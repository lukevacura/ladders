//! The batch left matrix profile: the whole series quantized and forward-filled up front, into
//! the same [`crate::row::Columns`] struct-of-arrays a stream builds incrementally, then every
//! row's diagonals advanced and ranked into per-row partials ([`crate::row::RowAcc`]) by the one
//! chunk pass every engine shares, never a second ranking path.
//!
//! Unlike a stream, batch sees every tick before running a single row. When the whole series fits
//! one narrow anchor, that anchor is taken once and the engines run on it. Otherwise independent
//! tiles each take an anchor for their own rows and another for their own columns, and only a
//! tile whose own range is still too wide runs the exact i64 pass; the result is the same bit for
//! bit.
//!
//! In every full profile (left, two-sided, top-k and AB-join), a window the hot passes cannot rank
//! at all (its variance numerator is past `2⁶³`) gets its exact nearest neighbors from the cold
//! route ([`crate::spill`]), and so can the windows that hold an outlier tick when the cost model
//! says that is cheaper than the exact i64 pass: the `cold` module wires it. A series inside one
//! anchor never reaches it.

mod ab_join;
mod cold;
mod derived;
mod kept;
mod left;
pub mod multi;
pub mod nonnorm;
pub(crate) mod one_loop;
mod pass;
mod plan;
mod quantized;
mod reanchor;
mod schedule;
mod sparse;
pub(crate) mod survivor;
mod tasks;
mod tile;
mod two_sided;

pub use self::ab_join::{
    ab_join, ab_join_report, ab_join_report_tiled, ab_join_single_threaded, ab_join_spill_report,
    ab_join_topk, ab_join_topk_over_tiled, ab_join_topk_over_walked, ab_join_topk_tiled,
};
pub(crate) use self::ab_join::{ab_join_hot, ab_join_over};
pub use self::cold::SpillReport;
pub(crate) use self::cold::{Cold, JoinSide, no_distance_cold, no_distance_cold_join};
pub use self::derived::{discords, fluss};
pub(crate) use self::kept::group_by_row;
pub(crate) use self::left::has_neighbor;
pub use self::left::{
    left_profile, left_profile_report, left_profile_report_tiled, left_profile_spill_report,
    left_profile_with_chunk, left_topk_profile, left_topk_profile_tiled,
};
#[cfg(all(feature = "parallel", not(loom)))]
pub use self::left::{left_profile_threads, left_profile_threads_tiled, left_profile_threads_with};
pub(crate) use self::plan::SlotRuns;
#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
pub(crate) use self::plan::break_even;
/// The cold route's cost verdict, public only for the route harness's spill driver.
#[doc(hidden)]
pub use self::plan::cold_route;
pub(crate) use self::plan::steps_through;
#[cfg(all(feature = "parallel", not(loom)))]
pub use self::plan::threaded_workers;
pub use self::plan::{chunk_lanes_for_bytes, default_chunk_lanes};
pub(crate) use self::quantized::Quantized;
pub use self::schedule::BatchReport;
/// The shared-floor table, public only for the model checkers' tests of its raise and read.
#[doc(hidden)]
pub use self::survivor::SharedFloors;
pub(crate) use self::two_sided::best_pair_profile;
pub(crate) use self::two_sided::{
    KeptList, Pairs, merge_kept, merged_two_sided, two_sided_sides_over,
};
pub use self::two_sided::{
    two_sided_profile, two_sided_profile_tiled, two_sided_spill_report, two_sided_topk_profile,
    two_sided_topk_profile_over_tiled, two_sided_topk_profile_over_walked,
    two_sided_topk_profile_tiled,
};

#[cfg(test)]
use self::ab_join::ab_join_chunked;
#[cfg(test)]
use self::kept::{join_list_len, side_counts};
#[cfg(test)]
use self::left::{left_profile_with_blocks, profile_on};
#[cfg(test)]
use self::pass::BLOCK_ROWS;
#[cfg(all(test, feature = "parallel", not(loom)))]
use self::plan::{
    PARTIALS_BYTES, SINGLE_THREADED_BELOW_ROWS, TILES_PER_THREAD, cut_tiles,
    single_threaded_below_rows, tile_plan,
};
#[cfg(test)]
use self::plan::{Tile, TilePlan};
#[cfg(test)]
use self::schedule::RIGHT_GROUP;
#[cfg(test)]
use self::tile::tile_columns;
#[cfg(test)]
use self::two_sided::{RightBook, is_finite};
#[cfg(all(test, feature = "parallel", not(loom)))]
use crate::row::RowAcc;

/// This row's eligible neighbors: `j` in `0..i − e`, with `e = ceil(m/4)` — the same half-width
/// [`advance_and_rank`] itself enforces (it returns [`Entry::NONE`] outright for any row at or
/// inside it).
#[cfg(test)]
fn eligible_count(row: usize, exclusion: usize) -> usize {
    row.saturating_sub(exclusion)
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
use alloc::vec::Vec;

/// Test-only: whether a left pass keeping one neighbor may leave the pair `(row, column)` out
/// because the constant class settles it, counted window by window from the samples' ticks. The
/// pair's correlation is fixed by its windows' kinds (they are not both ordinary); the column is
/// neither the first constant window nor the first finite one; and a window that settles the
/// row's offers lies between the column's zone and the row: a constant one for a constant row, a
/// finite one for an ordinary row, none needed for a non-finite row.
#[cfg(test)]
fn class_settles(
    samples: &[Option<i64>],
    m: usize,
    exclusion: usize,
) -> impl Fn(usize, usize) -> bool {
    const NON_FINITE: u8 = 0;
    const CONSTANT: u8 = 1;
    const ORDINARY: u8 = 2;
    let kinds: Vec<u8> = samples
        .windows(m)
        .map(|w| match w[0] {
            _ if w.iter().any(Option::is_none) => NON_FINITE,
            first if w.iter().all(|&tick| tick == first) => CONSTANT,
            _ => ORDINARY,
        })
        .collect();
    let first_constant = kinds.iter().position(|&kind| kind == CONSTANT);
    let first_finite = kinds.iter().position(|&kind| kind != NON_FINITE);
    move |row, column| {
        let between = &kinds[(column + exclusion + 1).min(row)..row];
        let beaten = match kinds[row] {
            NON_FINITE => true,
            CONSTANT => between.contains(&CONSTANT),
            _ => between.iter().any(|&kind| kind != NON_FINITE),
        };
        let fixed = kinds[row] != ORDINARY || kinds[column] != ORDINARY;
        let first = Some(column) == first_constant || Some(column) == first_finite;
        fixed && !first && beaten
    }
}

#[cfg(test)]
crate::sync::thread_local! {
    /// Test-only record: per finished row, every neighbor its [`RowAcc`] was fed.
    static VISITS: core::cell::RefCell<Vec<(usize, Vec<usize>)>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
crate::sync::thread_local! {
    /// Test-only: how many times the threaded engine ran on this thread.
    static PAR_RUNS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}
#[cfg(test)]
crate::sync::thread_local! {
    /// Test-only: the peak number of live bands of the last threaded engine run on this thread.
    static PEAK_LIVE_BANDS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}
#[cfg(test)]
crate::sync::thread_local! {
    /// Test-only: how many row groups the last threaded engine run on this thread made new
    /// instead of taking from a finished band.
    static FRESH_GROUPS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "parallel", not(loom)))]
crate::sync::thread_local! {
    /// Test-only: how many tiles of this thread's next threaded engine run must be inside at
    /// once before any proceeds (0: no gate).
    static TILE_RENDEZVOUS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "parallel", not(loom)))]
mod threaded_tests;
