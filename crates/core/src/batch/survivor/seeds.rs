//! The seed pre-pass: before a tiled self-join's tiles run, a few exact pairs are computed where
//! the bound is highest, and the floors they stand for are raised in the run's [`SharedFloors`],
//! so the pass meets near-final floors from its first block instead of building them from the
//! pairs it visits.
//!
//! A seed is a real pair of the run (outside the exclusion zone, both windows valid), ranked by
//! the same estimate the kernel ranks it by, and its floor is lowered by the same near-tie band
//! ([`band_floor`]): the floor of the `k`-th best of a window's seeds on a side is at most the
//! final floor of that side, exactly as a partial's is. The seeds only raise floors, never enter
//! a result: the pass still computes every pair it does not skip, so every output bit is the plain
//! pass's, and which blocks it skips is all that changes. A row's floor is raised from seeds that
//! pair it with earlier windows (its left list) and a column's from seeds that pair it with later
//! ones (its right list), a window needing `k` seeds on a side before its floor on that side is
//! raised. A run that reports a global floor, or walks chosen diagonals only, takes no seeds: its
//! floors are not the whole run's.
//!
//! Each group of [`GROUP_ROWS`] rows asks the group bound (the run's producer, [`Asked`], over the
//! union box of its rows) about every part of [`PART_LANES`] slots every row continues through,
//! in O(1) memory keeping the 8 highest; then each block of 8 rows of the group picks, in each of
//! those parts, the sub-block of 8 slots whose fine bound is highest and computes its pairs
//! exactly, one start sum for each of its 8 diagonals and the rest by the exact recurrence. Both
//! bounds are the producer's own values, so a producer that bounds more finely chooses the seeds
//! more finely with no seed code of its own.
//!
//! Memory is O(n): a group's pairs are held until its floors are raised.

use super::SharedFloors;
use super::asked::{Asked, Survey};
use crate::batch::tasks::run_tasks;
use crate::bound_paa::{Coefficients, Paa, Segments};
use crate::columns::valid_row_rank;
use crate::cross_term::{CrossTerms, Side};
use crate::estimate::{F64, Producer, band_floor};
use crate::kbest::BLOCK_LANES;
use crate::route::SeedForce;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Range;

/// Slots per part a group's bound is asked over: the keeper's lane block.
pub(super) const PART_LANES: usize = BLOCK_LANES;

/// Slots per sub-block a seed's pairs are chosen over.
pub(super) const SUB_SLOTS: usize = 8;

/// Rows per group the group bound is asked over: eight blocks.
pub(super) const GROUP_ROWS: usize = 64;

/// The parts of the best group bounds a group seeds.
pub(super) const BEST_PARTS: usize = 8;

/// What a run's seeds read: the exact cross terms of any pair of windows, the exact centered cross
/// term of a pair from its start sum, each window's kind, and the PAA coefficients the bound reads.
pub(crate) struct SeedInput<'a, S: Side> {
    pub(crate) pairs: CrossTerms<S>,
    /// The centered cross term `N` of the pair `(row, column)` whose cross term is the third
    /// argument, as the pass computes it.
    pub(crate) centered: &'a (dyn Fn(usize, usize, i64) -> i64 + Sync),
    pub(crate) r: &'a [f64],
    pub(crate) bias: &'a [f64],
    pub(crate) paa: &'a Coefficients,
    pub(crate) exclusion: usize,
    pub(crate) k: usize,
}

/// What the seeds of a task did.
#[derive(Clone, Copy, Default)]
struct Found {
    pairs: usize,
    raises: usize,
}

impl core::ops::AddAssign for Found {
    fn add_assign(&mut self, other: Self) {
        self.pairs += other.pairs;
        self.raises += other.raises;
    }
}

/// Whether a window with `(r, bias)` ranks: valid, neither constant nor non-finite.
#[inline(always)]
fn ranks(r: f64, bias: f64) -> bool {
    r != 0.0 && bias != f64::NEG_INFINITY
}

impl<S: Side> SeedInput<'_, S> {
    /// The estimate the kernel would rank the pair `(row, column)` by, from its cross term `qt`;
    /// `None` when either window does not rank.
    #[inline]
    fn estimate(&self, row: usize, column: usize, qt: i64) -> Option<f64> {
        let (ri, rj, bj) = (self.r[row], self.r[column], self.bias[column]);
        if !ranks(ri, self.bias[row]) || !ranks(rj, bj) {
            return None;
        }
        Some(valid_row_rank((self.centered)(row, column, qt), ri, rj, bj))
    }
}

/// The seed pre-pass, over `floors`. Does nothing when the route harness forces it off, when the
/// run has no shared floors, keeps a global floor (`global`) or walks chosen diagonals (`runs`).
pub(crate) fn seed_floors<S: Side + Sync>(
    floors: Option<&SharedFloors>,
    input: &SeedInput<S>,
    (runs, global): (bool, bool),
    threads: NonZeroUsize,
) {
    let force = crate::route::seeds();
    let Some(table) = floors.filter(|_| force != SeedForce::Off && !runs && !global) else {
        return;
    };
    #[cfg(feature = "std")]
    let started = std::time::Instant::now();
    let found = match input.paa.segments() {
        Segments::Four => top::<Paa<4>, S>(table, input, BEST_PARTS, threads),
        Segments::Eight => top::<Paa<8>, S>(table, input, BEST_PARTS, threads),
        Segments::Sixteen => top::<Paa<16>, S>(table, input, BEST_PARTS, threads),
    };
    #[cfg(feature = "std")]
    let nanos = usize::try_from(started.elapsed().as_nanos()).unwrap_or(usize::MAX);
    #[cfg(not(feature = "std"))]
    let nanos = 0;
    crate::route::count_seeds(found.pairs, found.raises, nanos);
}

/// The group's seeds: the pairs of the best `best` parts' best sub-blocks, per block of 8 rows.
fn top<P: Asked, S: Side + Sync>(
    table: &SharedFloors,
    input: &SeedInput<S>,
    best: usize,
    threads: NonZeroUsize,
) -> Found {
    let windows = input.r.len();
    let first_row = input.exclusion + 1;
    let groups = windows.saturating_sub(first_row).div_ceil(GROUP_ROWS);
    let mut total = Found::default();
    for found in run_tasks(threads, groups, |g| {
        let rows = first_row + g * GROUP_ROWS..(first_row + (g + 1) * GROUP_ROWS).min(windows);
        group_seeds::<P, S>(table, input, rows, best)
    }) {
        total += found;
    }
    total
}

fn group_seeds<P: Asked, S: Side>(
    table: &SharedFloors,
    input: &SeedInput<S>,
    group: Range<usize>,
    best: usize,
) -> Found {
    let view = |rows: &Range<usize>| Survey::of_seeds(rows.clone(), input.exclusion, input.paa);
    // Every row of the group has the slots `0..common`, whose pair with column `row − exclusion −
    // 1 − slot` is outside the exclusion zone.
    let common = group.start - input.exclusion;
    let group_view = view(&group);
    let Some(rows) = P::rows(&group_view) else {
        return Found::default();
    };
    // The parts of the best group bounds, highest first, ties to the lower part.
    let mut top: [(f64, usize); 8] = [(f64::NEG_INFINITY, 0); 8];
    for start in (0..common).step_by(PART_LANES) {
        let part = start..(start + PART_LANES).min(common);
        let value = P::value(&rows, &group_view, &part).ranked();
        if value > top[best - 1].0 {
            let at = top[..best]
                .iter()
                .position(|&(v, _)| value > v)
                .unwrap_or(best - 1);
            top.copy_within(at..best - 1, at + 1);
            top[at] = (value, start);
        }
    }
    let mut entries = Vec::new();
    for block in group.clone().step_by(SUB_SLOTS) {
        let block = block..(block + SUB_SLOTS).min(group.end);
        let block_view = view(&block);
        let Some(row_box) = P::rows(&block_view) else {
            continue;
        };
        for &(_, start) in top[..best].iter().filter(|(v, _)| *v > f64::NEG_INFINITY) {
            let part_end = (start + PART_LANES).min(common);
            let mut chosen: Option<(f64, usize)> = None;
            for sub in (start..part_end).step_by(SUB_SLOTS) {
                let slots = sub..(sub + SUB_SLOTS).min(part_end);
                let v = P::fine_value(&row_box, &block_view, &slots).ranked();
                if v > f64::NEG_INFINITY && chosen.is_none_or(|(best_v, _)| v > best_v) {
                    chosen = Some((v, sub));
                }
            }
            if let Some((_, sub)) = chosen {
                let slots = sub..(sub + SUB_SLOTS).min(part_end);
                sub_block(input, &block, &slots, &mut entries);
            }
        }
    }
    let mut found = Found {
        pairs: entries.len(),
        raises: 0,
    };
    found.raises = raise(table, input.k, &mut entries);
    found
}

/// The estimates of the pairs of the consecutive `rows` against the consecutive `slots` (at most
/// [`SUB_SLOTS`] each), pushed as `(row, column, estimate)`: one start sum per diagonal, for the
/// block's first row, and the exact recurrence along each diagonal for the others.
pub(super) fn sub_block<S: Side>(
    input: &SeedInput<S>,
    rows: &Range<usize>,
    slots: &Range<usize>,
    out: &mut Vec<(usize, usize, f64)>,
) {
    let lanes = slots.len();
    let first = rows.start;
    // Lane `l` is slot `slots.end − 1 − l`, column `first − exclusion − slots.end + l` at the first
    // row, and one more each row after.
    let column = first - input.exclusion - slots.end;
    let mut states = [0i64; SUB_SLOTS];
    let states = &mut states[..lanes];
    if input.pairs.resume_lanes((first + 1, column + 1), states) != lanes {
        return;
    }
    for t in 0..rows.len() {
        for (l, &qt) in states.iter().enumerate() {
            if let Some(estimate) = input.estimate(first + t, column + t + l, qt) {
                out.push((first + t, column + t + l, estimate));
            }
        }
        if t + 1 < rows.len() {
            input
                .pairs
                .advance_or_resume(1, (first + t + 2, column + t + 2), states);
        }
    }
}

/// Raises the floors the pairs `entries` stand for: a row's from its `k`-th best pair, a column's
/// from its `k`-th best, each lowered by the near-tie band. Returns how many floors rose.
pub(super) fn raise(table: &SharedFloors, k: usize, entries: &mut [(usize, usize, f64)]) -> usize {
    let band = |estimate: f64| band_floor(estimate, F64::DELTA);
    let mut raises = 0;
    if k == 1 {
        for &(row, column, estimate) in entries.iter() {
            raises += usize::from(table.raise_row(row, band(estimate)));
            raises += usize::from(table.raise_column(column, band(estimate)));
        }
        return raises;
    }
    // Best first within a window: the `k`-th entry of its run is its floor, when it has `k`.
    let mut by = |side: fn(&(usize, usize, f64)) -> usize, raise: &dyn Fn(usize, f64) -> bool| {
        entries.sort_unstable_by(|a, b| side(a).cmp(&side(b)).then(b.2.total_cmp(&a.2)));
        for run in entries.chunk_by(|a, b| side(a) == side(b)) {
            if let Some(&(_, _, kth)) = run.get(k - 1) {
                raises += usize::from(raise(side(&run[0]), band(kth)));
            }
        }
    };
    by(|e| e.0, &|row, floor| table.raise_row(row, floor));
    by(|e| e.1, &|column, floor| table.raise_column(column, floor));
    raises
}
