//! The fused per-row pass: one call advances every eligible diagonal's `QT` over a
//! struct-of-arrays window representation and reduces the row's candidates to its exact
//! nearest entry, without ever materializing the exact `u128` work for more than a near-tie.
//!
//! For any per-window state a stream can reach, one call to [`advance_and_rank`] leaves every
//! diagonal's `QT` exactly where `ZNorm`'s [`advance`](crate::kernel::Metric::advance) (and [`start`](crate::kernel::Metric::start) for the new
//! diagonal) would, and returns exactly the [`Entry`] that offering every eligible neighbor to
//! [`Nearest`] and finishing returns: same ρ, d² and index, bit for bit. [`advance_and_rank_narrow`]
//! is the same contract on i32-offset columns, its bulk lane work run through
//! [`crate::kernel_neon::rank_row`]; every window's kind is read through [`crate::columns`],
//! never a second encoding of its own.

use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::batch::survivor;
use crate::columns;
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::estimate::{F64, Producer, band_floor};
use crate::index::SampleIndex;
use crate::kernel_neon::{Conversion, RowScalars};
use crate::nearest::{ExactDistance, Nearest, rescan_near_tie};
use crate::right::ColumnSink;
use crate::stats::{Window, WindowKind};
use crate::znorm::{TickWindow, centered_cross};

mod acc;
mod chunk;
mod chunk_wide;
mod column;

pub use acc::RowAcc;
pub(crate) use acc::{RankedLane, RowId};
pub(crate) use chunk::{ChunkCols, RankRow, pass_chunk, pass_chunk_offering};
pub(crate) use chunk_wide::{WideChunkCols, WideRank, WideRow, pass_chunk_wide};
pub(crate) use column::ColumnAcc;

/// Per-window columns indexed by window start, struct-of-arrays, contiguous: window `j` is
/// `ticks[j..j + m]`. `r[j]`/`bias[j]` are `columns::r_bias`'s pair for that window, and
/// `variance[j]` (the exact `V`) is meaningful only where `r[j]` is positive (a Valid window);
/// elsewhere its value is never read. A window's kind is encoded here exactly as
/// `columns::encode` builds it, never by a second scheme of this struct's own.
pub struct Columns<'a> {
    pub ticks: &'a [i64],
    pub sum: &'a [i64],
    pub r: &'a [f64],
    pub bias: &'a [f64],
    pub variance: &'a [u64],
}

/// `window`'s own `(r, bias, variance)` triple (`columns::encode`), appended to a growing
/// engine's owned columns by one window: the one builder every engine calls.
pub fn push_window(
    window: Window,
    sum: &mut Vec<i64>,
    r: &mut Vec<f64>,
    bias: &mut Vec<f64>,
    variance: &mut Vec<u64>,
) {
    sum.push(window.sum);
    let (r_val, bias_val, variance_val) = columns::encode(&window);
    r.push(r_val);
    bias.push(bias_val);
    variance.push(variance_val);
}

/// The row completing this call: its window index and its own statistics.
pub struct RowInputs {
    pub row: usize,
    pub window: Window,
}

/// The `Window` [`Columns`] describes at `j`, decoded by [`columns::window_of`]: the one place
/// this reconstruction happens, matching [`push_window`]'s own encoding exactly.
fn window_at(cols: &Columns, j: usize) -> Window {
    columns::window_of(cols.sum[j], cols.r[j], cols.bias[j], cols.variance[j])
}

/// The [`TickWindow`] [`crate::znorm::ZNorm`] needs for window `j`: `leaving` is unused whenever `j` is
/// offered to [`start`](crate::kernel::Metric::start) (never to `ZNorm`'s [`advance`](crate::kernel::Metric::advance)), which is the only case where `j`
/// has no predecessor tick of its own.
fn tick_window_at(cols: &Columns, j: usize, m: usize, window: Window) -> TickWindow {
    TickWindow {
        entering: cols.ticks[j + m - 1],
        leaving: if j == 0 { 0 } else { cols.ticks[j - 1] },
        stats: window,
    }
}

/// The row's own kind, reduced to what the fast pass needs: a non-finite row never reaches
/// this (the caller returns [`Entry::NONE`] for it beforehand).
#[derive(Clone, Copy)]
enum RowFast {
    Constant,
    Valid { inv_sqrt: f64 },
}

fn row_fast(window: &Window) -> Option<RowFast> {
    match window.kind {
        WindowKind::NonFinite => None,
        WindowKind::Constant => Some(RowFast::Constant),
        WindowKind::Valid { inv_sqrt, .. } => Some(RowFast::Valid { inv_sqrt }),
    }
}

/// `previous`'s diagonal advanced by one step: the same wrapping terms `ZNorm`'s [`advance`](crate::kernel::Metric::advance)
/// computes from a [`TickWindow`] pair, read directly off the columns instead.
#[inline(always)]
pub(crate) fn advance_lane(
    previous: i64,
    row_enter: i64,
    row_leave: i64,
    entering: i64,
    leaving: i64,
) -> i64 {
    previous
        .wrapping_add(row_enter.wrapping_mul(entering))
        .wrapping_sub(row_leave.wrapping_mul(leaving))
}

/// `N`'s own wrapping terms, the same as [`centered_cross`], read directly off the columns.
#[inline(always)]
pub(crate) fn centered_cross_fast(qt: i64, m: i64, row_sum: i64, neighbor_sum: i64) -> i64 {
    m.wrapping_mul(qt)
        .wrapping_sub(row_sum.wrapping_mul(neighbor_sum))
}

/// Calls `lane(j)` for every `j` in `1..count`, folding into one running maximum together
/// with `seed` (`j = 0`'s own value). `lane` itself has no dependency on any other `j`, so
/// this is still shaped for a later SIMD node to vectorize directly. An 8-independent-lane
/// hand-unrolling of this reduction was tried and measured (`cargo bench -p ladde
/// --bench row`): slower here, not faster — this target's backend never turned either form
/// into real SIMD, so 8 separate running maxima were only ever extra register traffic around
/// the same scalar work. Kept plain; a dedicated SIMD node should vectorize `lane`, not this
/// reduction's shape.
#[inline(always)]
fn fold_max(count: usize, seed: f64, mut lane: impl FnMut(usize) -> f64) -> f64 {
    let mut max_rho_hat = seed;
    for j in 1..count {
        max_rho_hat = max_rho_hat.max(lane(j));
    }
    max_rho_hat
}

/// Advances every continuing diagonal with no ranking at all: the row is non-finite, so
/// `Nearest` would ignore every offer anyway.
fn advance_only(
    qt: &mut [i64],
    cols: &Columns,
    row_enter: i64,
    row_leave: i64,
    m: usize,
    count: usize,
) {
    for (j, qt_slot) in qt.iter_mut().enumerate().take(count).skip(1) {
        *qt_slot = advance_lane(
            *qt_slot,
            row_enter,
            row_leave,
            cols.ticks[j + m - 1],
            cols.ticks[j - 1],
        );
    }
}

/// Grows `qt` to `count` and puts every continuing diagonal where [`advance_and_rank`] and
/// [`advance_and_rank_narrow`] now expect to find it: `qt[j]` already holding pair `(i − 1, j
/// − 1)` for `j ≥ 1` (`qt[0]` is left as whatever `resize` fills it with; both functions
/// overwrite it with a fresh diagonal before ever reading it). Neither function shifts
/// anything itself any more, so every caller that keeps its own persistent, window-indexed
/// `qt` — `batch`'s own call sites, and this crate's own test and bench drivers, which drive
/// these functions directly rather than through a stream — does the equivalent of this through
/// this one function, rather than re-deriving the shift independently: a later change to the
/// contract then breaks whichever caller forgot to update, not a silent duplicate.
///
/// `#[doc(hidden)]`: an internal contract helper, not part of this crate's own stable public
/// surface, exposed only so code outside the crate (this crate's own benches) can still share
/// it instead of keeping their own copy.
#[doc(hidden)]
pub fn resize_and_shift(qt: &mut Vec<i64>, count: usize) {
    qt.resize(count, 0);
    if count > 0 {
        qt.copy_within(0..count - 1, 1);
    }
}

/// A buffer that only ever grows at the front, one slot at a time: exactly the shape
/// the growing stream's and batch's lag-ordered `qt` need (one owner), since every row adds one
/// fresh diagonal ahead of every continuing one and nothing is ever evicted. `buf[start..]` is
/// the live slice, in the same ascending-by-neighbor order the row functions read, and it always
/// runs to the end of `buf` (the live slice's far edge never moves). Growing it moves no existing
/// element — it only decrements `start` — unless the front is exhausted, in which case every live
/// element moves exactly once, to a doubled buffer's own back; amortized O(1) per row, same as
/// `Vec::push`'s own doubling, mirrored to grow from the other end.
pub(crate) struct FrontGrowBuffer {
    buf: Vec<i64>,
    start: usize,
}

impl FrontGrowBuffer {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            start: 0,
        }
    }

    /// The live slice: `qt[j]` for every eligible neighbor `j`, ascending.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [i64] {
        &mut self.buf[self.start..]
    }

    /// [`Self::as_mut_slice`], read-only: a checkpoint's own view of the live diagonals, in
    /// the same logical (never physical) order a restore rebuilds through repeated
    /// [`Self::grow_front`].
    pub(crate) fn as_slice(&self) -> &[i64] {
        &self.buf[self.start..]
    }

    /// Adds one fresh slot at the front — this row's own `j = 0` — leaving every existing
    /// element exactly where it was, so the caller never pays to move what grow-shift exists
    /// to stop moving.
    pub(crate) fn grow_front(&mut self) {
        if self.start == 0 {
            let live = self.buf.len();
            let new_cap = ((live + 1) * 2).max(4);
            let mut new_buf = vec![0i64; new_cap];
            let dest_start = new_cap - live;
            new_buf[dest_start..].copy_from_slice(&self.buf);
            self.buf = new_buf;
            self.start = dest_start;
        }
        self.start -= 1;
    }
}

/// This is not a second narrow row pass. It is the exact i64 fallback, different
/// arithmetic (wrapping i64 `QT`, no NEON lane type) on a cold path a stream enters at most once
/// and never leaves, so it keeps its own loop rather than a wide lane type for `pass_chunk`.
///
/// For any per-window state a stream can reach, one call leaves every diagonal's `QT` exactly
/// where `ZNorm`'s [`advance`](crate::kernel::Metric::advance) (and [`start`](crate::kernel::Metric::start) for the new diagonal) would, and returns
/// exactly the [`Entry`] that offering every eligible neighbor to [`Nearest`] and finishing
/// would: same ρ, d² and index, bit for bit.
///
/// **Layout.** `qt[j]` is the diagonal paired with neighbor `j`. A diagonal's lag is fixed for
/// its whole life, so the same diagonal that was `j − 1` last row is `j` this row: **the
/// caller's own contract is that `qt[j]` already holds pair `(i − 1, j − 1)` for every `j ≥
/// 1`** before this call ([`resize_and_shift`] satisfies it) — this call never shifts anything
/// itself, only advances each slot where it already sits, so that loop reads `qt` and `cols`
/// with the same ascending stride (never opposite ones). The one new diagonal (lag =
/// `row.row`) is always `j = 0`, started once per call, not part of the per-lane loop.
/// `scratch` holds each neighbor's `ρ̂`, indexed by neighbor.
///
/// **Near-tie strategy.** A first pass advances every diagonal and fills `scratch` with `ρ̂`
/// (unclamped — see the threshold comment below) while tracking this row's maximum, `M`; a
/// second pass offers [`Nearest`] only the neighbors with `ρ̂ ≥ M − 2ε` — usually exactly one —
/// and the exact `u128` comparison inside `Nearest` only ever runs there.
pub fn advance_and_rank(
    qt: &mut [i64],
    cols: &Columns,
    row: &RowInputs,
    m: usize,
    exclusion: usize,
    scratch: &mut [f64],
) -> Entry {
    advance_and_rank_exact(qt, cols, row, m, exclusion, scratch)
        .map_or(Entry::NONE, |exact| exact.entry())
}

/// [`advance_and_rank`], keeping the exact correlation the winner was ranked by; `None` where
/// that returns [`Entry::NONE`].
pub(crate) fn advance_and_rank_exact(
    qt: &mut [i64],
    cols: &Columns,
    row: &RowInputs,
    m: usize,
    exclusion: usize,
    scratch: &mut [f64],
) -> Option<ExactDistance> {
    let i = row.row;
    if i <= exclusion {
        return None;
    }
    let count = i - exclusion;
    assert_eq!(
        qt.len(),
        count,
        "qt must hold exactly this row's eligible diagonals"
    );
    assert_eq!(
        scratch.len(),
        count,
        "scratch must hold one ρ̂ per eligible neighbor"
    );
    // Checked once, so the per-lane loop below can index `cols` and `qt` with no further
    // bounds check: every `j` it reaches is below `count`, and every column it reads at `j`
    // or at `j + m - 1` is covered by these.
    assert!(
        cols.ticks.len() >= i + m,
        "ticks must cover this row's own window"
    );
    assert!(
        cols.sum.len() >= count
            && cols.r.len() >= count
            && cols.bias.len() >= count
            && cols.variance.len() >= count,
        "cols must cover every eligible neighbor"
    );

    let row_enter = cols.ticks[i + m - 1];
    let row_leave = cols.ticks[i - 1];
    let row_sum = row.window.sum;
    let m_i64 = m as i64;

    // `j = 0` is always a brand new diagonal, started once per call from the raw ticks;
    // `qt[j]` for `j ≥ 1` already holds pair `(i − 1, j − 1)` (the caller's own contract above),
    // so there is nothing to shift here.
    let start_qt = CrossTerms::of(cols.ticks, m).at(i, 0);
    qt[0] = start_qt;

    let Some(row_fast) = row_fast(&row.window) else {
        advance_only(qt, cols, row_enter, row_leave, m, count);
        return None;
    };

    let n0 = centered_cross_fast(start_qt, m_i64, row_sum, cols.sum[0]);
    let rho_hat_0 = match row_fast {
        RowFast::Constant => columns::constant_row_rank(cols.r[0], cols.bias[0]),
        RowFast::Valid { inv_sqrt: row_inv } => {
            columns::valid_row_rank(n0, row_inv, cols.r[0], cols.bias[0])
        }
    };
    scratch[0] = rho_hat_0;

    // Pass 1: advance every diagonal and this row's candidate ρ̂, ascending by neighbor
    // (positive-stride reads of `cols` and, after the shift above, of `qt` too); split by the
    // row's own kind so a constant row never computes the valid-valid product, and each
    // lane's update has no dependency on any other.
    let max_rho_hat = match row_fast {
        RowFast::Constant => fold_max(count, rho_hat_0, |j| {
            qt[j] = advance_lane(
                qt[j],
                row_enter,
                row_leave,
                cols.ticks[j + m - 1],
                cols.ticks[j - 1],
            );
            let rho_hat = columns::constant_row_rank(cols.r[j], cols.bias[j]);
            scratch[j] = rho_hat;
            rho_hat
        }),
        RowFast::Valid { inv_sqrt: row_inv } => fold_max(count, rho_hat_0, |j| {
            qt[j] = advance_lane(
                qt[j],
                row_enter,
                row_leave,
                cols.ticks[j + m - 1],
                cols.ticks[j - 1],
            );
            let n = centered_cross_fast(qt[j], m_i64, row_sum, cols.sum[j]);
            let rho_hat = columns::valid_row_rank(n, row_inv, cols.r[j], cols.bias[j]);
            scratch[j] = rho_hat;
            rho_hat
        }),
    };

    if max_rho_hat == f64::NEG_INFINITY {
        return None;
    }

    // Pass 2: the exact fallback, restricted to the near-tie band (usually one neighbor).
    // `scratch` and `max_rho_hat` both hold the fast pass's own *unclamped* ranking value,
    // never `Nearest`'s clamped one, and the threshold below compares them on that same
    // unclamped basis. Write `t` for the row's true exact nearest neighbor, `a` for whichever
    // candidate this pass's own maximum `M` came from, `ρ` for a candidate's true correlation,
    // and `err` for `rank_estimate`'s own error bound (`|ρ̂ − ρ| ≤ err`): then
    // `ρ̂_t ≥ ρ_t − err ≥ ρ_a − err ≥ M − 2·err ≥ M − δ > M − 2δ` (the second step from `t`
    // being the true maximum, `ρ_t ≥ ρ_a`; the third from `ρ_a ≥ ρ̂_a − err = M − err`; the
    // fourth from `δ`, `F64`'s declared bound, being `2·err`). The true winner's `ρ̂` therefore
    // always clears this threshold, with a whole `δ` of margin to spare. Clamping only ever moves an
    // estimate toward the true `ρ ∈ [-1, 1]` it approximates, so it can only shrink `|ρ̂ − ρ|`,
    // never grow it: a clamped `ρ̂` obeys the same `err` bound, and the chain above holds for
    // it too. `Nearest` recomputes each offered candidate's own (clamped) `ρ̂` and, only here,
    // the exact comparison, bit for bit with the reference.
    let row_tick_window = TickWindow {
        entering: row_enter,
        leaving: row_leave,
        stats: row.window,
    };
    let mut nearest = Nearest::new(&row.window, m);
    let threshold = band_floor(max_rho_hat, F64::DELTA);
    rescan_near_tie(scratch, threshold, |j| {
        let neighbor = window_at(cols, j);
        let neighbor_tick_window = tick_window_at(cols, j, m, neighbor);
        let n = centered_cross(qt[j], &row_tick_window, &neighbor_tick_window, m);
        let index = SampleIndex::new(j as u64).expect("j is below NONE's sentinel");
        nearest.offer(n, &neighbor, index);
    });
    nearest.finish_exact()
}

/// [`Columns`], but every tick and window sum is an i32 offset from an anchor instead of
/// the absolute i64 value: the same per-window identity `Columns` tracks (window `j` is still
/// `ticks[j..j + m]`), just in the narrowed arithmetic [`crate::kernel_neon::rank_row`] runs
/// on. `r`/`bias`/`variance` are anchor-independent, so a caller shares the exact slices its
/// own [`Columns`] already holds, never a second copy. `conversion` is how a row kernel converts
/// the cross terms of a row read from these columns with its neighbors, and every row built
/// from them carries it.
pub struct NarrowColumns<'a> {
    pub ticks: &'a [i32],
    pub sum: &'a [i32],
    pub r: &'a [f64],
    pub bias: &'a [f64],
    pub variance: &'a [u64],
    pub conversion: Conversion,
}

/// The `Window` [`NarrowColumns`] describes at `j`: like [`window_at`], but `sum` is never
/// meaningful for a narrowed neighbor (nothing [`Nearest`] reads depends on it), so it is
/// never reconstructed, only `0`.
pub(crate) fn window_at_narrow(cols: &NarrowColumns, j: usize) -> Window {
    columns::window_of(0, cols.r[j], cols.bias[j], cols.variance[j])
}

/// Like [`advance_and_rank`], but every live diagonal's `m·QT` is kept as an i32-offset
/// product from an anchor and ranked through `pass_row_narrow`; bit for bit the same
/// `Entry` as `advance_and_rank` on the same (un-narrowed) inputs. `scratch` (one slot per
/// eligible neighbor) only lends its prefix as the chunk tile.
#[allow(clippy::too_many_arguments)]
pub fn advance_and_rank_narrow(
    qt: &mut [i64],
    cols: &NarrowColumns,
    row: &RowInputs,
    row_sum_offset: i32,
    m: usize,
    exclusion: usize,
    scratch: &mut [f64],
) -> Entry {
    let i = row.row;
    if i <= exclusion {
        return Entry::NONE;
    }
    assert_eq!(
        scratch.len(),
        i - exclusion,
        "scratch must hold one ρ̂ per eligible neighbor"
    );
    let tile_len = scratch.len().min(DEFAULT_TILE_LANES);
    let mut acc = RowAcc::new();
    acc.begin(&row.window, NonZeroUsize::MIN, 1);
    let mut nearest = Nearest::new(&row.window, m);
    pass_row_narrow(
        qt,
        cols,
        row,
        row_sum_offset,
        m,
        exclusion,
        &mut acc,
        &mut scratch[..tile_len],
        &mut nearest,
    );
    nearest.finish()
}

/// Row `i`'s own [`RowScalars`] on narrowed columns: `me`/`ml` are its entering/leaving tick
/// scaled by `m` (with the same debug-only range check), `si` its own window-sum offset and `ri`
/// its `1/√V` (`0.0` for a Constant or non-finite row, which never ranks). One owner for the
/// row-pass engines (the stream's and batch's) alike.
pub(crate) fn narrow_row_scalars(
    cols: &NarrowColumns,
    window: &Window,
    i: usize,
    m: usize,
    row_sum_offset: i32,
) -> RowScalars {
    let me = i64::from(cols.ticks[i + m - 1]).wrapping_mul(m as i64);
    let ml = i64::from(cols.ticks[i - 1]).wrapping_mul(m as i64);
    debug_assert!(
        i32::try_from(me).is_ok() && i32::try_from(ml).is_ok(),
        "an in-bound offset scales by m within i32 (Narrow's own bound)"
    );
    RowScalars {
        me: me as i32,
        ml: ml as i32,
        si: row_sum_offset,
        ri: match window.kind {
            WindowKind::Valid { inv_sqrt, .. } => inv_sqrt,
            _ => 0.0,
        },
        conversion: cols.conversion,
    }
}

/// Lanes per chunk when the caller has no better number: the ρ̂ tile is 8 B per lane, so 1024
/// lanes (8 KiB) leave room in a 32 KiB L1d for the columns, and a w ≈ 10³ row is one chunk.
pub(crate) const DEFAULT_TILE_LANES: usize = 1024;

/// One whole narrow row over the growing layout (`qt[j]` is the diagonal paired with neighbor
/// `j`; the caller's contract is that `qt[j]` already holds pair `(i − 1, j − 1)` for `j ≥ 1`):
/// the fresh diagonal `j = 0`, then `tile.len()`-lane chunks through [`pass_chunk`], then
/// [`RowAcc::finish`] into `reducer`. The caller has already `begin`-ed `acc` for this row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass_row_narrow(
    qt: &mut [i64],
    cols: &NarrowColumns,
    row: &RowInputs,
    row_sum_offset: i32,
    m: usize,
    exclusion: usize,
    acc: &mut RowAcc,
    tile: &mut [f64],
    reducer: &mut Nearest,
) {
    pass_row_narrow_offering::<ColumnAcc>(
        qt,
        cols,
        row,
        row_sum_offset,
        m,
        exclusion,
        (acc, tile, reducer),
        None,
    );
}

/// [`pass_row_narrow`] that also offers every ranked pair to `right`, as the later window of it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass_row_narrow_offering<S: ColumnSink>(
    qt: &mut [i64],
    cols: &NarrowColumns,
    row: &RowInputs,
    row_sum_offset: i32,
    m: usize,
    exclusion: usize,
    (acc, tile, reducer): (&mut RowAcc, &mut [f64], &mut Nearest),
    mut right: Option<&mut S>,
) {
    let i = row.row;
    if i <= exclusion {
        return;
    }
    let count = i - exclusion;
    assert_eq!(
        qt.len(),
        count,
        "qt must hold exactly this row's eligible diagonals"
    );
    assert!(!tile.is_empty(), "a tile holds at least one lane");
    assert!(
        cols.ticks.len() >= i + m,
        "ticks must cover this row's own window"
    );
    assert!(
        cols.sum.len() >= count
            && cols.r.len() >= count
            && cols.bias.len() >= count
            && cols.variance.len() >= count,
        "cols must cover every eligible neighbor"
    );

    let scalars = narrow_row_scalars(cols, &row.window, i, m, row_sum_offset);

    // `j = 0` is always a brand new diagonal, started once per call from the narrowed ticks.
    let pairs = CrossTerms::of(cols.ticks, m);
    let qt0 = pairs.at(i, 0);
    qt[0] = qt0;

    let neighbor0 = |_: usize| window_at_narrow(cols, 0);
    let started = acc.start_lane(
        qt0,
        &ChunkCols {
            enter: &[],
            leave: &[],
            sum: &cols.sum[..1],
            r: &cols.r[..1],
            bias: &cols.bias[..1],
            first: 0,
            neighbor: &neighbor0,
        },
        scalars,
    );
    if let (Some(sink), Some(lane)) = (right.as_deref_mut(), started) {
        sink.offer_lane::<RankRow>(0, acc.id(i), lane, || columns::variance(&neighbor0(0)));
    }
    pairs.force_resume(i, (i, 1), &mut qt[1..]);
    let mut lo = 1;
    while lo < count {
        let hi = (lo + tile.len()).min(count);
        let visit = survivor::stream_visits(i, lo..hi);
        debug_assert!(visit, "a stream visits every block");
        let neighbor = |lane: usize| window_at_narrow(cols, lo + lane);
        pass_chunk_offering(
            &mut qt[lo..hi],
            ChunkCols {
                enter: &cols.ticks[lo + m - 1..hi + m - 1],
                leave: &cols.ticks[lo - 1..hi - 1],
                sum: &cols.sum[lo..hi],
                r: &cols.r[lo..hi],
                bias: &cols.bias[lo..hi],
                first: lo,
                neighbor: &neighbor,
            },
            scalars,
            acc,
            tile,
            right.as_deref_mut().map(|sink| (sink, i)),
        );
        lo = hi;
    }
    acc.finish(reducer);
}

/// [`pass_row_narrow`] on the exact i64 arithmetic over the growing layout: the fresh diagonal
/// `j = 0`, then `tile.len()`-lane chunks through [`pass_chunk_wide`], then [`RowAcc::finish`]
/// into `reducer`. It keeps the `k` nearest the caller `begin`-ed `acc` with, which the fused
/// [`advance_and_rank_exact`] (one nearest) cannot.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass_row_wide(
    qt: &mut [i64],
    cols: &Columns,
    row: &RowInputs,
    m: usize,
    exclusion: usize,
    acc: &mut RowAcc,
    tile: &mut [f64],
    reducer: &mut Nearest,
) {
    pass_row_wide_offering::<ColumnAcc>(qt, cols, row, m, exclusion, (acc, tile, reducer), None);
}

/// [`pass_row_wide`] that also offers every ranked pair to `right`, as the later window of it.
pub(crate) fn pass_row_wide_offering<S: ColumnSink>(
    qt: &mut [i64],
    cols: &Columns,
    row: &RowInputs,
    m: usize,
    exclusion: usize,
    (acc, tile, reducer): (&mut RowAcc, &mut [f64], &mut Nearest),
    mut right: Option<&mut S>,
) {
    let i = row.row;
    if i <= exclusion {
        return;
    }
    let count = i - exclusion;
    assert_eq!(
        qt.len(),
        count,
        "qt must hold exactly this row's eligible diagonals"
    );
    assert!(!tile.is_empty(), "a tile holds at least one lane");
    assert!(
        cols.ticks.len() >= i + m,
        "ticks must cover this row's own window"
    );
    assert!(
        cols.sum.len() >= count
            && cols.r.len() >= count
            && cols.bias.len() >= count
            && cols.variance.len() >= count,
        "cols must cover every eligible neighbor"
    );

    let scalars = WideRow {
        enter: cols.ticks[i + m - 1],
        leave: cols.ticks[i - 1],
        sum: row.window.sum,
    };
    let pairs = CrossTerms::of(cols.ticks, m);
    let qt0 = pairs.at(i, 0);
    qt[0] = qt0;
    let neighbor0 = |_: usize| window_at(cols, 0);
    let started = acc.start_lane_wide(
        qt0,
        &WideChunkCols {
            enter: &[],
            leave: &[],
            sum: &cols.sum[..1],
            r: &cols.r[..1],
            bias: &cols.bias[..1],
            first: 0,
            neighbor: &neighbor0,
        },
        scalars,
        m,
    );
    if let (Some(sink), Some(lane)) = (right.as_deref_mut(), started) {
        sink.offer_lane::<WideRank>(0, acc.id(i), lane, || columns::variance(&neighbor0(0)));
    }
    pairs.force_resume(i, (i, 1), &mut qt[1..]);
    let mut lo = 1;
    while lo < count {
        let hi = (lo + tile.len()).min(count);
        let visit = survivor::stream_visits(i, lo..hi);
        debug_assert!(visit, "a stream visits every block");
        let neighbor = |lane: usize| window_at(cols, lo + lane);
        pass_chunk_wide(
            &mut qt[lo..hi],
            WideChunkCols {
                enter: &cols.ticks[lo + m - 1..hi + m - 1],
                leave: &cols.ticks[lo - 1..hi - 1],
                sum: &cols.sum[lo..hi],
                r: &cols.r[lo..hi],
                bias: &cols.bias[lo..hi],
                first: lo,
                neighbor: &neighbor,
            },
            scalars,
            m,
            acc,
            tile,
        );
        if let Some(sink) = right.as_deref_mut().filter(|_| acc.ranks()) {
            let ranked = &qt[lo..hi];
            sink.offer_chunk::<WideRank>(
                lo,
                acc.id(i),
                &tile[..hi - lo],
                |lane| {
                    centered_cross_fast(ranked[lane], m as i64, scalars.sum, cols.sum[lo + lane])
                },
                |lane| columns::variance(&neighbor(lane)),
            );
        }
        lo = hi;
    }
    acc.finish(reducer);
}

#[cfg(test)]
mod tests;
