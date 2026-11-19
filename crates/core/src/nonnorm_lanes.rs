//! The non-normalized lane kernel: for one row, carries many diagonals' `u64` residues one step
//! with the O(1) recurrence and finds the row's nearest complete neighbor among them, in a tight
//! loop over lanes laid out like the z-normalized row kernel's.
//!
//! A lane is one diagonal: its residue, the ticks its neighbor's window takes in and drops, and
//! whether that window is complete. [`ascending`] (the batch tiles) reads lane `i` at slice
//! position `i`; [`descending`] (the streams) reads lane `k` at position `len − 1 − k`, so both
//! walk their neighbors' ticks forward through memory. The residues wrap modulo `2⁶⁴` exactly as
//! [`NonNormalized<u64>`](crate::nonnorm::NonNormalized) does, so the bits equal the per-diagonal
//! recurrence whenever the caller's row is narrow (`m·R² < 2⁶⁴`).
//!
//! On `aarch64` (NEON) and on `x86_64` with AVX2 (`kernel_avx2`) the lanes run four at a time when
//! [`Fit::I32`] holds: every tick difference the row reads fits `i32`, so each square is one
//! widening multiply (`vmull_s32`, `_mm256_mul_epi32`) of the low 32 bits of the two ticks.
//! Otherwise, and everywhere else (or without AVX2), the scalar lane body runs; it is also the
//! vector loop's remainder and the oracle the tests pin the vector loop against. Both sit behind
//! the same two entry points, chosen at compile time on `aarch64` and at run time on `x86_64`.
//!
//! As a producer of pair estimates the kernel is exact (its bound is zero): a ranking pass
//! reports the block's best, the nearest complete lane with ties to the neighbor that starts
//! first, and the block's values stay readable in the residues for a keeper that wants them all.

use crate::kernel::Metric;
use crate::nonnorm::{NonNormalized, Residue, TickStep};

/// How far apart the ticks a row reads can be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// Every two ticks differ by at most `2³¹ − 1`.
    I32,
    /// Any two ticks.
    U64,
}

impl Fit {
    /// The fit of ticks spanning `lo..=hi`; `hi < lo` means no ticks.
    pub(crate) fn of(lo: i64, hi: i64) -> Self {
        if hi < lo || i128::from(hi) - i128::from(lo) <= i128::from(i32::MAX) {
            Self::I32
        } else {
            Self::U64
        }
    }
}

/// What a row does with its lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    /// Steps every residue.
    Advance,
    /// Finds the nearest complete lane of the residues as they are.
    Rank,
    /// Steps every residue, then finds the nearest complete lane.
    AdvanceAndRank,
}

impl Pass {
    const fn ranks(self) -> bool {
        matches!(self, Self::Rank | Self::AdvanceAndRank)
    }
}

/// Every lane's neighbor: the tick its window takes in, the one it drops, and whether it is
/// complete; all three hold one element per lane.
#[derive(Clone, Copy)]
pub(crate) struct Lanes<'a> {
    pub(crate) enter: &'a [i64],
    pub(crate) leave: &'a [i64],
    pub(crate) complete: &'a [bool],
}

/// Each lane's column's nearest row so far, ranked with the row's own: a complete lane whose
/// residue is smaller than its column's takes it with `row`. Rows arrive in ascending order, so
/// equal residues keep the earlier row.
pub(crate) struct ColumnMins<'a> {
    pub(crate) squared: &'a mut [u64],
    pub(crate) rows: &'a mut [u64],
    pub(crate) row: u64,
}

/// A row's nearest complete lane: its residue and lane, ties to the lane whose neighbor starts
/// first.
pub(crate) type Nearest = Option<(u64, usize)>;

/// A residue no complete lane of a narrow row reaches: a distance is at most `m·R² ≤ 2⁶⁴ − 1`, and
/// equality would need `R² | 2⁶⁴ − 1`, which is square-free, so `R = 1` and `m = 2⁶⁴ − 1`.
const NONE: u64 = u64::MAX;

/// Where residues live: a batch tile's own `u64`s, or a stream's `u128` diagonals (whose high half
/// a narrow row ignores and clears).
trait Store {
    fn get(&self, lane: usize) -> u64;
    fn set(&mut self, lane: usize, residue: u64);
}

impl Store for [u64] {
    #[inline(always)]
    fn get(&self, lane: usize) -> u64 {
        self[lane]
    }

    #[inline(always)]
    fn set(&mut self, lane: usize, residue: u64) {
        self[lane] = residue;
    }
}

impl Store for [u128] {
    #[inline(always)]
    fn get(&self, lane: usize) -> u64 {
        u64::from_stored(self[lane])
    }

    #[inline(always)]
    fn set(&mut self, lane: usize, residue: u64) {
        self[lane] = residue.as_u128();
    }
}

/// Whether `candidate` displaces `best`: a smaller residue, or for [`descending`] (whose later
/// lanes start first) an equal one too.
#[inline(always)]
fn displaces<const DESC: bool>(candidate: u64, best: u64) -> bool {
    if DESC {
        candidate <= best
    } else {
        candidate < best
    }
}

/// The scalar lane body over lanes `from..to`, folding into `best`.
#[inline(always)]
fn scalar_lanes<S: Store + ?Sized, const DESC: bool, const ADVANCE: bool, const RANK: bool>(
    residues: &mut S,
    row: TickStep,
    lanes: Lanes,
    mut columns: Option<&mut ColumnMins>,
    (from, to): (usize, usize),
    mut best: (u64, usize),
) -> (u64, usize) {
    let len = lanes.enter.len();
    for lane in from..to {
        let at = if DESC { len - 1 - lane } else { lane };
        let mut residue = residues.get(lane);
        if ADVANCE {
            let neighbor = TickStep {
                entering: lanes.enter[at],
                leaving: lanes.leave[at],
            };
            residue = NonNormalized::<u64>::advance(residue, &row, &neighbor);
            residues.set(lane, residue);
        }
        if RANK && lanes.complete[at] {
            if displaces::<DESC>(residue, best.0) {
                best = (residue, lane);
            }
            if let Some(columns) = columns.as_deref_mut()
                && residue < columns.squared[lane]
            {
                columns.squared[lane] = residue;
                columns.rows[lane] = columns.row;
            }
        }
    }
    best
}

fn assert_lengths(residues: usize, lanes: Lanes, columns: Option<&ColumnMins>) {
    assert_eq!(lanes.enter.len(), residues, "one entering tick per lane");
    assert_eq!(lanes.leave.len(), residues, "one leaving tick per lane");
    assert_eq!(lanes.complete.len(), residues, "one completeness per lane");
    if let Some(columns) = columns {
        assert_eq!(
            columns.squared.len(),
            residues,
            "one column residue per lane"
        );
        assert_eq!(columns.rows.len(), residues, "one column row per lane");
    }
}

fn finish(best: (u64, usize)) -> Nearest {
    (best.0 != NONE).then_some(best)
}

/// The scalar body over every lane, as both the oracle and the path for [`Fit::U64`].
fn scalar_row<S: Store + ?Sized, const DESC: bool>(
    residues: &mut S,
    len: usize,
    row: TickStep,
    lanes: Lanes,
    pass: Pass,
    columns: Option<&mut ColumnMins>,
) -> Nearest {
    let none = (NONE, usize::MAX);
    let span = (0, len);
    let best = match pass {
        Pass::Advance => {
            scalar_lanes::<S, DESC, true, false>(residues, row, lanes, None, span, none)
        }
        Pass::Rank => {
            scalar_lanes::<S, DESC, false, true>(residues, row, lanes, columns, span, none)
        }
        Pass::AdvanceAndRank => {
            scalar_lanes::<S, DESC, true, true>(residues, row, lanes, columns, span, none)
        }
    };
    if pass.ranks() { finish(best) } else { None }
}

/// Steps (with `pass`) a batch row's lanes, lane `i` at position `i` of `lanes`, and returns its
/// nearest complete lane when `pass` ranks; with `columns`, a ranking pass also offers every
/// complete lane to its column. `fit` must hold for every tick of `row` and `lanes`.
pub(crate) fn ascending(
    residues: &mut [u64],
    row: TickStep,
    lanes: Lanes,
    fit: Fit,
    pass: Pass,
    columns: Option<&mut ColumnMins>,
) -> Nearest {
    assert_lengths(residues.len(), lanes, columns.as_deref());
    let len = residues.len();
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if fit == Fit::I32 {
        // SAFETY: lengths asserted above; `fit` is the caller's bound on every difference.
        return unsafe { neon::row::<[u64], false>(residues, row, lanes, pass, columns) };
    }
    #[cfg(target_arch = "x86_64")]
    if fit == Fit::I32 && crate::kernel_avx2::enabled(crate::kernel_avx2::Kernel::NonNormalized) {
        // SAFETY: AVX2 detected; lengths asserted above; `fit` is the caller's bound on every
        // difference.
        return unsafe { avx2::row::<[u64], false>(residues, row, lanes, pass, columns) };
    }
    let _ = fit;
    scalar_row::<[u64], false>(residues, len, row, lanes, pass, columns)
}

/// Steps (with `pass`) a stream row's lanes, lane `k` at position `len − 1 − k` of `lanes`, and
/// returns its nearest complete lane when `pass` ranks, ties to the larger lane (the neighbor that
/// starts first). `fit` must hold for every tick of `row` and `lanes`.
pub(crate) fn descending(
    residues: &mut [u128],
    row: TickStep,
    lanes: Lanes,
    fit: Fit,
    pass: Pass,
) -> Nearest {
    assert_lengths(residues.len(), lanes, None);
    let len = residues.len();
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if fit == Fit::I32 {
        // SAFETY: lengths asserted above; `fit` is the caller's bound on every difference.
        return unsafe { neon::row::<[u128], true>(residues, row, lanes, pass, None) };
    }
    #[cfg(target_arch = "x86_64")]
    if fit == Fit::I32 && crate::kernel_avx2::enabled(crate::kernel_avx2::Kernel::NonNormalized) {
        // SAFETY: AVX2 detected; lengths asserted above; `fit` is the caller's bound on every
        // difference.
        return unsafe { avx2::row::<[u128], true>(residues, row, lanes, pass, None) };
    }
    let _ = fit;
    scalar_row::<[u128], true>(residues, len, row, lanes, pass, None)
}

/// The oracle [`ascending`] is checked against: the scalar body over every lane.
#[cfg(test)]
fn ascending_scalar(
    residues: &mut [u64],
    row: TickStep,
    lanes: Lanes,
    pass: Pass,
    columns: Option<&mut ColumnMins>,
) -> Nearest {
    assert_lengths(residues.len(), lanes, columns.as_deref());
    let len = residues.len();
    scalar_row::<[u64], false>(residues, len, row, lanes, pass, columns)
}

/// The oracle [`descending`] is checked against: the scalar body over every lane.
#[cfg(test)]
fn descending_scalar(residues: &mut [u128], row: TickStep, lanes: Lanes, pass: Pass) -> Nearest {
    assert_lengths(residues.len(), lanes, None);
    let len = residues.len();
    scalar_row::<[u128], true>(residues, len, row, lanes, pass, None)
}

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use super::{ColumnMins, Lanes, NONE, Nearest, Pass, Store, finish, scalar_lanes};
    use crate::nonnorm::TickStep;
    use core::arch::aarch64::*;

    /// The nearer of two lane results, ties to the lane whose neighbor starts first.
    #[inline(always)]
    fn nearer<const DESC: bool>(a: (u64, usize), b: (u64, usize)) -> (u64, usize) {
        let b_wins = b.0 < a.0 || (b.0 == a.0 && if DESC { b.1 > a.1 } else { b.1 < a.1 });
        if b_wins { b } else { a }
    }

    /// Four lanes' residues as two vectors, lanes `(i, i+1)` and `(i+2, i+3)`.
    pub(super) trait VectorStore: Store {
        /// # Safety
        /// Lanes `i..i + 4` exist.
        unsafe fn load4(&self, i: usize) -> (uint64x2_t, uint64x2_t);
        /// # Safety
        /// Lanes `i..i + 4` exist.
        unsafe fn store4(&mut self, i: usize, residues: (uint64x2_t, uint64x2_t));
    }

    impl VectorStore for [u64] {
        #[inline(always)]
        unsafe fn load4(&self, i: usize) -> (uint64x2_t, uint64x2_t) {
            // SAFETY: the caller keeps `i + 4` within the slice.
            unsafe {
                let at = self.as_ptr().add(i);
                (vld1q_u64(at), vld1q_u64(at.add(2)))
            }
        }

        #[inline(always)]
        unsafe fn store4(&mut self, i: usize, (lo, hi): (uint64x2_t, uint64x2_t)) {
            // SAFETY: the caller keeps `i + 4` within the slice.
            unsafe {
                let at = self.as_mut_ptr().add(i);
                vst1q_u64(at, lo);
                vst1q_u64(at.add(2), hi);
            }
        }
    }

    impl VectorStore for [u128] {
        /// A `u128` is its low then its high `u64` (little-endian), so a de-interleaving load of
        /// two `u128`s puts their low halves in the first vector.
        #[inline(always)]
        unsafe fn load4(&self, i: usize) -> (uint64x2_t, uint64x2_t) {
            // SAFETY: the caller keeps `i + 4` within the slice; each load reads two `u128`s.
            unsafe {
                let at = self.as_ptr().add(i).cast::<u64>();
                (vld2q_u64(at).0, vld2q_u64(at.add(4)).0)
            }
        }

        #[inline(always)]
        unsafe fn store4(&mut self, i: usize, (lo, hi): (uint64x2_t, uint64x2_t)) {
            // SAFETY: the caller keeps `i + 4` within the slice; each store writes two `u128`s,
            // high halves cleared as `as_u128` clears them.
            unsafe {
                let at = self.as_mut_ptr().add(i).cast::<u64>();
                let zero = vdupq_n_u64(0);
                vst2q_u64(at, uint64x2x2_t(lo, zero));
                vst2q_u64(at.add(4), uint64x2x2_t(hi, zero));
            }
        }
    }

    /// The low 32 bits of four consecutive ticks from `at` (little-endian: each tick's low half
    /// is its first).
    #[inline(always)]
    unsafe fn low_halves(at: *const i64) -> int32x4_t {
        // SAFETY: the caller keeps four ticks readable from `at`.
        unsafe { vld2q_s32(at.cast::<i32>()).0 }
    }

    /// `(a − b)²` for four lanes, as two vectors of two: the difference of the low halves is the
    /// true difference when it fits `i32`, and its widening square is exact.
    #[inline(always)]
    fn squares(a: int32x4_t, b: int32x4_t) -> (uint64x2_t, uint64x2_t) {
        // SAFETY: register-only NEON arithmetic.
        unsafe {
            let d = vsubq_s32(a, b);
            (
                vreinterpretq_u64_s64(vmull_s32(vget_low_s32(d), vget_low_s32(d))),
                vreinterpretq_u64_s64(vmull_high_s32(d, d)),
            )
        }
    }

    /// Lane masks for four lanes from their completeness bytes: all ones for an incomplete lane.
    #[inline(always)]
    fn masks(bytes: u32) -> (uint64x2_t, uint64x2_t) {
        let mask = |lane: u32| u64::from((bytes >> (8 * lane)) & 1).wrapping_sub(1);
        // SAFETY: register-only NEON moves.
        unsafe {
            (
                vcombine_u64(vcreate_u64(mask(0)), vcreate_u64(mask(1))),
                vcombine_u64(vcreate_u64(mask(2)), vcreate_u64(mask(3))),
            )
        }
    }

    /// Four lanes at a time, in two pairs of vectors: the scalar body's recurrence restated with
    /// `vmull_s32` squares (exact under [`super::Fit::I32`]) and wrapping `u64` adds, the same
    /// value modulo `2⁶⁴` as subtracting the leaving square and adding the entering one. Each
    /// vector position keeps its own nearest `(residue, lane)`, displaced as the scalar body
    /// displaces, and the positions merge only at the end; lanes past the last group of four
    /// run the scalar body.
    ///
    /// # Safety
    /// `lanes` and `columns` hold one element per residue.
    #[inline(always)]
    pub(super) unsafe fn row<S: VectorStore + ?Sized, const DESC: bool>(
        residues: &mut S,
        row: TickStep,
        lanes: Lanes,
        pass: Pass,
        columns: Option<&mut ColumnMins>,
    ) -> Nearest {
        // SAFETY: forwarded from the caller.
        unsafe {
            match (pass, columns) {
                (Pass::Advance, _) => {
                    body::<S, DESC, true, false, false>(residues, row, lanes, None)
                }
                (Pass::Rank, None) => {
                    body::<S, DESC, false, true, false>(residues, row, lanes, None)
                }
                (Pass::Rank, columns) => {
                    body::<S, DESC, false, true, true>(residues, row, lanes, columns)
                }
                (Pass::AdvanceAndRank, None) => {
                    body::<S, DESC, true, true, false>(residues, row, lanes, None)
                }
                (Pass::AdvanceAndRank, columns) => {
                    body::<S, DESC, true, true, true>(residues, row, lanes, columns)
                }
            }
        }
    }

    /// # Safety
    /// `lanes` and `columns` (present exactly when `COLUMNS`) hold one element per residue.
    #[inline(always)]
    unsafe fn body<
        S: VectorStore + ?Sized,
        const DESC: bool,
        const ADVANCE: bool,
        const RANK: bool,
        const COLUMNS: bool,
    >(
        residues: &mut S,
        row: TickStep,
        lanes: Lanes,
        mut columns: Option<&mut ColumnMins>,
    ) -> Nearest {
        let len = lanes.enter.len();
        let vectored = len / 4 * 4;
        // SAFETY: every lane group below starts at `i ≤ vectored − 4`, so lanes `i..i + 4`, their
        // tick positions (`i..i + 4`, or `len − 4 − i..len − i` descending) and their column
        // slots are in bounds of slices the caller sized to `len`.
        let best = unsafe {
            let enter_row = vdupq_n_s32(row.entering as i32);
            let leave_row = vdupq_n_s32(row.leaving as i32);
            let row_vec = vdupq_n_u64(columns.as_ref().map_or(0, |c| c.row));
            let step = vdupq_n_u64(4);
            let first = vcombine_u64(vcreate_u64(0), vcreate_u64(1));
            let mut index = (first, vaddq_u64(first, vdupq_n_u64(2)));
            let mut best = (vdupq_n_u64(NONE), vdupq_n_u64(NONE));
            let mut best_lane = (vdupq_n_u64(u64::MAX), vdupq_n_u64(u64::MAX));
            let mut i = 0;
            while i < vectored {
                let at = if DESC { len - 4 - i } else { i };
                let mut current = residues.load4(i);
                if ADVANCE {
                    let (enter_lo, enter_hi) =
                        squares(enter_row, low_halves(lanes.enter.as_ptr().add(at)));
                    let (leave_lo, leave_hi) =
                        squares(leave_row, low_halves(lanes.leave.as_ptr().add(at)));
                    let step_lo = vsubq_u64(enter_lo, leave_lo);
                    let step_hi = vsubq_u64(enter_hi, leave_hi);
                    // Descending, tick positions `at..at + 4` are lanes `i + 3` down to `i`.
                    let (step_lo, step_hi) = if DESC {
                        (
                            vextq_u64(step_hi, step_hi, 1),
                            vextq_u64(step_lo, step_lo, 1),
                        )
                    } else {
                        (step_lo, step_hi)
                    };
                    current = (vaddq_u64(current.0, step_lo), vaddq_u64(current.1, step_hi));
                    residues.store4(i, current);
                }
                if RANK {
                    let bytes = lanes
                        .complete
                        .as_ptr()
                        .add(at)
                        .cast::<u32>()
                        .read_unaligned();
                    let bytes = if DESC { bytes.swap_bytes() } else { bytes };
                    let masked = if bytes == 0x0101_0101 {
                        current
                    } else {
                        let (lo, hi) = masks(bytes);
                        (vorrq_u64(current.0, lo), vorrq_u64(current.1, hi))
                    };
                    let wins = if DESC {
                        (vcleq_u64(masked.0, best.0), vcleq_u64(masked.1, best.1))
                    } else {
                        (vcltq_u64(masked.0, best.0), vcltq_u64(masked.1, best.1))
                    };
                    best = (
                        vbslq_u64(wins.0, masked.0, best.0),
                        vbslq_u64(wins.1, masked.1, best.1),
                    );
                    best_lane = (
                        vbslq_u64(wins.0, index.0, best_lane.0),
                        vbslq_u64(wins.1, index.1, best_lane.1),
                    );
                    index = (vaddq_u64(index.0, step), vaddq_u64(index.1, step));
                    if COLUMNS && let Some(columns) = columns.as_deref_mut() {
                        let squared = columns.squared.as_mut_ptr().add(i);
                        let rows = columns.rows.as_mut_ptr().add(i);
                        let held = (vld1q_u64(squared), vld1q_u64(squared.add(2)));
                        let takes = (vcltq_u64(masked.0, held.0), vcltq_u64(masked.1, held.1));
                        vst1q_u64(squared, vbslq_u64(takes.0, masked.0, held.0));
                        vst1q_u64(squared.add(2), vbslq_u64(takes.1, masked.1, held.1));
                        let held_rows = (vld1q_u64(rows), vld1q_u64(rows.add(2)));
                        vst1q_u64(rows, vbslq_u64(takes.0, row_vec, held_rows.0));
                        vst1q_u64(rows.add(2), vbslq_u64(takes.1, row_vec, held_rows.1));
                    }
                }
                i += 4;
            }
            let mut merged = (NONE, usize::MAX);
            for (residue, lane) in [
                (vgetq_lane_u64(best.0, 0), vgetq_lane_u64(best_lane.0, 0)),
                (vgetq_lane_u64(best.0, 1), vgetq_lane_u64(best_lane.0, 1)),
                (vgetq_lane_u64(best.1, 0), vgetq_lane_u64(best_lane.1, 0)),
                (vgetq_lane_u64(best.1, 1), vgetq_lane_u64(best_lane.1, 1)),
            ] {
                if residue != NONE {
                    merged = nearer::<DESC>(merged, (residue, lane as usize));
                }
            }
            merged
        };
        let tail_best = scalar_lanes::<S, DESC, ADVANCE, RANK>(
            residues,
            row,
            lanes,
            columns,
            (vectored, len),
            (NONE, usize::MAX),
        );
        if !RANK {
            return None;
        }
        let best = if tail_best.0 == NONE {
            best
        } else if best.0 == NONE {
            tail_best
        } else {
            nearer::<DESC>(best, tail_best)
        };
        finish(best)
    }
}

#[cfg(test)]
mod tests;
