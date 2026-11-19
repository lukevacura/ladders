//! The non-normalized lane kernel on AVX2: the scalar body's recurrence four lanes at a time
//! under [`super::Fit::I32`], equal to it on bits. All of the `unsafe` for this kernel's AVX2
//! path is here; the caller checks [`crate::kernel_avx2::enabled`].

use super::{ColumnMins, Lanes, NONE, Nearest, Pass, Store, finish, scalar_lanes};
use crate::nonnorm::TickStep;
use core::arch::x86_64::*;

/// The nearer of two lane results, ties to the lane whose neighbor starts first.
#[inline(always)]
fn nearer<const DESC: bool>(a: (u64, usize), b: (u64, usize)) -> (u64, usize) {
    let b_wins = b.0 < a.0 || (b.0 == a.0 && if DESC { b.1 > a.1 } else { b.1 < a.1 });
    if b_wins { b } else { a }
}

/// Four lanes' residues as one vector, lane `i` first.
pub(super) trait VectorStore: Store {
    /// # Safety
    /// AVX2 is available and lanes `i..i + 4` exist.
    unsafe fn load4(&self, i: usize) -> __m256i;
    /// # Safety
    /// AVX2 is available and lanes `i..i + 4` exist.
    unsafe fn store4(&mut self, i: usize, residues: __m256i);
}

impl VectorStore for [u64] {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load4(&self, i: usize) -> __m256i {
        // SAFETY: the caller keeps `i + 4` within the slice.
        unsafe { _mm256_loadu_si256(self.as_ptr().add(i).cast::<__m256i>()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store4(&mut self, i: usize, residues: __m256i) {
        // SAFETY: the caller keeps `i + 4` within the slice.
        unsafe { _mm256_storeu_si256(self.as_mut_ptr().add(i).cast::<__m256i>(), residues) }
    }
}

impl VectorStore for [u128] {
    /// A `u128` is its low then its high `u64` (little-endian): two loads of two `u128`s each,
    /// their low halves gathered in lane order.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load4(&self, i: usize) -> __m256i {
        // SAFETY: the caller keeps `i + 4` within the slice; each load reads two `u128`s.
        unsafe {
            let at = self.as_ptr().add(i).cast::<__m256i>();
            let (a, b) = (_mm256_loadu_si256(at), _mm256_loadu_si256(at.add(1)));
            // `[lo0, lo2, lo1, lo3]`, then into lane order.
            _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_unpacklo_epi64(a, b))
        }
    }

    /// Each store writes two `u128`s, high halves cleared as `as_u128` clears them.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store4(&mut self, i: usize, residues: __m256i) {
        // SAFETY: the caller keeps `i + 4` within the slice.
        unsafe {
            let at = self.as_mut_ptr().add(i).cast::<__m256i>();
            let zero = _mm256_setzero_si256();
            let even = _mm256_unpacklo_epi64(residues, zero);
            let odd = _mm256_unpackhi_epi64(residues, zero);
            _mm256_storeu_si256(at, _mm256_permute2x128_si256::<0x20>(even, odd));
            _mm256_storeu_si256(at.add(1), _mm256_permute2x128_si256::<0x31>(even, odd));
        }
    }
}

/// The four lanes in reverse order.
#[inline]
#[target_feature(enable = "avx2")]
fn reversed(v: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0b00_01_10_11>(v)
}

/// `(a − b)²` for four lanes: the difference of two ticks that differ by less than `2³¹` has the
/// true difference as its low 32 bits (signed), and `_mm256_mul_epi32` squares that exactly.
#[inline]
#[target_feature(enable = "avx2")]
fn squares(a: __m256i, b: __m256i) -> __m256i {
    let d = _mm256_sub_epi64(a, b);
    _mm256_mul_epi32(d, d)
}

/// All ones in the lanes where `a < b`, as unsigned.
#[inline]
#[target_feature(enable = "avx2")]
fn less(a: __m256i, b: __m256i) -> __m256i {
    let flip = _mm256_set1_epi64x(i64::MIN);
    _mm256_cmpgt_epi64(_mm256_xor_si256(b, flip), _mm256_xor_si256(a, flip))
}

/// Four lanes at a time: the scalar body's recurrence restated with exact `_mm256_mul_epi32`
/// squares (under [`super::Fit::I32`]) and wrapping `u64` adds, the same value modulo `2⁶⁴` as
/// subtracting the leaving square and adding the entering one. Each vector position keeps its
/// own nearest `(residue, lane)`, displaced as the scalar body displaces, and the positions
/// merge only at the end; lanes past the last group of four run the scalar body.
///
/// # Safety
/// The CPU has AVX2, `fit` is [`super::Fit::I32`] for every tick of `row` and `lanes`, and
/// `lanes` and `columns` hold one element per residue.
#[inline]
#[target_feature(enable = "avx2")]
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
            (Pass::Advance, _) => body::<S, DESC, true, false, false>(residues, row, lanes, None),
            (Pass::Rank, None) => body::<S, DESC, false, true, false>(residues, row, lanes, None),
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
/// As [`row`], with `columns` present exactly when `COLUMNS`.
#[inline]
#[target_feature(enable = "avx2")]
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
    // tick positions (`i..i + 4`, or `len − 4 − i..len − i` descending), their completeness
    // bytes and their column slots are in bounds of slices the caller sized to `len`.
    let best = unsafe {
        let enter_row = _mm256_set1_epi64x(row.entering);
        let leave_row = _mm256_set1_epi64x(row.leaving);
        let row_vec = _mm256_set1_epi64x(columns.as_ref().map_or(0, |c| c.row) as i64);
        let zero = _mm256_setzero_si256();
        let ones = _mm256_set1_epi64x(-1);
        let step = _mm256_set1_epi64x(4);
        let mut index = _mm256_setr_epi64x(0, 1, 2, 3);
        let mut best = _mm256_set1_epi64x(NONE as i64);
        let mut best_lane = _mm256_set1_epi64x(-1);
        let mut i = 0;
        while i < vectored {
            let at = if DESC { len - 4 - i } else { i };
            let mut current = residues.load4(i);
            if ADVANCE {
                let ticks = |from: *const i64| _mm256_loadu_si256(from.add(at).cast::<__m256i>());
                let entering = squares(enter_row, ticks(lanes.enter.as_ptr()));
                let leaving = squares(leave_row, ticks(lanes.leave.as_ptr()));
                let stepped = _mm256_sub_epi64(entering, leaving);
                // Descending, tick positions `at..at + 4` are lanes `i + 3` down to `i`.
                let stepped = if DESC { reversed(stepped) } else { stepped };
                current = _mm256_add_epi64(current, stepped);
                residues.store4(i, current);
            }
            if RANK {
                let bytes = lanes
                    .complete
                    .as_ptr()
                    .add(at)
                    .cast::<u32>()
                    .read_unaligned();
                let masked = if bytes == 0x0101_0101 {
                    current
                } else {
                    let flags = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(bytes as i32));
                    let incomplete = _mm256_cmpeq_epi64(flags, zero);
                    let incomplete = if DESC {
                        reversed(incomplete)
                    } else {
                        incomplete
                    };
                    _mm256_or_si256(current, incomplete)
                };
                let wins = if DESC {
                    _mm256_xor_si256(less(best, masked), ones)
                } else {
                    less(masked, best)
                };
                best = _mm256_blendv_epi8(best, masked, wins);
                best_lane = _mm256_blendv_epi8(best_lane, index, wins);
                index = _mm256_add_epi64(index, step);
                if COLUMNS && let Some(columns) = columns.as_deref_mut() {
                    let squared = columns.squared.as_mut_ptr().add(i).cast::<__m256i>();
                    let rows = columns.rows.as_mut_ptr().add(i).cast::<__m256i>();
                    let held = _mm256_loadu_si256(squared);
                    let takes = less(masked, held);
                    _mm256_storeu_si256(squared, _mm256_blendv_epi8(held, masked, takes));
                    let held_rows = _mm256_loadu_si256(rows);
                    _mm256_storeu_si256(rows, _mm256_blendv_epi8(held_rows, row_vec, takes));
                }
            }
            i += 4;
        }
        let mut residue = [0u64; 4];
        let mut lane = [0u64; 4];
        _mm256_storeu_si256(residue.as_mut_ptr().cast::<__m256i>(), best);
        _mm256_storeu_si256(lane.as_mut_ptr().cast::<__m256i>(), best_lane);
        let mut merged = (NONE, usize::MAX);
        for (residue, lane) in residue.into_iter().zip(lane) {
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
