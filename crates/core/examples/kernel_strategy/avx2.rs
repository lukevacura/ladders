//! The AVX2 bodies of `kernel_strategy`, for timing only.
//!
//! [`rank1`] is a copy of `kernel_avx2::rank::<true, true>` (the body `rank_row_groups` calls
//! with floors): the same `Lanes4` step, two max accumulators alternating per step, a group
//! closed every `GROUP_LANES` lanes through `close`, the same remainder loop and the same
//! returned row maximum. It leaves out only the scalar tail, since the harness's lane counts are
//! multiples of the group size. [`rows`] is that body with `R` rows per pass (the 2-row sweep and
//! the 4-row spill check): `qt` loaded and stored once per four lanes, each row with its own
//! accumulators, hit vector and group reports. `CONV` picks the conversion: production's
//! full-range `to_f64_full`, the fast `to_f64_fast` (each copied here), or `to_f64_emulated`, the
//! old production conversion, which the library no longer has and is kept here only as the
//! reference the full-range one replaced.

use super::harness::Cols;
use core::arch::x86_64::*;
use ladde::kernel_neon::{GROUP_LANES, RowScalars};

pub const TARGET: &str = "AVX2 (x86_64)";

/// Whether the fast conversion variants are candidates here (on NEON they are a control only).
pub const FAST_IS_CANDIDATE: bool = true;

/// The conversion bodies, as the bodies' `CONV` parameter: the old production one (emulated by
/// lane-crossing permutes), the full-range one by two magic-constant halves (exact on every
/// `i64`), and the fast one (exact within `2^51`).
pub const OLD: u8 = 0;
pub const FULL: u8 = 1;
pub const FAST: u8 = 2;
/// The conversion production runs where the fast one does not fit: the full-range one.
pub const PROD: u8 = FULL;

/// The bits of `2^84 + 2^63`, xor'd onto the high half (`hi + 2^31` units of `2^32` above `2^84`).
pub const HIGH_MAGIC_BITS: u64 = 0x4530_0000_8000_0000;
/// The bits of `2^52`; its upper half replaces the low half's.
pub const LOW_MAGIC_BITS: u64 = 0x4330_0000_0000_0000;
/// `2^84 + 2^63 + 2^52`.
pub const HIGH_MAGIC: f64 = f64::from_bits(0x4530_0000_8010_0000);

/// The CPU has AVX2: detected, or known at compile time (`-C target-feature=+avx2`).
pub fn available() -> bool {
    cfg!(target_feature = "avx2") || std::is_x86_feature_detected!("avx2")
}

/// `x as f64` for four lanes, correctly rounded: the old production conversion (the library's
/// `kernel_avx2::to_f64` before full-range-convert replaced it), kept here as the "old
/// production" reference only. Both 32-bit halves go through `i32` conversions after
/// lane-crossing permutes, the low one shifted into `i32` range and back, and one add rounds.
#[target_feature(enable = "avx2")]
#[inline]
fn to_f64_emulated(x: __m256i) -> __m256d {
    let low_halves = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let high_halves = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
    let low = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(x, low_halves));
    let high = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(x, high_halves));
    let low = _mm256_add_pd(
        _mm256_cvtepi32_pd(_mm_xor_si128(low, _mm_set1_epi32(i32::MIN))),
        _mm256_set1_pd(2_147_483_648.0),
    );
    let high = _mm256_mul_pd(_mm256_cvtepi32_pd(high), _mm256_set1_pd(4_294_967_296.0));
    _mm256_add_pd(high, low)
}

/// The bits of `1.5 * 2^52`: sign 0, biased exponent `0x433` (2^52), fraction field `2^51`.
pub const MAGIC_BITS: i64 = 0x4338_0000_0000_0000;
/// `1.5 * 2^52` as `f64`.
pub const MAGIC: f64 = 6_755_399_441_055_744.0;

/// `x as f64` for four lanes with `-2^51 <= x < 2^51`, by the magic constant: one integer add and
/// one `f64` subtract.
///
/// Proof that it equals `x as f64` bit for bit on that range. Adding `x` to [`MAGIC_BITS`] as
/// integers adds it to the 52-bit fraction field, which holds `2^51`: `2^51 + x` lies in
/// `[0, 2^52)`, so nothing carries into or borrows from the exponent, and the sum is the bit
/// pattern of the double `2^52 + (2^51 + x) = 1.5 * 2^52 + x`, exact because the spacing of
/// doubles in `[2^52, 2^53)` is 1. Subtracting [`MAGIC`] has the exact result `x`, an integer
/// of magnitude at most `2^51`, which is a double; an IEEE subtraction returns a representable
/// exact result unrounded, and for `x = 0` it returns `+0.0` (round to nearest). `x as f64` is
/// exact for `|x| <= 2^53` and gives `+0.0` for 0, so the two agree on every bit. Outside the
/// range the result is wrong (another exponent), so the route needs a proved bound on `|x|`.
#[target_feature(enable = "avx2")]
#[inline]
fn to_f64_fast(x: __m256i) -> __m256d {
    let biased = _mm256_add_epi64(x, _mm256_set1_epi64x(MAGIC_BITS));
    _mm256_sub_pd(_mm256_castsi256_pd(biased), _mm256_set1_pd(MAGIC))
}

/// `x as f64` for four lanes, every `i64` (copy of `kernel_avx2::to_f64_full`, whose proof is in
/// the library's `full_range` module): the high half shifted down and xor'd with the bits of
/// `2^84 + 2^63` is exactly `2^84 + 2^63 + hi * 2^32`; the low half blended under the upper half
/// of `2^52` is exactly `2^52 + lo`; subtracting `2^84 + 2^63 + 2^52` is exact (Sterbenz) and
/// the one add rounds `x` correctly. No permute, no `i32` conversion.
#[target_feature(enable = "avx2")]
#[inline]
fn to_f64_full(x: __m256i) -> __m256d {
    let high = _mm256_xor_si256(
        _mm256_srli_epi64::<32>(x),
        _mm256_set1_epi64x(HIGH_MAGIC_BITS as i64),
    );
    let low = _mm256_blend_epi32::<0b1010_1010>(x, _mm256_set1_epi64x(LOW_MAGIC_BITS as i64));
    _mm256_add_pd(
        _mm256_sub_pd(_mm256_castsi256_pd(high), _mm256_set1_pd(HIGH_MAGIC)),
        _mm256_castsi256_pd(low),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn convert<const CONV: u8>(x: __m256i) -> __m256d {
    match CONV {
        FAST => to_f64_fast(x),
        FULL => to_f64_full(x),
        _ => to_f64_emulated(x),
    }
}

/// [`to_f64_full`] on four values, for the check against `as f64`.
pub fn convert_full(x: [i64; 4]) -> [f64; 4] {
    assert!(available(), "AVX2 is required");
    let mut out = [0.0; 4];
    // SAFETY: AVX2 was just checked; the arrays hold four `i64`s and four `f64`s.
    unsafe { store_full(x.as_ptr(), out.as_mut_ptr()) };
    out
}

#[target_feature(enable = "avx2")]
unsafe fn store_full(from: *const i64, to: *mut f64) {
    _mm256_storeu_pd(to, to_f64_full(_mm256_loadu_si256(from.cast::<__m256i>())))
}

/// [`to_f64_fast`] on four values, for the check against `as f64`.
pub fn convert_fast(x: [i64; 4]) -> [f64; 4] {
    assert!(available(), "AVX2 is required");
    let mut out = [0.0; 4];
    // SAFETY: AVX2 was just checked; the arrays hold four `i64`s and four `f64`s.
    unsafe { store_fast(x.as_ptr(), out.as_mut_ptr()) };
    out
}

#[target_feature(enable = "avx2")]
unsafe fn store_fast(from: *const i64, to: *mut f64) {
    _mm256_storeu_pd(to, to_f64_fast(_mm256_loadu_si256(from.cast::<__m256i>())))
}

/// The lane slices of one row, already offset to its first lag, as production's `Lanes`.
pub struct Lanes<'a> {
    pub enter: &'a [i32],
    pub leave: &'a [i32],
    pub sum: &'a [i32],
    pub r: &'a [f64],
    pub bias: &'a [f64],
}

/// Copy of `kernel_avx2::Lanes4`.
struct Lanes4 {
    qt: *mut i64,
    enter: *const i32,
    leave: *const i32,
    sum: *const i32,
    r: *const f64,
    bias: *const f64,
    out: Option<*mut f64>,
    floor: *const f64,
    me: __m256i,
    ml: __m256i,
    si: __m256i,
    ri: __m256d,
}

impl Lanes4 {
    /// Copy of `kernel_avx2::Lanes4::rank`, with the conversion chosen by `CONV`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rank<const FLOORS: bool, const CONV: u8>(
        &self,
        c: usize,
        acc: &mut __m256d,
        hit: &mut __m256d,
    ) {
        let widen =
            |at: *const i32| _mm256_cvtepi32_epi64(_mm_loadu_si128(at.add(c).cast::<__m128i>()));
        let qt = self.qt.add(c).cast::<__m256i>();
        let mut q = _mm256_loadu_si256(qt);
        q = _mm256_add_epi64(q, _mm256_mul_epi32(self.me, widen(self.enter)));
        q = _mm256_sub_epi64(q, _mm256_mul_epi32(self.ml, widen(self.leave)));
        _mm256_storeu_si256(qt, q);
        let x = _mm256_sub_epi64(q, _mm256_mul_epi32(self.si, widen(self.sum)));
        let scaled = _mm256_mul_pd(
            _mm256_mul_pd(convert::<CONV>(x), self.ri),
            _mm256_loadu_pd(self.r.add(c)),
        );
        let value = _mm256_add_pd(scaled, _mm256_loadu_pd(self.bias.add(c)));
        if let Some(out) = self.out {
            _mm256_storeu_pd(out.add(c), value);
        }
        if FLOORS {
            let at = _mm256_cmp_pd::<_CMP_GE_OQ>(value, _mm256_loadu_pd(self.floor.add(c)));
            *hit = _mm256_or_pd(*hit, at);
        }
        *acc = _mm256_max_pd(*acc, value);
    }
}

/// Copy of `kernel_avx2::horizontal_max`.
#[target_feature(enable = "avx2")]
#[inline]
fn horizontal_max(v: __m256d) -> f64 {
    let mut lanes = [0.0f64; 4];
    // SAFETY: the array holds four `f64`s.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), v) };
    lanes.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// Copy of `kernel_avx2::rank::<GROUPS, FLOORS>` without the scalar tail; returns the row's
/// maximum and the hit mask.
///
/// # Safety
/// AVX2 is available; every lane slice, `scratch` and `floors` hold `qt_m.len()` elements, a
/// multiple of 4 (and of `GROUP_LANES` for exact group reports); `maxima` holds one slot per
/// group, at most 64 of them.
#[target_feature(enable = "avx2")]
pub unsafe fn rank1<const GROUPS: bool, const FLOORS: bool, const CONV: u8>(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    mut scratch: Option<&mut [f64]>,
    floors: &[f64],
    maxima: &mut [f64],
) -> (f64, u64) {
    let len = qt_m.len();
    let vectored = len / 4 * 4;
    debug_assert_eq!(vectored, len, "the harness has no scalar tail");
    let scratch_ptr: Option<*mut f64> = scratch.as_mut().map(|s| s.as_mut_ptr());
    let mut hits = 0u64;
    let at = Lanes4 {
        qt: qt_m.as_mut_ptr(),
        enter: lanes.enter.as_ptr(),
        leave: lanes.leave.as_ptr(),
        sum: lanes.sum.as_ptr(),
        r: lanes.r.as_ptr(),
        bias: lanes.bias.as_ptr(),
        out: scratch_ptr,
        floor: floors.as_ptr(),
        me: _mm256_set1_epi64x(i64::from(row.me)),
        ml: _mm256_set1_epi64x(i64::from(row.ml)),
        si: _mm256_set1_epi64x(i64::from(row.si)),
        ri: _mm256_set1_pd(row.ri),
    };
    let ninf = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut row_max = f64::NEG_INFINITY;
    let mut c = 0usize;
    let mut group = 0usize;
    let mut close = |acc: [__m256d; 2], hit: __m256d| {
        let group_max = horizontal_max(_mm256_max_pd(acc[0], acc[1]));
        row_max = row_max.max(group_max);
        if GROUPS {
            *maxima.get_unchecked_mut(group) = group_max;
            if FLOORS && _mm256_movemask_pd(hit) != 0 {
                hits |= 1 << group;
            }
            group += 1;
        }
    };
    if GROUPS {
        while c + GROUP_LANES <= vectored {
            let mut acc = [ninf; 2];
            let mut hit = _mm256_setzero_pd();
            for step in 0..GROUP_LANES / 4 {
                at.rank::<FLOORS, CONV>(c + 4 * step, &mut acc[step % 2], &mut hit);
            }
            c += GROUP_LANES;
            close(acc, hit);
        }
    }
    if c < vectored {
        let mut acc = [ninf; 2];
        let mut hit = _mm256_setzero_pd();
        let mut step = 0;
        while c < vectored {
            at.rank::<FLOORS, CONV>(c, &mut acc[step % 2], &mut hit);
            c += 4;
            step += 1;
        }
        close(acc, hit);
    }
    (row_max, hits)
}

/// One row's scalars as vectors, for [`rows`].
#[derive(Clone, Copy)]
struct Row4 {
    me: __m256i,
    ml: __m256i,
    si: __m256i,
    ri: __m256d,
}

/// [`rank1`] with groups and floors for `R` rows per pass: per four lanes `qt` is loaded once,
/// advanced by each row in turn (row `t` reads the columns from `col0 + t`), each row's value
/// taken right after its own step, and `qt` stored once. Each row keeps two accumulators
/// alternating per step and its own hit vector, and reports its group maxima and hit mask.
///
/// # Safety
/// AVX2 is available; `qt.len()` is a multiple of `GROUP_LANES`; each tile holds `qt.len()`
/// elements, each `maxima` one slot per group; the columns hold `col0 + R - 1 + qt.len()`
/// elements.
#[target_feature(enable = "avx2")]
pub unsafe fn rows<const R: usize, const CONV: u8>(
    qt: &mut [i64],
    cols: &Cols,
    col0: usize,
    rows: &[RowScalars; R],
    outs: &mut [Vec<f64>; R],
    maxima: &mut [Vec<f64>; R],
) -> [u64; R] {
    let vrows: [Row4; R] = core::array::from_fn(|t| Row4 {
        me: _mm256_set1_epi64x(i64::from(rows[t].me)),
        ml: _mm256_set1_epi64x(i64::from(rows[t].ml)),
        si: _mm256_set1_epi64x(i64::from(rows[t].si)),
        ri: _mm256_set1_pd(rows[t].ri),
    });
    let enter = cols.enter.as_ptr().add(col0);
    let leave = cols.leave.as_ptr().add(col0);
    let sum = cols.sum.as_ptr().add(col0);
    let r = cols.r.as_ptr().add(col0);
    let bias = cols.bias.as_ptr().add(col0);
    let floor = cols.floor.as_ptr().add(col0);
    let out: [*mut f64; R] = core::array::from_fn(|t| outs[t].as_mut_ptr());
    let qtp = qt.as_mut_ptr();
    let widen = |at: *const i32| _mm256_cvtepi32_epi64(_mm_loadu_si128(at.cast::<__m128i>()));
    let ninf = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut hits = [0u64; R];
    let mut c = 0usize;
    let mut group = 0usize;
    while c + GROUP_LANES <= qt.len() {
        let mut acc = [[ninf; 2]; R];
        let mut hit = [_mm256_setzero_pd(); R];
        for step in 0..GROUP_LANES / 4 {
            let l = c + 4 * step;
            let qp = qtp.add(l).cast::<__m256i>();
            let mut q = _mm256_loadu_si256(qp);
            for t in 0..R {
                let row = &vrows[t];
                let k = l + t;
                q = _mm256_add_epi64(q, _mm256_mul_epi32(row.me, widen(enter.add(k))));
                q = _mm256_sub_epi64(q, _mm256_mul_epi32(row.ml, widen(leave.add(k))));
                let x = _mm256_sub_epi64(q, _mm256_mul_epi32(row.si, widen(sum.add(k))));
                let scaled = _mm256_mul_pd(
                    _mm256_mul_pd(convert::<CONV>(x), row.ri),
                    _mm256_loadu_pd(r.add(k)),
                );
                let value = _mm256_add_pd(scaled, _mm256_loadu_pd(bias.add(k)));
                _mm256_storeu_pd(out[t].add(l), value);
                let at = _mm256_cmp_pd::<_CMP_GE_OQ>(value, _mm256_loadu_pd(floor.add(k)));
                hit[t] = _mm256_or_pd(hit[t], at);
                acc[t][step % 2] = _mm256_max_pd(acc[t][step % 2], value);
            }
            _mm256_storeu_si256(qp, q);
        }
        c += GROUP_LANES;
        for t in 0..R {
            *maxima[t].get_unchecked_mut(group) =
                horizontal_max(_mm256_max_pd(acc[t][0], acc[t][1]));
            if _mm256_movemask_pd(hit[t]) != 0 {
                hits[t] |= 1 << group;
            }
        }
        group += 1;
    }
    hits
}

/// One row with the `f32` estimate, eight lanes per step: the same `qt` step on two vectors of
/// four, the `i64` lanes converted to `f64` as the production body does where the fast
/// conversion does not fit (full range) and narrowed to `f32`, then the chain on `f32` columns into an `f32` tile.
///
/// # Safety
/// As [`rows`] for one row.
#[target_feature(enable = "avx2")]
pub unsafe fn row32(
    qt: &mut [i64],
    cols: &Cols,
    col0: usize,
    row: RowScalars,
    out: &mut [f32],
    maxima: &mut [f32],
) -> u64 {
    let widen = |at: *const i32| _mm256_cvtepi32_epi64(_mm_loadu_si128(at.cast::<__m128i>()));
    let max8 = |v: __m256| {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), v);
        lanes.into_iter().fold(f32::NEG_INFINITY, f32::max)
    };
    let me = _mm256_set1_epi64x(i64::from(row.me));
    let ml = _mm256_set1_epi64x(i64::from(row.ml));
    let si = _mm256_set1_epi64x(i64::from(row.si));
    let ri = _mm256_set1_ps(row.ri as f32);
    let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc = [ninf; 2];
    let mut hit = _mm256_setzero_ps();
    let mut mask = 0u64;
    let lanes = qt.len();
    let qt = qt.as_mut_ptr();
    let out = out.as_mut_ptr();
    let mut c = 0;
    let mut group = 0;
    while c < lanes {
        for step in 0..8 {
            let at = c + 8 * step;
            let col = at + col0;
            let mut x = [_mm_setzero_ps(); 2];
            for half in 0..2 {
                let off = 4 * half;
                let qp = qt.add(at + off).cast::<__m256i>();
                let mut q = _mm256_loadu_si256(qp);
                q = _mm256_add_epi64(
                    q,
                    _mm256_mul_epi32(me, widen(cols.enter.as_ptr().add(col + off))),
                );
                q = _mm256_sub_epi64(
                    q,
                    _mm256_mul_epi32(ml, widen(cols.leave.as_ptr().add(col + off))),
                );
                _mm256_storeu_si256(qp, q);
                let d = _mm256_sub_epi64(
                    q,
                    _mm256_mul_epi32(si, widen(cols.sum.as_ptr().add(col + off))),
                );
                x[half] = _mm256_cvtpd_ps(to_f64_full(d));
            }
            let x32 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(x[0]), x[1]);
            let scaled = _mm256_mul_ps(
                _mm256_mul_ps(x32, ri),
                _mm256_loadu_ps(cols.r32.as_ptr().add(col)),
            );
            let value = _mm256_add_ps(scaled, _mm256_loadu_ps(cols.bias32.as_ptr().add(col)));
            _mm256_storeu_ps(out.add(at), value);
            let ge =
                _mm256_cmp_ps::<_CMP_GE_OQ>(value, _mm256_loadu_ps(cols.floor32.as_ptr().add(col)));
            hit = _mm256_or_ps(hit, ge);
            acc[step % 2] = _mm256_max_ps(acc[step % 2], value);
        }
        c += 64;
        maxima[group] = max8(_mm256_max_ps(acc[0], acc[1]));
        acc = [ninf; 2];
        mask |= u64::from(_mm256_movemask_ps(hit) != 0) << group;
        hit = _mm256_setzero_ps();
        group += 1;
    }
    mask
}
