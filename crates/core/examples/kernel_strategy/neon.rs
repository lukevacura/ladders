//! The NEON bodies of `kernel_strategy`, for timing only.
//!
//! [`rank1`] is a copy of `kernel_neon::rank_neon::<true, true>` (the body `rank_row_groups`
//! inlines with floors): the same `Lanes8` step of eight lanes in two halves of four, four max
//! accumulators, a group closed every `GROUP_LANES` lanes through `close`, the same remainder
//! loop and the same returned row maximum. It leaves out only the scalar tail, since the
//! harness's lane counts are multiples of the group size. [`rows`] is that body with `R` rows per
//! pass (the 2-row sweep and the 4-row spill check): `qt` loaded and stored once per four lanes,
//! each row with its own four accumulators, hit vector and group reports. `CONV` picks the
//! conversion: `vcvtq_f64_s64` (production, now and before full-range-convert), [`to_f64_full`]
//! or [`to_f64_fast`]; NEON converts natively, so the last two are controls here.

use super::harness::Cols;
use core::arch::aarch64::*;
use ladde::kernel_neon::{GROUP_LANES, RowScalars};

pub const TARGET: &str = "NEON (aarch64)";

/// Whether the fast conversion variants are candidates here (on NEON they are a control only).
pub const FAST_IS_CANDIDATE: bool = false;

/// The conversion bodies, as the bodies' `CONV` parameter: production's own (native), the
/// full-range one by two magic-constant halves (exact on every `i64`), and the fast one (exact
/// within `2^51`).
pub const NATIVE: u8 = 0;
pub const FULL: u8 = 1;
pub const FAST: u8 = 2;
/// The conversion production runs: native, as it was before full-range-convert.
pub const PROD: u8 = NATIVE;
/// The old production conversion, the AVX2 harness's reference: native here too.
pub const OLD: u8 = NATIVE;

/// The bits of `2^84 + 2^63`, xor'd onto the high half (`hi + 2^31` units of `2^32` above `2^84`).
pub const HIGH_MAGIC_BITS: u64 = 0x4530_0000_8000_0000;
/// The bits of `2^52`; its upper half replaces the low half's.
pub const LOW_MAGIC_BITS: u64 = 0x4330_0000_0000_0000;
/// `2^84 + 2^63 + 2^52`.
pub const HIGH_MAGIC: f64 = f64::from_bits(0x4530_0000_8010_0000);

pub fn available() -> bool {
    true
}

/// The bits of `1.5 * 2^52`: sign 0, biased exponent `0x433` (2^52), fraction field `2^51`.
pub const MAGIC_BITS: i64 = 0x4338_0000_0000_0000;
/// `1.5 * 2^52` as `f64`.
pub const MAGIC: f64 = 6_755_399_441_055_744.0;

/// `x as f64` for two lanes with `-2^51 <= x < 2^51`, by the magic constant. The proof that it
/// equals `x as f64` bit for bit on that range is the one beside the AVX2 copy: the integer add
/// puts `2^51 + x`, in `[0, 2^52)`, in the fraction field of `2^52` without touching the
/// exponent, giving exactly `1.5 * 2^52 + x`, and the exact difference `x` is representable, so
/// the subtraction returns it unrounded (`+0.0` for 0).
#[inline(always)]
unsafe fn to_f64_fast(x: int64x2_t) -> float64x2_t {
    let biased = vaddq_s64(x, vdupq_n_s64(MAGIC_BITS));
    vsubq_f64(vreinterpretq_f64_s64(biased), vdupq_n_f64(MAGIC))
}

/// `x as f64` for two lanes, every `i64`, by two magic-constant halves: the NEON image of
/// `kernel_avx2::to_f64_full` (whose proof is in the library's `full_range` module). The high
/// half shifted down and xor'd with the bits of `2^84 + 2^63` is exactly `2^84 + 2^63 + hi * 2^32`;
/// the low half under the upper half of `2^52` is exactly `2^52 + lo`; subtracting
/// `2^84 + 2^63 + 2^52` is exact (Sterbenz) and the one add rounds `x` correctly.
#[inline(always)]
unsafe fn to_f64_full(x: int64x2_t) -> float64x2_t {
    let lane = vreinterpretq_u64_s64(x);
    let high = veorq_u64(vshrq_n_u64::<32>(lane), vdupq_n_u64(HIGH_MAGIC_BITS));
    let low = vorrq_u64(
        vandq_u64(lane, vdupq_n_u64(0xFFFF_FFFF)),
        vdupq_n_u64(LOW_MAGIC_BITS),
    );
    vaddq_f64(
        vsubq_f64(vreinterpretq_f64_u64(high), vdupq_n_f64(HIGH_MAGIC)),
        vreinterpretq_f64_u64(low),
    )
}

#[inline(always)]
unsafe fn convert<const CONV: u8>(x: int64x2_t) -> float64x2_t {
    match CONV {
        FAST => to_f64_fast(x),
        FULL => to_f64_full(x),
        _ => vcvtq_f64_s64(x),
    }
}

/// [`to_f64_full`] on four values, for the check against `as f64`.
pub fn convert_full(x: [i64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    // SAFETY: NEON is part of aarch64; the arrays hold four `i64`s and four `f64`s.
    unsafe {
        vst1q_f64(out.as_mut_ptr(), to_f64_full(vld1q_s64(x.as_ptr())));
        vst1q_f64(
            out.as_mut_ptr().add(2),
            to_f64_full(vld1q_s64(x.as_ptr().add(2))),
        );
    }
    out
}

/// [`to_f64_fast`] on four values, for the check against `as f64`.
pub fn convert_fast(x: [i64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    // SAFETY: NEON is part of aarch64; the arrays hold four `i64`s and four `f64`s.
    unsafe {
        vst1q_f64(out.as_mut_ptr(), to_f64_fast(vld1q_s64(x.as_ptr())));
        vst1q_f64(
            out.as_mut_ptr().add(2),
            to_f64_fast(vld1q_s64(x.as_ptr().add(2))),
        );
    }
    out
}

/// The lane slices of one row, already offset to its first lag, as production's `Lanes`.
pub struct Lanes<'a> {
    pub enter: &'a [i32],
    pub leave: &'a [i32],
    pub sum: &'a [i32],
    pub r: &'a [f64],
    pub bias: &'a [f64],
}

/// Copy of `kernel_neon::Lanes8`.
struct Lanes8 {
    qt: *mut i64,
    enter: *const i32,
    leave: *const i32,
    sum: *const i32,
    r: *const f64,
    bias: *const f64,
    out: Option<*mut f64>,
    floor: *const f64,
    me: int32x4_t,
    ml: int32x4_t,
    si: int32x4_t,
    ri: float64x2_t,
}

impl Lanes8 {
    /// Copy of `kernel_neon::Lanes8::rank`, with the conversion chosen by `CONV`.
    #[inline(always)]
    unsafe fn rank<const FLOORS: bool, const CONV: u8>(
        &self,
        c: usize,
        acc: &mut [float64x2_t; 4],
        hit: &mut uint64x2_t,
    ) {
        for h in 0..2 {
            let j = c + 4 * h;

            let ve = vld1q_s32(self.enter.add(j));
            let vl = vld1q_s32(self.leave.add(j));
            let vs = vld1q_s32(self.sum.add(j));

            let mut lo = vld1q_s64(self.qt.add(j));
            let mut hi = vld1q_s64(self.qt.add(j + 2));
            lo = vaddq_s64(lo, vmull_s32(vget_low_s32(ve), vget_low_s32(self.me)));
            hi = vaddq_s64(hi, vmull_high_s32(ve, self.me));
            lo = vsubq_s64(lo, vmull_s32(vget_low_s32(vl), vget_low_s32(self.ml)));
            hi = vsubq_s64(hi, vmull_high_s32(vl, self.ml));
            vst1q_s64(self.qt.add(j), lo);
            vst1q_s64(self.qt.add(j + 2), hi);

            let xlo = vsubq_s64(lo, vmull_s32(vget_low_s32(vs), vget_low_s32(self.si)));
            let xhi = vsubq_s64(hi, vmull_high_s32(vs, self.si));

            let rlo = vld1q_f64(self.r.add(j));
            let rhi = vld1q_f64(self.r.add(j + 2));
            let blo = vld1q_f64(self.bias.add(j));
            let bhi = vld1q_f64(self.bias.add(j + 2));

            let flo = vaddq_f64(
                vmulq_f64(vmulq_f64(convert::<CONV>(xlo), self.ri), rlo),
                blo,
            );
            let fhi = vaddq_f64(
                vmulq_f64(vmulq_f64(convert::<CONV>(xhi), self.ri), rhi),
                bhi,
            );

            if let Some(out) = self.out {
                vst1q_f64(out.add(j), flo);
                vst1q_f64(out.add(j + 2), fhi);
            }
            if FLOORS {
                let at_lo = vcgeq_f64(flo, vld1q_f64(self.floor.add(j)));
                let at_hi = vcgeq_f64(fhi, vld1q_f64(self.floor.add(j + 2)));
                *hit = vorrq_u64(*hit, vorrq_u64(at_lo, at_hi));
            }

            acc[2 * h] = vmaxq_f64(acc[2 * h], flo);
            acc[2 * h + 1] = vmaxq_f64(acc[2 * h + 1], fhi);
        }
    }
}

/// Copy of `kernel_neon::rank_neon::<GROUPS, FLOORS>` without the scalar tail; returns the
/// row's maximum and the hit mask.
///
/// # Safety
/// Every lane slice, `scratch` and `floors` hold `qt_m.len()` elements, a multiple of 8 (and of
/// `GROUP_LANES` for exact group reports); `maxima` holds one slot per group, at most 64 of them.
#[inline(always)]
pub unsafe fn rank1<const GROUPS: bool, const FLOORS: bool, const CONV: u8>(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    mut scratch: Option<&mut [f64]>,
    floors: &[f64],
    maxima: &mut [f64],
) -> (f64, u64) {
    let len = qt_m.len();
    let vectored = len / 8 * 8;
    debug_assert_eq!(vectored, len, "the harness has no scalar tail");
    let scratch_ptr: Option<*mut f64> = scratch.as_mut().map(|s| s.as_mut_ptr());
    let floor_ptr = floors.as_ptr();
    let mut hits = 0u64;
    let at = Lanes8 {
        qt: qt_m.as_mut_ptr(),
        enter: lanes.enter.as_ptr(),
        leave: lanes.leave.as_ptr(),
        sum: lanes.sum.as_ptr(),
        r: lanes.r.as_ptr(),
        bias: lanes.bias.as_ptr(),
        out: scratch_ptr,
        floor: floor_ptr,
        me: vdupq_n_s32(row.me),
        ml: vdupq_n_s32(row.ml),
        si: vdupq_n_s32(row.si),
        ri: vdupq_n_f64(row.ri),
    };
    let mut row_max = f64::NEG_INFINITY;
    let mut c = 0usize;
    let mut group = 0usize;
    let mut close = |acc: [float64x2_t; 4], hit: uint64x2_t| {
        let merged = vmaxq_f64(vmaxq_f64(acc[0], acc[1]), vmaxq_f64(acc[2], acc[3]));
        let group_max = vmaxvq_f64(merged);
        row_max = row_max.max(group_max);
        if GROUPS {
            *maxima.get_unchecked_mut(group) = group_max;
            if FLOORS && vmaxvq_u32(vreinterpretq_u32_u64(hit)) != 0 {
                hits |= 1 << group;
            }
            group += 1;
        }
    };
    if GROUPS {
        while c + GROUP_LANES <= vectored {
            let mut acc = [vdupq_n_f64(f64::NEG_INFINITY); 4];
            let mut hit = vdupq_n_u64(0);
            for step in 0..GROUP_LANES / 8 {
                at.rank::<FLOORS, CONV>(c + 8 * step, &mut acc, &mut hit);
            }
            c += GROUP_LANES;
            close(acc, hit);
        }
    }
    if c < vectored {
        let mut acc = [vdupq_n_f64(f64::NEG_INFINITY); 4];
        let mut hit = vdupq_n_u64(0);
        while c < vectored {
            at.rank::<FLOORS, CONV>(c, &mut acc, &mut hit);
            c += 8;
        }
        close(acc, hit);
    }
    (row_max, hits)
}

/// One row's scalars as vectors, for [`rows`].
struct Row8 {
    me: int32x4_t,
    ml: int32x4_t,
    si: int32x4_t,
    ri: float64x2_t,
}

/// [`rank1`] with groups and floors for `R` rows per pass: per four lanes `qt` is loaded once,
/// advanced by each row in turn (row `t` reads the columns from `col0 + t`), each row's values
/// taken right after its own step, and `qt` stored once. Each row keeps four accumulators and its
/// own hit vector, and reports its group maxima and hit mask.
///
/// # Safety
/// `qt.len()` is a multiple of `GROUP_LANES`; each tile holds `qt.len()` elements, each `maxima`
/// one slot per group; the columns hold `col0 + R - 1 + qt.len()` elements.
pub unsafe fn rows<const R: usize, const CONV: u8>(
    qt: &mut [i64],
    cols: &Cols,
    col0: usize,
    rows: &[RowScalars; R],
    outs: &mut [Vec<f64>; R],
    maxima: &mut [Vec<f64>; R],
) -> [u64; R] {
    let vrows: [Row8; R] = core::array::from_fn(|t| Row8 {
        me: vdupq_n_s32(rows[t].me),
        ml: vdupq_n_s32(rows[t].ml),
        si: vdupq_n_s32(rows[t].si),
        ri: vdupq_n_f64(rows[t].ri),
    });
    let enter = cols.enter.as_ptr().add(col0);
    let leave = cols.leave.as_ptr().add(col0);
    let sum = cols.sum.as_ptr().add(col0);
    let r = cols.r.as_ptr().add(col0);
    let bias = cols.bias.as_ptr().add(col0);
    let floor = cols.floor.as_ptr().add(col0);
    let out: [*mut f64; R] = core::array::from_fn(|t| outs[t].as_mut_ptr());
    let qtp = qt.as_mut_ptr();
    let ninf = vdupq_n_f64(f64::NEG_INFINITY);
    let mut hits = [0u64; R];
    let mut c = 0usize;
    let mut group = 0usize;
    while c + GROUP_LANES <= qt.len() {
        let mut acc = [[ninf; 4]; R];
        let mut hit = [vdupq_n_u64(0); R];
        for step in 0..GROUP_LANES / 8 {
            for h in 0..2 {
                let j = c + 8 * step + 4 * h;
                let mut lo = vld1q_s64(qtp.add(j));
                let mut hi = vld1q_s64(qtp.add(j + 2));
                for t in 0..R {
                    let row = &vrows[t];
                    let k = j + t;
                    let ve = vld1q_s32(enter.add(k));
                    let vl = vld1q_s32(leave.add(k));
                    let vs = vld1q_s32(sum.add(k));
                    lo = vaddq_s64(lo, vmull_s32(vget_low_s32(ve), vget_low_s32(row.me)));
                    hi = vaddq_s64(hi, vmull_high_s32(ve, row.me));
                    lo = vsubq_s64(lo, vmull_s32(vget_low_s32(vl), vget_low_s32(row.ml)));
                    hi = vsubq_s64(hi, vmull_high_s32(vl, row.ml));
                    let xlo = vsubq_s64(lo, vmull_s32(vget_low_s32(vs), vget_low_s32(row.si)));
                    let xhi = vsubq_s64(hi, vmull_high_s32(vs, row.si));
                    let rlo = vld1q_f64(r.add(k));
                    let rhi = vld1q_f64(r.add(k + 2));
                    let blo = vld1q_f64(bias.add(k));
                    let bhi = vld1q_f64(bias.add(k + 2));
                    let flo =
                        vaddq_f64(vmulq_f64(vmulq_f64(convert::<CONV>(xlo), row.ri), rlo), blo);
                    let fhi =
                        vaddq_f64(vmulq_f64(vmulq_f64(convert::<CONV>(xhi), row.ri), rhi), bhi);
                    vst1q_f64(out[t].add(j), flo);
                    vst1q_f64(out[t].add(j + 2), fhi);
                    let at_lo = vcgeq_f64(flo, vld1q_f64(floor.add(k)));
                    let at_hi = vcgeq_f64(fhi, vld1q_f64(floor.add(k + 2)));
                    hit[t] = vorrq_u64(hit[t], vorrq_u64(at_lo, at_hi));
                    acc[t][2 * h] = vmaxq_f64(acc[t][2 * h], flo);
                    acc[t][2 * h + 1] = vmaxq_f64(acc[t][2 * h + 1], fhi);
                }
                vst1q_s64(qtp.add(j), lo);
                vst1q_s64(qtp.add(j + 2), hi);
            }
        }
        c += GROUP_LANES;
        for t in 0..R {
            let a = &acc[t];
            let merged = vmaxq_f64(vmaxq_f64(a[0], a[1]), vmaxq_f64(a[2], a[3]));
            *maxima[t].get_unchecked_mut(group) = vmaxvq_f64(merged);
            if vmaxvq_u32(vreinterpretq_u32_u64(hit[t])) != 0 {
                hits[t] |= 1 << group;
            }
        }
        group += 1;
    }
    hits
}

/// One row with the `f32` estimate: the same `qt` step, the `i64` lane converted to `f64` as the
/// production body does and narrowed to `f32`, then the chain on `f32` columns into an `f32` tile.
///
/// # Safety
/// As [`rows`] for one row.
pub unsafe fn row32(
    qt: &mut [i64],
    cols: &Cols,
    col0: usize,
    row: RowScalars,
    out: &mut [f32],
    maxima: &mut [f32],
) -> u64 {
    let me = vdupq_n_s32(row.me);
    let ml = vdupq_n_s32(row.ml);
    let si = vdupq_n_s32(row.si);
    let ri = vdupq_n_f32(row.ri as f32);
    let ninf = vdupq_n_f32(f32::NEG_INFINITY);
    let mut acc = ninf;
    let mut hit = vdupq_n_u32(0);
    let mut mask = 0u64;
    let lanes = qt.len();
    let qt = qt.as_mut_ptr();
    let out = out.as_mut_ptr();
    let mut j = 0;
    let mut group = 0;
    while j < lanes {
        for _ in 0..8 {
            for h in 0..2 {
                let i = j + 4 * h;
                let at = i + col0;
                let ve = vld1q_s32(cols.enter.as_ptr().add(at));
                let vl = vld1q_s32(cols.leave.as_ptr().add(at));
                let vs = vld1q_s32(cols.sum.as_ptr().add(at));
                let mut lo = vld1q_s64(qt.add(i));
                let mut hi = vld1q_s64(qt.add(i + 2));
                lo = vaddq_s64(lo, vmull_s32(vget_low_s32(ve), vget_low_s32(me)));
                hi = vaddq_s64(hi, vmull_high_s32(ve, me));
                lo = vsubq_s64(lo, vmull_s32(vget_low_s32(vl), vget_low_s32(ml)));
                hi = vsubq_s64(hi, vmull_high_s32(vl, ml));
                vst1q_s64(qt.add(i), lo);
                vst1q_s64(qt.add(i + 2), hi);
                let xlo = vsubq_s64(lo, vmull_s32(vget_low_s32(vs), vget_low_s32(si)));
                let xhi = vsubq_s64(hi, vmull_high_s32(vs, si));
                let x32 = vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_s64(xlo)), vcvtq_f64_s64(xhi));
                let r = vld1q_f32(cols.r32.as_ptr().add(at));
                let b = vld1q_f32(cols.bias32.as_ptr().add(at));
                let f = vaddq_f32(vmulq_f32(vmulq_f32(x32, ri), r), b);
                vst1q_f32(out.add(i), f);
                hit = vorrq_u32(hit, vcgeq_f32(f, vld1q_f32(cols.floor32.as_ptr().add(at))));
                acc = vmaxq_f32(acc, f);
            }
            j += 8;
        }
        maxima[group] = vmaxvq_f32(acc);
        acc = ninf;
        mask |= u64::from(vmaxvq_u32(hit) != 0) << group;
        hit = vdupq_n_u32(0);
        group += 1;
    }
    mask
}
