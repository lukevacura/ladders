//! The fast row kernel: for one row, advances every lag's `QT` and ranks it against the row's
//! own scalars, in a tight loop meant to run once per neighbor on every row of a stream.
//! [`rank_row`] is the one entry point (NEON on `aarch64`, AVX2 on `x86_64` where the CPU has it
//! (`kernel_avx2`), the scalar body everywhere else); a
//! test-only scalar oracle takes the same slices and scalars and always agrees with it, bit
//! for bit, on the resulting maximum, the updated `qt_m`, and `scratch` when given one. The scalar lane ranks through
//! [`crate::estimate::rank_estimate`], the one place the ranking expression is written; the
//! NEON body restates that same product in vector form because a vector lane can't call a
//! scalar function, with its own written justification beside it.

use crate::estimate::rank_estimate;

/// This row's own scalars, needed by every lag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowScalars {
    /// `m · eᵢ`: this row's entering tick, scaled by the window length.
    pub me: i32,
    /// `m · lᵢ`: this row's leaving tick, scaled by the window length.
    pub ml: i32,
    /// `Sᵢ`: this row's own window sum.
    pub si: i32,
    /// `1/√Vᵢ`: this row's own inverse square root.
    pub ri: f64,
    /// Which conversion of this row's cross terms to `f64` a vector body may take.
    pub conversion: Conversion,
}

/// How a vector body may turn a lane's `i64` cross term into `f64`: every way gives `x as f64`
/// bit for bit, the fast one only on terms within `±2⁵¹`, so it is held only where every term a
/// row meets is proved in that range (the column builder's cross-term bound); the full-range one
/// is exact on every `i64` and runs everywhere else. Only the AVX2 body takes the choice, once
/// per row; NEON and the scalar body convert natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conversion {
    body: u8,
}

/// The conversion bodies a [`Conversion`] names, as the AVX2 kernels' constant parameter.
pub(crate) mod convert_body {
    /// Both 32-bit halves by magic constants, exact on every `i64` (`crate::full_range`).
    pub(crate) const FULL_RANGE: u8 = 0;
    /// One magic constant, exact on `−2⁵¹ ≤ x < 2⁵¹` only.
    pub(crate) const FAST: u8 = 1;
}

impl Conversion {
    /// The full-range conversion by two magic-constant halves, exact on every `i64`.
    pub const FULL_RANGE: Self = Self {
        body: convert_body::FULL_RANGE,
    };

    /// The conversion that is exact on `−2⁵¹ ≤ x < 2⁵¹` only.
    pub(crate) const FAST: Self = Self {
        body: convert_body::FAST,
    };

    /// Whether this is the fast conversion.
    #[cfg(target_arch = "x86_64")]
    pub(crate) const fn is_fast(self) -> bool {
        self.body == convert_body::FAST
    }

    /// Which body converts: a [`convert_body`] value.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) const fn body(self) -> u8 {
        self.body
    }
}

/// The per-lag columns, read at the same ascending index as `qt_m`: `enter[j]` and `leave[j]`
/// are the neighbor's entering and leaving ticks for lag `j`, `sum[j]` its window sum, `r[j]`
/// its `1/√V` (zero unless it is Valid), and `bias[j]` the additive term that stands in for a
/// branch on its kind (zero for Valid, `0.5` for Constant, `−∞` for non-finite).
pub(crate) struct Lanes<'a> {
    pub(crate) enter: &'a [i32],
    pub(crate) leave: &'a [i32],
    pub(crate) sum: &'a [i32],
    pub(crate) r: &'a [f64],
    pub(crate) bias: &'a [f64],
}

/// Checked before any lag runs: every slice holds one element per lag, and `row.ri` —
/// `1/√V`, always positive by construction — is positive here too.
fn assert_preconditions(len: usize, lanes: &Lanes, row: RowScalars, scratch: Option<&[f64]>) {
    debug_assert!(row.ri > 0.0, "row.ri (1/sqrt(V)) must be positive");
    // `vmaxq_f64` propagates NaN where `f64::max` would not, so no ranking lane may be NaN:
    // every lane's `r` and `bias` are finite, or `bias` is −∞ with `r` exactly 0.
    debug_assert!(
        row.ri.is_finite()
            && lanes
                .r
                .iter()
                .zip(lanes.bias)
                .all(|(&r, &b)| (r.is_finite() && b.is_finite())
                    || (r == 0.0 && b == f64::NEG_INFINITY)),
        "no ranking lane may be NaN: r and bias must be finite, or bias -inf with r = 0"
    );
    assert_eq!(lanes.enter.len(), len, "enter must have one tick per lag");
    assert_eq!(lanes.leave.len(), len, "leave must have one tick per lag");
    assert_eq!(lanes.sum.len(), len, "sum must have one window sum per lag");
    assert_eq!(lanes.r.len(), len, "r must have one scale per lag");
    assert_eq!(lanes.bias.len(), len, "bias must have one bias per lag");
    if let Some(scratch) = scratch {
        assert_eq!(scratch.len(), len, "scratch must have one slot per lag");
    }
}

/// One lag's new `qt_m[j]` and ranking value: widening products and a wrapping `i64`
/// accumulation reach `x`, then [`rank_estimate`] ranks it and `bias` is added last — no FMA.
#[inline]
fn lane_value(
    qt: i64,
    row: RowScalars,
    enter: i32,
    leave: i32,
    sum: i32,
    r: f64,
    bias: f64,
) -> (i64, f64) {
    let q = qt
        .wrapping_add((row.me as i64).wrapping_mul(enter as i64))
        .wrapping_sub((row.ml as i64).wrapping_mul(leave as i64));
    let x = q.wrapping_sub((row.si as i64).wrapping_mul(sum as i64));
    let v = rank_estimate(x, row.ri, r) + bias;
    (q, v)
}

/// Runs the scalar lane body over `start..end`, updating `qt_m` and `scratch` (when given) in
/// place, and returning the maximum ranking value over that range (`−∞` if it is empty). Used
/// both as the whole-row scalar path and as the vector kernel's own remainder.
pub(crate) fn scalar_range(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    mut scratch: Option<&mut [f64]>,
    start: usize,
    end: usize,
) -> f64 {
    let mut max_v = f64::NEG_INFINITY;
    for j in start..end {
        let (q, v) = lane_value(
            qt_m[j],
            row,
            lanes.enter[j],
            lanes.leave[j],
            lanes.sum[j],
            lanes.r[j],
            lanes.bias[j],
        );
        qt_m[j] = q;
        if let Some(s) = scratch.as_deref_mut() {
            s[j] = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    max_v
}

/// Lanes per group in [`rank_row_groups`]: each group reports its maximum and whether any lane
/// clears its floor, so a reader rescans only the groups that can hold a candidate: the keeper's
/// block.
pub const GROUP_LANES: usize = crate::kbest::BLOCK_LANES;

/// The most groups one [`rank_row_groups`] call reports: its hit mask has one bit per group.
pub(crate) const SPAN_GROUPS: usize = 16;

/// The most lanes one [`rank_row_groups`] call takes.
pub const SPAN_LANES: usize = GROUP_LANES * SPAN_GROUPS;

/// The NEON kernel: eight lags per iteration in two halves of four, each half widening its
/// three `i32 × i32` products to `i64` with `vmull_s32`/`vmull_high_s32` (sign-extending, the
/// same widening [`lane_value`] gets from `as i64`), then the same left-to-right `f64` chain
/// as the scalar body with `vmulq_f64`/`vaddq_f64` — never a fused multiply-add. Four max
/// accumulators carry the running maximum, merged into one only at the end of the row (or, with
/// `GROUPS`, of each [`GROUP_LANES`]-lane group, into `maxima`); any lags past the last full
/// group of eight fall through to [`scalar_range`], which is exactly the scalar body applied to
/// those lags. With `FLOORS`, bit `g` of the returned mask records whether a lane of group `g` is
/// at or above `floors` at the same index. Returns the row's maximum and that mask.
///
/// Written exception to routing every lane through one scalar function: a vector lane can't
/// call [`rank_estimate`] per element, so `vmulq_f64`/`vmulq_f64`/`vaddq_f64` below restates
/// its `((x as f64) * ri) * r + bias` product order, one vector lane at a time; the oracle
/// test pins both against each other on bits, so a reassociation in either copy is caught,
/// not just agreed on.
///
/// # Safety
/// Every lane slice and (when present) `scratch` hold `qt_m.len()` elements, `floors` too when
/// `FLOORS` (which needs `GROUPS` and `scratch`), and with `GROUPS` `maxima` holds one slot per
/// group and there are at most `SPAN_GROUPS` of them.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rank_neon<const GROUPS: bool, const FLOORS: bool>(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    mut scratch: Option<&mut [f64]>,
    floors: &[f64],
    maxima: &mut [f64],
) -> (f64, u64) {
    use core::arch::aarch64::*;

    let len = qt_m.len();
    let vectored = len / 8 * 8;
    let scratch_ptr: Option<*mut f64> = scratch.as_mut().map(|s| s.as_mut_ptr());
    let floor_ptr = floors.as_ptr();
    let mut hits = 0u64;

    // Safety: `qt_m`, `lanes.{enter,leave,sum,r,bias}`, (when present) the slice behind
    // `scratch_ptr` and (with `FLOORS`) `floors` all hold exactly `len` elements, as the caller
    // guarantees; every pointer below is offset only within `0..vectored`, a multiple of 8 no
    // greater than `len`, and each load or store below touches exactly 4 contiguous elements
    // starting at such an offset — so every access stays in bounds. A group index stays below
    // `maxima.len()`: groups split `0..vectored` into `GROUP_LANES`-lane runs, one slot each.
    let max_v = unsafe {
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
            // Whole groups, their eight steps unrolled.
            while c + GROUP_LANES <= vectored {
                let mut acc = [vdupq_n_f64(f64::NEG_INFINITY); 4];
                let mut hit = vdupq_n_u64(0);
                for step in 0..GROUP_LANES / 8 {
                    at.rank::<FLOORS>(c + 8 * step, &mut acc, &mut hit);
                }
                c += GROUP_LANES;
                close(acc, hit);
            }
        }
        if c < vectored {
            let mut acc = [vdupq_n_f64(f64::NEG_INFINITY); 4];
            let mut hit = vdupq_n_u64(0);
            while c < vectored {
                at.rank::<FLOORS>(c, &mut acc, &mut hit);
                c += 8;
            }
            close(acc, hit);
        }
        row_max
    };

    if vectored == len {
        return (max_v, hits);
    }
    let tail_max = scalar_range(qt_m, lanes, row, scratch.as_deref_mut(), vectored, len);
    if GROUPS {
        // The tail joins the last group, which the vector loop opened unless the tail starts it.
        let last = maxima.len() - 1;
        let opened = vectored > last * GROUP_LANES;
        maxima[last] = if opened {
            maxima[last].max(tail_max)
        } else {
            tail_max
        };
        let tail_hit = FLOORS
            && scratch.is_some_and(|s| {
                s[vectored..len]
                    .iter()
                    .zip(&floors[vectored..len])
                    .any(|(v, f)| v >= f)
            });
        hits |= u64::from(tail_hit) << last;
    }
    (max_v.max(tail_max), hits)
}

/// Where [`rank_neon`] reads and writes, and the row's scalars as vectors.
#[cfg(target_arch = "aarch64")]
struct Lanes8 {
    qt: *mut i64,
    enter: *const i32,
    leave: *const i32,
    sum: *const i32,
    r: *const f64,
    bias: *const f64,
    out: Option<*mut f64>,
    floor: *const f64,
    me: core::arch::aarch64::int32x4_t,
    ml: core::arch::aarch64::int32x4_t,
    si: core::arch::aarch64::int32x4_t,
    ri: core::arch::aarch64::float64x2_t,
}

#[cfg(target_arch = "aarch64")]
impl Lanes8 {
    /// Lanes `c..c + 8`, in two halves of four: each half widens its three `i32 × i32` products
    /// to `i64` with `vmull_s32`/`vmull_high_s32`, then runs the scalar body's left-to-right `f64`
    /// chain with `vmulq_f64`/`vaddq_f64`, folding into `acc` (and, with `FLOORS`, whether a lane
    /// is at or above its floor into `hit`).
    ///
    /// # Safety
    /// Every pointer is valid for lanes `c..c + 8` (`floor` only read with `FLOORS`).
    #[inline(always)]
    unsafe fn rank<const FLOORS: bool>(
        &self,
        c: usize,
        acc: &mut [core::arch::aarch64::float64x2_t; 4],
        hit: &mut core::arch::aarch64::uint64x2_t,
    ) {
        use core::arch::aarch64::*;
        // Safety: as the caller guarantees, each access below lies within lanes `c..c + 8`.
        unsafe {
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

                let flo = vaddq_f64(vmulq_f64(vmulq_f64(vcvtq_f64_s64(xlo), self.ri), rlo), blo);
                let fhi = vaddq_f64(vmulq_f64(vmulq_f64(vcvtq_f64_s64(xhi), self.ri), rhi), bhi);

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
}

/// Advances every lag's `qt_m` and returns this row's maximum ranking value, NEON on
/// `aarch64`, AVX2 on `x86_64` where the CPU has it, and the scalar body everywhere else — always bit for bit the same as
/// the scalar oracle on the same inputs. `qt_m`, `enter`, `leave`, `sum`, `r`, `bias` and
/// `scratch` (when given) must all hold exactly one element per lag; asserted before any
/// lag runs.
///
/// Precondition: no ranking lane may be NaN (every `r` and `bias` finite, or `bias = −∞` with
/// `r = 0`), because the vector maxima propagate NaN where `f64::max` does not; debug-asserted.
#[allow(clippy::too_many_arguments)] // the contract's own signature, not a free choice here
pub fn rank_row(
    qt_m: &mut [i64],
    enter: &[i32],
    leave: &[i32],
    sum: &[i32],
    r: &[f64],
    bias: &[f64],
    row: RowScalars,
    scratch: Option<&mut [f64]>,
) -> f64 {
    let lanes = Lanes {
        enter,
        leave,
        sum,
        r,
        bias,
    };
    assert_preconditions(qt_m.len(), &lanes, row, scratch.as_deref());
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: lengths asserted above; see `rank_neon`'s own safety comment for the bound it
        // relies on.
        unsafe { rank_neon::<false, false>(qt_m, &lanes, row, scratch, &[], &mut []).0 }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if crate::kernel_avx2::enabled(crate::kernel_avx2::Kernel::Narrow) {
            use crate::kernel_avx2::rank_converting;
            // Safety: AVX2 detected; lengths asserted above.
            return unsafe {
                rank_converting::<false, false>(qt_m, &lanes, row, scratch, &[], &mut []).0
            };
        }
        let len = qt_m.len();
        scalar_range(qt_m, &lanes, row, scratch, 0, len)
    }
}

/// [`rank_row`] that reports per group: every lane's ranking value goes to `scratch`, and the
/// maximum of each [`GROUP_LANES`]-lane group (the last may be shorter) to `maxima`. With
/// `floors` (one per lane), bit `g` of the returned mask is set when some lane of group `g` is at
/// or above its floor; without, the mask is zero. Takes at most [`SPAN_LANES`] lanes; `maxima`
/// holds one slot per group. Bit for bit the scalar oracle on the same inputs, as [`rank_row`].
#[allow(clippy::too_many_arguments)] // the contract's own signature, not a free choice here
pub fn rank_row_groups(
    qt_m: &mut [i64],
    enter: &[i32],
    leave: &[i32],
    sum: &[i32],
    r: &[f64],
    bias: &[f64],
    row: RowScalars,
    scratch: &mut [f64],
    floors: Option<&[f64]>,
    maxima: &mut [f64],
) -> u64 {
    let lanes = Lanes {
        enter,
        leave,
        sum,
        r,
        bias,
    };
    let len = qt_m.len();
    assert_preconditions(len, &lanes, row, Some(scratch));
    assert!(len <= SPAN_LANES, "a span holds at most SPAN_LANES lanes");
    assert_eq!(
        maxima.len(),
        len.div_ceil(GROUP_LANES),
        "maxima must have one slot per group"
    );
    if let Some(floors) = floors {
        assert_eq!(floors.len(), len, "floors must have one floor per lag");
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: lengths asserted above, at most `SPAN_GROUPS` groups.
        unsafe {
            match floors {
                Some(floors) => {
                    rank_neon::<true, true>(qt_m, &lanes, row, Some(scratch), floors, maxima).1
                }
                None => rank_neon::<true, false>(qt_m, &lanes, row, Some(scratch), &[], maxima).1,
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if crate::kernel_avx2::enabled(crate::kernel_avx2::Kernel::Narrow) {
            use crate::kernel_avx2::rank_converting;
            let scratch = Some(scratch);
            // Safety: AVX2 detected; lengths asserted above, at most `SPAN_GROUPS` groups.
            return unsafe {
                match floors {
                    Some(f) => rank_converting::<true, true>(qt_m, &lanes, row, scratch, f, maxima),
                    None => rank_converting::<true, false>(qt_m, &lanes, row, scratch, &[], maxima),
                }
                .1
            };
        }
        scalar_groups(qt_m, &lanes, row, scratch, floors, maxima)
    }
}

/// [`rank_row_groups`]' scalar body: [`scalar_range`] once per group, then the floor check.
#[cfg(any(test, not(target_arch = "aarch64")))]
pub(crate) fn scalar_groups(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    scratch: &mut [f64],
    floors: Option<&[f64]>,
    maxima: &mut [f64],
) -> u64 {
    let len = qt_m.len();
    let mut hits = 0u64;
    for (g, max) in maxima.iter_mut().enumerate() {
        let lo = g * GROUP_LANES;
        let hi = (lo + GROUP_LANES).min(len);
        *max = scalar_range(qt_m, lanes, row, Some(&mut *scratch), lo, hi);
        let hit = floors.is_some_and(|floors| {
            scratch[lo..hi]
                .iter()
                .zip(&floors[lo..hi])
                .any(|(v, f)| v >= f)
        });
        hits |= u64::from(hit) << g;
    }
    hits
}

/// The oracle [`rank_row`] is checked against: the scalar body run over every lag, with the
/// same preconditions.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // the contract's own signature, not a free choice here
fn rank_row_scalar(
    qt_m: &mut [i64],
    enter: &[i32],
    leave: &[i32],
    sum: &[i32],
    r: &[f64],
    bias: &[f64],
    row: RowScalars,
    scratch: Option<&mut [f64]>,
) -> f64 {
    let lanes = Lanes {
        enter,
        leave,
        sum,
        r,
        bias,
    };
    assert_preconditions(qt_m.len(), &lanes, row, scratch.as_deref());
    let len = qt_m.len();
    scalar_range(qt_m, &lanes, row, scratch, 0, len)
}

/// The oracle [`rank_row_groups`] is checked against: the scalar body group by group.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // the contract's own signature, not a free choice here
fn rank_row_groups_scalar(
    qt_m: &mut [i64],
    enter: &[i32],
    leave: &[i32],
    sum: &[i32],
    r: &[f64],
    bias: &[f64],
    row: RowScalars,
    scratch: &mut [f64],
    floors: Option<&[f64]>,
    maxima: &mut [f64],
) -> u64 {
    let lanes = Lanes {
        enter,
        leave,
        sum,
        r,
        bias,
    };
    assert_preconditions(qt_m.len(), &lanes, row, Some(scratch));
    scalar_groups(qt_m, &lanes, row, scratch, floors, maxima)
}

#[cfg(test)]
pub(crate) mod tests;
