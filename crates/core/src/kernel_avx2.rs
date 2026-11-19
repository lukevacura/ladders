//! The narrow row kernel on AVX2 (`x86_64`): the same lanes, ranking value and group reports as
//! the scalar body of [`crate::kernel_neon::rank_row_groups`], four lags per vector, equal to it
//! on bits. `enabled` says whether the running CPU takes this kernel; without `std` or AVX2
//! the scalar body runs. All of the `unsafe` for the narrow kernel's AVX2 path is here.
//!
//! AVX2 has no `i64 → f64` conversion. A row whose
//! [`Conversion`](crate::kernel_neon::Conversion) is fast converts by `to_f64_fast`, one magic
//! constant, exact on `±2⁵¹`; every other row converts by `to_f64_full`, which builds exact
//! doubles from a lane's two 32-bit halves by magic constants and adds them with one rounding,
//! the same correctly rounded conversion `x as f64` makes, on every `i64`. Every `f64` step is a
//! separate multiply or add, never a fused one.

use crate::full_range::{HIGH_MAGIC, HIGH_MAGIC_BITS, LOW_MAGIC_BITS};
use crate::kernel_neon::{GROUP_LANES, Lanes, RowScalars, convert_body, scalar_range};
use core::arch::x86_64::*;

/// The kernels that have an AVX2 body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum Kernel {
    /// The z-normalized narrow row kernel.
    Narrow,
    /// The non-normalized lane kernel.
    NonNormalized,
}

/// Whether `kernel`'s AVX2 body runs: the CPU has AVX2 (known at compile time, or detected under
/// `std`) and no test has forced the scalar body.
pub(crate) fn enabled(kernel: Kernel) -> bool {
    #[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
    {
        if forced::scalar() {
            return false;
        }
        let taken = detected();
        if taken {
            forced::count(kernel);
        }
        taken
    }
    #[cfg(not(all(any(test, feature = "oracle"), not(any(loom, shuttle)))))]
    {
        let _ = kernel;
        detected()
    }
}

/// Whether the CPU has AVX2, ignoring any forcing.
#[doc(hidden)]
pub fn detected() -> bool {
    if cfg!(target_feature = "avx2") {
        return true;
    }
    #[cfg(any(feature = "std", test))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(feature = "std", test)))]
    {
        false
    }
}

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
pub use forced::{Guard, fast_rows_taken, force_scalar, full_range_rows_taken, rows_taken};

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
mod forced {
    use super::Kernel;
    use crate::sync::{AtomicBool, AtomicU64, Ordering};

    static SCALAR: AtomicBool = AtomicBool::new(false);
    static ROWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    static FAST_ROWS: AtomicU64 = AtomicU64::new(0);
    static FULL_RANGE_ROWS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn scalar() -> bool {
        SCALAR.load(Ordering::Relaxed)
    }

    pub(super) fn count(kernel: Kernel) {
        ROWS[kernel as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// How many rows `kernel` has handed to its AVX2 body so far in this process.
    #[doc(hidden)]
    pub fn rows_taken(kernel: Kernel) -> u64 {
        ROWS[kernel as usize].load(Ordering::Relaxed)
    }

    pub(super) fn count_fast() {
        FAST_ROWS.fetch_add(1, Ordering::Relaxed);
    }

    /// How many narrow rows the AVX2 body has run with the fast conversion so far in this
    /// process.
    #[doc(hidden)]
    pub fn fast_rows_taken() -> u64 {
        FAST_ROWS.load(Ordering::Relaxed)
    }

    pub(super) fn count_full_range() {
        FULL_RANGE_ROWS.fetch_add(1, Ordering::Relaxed);
    }

    /// How many narrow rows the AVX2 body has run with the full-range conversion so far in this
    /// process.
    #[doc(hidden)]
    pub fn full_range_rows_taken() -> u64 {
        FULL_RANGE_ROWS.load(Ordering::Relaxed)
    }

    /// Restores the previous setting when dropped.
    #[must_use]
    pub struct Guard(bool);

    impl Drop for Guard {
        fn drop(&mut self) {
            SCALAR.store(self.0, Ordering::Relaxed);
        }
    }

    /// Makes every narrow and non-normalized row take the scalar body while the guard lives.
    /// Process-wide: callers hold the route guard of `route::forced_route` around it, as every
    /// route-forcing test does.
    #[doc(hidden)]
    pub fn force_scalar() -> Guard {
        Guard(SCALAR.swap(true, Ordering::Relaxed))
    }
}

/// The bits of `1.5 · 2⁵²`: sign 0, biased exponent `0x433` (`2⁵²`), fraction field `2⁵¹`.
const MAGIC_BITS: i64 = 0x4338_0000_0000_0000;

/// `1.5 · 2⁵²`.
const MAGIC: f64 = 6_755_399_441_055_744.0;

/// `x as f64` for four lanes with `−2⁵¹ ≤ x < 2⁵¹`: one integer add and one `f64` subtract.
///
/// Proof that it equals `x as f64` bit for bit on that range. Adding `x` to [`MAGIC_BITS`] as
/// integers adds it to the 52-bit fraction field, which holds `2⁵¹`: `2⁵¹ + x` lies in
/// `[0, 2⁵²)`, so nothing carries into or borrows from the exponent, and the sum is the bit
/// pattern of the double `2⁵² + (2⁵¹ + x) = 1.5 · 2⁵² + x`, exact because the spacing of doubles
/// in `[2⁵², 2⁵³)` is 1. Subtracting [`MAGIC`] has the exact result `x`, an integer of magnitude
/// at most `2⁵¹`, which is a double; an IEEE subtraction returns a representable exact result
/// unrounded, and for `x = 0` it returns `+0.0` (round to nearest). `x as f64` is exact for
/// `|x| ≤ 2⁵³` and gives `+0.0` for 0, so the two agree on every bit. Outside the range the
/// result is another number (near `2⁶¹·⁸` not even finite), so a row takes this conversion only
/// when the column builder's cross-term bound proves every lane in range.
#[target_feature(enable = "avx2")]
#[inline]
fn to_f64_fast(x: __m256i) -> __m256d {
    let biased = _mm256_add_epi64(x, _mm256_set1_epi64x(MAGIC_BITS));
    _mm256_sub_pd(_mm256_castsi256_pd(biased), _mm256_set1_pd(MAGIC))
}

/// `x as f64` for four lanes, every `i64`, by two magic-constant halves: a shift, an xor and a
/// blend build the exact doubles `H = 2⁸⁴ + 2⁶³ + hi · 2³²` and `L = 2⁵² + lo`, an exact
/// subtract removes `H`'s bias and one add rounds. The proof, and the scalar model these lines
/// mirror one for one, are in `crate::full_range`.
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

/// The conversion body `CONVERT` names (a [`convert_body`] value): [`to_f64_full`] or
/// [`to_f64_fast`], whose range a debug build checks lane by lane. `CONVERT` is a constant,
/// so each instance holds one body and no branch.
#[target_feature(enable = "avx2")]
#[inline]
fn convert<const CONVERT: u8>(x: __m256i) -> __m256d {
    match CONVERT {
        convert_body::FULL_RANGE => to_f64_full(x),
        _ => {
            #[cfg(debug_assertions)]
            {
                let mut lanes = [0i64; 4];
                // SAFETY: the array holds four `i64`s.
                unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), x) };
                debug_assert!(
                    lanes
                        .iter()
                        .all(|v| (-(1i64 << 51)..1i64 << 51).contains(v)),
                    "a fast-conversion lane is outside the proved range: {lanes:?}"
                );
            }
            to_f64_fast(x)
        }
    }
}

/// [`convert`] on an array, for the tests that pin every conversion against `as f64`.
#[cfg(all(test, not(any(loom, shuttle))))]
pub(crate) fn to_f64_four<const CONVERT: u8>(x: [i64; 4]) -> [f64; 4] {
    assert!(detected(), "AVX2 is required");
    let mut out = [0.0; 4];
    // SAFETY: AVX2 was just checked; the array holds four `i64`s and four `f64`s.
    unsafe { store_converted::<CONVERT>(x.as_ptr(), out.as_mut_ptr()) };
    out
}

/// # Safety
/// AVX2 is available; `from` is readable and `to` writable for four elements.
#[cfg(all(test, not(any(loom, shuttle))))]
#[target_feature(enable = "avx2")]
unsafe fn store_converted<const CONVERT: u8>(from: *const i64, to: *mut f64) {
    // SAFETY: as the caller guarantees.
    unsafe {
        _mm256_storeu_pd(
            to,
            convert::<CONVERT>(_mm256_loadu_si256(from.cast::<__m256i>())),
        )
    }
}

/// Where the kernel reads and writes, and the row's scalars as vectors.
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
    /// Lanes `c..c + 4`: `qt` stepped with three widening `i32 × i32` products, then the scalar
    /// body's left-to-right `f64` chain, folded into `acc` (and, with `FLOORS`, whether a lane
    /// is at or above its floor into `hit`). The cross term converts by the body `CONVERT`
    /// names ([`convert`]).
    ///
    /// # Safety
    /// AVX2 is available and every pointer is valid for lanes `c..c + 4` (`floor` only read with
    /// `FLOORS`).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rank<const FLOORS: bool, const CONVERT: u8>(
        &self,
        c: usize,
        acc: &mut __m256d,
        hit: &mut __m256d,
    ) {
        // SAFETY: as the caller guarantees, each access below lies within lanes `c..c + 4`.
        unsafe {
            let widen = |at: *const i32| {
                _mm256_cvtepi32_epi64(_mm_loadu_si128(at.add(c).cast::<__m128i>()))
            };
            let qt = self.qt.add(c).cast::<__m256i>();
            let mut q = _mm256_loadu_si256(qt);
            q = _mm256_add_epi64(q, _mm256_mul_epi32(self.me, widen(self.enter)));
            q = _mm256_sub_epi64(q, _mm256_mul_epi32(self.ml, widen(self.leave)));
            _mm256_storeu_si256(qt, q);
            let x = _mm256_sub_epi64(q, _mm256_mul_epi32(self.si, widen(self.sum)));
            let scaled = _mm256_mul_pd(
                _mm256_mul_pd(convert::<CONVERT>(x), self.ri),
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
}

/// The largest of four lanes (none is NaN).
#[target_feature(enable = "avx2")]
#[inline]
fn horizontal_max(v: __m256d) -> f64 {
    let mut lanes = [0.0f64; 4];
    // SAFETY: the array holds four `f64`s.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), v) };
    lanes.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// [`crate::kernel_neon::rank_row`] (with `GROUPS` off) or [`crate::kernel_neon::rank_row_groups`]
/// (on; with `FLOORS`, the hit mask too) on AVX2: four lags per step, lags past the last full
/// four through [`scalar_range`]. Returns the row's maximum and the hit mask. The vector lanes
/// convert by the body `CONVERT` names; with [`convert_body::FAST`] the caller holds the fast
/// [`Conversion`](crate::kernel_neon::Conversion) for this row (every cross term it meets within `±2⁵¹`).
///
/// # Safety
/// The CPU has AVX2 ([`enabled`] said so). Every lane slice and (when present) `scratch` hold
/// `qt_m.len()` elements, `floors` too when `FLOORS` (which needs `GROUPS` and `scratch`), and
/// with `GROUPS` `maxima` holds one slot per group and there are at most
/// [`crate::kernel_neon::SPAN_GROUPS`] of them.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rank<const GROUPS: bool, const FLOORS: bool, const CONVERT: u8>(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    mut scratch: Option<&mut [f64]>,
    floors: &[f64],
    maxima: &mut [f64],
) -> (f64, u64) {
    debug_assert!(
        CONVERT != convert_body::FAST || row.conversion.is_fast(),
        "the fast body needs the fast conversion"
    );
    #[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
    match CONVERT {
        convert_body::FAST => forced::count_fast(),
        _ => forced::count_full_range(),
    }
    let len = qt_m.len();
    let vectored = len / 4 * 4;
    let scratch_ptr: Option<*mut f64> = scratch.as_mut().map(|s| s.as_mut_ptr());
    let mut hits = 0u64;

    // SAFETY: `qt_m`, `lanes.{enter,leave,sum,r,bias}`, (when present) the slice behind
    // `scratch_ptr` and (with `FLOORS`) `floors` all hold exactly `len` elements, as the caller
    // guarantees; every access is four contiguous elements at an offset below `vectored`, a
    // multiple of 4 no greater than `len`. A group index stays below `maxima.len()`: groups
    // split `0..vectored` into `GROUP_LANES`-lane runs, one slot each.
    let max_v = unsafe {
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
                    at.rank::<FLOORS, CONVERT>(c + 4 * step, &mut acc[step % 2], &mut hit);
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
                at.rank::<FLOORS, CONVERT>(c, &mut acc[step % 2], &mut hit);
                c += 4;
                step += 1;
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

/// [`rank`] with the conversion body `row` holds, as the dispatcher picks it once per row.
///
/// # Safety
/// As [`rank`]'s.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rank_converting<const GROUPS: bool, const FLOORS: bool>(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    scratch: Option<&mut [f64]>,
    floors: &[f64],
    maxima: &mut [f64],
) -> (f64, u64) {
    use convert_body::{FAST, FULL_RANGE};
    // SAFETY: as the caller guarantees.
    unsafe {
        match row.conversion.body() {
            FAST => rank::<GROUPS, FLOORS, FAST>(qt_m, lanes, row, scratch, floors, maxima),
            _ => rank::<GROUPS, FLOORS, FULL_RANGE>(qt_m, lanes, row, scratch, floors, maxima),
        }
    }
}

/// `i32` products a step of [`narrow_dot`] takes.
const DOT_STEP: usize = 16;

/// Lanes a block of [`narrow_lanes`] takes together.
const LANE_STEP: usize = 8;

/// `Σ aₖ·bₖ` over the shorter of `a` and `b`, as `i64` products of `i32`s, wrapping: the narrow
/// start sum. `vpmuldq` multiplies the even 32-bit lanes of two vectors as signed 32 x 32 -> 64,
/// and a 32-bit right shift of each 64-bit lane brings the odd lanes down for a second multiply:
/// 16 products a step in four accumulators. The `i64` sums wrap, so any grouping is the same
/// bits as the scalar fold.
///
/// The wide (`i64 x i64`) sums have no AVX2 route: AVX2 has no 64-bit multiply, and they stay
/// scalar.
///
/// # Safety
/// The CPU has AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn narrow_dot(a: &[i32], b: &[i32]) -> i64 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    let mut a_steps = a.chunks_exact(DOT_STEP);
    let mut b_steps = b.chunks_exact(DOT_STEP);
    let (mut s0, mut s1, mut s2, mut s3) = (
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    );
    for (x, y) in (&mut a_steps).zip(&mut b_steps) {
        // SAFETY: `x` and `y` hold `DOT_STEP` = 16 `i32`s, so two unaligned 256-bit loads from
        // each (at 0 and 8 `i32`s) are in range.
        unsafe {
            let (x, y) = (x.as_ptr().cast::<__m256i>(), y.as_ptr().cast::<__m256i>());
            let (x0, y0) = (_mm256_loadu_si256(x), _mm256_loadu_si256(y));
            let (x1, y1) = (_mm256_loadu_si256(x.add(1)), _mm256_loadu_si256(y.add(1)));
            s0 = _mm256_add_epi64(s0, _mm256_mul_epi32(x0, y0));
            s1 = _mm256_add_epi64(s1, _mm256_mul_epi32(x1, y1));
            let (xo0, yo0) = (_mm256_srli_epi64::<32>(x0), _mm256_srli_epi64::<32>(y0));
            let (xo1, yo1) = (_mm256_srli_epi64::<32>(x1), _mm256_srli_epi64::<32>(y1));
            s2 = _mm256_add_epi64(s2, _mm256_mul_epi32(xo0, yo0));
            s3 = _mm256_add_epi64(s3, _mm256_mul_epi32(xo1, yo1));
        }
    }
    let lanes = four_i64(_mm256_add_epi64(
        _mm256_add_epi64(s0, s1),
        _mm256_add_epi64(s2, s3),
    ));
    let tail = a_steps
        .remainder()
        .iter()
        .zip(b_steps.remainder())
        .fold(0i64, |s, (&x, &y)| {
            s.wrapping_add(i64::from(x).wrapping_mul(i64::from(y)))
        });
    lanes.iter().fold(tail, |s, &l| s.wrapping_add(l))
}

/// The four `i64` lanes of `x`.
#[target_feature(enable = "avx2")]
#[inline]
fn four_i64(x: __m256i) -> [i64; 4] {
    let mut lanes = [0i64; 4];
    // SAFETY: the array holds 32 bytes.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), x) };
    lanes
}

/// `states[l] = Σₖ a[k]·b[l + k]` for every lane `l` of `states`, as `i64` products of `i32`s,
/// wrapping: the cross terms of one window with consecutive windows (`b` holds
/// `states.len() + a.len() - 1` values). Eight lanes go together, the even and the odd ones in
/// two accumulators, each value of `a` broadcast once per block; the lanes past the last block
/// take [`narrow_dot`]. Every lane is the same wrapping sum as the scalar fold, so the same bits.
///
/// # Safety
/// The CPU has AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn narrow_lanes(a: &[i32], b: &[i32], states: &mut [i64]) {
    let lanes = states.len();
    assert!(lanes == 0 || b.len() + 1 >= lanes + a.len());
    let mut first = 0;
    while first + LANE_STEP <= lanes {
        let (mut even, mut odd) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for (&x, ys) in a
            .iter()
            .zip(b[first..first + a.len() + LANE_STEP - 1].windows(LANE_STEP))
        {
            // SAFETY: `ys` holds `LANE_STEP` = 8 `i32`s, one unaligned 256-bit load.
            let y = unsafe { _mm256_loadu_si256(ys.as_ptr().cast::<__m256i>()) };
            let x = _mm256_set1_epi32(x);
            even = _mm256_add_epi64(even, _mm256_mul_epi32(x, y));
            odd = _mm256_add_epi64(odd, _mm256_mul_epi32(x, _mm256_srli_epi64::<32>(y)));
        }
        let (even, odd) = (four_i64(even), four_i64(odd));
        for (pair, (&e, &o)) in states[first..first + LANE_STEP]
            .chunks_exact_mut(2)
            .zip(even.iter().zip(&odd))
        {
            pair[0] = e;
            pair[1] = o;
        }
        first += LANE_STEP;
    }
    for (l, state) in states.iter_mut().enumerate().skip(first) {
        // SAFETY: AVX2, as the caller guarantees.
        *state = unsafe { narrow_dot(a, &b[l..]) };
    }
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod tests;
