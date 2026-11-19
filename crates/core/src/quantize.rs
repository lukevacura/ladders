//! Converts a sample to the integer tick count the kernel runs on: one correctly rounded
//! division by the scale, ties to even, rejected if it wouldn't survive the round trip.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::sanitize::{ForwardFill, Sanitized};
use crate::stats::{MonotonicDeque, WindowKind, WindowStats};

/// A tick's size in the sample's own units: positive and finite by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale(f64);

impl Scale {
    pub fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

/// Every tick count at or below this bound is exactly representable as `f64`.
pub const MAX_TICKS: i64 = 1 << 53;

/// At or beyond this magnitude, every representable `f64` is already an integer.
const INTEGRAL_THRESHOLD: f64 = 4_503_599_627_370_496.0; // 2^52

/// `core` has no `f64::round_ties_even` (it needs `libm`, which only `std` links), so this
/// rounds by hand: adding then subtracting [`INTEGRAL_THRESHOLD`] forces the FPU's own
/// round-to-nearest-ties-to-even onto the sum, which lands on the nearest integer to `x`.
pub(crate) fn round_ties_even(x: f64) -> f64 {
    if x.abs() >= INTEGRAL_THRESHOLD {
        return x;
    }
    if x.is_sign_negative() {
        -((-x + INTEGRAL_THRESHOLD) - INTEGRAL_THRESHOLD)
    } else {
        (x + INTEGRAL_THRESHOLD) - INTEGRAL_THRESHOLD
    }
}

/// `None` for a non-finite `value`, or a tick count whose magnitude exceeds [`MAX_TICKS`].
pub fn to_ticks(value: f64, scale: Scale) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let ticks = round_ties_even(value / scale.get());
    if ticks.abs() > MAX_TICKS as f64 {
        return None;
    }
    Some(ticks as i64)
}

/// Whether the window ending at each row (`series[row..row + m]`, one `row` per element, `0`
/// first) has no non-finite sample: scale-independent (a sample's own finiteness never
/// changes), so computed once and reused across every candidate [`Scale::finest_fitting`]
/// tries.
fn clean_windows(series: &[f64], m: usize) -> Vec<bool> {
    let mut non_finite = series[..m].iter().filter(|x| !x.is_finite()).count();
    let mut clean = Vec::with_capacity(series.len() - m + 1);
    clean.push(non_finite == 0);
    for i in m..series.len() {
        if !series[i - m].is_finite() {
            non_finite -= 1;
        }
        if !series[i].is_finite() {
            non_finite += 1;
        }
        clean.push(non_finite == 0);
    }
    clean
}

/// `2^exponent`, by repeated doubling or halving from `1.0`: exact for any exponent this
/// module ever calls it with, and needs no transcendental function (`core` alone, no `libm`).
fn pow2(exponent: i32) -> f64 {
    let mut value = 1.0_f64;
    if exponent >= 0 {
        for _ in 0..exponent {
            value *= 2.0;
        }
    } else {
        for _ in 0..-exponent {
            value /= 2.0;
        }
    }
    value
}

/// The smallest power of two at or above `min_scale`: climbs one exponent at a time from a
/// value already below it, so it is exact, verified by the real floating-point comparison
/// rather than by any float estimate alone. `1.0` for `min_scale <= 0.0` (every finite sample
/// is itself `0.0`), an arbitrary but valid starting scale.
fn smallest_power_of_two_at_least(min_scale: f64) -> f64 {
    if min_scale <= 0.0 {
        return 1.0;
    }
    // `min_scale`'s own IEEE754 exponent field is `floor(log2(min_scale))` exactly for any
    // normal value (every realistic `min_scale` here); four exponents below it is a margin
    // against a subnormal one, not a precision the climb below relies on.
    let biased_exponent =
        i32::try_from((min_scale.to_bits() >> 52) & 0x7ff).expect("11 bits fit i32");
    let mut exponent = biased_exponent - 1023 - 4;
    let mut candidate = pow2(exponent);
    while candidate < min_scale {
        exponent += 1;
        candidate = pow2(exponent);
    }
    candidate
}

/// Whether every clean window (`clean[row]`) is, at `scale`, anything but
/// [`WindowKind::NonFinite`] — run through the kernel's own quantization ([`to_ticks`],
/// [`ForwardFill`]) and window statistics ([`WindowStats`]), never estimated. A dirty window
/// (`!clean[row]`) is `NonFinite` at every scale regardless of this check, so it is skipped
/// rather than blocking a candidate it could never pass anyway.
fn passes(series: &[f64], m: NonZeroUsize, scale: Scale, clean: &[bool]) -> bool {
    let mut fill = ForwardFill::<i64>::default();
    let mut stats = WindowStats::new(m);
    let mut row = 0usize;
    for &x in series {
        let (tick, had_tick) = match fill.push(to_ticks(x, scale)) {
            Sanitized::Present(t) => (t, true),
            Sanitized::Filled(t) => (t, false),
            Sanitized::Leading => (0, false),
        };
        if let Some(window) = stats.push(tick, had_tick) {
            if clean[row] && window.kind == WindowKind::NonFinite {
                return false;
            }
            row += 1;
        }
    }
    true
}

impl Scale {
    /// The finest power-of-two scale under which every clean window (no non-finite sample
    /// inside it) of `series`, `m` long, passes the kernel's own range check: every tick
    /// within [`MAX_TICKS`] in magnitude, and every such window's variance numerator below
    /// `2⁶³` ([`WindowKind::NonFinite`]'s own gate, checked for real by `passes`). A dirty
    /// window is `NonFinite` at every scale regardless, so it never blocks a candidate.
    ///
    /// Starts from a candidate `series`' own float statistics suggest (the smallest power of
    /// two keeping every finite sample's tick within [`MAX_TICKS`]), then doubles — each
    /// doubling roughly quarters a clean window's variance numerator — until `passes`
    /// confirms it on the library's own exact integer quantization and statistics, never on
    /// that float estimate alone; the first candidate that passes is returned, so it is the
    /// finest one that does.
    ///
    /// `None` for a `series` shorter than `m`, or with no finite sample at all: no window
    /// could ever be clean, so there is nothing to fit a scale to.
    pub fn finest_fitting(series: &[f64], m: NonZeroUsize) -> Option<Scale> {
        let m_usize = m.get();
        if series.len() < m_usize || !series.iter().any(|x| x.is_finite()) {
            return None;
        }

        let clean = clean_windows(series, m_usize);
        let max_abs = series
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let mut value = smallest_power_of_two_at_least(max_abs / MAX_TICKS as f64);

        // Bounded all the same (f64's own exponent range is under 2048 wide): a pathological
        // input (or a `passes` that could never be satisfied) fails closed, never loops.
        for _ in 0..2048 {
            let scale = Scale::new(value)?;
            if passes(series, m, scale, &clean) {
                return Some(scale);
            }
            value *= 2.0;
        }
        None
    }
}

/// How many ulps of `v / scale` a sample may sit from a whole tick and still count as on the
/// grid: `v = fl(k·scale)` and `scale` itself carry at most `2⁻⁵³` relative error each and the
/// division one more, so `|v/scale − k| ≤ 2.5·2⁻⁵³·|k|`, under one ulp (`2⁻⁵²·|x|`); four ulps
/// is that bound with margin, and grows with the level, unlike a fixed fraction of a tick.
/// Measured limit: off-grid half-ticks are only accepted at `|k| ≥ 2⁴⁹` (about 5.6e12 at
/// `scale = 0.01`), where one ulp of `v / scale` is comparable to a tick.
const GRID_ULPS: f64 = 4.0;

impl Scale {
    /// Whether `v` lies on this scale's grid: its residual `|x − round(x)|`, for `x = v / scale`,
    /// is within [`GRID_ULPS`] ulps of `x` (the rounding error of computing `x`). `false` for
    /// a non-finite `v`. The one grid test: [`Self::holds`] and [`Self::infer`] both use it.
    fn on_grid(self, v: f64) -> bool {
        let x = v / self.0;
        x.is_finite() && (x - round_ties_even(x)).abs() <= GRID_ULPS * f64::EPSILON * x.abs()
    }

    /// Whether every sample lies on this scale's grid (see `on_grid`). `false` for a
    /// non-finite sample.
    pub fn holds(self, samples: &[f64]) -> bool {
        samples.iter().all(|&v| self.on_grid(v))
    }

    /// The scale to quantize `query` at, once: `self`, the stream's scale, when the query
    /// lies on its grid and the kernel accepts it there (so an exact repeat of stream data is
    /// the same ticks), else the query's own [`Self::finest_fitting`]. `None` for an empty
    /// query or none that fits.
    pub fn for_query(self, query: &[f64]) -> Option<Scale> {
        let m = NonZeroUsize::new(query.len())?;
        if self.holds(query) && passes(query, m, self, &clean_windows(query, m.get())) {
            return Some(self);
        }
        Self::finest_fitting(query, m)
    }
}

/// `10^exponent` for `0 <= exponent <= 22`: every such power of ten is exactly an `f64`, so
/// repeated multiplication from `1.0` is exact.
fn pow10(exponent: i32) -> f64 {
    let mut value = 1.0_f64;
    for _ in 0..exponent {
        value *= 10.0;
    }
    value
}

/// The default scale family, coarse to fine: first `c·10ᵏ` for `c` in `{5, 2.5, 2, 1}` (in
/// that order within one `k`) and `k` from `+9` down to `-12`, then the powers of two `2⁹`
/// down to `2⁻³⁰`. A decimal candidate is one exact operation on exact operands (`c` times
/// or divided by an exact power of ten), so it is the correctly rounded `f64` of the decimal
/// scale, the same value a literal such as `0.01` gives.
fn default_candidates() -> impl Iterator<Item = f64> {
    const DECIMAL_FACTORS: [f64; 4] = [5.0, 2.5, 2.0, 1.0];
    let decimals = (-12..=9).rev().flat_map(|k: i32| {
        DECIMAL_FACTORS
            .into_iter()
            .map(move |c| if k >= 0 { c * pow10(k) } else { c / pow10(-k) })
    });
    let powers_of_two = (-30..=9).rev().map(pow2);
    decimals.chain(powers_of_two)
}

/// The largest tick count `infer` accepts for any sample (`2⁴⁰`): the grid test is only
/// meaningful while one ulp of `v / scale` stays far below a tick (its measured limit is about
/// `2⁴⁹`), and at a larger count every float looks like a whole number of ticks.
const INFER_MAX_TICKS: f64 = (1u64 << 40) as f64;

impl Scale {
    /// The data's own grid, when it lies on one: the first of `default_candidates` (coarse to
    /// fine) that passes three tests.
    ///
    /// A candidate is accepted only if
    /// 1. every finite sample lies on its grid ([`Self::holds`]'s test, non-finite samples
    ///    skipped);
    /// 2. no finite sample is more than `2⁴⁰` ticks from zero, so a fine grid cannot "hold"
    ///    merely because floats at a large level are coarser than the grid; and
    /// 3. the kernel accepts it (`passes`).
    ///
    /// One edge is accepted: a series that is constant to `f64` precision at a huge level
    /// (its whole range a few ulps of its level) lies on a coarse grid too, and may infer it
    /// and come out constant.
    ///
    /// Cost: a candidate stops at its first off-grid or oversized sample, so an off-grid
    /// series fails all 128 candidates early; a series on a grid scans each sample once per
    /// candidate down to that grid, at most 128 passes of `n` samples. The only allocation is
    /// the window-cleanliness table `passes` needs, built once, when a candidate first clears
    /// tests 1 and 2.
    fn grid(series: &[f64], m: NonZeroUsize) -> Option<Scale> {
        let mut clean: Option<Vec<bool>> = None;
        for value in default_candidates() {
            let Some(scale) = Scale::new(value) else {
                continue;
            };
            let on_small_grid = series.iter().all(|&v| {
                !v.is_finite() || (v.abs() / scale.0 <= INFER_MAX_TICKS && scale.on_grid(v))
            });
            if !on_small_grid {
                continue;
            }
            let clean = clean.get_or_insert_with(|| clean_windows(series, m.get()));
            if passes(series, m, scale, clean) {
                return Some(scale);
            }
        }
        None
    }

    /// The default scale for `series` at window length `m`: the data's own grid when it lies on
    /// one, else [`Self::fast_fitting`]. A pure function of `series` and `m`. `None` for a
    /// `series` shorter than `m`, or with no finite sample.
    pub fn infer(series: &[f64], m: NonZeroUsize) -> Option<Scale> {
        ScaleChoice::Fast.resolve(series, m)
    }

    /// The opt-in finer default: the data's own grid when it lies on one, else
    /// [`Self::finest_fitting`]. `None` as [`Self::infer`] is.
    pub fn infer_finest(series: &[f64], m: NonZeroUsize) -> Option<Scale> {
        ScaleChoice::Finest.resolve(series, m)
    }
}

/// The numerator of the fast-path tick bound `FAST_TICKS(m) = FAST_TICKS_NUMERATOR / m`: a window
/// fits when one anchor holds every tick of it within that many ticks. A constant of the default
/// scale's rule, never read from a route, so no route can change the default scale.
pub const FAST_TICKS_NUMERATOR: i64 = (1 << 31) - 1;

/// `FAST_TICKS(m)`, the most ticks a fitting window's ticks may lie from its anchor.
pub fn fast_ticks(m: NonZeroUsize) -> i64 {
    FAST_TICKS_NUMERATOR / i64::try_from(m.get()).unwrap_or(i64::MAX)
}

/// At most one clean window in this many may hold an outlier at the default scale: where more do,
/// the outlier rule is excusing the data's own movement rather than isolated bad samples (at a
/// fine enough scale every tick lies past its limit), so the scale is too fine.
pub const OUTLIER_WINDOW_SHARE: usize = 16;

/// The buffers one [`fits_fast`] check fills, kept across the candidates of one search.
#[derive(Default)]
struct FitScratch {
    ticks: Vec<i64>,
    had: Vec<bool>,
    outliers: Vec<bool>,
}

/// The samples a clean span holds beyond its window: the default scale fits every run of
/// `m + SPAN_SLACK` finite samples holding no outlier to one anchor, not each window alone, so the
/// smallest cell the re-anchoring route cuts (one row by a kernel group of 64 lanes, `m + 64`
/// samples) fits as well. A constant of the rule, never read from a route; changing it changes
/// results.
pub const SPAN_SLACK: usize = 128;

/// The one series a grid shared by `parts` is found on: the parts end to end, each pair apart by
/// a seam of `m + SPAN_SLACK` non-finite samples (no wider than the longest part), so no clean
/// window or span crosses from one part into the next.
fn joined(parts: &[&[f64]], m: NonZeroUsize) -> Vec<f64> {
    let longest = parts.iter().map(|part| part.len()).max().unwrap_or(0);
    let seam = m.get().saturating_add(SPAN_SLACK).min(longest);
    let mut joined = Vec::new();
    for (at, part) in parts.iter().enumerate() {
        if at > 0 {
            joined.resize(joined.len() + seam, f64::NAN);
        }
        joined.extend_from_slice(part);
    }
    joined
}

/// The clean spans [`fits_fast`] checks: every `span` consecutive finite samples, and each run of
/// finite samples shorter than `span` but at least `m` long as a whole.
struct Spans {
    m: usize,
    span: usize,
    reach: i64,
    high: MonotonicDeque<true>,
    low: MonotonicDeque<false>,
    /// Finite samples ending at the current one, and the outliers among the last
    /// `min(run, span)` of them.
    run: usize,
    held: usize,
}

impl Spans {
    fn new(m: NonZeroUsize, reach: i64) -> Self {
        let span = m.saturating_add(SPAN_SLACK);
        Self {
            m: m.get(),
            span: span.get(),
            reach,
            high: MonotonicDeque::new(span),
            low: MonotonicDeque::new(span),
            run: 0,
            held: 0,
        }
    }

    /// Folds in the sample at `p` (`had[p]` when finite); `true` when a span ends there that
    /// holds no outlier and whose ticks span more than `reach`. A span ends at its `span`-th
    /// finite sample, or at the last of a shorter run at least `m` long.
    fn breaks(&mut self, p: usize, ticks: &[i64], had: &[bool], outliers: &[bool]) -> bool {
        if !had[p] {
            (self.run, self.held) = (0, 0);
            return false;
        }
        self.run += 1;
        self.held += usize::from(outliers[p]);
        if self.run > self.span {
            self.held -= usize::from(outliers[p - self.span]);
        }
        let len = self.run.min(self.span);
        self.high.push(p as u64, ticks[p], len);
        self.low.push(p as u64, ticks[p], len);
        let ends_run = had.get(p + 1) != Some(&true);
        let complete = self.run >= self.span || (ends_run && self.run >= self.m);
        complete && self.held == 0 && self.high.extreme() - self.low.extreme() > self.reach
    }
}

/// Whether, at `scale`, every finite sample of `series` has a tick, every clean window
/// (`clean[row]`) and every clean span ([`Spans`]) not holding an outlier fits one anchor (its
/// ticks spanning no more than `2·FAST_TICKS(m)`), and no more than one clean window in
/// [`OUTLIER_WINDOW_SHARE`] holds one.
fn fits_fast(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    clean: &[bool],
    scratch: &mut FitScratch,
) -> bool {
    let FitScratch {
        ticks,
        had,
        outliers,
    } = scratch;
    ticks.clear();
    had.clear();
    for &x in series {
        match to_ticks(x, scale) {
            Some(tick) => {
                ticks.push(tick);
                had.push(true);
            }
            None if x.is_finite() => return false,
            None => {
                ticks.push(0);
                had.push(false);
            }
        }
    }
    outliers.clear();
    outliers.resize(series.len(), false);
    crate::spill::each_outlier(ticks, had, m.get(), |p| {
        outliers[p] = true;
        true
    });
    let excusable = clean.iter().filter(|&&c| c).count() / OUTLIER_WINDOW_SHARE;
    let reach = 2 * fast_ticks(m);
    let mut high = MonotonicDeque::<true>::new(m);
    let mut low = MonotonicDeque::<false>::new(m);
    let mut spans = Spans::new(m, reach);
    let (mut held, mut excused) = (0usize, 0usize);
    for (p, &tick) in ticks.iter().enumerate() {
        if spans.breaks(p, ticks, had, outliers) {
            return false;
        }
        high.push(p as u64, tick, m.get());
        low.push(p as u64, tick, m.get());
        held += usize::from(outliers[p]);
        let Some(row) = (p + 1).checked_sub(m.get()) else {
            continue;
        };
        if clean[row] {
            if held > 0 {
                excused += 1;
                if excused > excusable {
                    return false;
                }
            } else if high.extreme() - low.extreme() > reach {
                return false;
            }
        }
        held -= usize::from(outliers[row]);
    }
    true
}

impl Scale {
    /// The finest power-of-two scale at which every clean span not holding an outlier fits the
    /// fast path's tick bound: every run of `m + `[`SPAN_SLACK`] finite samples (a whole run of
    /// finite samples when it is shorter, but at least `m` long), and every clean window (no
    /// non-finite sample inside it), that holds no tick the shared outlier rule withholds has its
    /// ticks within [`fast_ticks`] of one anchor. Every finite sample has a tick there, and at
    /// most one clean window in [`OUTLIER_WINDOW_SHARE`] holds an outlier.
    ///
    /// Tries each power of two from the finest at which every finite sample has a tick, coarser
    /// one at a time, and returns the first that passes: one where every tick is `0` always does.
    /// The check runs on the library's own quantization and outlier rule, never on a float
    /// estimate.
    ///
    /// `None` as [`Self::finest_fitting`] is: a `series` shorter than `m`, or with no finite
    /// sample.
    pub fn fast_fitting(series: &[f64], m: NonZeroUsize) -> Option<Scale> {
        let finite = || series.iter().copied().filter(|x| x.is_finite());
        if series.len() < m.get() || finite().next().is_none() {
            return None;
        }
        fast_fitting_each(&[series], m)
    }
}

/// [`Scale::fast_fitting`] of several series sharing one scale: the finest power of two at which
/// every finite sample of each has a tick and each at least `m` long fits the rule on its own,
/// with its own outliers and its own share of them.
fn fast_fitting_each(parts: &[&[f64]], m: NonZeroUsize) -> Option<Scale> {
    let max_abs = parts
        .iter()
        .flat_map(|part| part.iter().copied())
        .filter(|x| x.is_finite())
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let mut value = smallest_power_of_two_at_least(max_abs / MAX_TICKS as f64);
    let fitted: Vec<(&[f64], Vec<bool>)> = parts
        .iter()
        .filter(|part| part.len() >= m.get())
        .map(|&part| (part, clean_windows(part, m.get())))
        .collect();
    for (part, clean) in &fitted {
        value = value.max(fit_floor(part, m, clean));
    }
    let mut scratch = FitScratch::default();
    // Bounded all the same (f64's own exponent range is under 2048 wide).
    for _ in 0..2048 {
        let scale = Scale::new(value)?;
        if fitted
            .iter()
            .all(|(part, clean)| fits_fast(part, m, scale, clean, &mut scratch))
        {
            return Some(scale);
        }
        value *= 2.0;
    }
    None
}

/// The bits of `x` as an `i64` ordered as finite floats are; applied twice, the bits again.
fn order_key(bits: u64) -> i64 {
    let bits = bits as i64;
    bits ^ (((bits >> 63) as u64) >> 1) as i64
}

/// A power of two below which [`fits_fast`] fails, or `0.0`: a clean window spanning `d` has ticks
/// spanning at least `d/s − 1` at scale `s`, so it does not fit below `d/(2·FAST_TICKS(m) + 1)`,
/// and below the bound more clean windows than one in [`OUTLIER_WINDOW_SHARE`] do not fit.
/// Counted on windows, not spans: sound, since a scale no window fits no span fits either, but
/// weaker, so the climb may start a step or two below the scale the spans need.
fn fit_floor(series: &[f64], m: NonZeroUsize, clean: &[bool]) -> f64 {
    let mut high = MonotonicDeque::<true>::new(m);
    let mut low = MonotonicDeque::<false>::new(m);
    let mut half_spans = Vec::with_capacity(clean.iter().filter(|&&c| c).count());
    for (p, &x) in series.iter().enumerate() {
        let key = order_key(if x.is_finite() { x } else { 0.0 }.to_bits());
        high.push(p as u64, key, m.get());
        low.push(p as u64, key, m.get());
        if let Some(row) = (p + 1).checked_sub(m.get())
            && clean[row]
        {
            let extreme = |key: i64| f64::from_bits(order_key(key as u64) as u64);
            half_spans.push(extreme(high.extreme()) / 2.0 - extreme(low.extreme()) / 2.0);
        }
    }
    let excusable = half_spans.len() / OUTLIER_WINDOW_SHARE;
    let Some(needed) = half_spans.len().checked_sub(excusable + 1) else {
        return 0.0;
    };
    let (_, &mut half_span, _) = half_spans.select_nth_unstable_by(needed, f64::total_cmp);
    // `half_span` is at most two roundings low; the margin keeps the bound below the true one.
    let bound = half_span * (1.0 - 4.0 * f64::EPSILON) / (fast_ticks(m) as f64 + 1.0);
    if bound > 0.0 {
        smallest_power_of_two_at_least(bound)
    } else {
        0.0
    }
}

/// The number of the rule that picks a scale when the caller gives none ([`ScaleChoice::Fast`]
/// and [`ScaleChoice::Finest`]). It changes whenever the scale either picks for any input
/// changes, and at no other time. With a scale the caller gives, no result depends on it.
///
/// Rule 1 is everything those two read: the grid candidates and their order
/// (`default_candidates`), the grid test (`GRID_ULPS`) and its size limit (`INFER_MAX_TICKS`), the
/// fast-path bound ([`FAST_TICKS_NUMERATOR`], [`fast_ticks`]), the span ([`SPAN_SLACK`]), the
/// outlier rule and the share of windows it may excuse ([`OUTLIER_WINDOW_SHARE`]), how several
/// series share one scale ([`ScaleChoice::resolve_each`]) and [`Scale::finest_fitting`].
/// [`MAX_TICKS`] is not part of it: it bounds what a given scale can count.
///
/// A caller who needs the same bits under a later rule takes the scale a call would use
/// ([`Scale::infer`], [`Scale::infer_finest`], [`ScaleChoice::resolve_each`]) and gives it from
/// then on. The integration test `scale_rule` pins the scales rule 1 picks on fixed inputs.
pub const SCALE_RULE: u32 = 1;

/// How a series' scale is chosen: given by the caller, or inferred by one of the rules.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ScaleChoice {
    /// The data's own grid, else [`Scale::fast_fitting`]: the default.
    #[default]
    Fast,
    /// The data's own grid, else [`Scale::finest_fitting`]: finer input resolution, at the cost
    /// of the exact i64 pass on most off-grid data.
    Finest,
    /// This tick size.
    Given(Scale),
}

impl From<Scale> for ScaleChoice {
    fn from(scale: Scale) -> Self {
        Self::Given(scale)
    }
}

impl From<Option<Scale>> for ScaleChoice {
    /// A given scale, or the default rule for `None`.
    fn from(scale: Option<Scale>) -> Self {
        scale.map_or(Self::Fast, Self::Given)
    }
}

impl ScaleChoice {
    /// The scale for `series` at window length `m`: the given one, or the rule's. An inferred
    /// scale is `None` for a `series` shorter than `m`, or with no finite sample.
    pub fn resolve(self, series: &[f64], m: NonZeroUsize) -> Option<Scale> {
        self.resolve_each(&[series], m)
    }

    /// The one scale for several series whose distances compare in one unit: the given one, or
    /// the grid every finite sample of them lies on, else the rule's. The fast rule is the finest
    /// power of two at which each series at least `m` long fits [`Scale::fast_fitting`]'s rule on
    /// its own, with its own outliers and its own share of them; the finest exact rule, and the
    /// grid, read the series end to end, apart by seams no window or span crosses. `None` as
    /// [`Self::resolve`] is, for the series end to end.
    pub fn resolve_each(self, parts: &[&[f64]], m: NonZeroUsize) -> Option<Scale> {
        if let Self::Given(scale) = self {
            return Some(scale);
        }
        let owned;
        let series: &[f64] = match parts {
            [one] => one,
            _ => {
                owned = joined(parts, m);
                &owned
            }
        };
        if series.len() < m.get() || !series.iter().any(|x| x.is_finite()) {
            return None;
        }
        Scale::grid(series, m).or_else(|| match self {
            Self::Finest => Scale::finest_fitting(series, m),
            _ => fast_fitting_each(parts, m),
        })
    }
}

#[cfg(test)]
mod tests;
