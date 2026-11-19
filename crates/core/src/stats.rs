//! Exact running statistics of the last `m` ticks: their wrapped sum, and whether the window
//! is non-finite, constant, or valid with an exact variance numerator and its reciprocal
//! square root. The ring and both extreme-tracking deques are allocated once, in `new`; every
//! `push` after that is O(1) amortized with no further allocation.

use alloc::boxed::Box;
use alloc::vec;
use core::num::NonZeroUsize;

use crate::ring::MirroredRing;

mod constant_class;

pub(crate) use constant_class::{
    ClassBlocks, ConstantClass, JoinClass, JoinRows, RowClass, RowsClass,
};

/// One complete window: its ticks' sum, reduced mod 2⁶⁴, and which of three kinds it is.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub sum: i64,
    pub kind: WindowKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WindowKind {
    /// A sample in the window had no tick, or the window's range puts its variance numerator
    /// at or beyond 2⁶³.
    NonFinite,
    /// The window's maximum and minimum tick are equal.
    Constant,
    Valid {
        variance_numerator: u64,
        inv_sqrt: f64,
    },
}

impl Window {
    /// This window under the caller's flag for it, if it has one.
    pub(crate) fn under(self, flag: Option<bool>) -> Self {
        match flag {
            None => self,
            Some(flagged_constant) => self.under_flag(flagged_constant),
        }
    }

    /// Whether `self` is `classified`, a window as the default rule classifies it, under some
    /// flag or none: what a stored window of a stream whose caller flags windows can be.
    pub(crate) fn is_flagged_form_of(self, classified: Self) -> bool {
        self == classified
            || self == classified.under_flag(true)
            || self == classified.under_flag(false)
    }

    /// This window under the caller's flag for it: a finite window flagged constant is
    /// [`WindowKind::Constant`]; a window that is constant by its ticks but not flagged is
    /// [`FLAT_NOT_CONSTANT`]; a non-finite window is non-finite whatever the flag.
    pub(crate) fn under_flag(self, flagged_constant: bool) -> Self {
        let kind = match (self.kind, flagged_constant) {
            (WindowKind::NonFinite, _) => WindowKind::NonFinite,
            (_, true) => WindowKind::Constant,
            (WindowKind::Constant, false) => FLAT_NOT_CONSTANT,
            (kind @ WindowKind::Valid { .. }, false) => kind,
        };
        Self { kind, ..self }
    }
}

/// A window whose ticks are all equal that the caller did not flag constant: its variance
/// numerator is zero, so every centered cross term against it is exactly zero and its
/// correlation with any window is exactly zero, whatever finite reciprocal root it carries (`1`
/// here, because a zero one would read as [`WindowKind::Constant`] in the kernel's columns).
pub(crate) const FLAT_NOT_CONSTANT: WindowKind = WindowKind::Valid {
    variance_numerator: 0,
    inv_sqrt: 1.0,
};

/// The number of flags a series' windows need: one per window of `m` samples.
fn window_count(len: usize, m: NonZeroUsize) -> usize {
    (len + 1).saturating_sub(m.get())
}

/// A caller's flags for which windows are constant, of the wrong length for the series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstantFlagsLength {
    pub expected: usize,
    pub found: usize,
}

/// A series, and optionally the caller's own flag for each of its windows being constant (the
/// rule that replaces "all its ticks are equal"). Every z-normalized engine takes a window's
/// constancy from its series, so a flag reaches all of them the same way; a plain `&[f64]`
/// converts to a series with no flags, whose windows are constant exactly when their ticks are
/// equal. A flag on a window with a non-finite sample, or one whose variance is out of range, is
/// ignored: such a window has no distance.
#[derive(Clone, Copy, Debug)]
pub struct Series<'a> {
    values: &'a [f64],
    constant: Option<&'a [bool]>,
}

impl<'a> Series<'a> {
    /// `values` with `flags[i]` saying whether the window of `m` samples at `i` is constant: one
    /// flag per window, [`ConstantFlagsLength`] otherwise. A flag on a window that holds a
    /// non-finite sample is ignored: such a window is non-finite first.
    pub fn with_constant_windows(
        values: &'a [f64],
        m: NonZeroUsize,
        flags: &'a [bool],
    ) -> Result<Self, ConstantFlagsLength> {
        let expected = window_count(values.len(), m);
        if flags.len() != expected {
            return Err(ConstantFlagsLength {
                expected,
                found: flags.len(),
            });
        }
        Ok(Self {
            values,
            constant: Some(flags),
        })
    }

    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The caller's flags, `None` for the default rule.
    pub fn constant_windows(&self) -> Option<&'a [bool]> {
        self.constant
    }

    /// The one place a window's constancy is decided: the window at `row` as the default rule
    /// classified it, then under the caller's flag for it if there is one.
    pub(crate) fn window(&self, row: usize, window: Window, m: NonZeroUsize) -> Window {
        match self.constant {
            None => window,
            Some(flags) => {
                assert_eq!(
                    flags.len(),
                    window_count(self.values.len(), m),
                    "one constant flag per window"
                );
                window.under(Some(flags[row]))
            }
        }
    }
}

impl<'a> From<&'a [f64]> for Series<'a> {
    fn from(values: &'a [f64]) -> Self {
        Self {
            values,
            constant: None,
        }
    }
}

impl<'a> From<&'a alloc::vec::Vec<f64>> for Series<'a> {
    fn from(values: &'a alloc::vec::Vec<f64>) -> Self {
        values.as_slice().into()
    }
}

impl<'a, const N: usize> From<&'a [f64; N]> for Series<'a> {
    fn from(values: &'a [f64; N]) -> Self {
        values.as_slice().into()
    }
}

/// Tracks the current window's maximum (`MAX`) or minimum tick, in a fixed-capacity ring deque
/// allocated once, in O(1) amortized per push.
pub(crate) struct MonotonicDeque<const MAX: bool> {
    /// `(position, tick)` pairs, oldest at `head`, in order non-increasing (`MAX`) or
    /// non-decreasing by tick.
    entries: Box<[(u64, i64)]>,
    head: usize,
    len: usize,
}

impl<const MAX: bool> MonotonicDeque<MAX> {
    pub(crate) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            entries: vec![(0u64, 0i64); capacity.get()].into_boxed_slice(),
            head: 0,
            len: 0,
        }
    }

    fn slot(&self, offset: usize) -> usize {
        (self.head + offset) % self.entries.len()
    }

    /// Whether `candidate` will still be in the window for as long as `incumbent` could be, so
    /// `incumbent` can never again be the extreme.
    fn supersedes(candidate: i64, incumbent: i64) -> bool {
        if MAX {
            candidate >= incumbent
        } else {
            candidate <= incumbent
        }
    }

    /// Folds in the tick at `position`, having evicted whatever `window` ticks has aged out.
    pub(crate) fn push(&mut self, position: u64, tick: i64, window: usize) {
        while self.len > 0 && self.entries[self.head].0 + window as u64 <= position {
            self.head = self.slot(1);
            self.len -= 1;
        }
        while self.len > 0 && Self::supersedes(tick, self.entries[self.slot(self.len - 1)].1) {
            self.len -= 1;
        }
        self.entries[self.slot(self.len)] = (position, tick);
        self.len += 1;
    }

    pub(crate) fn extreme(&self) -> i64 {
        self.entries[self.head].1
    }
}

fn square(tick: i64) -> i128 {
    let tick = i128::from(tick);
    tick * tick
}

/// `V = m·Q − S²`, computed in wrapping i128 on `sum` and `sum_sq` as maintained. By this
/// point `range` has shown the true `V` fits i128 (`V ≤ m²R²/4`, and `m < 2⁶⁴` on any
/// reachable `usize` keeps that under `2¹²⁶`), so this wrapping arithmetic reproduces the
/// exact `V`, however far `m·Q` or `S²` individually run past i128 on the way there. The same
/// range also lower-bounds `V ≥ m·R²/2`, which is why failing it already proves `V ≥ 2⁶³`
/// without computing `V` at all: the range gate below never disagrees with this exact check,
/// only runs cheaper than it. `None` when `V ≥ 2⁶³`.
fn numerator(window: usize, sum: i128, sum_sq: i128, max: i64, min: i64) -> Option<u64> {
    let range = u128::from((i128::from(max) - i128::from(min)) as u64);
    // `m·R² ≥ 2⁶⁴ ⇔ m·R²/2 ≥ 2⁶³`; a checked multiply that overflows is an even larger bound.
    let out_of_range = range
        .checked_mul(range)
        .and_then(|r_squared| r_squared.checked_mul(window as u128))
        .is_none_or(|m_r_squared| m_r_squared >= 1u128 << 64);
    if out_of_range {
        return None;
    }
    let variance_numerator = (window as i128)
        .wrapping_mul(sum_sq)
        .wrapping_sub(sum.wrapping_mul(sum));
    u64::try_from(variance_numerator)
        .ok()
        .filter(|&v| v < 1 << 63)
}

/// A window with a missing tick is non-finite, one whose ticks are all equal constant, and any
/// other valid when its [`numerator`] is below `2⁶³`.
fn classify(
    window: usize,
    missing: usize,
    sum: i128,
    sum_sq: i128,
    max: i64,
    min: i64,
) -> WindowKind {
    if missing > 0 {
        return WindowKind::NonFinite;
    }
    if max == min {
        return WindowKind::Constant;
    }
    match numerator(window, sum, sum_sq, max, min) {
        None => WindowKind::NonFinite,
        Some(variance_numerator) => WindowKind::Valid {
            variance_numerator,
            inv_sqrt: 1.0 / libm::sqrt(variance_numerator as f64),
        },
    }
}

/// Exact running `S = Σx` and `Q = Σx²` of the last `m` ticks, their range, and how many of
/// them had no tick.
#[doc(hidden)]
pub struct WindowStats {
    ticks: MirroredRing<(i64, bool)>,
    sum: i128,
    sum_sq: i128,
    missing: usize,
    position: u64,
    max: MonotonicDeque<true>,
    min: MonotonicDeque<false>,
}

impl WindowStats {
    pub fn new(m: NonZeroUsize) -> Self {
        Self {
            ticks: MirroredRing::new(m, (0, false)),
            sum: 0,
            sum_sq: 0,
            missing: 0,
            position: 0,
            max: MonotonicDeque::new(m),
            min: MonotonicDeque::new(m),
        }
    }

    /// `tick` is the (possibly forward-filled) value; `had_tick` is false when it was filled.
    pub fn push(&mut self, tick: i64, had_tick: bool) -> Option<Window> {
        let window = self.ticks.capacity();
        if self.ticks.len() == window {
            let (departing, departing_had_tick) = self.ticks.as_slice()[0];
            self.sum = self.sum.wrapping_sub(i128::from(departing));
            self.sum_sq = self.sum_sq.wrapping_sub(square(departing));
            if !departing_had_tick {
                self.missing -= 1;
            }
        }
        self.ticks.push((tick, had_tick));
        self.sum = self.sum.wrapping_add(i128::from(tick));
        self.sum_sq = self.sum_sq.wrapping_add(square(tick));
        if !had_tick {
            self.missing += 1;
        }
        self.max.push(self.position, tick, window);
        self.min.push(self.position, tick, window);
        self.position += 1;

        if self.ticks.len() < window {
            return None;
        }
        Some(Window {
            sum: self.sum as i64,
            kind: classify(
                window,
                self.missing,
                self.sum,
                self.sum_sq,
                self.max.extreme(),
                self.min.extreme(),
            ),
        })
    }

    /// The exact `V = m·Q − S²` of the last `m` ticks pushed, filled ones at their filled value
    /// and whatever the window's kind, or `None` when it is `2⁶³` or more. Read only after
    /// [`Self::push`] has returned a window.
    pub(crate) fn numerator(&self) -> Option<u64> {
        numerator(
            self.ticks.capacity(),
            self.sum,
            self.sum_sq,
            self.max.extreme(),
            self.min.extreme(),
        )
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{MonotonicDeque, Window, WindowKind, WindowStats, classify};
    use crate::quantize::{Scale, to_ticks};
    use core::num::NonZeroUsize;
    use proptest::prelude::*;
    use std::vec::Vec;
    use testkit::generators::{MARKET_TICK, market_series_and_m};

    // ---------- matches_direct_sums ----------

    /// Non-finite or out-of-range samples are quantized first, then forward-filled on ticks:
    /// the filled value is the last tick that quantized, or 0 before any has.
    fn sanitize_to_ticks(series: &[f64], scale: Scale) -> Vec<(i64, bool)> {
        let mut last = 0i64;
        series
            .iter()
            .map(|&x| match to_ticks(x, scale) {
                Some(tick) => {
                    last = tick;
                    (tick, true)
                }
                None => (last, false),
            })
            .collect()
    }

    fn direct_stats(window: &[(i64, bool)]) -> (i128, i128, usize, i64, i64) {
        let sum: i128 = window.iter().map(|&(t, _)| i128::from(t)).sum();
        let sum_sq: i128 = window
            .iter()
            .map(|&(t, _)| {
                let t = i128::from(t);
                t * t
            })
            .sum();
        let missing = window.iter().filter(|&&(_, had_tick)| !had_tick).count();
        let max = window.iter().map(|&(t, _)| t).max().unwrap();
        let min = window.iter().map(|&(t, _)| t).min().unwrap();
        (sum, sum_sq, missing, max, min)
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// `push`'s rolling `S`, `Q` and missing count, over the actual last-`m` ticks, match a
        /// direct i128 recomputation over the same slice, on ticks small enough that the direct
        /// sums can't themselves overflow i128.
        #[test]
        fn matches_direct_sums((series, m) in market_series_and_m(2..64)) {
            let scale = Scale::new(MARKET_TICK).unwrap();
            let ticks = sanitize_to_ticks(&series, scale);
            let mut stats = WindowStats::new(NonZeroUsize::new(m).unwrap());

            for (k, &(tick, had_tick)) in ticks.iter().enumerate() {
                let window = stats.push(tick, had_tick);
                if k + 1 < m {
                    prop_assert_eq!(window, None);
                    continue;
                }
                let (sum, sum_sq, missing, max, min) = direct_stats(&ticks[k + 1 - m..=k]);
                let expected = Window {
                    sum: sum as i64,
                    kind: classify(m, missing, sum, sum_sq, max, min),
                };
                prop_assert_eq!(window, Some(expected));
            }
        }
    }

    // ---------- kind_precedence ----------

    #[test]
    fn missing_tick_beats_a_flat_window() {
        let m = NonZeroUsize::new(4).unwrap();
        let mut stats = WindowStats::new(m);
        for _ in 0..3 {
            stats.push(10, true);
        }
        // Filled with the level itself, so the window looks flat; the missing tick still wins.
        let window = stats.push(10, false).unwrap();
        assert_eq!(window.kind, WindowKind::NonFinite);
    }

    #[test]
    fn a_non_finite_window_still_has_its_numerator() {
        let mut stats = WindowStats::new(NonZeroUsize::new(3).unwrap());
        stats.push(0, true);
        stats.push(5, false);
        let window = stats.push(1, true).unwrap();
        assert_eq!(window.kind, WindowKind::NonFinite);
        // 3 · (0 + 25 + 1) − 6².
        assert_eq!(stats.numerator(), Some(42));
        stats.push(i64::MAX, true);
        assert_eq!(stats.numerator(), None);
    }

    #[test]
    fn a_numerator_that_wraps_to_a_small_value_is_still_out_of_range() {
        // `V = 4 · (2⁶³)² = 2¹²⁸`, which the wrapping sums read as 0; `m·R² = 2⁶⁶` rules it out.
        let mut stats = WindowStats::new(NonZeroUsize::new(4).unwrap());
        for tick in [-(1 << 62), -(1 << 62), 1 << 62, 1 << 62] {
            stats.push(tick, true);
        }
        assert_eq!(stats.numerator(), None);
        // At the gate's edge, `m·R² = 2⁶⁴`, every window has `V ≥ m·R²/2 = 2⁶³`.
        let mut edge = WindowStats::new(NonZeroUsize::new(4).unwrap());
        for tick in [0, 1 << 30, 1 << 30, 1 << 31] {
            edge.push(tick, true);
        }
        assert_eq!(edge.numerator(), None);
    }

    #[test]
    fn missing_count_decrements_once_the_gap_leaves_the_window() {
        let m = NonZeroUsize::new(3).unwrap();
        let mut stats = WindowStats::new(m);
        stats.push(5, true); // position 0
        stats.push(5, false); // position 1: the gap
        let with_gap = stats.push(5, true).unwrap(); // window [0, 1, 2]: gap inside
        assert_eq!(with_gap.kind, WindowKind::NonFinite);
        let gap_still_in_window = stats.push(5, true).unwrap(); // window [1, 2, 3]: still inside
        assert_eq!(gap_still_in_window.kind, WindowKind::NonFinite);
        let gap_evicted = stats.push(5, true).unwrap(); // window [2, 3, 4]: gap finally out
        assert_eq!(gap_evicted.kind, WindowKind::Constant);
    }

    // ---------- range_boundary ----------

    /// One tick at `0`, one at `2 · half_range`, and every other tick at `half_range`: the
    /// shape that minimizes `V` for a given range, so `V` lands exactly at `window · range² /
    /// 2` instead of somewhere in the wide gap up to the shape-dependent upper bound.
    fn midpoint_spread(m: NonZeroUsize, half_range: i64) -> Window {
        let mut stats = WindowStats::new(m);
        stats.push(0, true);
        for _ in 2..m.get() {
            stats.push(half_range, true);
        }
        stats.push(2 * half_range, true).unwrap()
    }

    /// Smallest `h` with `window · h² ≥ 2⁶²`, i.e. `window · (2h)²/2 ≥ 2⁶³`; found by binary
    /// search since `h` itself can run past a billion.
    fn smallest_half_range_at_the_bound(window: usize) -> u128 {
        let (mut low, mut high) = (0u128, 1u128 << 40);
        while low < high {
            let mid = low + (high - low) / 2;
            if (window as u128) * mid * mid >= 1u128 << 62 {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }

    #[test]
    fn range_boundary() {
        for &window in &[2usize, 1_000] {
            let h = smallest_half_range_at_the_bound(window);
            let m = NonZeroUsize::new(window).unwrap();

            let below = midpoint_spread(m, (h - 1) as i64);
            assert!(
                matches!(below.kind, WindowKind::Valid { .. }),
                "window {window}, h {h}: {below:?}"
            );

            let at = midpoint_spread(m, h as i64);
            assert_eq!(at.kind, WindowKind::NonFinite, "window {window}, h {h}");
        }
    }

    #[test]
    fn range_test_runs_before_v_can_wrap_into_looking_valid() {
        // R = 2⁶⁴ − 1: V = R² wraps to a small (even negative) i128 if read without the range
        // test first, which the range test alone must catch.
        let m = NonZeroUsize::new(2).unwrap();
        let mut stats = WindowStats::new(m);
        stats.push(i64::MIN, true);
        let window = stats.push(i64::MAX, true).unwrap();
        assert_eq!(window.kind, WindowKind::NonFinite);
    }

    /// `(m - 1)` zeros plus one tick at `spike`: range `spike`, `V = (m - 1) · spike²` exactly,
    /// so a caller can choose `m` and `spike` to land precisely on a threshold.
    fn single_spike(m: NonZeroUsize, spike: i64) -> Window {
        let mut stats = WindowStats::new(m);
        for _ in 1..m.get() {
            stats.push(0, true);
        }
        stats.push(spike, true).unwrap()
    }

    #[test]
    fn pre_check_threshold_is_closed_not_open() {
        // `midpoint_spread`'s `V = window · R²/2` exactly, so at `window = 4`, `R = 2 ·
        // half_range = 2³¹`, `window · R²` lands on exactly 2⁶⁴; one less `half_range` lands
        // just under it. `≥`, not `>`, is what the gate must use.
        let m = NonZeroUsize::new(4).unwrap();
        let half_range: i64 = 1 << 30;
        let window = i128::from(m.get() as i64);

        let r_at = 2 * i128::from(half_range);
        assert_eq!(window * r_at * r_at, 1i128 << 64);
        let at = midpoint_spread(m, half_range);
        assert_eq!(at.kind, WindowKind::NonFinite);

        let half_range_below = half_range - 1;
        let r_below = 2 * i128::from(half_range_below);
        assert!(window * r_below * r_below < (1i128 << 64));
        let below = midpoint_spread(m, half_range_below);
        assert!(matches!(below.kind, WindowKind::Valid { .. }));
    }

    #[test]
    fn variance_threshold_is_closed_not_open() {
        // `(m - 1) · spike²` lands on exactly 2⁶³ at m = 3, spike = 2³¹; one less `spike`
        // lands just under it. `≥`, not `>`, is what this check must use.
        let m = NonZeroUsize::new(3).unwrap();
        let spike: i64 = 1 << 31;

        let v_at = i128::from(m.get() as i64 - 1) * i128::from(spike) * i128::from(spike);
        assert_eq!(v_at, 1i128 << 63);
        let at = single_spike(m, spike);
        assert_eq!(at.kind, WindowKind::NonFinite);

        let spike_below = spike - 1;
        let v_below =
            i128::from(m.get() as i64 - 1) * i128::from(spike_below) * i128::from(spike_below);
        assert!(v_below < (1i128 << 63));
        let below = single_spike(m, spike_below);
        match below.kind {
            WindowKind::Valid {
                variance_numerator, ..
            } => assert_eq!(i128::from(variance_numerator), v_below),
            other => panic!("expected Valid, got {other:?}"),
        }
    }

    // ---------- deque_tracks_extremes ----------

    fn naive_extreme<const MAX: bool>(window: &[i64]) -> i64 {
        if MAX {
            *window.iter().max().unwrap()
        } else {
            *window.iter().min().unwrap()
        }
    }

    fn check_deque<const MAX: bool>(values: &[i64], window: usize) {
        let mut deque = MonotonicDeque::<MAX>::new(NonZeroUsize::new(window).unwrap());
        for (position, &value) in values.iter().enumerate() {
            deque.push(position as u64, value, window);
            if position + 1 >= window {
                let slice = &values[position + 1 - window..=position];
                assert_eq!(
                    deque.extreme(),
                    naive_extreme::<MAX>(slice),
                    "MAX {MAX}, window {window}, position {position}, values {values:?}"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// A small value range makes repeats and runs common, which is where a stale extreme
        /// would show up.
        #[test]
        fn deque_tracks_extremes(
            values in prop::collection::vec(-3i64..=3, 1..40),
            window_pick in 0usize..40,
        ) {
            let window = (window_pick % values.len()) + 1;
            check_deque::<true>(&values, window);
            check_deque::<false>(&values, window);
        }
    }

    #[test]
    fn deque_tracks_extremes_named_patterns() {
        let ascending: Vec<i64> = (0..20).collect();
        let descending: Vec<i64> = (0..20).rev().collect();
        let repeats = Vec::from([5i64; 20]);
        let alternating: Vec<i64> = (0..20).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        for values in [ascending, descending, repeats, alternating] {
            for window in [1, 3, 7, 20] {
                check_deque::<true>(&values, window);
                check_deque::<false>(&values, window);
            }
        }
    }

    // ---------- Q wrapping, not saturating ----------

    #[test]
    fn variance_numerator_exact_despite_sum_of_squares_overflowing_i128() {
        // V is shift-invariant, so three ticks near i64::MAX (whose squares already exceed
        // i128::MAX once summed) still have the tiny exact V of their small differences.
        let m = NonZeroUsize::new(3).unwrap();
        let mut stats = WindowStats::new(m);
        stats.push(i64::MAX, true);
        stats.push(i64::MAX - 1, true);
        let window = stats.push(i64::MAX - 2, true).unwrap();
        assert_eq!(
            window.kind,
            WindowKind::Valid {
                variance_numerator: 6,
                inv_sqrt: 1.0 / libm::sqrt(6.0),
            }
        );
    }
}
