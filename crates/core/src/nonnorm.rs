//! The non-normalized metric: squared Euclidean distance between raw tick windows, kept as a
//! residue the recurrence updates in O(1) per diagonal step. The residue is exact whenever the
//! true distance fits its width, so a row runs in `u64` while `m·R² < 2⁶⁴` holds over the range
//! `R` of every tick the row can see, and in `u128` after that.
//!
//! The reported distance is Euclidean (the square root of the exact squared distance, in the
//! caller's units), as STUMPY's `aamp` reports it; the kernel itself ranks the exact integer.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::index::SampleIndex;
use crate::kernel::{Metric, Reducer};
use crate::narrow::{Path, RouteCounts};
use crate::nonnorm_lanes::{self, Fit, Lanes, Pass};
use crate::quantize::{Scale, to_ticks};
use crate::sanitize::{ForwardFill, Sanitized};

/// The width a diagonal's squared distance is kept in, wrapping modulo `2^BITS`.
pub(crate) trait Residue: Copy {
    /// `(a − b)²` modulo the width, exact for any two ticks.
    fn squared_difference(a: i64, b: i64) -> Self;
    fn zero() -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn as_u128(self) -> u128;
    /// The low bits of a stored residue: exact when the true value fits the width.
    fn from_stored(stored: u128) -> Self;
}

impl Residue for u64 {
    #[inline]
    fn squared_difference(a: i64, b: i64) -> Self {
        // The difference modulo 2⁶⁴ squares to the true square modulo 2⁶⁴.
        let difference = a.wrapping_sub(b) as u64;
        difference.wrapping_mul(difference)
    }
    fn zero() -> Self {
        0
    }
    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        u64::wrapping_add(self, other)
    }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        u64::wrapping_sub(self, other)
    }
    fn as_u128(self) -> u128 {
        u128::from(self)
    }
    #[inline]
    fn from_stored(stored: u128) -> Self {
        stored as u64
    }
}

impl Residue for u128 {
    #[inline]
    fn squared_difference(a: i64, b: i64) -> Self {
        let difference = (i128::from(a) - i128::from(b)).unsigned_abs();
        difference.wrapping_mul(difference)
    }
    fn zero() -> Self {
        0
    }
    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        u128::wrapping_add(self, other)
    }
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        u128::wrapping_sub(self, other)
    }
    fn as_u128(self) -> u128 {
        self
    }
    #[inline]
    fn from_stored(stored: u128) -> Self {
        stored
    }
}

/// Squared Euclidean distance on raw ticks, with the diagonal kept as a `R` residue.
pub(crate) struct NonNormalized<R = u64>(PhantomData<R>);

/// The ticks sliding a window from its previous position.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TickStep {
    /// `x[i+m−1]`, the window's newest tick.
    pub(crate) entering: i64,
    /// `x[i−1]`, the tick the window just dropped.
    pub(crate) leaving: i64,
}

impl<R: Residue> Metric for NonNormalized<R> {
    type Step = TickStep;
    type Diagonal = R;

    #[inline]
    fn start(a: &[i64], _a_step: &TickStep, b: &[i64], _b_step: &TickStep) -> R {
        a.iter().zip(b).fold(R::zero(), |total, (&x, &y)| {
            total.wrapping_add(R::squared_difference(x, y))
        })
    }

    #[inline]
    fn advance(previous: R, a: &TickStep, b: &TickStep) -> R {
        previous
            .wrapping_sub(R::squared_difference(a.leaving, b.leaving))
            .wrapping_add(R::squared_difference(a.entering, b.entering))
    }
}

/// A row's nearest neighbor: the exact squared distance in ticks and the neighbor's window start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonNormNeighbor {
    pub(crate) squared_ticks: u128,
    pub(crate) index: SampleIndex,
}

impl NonNormNeighbor {
    pub(crate) const NONE: Self = Self {
        squared_ticks: u128::MAX,
        index: SampleIndex::NONE,
    };

    /// The neighbor's window start, [`SampleIndex::NONE`] for no neighbor.
    pub const fn index(self) -> SampleIndex {
        self.index
    }

    /// The exact squared distance in ticks; `u128::MAX` for no neighbor.
    pub const fn squared_ticks(self) -> u128 {
        self.squared_ticks
    }

    /// The nearer of two neighbors of one window by exact squared distance, the smaller index on
    /// equal distances, `self` when both are the same; no neighbor loses to any.
    pub(crate) fn nearer(self, other: Self) -> Self {
        let key = |n: Self| (n.squared_ticks, n.index.get().unwrap_or(u64::MAX));
        if key(other) < key(self) { other } else { self }
    }

    /// The Euclidean distance in the caller's units; infinite for no neighbor.
    pub fn euclidean(self, scale: Scale) -> f64 {
        if self.index.is_none() {
            return f64::INFINITY;
        }
        libm::sqrt(self.squared_ticks as f64) * scale.get()
    }
}

/// One dimension's exact squared distance in ticks, as the multi-dimensional layer sees a
/// non-normalized pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SquaredTicks(pub(crate) u128);

impl SquaredTicks {
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl crate::multidim::DimensionDistance for SquaredTicks {
    fn root(self) -> f64 {
        libm::sqrt(self.0 as f64)
    }

    /// The conversion to `f64` and the root each round once, by at most `2^-53`.
    const ROOT_RELATIVE_ERROR: f64 = 1.0 / (1u64 << 51) as f64;

    fn push_exact(self, negative: bool, terms: &mut crate::multidim::Terms) {
        terms.push_root(negative, self.0);
    }

    fn fixed(self) -> Option<bool> {
        (self.0 == 0).then_some(false)
    }
}

/// The left, right and nearer neighbor of a window under `NonNormalized`; the nearer of the two
/// goes to the smaller index on equal distances. Looking right trades on future information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonNormTwoSided {
    pub left: NonNormNeighbor,
    pub right: NonNormNeighbor,
    pub nearest: NonNormNeighbor,
}

/// [`NonNormTwoSided`] with the `k` nearest neighbors over both sides: `left` and `right` stay each
/// side's one nearest, and `nearest` holds up to `k` neighbors, nearest first (none for a window
/// with no neighbor at all).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonNormTwoSidedTopk {
    pub left: NonNormNeighbor,
    pub right: NonNormNeighbor,
    pub nearest: Vec<NonNormNeighbor>,
}

/// One row's reduction: the smallest exact squared distance, ties to the smaller index.
#[derive(Default)]
pub(crate) struct NearestSquared {
    best: Option<(u128, u64)>,
}

impl NearestSquared {
    /// Takes `other`'s candidate in, leaving it empty; order-independent.
    pub(crate) fn merge(&mut self, other: &mut Self) {
        if let Some(candidate) = other.best.take() {
            self.offer(candidate);
        }
    }
}

impl Reducer for NearestSquared {
    type Candidate = (u128, u64);
    type Output = NonNormNeighbor;

    fn offer(&mut self, candidate: (u128, u64)) {
        if self.best.is_none_or(|current| candidate < current) {
            self.best = Some(candidate);
        }
    }

    fn finish(self) -> NonNormNeighbor {
        self.best
            .map_or(NonNormNeighbor::NONE, |(squared_ticks, index)| {
                NonNormNeighbor {
                    squared_ticks,
                    index: SampleIndex::new(index).expect("window starts stay below the sentinel"),
                }
            })
    }
}

/// `m·R²` for the range `R = hi − lo`, or `None` past `u128`.
fn scaled_square_of_range(lo: i64, hi: i64, m: usize) -> Option<u128> {
    let range = (i128::from(hi) - i128::from(lo)) as u128;
    range.checked_mul(range)?.checked_mul(m as u128)
}

/// The width a row's squared distances are exact in, by the range of every tick the row sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowWidth {
    /// `m·R² < 2⁶⁴`: every distance fits `u64`.
    Narrow,
    /// `m·R² < 2¹²⁸`: every distance fits `u128`.
    Wide,
    /// `m·R² ≥ 2¹²⁸`: no width holds the distances, so the row has no neighbor.
    Unrepresentable,
}

/// The width for ticks spanning `lo..=hi`; `hi < lo` means no ticks.
pub(crate) fn row_width(lo: i64, hi: i64, m: usize) -> RowWidth {
    if hi < lo {
        return RowWidth::Narrow;
    }
    match scaled_square_of_range(lo, hi, m) {
        Some(scaled) if scaled < 1 << 64 => RowWidth::Narrow,
        Some(_) => RowWidth::Wide,
        None => RowWidth::Unrepresentable,
    }
}

/// [`row_width`], except that a test can force every representable row to [`RowWidth::Wide`].
pub(crate) fn routed_row_width(lo: i64, hi: i64, m: usize) -> RowWidth {
    match row_width(lo, hi, m) {
        RowWidth::Narrow if crate::route::wide(false) => RowWidth::Wide,
        width => width,
    }
}

/// The sample as the arithmetic sees it: its tick, the last present one when the sample is
/// missing (`0` before any), and whether the sample was present.
pub(crate) fn sanitized_tick(fill: &mut ForwardFill<i64>, x: f64, scale: Scale) -> (i64, bool) {
    match fill.push(to_ticks(x, scale)) {
        Sanitized::Present(tick) => (tick, true),
        Sanitized::Filled(tick) => (tick, false),
        Sanitized::Leading => (0, false),
    }
}

/// Counts updates by the arithmetic they ran on, and the switches between the two.
#[derive(Default)]
pub(crate) struct RouteLog {
    counts: RouteCounts,
    last: Option<Path>,
}

impl RouteLog {
    pub(crate) fn record(&mut self, width: RowWidth) {
        let path = if width == RowWidth::Narrow {
            Path::Narrow
        } else {
            Path::Wide
        };
        self.counts.count_update(path);
        match (self.last, path) {
            (Some(Path::Narrow), Path::Wide) => self.counts.switches_to_wide += 1,
            (Some(Path::Wide), Path::Narrow) => self.counts.switches_to_narrow += 1,
            _ => {}
        }
        self.last = Some(path);
    }

    pub(crate) fn counts(&self) -> RouteCounts {
        self.counts
    }
}

/// The squared distance, modulo `2¹²⁸` and exact in `width` when the row fits it, between the
/// window ending at `ticks`' last tick and the one `lag` ticks before it.
pub(crate) fn start_diagonal(ticks: &[i64], lag: usize, m: usize, width: RowWidth) -> u128 {
    let newest = ticks.len();
    squared_between(
        &ticks[newest - m..],
        &ticks[newest - m - lag..newest - lag],
        width,
    )
}

/// The squared distance between two equal-length tick windows, modulo `2¹²⁸` and exact in `width`
/// when the pair fits it.
fn squared_between(a: &[i64], b: &[i64], width: RowWidth) -> u128 {
    let unused = TickStep {
        entering: 0,
        leaving: 0,
    };
    match width {
        RowWidth::Narrow => NonNormalized::<u64>::start(a, &unused, b, &unused).as_u128(),
        _ => NonNormalized::<u128>::start(a, &unused, b, &unused),
    }
}

/// The exact squared distance between two equal-length tick windows, `None` when it needs more
/// than 128 bits (`m·R² ≥ 2¹²⁸` over the range `R` of both windows' ticks).
pub(crate) fn exact_squared(a: &[i64], b: &[i64]) -> Option<u128> {
    let lowest = a.iter().chain(b).copied().min()?;
    let highest = a.iter().chain(b).copied().max()?;
    match routed_row_width(lowest, highest, a.len()) {
        RowWidth::Unrepresentable => None,
        width => Some(squared_between(a, b, width)),
    }
}

/// One row's view of the stream: the ticks and window completeness it can see, oldest first and
/// ending at the row's own, and the lag of the first live diagonal.
pub(crate) struct RowView<'a> {
    pub(crate) ticks: &'a [i64],
    /// Whether each window has every tick present; the last is the row's.
    pub(crate) complete: &'a [bool],
    pub(crate) m: usize,
    pub(crate) row: u64,
    pub(crate) first_lag: usize,
    /// How far apart any two of `ticks` can be.
    pub(crate) fit: Fit,
}

/// A row reducer the lane kernel can feed: every complete pair, or only the row's nearest.
pub(crate) trait LaneReducer: Reducer<Candidate = (u128, u64)> {
    /// Whether the reducer keeps only the nearest pair, so the kernel ranks the lanes itself.
    const NEAREST_ONLY: bool = false;
}

impl LaneReducer for NearestSquared {
    const NEAREST_ONLY: bool = true;
}

impl LaneReducer for crate::batch::nonnorm::KeptNearest<'_> {}

/// Carries every live diagonal from the previous row to this one and offers each pair whose two
/// windows are complete to `nearest`. `diagonals[k]` is the pair at lag `first_lag + k`; with
/// `fresh`, the last has no previous row and is summed from the ticks instead. An
/// [`RowWidth::Unrepresentable`] row advances its residues and offers nothing.
pub(crate) fn advance_row<N: LaneReducer>(
    diagonals: &mut [u128],
    fresh: bool,
    view: &RowView,
    width: RowWidth,
    nearest: &mut N,
) {
    let offer = view.complete.last() == Some(&true) && width != RowWidth::Unrepresentable;
    let live = diagonals.len() - usize::from(fresh);
    let (advancing, started) = diagonals.split_at_mut(live);
    match width {
        RowWidth::Narrow => advance_lanes(advancing, view, offer, nearest),
        _ => advance_all::<u128, _>(advancing, view, offer, nearest),
    }
    if let Some(diagonal) = started.first_mut() {
        let lag = view.first_lag + live;
        *diagonal = start_diagonal(view.ticks, lag, view.m, width);
        offer_pair(*diagonal, lag, view, offer, nearest);
    }
}

/// [`advance_all`] in `u64` through the lane kernel: lane `k` is the diagonal at lag
/// `first_lag + k`, whose neighbor's ticks run backwards through `view.ticks` as `k` grows.
fn advance_lanes<N: LaneReducer>(
    diagonals: &mut [u128],
    view: &RowView,
    offer: bool,
    nearest: &mut N,
) {
    let live = diagonals.len();
    let newest = view.ticks.len() - 1;
    let Some(leaving) = newest.checked_sub(view.m) else {
        return;
    };
    if live == 0 {
        return;
    }
    let row = TickStep {
        entering: view.ticks[newest],
        leaving: view.ticks[leaving],
    };
    let lag = view.first_lag;
    let windows = view.complete.len();
    let lanes = Lanes {
        enter: &view.ticks[newest + 1 - lag - live..=newest - lag],
        leave: &view.ticks[leaving + 1 - lag - live..=leaving - lag],
        complete: &view.complete[windows - lag - live..windows - lag],
    };
    let pass = if offer && N::NEAREST_ONLY {
        Pass::AdvanceAndRank
    } else {
        Pass::Advance
    };
    let found = nonnorm_lanes::descending(diagonals, row, lanes, view.fit, pass);
    if N::NEAREST_ONLY {
        if let Some((squared, k)) = found {
            nearest.offer((u128::from(squared), view.row - (lag + k) as u64));
        }
    } else {
        for (k, &squared) in diagonals.iter().enumerate() {
            offer_pair(squared, lag + k, view, offer, nearest);
        }
    }
}

fn advance_all<R: Residue, N: Reducer<Candidate = (u128, u64)>>(
    diagonals: &mut [u128],
    view: &RowView,
    offer: bool,
    nearest: &mut N,
) {
    let newest = view.ticks.len() - 1;
    let Some(leaving) = newest.checked_sub(view.m) else {
        return;
    };
    let a = TickStep {
        entering: view.ticks[newest],
        leaving: view.ticks[leaving],
    };
    for (k, stored) in diagonals.iter_mut().enumerate() {
        let lag = view.first_lag + k;
        let b = TickStep {
            entering: view.ticks[newest - lag],
            leaving: view.ticks[leaving - lag],
        };
        *stored = NonNormalized::<R>::advance(R::from_stored(*stored), &a, &b).as_u128();
        offer_pair(*stored, lag, view, offer, nearest);
    }
}

#[inline]
fn offer_pair<N: Reducer<Candidate = (u128, u64)>>(
    squared: u128,
    lag: usize,
    view: &RowView,
    offer: bool,
    nearest: &mut N,
) {
    if offer && view.complete[view.complete.len() - 1 - lag] {
        nearest.offer((squared, view.row - lag as u64));
    }
}

/// Whether each window of `m` ticks has all of them present, oldest first.
pub(crate) fn complete_windows(present: &[bool], m: usize) -> Vec<bool> {
    if present.len() < m {
        return Vec::new();
    }
    let mut missing = present[..m].iter().filter(|&&p| !p).count();
    let mut complete = Vec::with_capacity(present.len() - m + 1);
    complete.push(missing == 0);
    for k in m..present.len() {
        missing += usize::from(!present[k]);
        missing -= usize::from(!present[k - m]);
        complete.push(missing == 0);
    }
    complete
}

/// The newest consecutive present ticks, up to `m`.
pub(crate) fn trailing_run(present: &[bool], m: usize) -> usize {
    present.iter().rev().take(m).take_while(|&&p| p).count()
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::NonNormNeighbor;
    use crate::batch::nonnorm::left_profile_report;
    use crate::index::SampleIndex;
    use crate::naive::{NonNormDistance, naive_left_profile_nonnorm};
    use crate::quantize::Scale;
    use proptest::prelude::*;
    use std::vec::Vec;

    type Row = (u64, u128, u64);

    fn kernel_report(samples: &[Option<i64>], m: usize) -> (Vec<Row>, crate::batch::BatchReport) {
        let series: Vec<f64> = samples
            .iter()
            .map(|s| s.map_or(f64::NAN, |tick| tick as f64))
            .collect();
        let scale = Scale::new(1.0).unwrap();
        let threads = core::num::NonZeroUsize::MIN;
        let (rows, report) = left_profile_report(
            &series,
            core::num::NonZeroUsize::new(m).unwrap(),
            scale,
            threads,
        );
        let rows = rows
            .into_iter()
            .map(|(row, n)| match n.index.get() {
                Some(j) => (row, n.squared_ticks, j),
                None => (row, 0, u64::MAX),
            })
            .collect();
        (rows, report)
    }

    fn kernel_rows(samples: &[Option<i64>], m: usize) -> Vec<Row> {
        kernel_report(samples, m).0
    }

    fn naive_rows(samples: &[Option<i64>], m: usize) -> Vec<Row> {
        naive_left_profile_nonnorm(samples, m)
            .into_iter()
            .map(|(row, e)| match e.index().get() {
                Some(j) => {
                    let squared = match e.distance() {
                        NonNormDistance::Narrow(d) => u128::from(d),
                        NonNormDistance::Wide(d) => u128::try_from(d).unwrap(),
                    };
                    (row, squared, j)
                }
                None => (row, 0, u64::MAX),
            })
            .collect()
    }

    fn assert_matches_naive(samples: &[Option<i64>], m: usize) {
        assert_eq!(
            kernel_rows(samples, m),
            naive_rows(samples, m),
            "m = {m}, samples = {samples:?}"
        );
    }

    const LEVELS: [i64; 6] = [0, 1 << 20, 1 << 31, 1 << 33, -(1 << 40), (1 << 53) - 16];

    /// Walks that stay, repeat, step by a few ticks, jump between levels (some past the `u64`
    /// bound) and drop samples.
    fn series() -> impl Strategy<Value = Vec<Option<i64>>> {
        prop::collection::vec((0u8..10, -3i64..=3, 0usize..LEVELS.len()), 0..40).prop_map(|ops| {
            let mut current = 0i64;
            ops.into_iter()
                .map(|(op, noise, level)| match op {
                    0 | 1 => None,
                    2..=4 => Some(current),
                    5 => {
                        current = LEVELS[level];
                        Some(current)
                    }
                    _ => {
                        current = (current + noise).clamp(-(1 << 53), 1 << 53);
                        Some(current)
                    }
                })
                .collect()
        })
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn random_series_equal_naive(m in 1usize..=6, samples in series()) {
            assert_matches_naive(&samples, m);
        }

        #[test]
        fn window_nearly_the_series_equals_naive(
            samples in series(),
            back in 0usize..3,
            small_levels in any::<bool>(),
        ) {
            let m = samples.len().saturating_sub(back).max(1);
            let samples: Vec<_> = if small_levels {
                samples.iter().map(|s| s.map(|t| t.rem_euclid(50))).collect()
            } else {
                samples
            };
            assert_matches_naive(&samples, m);
        }

        #[test]
        fn exact_repeats_at_a_level_equal_naive(
            period in 1usize..8,
            pattern in prop::collection::vec(-5i64..=5, 8),
            level in 0usize..LEVELS.len(),
            m in 1usize..=6,
            len in 0usize..40,
        ) {
            let samples: Vec<_> = (0..len)
                .map(|k| Some(LEVELS[level] + pattern[k % period]))
                .collect();
            assert_matches_naive(&samples, m);
        }
    }

    #[test]
    fn flat_series_and_flat_runs_between_levels_tie_to_the_smaller_index() {
        let flat = [Some(7i64); 20];
        assert_matches_naive(&flat, 4);
        assert_eq!(kernel_rows(&flat, 4)[10], (10, 0, 0));
        let runs: Vec<_> = (0..40)
            .map(|k| Some(if (k / 7) % 2 == 0 { 0 } else { 1 << 33 }))
            .collect();
        for m in [3, 5, 8] {
            assert_matches_naive(&runs, m);
        }
    }

    #[test]
    fn gaps_at_the_start_and_end_and_between_windows() {
        let mut samples: Vec<Option<i64>> = (0..30).map(|k| Some((k * 7 % 11) as i64)).collect();
        for gap in [0, 1, 13, 14, 28, 29] {
            samples[gap] = None;
        }
        assert_matches_naive(&samples, 4);
        assert_matches_naive(&samples[..14], 3);
    }

    /// Small values, then values past the `u64` bound, then small again: early rows run narrow,
    /// later rows wide, and every row equals the exact reference.
    #[test]
    fn rows_either_side_of_the_bound_agree_with_the_reference() {
        let mut samples: Vec<Option<i64>> = (0..16).map(|k| Some(k % 5)).collect();
        samples.extend((0..16).map(|k| Some((1 << 33) + k % 5)));
        samples.extend((0..8).map(|k| Some(k % 5)));
        let m = 4;
        assert_matches_naive(&samples, m);
        let rows = kernel_rows(&samples, m);
        assert!(rows[16].1 > u128::from(u64::MAX));
        assert!(rows[10].1 < u128::from(u64::MAX));
    }

    /// `m·R²` exactly `2⁶⁴`: the one pair that far apart would wrap to zero in `u64`.
    #[test]
    fn a_pair_exactly_two_to_the_64_apart_is_not_wrapped() {
        let mut samples = Vec::new();
        samples.extend([Some(0i64); 4]);
        samples.push(None);
        samples.extend([Some(1i64 << 31); 4]);
        assert_matches_naive(&samples, 4);
        assert_eq!(kernel_rows(&samples, 4)[5], (5, 1u128 << 64, 0));
    }

    #[test]
    fn just_under_the_bound_stays_narrow_and_exact() {
        let mut samples = Vec::new();
        samples.extend([Some(0i64); 4]);
        samples.push(None);
        samples.extend([Some((1i64 << 31) - 1); 4]);
        assert_eq!(kernel_report(&samples, 4).1.wide_tiles, 0);
        assert_matches_naive(&samples, 4);
        assert_eq!(
            kernel_rows(&samples, 4)[5].1,
            4 * ((1u128 << 31) - 1).pow(2)
        );
    }

    #[test]
    fn row_width_follows_m_times_the_squared_range() {
        use super::{RowWidth, row_width};
        assert_eq!(row_width(i64::MAX, i64::MIN, 4), RowWidth::Narrow);
        assert_eq!(row_width(0, (1 << 31) - 1, 4), RowWidth::Narrow);
        assert_eq!(row_width(0, 1 << 31, 4), RowWidth::Wide);
        assert_eq!(row_width(-(1 << 53), 1 << 53, 1 << 19), RowWidth::Wide);
        assert_eq!(
            row_width(-(1 << 53), 1 << 53, 1 << 20),
            RowWidth::Unrepresentable
        );
    }

    /// Lag 2 of row 2 is `[3, 1]` against `[1, 2]`: 4 + 1.
    #[test]
    fn an_unrepresentable_row_offers_nothing() {
        use super::{NearestSquared, RowView, RowWidth, advance_row};
        use crate::kernel::Reducer;
        let view = RowView {
            ticks: &[1, 2, 3, 1],
            complete: &[true; 3],
            m: 2,
            row: 2,
            first_lag: 2,
            fit: crate::nonnorm_lanes::Fit::I32,
        };
        for width in [RowWidth::Wide, RowWidth::Narrow] {
            let mut nearest = NearestSquared::default();
            advance_row(&mut [0], true, &view, width, &mut nearest);
            let neighbor = nearest.finish();
            assert_eq!((neighbor.squared_ticks, neighbor.index.get()), (5, Some(0)));
        }
        let mut nearest = NearestSquared::default();
        advance_row(
            &mut [0],
            true,
            &view,
            RowWidth::Unrepresentable,
            &mut nearest,
        );
        assert_eq!(nearest.finish(), NonNormNeighbor::NONE);
    }

    #[test]
    fn literal_distance_bits() {
        let scale = |x: f64| Scale::new(x).unwrap();
        let exact = |squared_ticks| NonNormNeighbor {
            squared_ticks,
            index: SampleIndex::new(0).unwrap(),
        };
        assert_eq!(
            exact(25).euclidean(scale(1.0)).to_bits(),
            0x4014_0000_0000_0000
        );
        assert_eq!(
            exact(25).euclidean(scale(0.5)).to_bits(),
            0x4004_0000_0000_0000
        );
        assert_eq!(
            exact(2).euclidean(scale(1.0)).to_bits(),
            0x3FF6_A09E_667F_3BCD
        );
        assert_eq!(NonNormNeighbor::NONE.euclidean(scale(1.0)), f64::INFINITY);
    }

    /// `m = 2`, exclusion 1, so row `i` sees `j ≤ i − 2`; rows worked by hand.
    #[test]
    fn small_series_matches_hand_worked_rows() {
        let samples = [1i64, 2, 3, 1, 2, 9].map(Some);
        // windows: [1,2] [2,3] [3,1] [1,2] [2,9]
        let rows = kernel_rows(&samples, 2);
        assert_eq!(rows[0], (0, 0, u64::MAX));
        assert_eq!(rows[1], (1, 0, u64::MAX));
        assert_eq!(rows[2], (2, 5, 0)); // [3,1] vs [1,2]: 4 + 1
        assert_eq!(rows[3], (3, 0, 0)); // [1,2] vs [1,2]
        assert_eq!(rows[4], (4, 36, 1)); // [2,9] vs [2,3]: 0 + 36
        assert_matches_naive(&samples, 2);
    }
}
