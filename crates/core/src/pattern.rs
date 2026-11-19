//! Pattern matching: the exact distance between a query and a window of a series, z-normalized
//! or not. The z-normalized query is quantized at its own finest-fitting scale and the series at
//! its own; the pair's entry comes from [`Nearest`], the conversion every profile uses.
//!
//! The metric is a type parameter, `Measure`, chosen once at the entry point: `ZNorm` (this
//! module's plain functions, reporting [`Entry`]) or `NonNormalized` (the `_nonnorm` functions,
//! reporting [`NonNormNeighbor`], whose `euclidean(scale)` is the distance in the caller's units).
//! Every derived procedure (matching, [`motifs`](crate::motif::motifs),
//! [`mpdist`](crate::mpdist::mpdist) and [`ostinato`](crate::ostinato::ostinato))
//! is written once over `TickedSeries<M>` with `M: Measure`, and each metric's public function is
//! a one-line instantiation of it. A procedure never branches on the metric: whatever differs
//! (the windows, the exact pair distance and its order, the self match, the reported distance,
//! the profile that picks motif candidates, the join of two series) is a `Measure` item. The
//! non-normalized metric quantizes the query at the series' own scale (the one scale every
//! distance shares, so a query off that grid is rounded to it), and a pair has no distance when
//! it needs more than 128 bits.
//!
//! The query is quantized once, at registration: at the stream's scale when every sample
//! lies on its grid ([`Scale::for_query`], the only place that decides), so an exact repeat
//! scores exactly 0 and every entry equals the stream's own computation on bits; otherwise at
//! the query's own finest-fitting scale. Results are never rounded afterwards.
//!
//! Why two scales are exact: ticks are `x / scale` rounded, and `ρ = N / √(V_q·V_x)` with
//! `N = m·Σ(q − q̄)(x − x̄)` is unchanged when either factor's units are multiplied by a positive
//! constant (both `N` and the matching `V` scale by it), so mixing two scales needs no
//! conversion; `|N| ≤ √(V_q·V_x) < 2⁶³` still holds (Cauchy–Schwarz), so the wrapping `N` is exact.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use core::cmp::Ordering;
use core::fmt::Debug;

use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::kernel::Metric;
use crate::multidim::DimensionDistance;
use crate::nearest::{ExactDistance, Nearest, Sharing};
use crate::nonnorm::{
    NonNormNeighbor, NonNormalized, Residue, SquaredTicks, TickStep, complete_windows,
    exact_squared, sanitized_tick,
};
use crate::quantize::{Scale, to_ticks};
use crate::ring::MirroredRing;
use crate::sanitize::ForwardFill;
use crate::stats::{Series, Window, WindowStats};
use crate::znorm::{TickWindow, ZNorm, centered_cross};

/// A quantized query: its ticks and the metric's per-window values for them.
pub(crate) struct Query<W> {
    ticks: Vec<i64>,
    window: W,
}

/// What derived procedures need of a metric (per-window validity, the exact pair distance and its
/// order, the reported hit, the distance in the caller's units), written once over `M: Measure`
/// (see the module docs). It extends [`Metric`], the kernel's per-diagonal arithmetic: one
/// concept in two layers. Implemented by `ZNorm` and `NonNormalized`.
pub(crate) trait Measure: Metric {
    /// What the metric keeps per window: its statistics (z-normalized) or whether it is complete
    /// (non-normalized).
    type Window: Copy + Send + Sync;
    /// A pair's exact distance, ordered by [`Self::compare`].
    type Exact: Copy;
    /// What a match reports.
    type Hit: Copy + Debug;

    /// A window with no distance.
    const NO_HIT: Self::Hit;

    /// Every window's values, from the forward-filled `ticks` and whether each had a tick, as
    /// `series` (the samples they came from, and the caller's constancy flags) says.
    fn windows(
        ticks: &[i64],
        present: &[bool],
        m: NonZeroUsize,
        series: &Series<'_>,
    ) -> Vec<Self::Window>;

    /// The query as a series quantized at `scale` sees it; `None` for an empty query or one that
    /// cannot be quantized.
    fn quantize_query(query: Series<'_>, scale: Scale) -> Option<Query<Self::Window>>;

    /// The exact distance from `query` to the window `ticks` (`window` its values) starting at
    /// `start`, `None` when either has no distance.
    fn exact(
        query: &Query<Self::Window>,
        ticks: &[i64],
        window: &Self::Window,
        start: u64,
    ) -> Option<Self::Exact>;

    /// The exact order, nearer first; `sharing` says what the two distances have in common.
    fn compare(a: &Self::Exact, b: &Self::Exact, sharing: Sharing) -> Ordering;

    fn hit(exact: &Self::Exact) -> Self::Hit;

    /// A window's match with itself, at distance zero.
    fn own(start: u64) -> Self::Hit;

    fn index(hit: &Self::Hit) -> Option<u64>;

    /// What the multi-dimensional layer averages of one dimension's exact distance.
    type Dimension: DimensionDistance;

    /// `exact` as the multi-dimensional layer sees it.
    fn dimension(exact: &Self::Exact) -> Self::Dimension;

    /// A mean of dimensions' roots in the caller's units: unchanged for the z-normalized
    /// metric, whose distances have none, and times `scale` for ticks.
    fn caller_units(mean: f64, scale: Scale) -> f64;

    /// The distance in the caller's units, infinite for [`Self::NO_HIT`].
    fn distance(hit: &Self::Hit, scale: Scale) -> f64;

    /// Every window's nearest neighbor on either side, over the whole series, on up to `threads`.
    fn nearest(
        series: Series<'_>,
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
    ) -> Vec<Self::Hit>;

    /// Every window of `a`'s nearest window of `b`, with no exclusion zone, on up to `threads`.
    /// `a` and `b` are quantized at `scale_a` and `scale_b`; a metric whose distances only
    /// compare in one unit takes both to be the same.
    fn ab_join(
        a: Series<'_>,
        b: Series<'_>,
        m: NonZeroUsize,
        scale_a: Scale,
        scale_b: Scale,
        threads: NonZeroUsize,
    ) -> Vec<Self::Hit>;

    /// The step of the window with these values along a diagonal, `entering` and `leaving` being
    /// the ticks it takes in and drops.
    fn step(window: &Self::Window, entering: i64, leaving: i64) -> Self::Step;

    /// One column of a sliding sweep: `rows` (windows of `m` samples) against `column` (the
    /// window starting at `start`), `states` their diagonal states. `pair(i, d)` gets the exact
    /// distance `d` of row `i` to the column for each row with one, in order. Returns the exactly
    /// nearest row's distance to the column; only the distance is observable, so which of equal
    /// rows is reported does not matter.
    fn scan_column(
        states: &[Self::Diagonal],
        rows: &[Self::Window],
        column: &Self::Window,
        m: usize,
        start: u64,
        pair: impl FnMut(usize, Self::Exact),
    ) -> Option<Self::Exact>;
}

impl Measure for ZNorm {
    type Window = Window;
    type Exact = ExactDistance;
    type Hit = Entry;
    type Dimension = ExactDistance;

    const NO_HIT: Entry = Entry::NONE;

    fn windows(
        ticks: &[i64],
        present: &[bool],
        m: NonZeroUsize,
        series: &Series<'_>,
    ) -> Vec<Window> {
        let mut stats = WindowStats::new(m);
        let mut windows = Vec::with_capacity((ticks.len() + 1).saturating_sub(m.get()));
        windows.extend(
            ticks
                .iter()
                .zip(present)
                .filter_map(|(&tick, &had_tick)| stats.push(tick, had_tick))
                .enumerate()
                .map(|(row, window)| series.window(row, window, m)),
        );
        windows
    }

    fn quantize_query(query: Series<'_>, scale: Scale) -> Option<Query<Window>> {
        quantize_query(query, scale)
    }

    fn exact(
        query: &Query<Window>,
        ticks: &[i64],
        window: &Window,
        start: u64,
    ) -> Option<ExactDistance> {
        entry_for(query, ticks, window, start).1
    }

    fn compare(a: &ExactDistance, b: &ExactDistance, sharing: Sharing) -> Ordering {
        a.cmp_sharing(b, sharing)
    }

    fn hit(exact: &ExactDistance) -> Entry {
        exact.entry()
    }

    fn own(start: u64) -> Entry {
        let own = SampleIndex::new(start).expect("a window start is below NONE's sentinel");
        Entry::new(1.0, 0.0, own).expect("neither is NaN")
    }

    fn index(hit: &Entry) -> Option<u64> {
        hit.index().get()
    }

    fn dimension(exact: &ExactDistance) -> ExactDistance {
        *exact
    }

    fn caller_units(mean: f64, _scale: Scale) -> f64 {
        mean
    }

    fn distance(hit: &Entry, _scale: Scale) -> f64 {
        libm::sqrt(hit.squared_distance())
    }

    fn nearest(
        series: Series<'_>,
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
    ) -> Vec<Entry> {
        crate::batch::two_sided_profile(series, m, scale, threads)
            .into_iter()
            .map(|(_, sides)| sides.nearest)
            .collect()
    }

    fn ab_join(
        a: Series<'_>,
        b: Series<'_>,
        m: NonZeroUsize,
        scale_a: Scale,
        scale_b: Scale,
        threads: NonZeroUsize,
    ) -> Vec<Entry> {
        // The hot passes alone: a window they cannot rank has no exact distance here either.
        crate::batch::ab_join_hot(a, b, m, (scale_a, scale_b), threads)
            .into_iter()
            .map(|(_, entry)| entry)
            .collect()
    }

    #[inline]
    fn step(window: &Window, entering: i64, leaving: i64) -> TickWindow {
        TickWindow {
            entering,
            leaving,
            stats: *window,
        }
    }

    #[inline]
    fn scan_column(
        states: &[i64],
        rows: &[Window],
        column: &Window,
        m: usize,
        start: u64,
        mut pair: impl FnMut(usize, ExactDistance),
    ) -> Option<ExactDistance> {
        let step = |window: &Window| ZNorm::step(window, 0, 0);
        let column_step = step(column);
        let mut nearest = Nearest::new(column, m);
        for (at, (&state, row)) in states.iter().zip(rows).enumerate() {
            let n = centered_cross(state, &step(row), &column_step, m);
            if let Some(distance) = exact_from_cross(n, row, column, m, start) {
                pair(at, distance);
            }
            nearest.offer(
                n,
                row,
                SampleIndex::new(at as u64).expect("a row is a small index"),
            );
        }
        nearest.finish_exact()
    }
}

impl<R: Residue> Measure for NonNormalized<R> {
    type Window = bool;
    type Exact = NonNormNeighbor;
    type Hit = NonNormNeighbor;
    type Dimension = SquaredTicks;

    const NO_HIT: NonNormNeighbor = NonNormNeighbor::NONE;

    fn windows(
        _ticks: &[i64],
        present: &[bool],
        m: NonZeroUsize,
        _series: &Series<'_>,
    ) -> Vec<bool> {
        complete_windows(present, m.get())
    }

    fn quantize_query(query: Series<'_>, scale: Scale) -> Option<Query<bool>> {
        NonZeroUsize::new(query.len())?;
        let ticks: Vec<i64> = query
            .values()
            .iter()
            .map(|&x| to_ticks(x, scale))
            .collect::<Option<_>>()?;
        Some(Query {
            ticks,
            window: true,
        })
    }

    fn exact(
        query: &Query<bool>,
        ticks: &[i64],
        window: &bool,
        start: u64,
    ) -> Option<NonNormNeighbor> {
        if !query.window || !window {
            return None;
        }
        Some(NonNormNeighbor {
            squared_ticks: exact_squared(&query.ticks, ticks)?,
            index: SampleIndex::new(start).expect("a window start is below NONE's sentinel"),
        })
    }

    fn compare(a: &NonNormNeighbor, b: &NonNormNeighbor, _sharing: Sharing) -> Ordering {
        a.squared_ticks.cmp(&b.squared_ticks)
    }

    fn hit(exact: &NonNormNeighbor) -> NonNormNeighbor {
        *exact
    }

    fn own(start: u64) -> NonNormNeighbor {
        NonNormNeighbor {
            squared_ticks: 0,
            index: SampleIndex::new(start).expect("a window start is below NONE's sentinel"),
        }
    }

    fn index(hit: &NonNormNeighbor) -> Option<u64> {
        hit.index().get()
    }

    fn dimension(exact: &NonNormNeighbor) -> SquaredTicks {
        SquaredTicks(exact.squared_ticks())
    }

    fn caller_units(mean: f64, scale: Scale) -> f64 {
        mean * scale.get()
    }

    fn distance(hit: &NonNormNeighbor, scale: Scale) -> f64 {
        hit.euclidean(scale)
    }

    fn nearest(
        series: Series<'_>,
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
    ) -> Vec<NonNormNeighbor> {
        crate::batch::nonnorm::two_sided_profile(series.values(), m, scale, threads)
            .into_iter()
            .map(|(_, sides)| sides.nearest)
            .collect()
    }

    fn ab_join(
        a: Series<'_>,
        b: Series<'_>,
        m: NonZeroUsize,
        scale_a: Scale,
        scale_b: Scale,
        threads: NonZeroUsize,
    ) -> Vec<NonNormNeighbor> {
        debug_assert!(
            scale_a == scale_b,
            "non-normalized distances compare in one unit"
        );
        crate::batch::nonnorm::ab_join(a.values(), b.values(), m, scale_a, threads)
            .into_iter()
            .map(|(_, neighbor)| neighbor)
            .collect()
    }

    #[inline]
    fn step(_window: &bool, entering: i64, leaving: i64) -> TickStep {
        TickStep { entering, leaving }
    }

    #[inline]
    fn scan_column(
        states: &[R],
        rows: &[bool],
        column: &bool,
        _m: usize,
        start: u64,
        mut pair: impl FnMut(usize, NonNormNeighbor),
    ) -> Option<NonNormNeighbor> {
        if !*column {
            return None;
        }
        let index = SampleIndex::new(start).expect("a window start is below NONE's sentinel");
        let mut best: Option<(u128, usize)> = None;
        for (at, (&state, &row)) in states.iter().zip(rows).enumerate() {
            if !row {
                continue;
            }
            let squared = state.as_u128();
            pair(
                at,
                NonNormNeighbor {
                    squared_ticks: squared,
                    index,
                },
            );
            if best.is_none_or(|(held, _)| squared < held) {
                best = Some((squared, at));
            }
        }
        best.map(|(squared_ticks, at)| NonNormNeighbor {
            squared_ticks,
            index: SampleIndex::new(at as u64).expect("a row is a small index"),
        })
    }
}

/// `None` for an empty query or one with a non-finite sample.
fn quantize_query(query: Series<'_>, stream_scale: Scale) -> Option<Query<Window>> {
    let m = NonZeroUsize::new(query.len())?;
    // A non-finite sample has no tick: the collect below rejects the query.
    let scale = stream_scale.for_query(query.values())?;
    let ticks: Vec<i64> = query
        .values()
        .iter()
        .map(|&x| to_ticks(x, scale))
        .collect::<Option<_>>()?;
    let mut stats = WindowStats::new(m);
    let window = ticks.iter().filter_map(|&t| stats.push(t, true)).last()?;
    let window = query.window(0, window, m);
    Some(Query { ticks, window })
}

/// The entry for `query` against the window `ticks` (`window` its statistics, the row), reported at
/// `start`: `N` from the kernel's own cross term, the entry from [`Nearest`], with the exact
/// distance it was ranked by (`None` for a window with no entry).
fn entry_for(
    query: &Query<Window>,
    ticks: &[i64],
    window: &Window,
    start: u64,
) -> (Entry, Option<ExactDistance>) {
    let m = query.ticks.len();
    let q = TickWindow {
        entering: 0,
        leaving: 0,
        stats: query.window,
    };
    let w = TickWindow {
        entering: 0,
        leaving: 0,
        stats: *window,
    };
    let n = centered_cross(ZNorm::start(&query.ticks, &q, ticks, &w), &q, &w, m);
    // The newer window is the row, the query its neighbor.
    match exact_from_cross(n, window, &query.window, m, start) {
        Some(exact) => (exact.entry(), Some(exact)),
        None => (Entry::NONE, None),
    }
}

/// The distance of `row` to `neighbor` (the window starting at `start`) from their exact centered
/// cross term `n`, as [`Nearest`] reports a lone candidate; `None` for a window with no distance.
pub(crate) fn exact_from_cross(
    n: i64,
    row: &Window,
    neighbor: &Window,
    m: usize,
    start: u64,
) -> Option<ExactDistance> {
    let mut nearest = Nearest::new(row, m);
    nearest.offer(
        n,
        neighbor,
        SampleIndex::new(start).expect("a window start is below NONE's sentinel"),
    );
    nearest.finish_exact()
}

/// The exact distance entry between `query` and every window of `series` (`series[i..i + m]`
/// for each `i`, entry `i` carrying index `i`), `m` the query's length. `series` is quantized
/// at `scale_series`, the query at the scale [`Scale::for_query`] picks. A window with a gap, or too
/// volatile for the kernel, is [`Entry::NONE`]. `None` for an empty query or one with a
/// non-finite sample; an empty profile if `series` is shorter than the query.
pub fn distance_profile<'q, 's>(
    query: impl Into<Series<'q>>,
    series: impl Into<Series<'s>>,
    scale_series: Scale,
) -> Option<Vec<Entry>> {
    profile_in::<ZNorm>(query.into(), series.into(), scale_series)
}

/// [`distance_profile`] under the non-normalized metric: the exact squared Euclidean distance on
/// ticks between `query` and every window of `series`, both quantized at `scale`. A window with a
/// gap, or whose distance needs more than 128 bits, has none. `None` for an empty query or one
/// that cannot be quantized.
pub fn distance_profile_nonnorm(
    query: &[f64],
    series: &[f64],
    scale: Scale,
) -> Option<Vec<NonNormNeighbor>> {
    profile_in::<NonNormalized>(query.into(), series.into(), scale)
}

fn profile_in<M: Measure>(
    query: Series<'_>,
    series: Series<'_>,
    scale: Scale,
) -> Option<Vec<M::Hit>> {
    Some(profile_of::<M>(
        &M::quantize_query(query, scale)?,
        series,
        scale,
    ))
}

fn profile_of<M: Measure>(
    query: &Query<M::Window>,
    series: Series<'_>,
    scale: Scale,
) -> Vec<M::Hit> {
    let m = NonZeroUsize::new(query.ticks.len()).expect("a query is never empty");
    TickedSeries::<M>::new(series, m, scale)
        .profile(query)
        .iter()
        .map(|exact| exact.as_ref().map_or(M::NO_HIT, M::hit))
        .collect()
}

/// A series as the arithmetic sees it: every sample's tick (forward-filled over a gap) and every
/// window's values, computed once so many queries can be matched against it. `scale` is the one
/// its distances are reported in.
pub(crate) struct TickedSeries<M: Measure = ZNorm> {
    m: usize,
    scale: Scale,
    ticks: Vec<i64>,
    windows: Vec<M::Window>,
}

impl<M: Measure> TickedSeries<M> {
    pub(crate) fn new<'a>(
        series: impl Into<Series<'a>>,
        m: NonZeroUsize,
        scale_series: Scale,
    ) -> Self {
        let series: Series<'a> = series.into();
        let mut fill = ForwardFill::default();
        let mut ticks = Vec::with_capacity(series.len());
        let mut present = Vec::with_capacity(series.len());
        for &x in series.values() {
            let (tick, had_tick) = sanitized_tick(&mut fill, x, scale_series);
            ticks.push(tick);
            present.push(had_tick);
        }
        Self {
            m: m.get(),
            scale: scale_series,
            windows: M::windows(&ticks, &present, m, &series),
            ticks,
        }
    }

    /// Every sample's tick, forward-filled over a gap.
    pub(crate) fn ticks(&self) -> &[i64] {
        &self.ticks
    }

    /// The values of every window, in order.
    pub(crate) fn windows(&self) -> &[M::Window] {
        &self.windows
    }

    /// The query that is the series' own window at `start`, ticks and all.
    pub(crate) fn query_at(&self, start: usize) -> Query<M::Window> {
        Query {
            ticks: self.ticks[start..start + self.m].to_vec(),
            window: self.windows[start],
        }
    }

    fn profile(&self, query: &Query<M::Window>) -> Vec<Option<M::Exact>> {
        (0..self.windows.len())
            .map(|i| self.exact_at(query, i))
            .collect()
    }

    /// The exact distance from `query` to the window at `start`.
    pub(crate) fn exact_at(&self, query: &Query<M::Window>, start: usize) -> Option<M::Exact> {
        M::exact(
            query,
            &self.ticks[start..start + self.m],
            &self.windows[start],
            start as u64,
        )
    }

    /// The starts on each side of a match that no other match may take.
    pub(crate) fn zone(&self) -> usize {
        crate::exclusion::zone(self.m)
    }

    /// A mean of exact distances, as this series' caller counts it.
    pub(crate) fn mean_in_caller_units(&self, mean: f64) -> f64 {
        M::caller_units(mean, self.scale)
    }

    /// The distance in the caller's units of an exact distance between windows of this series.
    pub(crate) fn distance_of(&self, exact: &M::Exact) -> f64 {
        M::distance(&M::hit(exact), self.scale)
    }

    /// The exact distance between the windows at `row` and `neighbor`, ranked as the batch
    /// profile ranks the pair with `row` the row; `None` when either has no distance.
    pub(crate) fn exact_pair(&self, row: usize, neighbor: usize) -> Option<M::Exact> {
        self.exact_across(row, self, neighbor)
    }

    /// [`Self::exact_pair`] with the neighbor a window of another series. A z-normalized pair does
    /// not depend on either series' scale; a non-normalized one needs both at the same.
    pub(crate) fn exact_across(
        &self,
        row: usize,
        other: &Self,
        neighbor: usize,
    ) -> Option<M::Exact> {
        M::exact(
            &other.query_at(neighbor),
            &self.ticks[row..row + self.m],
            &self.windows[row],
            neighbor as u64,
        )
    }

    /// The exactly nearest window of this series to `other`'s window at `at` (ties to the
    /// smaller start), `None` when that window or every window here has no distance.
    pub(crate) fn nearest_to(&self, other: &Self, at: usize) -> Option<M::Exact> {
        let query = other.query_at(at);
        let mut best: Option<M::Exact> = None;
        for (start, window) in self.windows.iter().enumerate() {
            let exact = M::exact(
                &query,
                &self.ticks[start..start + self.m],
                window,
                start as u64,
            );
            if let Some(exact) = exact
                && best.is_none_or(|b| M::compare(&exact, &b, Sharing::Neighbor).is_lt())
            {
                best = Some(exact);
            }
        }
        best
    }

    /// The matches of the series' own window at `start`: that window first, at distance zero
    /// (as STUMPY's `query_idx` forces), then the rest as [`find_matches`] selects them.
    pub(crate) fn matches_of_window(
        &self,
        start: usize,
        max_distance: MaxDistance,
        max_matches: Option<usize>,
        atol: f64,
    ) -> Vec<(u64, M::Hit)> {
        let profile = self.profile(&self.query_at(start));
        let rules = MatchRules {
            atol,
            query_idx: Some(start),
        };
        self.select_matches(&profile, max_distance, max_matches, rules)
    }

    /// The greedy selection over a distance profile, as [`select_starts`] makes it, each start
    /// with its hit: the anchor's own match, or the window's.
    fn select_matches(
        &self,
        full: &[Option<M::Exact>],
        max_distance: MaxDistance,
        max_matches: Option<usize>,
        rules: MatchRules,
    ) -> Vec<(u64, M::Hit)> {
        let anchor = rules.query_idx;
        let selection = Selection {
            zone: self.zone(),
            rules,
            max_distance,
            max_matches,
        };
        select_starts(
            full,
            &selection,
            |exact| self.distance_of(exact),
            |a, b| M::compare(a, b, Sharing::Neighbor),
        )
        .into_iter()
        .map(|i| {
            let hit = if Some(i) == anchor {
                M::own(i as u64)
            } else {
                M::hit(full[i].as_ref().expect("a selected window has a distance"))
            };
            (i as u64, hit)
        })
        .collect()
    }
}

/// What the greedy selection of matches is told besides the distances.
#[derive(Clone, Copy)]
pub(crate) struct Selection {
    /// The starts on each side of a match that no other match may take.
    pub(crate) zone: usize,
    /// STUMPY's `atol`, added to the threshold, and the `query_idx` that is taken first.
    pub(crate) rules: MatchRules,
    pub(crate) max_distance: MaxDistance,
    pub(crate) max_matches: Option<usize>,
}

/// STUMPY's `atol` and `query_idx` for [`find_matches_with`] and [`find_matches_nonnorm_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MatchRules {
    /// Added to the threshold when a distance is compared with it, as STUMPY's `atol` is
    /// (its default is `1e-8`; here the default is zero).
    pub atol: f64,
    /// The start of the window of `series` the query was taken from: that window is taken first,
    /// at distance zero, and counts as zero in the default threshold. `None` for a query that is
    /// not a window of the series.
    pub query_idx: Option<usize>,
}

/// The starts of the matches in a distance profile: with an `anchor` (the query is the series'
/// own window there), that window is taken first at distance zero and counts as zero in the
/// default threshold; every other window within the threshold goes by `compare`, ties to the
/// smaller start, each blocking the `zone` starts on each side of it.
pub(crate) fn select_starts<E: Copy>(
    full: &[Option<E>],
    selection: &Selection,
    distance_of: impl Fn(&E) -> f64,
    compare: impl Fn(&E, &E) -> Ordering,
) -> Vec<usize> {
    let Selection {
        zone,
        rules: MatchRules {
            atol,
            query_idx: anchor,
        },
        max_distance,
        max_matches,
    } = *selection;
    let mut distances: Vec<f64> = full
        .iter()
        .map(|exact| exact.as_ref().map_or(f64::INFINITY, &distance_of))
        .collect();
    if let Some(anchor) = anchor {
        distances[anchor] = 0.0;
    }
    let threshold = match max_distance {
        MaxDistance::Default => default_threshold(&distances),
        MaxDistance::Value(v) => Some(v),
    };
    let mut starts = Vec::new();
    let Some(threshold) = threshold.map(|threshold| threshold + atol) else {
        return starts;
    };
    let limit = max_matches.unwrap_or(usize::MAX);
    let mut blocked = alloc::vec![false; full.len()];
    if let Some(anchor) = anchor.filter(|_| limit > 0 && 0.0 <= threshold) {
        starts.push(anchor);
        let (from, to) = (
            anchor.saturating_sub(zone),
            (anchor + zone + 1).min(full.len()),
        );
        blocked[from..to].fill(true);
    }
    let mut candidates: Vec<usize> = Vec::with_capacity(full.len());
    candidates.extend((0..full.len()).filter(|&i| Some(i) != anchor).filter(|&i| {
        full[i]
            .as_ref()
            .is_some_and(|e| distance_of(e) <= threshold)
    }));
    candidates.sort_by(|&a, &b| {
        let (ea, eb) = (
            full[a].as_ref().expect("a candidate has a distance"),
            full[b].as_ref().expect("a candidate has a distance"),
        );
        compare(ea, eb).then(a.cmp(&b))
    });
    for i in candidates {
        if starts.len() >= limit {
            break;
        }
        if blocked[i] {
            continue;
        }
        starts.push(i);
        let (from, to) = (i.saturating_sub(zone), (i + zone + 1).min(full.len()));
        blocked[from..to].fill(true);
    }
    starts
}

/// How far a window may be from the query to match.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxDistance {
    /// STUMPY's default: `max(mean(D) - 2 * std(D), min(D))` over the finite distances `D` of
    /// the distance profile, so the nearest window always qualifies. The sums run in index
    /// order with plain multiplies and adds (no fused multiply-add), the standard deviation is
    /// the population one (divided by the count), so the value is the same on
    /// every target. NumPy sums pairwise, so the last bits of this threshold can differ from
    /// STUMPY's; parity at the threshold boundary is tolerance-based and measured, not exact.
    Default,
    /// A window matches when its distance is at most this (never when it is NaN).
    Value(f64),
}

/// The default threshold over the finite distances, `None` when there are none.
pub(crate) fn default_threshold(distances: &[f64]) -> Option<f64> {
    let mut count = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut min = f64::INFINITY;
    for &d in distances.iter().filter(|d| d.is_finite()) {
        count += 1.0;
        sum += d;
        if d < min {
            min = d;
        }
    }
    if count == 0.0 {
        return None;
    }
    let mean = sum / count;
    let mut squares = 0.0_f64;
    for &d in distances.iter().filter(|d| d.is_finite()) {
        let delta = d - mean;
        squares += delta * delta;
    }
    let spread = 2.0 * libm::sqrt(squares / count);
    let low = mean - spread;
    Some(if low > min { low } else { min })
}

/// Every window of `series` within `max_distance` of `query`, as `(start, entry)`, best first
/// (nearest first under the exact order, ties to the smaller start), at most `max_matches` of them, and
/// pairwise at least `ceil(m/4) + 1` apart: the same greedy selection as STUMPY's `match`
/// (take the nearest remaining window, drop the `ceil(m/4)` starts on each side of it, repeat).
/// Each entry is the one [`distance_profile`] gives; a window with a gap, or too volatile for the
/// kernel, never matches. A window qualifies when its reported distance is at most the
/// threshold (compared as the f64 distance); the qualifying windows are ordered by their exact
/// correlations, not by the rounded distances, which can reverse equal or nearly equal exact
/// distances. STUMPY also adds `1e-8` to its threshold; this does not (see [`find_matches_with`]). `None` for an empty or non-finite query, or a series shorter than the query.
pub fn find_matches<'q, 's>(
    query: impl Into<Series<'q>>,
    series: impl Into<Series<'s>>,
    scale_series: Scale,
    max_distance: MaxDistance,
    max_matches: Option<usize>,
) -> Option<Vec<(u64, Entry)>> {
    find_matches_with(
        query,
        series,
        scale_series,
        max_distance,
        max_matches,
        MatchRules::default(),
    )
}

/// [`find_matches`] with STUMPY's `atol` and `query_idx` (see [`MatchRules`]). `None` also for a
/// `query_idx` that is not the start of a window of `series`.
pub fn find_matches_with<'q, 's>(
    query: impl Into<Series<'q>>,
    series: impl Into<Series<'s>>,
    scale_series: Scale,
    max_distance: MaxDistance,
    max_matches: Option<usize>,
    rules: MatchRules,
) -> Option<Vec<(u64, Entry)>> {
    matches_in::<ZNorm>(
        query.into(),
        series.into(),
        scale_series,
        max_distance,
        max_matches,
        rules,
    )
}

/// [`find_matches`] under the non-normalized metric, as STUMPY's `aamp_match`: the windows within
/// `max_distance` (Euclidean, in the caller's units) of `query`, nearest first by the exact
/// squared distance on ticks, ties to the smaller start, pairwise at least `ceil(m/4) + 1` apart.
/// `series` and `query` are both quantized at `scale`. A window with a gap, or whose distance
/// needs more than 128 bits, never matches. `None` for an empty query or one that cannot be
/// quantized, or a series shorter than the query.
pub fn find_matches_nonnorm(
    query: &[f64],
    series: &[f64],
    scale: Scale,
    max_distance: MaxDistance,
    max_matches: Option<usize>,
) -> Option<Vec<(u64, NonNormNeighbor)>> {
    find_matches_nonnorm_with(
        query,
        series,
        scale,
        max_distance,
        max_matches,
        MatchRules::default(),
    )
}

/// [`find_matches_nonnorm`] with STUMPY's `atol` and `query_idx` (see [`MatchRules`]).
pub fn find_matches_nonnorm_with(
    query: &[f64],
    series: &[f64],
    scale: Scale,
    max_distance: MaxDistance,
    max_matches: Option<usize>,
    rules: MatchRules,
) -> Option<Vec<(u64, NonNormNeighbor)>> {
    matches_in::<NonNormalized>(
        query.into(),
        series.into(),
        scale,
        max_distance,
        max_matches,
        rules,
    )
}

fn matches_in<M: Measure>(
    query: Series<'_>,
    series: Series<'_>,
    scale: Scale,
    max_distance: MaxDistance,
    max_matches: Option<usize>,
    rules: MatchRules,
) -> Option<Vec<(u64, M::Hit)>> {
    if series.len() < query.len() {
        return None;
    }
    let query = M::quantize_query(query, scale)?;
    let m = NonZeroUsize::new(query.ticks.len()).expect("a query is never empty");
    let ticked = TickedSeries::<M>::new(series, m, scale);
    let full = ticked.profile(&query);
    if rules.query_idx.is_some_and(|at| at >= full.len()) {
        return None;
    }
    Some(ticked.select_matches(&full, max_distance, max_matches, rules))
}

/// Which registered query: the order of registration, from `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryId(usize);

impl QueryId {
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Why a query was not registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegisterError {
    /// The query is not the library's `m` samples long.
    WrongLength,
    /// A sample is NaN or infinite.
    NonFinite,
    /// No scale quantizes the query within the kernel's range.
    Unquantizable,
}

/// A library of queries, all `m` long, each quantized at its own scale with its window
/// statistics computed once, at registration.
pub struct PatternLibrary {
    m: NonZeroUsize,
    scale: Scale,
    queries: Vec<Query<Window>>,
}

impl PatternLibrary {
    /// `scale` is the scale of the streams this library will match.
    pub fn new(m: NonZeroUsize, scale: Scale) -> Self {
        Self {
            m,
            scale,
            queries: Vec::new(),
        }
    }

    pub fn m(&self) -> NonZeroUsize {
        self.m
    }

    pub fn len(&self) -> usize {
        self.queries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// Adds `query`, which must be `m` finite samples.
    pub fn register(&mut self, query: &[f64]) -> Result<QueryId, RegisterError> {
        if query.len() != self.m.get() {
            return Err(RegisterError::WrongLength);
        }
        if !query.iter().all(|x| x.is_finite()) {
            return Err(RegisterError::NonFinite);
        }
        let quantized =
            quantize_query(query.into(), self.scale).ok_or(RegisterError::Unquantizable)?;
        self.queries.push(quantized);
        Ok(QueryId(self.queries.len() - 1))
    }

    /// [`distance_profile`] of the registered query `id` over `series` at the library's scale;
    /// `None` if `id` is not in this library.
    pub fn distance_profile(&self, id: QueryId, series: &[f64]) -> Option<Vec<Entry>> {
        let query = self.queries.get(id.0)?;
        Some(profile_of::<ZNorm>(query, series.into(), self.scale))
    }
}

/// A library's queries matched against a stream, one sample at a time: after each
/// sample, one entry per query for the window ending there, bit for bit what
/// [`distance_profile`] reports at that window, and a function of that window and the query
/// alone, so a later sample never changes an earlier entry. Each tick costs `O(m·L)` for `L`
/// queries and allocates nothing: the ring and the output are sized at construction.
pub struct PatternStream {
    scale: Scale,
    library: PatternLibrary,
    fill: ForwardFill<i64>,
    stats: WindowStats,
    ring: MirroredRing<i64>,
    seen: u64,
    entries: Vec<Entry>,
}

impl PatternStream {
    /// The stream is quantized at the library's scale.
    pub fn new(library: PatternLibrary) -> Self {
        let m = library.m();
        let entries = Vec::with_capacity(library.len());
        Self {
            scale: library.scale,
            library,
            fill: ForwardFill::default(),
            stats: WindowStats::new(m),
            ring: MirroredRing::new(m, 0),
            seen: 0,
            entries,
        }
    }

    /// Takes the next sample. The entries are for the window ending at it, in registration
    /// order, each carrying the window's first sample's index; empty until `m` samples have
    /// arrived. A window with a gap, or too volatile for the kernel, yields [`Entry::NONE`].
    pub fn push(&mut self, sample: f64) -> &[Entry] {
        let (tick, had_tick) = sanitized_tick(&mut self.fill, sample, self.scale);
        self.ring.push(tick);
        self.seen += 1;
        self.entries.clear();
        if let Some(window) = self.stats.push(tick, had_tick) {
            let start = self.seen - self.library.m().get() as u64;
            let ticks = self.ring.as_slice();
            self.entries.extend(
                self.library
                    .queries
                    .iter()
                    .map(|query| entry_for(query, ticks, &window, start).0),
            );
        }
        &self.entries
    }
}

#[cfg(test)]
mod tests;
