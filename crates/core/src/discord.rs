//! The top-k discords of a stream: the windows whose nearest earlier neighbor is farthest.
//! Fed a stream's `update_exact` output, the score of a window is its exact left-profile
//! distance, fixed once published.
//!
//! The list is the greedy pick over every score still in the pool: farthest first by the exact
//! distance (never by the rounded published one), ties to the smaller window index, skipping
//! any window within the exclusion zone `ceil(m/4)` of one already
//! chosen. [`DiscordTracker::new`] pools every score ever published, which is exact DAMP on a
//! [`GrowingStream`](crate::stream::GrowingStream); [`DiscordTracker::windowed`] pools only the
//! windows still inside the last `w` samples, matching a
//! [`BoundedStream`](crate::bounded::BoundedStream) of that width.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZeroUsize;

use crate::capacity::{UNBOUNDED_RESERVE, reserve_for};
use crate::index::SampleIndex;
use crate::nearest::ExactDistance;

/// One listed discord: a window and its squared distance to its nearest earlier neighbor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Discord {
    index: SampleIndex,
    squared_distance: f64,
}

impl Discord {
    /// The window's own index (its first sample).
    pub const fn index(self) -> SampleIndex {
        self.index
    }

    pub const fn squared_distance(self) -> f64 {
        self.squared_distance
    }
}

/// A pooled score, ordered best first: the exactly farther distance, then the smaller index.
#[derive(Clone, Copy, Debug)]
struct Key {
    score: ExactDistance,
    index: u64,
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then(self.index.cmp(&other.index))
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Key {}

/// The best `k` discords of the pooled scores, pairwise outside the exclusion zone.
///
/// Each update inserts one score and re-picks greedily from the best. A pick skips at most
/// `2e` windows, so the walk visits at most `k(2e+1)` pooled scores: `O(log n + k·e·log k)`
/// per update, `n` the pool size. A pooled window keeps its exact distance in a 160-byte key:
/// in the score order, and, when windowed, again in the age order (a growing tracker keeps one).
/// A growing tracker keeps only the best `(k-1)(2e+1)+1` keys; a windowed one at most `w`.
pub struct DiscordTracker {
    k: usize,
    exclusion: u64,
    /// How far behind the newest row a window stays pooled; `None` keeps everything.
    reach: Option<u64>,
    /// The most scores a growing pool keeps: scores never change and the order is total, so the
    /// worst picks of `k` zones of `2e+1` windows reach only the first `(k-1)(2e+1)+1` entries
    /// and every later one can never be visited.
    pool_cap: usize,
    by_score: BTreeSet<Key>,
    /// Windowed only: the pooled windows by age, to forget the oldest.
    by_index: BTreeMap<u64, Key>,
    /// The picked discords, best first; reserved at `k` and never grown on a windowed tracker (at
    /// most the windows it pools), reserved up to a fixed limit on a growing one.
    list: Vec<Discord>,
    /// The picked indices, ascending, to test a candidate against the zone by binary search.
    picked: Vec<u64>,
}

impl DiscordTracker {
    /// Lists exactly what pooling every score ever published would, which is exact DAMP on a
    /// growing stream, while keeping only the scores that can still be listed: memory is
    /// `O(k·m)`. The list reserves `k` entries up to a fixed limit and grows past it, as the
    /// stream has no size to bound it by.
    pub fn new(m: NonZeroUsize, k: usize) -> Self {
        Self::reserving(m, k, reserve_for(k, UNBOUNDED_RESERVE))
    }

    fn reserving(m: NonZeroUsize, k: usize, reserve: usize) -> Self {
        let exclusion = crate::exclusion::zone(m.get()) as u64;
        Self {
            k,
            exclusion,
            reach: None,
            pool_cap: k
                .saturating_sub(1)
                .saturating_mul(2 * exclusion as usize + 1)
                .saturating_add(1),
            by_score: BTreeSet::new(),
            by_index: BTreeMap::new(),
            list: Vec::with_capacity(reserve),
            picked: Vec::with_capacity(reserve),
        }
    }

    /// Pools only windows inside the last `w` samples, as a bounded stream of width `w`
    /// scores them; an older window leaves the pool for good. Memory is `O(w)`. `None` when
    /// `w` is less than `m`. The list reserves `min(k, w - m + 1)` entries in full, up to a
    /// fixed limit of 2^24 past which it grows during warm-up only.
    pub fn windowed(m: NonZeroUsize, k: usize, w: NonZeroUsize) -> Option<Self> {
        let windows = w.get().checked_sub(m.get())?.saturating_add(1);
        Some(Self {
            reach: Some((windows - 1) as u64),
            pool_cap: usize::MAX,
            ..Self::reserving(m, k, reserve_for(k, windows))
        })
    }

    /// The discords of the scores so far, best first. At most `k`.
    pub fn top(&self) -> &[Discord] {
        &self.list
    }

    /// Offers one `update_exact` output. A window with no neighbor (a gap, or the first of the
    /// stream) is never a discord, though its row still ages the pool. Rows must arrive in
    /// increasing order, as a stream emits them.
    pub fn update(&mut self, row: SampleIndex, score: Option<ExactDistance>) {
        self.pool(row, score);
        self.pick();
    }

    fn pool(&mut self, row: SampleIndex, score: Option<ExactDistance>) {
        let Some(index) = row.get() else { return };
        if let Some(reach) = self.reach {
            while let Some((&old, &key)) = self.by_index.first_key_value() {
                if old + reach >= index {
                    break;
                }
                self.by_index.remove(&old);
                self.by_score.remove(&key);
            }
        }
        if let Some(score) = score {
            let key = Key { score, index };
            self.by_score.insert(key);
            if self.reach.is_some() {
                self.by_index.insert(index, key);
            } else if self.by_score.len() > self.pool_cap {
                self.by_score.pop_last();
            }
        }
    }

    /// The discords of a whole set of scores, rows in increasing order: pools them all, then
    /// picks once. Exactly the list [`Self::update`] would hold after the last row.
    pub fn pick_all(
        m: NonZeroUsize,
        k: usize,
        scores: impl IntoIterator<Item = (SampleIndex, Option<ExactDistance>)>,
    ) -> Vec<Discord> {
        let mut tracker = Self::new(m, k);
        for (row, score) in scores {
            tracker.pool(row, score);
        }
        tracker.pick();
        tracker.list
    }

    fn pick(&mut self) {
        self.list.clear();
        self.picked.clear();
        for key in &self.by_score {
            if self.list.len() == self.k {
                break;
            }
            let after = self.picked.partition_point(|&p| p < key.index);
            let clashes =
                |p: Option<&u64>| p.is_some_and(|&p| key.index.abs_diff(p) <= self.exclusion);
            if clashes(after.checked_sub(1).and_then(|at| self.picked.get(at)))
                || clashes(self.picked.get(after))
            {
                continue;
            }
            self.picked.insert(after, key.index);
            self.list.push(Discord {
                // A pooled index came from a real row.
                index: SampleIndex::new(key.index).unwrap_or(SampleIndex::NONE),
                squared_distance: key.score.entry().squared_distance(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{Discord, DiscordTracker, Key};
    use crate::bounded::BoundedStream;
    use crate::entry::Entry;
    use crate::index::SampleIndex;
    use crate::nearest::exact_oracle::{Exact, exactly_nearer};
    use crate::nearest::{ExactCorrelation, ExactDistance, variance_product};
    use crate::quantize::{Scale, to_ticks};
    use crate::stream::GrowingStream;
    use core::cmp::Ordering;
    use core::num::NonZeroUsize;
    use proptest::prelude::*;
    use std::vec::Vec;

    const TICK: f64 = 0.01;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn scale() -> Scale {
        Scale::new(TICK).unwrap()
    }

    fn tracker(m: usize, k: usize, reach: Option<usize>) -> DiscordTracker {
        match reach {
            None => DiscordTracker::new(nz(m), k),
            Some(r) => DiscordTracker::windowed(nz(m), k, nz(m + r)).unwrap(),
        }
    }

    fn exact_distance(rho: f64, squared: f64, correlation: ExactCorrelation) -> ExactDistance {
        let entry = Entry::new(rho, squared, SampleIndex::new(0).unwrap()).unwrap();
        ExactDistance::from_parts(entry, correlation, 1)
    }

    fn valid(cross: u128, row_variance: u64, neighbor_variance: u64) -> ExactCorrelation {
        ExactCorrelation::Valid {
            sign: 1,
            cross_squared: cross,
            row_variance,
            neighbor_variance,
            denominator: variance_product(row_variance, neighbor_variance),
        }
    }

    /// A score `s` in `0..32` as an exactly representable correlation `(32 - s) / 32`.
    fn synthetic(score: f64) -> ExactDistance {
        const VARIANCE: u64 = 1 << 26;
        let n = ((32.0 - score) * (1u64 << 21) as f64) as i64;
        let magnitude = u128::from(n.unsigned_abs());
        let correlation = ExactCorrelation::Valid {
            sign: n.signum() as i8,
            cross_squared: magnitude * magnitude,
            row_variance: VARIANCE,
            neighbor_variance: VARIANCE,
            denominator: variance_product(VARIANCE, VARIANCE),
        };
        exact_distance(n as f64 / VARIANCE as f64, score, correlation)
    }

    fn offer(tracker: &mut DiscordTracker, row: u64, score: Option<f64>) {
        tracker.update(SampleIndex::new(row).unwrap(), score.map(synthetic));
    }

    fn listed(tracker: &DiscordTracker) -> Vec<(u64, f64)> {
        tracker
            .top()
            .iter()
            .map(|d: &Discord| (d.index().get().unwrap(), d.squared_distance()))
            .collect()
    }

    fn listed_rows(tracker: &DiscordTracker) -> Vec<u64> {
        tracker
            .top()
            .iter()
            .map(|d| d.index().get().unwrap())
            .collect()
    }

    /// Independent of the tracker: sort every pooled score, then walk it taking whatever does
    /// not clash with a taken one. `scored` is `(row, score)` in arrival order, `None` for a
    /// row with no neighbor; `reach` drops windows more than that far behind the last row.
    fn oracle(
        scored: &[(u64, Option<f64>)],
        m: usize,
        k: usize,
        reach: Option<u64>,
    ) -> Vec<(u64, f64)> {
        let zone = crate::exclusion::zone(m) as u64;
        let newest = scored.last().map_or(0, |s| s.0);
        let mut pool: Vec<(u64, f64)> = scored
            .iter()
            .filter(|s| reach.is_none_or(|r| s.0 + r >= newest))
            .filter_map(|&(row, score)| Some((row, score?)))
            .collect();
        pool.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut taken: Vec<(u64, f64)> = Vec::new();
        for p in pool {
            if taken.len() < k && taken.iter().all(|t| t.0.abs_diff(p.0) > zone) {
                taken.push(p);
            }
        }
        taken
    }

    fn exact_of(ticks: &[i64], m: usize, row: usize, neighbor: usize) -> Exact {
        let stats = |start: usize| {
            let t = &ticks[start..start + m];
            let sum: i128 = t.iter().map(|&x| i128::from(x)).sum();
            let squares: i128 = t.iter().map(|&x| i128::from(x) * i128::from(x)).sum();
            (sum, m as i128 * squares - sum * sum)
        };
        let ((sum_row, var_row), (sum_neighbor, var_neighbor)) = (stats(row), stats(neighbor));
        match (var_row == 0, var_neighbor == 0) {
            (true, true) => Exact {
                sign: 1,
                num: 1,
                den: 1,
            },
            (false, false) => {
                let dot: i128 = (0..m)
                    .map(|k| i128::from(ticks[row + k]) * i128::from(ticks[neighbor + k]))
                    .sum();
                let n = m as i128 * dot - sum_row * sum_neighbor;
                Exact {
                    sign: n.signum() as i8,
                    num: (n * n) as u128,
                    den: (var_row * var_neighbor) as u128,
                }
            }
            _ => Exact {
                sign: 1,
                num: 1,
                den: 4,
            },
        }
    }

    /// Each row's exact nearest earlier neighbor outside the exclusion zone (and, when `reach`
    /// bounds the search, within it), ties to the smaller index; `None` for a row with none.
    fn exact_nearest_per_row(ticks: &[i64], m: usize, reach: Option<usize>) -> Vec<Option<Exact>> {
        let zone = crate::exclusion::zone(m);
        (0..=ticks.len() - m)
            .map(|row| {
                let oldest = reach.map_or(0, |r| row.saturating_sub(r));
                let mut best: Option<Exact> = None;
                for neighbor in oldest..row.saturating_sub(zone) {
                    let candidate = exact_of(ticks, m, row, neighbor);
                    if best.is_none_or(|b| exactly_nearer(candidate, b)) {
                        best = Some(candidate);
                    }
                }
                best
            })
            .collect()
    }

    /// The greedy pick over the exact scores of rows `0..=newest` still within `reach`.
    fn exact_reference(
        exact: &[Option<Exact>],
        newest: usize,
        m: usize,
        k: usize,
        reach: Option<usize>,
    ) -> Vec<usize> {
        let zone = crate::exclusion::zone(m);
        let mut pool: Vec<(usize, Exact)> = (0..=newest)
            .filter(|row| reach.is_none_or(|r| row + r >= newest))
            .filter_map(|row| Some((row, exact[row]?)))
            .collect();
        pool.sort_by(|a, b| {
            if exactly_nearer(b.1, a.1) {
                Ordering::Less
            } else if exactly_nearer(a.1, b.1) {
                Ordering::Greater
            } else {
                a.0.cmp(&b.0)
            }
        });
        let mut taken: Vec<usize> = Vec::new();
        for (row, _) in pool {
            if taken.len() < k && taken.iter().all(|&t| t.abs_diff(row) > zone) {
                taken.push(row);
            }
        }
        taken
    }

    /// Feeds `series` through a stream and a tracker; after every row, the list must be the
    /// exact reference's rows with the stream's own published distances.
    fn assert_matches_exact_reference(series: &[f64], m: usize, k: usize, extra: Option<usize>) {
        let ticks: Vec<i64> = series
            .iter()
            .map(|&x| to_ticks(x, scale()).unwrap())
            .collect();
        let exact = exact_nearest_per_row(&ticks, m, extra);
        let mut growing = GrowingStream::new(nz(m), scale());
        let mut bounded = extra.map(|e| BoundedStream::new(nz(m), nz(m + e), scale()).unwrap());
        let mut tracker = tracker(m, k, extra);
        let mut published: Vec<f64> = Vec::new();
        for &x in series {
            let output = match &mut bounded {
                Some(stream) => stream.update_exact(x),
                None => growing.update_exact(x),
            };
            let Some((row, score)) = output else { continue };
            published.push(score.map_or(f64::NAN, |s| s.entry().squared_distance()));
            tracker.update(row, score);
            let newest = row.get().unwrap() as usize;
            let expected: Vec<(u64, u64)> = exact_reference(&exact, newest, m, k, extra)
                .into_iter()
                .map(|r| (r as u64, published[r].to_bits()))
                .collect();
            let actual: Vec<(u64, u64)> = tracker
                .top()
                .iter()
                .map(|d| (d.index().get().unwrap(), d.squared_distance().to_bits()))
                .collect();
            assert_eq!(actual, expected, "after row {newest}");
        }
    }

    /// A walk of small steps, so exactly equal distances and near-ties are common.
    fn small_walk() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-2i64..=2, 10..70).prop_map(|steps| {
            steps
                .into_iter()
                .scan(0i64, |level, step| {
                    *level += step;
                    Some(*level as f64 * TICK)
                })
                .collect()
        })
    }

    /// One shape repeated at several scales and offsets: each copy is a scaled copy of the
    /// others, so windows inside different copies are exactly equally far from their kin.
    fn scaled_copies() -> impl Strategy<Value = Vec<f64>> {
        (
            prop::collection::vec(-3i64..=3, 5..9),
            prop::collection::vec((1i64..=3, -4i64..=4), 2..5),
        )
            .prop_map(|(shape, copies)| {
                copies
                    .into_iter()
                    .flat_map(|(factor, offset)| {
                        shape
                            .iter()
                            .map(move |&t| (factor * t + offset) as f64 * TICK)
                            .collect::<Vec<_>>()
                    })
                    .collect()
            })
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn growing_stream_matches_exact_reference(
            series in prop_oneof![small_walk(), scaled_copies()],
            m in 2usize..8,
            k in 1usize..6,
        ) {
            prop_assume!(series.len() >= m);
            assert_matches_exact_reference(&series, m, k, None);
        }

        #[test]
        fn bounded_stream_matches_exact_reference(
            series in prop_oneof![small_walk(), scaled_copies()],
            m in 2usize..8,
            extra in 0usize..30,
            k in 1usize..6,
        ) {
            prop_assume!(series.len() >= m);
            assert_matches_exact_reference(&series, m, k, Some(extra));
        }

        /// Few distinct scores, so ties and exclusion clashes are the common case.
        #[test]
        fn tied_scores_match_oracle(
            steps in prop::collection::vec((1u64..4, prop::option::weighted(0.85, 0u8..4)), 1..60),
            m in 1usize..14,
            k in 1usize..6,
            reach in prop::option::of(0usize..12),
        ) {
            let mut tracker = tracker(m, k, reach);
            let (mut row, mut scored) = (0, Vec::new());
            for (gap, score) in steps {
                row += gap;
                let score = score.map(f64::from);
                scored.push((row, score));
                offer(&mut tracker, row, score);
                prop_assert_eq!(listed(&tracker), oracle(&scored, m, k, reach.map(|r| r as u64)));
            }
            let zone = crate::exclusion::zone(m) as u64;
            for pair in listed(&tracker).windows(2) {
                prop_assert!(pair[0].1 > pair[1].1 || (pair[0].1 == pair[1].1 && pair[0].0 < pair[1].0));
            }
            let rows = listed_rows(&tracker);
            for (a, &x) in rows.iter().enumerate() {
                for &y in &rows[a + 1..] {
                    prop_assert!(x.abs_diff(y) > zone);
                }
            }
        }

        #[test]
        fn same_input_same_list(series in small_walk(), m in 2usize..8, k in 1usize..7) {
            prop_assume!(series.len() >= m);
            let run = || {
                let mut stream = GrowingStream::new(nz(m), scale());
                let mut tracker = DiscordTracker::new(nz(m), k);
                for &x in &series {
                    if let Some((row, score)) = stream.update_exact(x) {
                        tracker.update(row, score);
                    }
                }
                listed(&tracker).iter().map(|&(i, s)| (i, s.to_bits())).collect::<Vec<_>>()
            };
            prop_assert_eq!(run(), run());
        }
    }

    /// Two windows whose distances are exactly equal but published a rounding apart, in either
    /// direction, list in index order.
    #[test]
    fn exactly_equal_distances_list_in_index_order() {
        // rho^2 = (4n^2) / (2V)^2 = n^2 / V^2: one window is a scaled copy of the other.
        let variance = 1u64 << 40;
        let n = u128::from(variance) - 3;
        let rho = n as f64 / variance as f64;
        let up = f64::from_bits(rho.to_bits() + 1);
        for (first_rho, second_rho) in [(rho, up), (up, rho)] {
            let squared = |rho: f64| 8.0 * (1.0 - rho);
            let first = exact_distance(
                first_rho,
                squared(first_rho),
                valid(n * n, variance, variance),
            );
            let second = exact_distance(
                second_rho,
                squared(second_rho),
                valid(4 * n * n, 2 * variance, 2 * variance),
            );
            for (early, late) in [(first, second), (second, first)] {
                let mut tracker = DiscordTracker::new(nz(4), 3);
                tracker.update(SampleIndex::new(30).unwrap(), Some(late));
                tracker.update(SampleIndex::new(10).unwrap(), Some(early));
                assert_eq!(listed_rows(&tracker), [10, 30]);
            }
        }
    }

    /// Two distances one tick of correlation apart, published in the opposite order by their
    /// own rounded values: the exact distance decides.
    #[test]
    fn a_near_tie_orders_by_the_exact_distance() {
        let variance = 1u64 << 40;
        // `farther` has correlation 1 - 1/V; `nearer` has 1 - 1/(V + 1): the larger variance
        // leaves less to lose.
        let farther_n = u128::from(variance) - 1;
        let nearer_variance = variance + 1;
        let nearer_n = u128::from(nearer_variance) - 1;
        let nearer_rho = nearer_n as f64 / nearer_variance as f64;
        let farther_rho = f64::from_bits(nearer_rho.to_bits() + 1);
        let farther = exact_distance(
            farther_rho,
            8.0 * (1.0 - farther_rho),
            valid(farther_n * farther_n, variance, variance),
        );
        let nearer = exact_distance(
            nearer_rho,
            8.0 * (1.0 - nearer_rho),
            valid(nearer_n * nearer_n, nearer_variance, nearer_variance),
        );
        assert!(farther.entry().squared_distance() < nearer.entry().squared_distance());
        let mut tracker = DiscordTracker::new(nz(4), 2);
        tracker.update(SampleIndex::new(10).unwrap(), Some(nearer));
        tracker.update(SampleIndex::new(30).unwrap(), Some(farther));
        assert_eq!(listed_rows(&tracker), [30, 10]);
    }

    #[test]
    fn a_pooled_window_costs_a_known_number_of_bytes() {
        assert_eq!(core::mem::size_of::<Key>(), 160);
    }

    /// A wave of period 20 with one spike at `at`.
    fn wave_with_spike(len: usize, at: usize) -> Vec<f64> {
        (0..len)
            .map(|t| {
                100.0
                    + 5.0 * libm::sin(t as f64 * core::f64::consts::TAU / 20.0)
                    + if t == at { 20.0 } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn a_planted_anomaly_is_found() {
        let (m, at) = (20, 200);
        let series = wave_with_spike(300, at);
        let mut growing = GrowingStream::new(nz(m), scale());
        let mut bounded = BoundedStream::new(nz(m), nz(120), scale()).unwrap();
        let mut a = DiscordTracker::new(nz(m), 3);
        let mut b = tracker(m, 3, Some(120 - m));
        // The first windows have almost no past to match, so any wave looks odd there.
        let warmed_up = |row: SampleIndex| row.get().unwrap() >= 2 * m as u64;
        for &x in &series {
            if let Some((row, score)) = growing.update_exact(x).filter(|&(r, _)| warmed_up(r)) {
                a.update(row, score);
            }
            if let Some((row, score)) = bounded.update_exact(x).filter(|&(r, _)| warmed_up(r)) {
                b.update(row, score);
            }
        }
        for tracker in [&a, &b] {
            let best = tracker.top()[0].index().get().unwrap() as usize;
            assert!((at + 1 - m..=at).contains(&best), "found window {best}");
        }
    }

    #[test]
    fn a_tie_goes_to_the_smaller_index() {
        let mut tracker = DiscordTracker::new(nz(8), 3); // zone 2
        offer(&mut tracker, 10, Some(5.0));
        offer(&mut tracker, 11, Some(5.0));
        assert_eq!(listed(&tracker), [(10, 5.0)]);
        // Full list: an equal score at a later index does not push out the earlier one.
        let mut tracker = DiscordTracker::new(nz(8), 1);
        offer(&mut tracker, 10, Some(5.0));
        offer(&mut tracker, 20, Some(5.0));
        assert_eq!(listed(&tracker), [(10, 5.0)]);
    }

    #[test]
    fn the_exclusion_zone_is_inclusive_of_its_edge() {
        let mut tracker = DiscordTracker::new(nz(8), 3); // zone 2
        offer(&mut tracker, 10, Some(5.0));
        offer(&mut tracker, 12, Some(4.0)); // exactly the zone away: excluded
        assert_eq!(listed(&tracker), [(10, 5.0)]);
        offer(&mut tracker, 13, Some(4.0)); // one past: kept
        assert_eq!(listed(&tracker), [(10, 5.0), (13, 4.0)]);
        offer(&mut tracker, 15, Some(6.0)); // beats and replaces 13
        assert_eq!(listed(&tracker), [(15, 6.0), (10, 5.0)]);
    }

    #[test]
    fn no_neighbor_is_never_a_discord() {
        let mut tracker = DiscordTracker::new(nz(8), 2);
        offer(&mut tracker, 3, None);
        tracker.update(SampleIndex::new(4).unwrap(), None);
        assert!(tracker.top().is_empty());
        offer(&mut tracker, 9, Some(0.0));
        offer(&mut tracker, 20, None);
        assert_eq!(listed(&tracker), [(9, 0.0)]);
    }

    #[test]
    fn a_dominated_window_returns_when_its_dominator_is_dominated() {
        let mut tracker = DiscordTracker::new(nz(8), 2); // zone 2
        offer(&mut tracker, 10, Some(10.0)); // A
        offer(&mut tracker, 12, Some(12.0)); // B, within the zone of A
        assert_eq!(listed(&tracker), [(12, 12.0)]);
        offer(&mut tracker, 14, Some(15.0)); // C, within the zone of B but not of A
        assert_eq!(listed(&tracker), [(14, 15.0), (10, 10.0)]);
    }

    #[test]
    fn a_window_leaves_the_pool_once_it_is_older_than_the_width() {
        let mut tracker = DiscordTracker::windowed(nz(4), 2, nz(6)).unwrap(); // reach 2
        offer(&mut tracker, 0, Some(9.0));
        offer(&mut tracker, 2, Some(1.0));
        assert_eq!(listed(&tracker), [(0, 9.0), (2, 1.0)]);
        offer(&mut tracker, 3, Some(1.0)); // 0 is gone; 2 wins the tie against 3
        assert_eq!(listed(&tracker), [(2, 1.0)]);
        offer(&mut tracker, 5, None); // a row with no neighbor still ages the pool
        assert_eq!(listed(&tracker), [(3, 1.0)]);
        assert!(DiscordTracker::windowed(nz(4), 2, nz(3)).is_none());
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// Dropping the pool's tail never changes the list, ties and rows without a neighbor
        /// included, and the pool stays within the bound the cap argues.
        #[test]
        fn a_capped_pool_lists_what_an_uncapped_one_does(
            steps in prop::collection::vec((1u64..4, prop::option::weighted(0.9, 0u8..6)), 1..120),
            exclusion in 1usize..25,
            k in 1usize..6,
        ) {
            let mut capped = DiscordTracker::new(nz(4 * exclusion), k);
            let mut uncapped = DiscordTracker::new(nz(4 * exclusion), k);
            uncapped.pool_cap = usize::MAX;
            let mut row = 0;
            for (gap, score) in steps {
                row += gap;
                let score = score.map(f64::from);
                offer(&mut capped, row, score);
                offer(&mut uncapped, row, score);
                prop_assert_eq!(listed(&capped), listed(&uncapped));
                prop_assert!(capped.by_score.len() <= (k - 1) * (2 * exclusion + 1) + 1);
            }
        }
    }

    /// The best picks sit in the middle of their zones, so the last list entry is exactly the
    /// last key the cap keeps: scores 5 > 4 > 3 > 2 at rows 5, 4, 6, 3 with zone 1 and k = 2.
    #[test]
    fn the_cap_keeps_the_deepest_key_a_pick_can_reach() {
        let mut capped = DiscordTracker::new(nz(4), 2);
        for (row, score) in [(3, 2.0), (4, 4.0), (5, 5.0), (6, 3.0)] {
            offer(&mut capped, row, Some(score));
        }
        assert_eq!(listed(&capped), [(5, 5.0), (3, 2.0)]);
        offer(&mut capped, 20, Some(0.0));
        assert_eq!(capped.by_score.len(), 4);
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// Picking once over every score lists what an update per row ends with.
        #[test]
        fn picking_all_at_once_lists_what_every_update_ends_with(
            steps in prop::collection::vec((1u64..4, prop::option::weighted(0.9, 0u8..6)), 1..80),
            exclusion in 1usize..12,
            k in 0usize..6,
        ) {
            let mut tracker = DiscordTracker::new(nz(4 * exclusion), k);
            let mut row = 0;
            let mut scores = Vec::new();
            for (gap, score) in steps {
                row += gap;
                offer(&mut tracker, row, score.map(f64::from));
                scores.push((SampleIndex::new(row).unwrap(), score.map(|s| synthetic(f64::from(s)))));
            }
            let all = DiscordTracker::pick_all(nz(4 * exclusion), k, scores);
            prop_assert_eq!(all.as_slice(), tracker.top());
        }
    }

    #[test]
    fn zero_slots_list_nothing() {
        let mut tracker = DiscordTracker::new(nz(8), 0);
        offer(&mut tracker, 9, Some(1.0));
        assert!(tracker.top().is_empty());
    }
}
