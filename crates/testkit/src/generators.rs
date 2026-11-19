//! Market-shaped proptest strategies for input series and window lengths.

use proptest::collection::{SizeRange, vec};
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

/// The seed every property and schedule-exploring run starts from, so a run is the same on every
/// machine and time.
pub const TEST_SEED: u64 = 0x4d50_5445_5354_0001;

/// Proptest's default configuration (its `PROPTEST_CASES` override included) with a fixed seed.
/// Setting `PROPTEST_RNG_SEED` explores another seed: proptest reads it, and it wins.
pub fn seeded_default() -> ProptestConfig {
    let mut config = ProptestConfig::default();
    // timing-ok: replaces proptest's random seed with the fixed one
    if config.rng_seed == RngSeed::Random {
        config.rng_seed = RngSeed::Fixed(TEST_SEED);
    }
    config
}

/// [`seeded_default`] with exactly `cases` cases.
pub fn seeded(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        ..seeded_default()
    }
}

pub const MARKET_OFFSET: f64 = 1e8;
pub const MARKET_TICK: f64 = 0.01;
pub const MAX_WALK_STEP: f64 = 1.0;
pub const MAX_TICK_STEP: i64 = 2;

/// A walk starting anywhere in `0..=max_offset`, often exactly at `max_offset`.
pub fn random_walk(len: impl Into<SizeRange>, max_offset: f64) -> impl Strategy<Value = Vec<f64>> {
    let start = prop_oneof![3 => 0.0..=max_offset, 1 => Just(max_offset)];
    let steps = vec(-MAX_WALK_STEP..=MAX_WALK_STEP, len);
    (start, steps).prop_map(|(start, steps)| {
        steps
            .into_iter()
            .scan(start, |level, step| {
                let value = *level;
                *level += step;
                Some(value)
            })
            .collect()
    })
}

/// A walk over integer multiples of `tick`, so equal levels are bitwise-equal values; it often
/// starts exactly at the tick nearest `max_offset`.
pub fn quantized(
    len: impl Into<SizeRange>,
    tick: f64,
    max_offset: f64,
) -> impl Strategy<Value = Vec<f64>> {
    let max_start_ticks = (max_offset / tick).round() as i64;
    let start_ticks = prop_oneof![3 => 0..=max_start_ticks, 1 => Just(max_start_ticks)];
    let steps = vec(-MAX_TICK_STEP..=MAX_TICK_STEP, len);
    (start_ticks, steps).prop_map(move |(start_ticks, steps)| {
        steps
            .into_iter()
            .scan(start_ticks, |ticks, step| {
                let value = *ticks as f64 * tick;
                *ticks += step;
                Some(value)
            })
            .collect()
    })
}

/// Inserts a maximal run of `m..=2m` equal values.
pub fn with_flat_runs(
    base: impl Strategy<Value = Vec<f64>>,
    m: usize,
) -> impl Strategy<Value = Vec<f64>> {
    (base, any::<Index>(), any::<Index>(), m..=2 * m).prop_map(
        |(mut series, at, level_source, run_len)| {
            insert_flat_run(&mut series, at, level_source, run_len);
            series
        },
    )
}

/// The run's level is the first base value from `level_source` on, cyclically, that differs
/// bitwise from both neighbours, so the inserted run is maximal at exactly `run_len`.
fn insert_flat_run(series: &mut Vec<f64>, at: Index, level_source: Index, run_len: usize) {
    let at = at.index(series.len() + 1);
    let neighbours = [at.checked_sub(1), Some(at)].map(|i| i.and_then(|i| series.get(i)).copied());
    let differs_from_neighbours = |level: &f64| {
        neighbours
            .iter()
            .flatten()
            .all(|n| n.to_bits() != level.to_bits())
    };
    let off_grid_fallback = std::iter::successors(Some(0.0_f64), |level| Some(level.next_up()));
    let level = series
        .iter()
        .cycle()
        .skip(level_source.index(series.len().max(1)))
        .take(series.len())
        .copied()
        .chain(off_grid_fallback)
        .find(differs_from_neighbours)
        .expect("two neighbours exclude at most two of infinitely many distinct levels");
    series.splice(at..at, std::iter::repeat_n(level, run_len));
}

/// Inserts one to three runs of one to three NaN, +inf or −inf values.
pub fn with_gaps(base: impl Strategy<Value = Vec<f64>>) -> impl Strategy<Value = Vec<f64>> {
    let gap_value = prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)];
    let gaps = vec((any::<Index>(), gap_value, 1usize..=3), 1..=3);
    (base, gaps).prop_map(|(mut series, gaps)| {
        for (at, value, gap_len) in gaps {
            let at = at.index(series.len() + 1);
            series.splice(at..at, std::iter::repeat_n(value, gap_len));
        }
        series
    })
}

/// Inserts two non-overlapping copies of one `m..=2m` segment read cyclically from the base.
pub fn with_repeats(
    base: impl Strategy<Value = Vec<f64>>,
    m: usize,
) -> impl Strategy<Value = Vec<f64>> {
    (
        base,
        any::<Index>(),
        [any::<Index>(), any::<Index>()],
        m..=2 * m,
    )
        .prop_map(|(series, source, targets, segment_len)| {
            let source_values = if series.is_empty() {
                vec![0.0]
            } else {
                series.clone()
            };
            let segment: Vec<f64> = source_values
                .iter()
                .cycle()
                .skip(source.index(source_values.len()))
                .take(segment_len)
                .copied()
                .collect();
            let mut targets = targets.map(|target| target.index(series.len() + 1));
            targets.sort_unstable();
            let [first, second] = targets;
            let mut repeated = Vec::with_capacity(series.len() + 2 * segment_len);
            repeated.extend_from_slice(&series[..first]);
            repeated.extend_from_slice(&segment);
            repeated.extend_from_slice(&series[first..second]);
            repeated.extend_from_slice(&segment);
            repeated.extend_from_slice(&series[second..]);
            repeated
        })
}

/// Any of the walks above, plain or with flat runs, gaps or repeats at window length `m`.
pub fn market_series(len: impl Into<SizeRange>, m: usize) -> impl Strategy<Value = Vec<f64>> {
    let level = market_level(len);
    prop_oneof![
        level.clone(),
        with_flat_runs(level.clone(), m),
        with_gaps(level.clone()),
        with_repeats(level, m),
    ]
}

fn market_level(len: impl Into<SizeRange>) -> BoxedStrategy<Vec<f64>> {
    let len = len.into();
    prop_oneof![
        random_walk(len.clone(), MARKET_OFFSET),
        quantized(len, MARKET_TICK, MARKET_OFFSET),
    ]
    .boxed()
}

/// A market series and a window length `m` its features hold at: `m` in `2..=4` with any
/// feature, or `m` in `n − 2..=n` with gaps or a flat run (repeats need `n ≥ 2m`).
pub fn market_series_and_m(len: impl Into<SizeRange>) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let len = len.into();
    prop_oneof![
        tiny_m_series(len.clone(), market_series),
        near_n_series(len.clone()),
        near_n_flat_series(len),
    ]
}

fn at_least(len: &SizeRange, min: usize) -> SizeRange {
    let (start, end) = len.start_end_incl();
    SizeRange::new(start.max(min)..=end.max(min))
}

fn tiny_m_series<S: Strategy<Value = Vec<f64>>>(
    len: SizeRange,
    series_at: impl Fn(SizeRange, usize) -> S,
) -> impl Strategy<Value = (Vec<f64>, usize)> {
    (2usize..=4)
        .prop_flat_map(move |m| series_at(at_least(&len, m), m).prop_map(move |series| (series, m)))
}

fn near_n_series(len: SizeRange) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let level = market_level(at_least(&len, 2));
    let series = prop_oneof![level.clone(), with_gaps(level)];
    (series, 0usize..=2).prop_map(|(series, k)| {
        let m = (series.len() - k).max(2);
        (series, m)
    })
}

fn near_n_flat_series(len: SizeRange) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let (start, end) = at_least(&len, 2).start_end_incl();
    (
        market_level(0..=2),
        any::<Index>(),
        any::<Index>(),
        start..=end,
    )
        .prop_map(|(mut series, at, level_source, m)| {
            insert_flat_run(&mut series, at, level_source, m);
            (series, m)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    const M: usize = 3;

    fn maximal_run_lengths(series: &[f64]) -> Vec<usize> {
        series
            .chunk_by(|a, b| a.to_bits() == b.to_bits())
            .map(<[f64]>::len)
            .collect()
    }

    fn draws<T: Strategy>(strategy: &T) -> Vec<T::Value> {
        let mut runner = TestRunner::deterministic();
        (0..256)
            .map(|_| strategy.new_tree(&mut runner).unwrap().current())
            .collect()
    }

    fn has_disjoint_bitwise_repeat(series: &[f64], len: usize) -> bool {
        let bits: Vec<u64> = series.iter().map(|x| x.to_bits()).collect();
        let windows: Vec<&[u64]> = bits.windows(len).collect();
        windows
            .iter()
            .enumerate()
            .any(|(i, earlier)| windows.iter().skip(i + len).any(|later| earlier == later))
    }

    proptest! {
        #![proptest_config(super::seeded_default())]
        #[test]
        fn walk_starts_within_offset_and_steps_by_at_most_one(
            walk in random_walk(1..64, MARKET_OFFSET),
        ) {
            prop_assert!((0.0..=MARKET_OFFSET).contains(&walk[0]));
            for pair in walk.windows(2) {
                let rounding = 2.0 * f64::EPSILON * pair[0].abs().max(pair[1].abs());
                prop_assert!((pair[1] - pair[0]).abs() <= MAX_WALK_STEP + rounding);
            }
        }

        #[test]
        fn quantized_values_are_on_grid(series in quantized(0..64, MARKET_TICK, MARKET_OFFSET)) {
            for value in series {
                prop_assert_eq!((value / MARKET_TICK).round() * MARKET_TICK, value);
            }
        }

        #[test]
        fn flat_runs_have_length_at_least_m(
            series in with_flat_runs(random_walk(0..64, MARKET_OFFSET), M),
        ) {
            prop_assert!(maximal_run_lengths(&series).iter().any(|len| (M..=2 * M).contains(len)));
        }

        #[test]
        fn gaps_contain_nonfinite(series in with_gaps(random_walk(0..64, MARKET_OFFSET))) {
            prop_assert!(series.iter().any(|x| !x.is_finite()));
        }

        #[test]
        fn repeats_are_bitwise_copies(series in with_repeats(random_walk(0..64, MARKET_OFFSET), M)) {
            prop_assert!(has_disjoint_bitwise_repeat(&series, M));
        }

        #[test]
        fn m_within_bounds((series, m) in market_series_and_m(2..64)) {
            prop_assert!((2..=series.len()).contains(&m));
        }

        #[test]
        fn tiny_m_features_hold_at_m(
            (series, m) in tiny_m_series((0..64).into(), |len, m| {
                with_flat_runs(random_walk(len, MARKET_OFFSET), m)
            }),
        ) {
            prop_assert!(maximal_run_lengths(&series).iter().any(|len| (m..=2 * m).contains(len)));
        }

        #[test]
        fn near_n_flat_runs_have_length_at_least_m((series, m) in near_n_flat_series((2..64).into())) {
            prop_assert!(series.len() - m <= 2);
            prop_assert!(maximal_run_lengths(&series).iter().any(|&len| len >= m));
        }
    }

    #[test]
    fn offset_reaches_requested_level() {
        let walks = draws(&random_walk(1..8, MARKET_OFFSET));
        let quantized_walks = draws(&quantized(1..8, MARKET_TICK, MARKET_OFFSET));
        assert!(walks.iter().any(|walk| walk[0] == MARKET_OFFSET));
        assert!(quantized_walks.iter().any(|walk| walk[0] == MARKET_OFFSET));
    }

    #[test]
    fn quantized_start_rounds_to_nearest_tick() {
        let tick = 0.03;
        let walks = draws(&quantized(1..8, tick, 0.05));
        assert!(walks.iter().any(|walk| walk[0] == 2.0 * tick));
    }

    #[test]
    fn flat_run_of_exactly_m_is_drawn() {
        let series = draws(&with_flat_runs(random_walk(1..64, MARKET_OFFSET), M));
        assert!(
            series
                .iter()
                .any(|series| maximal_run_lengths(series).contains(&M))
        );
    }

    #[test]
    fn gaps_cover_nan_and_both_infinities() {
        let values: Vec<f64> = draws(&with_gaps(random_walk(0..64, MARKET_OFFSET)))
            .into_iter()
            .flatten()
            .collect();
        assert!(values.iter().any(|x| x.is_nan()));
        assert!(values.contains(&f64::INFINITY));
        assert!(values.contains(&f64::NEG_INFINITY));
    }

    #[test]
    fn m_covers_tiny_and_near_n() {
        let pairs = draws(&market_series_and_m(16..=16));
        for m in [2, 3, 4] {
            assert!(
                pairs.iter().any(|(_, drawn)| *drawn == m),
                "m = {m} never drawn"
            );
        }
        for k in [0, 1, 2] {
            assert!(
                pairs.iter().any(|(series, m)| series.len() - m == k),
                "m = n - {k} never drawn"
            );
        }
    }
}
