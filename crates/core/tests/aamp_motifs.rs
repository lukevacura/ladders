#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Non-normalized matching and motifs (STUMPY's `aamp_match` and `aamp_motifs`) equal a plain,
//! independent statement of the procedure on exact integer ticks: on random series with planted
//! exact repeats, series of ties, NaN gaps and the edges of the window count, at every thread
//! count; the z-normalized and non-normalized dispatches see different distances.

use core::num::NonZeroUsize;

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/nonnorm_derived.rs"]
mod reference;

use ladde::motif::{MotifParams, motifs, motifs_nonnorm};
use ladde::nonnorm::NonNormNeighbor;
use ladde::pattern::{MaxDistance, distance_profile_nonnorm, find_matches, find_matches_nonnorm};
use ladde::quantize::Scale;
use nonnorm::{series as walk_series, tie_series};
use proptest::prelude::*;
use reference::{
    Group, brute_matches, brute_motifs, distance, profile, squared, ticks, window, windows,
};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn group(matches: &[(u64, NonNormNeighbor)], scale: Scale) -> Group {
    matches
        .iter()
        .map(|&(start, hit)| {
            assert_eq!(hit.index().get(), Some(start));
            assert_eq!(
                hit.euclidean(scale).to_bits(),
                distance(hit.squared_ticks(), scale).to_bits()
            );
            (start, hit.squared_ticks())
        })
        .collect()
}

fn library_motifs(
    series: &[f64],
    m: usize,
    scale: Scale,
    threads: usize,
    params: &MotifParams,
) -> Vec<Group> {
    motifs_nonnorm(series, nz(m), scale, nz(threads), params)
        .iter()
        .map(|g| group(g, scale))
        .collect()
}

/// Representatives are outside every earlier group's zones, and each group is outside its own.
fn assert_zones(groups: &[Group], m: usize) {
    let zone = ladde::exclusion::zone(m) as u64;
    for (g, found) in groups.iter().enumerate() {
        for (k, a) in found.iter().enumerate() {
            for b in &found[k + 1..] {
                assert!(a.0.abs_diff(b.0) > zone, "inside one group: {found:?}");
            }
        }
        for earlier in &groups[..g] {
            for b in earlier {
                assert!(found[0].0.abs_diff(b.0) > zone, "{groups:?}");
            }
        }
    }
}

fn max_distances(reach: f64) -> [MaxDistance; 5] {
    [
        MaxDistance::Default,
        MaxDistance::Value(0.0),
        MaxDistance::Value(0.4 * reach),
        MaxDistance::Value(reach),
        MaxDistance::Value(f64::INFINITY),
    ]
}

fn motif_grid(reach: f64) -> Vec<MotifParams> {
    let mut grid = Vec::new();
    for (min_neighbors, max_matches, max_motifs) in [
        (0, Some(10), 1),
        (1, Some(10), 3),
        (1, None, 100),
        (2, Some(3), 4),
        (0, Some(1), 100),
        (1, Some(0), 2),
    ] {
        for max_distance in max_distances(reach) {
            for cutoff in [None, Some(0.0), Some(0.6 * reach), Some(f64::INFINITY)] {
                grid.push(MotifParams {
                    min_neighbors,
                    max_distance,
                    cutoff,
                    max_matches,
                    max_motifs,
                    ..MotifParams::default()
                });
            }
        }
    }
    grid
}

/// A distance scale to size thresholds by: the span of the series in its own units.
fn reach_of(series: &[f64], m: usize) -> f64 {
    let finite = series.iter().copied().filter(|x| x.is_finite());
    let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
        (lo.min(x), hi.max(x))
    });
    if hi < lo {
        return 1.0;
    }
    libm::sqrt(m as f64) * (hi - lo) * 0.5
}

fn assert_motifs_equal_brute(series: &[f64], m: usize, scale: Scale) -> usize {
    let mut nonempty = 0;
    for params in motif_grid(reach_of(series, m)) {
        let want = brute_motifs(series, m, scale, &params);
        let got = library_motifs(series, m, scale, 1, &params);
        assert_eq!(got, want, "{params:?}");
        assert_zones(&got, m);
        nonempty += usize::from(!got.is_empty());
        for threads in [2, 7] {
            assert_eq!(
                library_motifs(series, m, scale, threads, &params),
                got,
                "threads {threads} {params:?}"
            );
        }
    }
    nonempty
}

/// Every query worth asking of `series`: each of its windows' starts (a sample), a window
/// nudged off the grid, and a query that is not in the series at all.
fn queries(series: &[f64], m: usize, scale: Scale) -> Vec<Vec<f64>> {
    let t = ticks(series, scale);
    let mut out = Vec::new();
    for start in (0..windows(&t, m)).step_by(3) {
        if let Some(w) = window(&t, start, m) {
            let on_grid: Vec<f64> = w.iter().map(|&x| x as f64 * scale.get()).collect();
            let mut nudged = on_grid.clone();
            nudged[start % m] += 0.6 * scale.get();
            out.push(on_grid);
            out.push(nudged);
        }
    }
    out.push((0..m).map(|k| (k * k % 5) as f64 * scale.get()).collect());
    out
}

fn assert_matches_equal_brute(series: &[f64], m: usize, scale: Scale) -> usize {
    let t = ticks(series, scale);
    let mut nonempty = 0;
    let reach = reach_of(series, m);
    for query in queries(series, m, scale) {
        let q: Vec<i64> = query
            .iter()
            .map(|&x| ladde::quantize::to_ticks(x, scale).unwrap())
            .collect();
        let exact = profile(&q, &t);
        let hits = distance_profile_nonnorm(&query, series, scale).unwrap();
        assert_eq!(hits.len(), exact.len());
        for (i, (hit, want)) in hits.iter().zip(&exact).enumerate() {
            match want {
                Some(sq) => {
                    assert_eq!(
                        (hit.squared_ticks(), hit.index().get()),
                        (*sq, Some(i as u64))
                    );
                }
                None => assert_eq!(hit.index().get(), None, "window {i}"),
            }
        }
        if series.len() < query.len() {
            let refused = find_matches_nonnorm(&query, series, scale, MaxDistance::Default, None);
            assert!(refused.is_none());
            continue;
        }
        let mut thresholds = max_distances(reach).to_vec();
        // A threshold exactly at a window's distance includes it, and one just below does not.
        if let Some(Some(sq)) = exact.iter().find(|e| e.is_some_and(|s| s > 0)) {
            let at = distance(*sq, scale);
            thresholds.push(MaxDistance::Value(at));
            thresholds.push(MaxDistance::Value(f64::from_bits(at.to_bits() - 1)));
        }
        for max_distance in thresholds {
            for limit in [None, Some(0), Some(1), Some(2), Some(5)] {
                let want = brute_matches(&exact, m, scale, max_distance, limit, None);
                let got = find_matches_nonnorm(&query, series, scale, max_distance, limit).unwrap();
                assert_eq!(group(&got, scale), want, "{max_distance:?} {limit:?}");
                nonempty += usize::from(!got.is_empty());
            }
        }
    }
    nonempty
}

#[derive(Debug, Clone)]
struct Plant {
    from: prop::sample::Index,
    to: prop::sample::Index,
    shift: i32,
}

/// Small integers with exact copies planted (some off by one in one sample), and NaN gaps.
fn planted() -> impl Strategy<Value = (Vec<f64>, usize)> {
    (
        proptest::collection::vec(-12i32..12, 24..64),
        proptest::collection::vec(
            (
                any::<prop::sample::Index>(),
                any::<prop::sample::Index>(),
                0i32..4,
            )
                .prop_map(|(from, to, shift)| Plant { from, to, shift }),
            0..4,
        ),
        proptest::collection::vec(any::<prop::sample::Index>(), 0..3),
        4usize..9,
    )
        .prop_map(|(base, plants, gaps, m)| {
            let mut series: Vec<f64> = base.iter().map(|&x| f64::from(x)).collect();
            let n = series.len();
            for plant in plants {
                let (from, to) = (plant.from.index(n - m), plant.to.index(n - m));
                for k in 0..m {
                    series[to + k] = f64::from(base[from + k]);
                }
                series[to + m / 2] += f64::from(plant.shift.min(1));
            }
            for gap in gaps {
                series[gap.index(n)] = f64::NAN;
            }
            (series, m)
        })
}

#[test]
fn motifs_equal_the_brute_force_procedure() {
    let (cases, nonempty) = (core::cell::Cell::new(0usize), core::cell::Cell::new(0usize));
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(24))
        .run(&planted(), |(series, m)| {
            nonempty.set(nonempty.get() + assert_motifs_equal_brute(&series, m, scale(1.0)));
            cases.set(cases.get() + 1);
            Ok(())
        })
        .unwrap();
    assert!(nonempty.get() >= cases.get() * 20, "{}", nonempty.get());
}

#[test]
fn motifs_on_a_finer_scale_equal_the_brute_force_procedure() {
    let nonempty = core::cell::Cell::new(0usize);
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(12))
        .run(&planted(), |(series, m)| {
            let quarter: Vec<f64> = series.iter().map(|x| x * 0.25).collect();
            nonempty.set(nonempty.get() + assert_motifs_equal_brute(&quarter, m, scale(0.25)));
            Ok(())
        })
        .unwrap();
    assert!(nonempty.get() > 100);
}

#[test]
fn matches_equal_the_brute_force_procedure() {
    let nonempty = core::cell::Cell::new(0usize);
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(24))
        .run(&planted(), |(series, m)| {
            nonempty.set(nonempty.get() + assert_matches_equal_brute(&series, m, scale(1.0)));
            Ok(())
        })
        .unwrap();
    assert!(nonempty.get() > 500);
}

#[test]
fn matches_on_a_finer_scale_equal_the_brute_force_procedure() {
    let nonempty = core::cell::Cell::new(0usize);
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(12))
        .run(&planted(), |(series, m)| {
            let quarter: Vec<f64> = series.iter().map(|x| x * 0.25).collect();
            nonempty.set(nonempty.get() + assert_matches_equal_brute(&quarter, m, scale(0.25)));
            Ok(())
        })
        .unwrap();
    assert!(nonempty.get() > 200);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    /// Series of exact ties (constants, flat runs between levels, short periods, few levels):
    /// every order is a tie that only the start decides.
    #[test]
    fn tie_series_equal_the_brute_force_procedure(series in tie_series(), m in 1usize..=6) {
        assert_motifs_equal_brute(&series, m, scale(1.0));
        assert_matches_equal_brute(&series, m, scale(1.0));
    }

    /// Walks that jump between levels around the `u64` bound: pairs whose squared distance
    /// needs the wide arithmetic.
    #[test]
    fn jumping_series_equal_the_brute_force_procedure(series in walk_series(), m in 1usize..=6) {
        assert_motifs_equal_brute(&series, m, scale(1.0));
        assert_matches_equal_brute(&series, m, scale(1.0));
    }
}

#[test]
fn a_constant_series_matches_by_start() {
    let series = [7.0; 40];
    let m = 6;
    let got = find_matches_nonnorm(
        &series[..m],
        &series,
        scale(1.0),
        MaxDistance::Default,
        None,
    )
    .unwrap();
    let starts: Vec<u64> = got.iter().map(|&(s, _)| s).collect();
    assert_eq!(starts, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]);
    assert!(got.iter().all(|&(_, h)| h.squared_ticks() == 0));
    let params = MotifParams {
        max_motifs: 3,
        max_matches: None,
        ..MotifParams::default()
    };
    let found = library_motifs(&series, m, scale(1.0), 1, &params);
    assert_eq!(found, brute_motifs(&series, m, scale(1.0), &params));
    assert_eq!(found[0][0].0, 0);
}

#[test]
fn exact_repeats_are_ordered_by_start_not_by_rounded_distance() {
    let shape = [3.0, -7.0, 5.0, 1.0, -2.0, 8.0, -4.0];
    let mut series: Vec<f64> = Vec::new();
    for offset in [0.0, 24.0, 5.0, -11.0, 3.0, 24.0, 0.0] {
        series.extend(shape.iter().map(|x| x + offset));
        series.extend([1.0, 4.0, 1.0]);
    }
    for max_distance in [MaxDistance::Default, MaxDistance::Value(f64::INFINITY)] {
        let params = MotifParams {
            max_motifs: 3,
            max_matches: None,
            max_distance,
            cutoff: Some(f64::INFINITY),
            ..MotifParams::default()
        };
        let got = library_motifs(&series, 7, scale(1.0), 1, &params);
        assert_eq!(got, brute_motifs(&series, 7, scale(1.0), &params));
        assert_zones(&got, 7);
    }
}

#[test]
fn nan_gaps_at_both_ends_and_inside_equal_the_brute_force_procedure() {
    let mut series: Vec<f64> = (0..70).map(|k| f64::from((k * 7 % 11) as u8)).collect();
    series[..3].fill(f64::NAN);
    series[8..24].fill(5.0);
    series[33] = f64::NAN;
    series[50..53].fill(f64::NAN);
    series[67..].fill(f64::NAN);
    assert!(assert_motifs_equal_brute(&series, 5, scale(1.0)) > 0);
    assert!(assert_matches_equal_brute(&series, 5, scale(1.0)) > 0);
    let all_gap = [f64::NAN; 20];
    assert!(library_motifs(&all_gap, 4, scale(1.0), 1, &MotifParams::default()).is_empty());
    let hits = distance_profile_nonnorm(&[1.0, 2.0, 3.0], &all_gap, scale(1.0)).unwrap();
    assert!(hits.iter().all(|h| h.index().get().is_none()));
    let none = find_matches_nonnorm(
        &[1.0, 2.0, 3.0],
        &all_gap,
        scale(1.0),
        MaxDistance::Default,
        None,
    );
    assert_eq!(none.unwrap().len(), 0);
}

#[test]
fn the_edges_of_the_window_count_have_the_documented_answers() {
    let unit = scale(1.0);
    let series: Vec<f64> = (0..12).map(|k| f64::from(k % 5)).collect();
    // A series shorter than the window has no motifs and no profile.
    assert!(library_motifs(&series[..3], 4, unit, 1, &MotifParams::default()).is_empty());
    assert!(library_motifs(&[], 4, unit, 1, &MotifParams::default()).is_empty());
    // One window has no neighbor, so no motif; but it matches itself.
    assert!(library_motifs(&series[..4], 4, unit, 1, &MotifParams::default()).is_empty());
    let one = find_matches_nonnorm(&series[..4], &series[..4], unit, MaxDistance::Default, None);
    assert_eq!(group(&one.unwrap(), unit), [(0, 0)]);
    // A query longer than the series, an empty one, and a non-finite one are refused.
    let long = &series[..8];
    for query in [long, &[][..], &[1.0, f64::NAN][..], &[f64::INFINITY][..]] {
        let refused = find_matches_nonnorm(query, &series[..5], unit, MaxDistance::Default, None);
        assert!(refused.is_none(), "{query:?}");
    }
    for query in [&[][..], &[1.0, f64::NAN][..], &[f64::INFINITY][..]] {
        assert!(
            distance_profile_nonnorm(query, &series[..5], unit).is_none(),
            "{query:?}"
        );
    }
    // A query as long as the series meets exactly one window.
    let whole = distance_profile_nonnorm(&series, &series, unit).unwrap();
    assert_eq!(whole.len(), 1);
    assert_eq!(whole[0].squared_ticks(), 0);
    // A longer query than the series has an empty profile.
    assert!(
        distance_profile_nonnorm(long, &series[..5], unit)
            .unwrap()
            .is_empty()
    );
    // A query that no tick can hold (past the quantizer's range) is refused, not wrapped.
    assert!(distance_profile_nonnorm(&[1e300, 0.0], &series, unit).is_none());
}

#[test]
fn a_pair_whose_squared_distance_needs_more_than_128_bits_has_no_distance() {
    let unit = scale(1.0);
    let top = (1u64 << 53) as f64;
    // The rule is on the range of both windows' ticks: `m·R² < 2¹²⁸` or the pair has none, even
    // when this pair's own distance is small.
    for m in [(1usize << 20) - 1, 1usize << 20] {
        let query = vec![top; m];
        let mut series = vec![-top];
        series.extend(&query);
        let hits = distance_profile_nonnorm(&query, &series, unit).unwrap();
        assert_eq!(hits[1].squared_ticks(), 0, "m {m}");
        assert_eq!(hits[0].index().get().is_some(), m < 1 << 20, "m {m}");
        if m < 1 << 20 {
            assert_eq!(hits[0].squared_ticks(), 1u128 << 108);
        }
    }
}

#[test]
fn the_two_metrics_dispatch_to_different_distances() {
    // A copy of the first shape raised by a constant is the same shape to the z-normalized
    // distance and far away to the raw one.
    let shape = [3.0, -7.0, 5.0, 1.0, -2.0, 8.0, -4.0];
    let mut series: Vec<f64> = shape.to_vec();
    series.extend([40.0, -30.0, 22.0, 9.0, 17.0, -25.0, 31.0]);
    series.extend(shape.iter().map(|x| x + 100.0));
    series.extend([-12.0, 50.0, -6.0, 14.0, 2.0, -33.0, 27.0]);
    let params = MotifParams {
        cutoff: Some(f64::INFINITY),
        max_distance: MaxDistance::Value(1e-6),
        max_matches: None,
        ..MotifParams::default()
    };
    let z = motifs(&series, nz(7), scale(1.0), nz(1), &params);
    assert_eq!(z[0].iter().map(|m| m.0).collect::<Vec<_>>(), [0, 14]);
    let raw = motifs_nonnorm(&series, nz(7), scale(1.0), nz(1), &params);
    assert!(raw.is_empty(), "{raw:?}");
    let z = find_matches(&shape, &series, scale(1.0), MaxDistance::Value(1e-6), None).unwrap();
    assert_eq!(z.iter().map(|m| m.0).collect::<Vec<_>>(), [0, 14]);
    let raw =
        find_matches_nonnorm(&shape, &series, scale(1.0), MaxDistance::Value(1e-6), None).unwrap();
    assert_eq!(raw.iter().map(|m| m.0).collect::<Vec<_>>(), [0]);
}

#[test]
fn an_off_grid_query_is_rounded_to_the_series_scale() {
    let series: Vec<f64> = (0..30)
        .map(|k| f64::from((k * 5 % 7) as u8) * 0.5)
        .collect();
    let on_grid = &series[3..9];
    let nudged: Vec<f64> = on_grid.iter().map(|x| x + 0.1).collect();
    let a = find_matches_nonnorm(on_grid, &series, scale(0.5), MaxDistance::Default, None);
    let b = find_matches_nonnorm(&nudged, &series, scale(0.5), MaxDistance::Default, None);
    assert_eq!(
        group(&a.unwrap(), scale(0.5)),
        group(&b.unwrap(), scale(0.5))
    );
}

#[test]
fn the_squared_distance_is_the_sum_of_squared_tick_differences() {
    let unit = scale(1.0);
    let t = ticks(&[1.0, 4.0, 2.0, 8.0, 5.0, 7.0], unit);
    let (a, b) = (window(&t, 0, 3).unwrap(), window(&t, 2, 3).unwrap());
    assert_eq!(squared(&a, &b), 1 + 16 + 9);
}
