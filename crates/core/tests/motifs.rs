#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Batch motifs equal a plain, independent statement of STUMPY's `motifs` procedure with exact
//! rational orders: on random integer series with planted repeats, scaled copies (exactly equal
//! distances) and NaN gaps, at every thread count; motifs are outside each other's zones.

use core::cmp::Ordering;
use core::num::NonZeroUsize;

#[path = "common/exact_pair.rs"]
mod exact_pair;

use exact_pair::{Rho, cmp_rho, pair};
use ladde::entry::Entry;
use ladde::motif::{MotifParams, motifs};
use ladde::naive::{Ticks, naive_two_sided};
use ladde::pattern::MaxDistance;
use ladde::quantize::{Scale, to_ticks};
use ladde::stats::Series;
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// STUMPY's default threshold over the finite values: `max(mean - 2 std, min)`.
fn default_threshold(values: &[f64]) -> Option<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().fold(0.0, |a, x| a + x) / n;
    let var = finite.iter().fold(0.0, |a, x| a + (x - mean) * (x - mean)) / n;
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    Some((mean - 2.0 * libm::sqrt(var)).max(min))
}

type Group = Vec<(u64, u64)>;

/// `match` of the window at `start`: itself first, then the exactly nearest remaining window
/// within the threshold, blanking the zone around each, until `limit`.
fn brute_matches(
    ticks: Ticks<'_>,
    m: usize,
    start: usize,
    max_distance: MaxDistance,
    limit: Option<usize>,
) -> Group {
    let windows = ticks.len() - m + 1;
    let zone = ladde::exclusion::zone(m);
    let mut d: Vec<f64> = (0..windows)
        .map(|i| pair(ticks, m, i, start).map_or(f64::INFINITY, |(_, d2)| libm::sqrt(d2)))
        .collect();
    d[start] = 0.0;
    let threshold = match max_distance {
        MaxDistance::Value(v) => Some(v),
        MaxDistance::Default => default_threshold(&d),
    };
    let mut out = Vec::new();
    let Some(threshold) = threshold else {
        return out;
    };
    let mut alive: Vec<bool> = d.iter().map(|&x| x.is_finite() && x <= threshold).collect();
    let mut next = Some(start).filter(|_| alive[start]);
    while out.len() < limit.unwrap_or(usize::MAX) {
        let best = next.take().or_else(|| {
            let mut best: Option<usize> = None;
            for i in (0..windows).filter(|&i| alive[i]) {
                let rho = |w: usize| pair(ticks, m, w, start).unwrap().0;
                if best.is_none_or(|b| cmp_rho(rho(i), rho(b)) == Ordering::Greater) {
                    best = Some(i);
                }
            }
            best
        });
        let Some(best) = best else { break };
        out.push((best as u64, d[best].to_bits()));
        for (i, a) in alive.iter_mut().enumerate() {
            if i.abs_diff(best) <= zone {
                *a = false;
            }
        }
    }
    out
}

/// Each window's nearest-neighbor distance (f64) and exact correlation, from the oracle.
fn profile_view<'a>(ticks: impl Into<Ticks<'a>>, m: usize) -> (Vec<f64>, Vec<Option<Rho>>) {
    let ticks: Ticks<'a> = ticks.into();
    let profile = naive_two_sided(ticks, m);
    let distances = profile
        .iter()
        .map(|(_, sides)| libm::sqrt(sides.nearest.squared_distance()))
        .collect();
    let exact = profile
        .iter()
        .map(|(row, sides)| {
            let neighbor = sides.nearest.index().get()? as usize;
            pair(ticks, m, *row as usize, neighbor).map(|(rho, _)| rho)
        })
        .collect();
    (distances, exact)
}

/// The exactly nearest window not excluded, ties to the smaller start.
fn pick(exact: &[Option<Rho>], excluded: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for i in (0..exact.len()).filter(|&i| !excluded[i] && exact[i].is_some()) {
        if best.is_none_or(|b| cmp_rho(exact[i].unwrap(), exact[b].unwrap()) == Ordering::Greater) {
            best = Some(i);
        }
    }
    best
}

/// STUMPY's `motifs` loop on the two-sided profile, windows ordered by exact correlation.
fn brute_motifs<'a>(ticks: impl Into<Ticks<'a>>, m: usize, params: &MotifParams) -> Vec<Group> {
    let ticks: Ticks<'a> = ticks.into();
    let zone = ladde::exclusion::zone(m);
    let (distances, exact) = profile_view(ticks, m);
    let mut found = Vec::new();
    let mut excluded = vec![false; exact.len()];
    let Some(first) = pick(&exact, &excluded) else {
        return found;
    };
    // The default cutoff is never below the exactly nearest window's distance.
    let cutoff = params
        .cutoff
        .unwrap_or_else(|| default_threshold(&distances).unwrap().max(distances[first]));
    while found.len() < params.max_motifs {
        let Some(candidate) = pick(&exact, &excluded) else {
            break;
        };
        let d = distances[candidate];
        if d > cutoff || !d.is_finite() {
            break;
        }
        if matches!(params.max_distance, MaxDistance::Value(v) if d > v) {
            break;
        }
        let matches = brute_matches(ticks, m, candidate, params.max_distance, params.max_matches);
        let blocked: Vec<usize> = if matches.is_empty() {
            vec![candidate]
        } else {
            matches.iter().map(|&(i, _)| i as usize).collect()
        };
        for at in blocked {
            for (i, flag) in excluded.iter_mut().enumerate() {
                if i.abs_diff(at) <= zone {
                    *flag = true;
                }
            }
        }
        if matches.len() > params.min_neighbors {
            found.push(matches);
        }
    }
    found
}

fn bits(group: &[(u64, Entry)]) -> Group {
    group
        .iter()
        .map(|&(start, entry)| {
            assert_eq!(entry.index().get(), Some(start));
            (start, libm::sqrt(entry.squared_distance()).to_bits())
        })
        .collect()
}

fn library<'a>(
    series: impl Into<Series<'a>>,
    m: usize,
    threads: usize,
    params: &MotifParams,
) -> Vec<Group> {
    motifs(series, nz(m), Scale::new(1.0).unwrap(), nz(threads), params)
        .iter()
        .map(|group| bits(group))
        .collect()
}

fn ticks_of(series: &[f64]) -> Vec<Option<i64>> {
    let scale = Scale::new(1.0).unwrap();
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

/// Representatives are outside every earlier group's zones, and each group is outside its own.
fn assert_zones(groups: &[Group], m: usize) {
    let zone = ladde::exclusion::zone(m) as u64;
    for (g, group) in groups.iter().enumerate() {
        for (k, a) in group.iter().enumerate() {
            for b in &group[k + 1..] {
                assert!(a.0.abs_diff(b.0) > zone, "inside one group: {group:?}");
            }
        }
        for earlier in &groups[..g] {
            for b in earlier {
                assert!(group[0].0.abs_diff(b.0) > zone, "{groups:?}");
            }
        }
    }
}

fn param_grid(m: usize) -> Vec<MotifParams> {
    let reach = libm::sqrt(2.0 * m as f64);
    let mut grid = Vec::new();
    for (min_neighbors, max_matches, max_motifs) in [
        (0, Some(10), 1),
        (1, Some(10), 3),
        (1, None, 100),
        (2, Some(3), 4),
        (0, Some(1), 100),
        (1, Some(0), 2),
    ] {
        for max_distance in [
            MaxDistance::Default,
            MaxDistance::Value(0.0),
            MaxDistance::Value(0.8 * reach),
            MaxDistance::Value(f64::INFINITY),
        ] {
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

fn assert_equals_brute(series: &[f64], m: usize) -> usize {
    let ticks = ticks_of(series);
    let mut nonempty = 0;
    for params in param_grid(m) {
        let want = brute_motifs(&ticks, m, &params);
        let got = library(series, m, 1, &params);
        assert_eq!(got, want, "{params:?}");
        assert_zones(&got, m);
        nonempty += usize::from(!got.is_empty());
        for threads in [2, 7] {
            assert_eq!(library(series, m, threads, &params), got, "{threads}");
        }
    }
    nonempty
}

#[derive(Debug, Clone)]
struct Plant {
    from: prop::sample::Index,
    to: prop::sample::Index,
    scale: i32,
    shift: i32,
}

fn series_strategy() -> impl Strategy<Value = (Vec<f64>, usize)> {
    (
        proptest::collection::vec(-12i32..12, 24..64),
        proptest::collection::vec(
            (
                any::<prop::sample::Index>(),
                any::<prop::sample::Index>(),
                prop_oneof![Just(1), Just(-1), Just(2), Just(3), Just(-2)],
                -9i32..9,
            )
                .prop_map(|(from, to, scale, shift)| Plant {
                    from,
                    to,
                    scale,
                    shift,
                }),
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
                    series[to + k] = f64::from(plant.scale * base[from + k] + plant.shift);
                }
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
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(20))
        .run(&series_strategy(), |(series, m)| {
            nonempty.set(nonempty.get() + assert_equals_brute(&series, m));
            cases.set(cases.get() + 1);
            Ok(())
        })
        .unwrap();
    assert!(nonempty.get() >= cases.get() * 20, "{}", nonempty.get());
}

/// A shape repeated as scaled and shifted copies: every pair among them is exactly equally
/// near, though their f64 distances can differ in the last bit.
#[test]
fn scaled_copies_are_ordered_by_index_not_by_rounded_distance() {
    let shape = [3, -7, 5, 1, -2, 8, -4];
    let mut series: Vec<f64> = Vec::new();
    for (k, c) in [(1, 0), (3, 24), (-2, 5), (5, -11), (7, 3)] {
        series.extend(shape.iter().map(|&x| f64::from(k * x + c)));
        series.extend([1.0, 4.0, 1.0]);
    }
    let params = MotifParams {
        max_motifs: 3,
        max_matches: None,
        max_distance: MaxDistance::Value(f64::INFINITY),
        cutoff: Some(f64::INFINITY),
        ..MotifParams::default()
    };
    let got = library(&series, 7, 1, &params);
    assert_eq!(got, brute_motifs(&ticks_of(&series), 7, &params));
    assert_eq!(got[0][0].0, 0);
    assert_zones(&got, 7);
    let defaults = MotifParams::default();
    let got = library(&series, 7, 1, &defaults);
    assert_eq!(got, brute_motifs(&ticks_of(&series), 7, &defaults));
    assert_eq!(got[0][0].0, 0);
}

#[test]
fn nan_gaps_and_flat_runs_equal_the_brute_force_procedure() {
    let mut series: Vec<f64> = (0..70).map(|k| f64::from((k * 7 % 11) as u8)).collect();
    series[8..24].fill(5.0);
    series[33] = f64::NAN;
    series[50..53].fill(f64::NAN);
    assert!(assert_equals_brute(&series, 5) > 0);
}

#[test]
fn a_series_without_a_finite_window_has_no_motifs() {
    let series = [f64::NAN; 20];
    assert!(library(&series, 4, 1, &MotifParams::default()).is_empty());
    assert!(library(&series[..3], 4, 1, &MotifParams::default()).is_empty());
}

/// Two exactly equal distances can round apart, so the smallest f64 distance need not be the
/// exactly nearest window's: the default cutoff must still admit the exactly nearest window.
#[test]
fn the_default_cutoff_admits_the_exactly_nearest_window_when_roundings_differ() {
    let mut hits = 0;
    for seed in 0..70u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        let mut level = 0.0;
        let series: Vec<f64> = (0..120)
            .map(|_| {
                level += next(2001) as f64 - 1000.0;
                level
            })
            .collect();
        let ticks = ticks_of(&series);
        let (distances, exact) = profile_view(&ticks, 8);
        let first = pick(&exact, &vec![false; exact.len()]).unwrap();
        let unfloored = default_threshold(&distances).unwrap();
        if distances[first] > unfloored {
            hits += 1;
            let params = MotifParams {
                max_distance: MaxDistance::Value(f64::INFINITY),
                ..MotifParams::default()
            };
            let got = library(&series, 8, 1, &params);
            assert_eq!(
                got.first().map(|g| g[0].0),
                Some(first as u64),
                "seed {seed}"
            );
            assert_eq!(got, brute_motifs(&ticks, 8, &params));
            break;
        }
    }
    assert!(hits > 0, "no series had the nearest pair round apart");
}

#[test]
fn flagged_windows_equal_the_brute_force() {
    let mut state = 0x1F83_D9AB_FB41_BD6Bu64;
    let mut next = |every: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % every
    };
    let mut planted: Vec<f64> = (0..44).map(|_| next(9) as f64).collect();
    let pattern = [0.0, 6.0, -3.0, 9.0, 2.0, -7.0];
    for at in [3, 17, 30] {
        planted[at..at + 6].copy_from_slice(&pattern);
    }
    let mut gapped: Vec<f64> = (0..40).map(|_| next(4) as f64).collect();
    gapped[12] = f64::NAN;
    let mut flat_runs: Vec<f64> = (0..42).map(|_| next(3) as f64).collect();
    flat_runs[5..14].fill(2.0);
    flat_runs[25..30].fill(0.0);
    for series in [planted, gapped, flat_runs] {
        let ticks = ticks_of(&series);
        for m in [4usize, 6] {
            for which in 0..5 {
                let flags: Vec<bool> = (0..series.len() + 1 - m)
                    .map(|i| match which {
                        0 => series[i..i + m].iter().all(|x| *x == series[i]),
                        1 => false,
                        2 => true,
                        3 => series[i..i + m].iter().copied().fold(0.0, f64::max) <= 2.0,
                        _ => next(4) == 0,
                    })
                    .collect();
                let reference = Ticks::with_constant_windows(&ticks, &flags);
                let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
                for params in param_grid(m) {
                    let want = brute_motifs(reference, m, &params);
                    for threads in [1, 3] {
                        let got = library(flagged, m, threads, &params);
                        assert_eq!(got, want, "m {m} flags {which} {params:?}");
                    }
                }
            }
        }
    }
}
