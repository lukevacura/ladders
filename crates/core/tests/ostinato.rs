#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Ostinato equals a plain statement of STUMPY's procedure with exact rational orders (the join
//! order of windows, the running best radius, the strict improvement, the central-motif pass),
//! and its radius equals the smallest radius over every window of every series, on random sets
//! with gaps, repeats and flat runs, tie-heavy sets, every forced route and every thread count.

#[path = "common/exact_pair.rs"]
mod exact_pair;
#[path = "common/route.rs"]
mod route;

use core::cmp::Ordering;

use exact_pair::{Rho, cmp_rho, cross_pair};
use ladde::naive::Ticks;
use ladde::ostinato::{Consensus, ostinato};
use ladde::quantize::{Scale, to_ticks};
use ladde::route::{RouteForce, forced_route};
use ladde::stats::Series;
use proptest::prelude::*;
use route::{forces, jumpy, nz};
use testkit::generators::{MARKET_TICK, market_series};

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn ticks_of(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

/// A distance as the exact correlation it is ranked by and the f64 squared distance reported.
type Dist = (Rho, f64);

/// `a` is strictly nearer than `b`; no distance is infinitely far.
fn nearer(a: Option<Dist>, b: Option<Dist>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => cmp_rho(a.0, b.0) == Ordering::Greater,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

/// The exactly nearest window of `target` to the window of `from` at `at`, ties to the smaller
/// start: its start and distance, `None` when no window of `target` has one.
fn nearest(target: Ticks<'_>, from: Ticks<'_>, m: usize, at: usize) -> Option<(usize, Dist)> {
    let mut best: Option<(usize, Dist)> = None;
    for w in 0..target.len() + 1 - m {
        let Some(d) = cross_pair(target, from, m, w, at) else {
            continue;
        };
        if nearer(Some(d), best.map(|b| b.1)) {
            best = Some((w, d));
        }
    }
    best
}

fn distance(d: Dist) -> f64 {
    libm::sqrt(d.1)
}

fn is_close(a: f64, b: f64) -> bool {
    a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-8 + 1e-5 * b.abs())
}

/// One window's distance to the nearest window of every series.
fn nearest_everywhere(sets: &[Ticks<'_>], m: usize, series: usize, at: usize) -> Vec<(usize, f64)> {
    sets.iter()
        .map(|t| {
            nearest(*t, sets[series], m, at).map_or((0, f64::INFINITY), |(w, d)| (w, distance(d)))
        })
        .collect()
}

/// STUMPY's `_ostinato` and `_get_central_motif`, windows and distances ordered exactly.
fn brute(sets: &[Ticks<'_>], m: usize) -> Option<Consensus> {
    let k = sets.len();
    if k == 0 || sets.iter().any(|t| t.len() < m) {
        return None;
    }
    let mut best: Option<(Dist, usize, usize)> = None;
    for j in 0..k {
        let h = (j + 1) % k;
        let mut joined: Vec<(Dist, usize)> = (0..sets[j].len() + 1 - m)
            .filter_map(|q| {
                let (w, _) = nearest(sets[h], sets[j], m, q)?;
                Some((cross_pair(sets[j], sets[h], m, q, w)?, q))
            })
            .collect();
        joined.sort_by(|x, y| cmp_rho(y.0.0, x.0.0).then(x.1.cmp(&y.1)));
        for (mut radius, q) in joined {
            if best.is_some_and(|b| !nearer(Some(radius), Some(b.0))) {
                break;
            }
            let mut finite = true;
            for i in (0..k).filter(|&i| i != j && i != h) {
                let Some((_, d)) = nearest(sets[i], sets[j], m, q) else {
                    finite = false;
                    break;
                };
                if nearer(Some(radius), Some(d)) {
                    radius = d;
                }
                if best.is_some_and(|b| !nearer(Some(radius), Some(b.0))) {
                    break;
                }
            }
            if finite && best.is_none_or(|b| nearer(Some(radius), Some(b.0))) {
                best = Some((radius, j, q));
            }
        }
    }
    let Some((radius, mut series, mut window)) = best else {
        return Some(Consensus {
            radius: f64::INFINITY,
            series: 0,
            window: 0,
        });
    };
    let radius = distance(radius);
    let around = nearest_everywhere(sets, m, series, window);
    let mut best_mean = around.iter().fold(0.0, |s, x| s + x.1) / k as f64;
    for (candidate, &(start, closest)) in around.iter().enumerate() {
        if !is_close(closest, radius) {
            continue;
        }
        let radii: Vec<f64> = nearest_everywhere(sets, m, candidate, start)
            .iter()
            .map(|x| x.1)
            .collect();
        let farthest = radii.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x));
        let candidate_mean = radii.iter().fold(0.0, |s, x| s + x) / k as f64;
        if is_close(farthest, radius) && candidate_mean < best_mean {
            series = candidate;
            window = start;
            best_mean = candidate_mean;
        }
    }
    Some(Consensus {
        radius,
        series,
        window,
    })
}

/// The smallest over all windows of the farthest nearest distance to the other series: the
/// radius the search must find whatever its pruning.
fn smallest_radius(sets: &[Ticks<'_>], m: usize) -> f64 {
    let mut smallest = f64::INFINITY;
    for j in 0..sets.len() {
        for q in 0..sets[j].len() + 1 - m {
            let farthest = (0..sets.len())
                .filter(|&i| i != j || sets.len() == 1)
                .map(|i| {
                    nearest(sets[i], sets[j], m, q).map_or(f64::INFINITY, |(_, d)| distance(d))
                })
                .fold(0.0, f64::max);
            smallest = smallest.min(farthest);
        }
    }
    smallest
}

fn same(a: Option<Consensus>, b: Option<Consensus>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => {
            a.radius.to_bits() == b.radius.to_bits() && a.series == b.series && a.window == b.window
        }
        (a, b) => a.is_none() && b.is_none(),
    }
}

fn run(series: &[Vec<f64>], scales: &[Scale], m: usize, threads: usize) -> Option<Consensus> {
    let set: Vec<(&[f64], Scale)> = series
        .iter()
        .zip(scales)
        .map(|(s, &scale)| (&s[..], scale))
        .collect();
    ostinato(&set, nz(m), nz(threads))
}

/// The library at each thread count equal to the brute force; and its radius the smallest one.
fn assert_equals_brute(series: &[Vec<f64>], scales: &[Scale], m: usize, threads: &[usize]) {
    let owned: Vec<Vec<Option<i64>>> = series
        .iter()
        .zip(scales)
        .map(|(s, &scale)| ticks_of(s, scale))
        .collect();
    let sets: Vec<Ticks> = owned.iter().map(Ticks::from).collect();
    let want = brute(&sets, m);
    for &t in threads {
        let got = run(series, scales, m, t);
        assert!(same(got, want), "threads {t} m {m}: {got:?} vs {want:?}");
    }
    if let Some(found) = want.filter(|c| c.radius.is_finite()) {
        let smallest = smallest_radius(&sets, m);
        assert!(
            (found.radius - smallest).abs() <= 1e-9 * smallest.max(1.0),
            "{found:?} vs smallest {smallest}"
        );
    }
}

fn scales() -> impl Strategy<Value = Scale> {
    prop_oneof![Just(MARKET_TICK), Just(0.5), Just(1.0)].prop_map(scale)
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn random_sets_equal_the_brute_force(
        m in 1usize..=5,
        set in prop::collection::vec(
            ((1usize..=5).prop_flat_map(|m| market_series(0..30, m)), scales()), 0..5),
    ) {
        let _unforced = forced_route(RouteForce::default());
        let (series, scales): (Vec<_>, Vec<_>) = set.into_iter().unzip();
        assert_equals_brute(&series, &scales, m, &[1, 3]);
    }

    #[test]
    fn sets_sharing_a_pattern_equal_the_brute_force(
        m in 2usize..=4,
        base in (2usize..=4).prop_flat_map(|m| market_series(m + 4..24, m)),
        copies in 1usize..5,
        offsets in prop::collection::vec(0usize..6, 5),
    ) {
        let _unforced = forced_route(RouteForce::default());
        let tick = scale(MARKET_TICK);
        let series: Vec<Vec<f64>> = (0..copies)
            .map(|c| {
                let cut = offsets[c] % base.len();
                let mut s = base[cut..].to_vec();
                s.extend(base.iter().map(|x| x + 0.01 * c as f64));
                s
            })
            .collect();
        assert_equals_brute(&series, &vec![tick; copies], m, &[1, 2]);
    }

    #[test]
    fn every_forced_route_equals_the_brute_force(
        (first, second, third, m) in (2usize..=4).prop_flat_map(|m| {
            (jumpy(m), jumpy(m), jumpy(m), Just(m))
        }),
        seed in any::<u64>(),
    ) {
        let series = vec![first.0, second.0, third.0];
        let one = scale(1.0);
        for force in forces(seed) {
            let _forced = forced_route(force);
            assert_equals_brute(&series, &[one, one, one], m, &[2]);
        }
    }
}

#[test]
fn a_forced_wide_route_fires_through_the_joins() {
    let series: Vec<Vec<f64>> = (0..3usize)
        .map(|s| {
            (0..40usize)
                .map(|i| f64::from(((i + 3 * s) * 7 + i * i) as u32 % 11))
                .collect()
        })
        .collect();
    let one = scale(1.0);
    let wide = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    let guard = forced_route(wide);
    let got = run(&series, &[one, one, one], 5, 2);
    assert!(guard.fired().wide_switches > 0, "{:?}", guard.fired());
    drop(guard);
    let _unforced = forced_route(RouteForce::default());
    assert!(same(got, run(&series, &[one, one, one], 5, 2)));
}

fn period(n: usize, shift: usize, cycle: &[f64]) -> Vec<f64> {
    (0..n).map(|i| cycle[(i + shift) % cycle.len()]).collect()
}

#[test]
fn tie_heavy_sets_equal_the_brute_force() {
    let _unforced = forced_route(RouteForce::default());
    let one = scale(1.0);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut levels = |n: usize, levels: u64| -> Vec<f64> {
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % levels) as f64
            })
            .collect()
    };
    let cycle7 = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let sets: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![3.0; 20], vec![3.0; 17], vec![3.0; 25]],
        vec![vec![3.0; 20], vec![4.0; 17], vec![3.0; 25]],
        vec![
            [vec![1.0; 10], vec![5.0; 10], vec![1.0; 10]].concat(),
            [vec![5.0; 9], vec![1.0; 9]].concat(),
            vec![2.0; 14],
        ],
        vec![period(30, 0, &[3.0, 1.0]), period(25, 1, &[3.0, 1.0])],
        vec![
            period(33, 0, &[3.0, 1.0, 4.0, 1.0]),
            period(28, 2, &[3.0, 1.0, 4.0, 1.0]),
            period(31, 1, &[3.0, 1.0, 4.0, 1.0]),
            period(26, 3, &[3.0, 1.0, 4.0, 1.0]),
        ],
        vec![
            period(40, 0, &cycle7),
            period(35, 3, &cycle7),
            period(30, 5, &cycle7),
        ],
        vec![levels(36, 2), levels(30, 2), levels(33, 2)],
        vec![levels(36, 3), levels(30, 3)],
        vec![levels(30, 3)],
        vec![
            period(24, 0, &cycle7),
            period(24, 0, &cycle7),
            period(24, 0, &cycle7),
        ],
    ];
    for set in &sets {
        for m in [1, 2, 3, 5, 7] {
            assert_equals_brute(set, &vec![one; set.len()], m, &[1, 2, 7]);
        }
    }
}

#[test]
fn gaps_at_the_start_and_the_end_equal_the_brute_force() {
    let _unforced = forced_route(RouteForce::default());
    let one = scale(1.0);
    let walk = |n: usize, seed: u64| -> Vec<f64> {
        let mut state = seed | 1;
        let mut x = 0.0;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                x += ((state >> 58) as i64 % 5 - 2) as f64;
                x
            })
            .collect()
    };
    for (lead, trail, every) in [
        (0, 0, 9),
        (4, 0, 0),
        (0, 5, 0),
        (6, 6, 7),
        (22, 0, 0),
        (0, 22, 0),
    ] {
        let gapped = |n: usize, seed: u64| {
            let mut s = walk(n, seed);
            for k in (0..n).filter(|k| every != 0 && k % every == every - 1) {
                s[k] = f64::NAN;
            }
            s[..lead.min(n)].fill(f64::NAN);
            s[n - trail.min(n)..].fill(f64::INFINITY);
            s
        };
        for m in [2, 4, 6] {
            let set = vec![gapped(30, 5), walk(26, 6), gapped(28, 7)];
            assert_equals_brute(&set, &[one, one, one], m, &[1, 3]);
            let set = vec![gapped(30, 8), gapped(27, 9)];
            assert_equals_brute(&set, &[one, one], m, &[1]);
        }
    }
    let nothing = vec![f64::NAN; 20];
    let set = vec![nothing.clone(), walk(20, 3), nothing.clone()];
    assert_equals_brute(&set, &[one, one, one], 3, &[1]);
    let found = run(&set, &[one, one, one], 3, 1).unwrap();
    assert_eq!(
        found,
        Consensus {
            radius: f64::INFINITY,
            series: 0,
            window: 0
        }
    );
}

#[test]
fn a_pattern_planted_in_every_series_is_the_consensus() {
    let _unforced = forced_route(RouteForce::default());
    let one = scale(1.0);
    let pattern = [0.0, 6.0, -3.0, 9.0, 2.0, -7.0, 4.0, 1.0];
    let noise = |n: usize, seed: u64| -> Vec<f64> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % 21) as f64 - 10.0
            })
            .collect()
    };
    let plant = |at: usize, seed: u64, level: f64| -> Vec<f64> {
        let mut s = noise(60, seed);
        for (i, p) in pattern.iter().enumerate() {
            s[at + i] = level + 3.0 * p;
        }
        s
    };
    let set = vec![
        plant(5, 1, 0.0),
        plant(31, 2, 40.0),
        plant(17, 3, -25.0),
        plant(44, 4, 7.0),
    ];
    let found = run(&set, &[one; 4], 8, 2).unwrap();
    assert_eq!(found.radius, 0.0);
    let starts = [5, 31, 17, 44];
    assert_eq!(found.window, starts[found.series]);
}

#[test]
fn an_empty_set_or_a_series_without_a_window_has_no_consensus() {
    let _unforced = forced_route(RouteForce::default());
    let one = scale(1.0);
    assert!(run(&[], &[], 3, 1).is_none());
    let short = vec![1.0, 2.0];
    let long: Vec<f64> = (0..20).map(|i| f64::from(i * i % 7)).collect();
    assert!(run(&[long.clone(), short.clone()], &[one, one], 3, 1).is_none());
    assert!(run(&[short, long.clone()], &[one, one], 3, 1).is_none());
    assert!(run(std::slice::from_ref(&long), &[one], 21, 1).is_none());
    let alone = run(&[long], &[one], 3, 1).unwrap();
    assert_eq!(alone.radius, 0.0);
    assert_eq!(alone.series, 0);
}

#[test]
fn threads_change_no_bit_on_series_long_enough_to_run_threaded() {
    let _unforced = forced_route(RouteForce::default());
    let walk = |n: usize, seed: u64, level: f64| -> Vec<f64> {
        let mut state = seed | 1;
        let mut x = level;
        (0..n)
            .map(|i| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                x += ((state >> 58) as i64 % 3 - 1) as f64 * 0.01;
                if i % 97 == 41 { f64::NAN } else { x }
            })
            .collect()
    };
    let tick = scale(MARKET_TICK);
    let set = vec![
        walk(2400, 7, 100.0),
        walk(2100, 11, 100.5),
        walk(1800, 13, 99.0),
    ];
    let reference = run(&set, &[tick; 3], 24, 1);
    assert!(reference.is_some_and(|c| c.radius.is_finite()));
    for threads in [2, 3, 7, 16] {
        assert!(
            same(run(&set, &[tick; 3], 24, threads), reference),
            "threads {threads}"
        );
    }
}

#[test]
fn raw_bit_samples_never_panic() {
    let _unforced = forced_route(RouteForce::default());
    let state = std::cell::Cell::new(0xD1B5_4A32_D192_ED03u64);
    let next = || {
        let x = state
            .get()
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state.set(x);
        x >> 11
    };
    let series = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| match next() % 6 {
                0 => f64::from_bits(next() << 11 ^ next()),
                1 => f64::NAN,
                2 => (next() % 5) as f64,
                _ => (next() % 1000) as f64 * 1e12,
            })
            .collect()
    };
    let scales = [
        scale(1.0),
        scale(1e-300),
        scale(1e300),
        scale(f64::MIN_POSITIVE),
    ];
    for round in 0..200 {
        let count = (next() % 4) as usize;
        let set: Vec<Vec<f64>> = (0..count).map(|_| series((next() % 30) as usize)).collect();
        let used: Vec<Scale> = (0..count).map(|i| scales[(round + i) % 4]).collect();
        let _ = run(&set, &used, 1 + (next() % 8) as usize, 2);
    }
}

/// Flags for every window of `series` at `m`: the default rule, none constant, all constant, a
/// near-constant predicate, and a spread of pseudo-random ones.
fn flag_family(series: &[f64], m: usize, which: usize, state: &mut u64) -> Vec<bool> {
    let windows = (series.len() + 1).saturating_sub(m);
    let mut coin = |every: u64| {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33).is_multiple_of(every)
    };
    (0..windows)
        .map(|i| {
            let w = &series[i..i + m];
            match which % 5 {
                0 => w.iter().all(|x| *x == w[0]),
                1 => false,
                2 => true,
                3 => {
                    let hi = w.iter().copied().fold(f64::MIN, f64::max);
                    let lo = w.iter().copied().fold(f64::MAX, f64::min);
                    hi - lo <= 2.0
                }
                _ => coin(3),
            }
        })
        .collect()
}

#[test]
fn flagged_windows_equal_the_brute_force() {
    let _unforced = forced_route(RouteForce::default());
    let one = scale(1.0);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut levels = |n: usize, levels: u64| -> Vec<f64> {
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % levels) as f64
            })
            .collect()
    };
    let mut gapped = levels(26, 4);
    gapped[9] = f64::NAN;
    gapped[10] = f64::NAN;
    let sets: Vec<Vec<Vec<f64>>> = vec![
        vec![levels(30, 3), levels(26, 3), levels(28, 5)],
        vec![levels(24, 2), levels(24, 2)],
        vec![vec![3.0; 20], levels(22, 4), vec![3.0; 17]],
        vec![gapped, levels(25, 3), levels(27, 6)],
        vec![levels(34, 7), levels(30, 7), levels(31, 7), levels(29, 7)],
    ];
    let mut flag_state = 11u64;
    for (n, set) in sets.iter().enumerate() {
        for m in [2, 4, 5] {
            for which in 0..5 {
                let ticks: Vec<Vec<Option<i64>>> = set.iter().map(|s| ticks_of(s, one)).collect();
                let flags: Vec<Vec<bool>> = set
                    .iter()
                    .enumerate()
                    .map(|(i, s)| flag_family(s, m, which + i + n, &mut flag_state))
                    .collect();
                let reference: Vec<Ticks> = ticks
                    .iter()
                    .zip(&flags)
                    .map(|(t, f)| Ticks::with_constant_windows(t, f))
                    .collect();
                let want = brute(&reference, m);
                let flagged: Vec<(Series, Scale)> = set
                    .iter()
                    .zip(&flags)
                    .map(|(s, f)| (Series::with_constant_windows(s, nz(m), f).unwrap(), one))
                    .collect();
                for threads in [1, 3] {
                    let got = ostinato(&flagged, nz(m), nz(threads));
                    assert!(
                        same(got, want),
                        "set {n} m {m} flags {which} threads {threads}"
                    );
                }
                if let Some(found) = want.filter(|c| c.radius.is_finite()) {
                    let smallest = smallest_radius(&reference, m);
                    assert!((found.radius - smallest).abs() <= 1e-9 * smallest.max(1.0));
                }
            }
        }
    }
}
