#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! MPdist equals a plain statement of STUMPY's procedure with exact rational orders: the AB-join
//! and BA-join nearest distances sorted exactly (exactly equal ones AB-join first, then by
//! window), the value at the clamped position, and STUMPY's fallback when that value is infinite.
//! On random pairs with gaps, repeats and flat runs, tie-heavy series, every forced route, every
//! selection extreme and every thread count.

#[path = "common/exact_pair.rs"]
mod exact_pair;
#[path = "common/mpdist_brute.rs"]
mod mpdist_brute;
#[path = "common/route.rs"]
mod route;

use ladde::mpdist::{Selection, mpdist};
use ladde::naive::Ticks;
use ladde::quantize::{Scale, to_ticks};
use ladde::route::{RouteForce, forced_route};
use ladde::stats::Series;
use mpdist_brute::{Pick, brute};
use proptest::prelude::*;
use route::{forces, jumpy, nz};
use testkit::generators::{MARKET_TICK, market_series};

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn ticks_of(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

/// The selection `pick` names, or `None` for a percentage outside `[0, 1]`.
fn selection(pick: Pick) -> Option<Selection> {
    match pick {
        Pick::Percentage(p) => Selection::percentage(p),
        Pick::Rank(k) => Some(Selection::rank(k)),
    }
}

fn bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

fn picks() -> Vec<Pick> {
    vec![
        Pick::Percentage(0.05),
        Pick::Percentage(0.0),
        Pick::Percentage(0.5),
        Pick::Percentage(1.0),
        Pick::Percentage(7.5),
        Pick::Percentage(-3.0),
        Pick::Percentage(f64::INFINITY),
        Pick::Percentage(f64::NEG_INFINITY),
        Pick::Rank(0),
        Pick::Rank(1),
        Pick::Rank(7),
        Pick::Rank(usize::MAX),
    ]
}

/// The library at every pick, every thread count in `threads`, equal to the brute force.
fn assert_equals_brute(a: &[f64], b: &[f64], m: usize, scales: (Scale, Scale), threads: &[usize]) {
    let (ta, tb) = (ticks_of(a, scales.0), ticks_of(b, scales.1));
    for pick in picks() {
        let want = bits(brute((a, &ta), (b, &tb), m, pick));
        for &t in threads {
            let got = selection(pick)
                .and_then(|selection| mpdist(a, b, nz(m), scales.0, scales.1, nz(t), selection));
            assert_eq!(bits(got), want, "{pick:?} threads {t} m {m}");
        }
    }
}

fn scales() -> impl Strategy<Value = Scale> {
    prop_oneof![Just(MARKET_TICK), Just(0.5), Just(1.0)].prop_map(scale)
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn random_pairs_equal_the_brute_force(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..40, m)),
        b in (1usize..=6).prop_flat_map(|m| market_series(0..40, m)),
        scale_a in scales(),
        scale_b in scales(),
    ) {
        let _unforced = forced_route(RouteForce::default());
        assert_equals_brute(&a, &b, m, (scale_a, scale_b), &[1, 3]);
    }

    #[test]
    fn a_series_against_a_repeat_of_part_of_itself_equals_the_brute_force(
        m in 1usize..=4,
        a in (1usize..=4).prop_flat_map(|m| market_series(8..30, m)),
        cut in 0usize..8,
        copies in 1usize..4,
    ) {
        let _unforced = forced_route(RouteForce::default());
        let piece = a[cut..].to_vec();
        let b: Vec<f64> = (0..copies).flat_map(|_| piece.iter().copied()).collect();
        let tick = scale(MARKET_TICK);
        assert_equals_brute(&a, &b, m, (tick, tick), &[1, 2]);
        assert_equals_brute(&a, &a, m, (tick, tick), &[1]);
    }

    #[test]
    fn every_forced_route_equals_the_brute_force(
        ((a, m, one), (b, _, _)) in (2usize..=5).prop_flat_map(|m| (jumpy(m), jumpy(m))),
        seed in any::<u64>(),
    ) {
        for force in forces(seed) {
            let _forced = forced_route(force);
            assert_equals_brute(&a, &b, m, (one, one), &[2]);
        }
    }
}

#[test]
fn a_forced_wide_route_fires_through_the_joins() {
    let walk: Vec<f64> = (0..60).map(|i| f64::from((i * 7 + i * i) % 11)).collect();
    let other: Vec<f64> = (0..50).map(|i| f64::from((i * 5 + i * i) % 13)).collect();
    let one = scale(1.0);
    let wide = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    let guard = forced_route(wide);
    let got = mpdist(&walk, &other, nz(6), one, one, nz(2), Selection::default());
    assert!(guard.fired().wide_switches > 0, "{:?}", guard.fired());
    drop(guard);
    let _unforced = forced_route(RouteForce::default());
    let natural = mpdist(&walk, &other, nz(6), one, one, nz(2), Selection::default());
    assert_eq!(bits(got), bits(natural));
}

fn assert_pairs_equal_brute(pairs: &[(Vec<f64>, Vec<f64>)], ms: &[usize], threads: &[usize]) {
    let one = scale(1.0);
    for (a, b) in pairs {
        for &m in ms {
            assert_equals_brute(a, b, m, (one, one), threads);
        }
    }
}

fn period(n: usize, shift: usize, cycle: &[f64]) -> Vec<f64> {
    (0..n).map(|i| cycle[(i + shift) % cycle.len()]).collect()
}

#[test]
fn tie_heavy_series_equal_the_brute_force() {
    let _unforced = forced_route(RouteForce::default());
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
    let pairs = vec![
        (vec![3.0; 25], vec![3.0; 19]),
        (vec![3.0; 25], vec![4.0; 19]),
        (
            [vec![1.0; 12], vec![5.0; 12], vec![1.0; 12]].concat(),
            [vec![5.0; 9], vec![1.0; 9]].concat(),
        ),
        (period(40, 0, &[3.0, 1.0]), period(31, 1, &[3.0, 1.0])),
        (
            period(44, 0, &[3.0, 1.0, 4.0, 1.0]),
            period(37, 2, &[3.0, 1.0, 4.0, 1.0]),
        ),
        (
            period(50, 0, &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]),
            period(41, 3, &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]),
        ),
        (levels(45, 2), levels(38, 2)),
        (levels(45, 3), levels(52, 3)),
    ];
    assert_pairs_equal_brute(&pairs, &[1, 2, 3, 5, 8], &[1, 2, 7]);
    let same = vec![(
        period(33, 0, &[2.0, 7.0, 1.0]),
        period(33, 0, &[2.0, 7.0, 1.0]),
    )];
    assert_pairs_equal_brute(&same, &[3, 4], &[1]);
}

#[test]
fn gaps_at_the_start_and_the_end_equal_the_brute_force() {
    let _unforced = forced_route(RouteForce::default());
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
    let mut pairs = Vec::new();
    for (lead, trail, every) in [
        (0, 0, 9),
        (4, 0, 0),
        (0, 5, 0),
        (6, 6, 7),
        (30, 0, 0),
        (0, 30, 0),
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
        pairs.push((gapped(40, 5), walk(35, 6)));
        pairs.push((walk(35, 7), gapped(40, 8)));
        pairs.push((gapped(36, 9), gapped(41, 10)));
    }
    assert_pairs_equal_brute(&pairs, &[2, 4, 7], &[1, 3]);
    let nothing = vec![f64::NAN; 30];
    assert_pairs_equal_brute(
        &[(nothing.clone(), walk(30, 3)), (nothing.clone(), nothing)],
        &[3],
        &[1],
    );
}

#[test]
fn the_selection_extremes_are_refused_instead_of_clamped() {
    let _unforced = forced_route(RouteForce::default());
    let a: Vec<f64> = (0..30).map(|i| f64::from((i * i + 3 * i) % 17)).collect();
    let b: Vec<f64> = (0..24).map(|i| f64::from((i * i + 5 * i) % 19)).collect();
    let one = scale(1.0);
    let run = |s| mpdist(&a, &b, nz(5), one, one, nz(1), s);
    let all = (a.len() - 4) + (b.len() - 4);
    let largest = run(Selection::rank(all - 1)).unwrap();
    assert_eq!(run(Selection::rank(all)), None);
    assert_eq!(run(Selection::rank(usize::MAX)), None);
    assert_eq!(run(Selection::percentage(1.0).unwrap()), Some(largest));
    for outside in [
        f64::INFINITY,
        1.0 + f64::EPSILON,
        -1.0,
        -f64::MIN_POSITIVE,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        assert!(Selection::percentage(outside).is_none(), "{outside}");
    }
    let smallest = run(Selection::rank(0)).unwrap();
    assert_eq!(run(Selection::percentage(0.0).unwrap()), Some(smallest));
    assert!(smallest <= run(Selection::default()).unwrap());
    assert!(run(Selection::default()).unwrap() <= largest);
    assert!(Selection::rank(all - 1).fits(all) && !Selection::rank(all).fits(all));
    assert!(Selection::percentage(1.0).unwrap().fits(0));
}

#[test]
fn a_series_without_a_window_has_no_distance() {
    let _unforced = forced_route(RouteForce::default());
    let a = [1.0, 2.0, 3.0, 4.0];
    let one = scale(1.0);
    for (x, y) in [(&a[..], &a[..3]), (&a[..3], &a[..]), (&a[..3], &a[..3])] {
        assert_eq!(
            mpdist(x, y, nz(4), one, one, nz(1), Selection::default()),
            None
        );
    }
    assert_eq!(
        mpdist(&a, &a, nz(4), one, one, nz(1), Selection::default()),
        Some(0.0)
    );
    assert_eq!(
        mpdist(&[], &a, nz(1), one, one, nz(1), Selection::default()),
        None
    );
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
    let (a, b) = (walk(3200, 7, 100.0), walk(2700, 11, 100.5));
    for pick in [Pick::Percentage(0.05), Pick::Rank(0), Pick::Percentage(1.0)] {
        let reference = selection(pick)
            .and_then(|selection| mpdist(&a, &b, nz(24), tick, tick, nz(1), selection));
        assert!(reference.is_some_and(f64::is_finite));
        for threads in [2, 3, 7, 16] {
            let got = selection(pick)
                .and_then(|selection| mpdist(&a, &b, nz(24), tick, tick, nz(threads), selection));
            assert_eq!(bits(got), bits(reference), "{pick:?} threads {threads}");
        }
    }
}

#[test]
fn raw_bit_samples_and_extreme_selections_never_panic() {
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
    for round in 0..300 {
        let (a, b) = (
            series((next() % 40) as usize),
            series((next() % 40) as usize),
        );
        let m = nz(1 + (next() % 9) as usize);
        let scales = [
            scale(1.0),
            scale(1e-300),
            scale(1e300),
            scale(f64::MIN_POSITIVE),
        ];
        let (scale_a, scale_b) = (scales[round % 4], scales[(round / 4) % 4]);
        for pick in picks() {
            let _ = selection(pick)
                .and_then(|selection| mpdist(&a, &b, m, scale_a, scale_b, nz(2), selection));
        }
    }
}

#[test]
fn flagged_windows_equal_the_brute_force() {
    let _unforced = forced_route(RouteForce::default());
    let one = scale(1.0);
    let mut state = 0x6A09_E667_F3BC_C909u64;
    let mut next = |every: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % every
    };
    let mut series = Vec::new();
    for (n, levels) in [(34, 3), (29, 2), (40, 5), (25, 3)] {
        let mut s: Vec<f64> = (0..n).map(|_| next(levels) as f64).collect();
        for x in &mut s[5..11] {
            *x = 2.0;
        }
        series.push(s);
    }
    series[2][14] = f64::NAN;
    for (a, b) in [(0, 1), (1, 0), (2, 3), (3, 2), (0, 0), (2, 1)] {
        for m in [2, 4, 6] {
            let flag_sets = |s: &[f64], which: usize, next: &mut dyn FnMut(u64) -> u64| {
                (0..s.len() + 1 - m)
                    .map(|i| match which {
                        0 => s[i..i + m].iter().all(|x| *x == s[i]),
                        1 => false,
                        2 => true,
                        _ => next(3) == 0,
                    })
                    .collect::<Vec<bool>>()
            };
            for which in 0..4 {
                let (fa, fb) = (
                    flag_sets(&series[a], which, &mut next),
                    flag_sets(&series[b], (which + a) % 4, &mut next),
                );
                let (ta, tb) = (ticks_of(&series[a], one), ticks_of(&series[b], one));
                let (ra, rb) = (
                    Ticks::with_constant_windows(&ta, &fa),
                    Ticks::with_constant_windows(&tb, &fb),
                );
                let (sa, sb) = (
                    Series::with_constant_windows(&series[a], nz(m), &fa).unwrap(),
                    Series::with_constant_windows(&series[b], nz(m), &fb).unwrap(),
                );
                for pick in picks() {
                    let want = bits(brute((&series[a], ra), (&series[b], rb), m, pick));
                    for threads in [1, 3] {
                        let got = selection(pick).and_then(|selection| {
                            mpdist(sa, sb, nz(m), one, one, nz(threads), selection)
                        });
                        assert_eq!(bits(got), want, "{pick:?} {a}/{b} m {m} flags {which}");
                    }
                }
            }
        }
    }
}
