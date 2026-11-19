#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Non-normalized MPdist and consensus motif (STUMPY's `aampdist` and `aamp_ostinato`) equal a
//! plain, independent statement of the procedure on exact integer ticks: on random series with
//! planted exact repeats, series of ties, NaN and infinite gaps, jumps past the `u64` bound and
//! the edges of the window count, at every selection and thread count; the z-normalized and
//! non-normalized dispatches see different distances, and the shared scale is applied once.

use core::num::NonZeroUsize;

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/nonnorm_derived.rs"]
mod reference;

use ladde::mpdist::{Selection, aampdist, mpdist};
use ladde::ostinato::{aamp_ostinato, ostinato};
use ladde::quantize::Scale;
use nonnorm::{planted, series as walk_series, tie_series};
use proptest::prelude::*;
use reference::{Pick, brute_aamp_ostinato, brute_aampdist};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

/// The selection `pick` names, or `None` for a percentage outside `[0, 1]`.
fn selection(pick: Pick) -> Option<Selection> {
    match pick {
        Pick::Percentage(p) => Selection::percentage(p),
        Pick::Rank(k) => Some(Selection::rank(k)),
    }
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

fn assert_dist_equals_brute(a: &[f64], b: &[f64], m: usize, scale: Scale, threads: &[usize]) {
    for pick in picks() {
        let want = bits(brute_aampdist(a, b, m, scale, pick));
        for &t in threads {
            let got = selection(pick)
                .and_then(|selection| aampdist(a, b, nz(m), scale, nz(t), selection));
            assert_eq!(bits(got), want, "{pick:?} threads {t} m {m}");
        }
    }
}

fn consensus_bits(found: Option<(f64, usize, usize)>) -> Option<(u64, usize, usize)> {
    found.map(|(radius, series, window)| (radius.to_bits(), series, window))
}

fn assert_consensus_equals_brute(set: &[Vec<f64>], m: usize, scale: Scale, threads: &[usize]) {
    let want = consensus_bits(brute_aamp_ostinato(set, m, scale));
    let refs: Vec<&[f64]> = set.iter().map(Vec::as_slice).collect();
    for &t in threads {
        let got = aamp_ostinato(&refs, scale, nz(m), nz(t)).map(|c| (c.radius, c.series, c.window));
        assert_eq!(consensus_bits(got), want, "threads {t} m {m} on {set:?}");
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    #[test]
    fn random_pairs_equal_the_brute_force(
        (a, m) in planted(),
        (b, _) in planted(),
        quarter in any::<bool>(),
    ) {
        let tick = if quarter { 0.25 } else { 1.0 };
        let fine = |s: &[f64]| s.iter().map(|x| x * tick).collect::<Vec<f64>>();
        assert_dist_equals_brute(&fine(&a), &fine(&b), m, scale(tick), &[1, 3]);
    }

    /// Series of exact ties: every order is a tie that only the side and window decide.
    #[test]
    fn tie_pairs_equal_the_brute_force(a in tie_series(), b in tie_series(), m in 1usize..=6) {
        assert_dist_equals_brute(&a, &b, m, scale(1.0), &[1, 2]);
    }

    /// Walks that jump between levels around the `u64` bound.
    #[test]
    fn jumping_pairs_equal_the_brute_force(a in walk_series(), b in walk_series(), m in 1usize..=6) {
        assert_dist_equals_brute(&a, &b, m, scale(1.0), &[1, 2]);
    }

    #[test]
    fn random_sets_equal_the_brute_force(
        sets in proptest::collection::vec(planted(), 1..6),
        quarter in any::<bool>(),
    ) {
        let m = sets[0].1;
        let tick = if quarter { 0.25 } else { 1.0 };
        let set: Vec<Vec<f64>> = sets
            .into_iter()
            .map(|(s, _)| s.iter().map(|x| x * tick).collect())
            .collect();
        assert_consensus_equals_brute(&set, m, scale(tick), &[1, 3]);
    }

    #[test]
    fn tie_sets_equal_the_brute_force(
        set in proptest::collection::vec(tie_series(), 1..6),
        m in 1usize..=6,
    ) {
        assert_consensus_equals_brute(&set, m, scale(1.0), &[1, 2]);
    }

    #[test]
    fn jumping_sets_equal_the_brute_force(
        set in proptest::collection::vec(walk_series(), 1..5),
        m in 1usize..=5,
    ) {
        assert_consensus_equals_brute(&set, m, scale(1.0), &[1, 2]);
    }
}

#[test]
fn a_planted_pattern_is_the_consensus_motif() {
    let pattern = [3.0, -2.0, 5.0, 0.0, 4.0];
    let mut set = Vec::new();
    for k in 0..4usize {
        let mut s: Vec<f64> = (0..40 + 7 * k)
            .map(|i| ((i * i * (k + 3)) % 11) as f64)
            .collect();
        let at = 5 + 6 * k;
        s[at..at + 5].copy_from_slice(&pattern);
        set.push(s);
    }
    let refs: Vec<&[f64]> = set.iter().map(Vec::as_slice).collect();
    let found = aamp_ostinato(&refs, scale(1.0), nz(5), nz(1)).unwrap();
    assert_eq!(found.radius, 0.0);
    assert_eq!(found.window, 5 + 6 * found.series);
    assert_consensus_equals_brute(&set, 5, scale(1.0), &[1, 4]);
}

#[test]
fn ties_between_windows_are_decided_by_the_join_order() {
    let one = scale(1.0);
    let raw: [&[f64]; 3] = [&[0.0, 0.0, 1.0], &[1.0, 1.0, 0.0], &[0.0, 1.0, 1.0, 1.0]];
    let found = aamp_ostinato(&raw, one, nz(3), nz(1)).unwrap();
    assert_eq!(
        (found.radius, found.series, found.window),
        (core::f64::consts::SQRT_2, 2, 0)
    );
    let normalized: [&[f64]; 3] = [
        &[1.0, 2.0, 0.0, 0.0, 2.0],
        &[2.0, 0.0, 1.0, 1.0],
        &[1.0, 2.0, 0.0],
    ];
    let set: Vec<(&[f64], Scale)> = normalized.iter().map(|&s| (s, one)).collect();
    let found = ostinato(&set, nz(3), nz(1)).unwrap();
    assert_eq!((found.series, found.window), (2, 0));
    assert!((found.radius - 2.449489742783178).abs() < 1e-12);
}

#[test]
fn the_metric_decides_the_distance() {
    let a: Vec<f64> = (0..30).map(|i| f64::from(i % 7)).collect();
    let b: Vec<f64> = a.iter().map(|x| x + 100.0).collect();
    let one = scale(1.0);
    let normalized = mpdist(&a, &b, nz(5), one, one, nz(1), Selection::default()).unwrap();
    let raw = aampdist(&a, &b, nz(5), one, nz(1), Selection::default()).unwrap();
    assert!(normalized < 1e-6, "{normalized}");
    assert!(raw > 200.0, "{raw}");
    let z = ostinato(&[(&a, one), (&b, one)], nz(5), nz(1)).unwrap();
    let plain = aamp_ostinato(&[&a, &b], one, nz(5), nz(1)).unwrap();
    assert!(z.radius < 1e-6, "{z:?}");
    assert!(plain.radius > 200.0, "{plain:?}");
}

#[test]
fn the_shared_scale_is_applied_once() {
    let a: Vec<f64> = (0..40).map(|i| f64::from((i * i + 3 * i) % 13)).collect();
    let b: Vec<f64> = (0..33).map(|i| f64::from((i * i + 5 * i) % 17)).collect();
    let quarter = |s: &[f64]| s.iter().map(|x| x * 0.25).collect::<Vec<f64>>();
    let (qa, qb) = (quarter(&a), quarter(&b));
    for pick in picks() {
        let unit = selection(pick)
            .and_then(|selection| aampdist(&a, &b, nz(6), scale(1.0), nz(1), selection));
        let fine = selection(pick)
            .and_then(|selection| aampdist(&qa, &qb, nz(6), scale(0.25), nz(1), selection));
        let (Some(unit), Some(fine)) = (unit, fine) else {
            assert_eq!((unit, fine), (None, None), "{pick:?}");
            continue;
        };
        assert!(unit > 0.0 && unit.is_finite(), "{pick:?} {unit}");
        assert_eq!(fine.to_bits(), (unit * 0.25).to_bits(), "{pick:?}");
    }
    let unit = aamp_ostinato(&[&a, &b, &a[3..]], scale(1.0), nz(6), nz(1)).unwrap();
    let small = aamp_ostinato(&[&qa, &qb, &qa[3..]], scale(0.25), nz(6), nz(1)).unwrap();
    assert!(unit.radius > 0.0 && unit.radius.is_finite(), "{unit:?}");
    assert_eq!((small.series, small.window), (unit.series, unit.window));
    assert_eq!(small.radius.to_bits(), (unit.radius * 0.25).to_bits());
}

#[test]
fn gaps_at_the_start_and_the_end_equal_the_brute_force() {
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
        for m in [2, 4, 7] {
            assert_dist_equals_brute(&gapped(40, 5), &walk(35, 6), m, scale(1.0), &[1, 3]);
            assert_dist_equals_brute(&gapped(36, 9), &gapped(41, 10), m, scale(1.0), &[1]);
            let three = [gapped(40, 5), walk(35, 6), gapped(38, 7)];
            assert_consensus_equals_brute(&three, m, scale(1.0), &[1, 2]);
            assert_consensus_equals_brute(&[gapped(36, 9), gapped(41, 10)], m, scale(1.0), &[1]);
        }
    }
    let nothing = vec![f64::NAN; 30];
    assert_dist_equals_brute(&nothing, &walk(30, 3), 3, scale(1.0), &[1]);
    assert_dist_equals_brute(&nothing, &nothing, 3, scale(1.0), &[1]);
    assert_consensus_equals_brute(&[nothing.clone(), walk(30, 3)], 3, scale(1.0), &[1]);
    let none = [nothing.clone(), nothing];
    assert_eq!(
        brute_aamp_ostinato(&none, 3, scale(1.0)),
        Some((f64::INFINITY, 0, 0))
    );
    assert_consensus_equals_brute(&none, 3, scale(1.0), &[1]);
}

#[test]
fn a_series_without_a_window_has_no_result() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let one = scale(1.0);
    for (x, y) in [(&a[..], &a[..3]), (&a[..3], &a[..]), (&a[..3], &a[..3])] {
        assert_eq!(
            aampdist(x, y, nz(4), one, nz(1), Selection::default()),
            None
        );
    }
    assert_eq!(
        aampdist(&a, &a, nz(4), one, nz(1), Selection::default()),
        Some(0.0)
    );
    assert_eq!(
        aampdist(&[], &a, nz(1), one, nz(1), Selection::default()),
        None
    );
    assert!(aamp_ostinato(&[], one, nz(2), nz(1)).is_none());
    assert!(aamp_ostinato(&[&a[..], &a[..3]], one, nz(4), nz(1)).is_none());
    let alone = aamp_ostinato(&[&a[..]], one, nz(4), nz(1)).unwrap();
    assert_eq!((alone.radius, alone.series, alone.window), (0.0, 0, 0));
}

#[test]
fn threads_change_no_bit_on_series_long_enough_to_run_threaded() {
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
    let tick = scale(0.01);
    let (a, b, c) = (
        walk(3200, 7, 100.0),
        walk(2700, 11, 100.5),
        walk(2000, 13, 99.0),
    );
    for pick in [Pick::Percentage(0.05), Pick::Rank(0), Pick::Percentage(1.0)] {
        let reference =
            selection(pick).and_then(|selection| aampdist(&a, &b, nz(24), tick, nz(1), selection));
        assert!(reference.is_some_and(f64::is_finite));
        for threads in [2, 3, 7, 16] {
            let got = selection(pick)
                .and_then(|selection| aampdist(&a, &b, nz(24), tick, nz(threads), selection));
            assert_eq!(bits(got), bits(reference), "{pick:?} threads {threads}");
        }
    }
    let set = [&a[..], &b[..], &c[..]];
    let reference = aamp_ostinato(&set, tick, nz(24), nz(1)).unwrap();
    assert!(reference.radius.is_finite());
    for threads in [2, 3, 7, 16] {
        let got = aamp_ostinato(&set, tick, nz(24), nz(threads)).unwrap();
        assert_eq!(
            got.radius.to_bits(),
            reference.radius.to_bits(),
            "threads {threads}"
        );
        assert_eq!(
            (got.series, got.window),
            (reference.series, reference.window)
        );
    }
}

#[test]
fn raw_bit_samples_and_extreme_selections_never_panic() {
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
    for round in 0..300 {
        let a = series((next() % 40) as usize);
        let b = series((next() % 40) as usize);
        let c = series((next() % 40) as usize);
        let m = nz(1 + (next() % 9) as usize);
        let tick = scales[round % 4];
        for pick in picks() {
            let _ =
                selection(pick).and_then(|selection| aampdist(&a, &b, m, tick, nz(2), selection));
        }
        let _ = aamp_ostinato(&[&a, &b, &c], tick, m, nz(2));
        let _ = aamp_ostinato(&[&a, &b], tick, m, nz(1));
    }
}
