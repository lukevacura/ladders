extern crate std;

use super::{
    MatchRules, MaxDistance, PatternLibrary, PatternStream, QueryId, RegisterError,
    distance_profile, find_matches, find_matches_nonnorm, find_matches_nonnorm_with,
    find_matches_with,
};
use crate::entry::Entry;
use crate::nearest::exact_oracle::{Exact, exactly_nearer};
use crate::quantize::{Scale, to_ticks};
use crate::stream::GrowingStream;
use core::num::NonZeroUsize;
use proptest::prelude::*;
use proptest::sample::Index;
use std::format;
use std::vec::Vec;
use testkit::generators::{MARKET_OFFSET, MARKET_TICK, market_series_and_m};

fn scale(value: f64) -> Scale {
    Scale::new(value).unwrap()
}

/// What one window must report, bit for bit: the nearest-neighbor definition written out
/// again from direct i128 sums over ticks, sharing no code with `Nearest`.
#[derive(Debug, PartialEq)]
enum Expected {
    None,
    Pair { rho: u64, d2: u64 },
}

fn expected(query: &[i64], window: &[Option<i64>]) -> Expected {
    let Some(window) = window.iter().copied().collect::<Option<Vec<i64>>>() else {
        return Expected::None;
    };
    let m = query.len() as i128;
    let stats = |t: &[i64]| {
        let s: i128 = t.iter().map(|&x| i128::from(x)).sum();
        let q: i128 = t.iter().map(|&x| i128::from(x) * i128::from(x)).sum();
        (s, m * q - s * s)
    };
    let ((sq, vq), (sx, vx)) = (stats(query), stats(&window));
    if vx >= 1 << 63 {
        return Expected::None;
    }
    let two_m = 2.0 * query.len() as f64;
    let (rho, d2) = match (vq == 0, vx == 0) {
        (true, true) => (1.0, 0.0),
        (false, false) => {
            let qt: i128 = query
                .iter()
                .zip(&window)
                .map(|(&a, &b)| i128::from(a) * i128::from(b))
                .sum();
            let n = m * qt - sq * sx;
            // The window is the row (`i`), the query its neighbor (`j`).
            let ri = 1.0 / libm::sqrt(vx as f64);
            let rj = 1.0 / libm::sqrt(vq as f64);
            let rho = (((n as i64 as f64) * ri) * rj).clamp(-1.0, 1.0);
            let d2 = if rho > 0.0 {
                let d = (vq * vx - n * n) as u128;
                let r = ri * rj;
                (((two_m * (d as f64)) * r) * r) / (1.0 + rho)
            } else {
                two_m * (1.0 - rho)
            };
            (rho, d2)
        }
        _ => (0.5, two_m * (1.0 - 0.5)),
    };
    Expected::Pair {
        rho: rho.to_bits(),
        d2: d2.to_bits(),
    }
}

/// The oracle's profile: quantized and gap-marked by `to_ticks` alone (a missing tick is
/// `None`; no fill is needed since any window with one is `NONE`).
fn oracle(query: &[f64], series: &[f64], series_scale: Scale) -> Vec<Expected> {
    let query_scale = series_scale.for_query(query).unwrap();
    let q: Vec<i64> = query
        .iter()
        .map(|&x| to_ticks(x, query_scale).unwrap())
        .collect();
    let x: Vec<Option<i64>> = series.iter().map(|&v| to_ticks(v, series_scale)).collect();
    x.windows(query.len()).map(|w| expected(&q, w)).collect()
}

fn check(query: &[f64], series: &[f64], series_scale: Scale) -> Result<usize, TestCaseError> {
    let profile = distance_profile(query, series, series_scale).unwrap();
    let want = oracle(query, series, series_scale);
    prop_assert_eq!(profile.len(), want.len());
    let mut finite = 0;
    for (i, (entry, want)) in profile.iter().zip(&want).enumerate() {
        match want {
            Expected::None => {
                prop_assert!(entry.index().is_none(), "window {i}");
                prop_assert_eq!(entry.squared_distance(), f64::INFINITY);
            }
            &Expected::Pair { rho, d2 } => {
                finite += 1;
                prop_assert_eq!(entry.index().get(), Some(i as u64));
                prop_assert_eq!(entry.rho().to_bits(), rho, "window {}", i);
                prop_assert_eq!(entry.squared_distance().to_bits(), d2, "window {}", i);
            }
        }
    }
    Ok(finite)
}

fn bits(entry: &Entry) -> (u64, u64, Option<u64>) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get(),
    )
}

/// Random, market-level (offset 1e8), flat-run and gapped series; the query is either a
/// window of the series (at its own, usually different, scale) or a fresh walk. Run by hand
/// so the count of cases that compared real entries can be asserted afterwards.
#[test]
fn matches_the_exact_oracle() {
    let strategy = (
        market_series_and_m(8..80),
        any::<Index>(),
        any::<bool>(),
        proptest::collection::vec(-1.0f64..=1.0, 200),
        prop_oneof![Just(0.0), Just(MARKET_OFFSET)],
    );
    let (cases, compared) = (core::cell::Cell::new(0usize), core::cell::Cell::new(0usize));
    proptest::test_runner::TestRunner::new(testkit::generators::seeded_default())
        .run(
            &strategy,
            |((series, m), pick, from_series, steps, level)| {
                let query: Vec<f64> = if from_series && series.len() >= m {
                    let start = pick.index(series.len() - m + 1);
                    series[start..start + m].to_vec()
                } else {
                    // A walk at its own level, not the series' tick grid.
                    steps[..m]
                        .iter()
                        .scan(level, |x, step| {
                            *x += step;
                            Some(*x)
                        })
                        .collect()
                };
                cases.set(cases.get() + 1);
                if query.iter().any(|x| !x.is_finite()) {
                    prop_assert!(distance_profile(&query, &series, scale(MARKET_TICK)).is_none());
                } else {
                    let finite = check(&query, &series, scale(MARKET_TICK))?;
                    compared.set(compared.get() + usize::from(finite > 0));
                }
                Ok(())
            },
        )
        .unwrap();
    let (cases, compared) = (cases.get(), compared.get());
    assert!(
        cases >= 100 && compared * 10 >= cases * 7,
        "{compared} of {cases}"
    );
}

#[test]
fn rejects_empty_and_non_finite_queries() {
    let s = [1.0, 2.0, 3.0];
    for q in [&[][..], &[1.0, f64::NAN], &[f64::INFINITY, 1.0]] {
        assert!(distance_profile(q, &s, scale(1.0)).is_none());
    }
    assert_eq!(
        distance_profile(&[1.0, 2.0, 3.0, 4.0], &s, scale(1.0))
            .unwrap()
            .len(),
        0
    );
}

/// The query's own scale: a shape 16 times finer than the series' tick still matches
/// exactly. Quantized at the series' scale it would be all zeros, a constant.
#[test]
fn query_finer_than_the_series_tick_matches_exactly() {
    let query = [0.0, 0.125, 0.0625, 0.25];
    let series = [5.0, 7.0, 6.0, 9.0, 5.0, 7.0, 6.0, 9.0, 3.0];
    let profile = distance_profile(&query, &series, scale(1.0)).unwrap();
    assert_eq!(profile.len(), 6);
    for (i, entry) in profile.iter().enumerate() {
        if i % 4 == 0 {
            assert_eq!(entry.squared_distance().to_bits(), 0, "window {i}");
        } else {
            assert!(entry.squared_distance() > 0.5, "window {i}");
        }
    }
    check(&query, &series, scale(1.0)).unwrap();
}

/// A planted repeat among varied windows, at market level: non-trivial, every entry
/// finite and distinct enough to tell an off-by-one alignment apart.
#[test]
fn planted_repeat_is_the_unique_exact_match() {
    let base = 1.23e8;
    let steps = [
        0.0, 0.03, -0.02, 0.07, 0.01, -0.05, 0.04, 0.0, 0.09, -0.01, 0.02,
    ];
    let series: Vec<f64> = steps.iter().map(|s| base + s).collect();
    let query = &series[3..8];
    let profile = distance_profile(query, &series, scale(0.01)).unwrap();
    assert_eq!(profile.len(), 7);
    assert_eq!(check(query, &series, scale(0.01)).unwrap(), 7);
    let nearest = profile
        .iter()
        .min_by(|a, b| a.squared_distance().total_cmp(&b.squared_distance()))
        .unwrap();
    assert_eq!(nearest.index().get(), Some(3));
    // On the stream's 0.01 grid at market level, so the repeat is the same ticks: exactly 0.
    assert_eq!(nearest.squared_distance().to_bits(), 0);
    assert!(
        profile
            .iter()
            .filter(|e| e.squared_distance() > 1e-3)
            .count()
            == 6
    );
}

/// A series that starts with gaps: the windows holding one are NONE (a leading gap is
/// never counted as a present tick).
#[test]
fn leading_gaps_make_the_first_windows_none() {
    let series = [f64::NAN, f64::NAN, 3.0, 5.0, 4.0, 9.0, 2.0];
    let q = [3.0, 5.0, 4.0];
    let profile = distance_profile(&q, &series, scale(1.0)).unwrap();
    assert_eq!(profile.len(), 5);
    assert!(profile[0].index().is_none() && profile[1].index().is_none());
    assert_eq!(profile[2].squared_distance(), 0.0);
    assert!(profile[3].index().get() == Some(3) && profile[4].index().get() == Some(4));
    assert_eq!(check(&q, &series, scale(1.0)).unwrap(), 3);
    let flat = distance_profile(&[2.0, 2.0, 2.0], &[f64::NAN, 2.0, 2.0, 2.0], scale(1.0));
    assert!(flat.unwrap()[0].index().is_none());
}

#[test]
fn flat_and_gapped_series() {
    let series = [4.0, 4.0, 4.0, 4.0, 1.0, 2.0, f64::NAN, 2.0, 1.0, 2.0, 1.0];
    let flat = distance_profile(&[7.0, 7.0, 7.0], &series, scale(1.0)).unwrap();
    assert_eq!(flat[0].squared_distance(), 0.0);
    assert_eq!(flat[1].squared_distance(), 0.0);
    assert_eq!(flat[2].squared_distance(), 3.0); // one constant, one valid: m
    assert!(flat[5].index().is_none() && flat[6].index().is_none());
    assert!(!flat[8].index().is_none());
    assert_eq!(check(&[7.0, 7.0, 7.0], &series, scale(1.0)).unwrap(), 6);
    assert_eq!(check(&[1.0, 2.0, 1.0], &series, scale(1.0)).unwrap(), 6);
}

/// Literal bits on a generic case, so a change in operation order fails.
#[test]
fn generic_case_pins_bits() {
    let series = [1.0, 8.0, 2.0, 50.0, 60.0, 10.0, 3.0, 7.0];
    let profile = distance_profile(&[2.0, 9.0, 4.0], &series, scale(1.0)).unwrap();
    let got: Vec<_> = profile.iter().map(bits).collect();
    let pinned = [
        (4607083179624806880, 4589427901073270209, Some(0)),
        (13823347914779848050, 4620749366151629957, Some(1)),
        (4603573505625062115, 4612595788562450346, Some(2)),
        (4601225308237968276, 4615027951442082128, Some(3)),
        (13827293190695215183, 4621722043135972382, Some(4)),
        (13830439019273615692, 4622901728852872572, Some(5)),
    ];
    assert_eq!(got, pinned);
}

fn library(m: usize, stream_scale: f64) -> PatternLibrary {
    PatternLibrary::new(NonZeroUsize::new(m).unwrap(), scale(stream_scale))
}

#[test]
fn registration_rejects_wrong_length_and_non_finite() {
    let mut lib = library(3, 1.0);
    assert_eq!(lib.register(&[1.0, 2.0]), Err(RegisterError::WrongLength));
    assert_eq!(
        lib.register(&[1.0, 2.0, 3.0, 4.0]),
        Err(RegisterError::WrongLength)
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            lib.register(&[1.0, bad, 3.0]),
            Err(RegisterError::NonFinite)
        );
    }
    assert!(lib.is_empty());
    let a = lib.register(&[1.0, 2.0, 4.0]).unwrap();
    let b = lib.register(&[4.0, 2.0, 1.0]).unwrap();
    assert_eq!((a.get(), b.get(), lib.len()), (0, 1, 2));
    assert_eq!(lib.m().get(), 3);
    assert!(lib.distance_profile(QueryId(2), &[1.0; 4]).is_none());
}

/// Each registered query's profile equals `distance_profile` on the same query, on bits,
/// and the queries' profiles differ from one another.
#[test]
fn library_profiles_equal_distance_profile_on_bits() {
    let series: Vec<f64> = [3.0, 9.0, 4.0, 4.5, 8.0, 1.0, 7.0, 2.0, 2.5, 6.0, 0.5, 5.0]
        .iter()
        .map(|v| 1e6 + v * 0.01)
        .collect();
    let queries = [
        [1e6 + 0.04, 1e6 + 0.09, 1e6 + 0.03, 1e6 + 0.05],
        [0.0, 1.0, 0.5, 2.0],
        [7.0, 7.0, 7.0, 7.0],
    ];
    let mut lib = library(4, 0.01);
    let ids: Vec<QueryId> = queries.iter().map(|q| lib.register(q).unwrap()).collect();
    let mut profiles = Vec::new();
    for (id, q) in ids.iter().zip(&queries) {
        let from_lib = lib.distance_profile(*id, &series).unwrap();
        let direct = distance_profile(q, &series, scale(0.01)).unwrap();
        assert_eq!(from_lib.len(), 9);
        assert_eq!(
            from_lib.iter().map(bits).collect::<Vec<_>>(),
            direct.iter().map(bits).collect::<Vec<_>>()
        );
        assert!(direct.iter().all(|e| e.squared_distance().is_finite()));
        profiles.push(direct.iter().map(bits).collect::<Vec<_>>());
    }
    assert_ne!(profiles[0], profiles[1]);
    assert_ne!(profiles[1], profiles[2]);
}

/// Every entry the stream emits, per query, over all of `series`.
fn stream_all(queries: &[Vec<f64>], series: &[f64], series_scale: Scale) -> Vec<Vec<Entry>> {
    let mut lib = library(queries[0].len(), series_scale.get());
    for q in queries {
        lib.register(q).unwrap();
    }
    let mut stream = PatternStream::new(lib);
    let mut per_query = std::vec![Vec::new(); queries.len()];
    for (k, &x) in series.iter().enumerate() {
        let out = stream.push(x);
        if k + 1 < queries[0].len() {
            assert!(out.is_empty());
        } else {
            assert_eq!(out.len(), queries.len());
        }
        for (acc, e) in per_query.iter_mut().zip(out) {
            acc.push(*e);
        }
    }
    per_query
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// The stream equals `distance_profile` on bits, and a prefix of the stream is a prefix
    /// of the whole (causality), on random, market-level, flat and gapped series.
    #[test]
    fn stream_equals_profile_and_is_causal(
        (series, m) in market_series_and_m(8..80),
        pick in any::<Index>(),
        steps in proptest::collection::vec(-1.0f64..=1.0, 200),
        cut in any::<Index>(),
    ) {
        let start = pick.index(series.len().saturating_sub(m) + 1);
        let from_series = series[start.min(series.len())..].iter().take(m).copied().collect::<Vec<_>>();
        let walk: Vec<f64> = steps[..m].iter().scan(1e8, |x, s| { *x += s; Some(*x) }).collect();
        let queries: Vec<Vec<f64>> = [from_series, walk]
            .into_iter()
            .filter(|q| q.len() == m && q.iter().all(|x| x.is_finite()))
            .collect();
        // The walk is always one of them.
        let full = stream_all(&queries, &series, scale(MARKET_TICK));
        for (q, got) in queries.iter().zip(&full) {
            let want = distance_profile(q, &series, scale(MARKET_TICK)).unwrap();
            prop_assert_eq!(got.iter().map(bits).collect::<Vec<_>>(), want.iter().map(bits).collect::<Vec<_>>());
        }
        let k = cut.index(series.len() + 1);
        let prefix = stream_all(&queries, &series[..k], scale(MARKET_TICK));
        for (p, f) in prefix.iter().zip(&full) {
            prop_assert_eq!(p.iter().map(bits).collect::<Vec<_>>(), f[..p.len()].iter().map(bits).collect::<Vec<_>>());
        }
    }
}

#[test]
fn stream_reports_start_index_none_on_gaps_and_reuses_its_buffer() {
    let series = [1.0, 2.0, 4.0, f64::NAN, 3.0, 5.0, 4.0, 8.0, 7.0];
    let mut lib = library(3, 1.0);
    lib.register(&[1.0, 2.0, 4.0]).unwrap();
    lib.register(&[4.0, 2.0, 1.0]).unwrap();
    let mut stream = PatternStream::new(lib);
    let mut pointer = None;
    let mut seen = Vec::new();
    for &x in &series {
        let out = stream.push(x);
        if !out.is_empty() {
            let p = out.as_ptr();
            assert_eq!(*pointer.get_or_insert(p), p);
        }
        seen.push(out.to_vec());
    }
    assert!(seen[0].is_empty() && seen[1].is_empty());
    // Windows 1..=3 hold the gap.
    assert_eq!(seen[2][0].squared_distance(), 0.0);
    assert_eq!(seen[2][0].index().get(), Some(0));
    for window in 1..=3 {
        assert!(
            seen[window + 2].iter().all(|e| e.index().is_none()),
            "{window}"
        );
    }
    assert_eq!(seen[6].len(), 2);
    assert_eq!(seen[6][0].index().get(), Some(4));
    assert!(seen[6][0].squared_distance() > 0.0 && seen[6][0].squared_distance().is_finite());
    assert_eq!(seen[8][1].index().get(), Some(6));
    assert_ne!(seen[8][0].squared_distance(), seen[8][1].squared_distance());
}

/// The stream's tick is coarse; a finer query still matches exactly.
#[test]
fn stream_query_finer_than_the_stream_tick_matches_exactly() {
    let queries = [std::vec![0.0, 0.125, 0.0625, 0.25]];
    let series = [5.0, 7.0, 6.0, 9.0, 5.0, 7.0, 6.0, 9.0, 3.0];
    let got = stream_all(&queries, &series, scale(1.0));
    assert_eq!(got[0].len(), 6);
    for (i, e) in got[0].iter().enumerate() {
        assert_eq!(e.squared_distance() == 0.0, i % 4 == 0, "window {i}");
    }
}

/// A near-repeat and an exact repeat on the stream's grid: the pattern entry equals the
/// `GrowingStream` left-profile entry for the same two windows on bits (the newer window is
/// the row in both), and the exact repeat scores exactly 0.
#[test]
fn on_grid_entries_equal_the_growing_streams_on_bits() {
    let m = 12;
    let tick = |k: i64| 1.23e8 + k as f64 * 0.01;
    let a: Vec<i64> = std::vec![0, 3, 1, 6, 2, 8, 5, 4, 9, 3, 7, 2];
    let junk: Vec<i64> = (0..20).map(|i| (i * i * 7 + 3 * i) % 41 - 10).collect();
    for (name, b) in [
        ("near", [0, 3, 2, 6, 2, 8, 4, 4, 9, 3, 7, 3]),
        ("exact", [0, 3, 1, 6, 2, 8, 5, 4, 9, 3, 7, 2]),
    ] {
        let ticks: Vec<i64> = a.iter().chain(&junk).chain(&b).copied().collect();
        let series: Vec<f64> = ticks.iter().map(|&k| tick(k)).collect();
        let query: Vec<f64> = a.iter().map(|&k| tick(k)).collect();
        let mut growing = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale(0.01));
        let last = series
            .iter()
            .filter_map(|&x| growing.update(x))
            .last()
            .unwrap()
            .1;
        assert_eq!(last.index().get(), Some(0), "{name}: the pair is A and B");
        let ours = *distance_profile(&query, &series, scale(0.01))
            .unwrap()
            .last()
            .unwrap();
        assert_eq!(ours.rho().to_bits(), last.rho().to_bits(), "{name}");
        assert_eq!(
            ours.squared_distance().to_bits(),
            last.squared_distance().to_bits(),
            "{name}"
        );
        assert_eq!(name == "exact", ours.squared_distance() == 0.0);
        assert!(name == "exact" || ours.squared_distance() > 0.1);
    }
}

/// A sine template is off the 0.01 grid: it keeps its own scale (not rounded to the
/// stream's), so even the series built by rounding it does not score exactly 0.
#[test]
fn off_grid_template_keeps_its_own_scale() {
    let m = 16;
    let sine: Vec<f64> = (0..m)
        .map(|i| libm::sin(i as f64 * 0.7) * 3.0 + 50.0)
        .collect();
    let stream = scale(0.01);
    assert!(!stream.holds(&sine));
    let own = stream.for_query(&sine).unwrap();
    assert_ne!(own, stream);
    let rounded: Vec<f64> = sine.iter().map(|v| (v / 0.01).round() * 0.01).collect();
    let series: Vec<f64> = (0..10)
        .map(|i| 50.0 + (i * 5 % 7) as f64 * 0.31)
        .chain(rounded)
        .chain((0..10).map(|i| 49.0 + (i * 3 % 5) as f64 * 0.47))
        .collect();
    let profile = distance_profile(&sine, &series, stream).unwrap();
    assert_eq!(check(&sine, &series, stream).unwrap(), profile.len());
    let best = profile
        .iter()
        .min_by(|a, b| a.squared_distance().total_cmp(&b.squared_distance()))
        .unwrap();
    assert_eq!(best.index().get(), Some(10));
    assert!(best.squared_distance() > 0.0 && best.squared_distance() < 0.05);
}

/// 0.001-resolution data against a 0.01 stream is not rounded to its ticks: the shape
/// matches the stream's coarse window, where rounding would flatten it to a constant.
#[test]
fn finer_tick_query_on_a_coarser_stream_is_not_rounded() {
    let query: Vec<f64> = [0, 2, 1, 4, 3].iter().map(|&k| k as f64 * 0.001).collect();
    let series: Vec<f64> = [9, 5, 7, 6, 9, 8, 2, 9]
        .iter()
        .map(|&k| k as f64 * 0.01)
        .collect();
    let profile = distance_profile(&query, &series, scale(0.01)).unwrap();
    assert_eq!(check(&query, &series, scale(0.01)).unwrap(), 4);
    assert_eq!(profile[1].index().get(), Some(1));
    assert!(profile[1].squared_distance() < 1e-12);
    assert!(profile[0].squared_distance() > 0.1 && profile[2].squared_distance() > 0.1);
}

// ---- find_matches ----

fn exact_of(query: &[i64], window: &[Option<i64>]) -> Option<Exact> {
    let window = window.iter().copied().collect::<Option<Vec<i64>>>()?;
    let m = query.len() as i128;
    let stats = |t: &[i64]| {
        let s: i128 = t.iter().map(|&x| i128::from(x)).sum();
        let q: i128 = t.iter().map(|&x| i128::from(x) * i128::from(x)).sum();
        (s, m * q - s * s)
    };
    let ((sq, vq), (sx, vx)) = (stats(query), stats(&window));
    if vx >= 1 << 63 {
        return None;
    }
    Some(match (vq == 0, vx == 0) {
        (true, true) => Exact {
            sign: 1,
            num: 1,
            den: 1,
        },
        (false, false) => {
            let qt: i128 = query
                .iter()
                .zip(&window)
                .map(|(&a, &b)| i128::from(a) * i128::from(b))
                .sum();
            let n = m * qt - sq * sx;
            Exact {
                sign: n.signum() as i8,
                num: (n * n) as u128,
                den: (vq * vx) as u128,
            }
        }
        _ => Exact {
            sign: 1,
            num: 1,
            den: 4,
        },
    })
}

fn oracle_exact(query: &[f64], series: &[f64], series_scale: Scale) -> Vec<Option<Exact>> {
    let query_scale = series_scale.for_query(query).unwrap();
    let q: Vec<i64> = query
        .iter()
        .map(|&x| to_ticks(x, query_scale).unwrap())
        .collect();
    let x: Vec<Option<i64>> = series.iter().map(|&v| to_ticks(v, series_scale)).collect();
    x.windows(query.len()).map(|w| exact_of(&q, w)).collect()
}

/// The greedy selection written plainly: among the windows within the threshold (by their
/// f64 distance), the exactly nearest remaining one (ties to the smaller start), then blank
/// the `ceil(m/4)` starts each side, repeat until `limit`.
fn reference_matches(
    query: &[f64],
    series: &[f64],
    series_scale: Scale,
    max_distance: MaxDistance,
    limit: Option<usize>,
) -> Vec<(u64, u64)> {
    let d: Vec<f64> = oracle(query, series, series_scale)
        .iter()
        .map(|e| match e {
            Expected::None => f64::INFINITY,
            Expected::Pair { d2, .. } => libm::sqrt(f64::from_bits(*d2)),
        })
        .collect();
    let exact = oracle_exact(query, series, series_scale);
    let finite: Vec<f64> = d.iter().copied().filter(|x| x.is_finite()).collect();
    let threshold = match max_distance {
        MaxDistance::Value(v) => v,
        MaxDistance::Default if finite.is_empty() => f64::NAN,
        MaxDistance::Default => {
            let n = finite.len() as f64;
            let mean = finite.iter().fold(0.0, |a, x| a + x) / n;
            let var = finite.iter().fold(0.0, |a, x| a + (x - mean) * (x - mean)) / n;
            let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
            (mean - 2.0 * libm::sqrt(var)).max(min)
        }
    };
    let mut alive: Vec<bool> = d.iter().map(|&x| x.is_finite() && x <= threshold).collect();
    let zone = crate::exclusion::zone(query.len());
    let mut out = Vec::new();
    while out.len() < limit.unwrap_or(usize::MAX) {
        let mut best: Option<usize> = None;
        for i in 0..d.len() {
            if alive[i] && best.is_none_or(|b| exactly_nearer(exact[i].unwrap(), exact[b].unwrap()))
            {
                best = Some(i);
            }
        }
        let Some(b) = best else { break };
        out.push((b as u64, d[b].to_bits()));
        for (i, a) in alive.iter_mut().enumerate() {
            if i.abs_diff(b) <= zone {
                *a = false;
            }
        }
    }
    out
}

fn found(
    query: &[f64],
    series: &[f64],
    series_scale: Scale,
    max_distance: MaxDistance,
    limit: Option<usize>,
) -> Vec<(u64, u64)> {
    find_matches(query, series, series_scale, max_distance, limit)
        .unwrap()
        .iter()
        .map(|(start, e)| {
            assert_eq!(e.index().get(), Some(*start));
            (*start, libm::sqrt(e.squared_distance()).to_bits())
        })
        .collect()
}

#[test]
fn matches_equal_the_plain_greedy_reference() {
    let strategy = (
        market_series_and_m(8..80),
        any::<Index>(),
        any::<bool>(),
        proptest::collection::vec(-1.0f64..=1.0, 200),
        0usize..4,
        proptest::option::of(0usize..6),
        0.0f64..2.0,
    );
    let (cases, nonempty) = (core::cell::Cell::new(0usize), core::cell::Cell::new(0usize));
    proptest::test_runner::TestRunner::new(testkit::generators::seeded_default())
        .run(
            &strategy,
            |((series, m), pick, from_series, steps, mode, limit, factor)| {
                if series.len() < m {
                    return Ok(());
                }
                let sc = scale(MARKET_TICK);
                let start = pick.index(series.len() - m + 1);
                let query: Vec<f64> = if from_series {
                    series[start..start + m].to_vec()
                } else {
                    steps[..m]
                        .iter()
                        .scan(0.0, |x, step| {
                            *x += step;
                            Some(*x)
                        })
                        .collect()
                };
                if query.iter().any(|x| !x.is_finite()) {
                    return Ok(());
                }
                let max_distance = match mode {
                    0 | 1 => MaxDistance::Default,
                    2 => MaxDistance::Value(factor * libm::sqrt(2.0 * m as f64)),
                    _ => MaxDistance::Value(f64::INFINITY),
                };
                let want = reference_matches(&query, &series, sc, max_distance, limit);
                let got = found(&query, &series, sc, max_distance, limit);
                prop_assert_eq!(&got, &want);
                cases.set(cases.get() + 1);
                nonempty.set(nonempty.get() + usize::from(!got.is_empty()));
                Ok(())
            },
        )
        .unwrap();
    assert!(
        cases.get() >= 100 && nonempty.get() * 10 >= cases.get() * 6,
        "{} of {}",
        nonempty.get(),
        cases.get()
    );
}

#[test]
fn matches_reject_bad_input_and_respect_limits() {
    let sc = scale(1.0);
    let s = [
        1.0, 2.0, 4.0, 3.0, 1.0, 2.0, 4.0, 3.0, 9.0, 1.0, 2.0, 4.0, 3.0,
    ];
    let q = [1.0, 2.0, 4.0, 3.0];
    for bad in [&[][..], &[1.0, f64::NAN], &[f64::INFINITY, 1.0]] {
        assert!(find_matches(bad, &s, sc, MaxDistance::Default, None).is_none());
    }
    assert!(find_matches(&q, &s[..3], sc, MaxDistance::Default, None).is_none());
    let all = find_matches(&q, &s, sc, MaxDistance::Value(0.0), None).unwrap();
    let starts: Vec<u64> = all.iter().map(|m| m.0).collect();
    assert_eq!(starts, [0, 4, 9]);
    assert!(all.iter().all(|m| m.1.squared_distance() == 0.0));
    let two = find_matches(&q, &s, sc, MaxDistance::Value(0.0), Some(2)).unwrap();
    assert_eq!(two.len(), 2);
    let none = find_matches(&q, &s, sc, MaxDistance::Value(0.0), Some(0)).unwrap();
    assert!(none.is_empty());
    let nan = find_matches(&q, &s, sc, MaxDistance::Value(f64::NAN), None).unwrap();
    assert!(nan.is_empty());
    // Starts 0 and 1 are within the zone (m = 4, zone 1) of each other: only one survives.
    let flat = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let q = [3.0, 3.0, 3.0, 3.0];
    let starts: Vec<u64> = find_matches(&q, &flat, sc, MaxDistance::Default, None)
        .unwrap()
        .iter()
        .map(|m| m.0)
        .collect();
    assert_eq!(starts, [0, 2, 4]);
}

/// Windows with identical shapes (a window and a scaled, shifted copy) are exactly equally
/// near; each has its own rounded reciprocal root, so their f64 distances can differ in the
/// last bit. They must still come out in index order.
#[test]
fn scaled_copies_come_out_in_index_order() {
    let a = [-16.0, -1.0, 14.0, -16.0, -19.0, -11.0, 7.0];
    let mut series = a.to_vec();
    series.extend(a.iter().map(|x| 3.0 * x + 24.0));
    let query = [-13.0, 7.0, 10.0, -11.0, -14.0, -11.0, 16.0];
    let got = find_matches(
        &query,
        &series,
        scale(1.0),
        MaxDistance::Value(f64::INFINITY),
        None,
    )
    .unwrap();
    let starts: Vec<u64> = got.iter().map(|m| m.0).collect();
    assert_eq!(starts[..2], [0, 7]);
    let limited = find_matches(
        &query,
        &series,
        scale(1.0),
        MaxDistance::Value(f64::INFINITY),
        Some(1),
    )
    .unwrap();
    assert_eq!(limited[0].0, 0);
}

/// Random integer shapes repeated as scaled, shifted (and sometimes negated) copies, so
/// many windows are exactly equally near: the result must equal the exact-rational reference.
#[test]
fn scaled_copies_match_the_exact_reference() {
    let strategy = (
        proptest::collection::vec(-20i32..20, 5..9),
        proptest::collection::vec(-20i32..20, 5..9),
        proptest::collection::vec((-4i32..5, -30i32..30), 1..4),
        proptest::option::of(1usize..5),
    );
    proptest::test_runner::TestRunner::new(testkit::generators::seeded_default())
        .run(&strategy, |(base, query, copies, limit)| {
            let m = base.len().min(query.len());
            let mut series: Vec<f64> = base[..m].iter().map(|&x| f64::from(x)).collect();
            for (k, c) in copies {
                let k = if k == 0 { 3 } else { k };
                series.extend(base[..m].iter().map(|&x| f64::from(k * x + c)));
            }
            let query: Vec<f64> = query[..m].iter().map(|&x| f64::from(x)).collect();
            let sc = scale(1.0);
            let all = MaxDistance::Value(f64::INFINITY);
            prop_assert_eq!(
                found(&query, &series, sc, all, limit),
                reference_matches(&query, &series, sc, all, limit)
            );
            Ok(())
        })
        .unwrap();
}

#[test]
fn a_series_with_no_finite_window_has_no_matches() {
    let s = [f64::NAN; 10];
    let r = find_matches(&[1.0, 2.0, 3.0], &s, scale(1.0), MaxDistance::Default, None);
    assert!(r.unwrap().is_empty());
}

/// STUMPY's `match` on benchmark slices (`scripts/match_stumpy.py`): the same indices and
/// counts. Distances are compared to 0.1 only: STUMPY's own distance for an exact repeat is
/// 0.043 (cancellation in its sliding statistics), and series quantized to their tick differ
/// from the raw floats by up to 6e-4. A length difference would be allowed only for a
/// distance at an explicit threshold's edge; none occurs in these fixtures.
#[test]
fn matches_agree_with_stumpy_match() {
    fn hex(tokens: &[&str]) -> Vec<f64> {
        tokens
            .iter()
            .map(|t| f64::from_bits(u64::from_str_radix(t, 16).unwrap()))
            .collect()
    }
    // timing-ok: a compile-time path, fixed per build
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "matches"))
        .collect();
    files.sort();
    assert!(files.len() >= 3);
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let field = |key: &str| -> Vec<&str> {
            text.lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix(' '))
                .unwrap()
                .split_ascii_whitespace()
                .collect()
        };
        let sc = scale(hex(&field("scale"))[0]);
        let (series, query) = (hex(&field("series")), hex(&field("query")));
        for case in ["default", "limited", "value"] {
            let max = match field(&format!("{case}_max"))[0] {
                "default" => MaxDistance::Default,
                h => MaxDistance::Value(hex(&[h])[0]),
            };
            let limit = field(&format!("{case}_limit"))[0].parse().ok();
            let theirs: Vec<(u64, f64)> = field(&format!("{case}_i"))
                .iter()
                .map(|t| t.parse().unwrap())
                .zip(hex(&field(&format!("{case}_d"))))
                .collect();
            let ours = find_matches(&query, &series, sc, max, limit).unwrap();
            let name = format!("{} {case}", path.display());
            let tol = 0.1;
            let n = ours.len().min(theirs.len());
            for k in 0..n {
                assert_eq!(ours[k].0, theirs[k].0, "{name} match {k}");
                let d = libm::sqrt(ours[k].1.squared_distance());
                assert!(
                    (d - theirs[k].1).abs() <= tol,
                    "{name} match {k}: {d} vs {}",
                    theirs[k].1
                );
            }
            // A length difference is allowed only for a distance at an explicit threshold's
            // edge; the default threshold must agree here outright.
            if ours.len() != theirs.len() {
                let extra = if ours.len() > n {
                    libm::sqrt(ours[n].1.squared_distance())
                } else {
                    theirs[n].1
                };
                let MaxDistance::Value(edge) = max else {
                    panic!("{name}: {} matches against {}", ours.len(), theirs.len());
                };
                assert!((extra - edge).abs() <= tol, "{name}");
            }
        }
    }
}

/// STUMPY forces the query window's own distance to 0 (`query_idx`), so a window with no distance
/// at all still matches itself and its zero counts in the default threshold: with every other
/// distance missing, the threshold is 0 and the window is the one match, under both metrics.
fn assert_a_distanceless_window_matches_itself<M: super::Measure>() {
    let mut series: Vec<f64> = (0..30).map(|k| f64::from((k * 7 % 11) as u8)).collect();
    series[2] = f64::NAN;
    let m = NonZeroUsize::new(5).unwrap();
    let ticked = super::TickedSeries::<M>::new(&series, m, scale(1.0));
    let matches = ticked.matches_of_window(0, MaxDistance::Default, None, 0.0);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, 0);
    assert_eq!(M::index(&matches[0].1), Some(0));
    assert_eq!(M::distance(&matches[0].1, scale(1.0)), 0.0);
}

#[test]
fn a_window_with_no_distance_still_matches_itself() {
    assert_a_distanceless_window_matches_itself::<crate::znorm::ZNorm>();
    assert_a_distanceless_window_matches_itself::<crate::nonnorm::NonNormalized>();
}

fn repeated_series() -> Vec<f64> {
    let mut state = 12345_u64;
    (0..80)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f64::from((state >> 40) as u32 % 64)
        })
        .collect()
}

/// `atol` widens the threshold by exactly its size: a distance one `atol` past a given
/// `max_distance` matches with it and not without.
#[test]
fn atol_is_added_to_the_threshold() {
    let series = repeated_series();
    let query = &series[0..10];
    let all = find_matches(
        query,
        &series,
        scale(1.0),
        MaxDistance::Value(f64::MAX),
        None,
    )
    .expect("a query of the series");
    let (_, third) = all[2];
    let edge = third.squared_distance().sqrt();
    let rules = |atol| MatchRules {
        atol,
        query_idx: None,
    };
    let count = |max: f64, atol| {
        find_matches_with(
            query,
            &series,
            scale(1.0),
            MaxDistance::Value(max),
            None,
            rules(atol),
        )
        .unwrap()
        .len()
    };
    let slack = 1e-6;
    assert_eq!(count(edge, 0.0), 3);
    assert_eq!(count(edge - slack, 0.0), 2);
    assert_eq!(count(edge - slack, slack), 3);
    assert_eq!(count(edge - 2.0 * slack, slack), 2);
}

/// `query_idx` takes that window first, at distance zero, and a start beyond the windows is
/// refused; the non-normalized search does the same.
#[test]
fn query_idx_takes_the_window_first() {
    let series = repeated_series();
    let at = 20;
    let query = &series[at..at + 10];
    let rules = MatchRules {
        atol: 0.0,
        query_idx: Some(at),
    };
    let found = find_matches_with(
        query,
        &series,
        scale(1.0),
        MaxDistance::Default,
        None,
        rules,
    )
    .unwrap();
    assert_eq!(found[0].0, at as u64);
    assert_eq!(found[0].1.squared_distance(), 0.0);
    let found = find_matches_nonnorm_with(
        query,
        &series,
        scale(1.0),
        MaxDistance::Default,
        None,
        rules,
    )
    .unwrap();
    assert_eq!(found[0].0, at as u64);
    let beyond = MatchRules {
        atol: 0.0,
        query_idx: Some(series.len() - 9),
    };
    assert!(
        find_matches_with(
            query,
            &series,
            scale(1.0),
            MaxDistance::Default,
            None,
            beyond
        )
        .is_none()
    );
    assert!(
        find_matches_nonnorm_with(
            query,
            &series,
            scale(1.0),
            MaxDistance::Default,
            None,
            beyond
        )
        .is_none()
    );
    assert_eq!(
        find_matches_nonnorm(query, &series, scale(1.0), MaxDistance::Default, None)
            .unwrap()
            .len(),
        find_matches_nonnorm_with(
            query,
            &series,
            scale(1.0),
            MaxDistance::Default,
            None,
            MatchRules::default()
        )
        .unwrap()
        .len()
    );
}
