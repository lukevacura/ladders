#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Streams equal the exact reference on bits under every forced route, and their route counters
//! follow the forced route.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::narrow::{Path, RouteCounts};
use ladde::quantize::Scale;
use ladde::route::{Fired, RouteForce, forced_route};
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::*;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn growing_equals_the_exact_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
    ) {
        let expected = stream_profile_bits(&samples(&series, scale), m);
        for force in forces(seed) {
            let _forced = forced_route(force);
            let mut one_by_one = GrowingStream::new(nz(m), scale);
            let rows: Vec<_> = series.iter().filter_map(|&x| one_by_one.update(x)).collect();
            prop_assert_eq!(&stream_bits(&rows), &expected, "{:?} update", force);
            if force.wide {
                prop_assert_eq!(one_by_one.path(), Path::Wide);
            }
            for chunk in [1, 2, 5, series.len().max(1)] {
                let mut stream = GrowingStream::new(nz(m), scale);
                let rows = chunked(&mut stream, &series, chunk);
                prop_assert_eq!(&stream_bits(&rows), &expected, "{:?} chunk {}", force, chunk);
            }
        }
    }

    #[test]
    fn bounded_equals_the_windowed_exact_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..40,
    ) {
        let samples = samples(&series, scale);
        let w = m + extra;
        let mut exact = vec![None; series.len()];
        for force in forces(seed) {
            let _forced = forced_route(force);
            let new = || BoundedStream::new(nz(m), nz(w), scale).expect("w is at least m");
            let mut one_by_one = new();
            let rows: Vec<_> = series.iter().filter_map(|&x| one_by_one.update(x)).collect();
            if force.wide {
                prop_assert_eq!(one_by_one.path(), Path::Wide);
            }
            let by_update = stream_bits(&rows);
            for &got in &by_update {
                prop_assert_eq!(
                    got,
                    exact_row(&mut exact, &samples, m, w, got.0 as usize),
                    "{:?} w {}", force, w
                );
            }
            for chunk in [1, 2, 5, series.len().max(1)] {
                let mut stream = new();
                let rows = chunked(&mut stream, &series, chunk);
                prop_assert_eq!(&stream_bits(&rows), &by_update, "{:?} chunk {}", force, chunk);
            }
        }
    }
}

/// Row `i`'s exact entry on only the last `w` samples, searched for that row alone, once per case
/// and shared by every forced route.
fn exact_row(
    cache: &mut [Option<Bits>],
    samples: &[Option<i64>],
    m: usize,
    w: usize,
    i: usize,
) -> Bits {
    *cache[i].get_or_insert_with(|| stream_window_bits(samples, m, w, i))
}

fn growing_counts(force: RouteForce) -> (RouteCounts, Fired, usize) {
    let (series, m, scale) = long_walk();
    let guard = forced_route(force);
    let mut stream = GrowingStream::new(nz(m), scale);
    for &x in &series {
        let _ = stream.update(x);
    }
    (stream.route_counts(), guard.fired(), series.len())
}

fn bounded_counts(force: RouteForce) -> (RouteCounts, Fired, usize) {
    let (series, m, scale) = long_walk();
    let guard = forced_route(force);
    let mut stream = BoundedStream::new(nz(m), nz(40), scale).expect("w is at least m");
    for &x in &series {
        let _ = stream.update(x);
    }
    (stream.route_counts(), guard.fired(), series.len())
}

#[test]
fn stream_counts_follow_the_forced_route() {
    for run in [growing_counts, bounded_counts] {
        let (counts, _, n) = run(RouteForce::default());
        assert_eq!(
            counts,
            RouteCounts {
                narrow_updates: n as u64,
                ..RouteCounts::default()
            }
        );

        let wide = RouteForce {
            wide: true,
            ..RouteForce::default()
        };
        let (counts, fired, n) = run(wide);
        assert_eq!(
            counts,
            RouteCounts {
                wide_updates: n as u64,
                switches_to_wide: 1,
                ..RouteCounts::default()
            }
        );
        assert_eq!(fired.wide_switches, 1);

        let reanchoring = RouteForce {
            reanchor_every: 3,
            ..RouteForce::default()
        };
        let (counts, fired, n) = run(reanchoring);
        assert_eq!(counts.reanchors, fired.extra_reanchors as u64);
        assert!(counts.reanchors > 0);
        assert_eq!(counts.narrow_updates, n as u64);
        assert_eq!(counts.switches_to_wide, 0);
    }
}

#[test]
fn counts_start_from_zero_after_restore() {
    let _unforced = forced_route(RouteForce::default());
    let (series, m, scale) = long_walk();
    let mut stream = GrowingStream::new(nz(m), scale);
    for &x in &series {
        let _ = stream.update(x);
    }
    assert!(stream.route_counts().narrow_updates > 0);
    let restored = GrowingStream::restore(&stream.checkpoint()).expect("own checkpoint restores");
    assert_eq!(restored.route_counts(), RouteCounts::default());
}

/// A scale-1 series of plateaus at levels around the one-anchor bound, some with gaps, each long
/// enough that a bounded stream sees several return checks while the shocks come and go.
fn shocked(m: usize) -> impl Strategy<Value = (Vec<f64>, usize, Scale)> {
    let bound = (i32::MAX as usize / m) as f64;
    let levels = [
        0.0,
        0.0,
        bound / 2.0,
        bound,
        2.0 * bound,
        3.0 * bound,
        -bound,
    ];
    prop::collection::vec((0usize..7, 1usize..45, -4i32..=4, 0u8..12), 4..16).prop_map(
        move |segments| {
            let mut series = Vec::new();
            for (level, len, noise, kind) in segments {
                for k in 0..len {
                    let wobble = ((k as i32 * 7 + noise) % 9 - 4) as f64;
                    series.push(if kind == 0 && k % 5 == 3 {
                        f64::NAN
                    } else {
                        levels[level] + wobble
                    });
                }
            }
            (series, m, Scale::new(1.0).unwrap())
        },
    )
}

proptest! {
    #![proptest_config(testkit::generators::seeded(300))]

    #[test]
    fn bounded_returning_to_narrow_equals_the_exact_reference_on_every_route(
        (series, m, scale) in (2usize..=6).prop_flat_map(shocked),
        seed in any::<u64>(),
        extra in 0usize..30,
        pick in any::<usize>(),
    ) {
        let w = m + extra;
        let samples = samples(&series, scale);
        let split = pick % (series.len() + 1);
        let mut exact = vec![None; series.len()];
        for force in forces(seed) {
            let _forced = forced_route(force);
            let mut stream = BoundedStream::new(nz(m), nz(w), scale).expect("w is at least m");
            let mut rows = Vec::new();
            for &x in &series[..split] {
                rows.extend(stream.update(x));
            }
            let mut stream = BoundedStream::restore(&stream.checkpoint())
                .expect("a genuine checkpoint restores");
            for &x in &series[split..] {
                rows.extend(stream.update(x));
            }
            for row in stream_bits(&rows) {
                prop_assert_eq!(
                    row,
                    exact_row(&mut exact, &samples, m, w, row.0 as usize),
                    "{:?} w {} split {}", force, w, split
                );
            }
        }
    }
}

#[test]
fn a_shock_that_leaves_the_window_returns_the_stream_to_narrow_unless_wide_is_forced() {
    let m = 4;
    let bound = (i32::MAX as usize / m) as f64;
    let series: Vec<f64> = (0..300)
        .map(|k| match k {
            40..=43 => 3.0 * bound,
            _ => ((k * 5) % 11) as f64,
        })
        .collect();
    let run = |force| {
        let _forced = forced_route(force);
        let mut stream = BoundedStream::new(nz(m), nz(20), Scale::new(1.0).unwrap()).unwrap();
        for &x in &series {
            let _ = stream.update(x);
        }
        (stream.path(), stream.route_counts())
    };
    let (path, counts) = run(RouteForce::default());
    assert_eq!(path, Path::Narrow);
    assert_eq!((counts.switches_to_wide, counts.switches_to_narrow), (1, 1));
    assert!(counts.wide_updates > 0 && counts.narrow_updates > counts.wide_updates);

    let (path, counts) = run(RouteForce {
        wide: true,
        ..RouteForce::default()
    });
    assert_eq!(path, Path::Wide);
    assert_eq!(counts.switches_to_narrow, 0);
}
