#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The AB-join equals the naive oracle on bits under every forced route.

#[path = "common/route.rs"]
mod route;
use route::spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    ab_join_report_tiled, ab_join_single_threaded, ab_join_topk, ab_join_topk_tiled,
};
use ladde::entry::Entry;
use ladde::route::{ConversionForce, Fired, RouteForce, forced_route};
use proptest::prelude::*;
use route::*;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn ab_join_equals_naive_on_every_route(
        (a, m, scale_a) in cases(),
        (b, _, scale_b) in cases(),
        seed in any::<u64>(),
    ) {
        let expected = spill_run::exact_join(&a, scale_a, &b, scale_b, m);
        for force in forces(seed) {
            let _forced = forced_route(force);
            let profile = ab_join_single_threaded(&a, &b, nz(m), scale_a, scale_b);
            prop_assert_eq!(&profile_bits(&profile), &expected, "{:?}", force);
            for threads in thread_counts() {
                for (band, chunk, tile) in [(1, 1, 1), (3, 2, 5), (64, 8, 64)] {
                    let (profile, report) = ab_join_report_tiled(
                        &a, &b, nz(m), scale_a, scale_b, nz(threads), nz(band), nz(chunk), nz(tile),
                    );
                    if force.wide {
                        prop_assert_eq!(report.wide_tiles, report.tiles, "{:?}", force);
                    }
                    prop_assert_eq!(
                        &profile_bits(&profile),
                        &expected,
                        "{:?} threads {} tiling {:?}", force, threads, (band, chunk, tile)
                    );
                }
            }
        }
    }
}

fn topk_bits(profile: &[(u64, Vec<Entry>)]) -> Vec<Vec<Bits>> {
    profile
        .iter()
        .map(|(row, entries)| entries.iter().map(|&e| bits(*row, e)).collect())
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn ab_join_topk_equals_naive_on_every_route(
        (a, m, scale_a) in cases(),
        (b, _, scale_b) in cases(),
        seed in any::<u64>(),
        k in prop_oneof![Just(1usize), Just(2), Just(3), Just(8)],
    ) {
        let expected = spill_run::exact_join_topk(&a, scale_a, &b, scale_b, m, k);
        for force in forces(seed) {
            let _forced = forced_route(force);
            for threads in thread_counts() {
                let default = ab_join_topk(&a, &b, nz(m), scale_a, scale_b, nz(threads), nz(k)).unwrap();
                prop_assert_eq!(
                    &topk_bits(&default), &expected, "{:?} threads {}", force, threads
                );
                for (band, chunk, tile) in [(1, 1, 1), (3, 2, 5), (64, 8, 64)] {
                    let (profile, report) = ab_join_topk_tiled(
                        &a, &b, nz(m), scale_a, scale_b, nz(threads), nz(band), nz(chunk),
                        nz(tile), nz(k),
                    ).unwrap();
                    if force.wide {
                        prop_assert_eq!(report.wide_tiles, report.tiles, "{:?}", force);
                    }
                    prop_assert_eq!(
                        &topk_bits(&profile),
                        &expected,
                        "{:?} threads {} tiling {:?}", force, threads, (band, chunk, tile)
                    );
                }
            }
        }
    }
}

fn ab_run(force: RouteForce) -> Fired {
    let (series, m, scale) = long_walk();
    let other: Vec<f64> = series.iter().rev().map(|x| x + 1.0).collect();
    let guard = forced_route(force);
    let _ = ab_join_single_threaded(&series, &other, nz(m), scale, scale);
    for (band, chunk, tile) in [(1, 1, 7), (4, 2, 16), (64, 8, 64)] {
        let _ = ab_join_report_tiled(
            &series,
            &other,
            nz(m),
            scale,
            scale,
            nz(2),
            nz(band),
            nz(chunk),
            nz(tile),
        );
    }
    guard.fired()
}

#[test]
fn ab_join_fires_only_the_routes_forced() {
    assert_eq!(ab_run(RouteForce::default()), Fired::default());
    let wide = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    assert!(ab_run(wide).wide_switches > 0);
    let seeded = RouteForce {
        anchor_seed: 7,
        ..RouteForce::default()
    };
    assert!(ab_run(seeded).off_center_anchors > 0);
    let full_range = RouteForce {
        conversion: ConversionForce::FullRange,
        ..RouteForce::default()
    };
    assert!(ab_run(full_range).conversions_forced_full_range > 0);
}

fn ab_topk_run(force: RouteForce) -> Fired {
    let (series, m, scale) = long_walk();
    let other: Vec<f64> = series.iter().rev().map(|x| x + 1.0).collect();
    let guard = forced_route(force);
    for (band, chunk, tile) in [(1, 1, 7), (4, 2, 16), (64, 8, 64)] {
        let _ = ab_join_topk_tiled(
            &series,
            &other,
            nz(m),
            scale,
            scale,
            nz(2),
            nz(band),
            nz(chunk),
            nz(tile),
            nz(3),
        )
        .unwrap();
    }
    guard.fired()
}

#[test]
fn ab_join_topk_fires_only_the_routes_forced() {
    assert_eq!(ab_topk_run(RouteForce::default()), Fired::default());
    let wide = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    assert!(ab_topk_run(wide).wide_switches > 0);
    let seeded = RouteForce {
        anchor_seed: 7,
        ..RouteForce::default()
    };
    assert!(ab_topk_run(seeded).off_center_anchors > 0);
}
