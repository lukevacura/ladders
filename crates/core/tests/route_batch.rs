#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Batch profiles equal the naive oracle on bits under every forced route, thread count and
//! tiling.

#[path = "common/route.rs"]
mod route;
use route::spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    left_profile, left_profile_report_tiled, two_sided_profile, two_sided_profile_tiled,
    two_sided_topk_profile, two_sided_topk_profile_tiled,
};
use ladde::entry::{TwoSided, TwoSidedTopk};
use ladde::quantize::Scale;
use ladde::route::{Fired, RouteForce, forced_route};
use proptest::prelude::*;
use route::*;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn batch_equals_naive_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
    ) {
        // The exact reference: a window the hot passes cannot rank has a distance too.
        let expected = spill_run::reference(&series, m, scale);
        for force in forces(seed) {
            let _forced = forced_route(force);
            let default = left_profile(&series, nz(m), scale);
            prop_assert_eq!(&profile_bits(&default), &expected, "{:?} default", force);
            for threads in thread_counts() {
                for (band, chunk, tile) in [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)] {
                    let (profile, report) = left_profile_report_tiled(
                        &series, nz(m), scale, nz(threads), nz(band), nz(chunk), nz(tile),
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

fn two_sided_bits(profile: &[(u64, TwoSided)]) -> Vec<(u64, [Bits; 3])> {
    profile
        .iter()
        .map(|&(row, sides)| {
            (
                row,
                [sides.left, sides.right, sides.nearest].map(|e| bits(0, e)),
            )
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn two_sided_equals_naive_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
    ) {
        let expected = exact_two_sided_bits(&series, scale, m);
        for force in forces(seed) {
            let _forced = forced_route(force);
            for threads in thread_counts() {
                let default = two_sided_profile(&series, nz(m), scale, nz(threads));
                prop_assert_eq!(
                    &two_sided_bits(&default), &expected, "{:?} threads {}", force, threads
                );
                for (band, chunk, tile) in [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)] {
                    let (profile, _) = two_sided_profile_tiled(
                        &series, nz(m), scale, nz(threads), nz(band), nz(chunk), nz(tile),
                    );
                    prop_assert_eq!(
                        &two_sided_bits(&profile),
                        &expected,
                        "{:?} threads {} tiling {:?}", force, threads, (band, chunk, tile)
                    );
                }
            }
        }
    }

}

/// The exact reference two-sided profile, in `two_sided_bits`' shape.
fn exact_two_sided_bits(series: &[f64], scale: Scale, m: usize) -> Vec<(u64, [Bits; 3])> {
    spill_run::exact_two_sided(series, scale, m)
        .into_iter()
        .enumerate()
        .map(|(row, sides)| (row as u64, sides.map(|b| (0, b.1, b.2, b.3))))
        .collect()
}

type TopkBits = Vec<(u64, [Bits; 2], Vec<Bits>)>;

/// The exact reference two-sided top-k profile, in `two_sided_topk_bits`' shape.
fn exact_two_sided_topk_bits(series: &[f64], scale: Scale, m: usize, k: usize) -> TopkBits {
    let unrowed = |b: Bits| (0, b.1, b.2, b.3);
    spill_run::exact_two_sided_topk(series, scale, m, k)
        .into_iter()
        .enumerate()
        .map(|(row, (left, right, nearest))| {
            (
                row as u64,
                [unrowed(left), unrowed(right)],
                nearest.into_iter().map(unrowed).collect(),
            )
        })
        .collect()
}

fn two_sided_topk_bits(profile: &[(u64, TwoSidedTopk)]) -> TopkBits {
    profile
        .iter()
        .map(|(row, sides)| {
            (
                *row,
                [sides.left, sides.right].map(|e| bits(0, e)),
                sides.nearest.iter().map(|&e| bits(0, e)).collect(),
            )
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn two_sided_topk_equals_naive_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        k in prop_oneof![Just(1usize), Just(2), Just(3), Just(8)],
    ) {
        let expected = exact_two_sided_topk_bits(&series, scale, m, k);
        for force in forces(seed) {
            let _forced = forced_route(force);
            for threads in thread_counts() {
                let default = two_sided_topk_profile(&series, nz(m), scale, nz(threads), nz(k)).unwrap();
                prop_assert_eq!(
                    &two_sided_topk_bits(&default), &expected, "{:?} threads {}", force, threads
                );
                for (band, chunk, tile) in [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)] {
                    let (profile, report) = two_sided_topk_profile_tiled(
                        &series, nz(m), scale, nz(threads), nz(band), nz(chunk), nz(tile), nz(k),
                    ).unwrap();
                    if force.wide {
                        prop_assert_eq!(report.wide_tiles, report.tiles, "{:?}", force);
                    }
                    prop_assert_eq!(
                        &two_sided_topk_bits(&profile),
                        &expected,
                        "{:?} threads {} tiling {:?}", force, threads, (band, chunk, tile)
                    );
                }
            }
        }
    }
}

#[test]
fn forced_per_tile_anchors_fire_in_batch() {
    let force = RouteForce {
        per_tile: true,
        ..RouteForce::default()
    };
    assert!(batch_run(force).per_tile_series > 0);
}

fn two_sided_topk_run(force: RouteForce) -> Fired {
    let (series, m, scale) = long_walk();
    let guard = forced_route(force);
    for (band, chunk, tile) in [(1, 1, 7), (4, 2, 16), (64, 8, 64)] {
        let _ = two_sided_topk_profile_tiled(
            &series,
            nz(m),
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
fn two_sided_topk_fires_only_the_routes_forced() {
    assert_eq!(two_sided_topk_run(RouteForce::default()), Fired::default());
    let wide = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    assert!(two_sided_topk_run(wide).wide_switches > 0);
    let per_tile = RouteForce {
        per_tile: true,
        ..RouteForce::default()
    };
    assert!(two_sided_topk_run(per_tile).per_tile_series > 0);
    let seeded = RouteForce {
        anchor_seed: 7,
        ..RouteForce::default()
    };
    assert!(two_sided_topk_run(seeded).off_center_anchors > 0);
}
