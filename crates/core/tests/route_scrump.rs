#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The anytime profiles equal the exact reference over the diagonals visited (and the pairs of the
//! windows with no distance in the hot passes, ranked up front) on bits under every forced route,
//! and the tiles over a subset of the diagonals take the forced route: every tile runs the exact
//! i64 pass when it is forced, and the whole-series anchor is dropped when that is.

#[path = "common/route.rs"]
mod route;
use route::spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::two_sided_topk_profile_over_tiled;
use ladde::entry::{Entry, TwoSidedTopk};
use ladde::naive::naive_two_sided_topk_on;
use ladde::route::{RouteForce, forced_route};
use ladde::scrump::{Scrump, Shuffled};
use proptest::prelude::*;
use route::*;

type RowBits = (u64, [Bits; 2], Vec<Bits>);

fn row_bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<RowBits> {
    let one = |e: Entry| bits(0, e);
    profile
        .iter()
        .map(|(row, s)| {
            (
                *row,
                [s.left, s.right].map(one),
                s.nearest.iter().copied().map(one).collect(),
            )
        })
        .collect()
}

/// The reference's rows as [`row_bits`] lays them out.
fn reference_bits(rows: Vec<(Bits, Bits, Vec<Bits>)>) -> Vec<RowBits> {
    let one = |(_, rho, distance, index): Bits| (0, rho, distance, index);
    rows.into_iter()
        .enumerate()
        .map(|(row, (left, right, nearest))| {
            (
                row as u64,
                [one(left), one(right)],
                nearest.into_iter().map(one).collect(),
            )
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn anytime_equals_the_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        order_seed in any::<u64>(),
    ) {
        let Ok(_) = Scrump::diagonals(series.len(), nz(m)) else {
            return Ok(());
        };
        for force in forces(seed) {
            let guard = forced_route(force);
            for threads in [1, 3] {
                let mut scrump = Scrump::new(
                    &series, nz(m), scale, nz(threads), nz(2), 0.3, Shuffled::seeded(order_seed),
                ).unwrap();
                while !scrump.is_complete() {
                    scrump.update().unwrap();
                    let visited = scrump.visited();
                    let expected = spill_run::anytime_two_sided_topk(&series, scale, m, 2, visited);
                    prop_assert_eq!(
                        row_bits(&scrump.profile().unwrap()),
                        reference_bits(expected),
                        "{:?} threads {} after {} diagonals", force, threads, scrump.visited().len()
                    );
                }
            }
            drop(guard);
        }
    }

    #[test]
    fn subset_tiles_take_the_forced_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
    ) {
        let Ok((first, count)) = Scrump::diagonals(series.len(), nz(m)) else {
            return Ok(());
        };
        let some: Vec<i64> = (first..first + count as i64).step_by(3).collect();
        let samples = samples(&series, scale);
        let expected = row_bits(&naive_two_sided_topk_on(&samples, m, nz(2), &some));
        for force in forces(seed) {
            let _guard = forced_route(force);
            let (profile, report) = two_sided_topk_profile_over_tiled(
                &series, nz(m), scale, nz(2), nz(3), nz(2), nz(5), nz(2), &some,
            ).unwrap();
            prop_assert_eq!(&row_bits(&profile), &expected, "{:?}", force);
            if force.wide {
                prop_assert_eq!(report.wide_tiles, report.tiles, "{:?}", force);
            }
        }
    }
}

/// A calm series, so the natural route is the whole-series narrow anchor on every tile.
#[test]
fn forced_routes_fire_on_a_subset_run() {
    let series: Vec<f64> = (0..80)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let scale = ladde::quantize::Scale::new(1.0).unwrap();
    let (first, count) = Scrump::diagonals(series.len(), nz(5)).unwrap();
    let some: Vec<i64> = (first..first + count as i64).step_by(4).collect();
    let run = || {
        two_sided_topk_profile_over_tiled(
            &series,
            nz(5),
            scale,
            nz(1),
            nz(8),
            nz(4),
            nz(8),
            nz(1),
            &some,
        )
        .unwrap()
        .1
    };
    let natural = {
        let guard = forced_route(RouteForce::default());
        let report = run();
        assert_eq!(guard.fired(), Default::default());
        report
    };
    assert!(natural.tiles > 0);
    assert_eq!(natural.wide_tiles, 0);

    let guard = forced_route(RouteForce {
        wide: true,
        ..RouteForce::default()
    });
    let wide = run();
    assert_eq!(wide.tiles, natural.tiles);
    assert_eq!(wide.wide_tiles, wide.tiles);
    assert!(guard.fired().wide_switches > 0);
    drop(guard);

    let guard = forced_route(RouteForce {
        per_tile: true,
        ..RouteForce::default()
    });
    run();
    assert_eq!(guard.fired().per_tile_series, 1);
}
