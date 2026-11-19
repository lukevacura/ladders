#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Both multi-dimensional batch profiles equal the reference under every forced route: the
//! estimate screen that settles most cells, and the forced wide route on which every cell takes
//! the exact path, on inputs full of ties and on near-ties in the dimensions' order and in the
//! means. The non-normalized tiles it reports as wide follow the route that ran (every
//! z-normalized tile runs the exact `i64` arithmetic), and the screen settles cells only on the
//! route that has it.

#[path = "common/multi.rs"]
mod multi;

use ladde::batch::multi::{maamp_report, maamp_report_tiled, mstump_report, mstump_report_tiled};
use ladde::multidim::{NonFinitePolicy, screened_cells};
use ladde::naive::{naive_multi_profile, naive_multi_profile_nonnorm};
use ladde::route::{RouteForce, forced_route};
use multi::*;
use proptest::prelude::*;

fn wide() -> RouteForce {
    RouteForce {
        wide: true,
        ..RouteForce::default()
    }
}

/// Walks of a few ticks around zero in three dimensions: no tile needs more than `u64`.
fn calm() -> Vec<Vec<f64>> {
    (0..3).map(|a| kind_series(0, 90, 40 + a)).collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]

    #[test]
    fn equals_the_reference_on_every_route(input in input(9)) {
        let samples: Vec<_> = input.dimensions.iter().map(|s| ticks(s, unit())).collect();
        let refs: Vec<&[Option<i64>]> = samples.iter().map(Vec::as_slice).collect();
        let expected = naive_multi_profile_nonnorm(
            &refs, input.m, &input.include, input.discords, NonFinitePolicy::Missing,
        ).unwrap();
        let dimensions = slices(&input.dimensions);
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            let (profile, _) = maamp_report_tiled(
                &dimensions, nz(input.m), unit(), &input.include, input.discords,
                nz(2), nz(3), nz(2), nz(4),
            ).unwrap();
            prop_assert_eq!(&profile, &expected, "{:?}", force);
        }
    }

    #[test]
    fn z_normalized_equals_the_reference_on_every_route(input in input(8)) {
        let scales = vec![unit(); input.dimensions.len()];
        assert_routes_equal_the_reference(&input, &scales);
    }

    #[test]
    fn near_ties_equal_the_reference_on_every_route(input in near_ties()) {
        let scales = vec![unit(); input.dimensions.len()];
        assert_routes_equal_the_reference(&input, &scales);
        assert_nonnorm_routes_equal_the_reference(&input);
    }
}

/// `input` under both metrics' routes against the reference, at a few tile shapes and threads.
fn assert_routes_equal_the_reference(input: &Input, scales: &[ladde::quantize::Scale]) {
    let samples: Vec<_> = input.dimensions.iter().map(|s| ticks(s, unit())).collect();
    let refs: Vec<&[Option<i64>]> = samples.iter().map(Vec::as_slice).collect();
    let expected = naive_multi_profile(
        &refs,
        input.m,
        &input.include,
        input.discords,
        NonFinitePolicy::Missing,
    )
    .unwrap();
    let dimensions = slices(&input.dimensions);
    for force in [RouteForce::default(), wide()] {
        let _forced = forced_route(force);
        for (threads, band, chunk, tile) in [(1, 1, 1, 1), (2, 3, 2, 4), (1, 64, 64, 64)] {
            let (profile, _) = mstump_report_tiled(
                &dimensions,
                nz(input.m),
                scales,
                &input.include,
                input.discords,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            assert_eq!(
                bits(&profile),
                bits(&expected),
                "{input:?} {force:?} {band}"
            );
            assert_eq!(profile, expected, "{force:?}");
        }
    }
}

fn assert_nonnorm_routes_equal_the_reference(input: &Input) {
    let samples: Vec<_> = input.dimensions.iter().map(|s| ticks(s, unit())).collect();
    let refs: Vec<&[Option<i64>]> = samples.iter().map(Vec::as_slice).collect();
    let expected = naive_multi_profile_nonnorm(
        &refs,
        input.m,
        &input.include,
        input.discords,
        NonFinitePolicy::Missing,
    )
    .unwrap();
    let dimensions = slices(&input.dimensions);
    for force in [RouteForce::default(), wide()] {
        let _forced = forced_route(force);
        for (threads, band, chunk, tile) in [(1, 1, 1, 1), (2, 3, 2, 4), (1, 64, 64, 64)] {
            let (profile, _) = maamp_report_tiled(
                &dimensions,
                nz(input.m),
                unit(),
                &input.include,
                input.discords,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            assert_eq!(
                bits(&profile),
                bits(&expected),
                "{input:?} {force:?} {band}"
            );
        }
    }
}

/// Copies of one walk, each scaled and shifted (equal z-normalized distances in every cell, so
/// the dimensions' order ties) or the same (equal in both metrics), some then moved by one tick
/// at one sample: distances that differ in their last bits, so the order of the dimensions and
/// the means of a row's candidates are near-ties, the cells a screen must leave to the exact path.
fn near_ties() -> impl Strategy<Value = Input> {
    (
        2usize..=6,
        24usize..48,
        3usize..=6,
        any::<u64>(),
        prop::collection::vec((0u8..3, 1u8..=3, any::<u16>()), 6),
        prop::collection::vec(0usize..6, 0..3),
        any::<bool>(),
    )
        .prop_map(|(d, n, m, seed, copies, include, discords)| {
            let base = kind_series(0, n, seed);
            let dimensions = copies
                .iter()
                .take(d)
                .map(|&(kind, factor, at)| {
                    let mut copy: Vec<f64> = base
                        .iter()
                        .map(|&x| match kind {
                            0 => x,
                            _ => x * f64::from(factor) + 7.0,
                        })
                        .collect();
                    if kind == 2 {
                        copy[usize::from(at) % n] += 1.0;
                    }
                    copy
                })
                .collect();
            Input {
                dimensions,
                m,
                include: include.into_iter().map(|a| a % d).collect(),
                discords,
            }
        })
}

/// Random walks, one per dimension: almost every cell is settled by the screen.
fn walks(dimensions: usize, n: usize) -> Vec<Vec<f64>> {
    (0..dimensions)
        .map(|a| kind_series(0, n, 40 + a as u64))
        .collect()
}

#[test]
fn the_screen_settles_most_cells_and_none_on_the_wide_route() {
    let (n, m) = (300, 8);
    let series = walks(4, n);
    let dimensions = slices(&series);
    let windows = n - m + 1;
    let scales = vec![unit(); dimensions.len()];
    let cells = windows * windows;
    let z_normalized = |force: RouteForce| {
        let guard = forced_route(force);
        let before = screened_cells();
        let (profile, report) =
            mstump_report(&dimensions, nz(m), &scales, &[], false, nz(1)).unwrap();
        (
            profile,
            report,
            screened_cells() - before,
            guard.fired().wide_switches,
        )
    };
    let (natural, report, screened, fired) = z_normalized(RouteForce::default());
    assert_eq!((report.wide_tiles, fired), (report.tiles, 0));
    assert!(screened > cells * 3 / 4, "{screened} of {cells}");
    let (forced, report, screened, fired) = z_normalized(wide());
    assert_eq!(report.wide_tiles, report.tiles);
    assert_eq!(fired, report.tiles);
    assert_eq!(screened, 0);
    assert_eq!(natural, forced);

    let non_normalized = |force: RouteForce| {
        let _guard = forced_route(force);
        let before = screened_cells();
        let (profile, _) = maamp_report(&dimensions, nz(m), unit(), &[], false, nz(1)).unwrap();
        (profile, screened_cells() - before)
    };
    let (natural, screened) = non_normalized(RouteForce::default());
    assert!(screened > cells * 3 / 4, "{screened} of {cells}");
    let (forced, screened) = non_normalized(wide());
    assert_eq!(screened, 0);
    assert_eq!(natural, forced);
}

#[test]
fn wide_tiles_follow_the_route() {
    let series = calm();
    let dimensions = slices(&series);
    let run = |force: RouteForce| {
        let guard = forced_route(force);
        let (profile, report) =
            maamp_report(&dimensions, nz(8), unit(), &[], false, nz(1)).unwrap();
        (profile, report, guard.fired().wide_switches)
    };
    let (natural, report, fired) = run(RouteForce::default());
    assert!(report.tiles > 0);
    assert_eq!((report.wide_tiles, fired), (0, 0));
    let (forced, report, fired) = run(wide());
    assert_eq!(report.wide_tiles, report.tiles);
    assert!(fired > 0);
    assert_eq!(natural, forced);
}

#[test]
fn a_dimension_past_the_bound_makes_only_its_tiles_wide() {
    let mut series = calm();
    for x in &mut series[1][40..] {
        *x += (1u64 << 40) as f64;
    }
    let dimensions = slices(&series);
    let _unforced = forced_route(RouteForce::default());
    let (_, report) = maamp_report_tiled(
        &dimensions,
        nz(8),
        unit(),
        &[],
        false,
        nz(2),
        nz(10),
        nz(10),
        nz(10),
    )
    .unwrap();
    assert!(
        report.wide_tiles > 0 && report.wide_tiles < report.tiles,
        "{report:?}"
    );
}
