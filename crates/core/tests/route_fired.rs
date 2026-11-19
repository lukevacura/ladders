#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Each forced route fires in the engines that take it, and nothing fires when nothing is forced.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{ab_join, ab_join_report_tiled, left_profile, left_profile_report_tiled};
use ladde::quantize::Scale;
use ladde::route::{ConversionForce, Fired, RouteForce, forced_route};
use route::*;

#[test]
fn nothing_fires_when_nothing_is_forced() {
    let none = Fired::default();
    let force = RouteForce::default();
    assert_eq!(batch_run(force), none);
    assert_eq!(growing_run(force), none);
    assert_eq!(bounded_run(force), none);
}

#[test]
fn forced_wide_fires_in_every_engine() {
    let force = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    assert!(batch_run(force).wide_switches > 0);
    assert!(growing_run(force).wide_switches > 0);
    assert!(bounded_run(force).wide_switches > 0);
}

#[test]
fn forced_reanchors_fire_in_both_streams() {
    let force = RouteForce {
        reanchor_every: 3,
        ..RouteForce::default()
    };
    // The growing stream asks once per tick, at tick counts 0, 3, 6, ..., so the count is exact.
    assert_eq!(
        growing_run(force).extra_reanchors,
        long_walk().0.len().div_ceil(3)
    );
    assert!(bounded_run(force).extra_reanchors > 0);
}

#[test]
fn seeded_anchors_move_off_center_in_every_engine() {
    let seeded = RouteForce {
        anchor_seed: 7,
        ..RouteForce::default()
    };
    assert!(batch_run(seeded).off_center_anchors > 0);
    let with_reanchors = RouteForce {
        reanchor_every: 1,
        ..seeded
    };
    assert!(growing_run(with_reanchors).off_center_anchors > 0);
    assert!(bounded_run(with_reanchors).off_center_anchors > 0);
}

#[test]
fn a_forced_resume_fires_in_every_engine() {
    let force = RouteForce {
        resume_every: 2,
        ..RouteForce::default()
    };
    assert!(batch_run(force).resumes > 0);
    assert!(growing_run(force).resumes > 0);
    assert!(bounded_run(force).resumes > 0);
}

#[test]
fn a_forced_full_range_conversion_fires_in_batch_and_not_in_streams() {
    let force = RouteForce {
        conversion: ConversionForce::FullRange,
        ..RouteForce::default()
    };
    // The long walk fits: forcing the full-range conversion changes the decision.
    assert!(batch_run(force).conversions_forced_full_range > 0);
    assert_eq!(growing_run(force), Fired::default());
    assert_eq!(bounded_run(force), Fired::default());
    // Windows that swing 10⁸ ticks at m 8 do not fit: the full-range conversion is natural.
    let series: Vec<f64> = (0..200).map(|k| f64::from(k % 2) * 1.0e8).collect();
    let guard = forced_route(force);
    let _ = left_profile(&series, nz(8), Scale::new(1.0).unwrap());
    assert_eq!(guard.fired(), Fired::default());
}

#[test]
fn a_forced_fast_conversion_is_refused_where_the_bound_does_not_fit() {
    let force = RouteForce {
        conversion: ConversionForce::Fast,
        ..RouteForce::default()
    };
    // The long walk fits: nothing to refuse.
    assert_eq!(batch_run(force), Fired::default());
    // Windows that swing 10⁸ ticks at m 8 have `V` near 2⁵⁷.
    let series: Vec<f64> = (0..200).map(|k| f64::from(k % 2) * 1.0e8).collect();
    let guard = forced_route(force);
    let _ = left_profile(&series, nz(8), Scale::new(1.0).unwrap());
    assert!(guard.fired().fast_conversions_refused > 0);
}

/// The path shapes whose column sets choose a conversion: the whole-series anchor, per-tile
/// anchors and re-anchored cells.
fn shapes() -> [RouteForce; 3] {
    [
        RouteForce::default(),
        RouteForce {
            per_tile: true,
            ..RouteForce::default()
        },
        RouteForce {
            reanchor_cell: 2,
            ..RouteForce::default()
        },
    ]
}

/// What `run` fired under `shape` with the conversion forced to `conversion`.
fn fired_with(shape: RouteForce, conversion: ConversionForce, run: impl FnOnce()) -> Fired {
    let guard = forced_route(RouteForce {
        conversion,
        ..shape
    });
    run();
    guard.fired()
}

#[test]
fn every_tile_and_cell_asks_the_bound_for_its_conversion() {
    let (walk, m, scale) = long_walk();
    let tiled = || {
        let _ = left_profile_report_tiled(&walk[..], nz(m), scale, nz(4), nz(4), nz(2), nz(16));
    };
    let [whole, per_tile, reanchored] =
        shapes().map(|shape| fired_with(shape, ConversionForce::FullRange, tiled));
    assert!(whole.conversions_forced_full_range > 0);
    // Each tile or cell builds its own columns, and each asks.
    assert!(per_tile.conversions_forced_full_range > whole.conversions_forced_full_range);
    assert!(reanchored.conversions_forced_full_range > whole.conversions_forced_full_range);
}

#[test]
fn a_join_takes_the_fast_conversion_only_when_both_sides_fit() {
    let (walk, m, scale) = long_walk();
    // The walk fits; windows that swing 10⁸ ticks do not.
    let wide: Vec<f64> = (0..walk.len())
        .map(|k| f64::from(k % 2 == 1) * 1.0e8)
        .collect();
    let unit = Scale::new(1.0).unwrap();
    for shape in shapes() {
        for (a, scale_a, b, scale_b) in [(&walk, scale, &wide, unit), (&wide, unit, &walk, scale)] {
            // One pass where the join allows it, and always cut into tiles.
            let one_pass = || {
                let _ = ab_join(&a[..], &b[..], nz(m), scale_a, scale_b, nz(3));
            };
            let tiled = || {
                let _ = ab_join_report_tiled(
                    &a[..],
                    &b[..],
                    nz(m),
                    scale_a,
                    scale_b,
                    nz(3),
                    nz(4),
                    nz(2),
                    nz(16),
                );
            };
            for (cut, join) in [("one pass", &one_pass as &dyn Fn()), ("tiles", &tiled)] {
                let at = format!("{cut} {shape:?}");
                assert!(
                    fired_with(shape, ConversionForce::Fast, join).fast_conversions_refused > 0,
                    "{at}"
                );
                assert_eq!(
                    fired_with(shape, ConversionForce::FullRange, join)
                        .conversions_forced_full_range,
                    0,
                    "{at}"
                );
            }
        }
    }
}
