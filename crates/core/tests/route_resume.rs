#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A diagonal resumed by its exact cross term after a skipped run, instead of the recurrence,
//! leaves every batch, AB-join, two-sided and stream result equal on bits to the plain pass, and
//! the forced resume fires in each of them.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join_report_tiled, ab_join_single_threaded, left_profile_report_tiled,
    two_sided_profile_tiled, two_sided_topk_profile_over_walked,
};
use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, SpillForce, forced_route};
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::*;

type Run = fn(&[f64], usize, Scale) -> Vec<Bits>;

fn left_tiled(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    [(1, 1, 3), (4, 2, 16), (64, 8, 64)]
        .into_iter()
        .flat_map(|(band, chunk, tile)| {
            let (profile, _) = left_profile_report_tiled(
                series,
                nz(m),
                scale,
                nz(2),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            profile_bits(&profile)
        })
        .collect()
}

fn two_sided(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let (profile, _) = two_sided_profile_tiled(series, nz(m), scale, nz(2), nz(3), nz(2), nz(8));
    profile
        .iter()
        .flat_map(|&(row, t)| [t.left, t.right, t.nearest].map(|e| bits(row, e)))
        .collect()
}

fn walked(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let diagonals: Vec<i64> = (0..series.len() as i64).step_by(3).collect();
    two_sided_topk_profile_over_walked(series, nz(m), scale, nz(2), nz(2), &diagonals)
        .expect("small series fit")
        .iter()
        .flat_map(|(row, t)| {
            let entries: Vec<Entry> = [t.left, t.right]
                .into_iter()
                .chain(t.nearest.clone())
                .collect();
            entries.into_iter().map(move |e| bits(*row, e))
        })
        .collect()
}

/// The series against itself shifted by a third, so the two sides differ.
fn other_side(series: &[f64]) -> Vec<f64> {
    let cut = series.len() / 3;
    series[cut..]
        .iter()
        .chain(&series[..cut])
        .copied()
        .collect()
}

fn ab_one_pass(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let b = other_side(series);
    profile_bits(&ab_join_single_threaded(
        series,
        &b[..],
        nz(m),
        scale,
        scale,
    ))
}

fn ab_tiled(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let b = other_side(series);
    let (profile, _) = ab_join_report_tiled(
        series,
        &b[..],
        nz(m),
        scale,
        scale,
        nz(2),
        nz(3),
        nz(2),
        nz(5),
    );
    profile_bits(&profile)
}

fn growing(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let mut stream = GrowingStream::new(nz(m), scale);
    let rows: Vec<_> = series.iter().filter_map(|&x| stream.update(x)).collect();
    stream_bits(&rows)
}

fn bounded(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let mut stream = BoundedStream::new(nz(m), nz(3 * m + 5), scale).expect("w is at least m");
    let rows: Vec<_> = series.iter().filter_map(|&x| stream.update(x)).collect();
    stream_bits(&rows)
}

const FAMILIES: [(&str, Run); 7] = [
    ("left tiled", left_tiled),
    ("two-sided", two_sided),
    ("walked", walked),
    ("ab one-pass", ab_one_pass),
    ("ab tiled", ab_tiled),
    ("growing", growing),
    ("bounded", bounded),
];

/// Forced resumes at every row, every second and every third: each replaces a continuing
/// diagonal's state by its exact cross term, whatever run it would have skipped.
fn resumes(base: RouteForce) -> Vec<RouteForce> {
    [1, 2, 3]
        .into_iter()
        .map(|every| RouteForce {
            resume_every: every,
            ..base
        })
        .collect()
}

/// Each family's result under each forced resume equals its plain pass, and the resume fired.
fn check(series: &[f64], m: usize, scale: Scale, base: RouteForce) {
    for (name, run) in FAMILIES {
        let plain = {
            let _g = forced_route(base);
            run(series, m, scale)
        };
        assert!(
            plain
                .iter()
                .any(|&(_, rho, _, _)| f64::from_bits(rho).is_finite()),
            "{name}: some pair is ranked"
        );
        for force in resumes(base) {
            let guard = forced_route(force);
            let resumed = run(series, m, scale);
            assert_eq!(resumed, plain, "{name} {force:?}");
            assert!(guard.fired().resumes > 0, "{name} {force:?}");
        }
    }
}

#[test]
fn a_resume_keeps_every_bit_in_every_family() {
    let (series, m, scale) = long_walk();
    for base in [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
        RouteForce {
            reanchor_cell: 3,
            ..RouteForce::default()
        },
        RouteForce {
            per_tile: true,
            anchor_seed: 9,
            reanchor_every: 2,
            ..RouteForce::default()
        },
    ] {
        check(&series[..120], m, scale, base);
    }
}

#[test]
fn a_resume_across_a_gap_keeps_every_bit() {
    let (mut series, m, scale) = long_walk();
    series.truncate(100);
    // Gaps at the very start, inside a run, across what tiles and chunks cut, and at the end.
    for at in [0, 1, 17, 18, 19, 40, 64, 65, 97, 98, 99] {
        series[at] = f64::NAN;
    }
    check(&series, m, scale, RouteForce::default());
    check(
        &series,
        m,
        scale,
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    );
}

#[test]
fn a_resume_across_a_stand_in_window_keeps_every_bit() {
    let (mut series, m, scale) = long_walk();
    series.truncate(100);
    // One print far off the level: its windows are stood in and ranked by the cold route.
    series[50] += 1.0e9;
    let mut stream = GrowingStream::new(nz(m), scale);
    for &x in &series {
        let _ = stream.update(x);
    }
    assert!(
        stream.route_counts().withheld_ticks > 0,
        "the print is stood in"
    );
    check(
        &series,
        m,
        scale,
        RouteForce {
            spill: SpillForce::On,
            ..RouteForce::default()
        },
    );
}

#[test]
fn a_resume_near_the_integer_limits_keeps_every_bit() {
    // Ticks near the largest exact tick count, on two levels no anchor covers together: every
    // product wraps i64 and the pass runs the i64 arithmetic.
    let top = (1i64 << 53) as f64 - 4096.0;
    let series: Vec<f64> = (0..90)
        .map(|k| {
            let level = if (k / 23) % 2 == 0 { top } else { -top };
            level + ((k * 37) % 11) as f64
        })
        .collect();
    let scale = Scale::new(1.0).unwrap();
    let (_, report) = left_profile_report_tiled(&series, nz(4), scale, nz(1), nz(4), nz(2), nz(16));
    assert!(report.wide_tiles > 0, "a tile runs the i64 arithmetic");
    let mut stream = GrowingStream::new(nz(4), scale);
    for &x in &series {
        let _ = stream.update(x);
    }
    assert!(
        stream.route_counts().wide_updates > 0,
        "the stream runs the i64 arithmetic"
    );
    check(&series, 4, scale, RouteForce::default());
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]
    #[test]
    fn a_resume_keeps_every_bit_on_any_series(
        (series, m, scale) in cases(),
        every in 1usize..4,
    ) {
        for (name, family) in FAMILIES {
            let plain = {
                let _g = forced_route(RouteForce::default());
                family(&series, m, scale)
            };
            let _g = forced_route(RouteForce {
                resume_every: every,
                ..RouteForce::default()
            });
            prop_assert_eq!(family(&series, m, scale), plain, "{}", name);
        }
    }
}
