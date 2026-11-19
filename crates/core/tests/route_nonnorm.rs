#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized streams equal the exact reference on bits under every forced route, and
//! their route counts follow the route that ran.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/nonnorm_topk.rs"]
mod nonnorm_topk;

use ladde::batch::{BatchReport, nonnorm as batch};
use ladde::narrow::RouteCounts;
use ladde::nonnorm::NonNormNeighbor;
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::route::{RouteForce, forced_route};
use nonnorm::*;
use proptest::prelude::*;

fn wide() -> RouteForce {
    RouteForce {
        wide: true,
        ..RouteForce::default()
    }
}

/// A walk of a few ticks around zero: no row needs more than `u64`.
fn calm_walk() -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut price = 0i64;
    (0..200)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            price += ((state >> 33) % 7) as i64 - 3;
            price as f64
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn growing_equals_the_reference_on_every_route(
        series in prop_oneof![series(), tie_series()],
        m in 1usize..=6,
    ) {
        let expected = naive_rows(&series, unit(), m);
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            let mut stream = GrowingNonNormStream::new(nz(m), unit());
            let rows: Vec<_> =
                series.iter().filter_map(|&x| stream.update(x)).map(stream_row).collect();
            prop_assert_eq!(&rows, &expected, "{:?}", force);
        }
    }
}

fn counts(force: RouteForce, series: &[f64], m: usize) -> (RouteCounts, usize) {
    let guard = forced_route(force);
    let mut stream = GrowingNonNormStream::new(nz(m), unit());
    for &x in series {
        let _ = stream.update(x);
    }
    (stream.route_counts(), guard.fired().wide_switches)
}

#[test]
fn growing_counts_follow_the_route() {
    let series = calm_walk();
    let (natural, fired) = counts(RouteForce::default(), &series, 8);
    assert_eq!(
        natural,
        RouteCounts {
            narrow_updates: series.len() as u64,
            ..RouteCounts::default()
        }
    );
    assert_eq!(fired, 0);

    let (forced, fired) = counts(wide(), &series, 8);
    assert_eq!(forced.narrow_updates, 7);
    assert_eq!(forced.wide_updates, series.len() as u64 - 7);
    assert_eq!(forced.switches_to_wide, 1);
    assert!(fired > 0);
}

#[test]
fn a_range_past_the_bound_switches_once_and_stays_wide() {
    let mut series = calm_walk();
    series.extend(calm_walk().iter().map(|x| x + (1u64 << 40) as f64));
    series.extend(calm_walk());
    let (natural, fired) = counts(RouteForce::default(), &series, 8);
    assert_eq!(natural.switches_to_wide, 1);
    assert_eq!(natural.switches_to_narrow, 0);
    assert_eq!(natural.narrow_updates, 200);
    assert_eq!(
        natural.narrow_updates + natural.wide_updates,
        series.len() as u64
    );
    assert_eq!(fired, 0);
}

#[test]
fn counts_start_from_zero_after_restore() {
    let _unforced = forced_route(RouteForce::default());
    let mut stream = GrowingNonNormStream::new(nz(8), unit());
    for &x in &calm_walk() {
        let _ = stream.update(x);
    }
    assert!(stream.route_counts().narrow_updates > 0);
    let restored = GrowingNonNormStream::restore(&stream.checkpoint()).unwrap();
    assert_eq!(restored.route_counts(), RouteCounts::default());
}

fn bounded(m: usize, w: usize) -> BoundedNonNormStream {
    BoundedNonNormStream::new(nz(m), nz(w), unit()).expect("w is at least m")
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn bounded_equals_the_windowed_reference_on_every_route(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
        extra in 0usize..30,
    ) {
        let w = m + extra;
        let expected = naive_window_rows(&series, unit(), m, w);
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            let mut stream = bounded(m, w);
            let rows: Vec<_> =
                series.iter().filter_map(|&x| stream.update(x)).map(stream_row).collect();
            prop_assert_eq!(&rows, &expected, "{:?} w {}", force, w);
        }
    }
}

fn bounded_counts(force: RouteForce, series: &[f64], w: usize) -> (RouteCounts, usize) {
    let guard = forced_route(force);
    let mut stream = bounded(8, w);
    for &x in series {
        let _ = stream.update(x);
    }
    (stream.route_counts(), guard.fired().wide_switches)
}

#[test]
fn bounded_counts_follow_the_route() {
    let series = calm_walk();
    let (natural, fired) = bounded_counts(RouteForce::default(), &series, 40);
    assert_eq!(
        natural,
        RouteCounts {
            narrow_updates: series.len() as u64,
            ..RouteCounts::default()
        }
    );
    assert_eq!(fired, 0);

    let (forced, fired) = bounded_counts(wide(), &series, 40);
    assert_eq!(forced.narrow_updates, 7);
    assert_eq!(forced.wide_updates, series.len() as u64 - 7);
    assert_eq!(forced.switches_to_wide, 1);
    assert_eq!(forced.switches_to_narrow, 0);
    assert!(fired > 0);
}

/// A four-tick shock at `60` is inside the last `40` samples for ticks `60 ..= 102`.
#[test]
fn a_shock_that_leaves_the_horizon_returns_the_stream_to_narrow_unless_wide_is_forced() {
    for sign in [1.0, -1.0] {
        let mut series = calm_walk();
        for x in &mut series[60..64] {
            *x += sign * (1u64 << 40) as f64;
        }
        let (natural, fired) = bounded_counts(RouteForce::default(), &series, 40);
        assert_eq!(
            (natural.switches_to_wide, natural.switches_to_narrow),
            (1, 1)
        );
        assert_eq!(natural.wide_updates, 43, "sign {sign}");
        assert_eq!(natural.narrow_updates, series.len() as u64 - 43);
        assert_eq!(fired, 0);

        let (forced, _) = bounded_counts(wide(), &series, 40);
        assert_eq!(forced.switches_to_narrow, 0);
    }
}

#[test]
fn bounded_counts_start_from_zero_after_restore() {
    let _unforced = forced_route(RouteForce::default());
    let mut stream = bounded(8, 40);
    for &x in &calm_walk() {
        let _ = stream.update(x);
    }
    assert!(stream.route_counts().narrow_updates > 0);
    let restored = BoundedNonNormStream::restore(&stream.checkpoint()).unwrap();
    assert_eq!(restored.route_counts(), RouteCounts::default());
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn batch_equals_the_reference_on_every_route(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
    ) {
        let other = &series[series.len() / 2..];
        let left = naive_rows(&series, unit(), m);
        let sides = naive_sides(&series, unit(), m);
        let join = naive_join_rows(&series, other, unit(), m);
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            let (band, chunk, tile) = (nz(3), nz(2), nz(4));
            let (rows, _) =
                batch::left_profile_report_tiled(&series, nz(m), unit(), nz(2), band, chunk, tile);
            prop_assert_eq!(&batch_rows(&rows), &left, "left {:?}", force);
            let (rows, _) =
                batch::two_sided_profile_tiled(&series, nz(m), unit(), nz(2), band, chunk, tile);
            prop_assert_eq!(&batch_sides(&rows), &sides, "two-sided {:?}", force);
            let (rows, _) = batch::ab_join_report_tiled(
                &series, other, nz(m), unit(), nz(2), band, chunk, tile,
            );
            prop_assert_eq!(&batch_rows(&rows), &join, "join {:?}", force);
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn batch_top_k_equals_the_reference_on_every_route(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
        k in 1usize..=5,
    ) {
        let other = &series[series.len() / 2..];
        let sides = nonnorm_topk::exact_two_sided(&series, unit(), m, nz(k));
        let join = nonnorm_topk::exact_join(&series, other, unit(), m, nz(k));
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            let (band, chunk, tile) = (nz(3), nz(2), nz(4));
            let (rows, _) = batch::two_sided_topk_profile_tiled(
                &series, nz(m), unit(), nz(2), band, chunk, tile, nz(k),
            )
            .unwrap();
            prop_assert_eq!(&nonnorm_topk::batch_sides(&rows), &sides, "two-sided {:?}", force);
            let (rows, _) = batch::ab_join_topk_tiled(
                &series, other, nz(m), unit(), nz(2), band, chunk, tile, nz(k),
            )
            .unwrap();
            prop_assert_eq!(&nonnorm_topk::batch_join(&rows), &join, "join {:?}", force);
        }
    }
}

#[test]
fn top_k_tile_widths_follow_the_route() {
    let series = calm_walk();
    let tiled = |force: RouteForce| {
        let guard = forced_route(force);
        let (_, report) = batch::two_sided_topk_profile_tiled(
            &series,
            nz(8),
            unit(),
            nz(2),
            nz(8),
            nz(8),
            nz(8),
            nz(3),
        )
        .unwrap();
        (report, guard.fired().wide_switches)
    };
    let (natural, fired) = tiled(RouteForce::default());
    assert!(natural.tiles > 4);
    assert_eq!((natural.wide_tiles, fired), (0, 0));
    let (forced, fired) = tiled(wide());
    assert_eq!(forced.tiles, natural.tiles);
    assert_eq!(forced.wide_tiles, forced.tiles);
    assert_eq!(fired, forced.tiles);
}

fn tiled_left(series: &[f64], force: RouteForce) -> (BatchReport, usize) {
    let guard = forced_route(force);
    let (_, report) =
        batch::left_profile_report_tiled(series, nz(8), unit(), nz(2), nz(8), nz(8), nz(8));
    (report, guard.fired().wide_switches)
}

#[test]
fn batch_tile_widths_follow_the_route() {
    let series = calm_walk();
    let (natural, fired) = tiled_left(&series, RouteForce::default());
    assert!(natural.tiles > 4);
    assert_eq!((natural.wide_tiles, fired), (0, 0));

    let (forced, fired) = tiled_left(&series, wide());
    assert_eq!(forced.tiles, natural.tiles);
    assert_eq!(forced.wide_tiles, forced.tiles);
    assert_eq!(fired, forced.tiles);

    let mut shocked = series.clone();
    for x in &mut shocked[120..128] {
        *x += (1u64 << 40) as f64;
    }
    let (report, fired) = tiled_left(&shocked, RouteForce::default());
    assert!(
        report.wide_tiles > 0 && report.wide_tiles < report.tiles,
        "{report:?}"
    );
    assert_eq!(fired, 0);
}

type KeptRows = Vec<(u64, Vec<nonnorm_topk::Pair>)>;

fn kept_row(
    (row, kept): (ladde::index::SampleIndex, &[NonNormNeighbor]),
) -> (u64, Vec<nonnorm_topk::Pair>) {
    (row.get().unwrap(), nonnorm_topk::kept_pairs(kept))
}

fn growing_topk(series: &[f64], m: usize, k: usize) -> KeptRows {
    let mut stream = GrowingNonNormStream::with_k(nz(m), unit(), nz(k));
    series
        .iter()
        .filter_map(|&x| stream.update_topk(x).map(kept_row))
        .collect()
}

fn bounded_topk(series: &[f64], m: usize, w: usize, k: usize) -> KeptRows {
    let mut stream = BoundedNonNormStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
    series
        .iter()
        .filter_map(|&x| stream.update_topk(x).map(kept_row))
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn stream_top_k_equals_the_reference_on_every_route(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
        k in prop::sample::select(vec![1usize, 2, 3, 8, 200]),
        extra in 0usize..30,
    ) {
        let w = m + extra;
        let growing = nonnorm_topk::exact_left(&series, unit(), m, nz(k), None);
        let windowed = nonnorm_topk::exact_left(&series, unit(), m, nz(k), Some(w));
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            prop_assert_eq!(&growing_topk(&series, m, k), &growing, "growing {:?}", force);
            prop_assert_eq!(&bounded_topk(&series, m, w, k), &windowed, "bounded {:?} w {}", force, w);
        }
    }
}

#[test]
fn top_k_stream_counts_follow_the_route() {
    let series = calm_walk();
    for force in [RouteForce::default(), wide()] {
        let guard = forced_route(force);
        let mut growing = GrowingNonNormStream::with_k(nz(8), unit(), nz(3));
        let mut bounded = BoundedNonNormStream::with_k(nz(8), nz(40), unit(), nz(3)).unwrap();
        for &x in &series {
            let _ = growing.update_topk(x);
            let _ = bounded.update_topk(x);
        }
        let wide_updates = if force.wide {
            series.len() as u64 - 7
        } else {
            0
        };
        assert_eq!(
            growing.route_counts().wide_updates,
            wide_updates,
            "{force:?}"
        );
        assert_eq!(
            bounded.route_counts().wide_updates,
            wide_updates,
            "{force:?}"
        );
        assert_eq!(guard.fired().wide_switches > 0, force.wide);
    }
}
