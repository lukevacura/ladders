#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The one-loop route of a tiled self-join, two-sided or one-sided (the left profile): forced on,
//! it returns every output bit the tiled path returns, with the PAA filter on in both.
//!
//! Every bit is equal at 1, 2 and 8 threads on walks, ties, repeats, constant runs at band and
//! block edges, NaN gaps, a far best pair, and window counts around the band, block and range
//! boundaries (and tiny n and m), in the two-sided and the left profile and in every run the route
//! does not take (a filter that is not PAA, some diagonals only);
//! the handoff forced to miss changes no bit; and a random series equals the exact reference.
//! The route's own counters are tested here too: every cell of the triangle is skipped or
//! computed, no pair is evaluated twice or missed, and the ladder skips no pair with a constant
//! window (the pair-log tests force the constant class's block skip off; `route_constant_class`
//! tests that skip).

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join, ab_join_topk, left_profile_report_tiled, left_topk_profile_tiled,
    two_sided_profile_tiled, two_sided_topk_profile_over_tiled, two_sided_topk_profile_tiled,
};
use ladde::entry::TwoSidedTopk;
use ladde::motif::motif_pairs;
use ladde::naive::{naive_left_profile, naive_left_topk, naive_two_sided, naive_two_sided_topk};
use ladde::quantize::Scale;
use ladde::route::{OneLoop, RouteForce, SurvivorForce, forced_route};
use proptest::prelude::*;
use route::*;
use testkit::generators::{MARKET_TICK, market_series_and_m, seeded};

const THREADS: [usize; 3] = [1, 2, 8];

/// Bands, chunks and tiles: one row and lane (a band of one block of 16 rows), a band of one
/// block, a narrow chunk and a band past a block.
const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 3), (8, 64, 64), (16, 24, 48), (64, 64, 128)];

fn scale() -> Scale {
    Scale::new(MARKET_TICK).unwrap()
}

/// A series of `n` samples from a seeded generator: a random walk within one narrow anchor, or
/// one of `levels` distinct tick levels per sample (ties everywhere) when `levels` is set.
fn lcg(seed: u64, n: usize, levels: Option<u64>) -> Vec<f64> {
    let mut state = seed | 1;
    let mut price = 100.0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            match levels {
                Some(levels) => 100.0 + ((state >> 33) % levels) as f64 * MARKET_TICK,
                None => {
                    price += ((state >> 33) % 7) as f64 * MARKET_TICK - 3.0 * MARKET_TICK;
                    price
                }
            }
        })
        .collect()
}

/// `series` with `len` samples from `start` held at one value.
fn flat(mut series: Vec<f64>, start: usize, len: usize) -> Vec<f64> {
    let end = (start + len).min(series.len());
    if let Some(&level) = series.get(start) {
        series[start..end].fill(level);
    }
    series
}

/// `series` with NaN at every `every`th sample from `first`, and a run of three at `first`.
fn gaps(mut series: Vec<f64>, first: usize, every: usize) -> Vec<f64> {
    let n = series.len();
    for at in (first..n).step_by(every) {
        series[at] = f64::NAN;
    }
    for x in series.iter_mut().take((first + 3).min(n)).skip(first) {
        *x = f64::NAN;
    }
    series
}

/// A rough series whose best pair is its first and last stretch, far apart.
fn far_pair(n: usize, m: usize) -> Vec<f64> {
    let mut series: Vec<f64> = lcg(0xA11CE, n, None)
        .iter()
        .enumerate()
        .map(|(i, x)| x + ((i * 7919) % 13) as f64 * 30.0 * MARKET_TICK)
        .collect();
    let len = (m + 4).min(n / 2);
    let head: Vec<f64> = series[..len].to_vec();
    series[n - len..].copy_from_slice(&head);
    series
}

/// `series` with its first stretch copied over its last: an exact repeat.
fn repeated(mut series: Vec<f64>, m: usize) -> Vec<f64> {
    let n = series.len();
    if n >= 3 * m {
        let copy: Vec<f64> = series[..m + 2].to_vec();
        series[n - m - 2..].copy_from_slice(&copy);
    }
    series
}

/// One named series and its window length.
type Case = (String, Vec<f64>, usize);

/// Window counts around the band (16, 64), block (16, 64, 256) and tile boundaries, with every
/// kind of series, plus tiny n and m and longer series with constant runs starting at band and
/// block edges.
fn cases() -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();
    let mut add = |name: &str, n: usize, m: usize, series: Vec<f64>| {
        out.push((format!("{name} n={n} m={m}"), series, m));
    };
    let m = 8;
    for windows in [
        2, 9, 16, 17, 18, 24, 25, 31, 32, 33, 40, 63, 64, 65, 72, 73, 100, 128, 129,
    ] {
        let n = windows + m - 1;
        add("walk", n, m, lcg(1, n, None));
        add("ties", n, m, lcg(2, n, Some(3)));
        add("repeat", n, m, repeated(lcg(3, n, None), m));
        add("flat", n, m, flat(lcg(4, n, None), n / 3, m + 3));
        add("gaps", n, m, gaps(lcg(5, n, None), 5, 17));
        add("far", n, m, far_pair(n, m));
    }
    // Tiny: one window, two, the exclusion zone's edge, and tiny m.
    for (m, n) in [
        (8, 8),
        (8, 9),
        (8, 10),
        (2, 4),
        (2, 9),
        (3, 5),
        (3, 12),
        (4, 13),
    ] {
        add("tiny walk", n, m, lcg(6, n, None));
        add("tiny ties", n, m, lcg(7, n, Some(2)));
        add("tiny flat", n, m, flat(lcg(8, n, None), 1, m));
        add("tiny gaps", n, m, gaps(lcg(9, n, None), 1, 4));
    }
    // Longer, with a constant run beginning at each band and block edge.
    let (n, m) = (230, 12);
    for start in [7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128] {
        add(
            &format!("flat at {start}"),
            n,
            m,
            flat(lcg(10, n, None), start, m + 6),
        );
    }
    add("long walk", 300, 24, lcg(11, 300, None));
    add("long far", 300, 24, far_pair(300, 24));
    add("long ties", 300, 24, lcg(12, 300, Some(4)));
    out
}

fn force(route: OneLoop, filter: SurvivorForce) -> RouteForce {
    RouteForce {
        survivor: filter,
        one_loop: route,
        ..RouteForce::default()
    }
}

/// One entry point over the shapes, as bits.
type Family = fn(&[f64], usize, usize) -> Vec<Bits>;

fn two_sided(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    SHAPES
        .iter()
        .flat_map(|&(band, chunk, tile)| {
            let (profile, _) = two_sided_profile_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            profile
                .iter()
                .flat_map(|&(row, t)| [t.left, t.right, t.nearest].map(|e| bits(row, e)))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// One neighbor per row, offered to the rows only: the left profile.
fn left_only(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    SHAPES
        .iter()
        .flat_map(|&(band, chunk, tile)| {
            let (profile, _) = left_profile_report_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            profile_bits(&profile)
        })
        .collect()
}

fn topk_rows(rows: &[(u64, TwoSidedTopk)]) -> Vec<Bits> {
    rows.iter()
        .flat_map(|(row, t)| {
            let entries: Vec<_> = [t.left, t.right]
                .into_iter()
                .chain(t.nearest.clone())
                .collect();
            topk_bits(*row, &entries)
        })
        .collect()
}

/// `k` neighbors per row, two-sided.
fn topk_of(k: usize, series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    SHAPES
        .iter()
        .flat_map(|&(band, chunk, tile)| {
            let (profile, _) = two_sided_topk_profile_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(k),
            )
            .expect("small series fit");
            topk_rows(&profile)
        })
        .collect()
}

/// `k` neighbors per row, offered to the rows only, in one shape: the left top-k.
fn left_topk_of_shape(
    k: usize,
    series: &[f64],
    m: usize,
    threads: usize,
    (band, chunk, tile): (usize, usize, usize),
) -> Vec<Bits> {
    let (profile, _) = left_topk_profile_tiled(
        series,
        nz(m),
        scale(),
        nz(threads),
        nz(band),
        nz(chunk),
        nz(tile),
        nz(k),
    );
    profile
        .iter()
        .flat_map(|(row, entries)| topk_bits(*row, entries))
        .collect()
}

/// [`left_topk_of_shape`] over every shape.
fn left_topk_of(k: usize, series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    SHAPES
        .iter()
        .flat_map(|&shape| left_topk_of_shape(k, series, m, threads, shape))
        .collect()
}

fn topk2(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    topk_of(2, series, m, threads)
}

fn topk3(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    topk_of(3, series, m, threads)
}

fn topk8(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    topk_of(8, series, m, threads)
}

fn left_topk2(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    left_topk_of(2, series, m, threads)
}

fn left_topk3(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    left_topk_of(3, series, m, threads)
}

fn left_topk8(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    left_topk_of(8, series, m, threads)
}

/// Every top-k family the route takes, over `k` of 2, 3 and 8.
const TOPK_FAMILIES: [(&str, Family); 6] = [
    ("top-2", topk2),
    ("top-3", topk3),
    ("top-8", topk8),
    ("left top-2", left_topk2),
    ("left top-3", left_topk3),
    ("left top-8", left_topk8),
];

/// The AB-join of the series against its reverse, one neighbor per row.
fn ab_of_reverse(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    let other: Vec<f64> = series.iter().rev().copied().collect();
    profile_bits(&ab_join(
        series,
        &other[..],
        nz(m),
        scale(),
        scale(),
        nz(threads),
    ))
}

/// The AB-join of the series against its reverse, three neighbors per row.
fn ab_topk_of_reverse(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    let other: Vec<f64> = series.iter().rev().copied().collect();
    ab_join_topk(
        series,
        &other[..],
        nz(m),
        scale(),
        scale(),
        nz(threads),
        nz(3),
    )
    .expect("small series fit")
    .iter()
    .flat_map(|(row, entries)| topk_bits(*row, entries))
    .collect()
}

/// One neighbor per row, but only some diagonals walked.
fn some_diagonals(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    let diagonals: Vec<i64> = (1..series.len() as i64).filter(|d| d % 3 != 0).collect();
    SHAPES
        .iter()
        .flat_map(|&(band, chunk, tile)| {
            let (profile, _) = two_sided_topk_profile_over_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(1),
                &diagonals,
            )
            .expect("small series fit");
            topk_rows(&profile)
        })
        .collect()
}

/// The best-pair search, which keeps a global floor.
fn best_pair(series: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    motif_pairs(series, nz(m), scale(), nz(threads))
        .expect("small series fit")
        .iter()
        .map(|p| (p.a as u64, p.b as u64, p.distance.to_bits(), 0))
        .collect()
}

/// The entry points behind the route's entry the route does not take, which keep today's path:
/// some diagonals.
const OTHER_FAMILIES: [(&str, Family); 1] = [("some diagonals", some_diagonals)];

/// `family` under `force` at `threads`.
fn run(family: Family, series: &[f64], m: usize, threads: usize, force: RouteForce) -> Vec<Bits> {
    let _guard = forced_route(force);
    family(series, m, threads)
}

/// Forced on, at 1, 2 and 8 threads, every run equals the tiled path's at one thread.
fn assert_family(name: &str, family: Family, filter: SurvivorForce, cases: &[&Case]) {
    for (what, series, m) in cases {
        let reference = run(family, series, *m, 1, force(OneLoop::Off, filter));
        for threads in THREADS {
            assert_eq!(
                run(family, series, *m, threads, force(OneLoop::On, filter)),
                reference,
                "{name} {filter:?} on {threads} threads, {what}"
            );
            if threads > 1 {
                assert_eq!(
                    run(family, series, *m, threads, force(OneLoop::Off, filter)),
                    reference,
                    "{name} {filter:?} off {threads} threads, {what}"
                );
            }
        }
    }
}

#[test]
fn the_route_equals_the_tiled_path_in_the_two_sided_profile() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().collect();
    assert_family("two-sided", two_sided, SurvivorForce::Paa, &refs);
}

/// The left profile is one-sided: the route reads only its rows' floors, and at 1, 2 and 8 threads
/// returns the tiled path's bits (one thread included, where today's path is not the tile engine).
#[test]
fn the_route_equals_the_tiled_path_in_the_left_profile() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().collect();
    assert_family("left-only", left_only, SurvivorForce::Paa, &refs);
}

#[test]
fn the_route_equals_the_tiled_path_without_the_paa_filter() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(3).collect();
    for filter in [SurvivorForce::Gate, SurvivorForce::Off] {
        assert_family("two-sided", two_sided, filter, &refs);
        assert_family("left-only", left_only, filter, &refs);
    }
}

/// More than one neighbor per row: each row's `k`-th floor is the floor the ladder reads, and
/// forced on, at 1, 2 and 8 threads, every run equals the tiled path's.
#[test]
fn the_route_equals_the_tiled_path_with_k_neighbors() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(2).collect();
    for &(name, family) in &TOPK_FAMILIES {
        assert_family(name, family, SurvivorForce::Paa, &refs);
    }
}

#[test]
fn runs_the_route_does_not_take_keep_todays_bits_with_it_forced() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(4).collect();
    for &(name, family) in &OTHER_FAMILIES {
        assert_family(name, family, SurvivorForce::Paa, &refs);
    }
}

/// The best-pair search keeps a global floor, which the route reads as one more floor of every
/// block: forced on, at 1, 2 and 8 threads, it returns the tiled path's pair on every kind of
/// series.
#[test]
fn the_route_equals_the_tiled_path_in_the_best_pair_search() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(2).collect();
    assert_family("best pair", best_pair, SurvivorForce::Paa, &refs);
}

/// A handoff that always misses starts every lane lazily and changes no bit.
#[test]
fn a_handoff_that_always_misses_changes_no_bit() {
    let all = cases();
    for (name, family) in [
        ("two-sided", two_sided as Family),
        ("left-only", left_only),
        ("top-3", topk3),
        ("left top-3", left_topk3),
    ] {
        for (what, series, m) in all.iter().step_by(2) {
            let reference = run(
                family,
                series,
                *m,
                1,
                force(OneLoop::Off, SurvivorForce::Paa),
            );
            for threads in THREADS {
                for no_handoff in [false, true] {
                    let forced = RouteForce {
                        one_loop_no_handoff: no_handoff,
                        ..force(OneLoop::On, SurvivorForce::Paa)
                    };
                    assert_eq!(
                        run(family, series, *m, threads, forced),
                        reference,
                        "{name} no_handoff {no_handoff} {threads} threads, {what}"
                    );
                }
            }
        }
    }
}

/// The profiles the route takes, to run a counter test over each: both joins with one neighbor
/// and with three.
#[derive(Clone, Copy, Debug)]
enum Join {
    TwoSided,
    Left,
    TwoSidedTopk,
    LeftTopk,
}

const JOINS: [Join; 4] = [
    Join::TwoSided,
    Join::Left,
    Join::TwoSidedTopk,
    Join::LeftTopk,
];

/// The neighbors per row of the top-k joins of [`JOINS`].
const JOIN_K: usize = 3;

/// `join` over `series` in the shape `(band, chunk, tile)`, as bits.
fn walk(
    join: Join,
    series: &[f64],
    m: usize,
    threads: usize,
    (band, chunk, tile): (usize, usize, usize),
) -> Vec<Bits> {
    match join {
        Join::TwoSided => sides_bits(
            &two_sided_profile_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .0,
        ),
        Join::Left => profile_bits(
            &left_profile_report_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .0,
        ),
        Join::TwoSidedTopk => topk_rows(
            &two_sided_topk_profile_tiled(
                series,
                nz(m),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(JOIN_K),
            )
            .expect("small series fit")
            .0,
        ),
        Join::LeftTopk => left_topk_of_shape(JOIN_K, series, m, threads, (band, chunk, tile)),
    }
}

/// The pairs of a triangle of `windows` windows with exclusion `zone`: those `zone + 1` or more
/// apart.
fn triangle_area(windows: usize, zone: usize) -> usize {
    let diagonals = windows.saturating_sub(zone + 1);
    diagonals * (diagonals + 1) / 2
}

/// Every pair of the triangle is skipped on a proof or computed: the two areas add up to the
/// triangle's, from the run's own counters. A pair that touches a non-finite window is a cell like
/// any other (`Counts::computed_area`, `skipped_area`).
fn assert_area(
    join: Join,
    series: &[f64],
    m: usize,
    threads: usize,
    (band, chunk, tile): (usize, usize, usize),
) {
    let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
    let _ = walk(join, series, m, threads, (band, chunk, tile));
    let work = guard.work();
    let windows = series.len() - m + 1;
    let area = triangle_area(windows, ladde::exclusion::zone(m));
    assert_eq!(
        work.skipped_area + work.computed_area,
        area,
        "{join:?} n {} m {m} {threads} threads, shape {band}/{chunk}/{tile}: {work:?}",
        series.len()
    );
}

/// The series of one window count the area and evaluation tests sweep: a walk, one with NaN gaps,
/// one with a constant run, and one with ties everywhere.
fn area_series(windows: usize, m: usize) -> Vec<(&'static str, Vec<f64>)> {
    let n = windows + m - 1;
    vec![
        ("walk", lcg(21, n, None)),
        ("gaps", gaps(lcg(23, n, None), 5, 11)),
        ("flat", flat(lcg(24, n, None), n / 3, m + 3)),
        ("ties", lcg(25, n, Some(3))),
    ]
}

#[test]
fn every_cell_of_the_triangle_is_skipped_or_computed() {
    let m = 8;
    for windows in [2, 9, 16, 17, 33, 64, 65, 129, 200] {
        for (what, series) in area_series(windows, m) {
            let _ = what;
            for join in JOINS {
                for threads in THREADS {
                    for shape in SHAPES {
                        assert_area(join, &series, m, threads, shape);
                    }
                }
            }
        }
    }
}

#[test]
fn the_ladder_evaluates_and_computes_where_it_applies() {
    let series = lcg(22, 600, None);
    for join in JOINS {
        let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let _ = walk(join, &series, 24, 1, (64, 64, 128));
        let work = guard.work();
        assert!(work.ladder_entries[0] > 0, "{join:?} {work:?}");
        assert!(
            work.survivor_runs > 0 && work.computed_area > 0,
            "{join:?} {work:?}"
        );
    }
}

/// A run the route does not take (some diagonals only) reports nothing of the ladder, and a left
/// profile with the route off reports none either.
#[test]
fn a_run_the_route_does_not_take_counts_no_ladder() {
    let series = lcg(22, 600, None);
    for route in [OneLoop::Off, OneLoop::On] {
        let guard = forced_route(force(route, SurvivorForce::Paa));
        let _ = some_diagonals(&series, 24, 1);
        let work = guard.work();
        assert_eq!(work.ladder_entries, [0; 3], "{route:?}");
        assert_eq!(work.survivor_runs, 0, "{route:?}");
        assert_eq!(work.skipped_area + work.computed_area, 0, "{route:?}");
    }
}

/// The left profile with the route off reports nothing of the ladder.
#[test]
fn a_left_profile_with_the_route_off_counts_no_ladder() {
    let series = lcg(22, 600, None);
    let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
    let _ = left_only(&series, 24, 1);
    let work = guard.work();
    assert_eq!(work.ladder_entries, [0; 3]);
    assert_eq!(work.skipped_area + work.computed_area, 0);
}

/// The default (`OneLoop::Auto`, production's choice) takes the route for every family the box
/// accepted (the two-sided self-join of one neighbor or k, the best pair, the left profile of one
/// neighbor or k), and the plain pass for some diagonals; off takes none, and every family's bits
/// are the same either way.
#[test]
fn the_default_takes_the_loop_for_the_accepted_families() {
    let series = lcg(22, 600, None);
    let families: [(&str, Family, bool); 8] = [
        ("two-sided", two_sided, true),
        ("best pair", best_pair, true),
        ("left", left_only, true),
        ("top-k", topk2, true),
        ("left top-k", left_topk3, true),
        ("ab-join", ab_of_reverse, true),
        ("ab-join top-k", ab_topk_of_reverse, true),
        ("some diagonals", some_diagonals, false),
    ];
    for (name, family, on_the_loop) in families {
        for threads in [1, 4] {
            let guard = forced_route(force(OneLoop::Auto, SurvivorForce::Paa));
            let got = family(&series, 24, threads);
            let work = guard.work();
            drop(guard);
            assert_eq!(
                work.ladder_entries[0] > 0,
                on_the_loop,
                "{name} {threads} threads {work:?}"
            );
            assert_eq!(
                work.skipped_area + work.computed_area > 0,
                on_the_loop,
                "{name} {threads} threads {work:?}"
            );
            let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            let reference = family(&series, 24, threads);
            let work = guard.work();
            drop(guard);
            assert_eq!(work.ladder_entries, [0; 3], "{name} off {work:?}");
            assert_eq!(got, reference, "{name} {threads} threads");
        }
    }
}

/// A series with a repeating shape prunes: the ladder skips a share of the triangle, and the
/// skipped cells are cells the run never evaluated.
#[test]
fn a_repeating_series_is_pruned() {
    let base = lcg(41, 600, None);
    let series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    for join in JOINS {
        let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let _ = walk(join, &series, 32, 1, (64, 64, 128));
        let work = guard.work();
        assert!(work.skipped_area > 0, "{join:?} skipped nothing: {work:?}");
        assert!(
            work.ladder_entries.iter().all(|&e| e > 0),
            "{join:?} {work:?}"
        );
    }
}

/// A series of more windows than one range of slots (4096) holds: the floors are read, the ladder
/// passed and the floors raised more than once per band, at every shape. No bit changes, every
/// cell is covered once, and the ladder skips a share of it.
#[test]
fn a_series_longer_than_one_range_changes_no_bit_and_covers_every_cell() {
    let m = 64;
    let base = lcg(51, 5000, None);
    let series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 5) as f64 * 1e-6)
        .collect();
    let windows = series.len() - m + 1;
    let area = triangle_area(windows, ladde::exclusion::zone(m));
    for shape in [(64, 64, 4096), (256, 512, 8192), (1024, 256, 1024)] {
        let (band, chunk, tile) = shape;
        for join in JOINS {
            let reference = {
                let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                walk(join, &series, m, 1, (band, chunk, tile))
            };
            for threads in [1, 4] {
                let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
                let got = walk(join, &series, m, threads, (band, chunk, tile));
                assert_eq!(got, reference, "{join:?} {shape:?} {threads} threads");
                let work = guard.work();
                assert_eq!(
                    work.skipped_area + work.computed_area,
                    area,
                    "{join:?} {shape:?} {threads} threads: {work:?}"
                );
                assert!(work.skipped_area > 0, "{join:?} {shape:?}: {work:?}");
            }
        }
    }
}

/// The spans of pairs the run evaluated, expanded to `(row, column)`, each once.
fn evaluated_pairs(spans: &[(usize, usize, usize)]) -> (Vec<(usize, usize)>, usize) {
    let mut pairs: Vec<(usize, usize)> = spans
        .iter()
        .flat_map(|&(row, from, to)| (from..to).map(move |column| (row, column)))
        .collect();
    let total = pairs.len();
    pairs.sort_unstable();
    pairs.dedup();
    (pairs, total)
}

/// No pair is evaluated twice, each evaluated pair is a pair of the triangle, and the pairs
/// evaluated are the pairs the run counts as computed; on a series that cannot be pruned (every
/// window constant) they are the whole triangle, so none is missed either.
#[test]
fn no_pair_is_evaluated_twice_or_missed() {
    let m = 4;
    let zone = ladde::exclusion::zone(m);
    for windows in [2, 5, 9, 16, 17, 18, 25, 33, 40, 64, 65] {
        let n = windows + m - 1;
        let constant = vec![100.0; n];
        for (what, series) in area_series(windows, m)
            .into_iter()
            .chain([("constant", constant)])
        {
            for (join, shape) in JOINS.into_iter().flat_map(|j| SHAPES.map(|s| (j, s))) {
                let guard = forced_route(RouteForce {
                    log_pairs: true,
                    no_constant_blocks: true,
                    ..force(OneLoop::On, SurvivorForce::Paa)
                });
                let _ = walk(join, &series, m, 1, shape);
                let work = guard.work();
                let (pairs, total) = evaluated_pairs(&guard.pair_log());
                let at = format!("{join:?} {what} n {n} shape {shape:?}");
                assert_eq!(pairs.len(), total, "a pair evaluated twice: {at}");
                assert!(
                    pairs
                        .iter()
                        .all(|&(row, column)| row < windows && column + zone < row),
                    "a pair outside the triangle: {at}"
                );
                assert_eq!(pairs.len(), work.computed_area, "{at}: {work:?}");
                if what == "constant" {
                    assert_eq!(pairs.len(), triangle_area(windows, zone), "{at}");
                }
            }
        }
    }
}

/// I5 at the route: a constant window is never skipped. On a repeating series (which the ladder
/// prunes hard) with flat runs at several places, every pair with a constant window is among the
/// pairs the run evaluated, and the profile equals the tiled path's; on a series of one level the
/// whole triangle is evaluated, at windows that span every layer of the ladder.
#[test]
fn a_constant_window_is_never_skipped() {
    let m = 32;
    let zone = ladde::exclusion::zone(m);
    let base = lcg(61, 700, None);
    let mut series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    for start in [100, 333, 1000] {
        series = flat(series, start, 3 * m);
    }
    let windows = series.len() - m + 1;
    let constant = |w: usize| series[w..w + m].iter().all(|&x| x == series[w]);
    assert!((0..windows).filter(|&w| constant(w)).count() > 60);
    for (join, shape) in JOINS
        .into_iter()
        .flat_map(|j| [(16, 16, 64), (64, 64, 128), (256, 256, 512)].map(|s| (j, s)))
    {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, &series, m, 1, shape)
        };
        let guard = forced_route(RouteForce {
            log_pairs: true,
            no_constant_blocks: true,
            ..force(OneLoop::On, SurvivorForce::Paa)
        });
        let got = walk(join, &series, m, 1, shape);
        assert_eq!(got, reference, "{join:?} {shape:?}");
        let work = guard.work();
        assert!(
            work.skipped_area > 0,
            "{join:?} the series prunes: {work:?}"
        );
        let (pairs, _) = evaluated_pairs(&guard.pair_log());
        let seen: std::collections::BTreeSet<(usize, usize)> = pairs.into_iter().collect();
        for row in 0..windows {
            for column in 0..row.saturating_sub(zone) {
                if constant(row) || constant(column) {
                    assert!(
                        seen.contains(&(row, column)),
                        "{join:?} {shape:?}: pair ({row}, {column}) with a constant window was skipped"
                    );
                }
            }
        }
    }
    // One level everywhere: nothing can be skipped.
    let level = vec![100.0; 700 + m - 1];
    let windows = level.len() - m + 1;
    for join in JOINS {
        let guard = forced_route(RouteForce {
            log_pairs: true,
            no_constant_blocks: true,
            ..force(OneLoop::On, SurvivorForce::Paa)
        });
        let _ = walk(join, &level, m, 1, (64, 64, 256));
        let (pairs, total) = evaluated_pairs(&guard.pair_log());
        assert_eq!(pairs.len(), total, "{join:?}");
        assert_eq!(pairs.len(), triangle_area(windows, zone), "{join:?}");
        assert_eq!(guard.work().skipped_area, 0, "{join:?}");
    }
}

fn sides_bits(profile: &[(u64, ladde::entry::TwoSided)]) -> Vec<Bits> {
    profile
        .iter()
        .flat_map(|&(row, t)| [t.left, t.right, t.nearest].map(|e| bits(row, e)))
        .collect()
}

proptest! {
    #![proptest_config(seeded(40))]
    /// A random series, a shape and a thread count: the profile equals the exact reference on
    /// bits.
    #[test]
    fn a_random_series_equals_the_exact_profile(
        (series, m) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
    ) {
        let (band, chunk, tile) = shape;
        let exact = sides_bits(&naive_two_sided(&samples(&series, scale())[..], m));
        let _guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let (profile, _) = two_sided_profile_tiled(
            &series, nz(m), scale(), nz(threads), nz(band), nz(chunk), nz(tile),
        );
        prop_assert_eq!(sides_bits(&profile), exact, "{} threads {:?}", threads, shape);
    }

    /// The same for the left profile, against the exact left reference.
    #[test]
    fn a_random_series_equals_the_exact_left_profile(
        (series, m) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
    ) {
        let exact = profile_bits(&naive_left_profile(&samples(&series, scale())[..], m));
        let _guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        prop_assert_eq!(
            walk(Join::Left, &series, m, threads, shape),
            exact,
            "{} threads {:?}",
            threads,
            shape
        );
    }

    /// `k` neighbors per row, two-sided, against the exact top-k.
    #[test]
    fn a_random_series_equals_the_exact_topk(
        (series, m) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
        k in prop::sample::select(vec![2usize, 3, 8]),
    ) {
        let exact = topk_rows(&naive_two_sided_topk(&samples(&series, scale())[..], m, nz(k)));
        let _guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let (profile, _) = two_sided_topk_profile_tiled(
            &series, nz(m), scale(), nz(threads), nz(shape.0), nz(shape.1), nz(shape.2), nz(k),
        )
        .expect("small series fit");
        prop_assert_eq!(topk_rows(&profile), exact, "{} threads {:?} k {}", threads, shape, k);
    }

    /// `k` neighbors per row, one-sided, against the exact left top-k.
    #[test]
    fn a_random_series_equals_the_exact_left_topk(
        (series, m) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
        k in prop::sample::select(vec![2usize, 3, 8]),
    ) {
        let exact: Vec<Bits> = naive_left_topk(&samples(&series, scale())[..], m, nz(k))
            .iter()
            .flat_map(|(row, entries)| topk_bits(*row, entries))
            .collect();
        let _guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        prop_assert_eq!(
            left_topk_of_shape(k, &series, m, threads, shape),
            exact,
            "{} threads {:?} k {}",
            threads,
            shape,
            k
        );
    }
}

#[test]
fn the_route_equals_the_tiled_path_on_a_small_walk() {
    let series = lcg(0x9E37_79B9_7F4A_7C15, 1200, None);
    let off = run(
        two_sided,
        &series,
        24,
        1,
        force(OneLoop::Off, SurvivorForce::Paa),
    );
    assert!(off.len() > 3 * 1000, "the profile is not trivial");
    assert_eq!(
        run(
            two_sided,
            &series,
            24,
            1,
            force(OneLoop::On, SurvivorForce::Paa)
        ),
        off
    );
}

/// The best-pair search under the route: every cell of the triangle is skipped or computed, and
/// the ladder skips a share of it on a repeating series.
#[test]
fn the_best_pair_search_covers_every_cell_and_the_global_floor_skips() {
    let m = 8;
    let zone = ladde::exclusion::zone(m);
    for windows in [2, 9, 17, 33, 65, 129, 200, 700] {
        let mut all = area_series(windows, m);
        all.push(("far", far_pair(windows + m - 1, m)));
        for (what, series) in all {
            for threads in THREADS {
                let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
                let _ =
                    motif_pairs(&series, nz(m), scale(), nz(threads)).expect("small series fit");
                let work = guard.work();
                assert_eq!(
                    work.skipped_area + work.computed_area,
                    triangle_area(windows, zone),
                    "{what} windows {windows} {threads} threads: {work:?}"
                );
            }
        }
    }
    let base = lcg(41, 600, None);
    let series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
    let _ = motif_pairs(&series, nz(32), scale(), nz(1)).expect("small series fit");
    assert!(guard.work().skipped_area > 0, "{:?}", guard.work());
}

/// The best-pair search: no pair is evaluated twice, each is a pair of the triangle, the pairs
/// evaluated are the pairs counted as computed, and every pair with a constant window is
/// evaluated (the global floor never skips a block that holds a constant window).
#[test]
fn the_best_pair_search_evaluates_no_pair_twice_and_skips_no_constant_window() {
    let m = 16;
    let zone = ladde::exclusion::zone(m);
    let base = lcg(71, 400, None);
    let mut series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    for start in [90, 301, 650] {
        series = flat(series, start, 2 * m);
    }
    let windows = series.len() - m + 1;
    let constant = |w: usize| series[w..w + m].iter().all(|&x| x == series[w]);
    let guard = forced_route(RouteForce {
        log_pairs: true,
        no_constant_blocks: true,
        ..force(OneLoop::On, SurvivorForce::Paa)
    });
    let _ = motif_pairs(&series, nz(m), scale(), nz(1)).expect("small series fit");
    let work = guard.work();
    let (pairs, total) = evaluated_pairs(&guard.pair_log());
    assert_eq!(pairs.len(), total, "a pair evaluated twice");
    assert!(
        pairs
            .iter()
            .all(|&(row, column)| row < windows && column + zone < row)
    );
    assert_eq!(pairs.len(), work.computed_area, "{work:?}");
    assert!(work.skipped_area > 0, "the series prunes: {work:?}");
    let seen: std::collections::BTreeSet<(usize, usize)> = pairs.into_iter().collect();
    for row in 0..windows {
        for column in 0..row.saturating_sub(zone) {
            if constant(row) || constant(column) {
                assert!(
                    seen.contains(&(row, column)),
                    "pair ({row}, {column}) with a constant window was skipped"
                );
            }
        }
    }
}

// Per-band anchors: a series no one anchor covers narrows each band to anchors of its own, and
// returns the tiled path's bits.

/// Every tile takes anchors of its own (placed from `seed`, `0` keeping them centered), the one-loop
/// route as `route` says.
fn per_band(route: OneLoop, seed: u64) -> RouteForce {
    RouteForce {
        per_tile: true,
        anchor_seed: seed,
        ..force(route, SurvivorForce::Paa)
    }
}

/// A walk of `n` samples that climbs `slope` ticks a sample, so its ticks span more than one
/// anchor covers at `m` while a window, a band and a tile span little.
fn ramp(seed: u64, n: usize, slope: f64) -> Vec<f64> {
    lcg(seed, n, None)
        .iter()
        .enumerate()
        .map(|(i, x)| x + i as f64 * slope * MARKET_TICK)
        .collect()
}

/// Forced per-band, at 1, 2 and 8 threads and with the anchors centered and placed anywhere, every
/// run equals the whole-anchor tiled path's at one thread, in every family the route takes.
#[test]
fn per_band_anchors_equal_the_whole_anchor_tiled_path() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(4).collect();
    let families: [(&str, Family); 7] = [
        ("two-sided", two_sided),
        ("left-only", left_only),
        ("best pair", best_pair),
        ("top-2", topk2),
        ("top-8", topk8),
        ("left top-3", left_topk3),
        ("left top-8", left_topk8),
    ];
    for (name, family) in families {
        for (what, series, m) in &refs {
            let reference = run(
                family,
                series,
                *m,
                1,
                force(OneLoop::Off, SurvivorForce::Paa),
            );
            for seed in [0, 0x5EED] {
                for threads in THREADS {
                    assert_eq!(
                        run(family, series, *m, threads, per_band(OneLoop::On, seed)),
                        reference,
                        "{name} seed {seed} on {threads} threads, {what}"
                    );
                }
            }
        }
    }
}

/// A series whose ticks no one anchor covers takes the route by itself, with each band's own
/// anchors, where the route is forced on, and keeps today's path by default and when off; every
/// run returns the same bits. The slopes are chosen so a tile's columns fit no anchor either at
/// the wider shapes, which run on the exact ticks.
#[test]
fn a_series_no_anchor_covers_runs_on_per_band_anchors() {
    for (m, slope, n) in [(24usize, 2.0e6, 700usize), (8, 5.0e6, 500)] {
        let series = ramp(81, n, slope);
        for (name, family) in [
            ("two-sided", two_sided as Family),
            ("left-only", left_only),
            ("top-3", topk3),
            ("left top-3", left_topk3),
            ("best pair", best_pair),
        ] {
            let off = {
                let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                let got = family(&series, m, 1);
                assert_eq!(guard.work().ladder_entries, [0; 3], "{name} off");
                got
            };
            for threads in THREADS {
                let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
                let got = family(&series, m, threads);
                let work = guard.work();
                drop(guard);
                assert_eq!(got, off, "{name} on {threads} threads, m {m}");
                assert!(
                    work.ladder_entries[0] > 0 && work.computed_area > 0,
                    "{name} m {m}: {work:?}"
                );
                let guard = forced_route(force(OneLoop::Auto, SurvivorForce::Paa));
                let got = family(&series, m, threads);
                assert!(
                    guard.work().ladder_entries[0] > 0,
                    "{name} auto, m {m}: the default takes the wide route too"
                );
                drop(guard);
                assert_eq!(got, off, "{name} auto {threads} threads, m {m}");
            }
        }
    }
}

/// Adjacent bands with different anchors hand lanes down and move them: lanes taken from the band
/// above are moved to the band's anchors, and no bit changes; with one anchor for the series no
/// lane is moved.
#[test]
fn lanes_are_handed_down_and_moved_between_bands_of_different_anchors() {
    let series = ramp(82, 900, 3.0e6);
    let m = 24;
    for join in JOINS {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, &series, m, 1, (16, 16, 64))
        };
        for seed in [0, 0x5EED] {
            let guard = forced_route(per_band(OneLoop::On, seed));
            let got = walk(join, &series, m, 1, (16, 16, 64));
            let work = guard.work();
            assert_eq!(got, reference, "{join:?} seed {seed}");
            assert!(
                work.lanes_handed_down > 0 && work.lanes_moved > 0,
                "{join:?} seed {seed}: {work:?}"
            );
            assert!(work.lanes_moved <= work.lanes_handed_down);
        }
    }
    let whole = lcg(83, 900, None);
    let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
    let _ = walk(Join::TwoSided, &whole, m, 1, (16, 16, 64));
    let work = guard.work();
    assert!(work.lanes_handed_down > 0, "{work:?}");
    assert_eq!(work.lanes_moved, 0, "{work:?}");
}

/// A series no one anchor covers, under a plan whose band is too tall for one anchor per side (a
/// band of 4096 rows), runs on the route (bands kept or cut one grain tall where that leaves fewer
/// pairs uncovered; tiles with no anchor on the exact ticks) and no bit differs from the plain
/// pass's, at 1, 2 and 8 threads in every family. Which bands are cut is the `cut_tiles` unit
/// tests' to pin.
#[test]
fn a_plan_band_too_tall_for_one_anchor_runs_on_the_route() {
    let (m, shape) = (24, (4096, 64, 4096));
    let series = ramp(85, 7000, 2.6e4);
    for join in JOINS {
        let reference = {
            let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            let got = walk(join, &series, m, 1, shape);
            assert_eq!(guard.work().ladder_entries, [0; 3]);
            got
        };
        for threads in THREADS {
            let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
            let got = walk(join, &series, m, threads, shape);
            let work = guard.work();
            drop(guard);
            assert_eq!(got, reference, "{join:?} {threads} threads");
            assert!(work.ladder_entries[0] > 0, "{join:?}: {work:?}");
        }
    }
}

/// A handoff that always misses changes no bit on per-band anchors either, and a band whose
/// anchors cannot be placed (the forced cell cut or the exact i64 pass) runs on the exact ticks.
#[test]
fn per_band_runs_keep_the_bits_with_missed_handoffs_and_tiles_the_loop_declines() {
    let series = ramp(84, 600, 2.5e6);
    let m = 16;
    for join in JOINS {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, &series, m, 1, (16, 16, 64))
        };
        for threads in [1, 4] {
            for forced in [
                RouteForce {
                    one_loop_no_handoff: true,
                    ..per_band(OneLoop::On, 0x5EED)
                },
                RouteForce {
                    reanchor_cell: 8,
                    ..per_band(OneLoop::On, 0)
                },
                RouteForce {
                    wide: true,
                    ..per_band(OneLoop::On, 0)
                },
            ] {
                let _guard = forced_route(forced);
                assert_eq!(
                    walk(join, &series, m, threads, (16, 16, 64)),
                    reference,
                    "{join:?} {threads} threads {forced:?}"
                );
            }
        }
    }
}

/// I1 on per-band anchors: every cell of the triangle is skipped or computed, no pair is
/// evaluated twice or outside the triangle, and the pairs evaluated are the pairs counted.
#[test]
fn per_band_runs_cover_every_cell_once() {
    let m = 8;
    let zone = ladde::exclusion::zone(m);
    for windows in [2, 9, 17, 33, 65, 129, 200] {
        for (what, series) in area_series(windows, m) {
            for join in JOINS {
                for shape in SHAPES {
                    for threads in [1, 4] {
                        let guard = forced_route(RouteForce {
                            log_pairs: true,
                            no_constant_blocks: true,
                            ..per_band(OneLoop::On, 0x5EED)
                        });
                        let _ = walk(join, &series, m, threads, shape);
                        let work = guard.work();
                        let at = format!(
                            "{join:?} {what} windows {windows} {threads} threads {shape:?}"
                        );
                        assert_eq!(
                            work.skipped_area + work.computed_area,
                            triangle_area(windows, zone),
                            "{at}: {work:?}"
                        );
                        let (pairs, total) = evaluated_pairs(&guard.pair_log());
                        assert_eq!(pairs.len(), total, "a pair evaluated twice: {at}");
                        assert_eq!(pairs.len(), work.computed_area, "{at}: {work:?}");
                        assert!(
                            pairs
                                .iter()
                                .all(|&(row, column)| row < windows && column + zone < row),
                            "a pair outside the triangle: {at}"
                        );
                    }
                }
            }
        }
    }
}

/// I5 on per-band anchors: a repeating series with flat runs, pruned hard, evaluates every pair
/// with a constant window and returns the whole-anchor tiled path's bits.
#[test]
fn per_band_runs_never_skip_a_constant_window() {
    let m = 32;
    let zone = ladde::exclusion::zone(m);
    let base = lcg(61, 700, None);
    let mut series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    for start in [100, 333, 1000] {
        series = flat(series, start, 3 * m);
    }
    let windows = series.len() - m + 1;
    let constant = |w: usize| series[w..w + m].iter().all(|&x| x == series[w]);
    for (join, shape) in JOINS
        .into_iter()
        .flat_map(|j| [(16, 16, 64), (64, 64, 128)].map(|s| (j, s)))
    {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, &series, m, 1, shape)
        };
        let guard = forced_route(RouteForce {
            log_pairs: true,
            no_constant_blocks: true,
            ..per_band(OneLoop::On, 0x5EED)
        });
        assert_eq!(
            walk(join, &series, m, 1, shape),
            reference,
            "{join:?} {shape:?}"
        );
        let work = guard.work();
        assert!(
            work.skipped_area > 0,
            "{join:?} the series prunes: {work:?}"
        );
        let (pairs, _) = evaluated_pairs(&guard.pair_log());
        let seen: std::collections::BTreeSet<(usize, usize)> = pairs.into_iter().collect();
        for row in 0..windows {
            for column in 0..row.saturating_sub(zone) {
                if constant(row) || constant(column) {
                    assert!(
                        seen.contains(&(row, column)),
                        "{join:?} {shape:?}: pair ({row}, {column}) with a constant window was skipped"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(seeded(30))]
    /// I7 on per-band anchors: a random series, shape, thread count and anchor placement equal the
    /// exact reference on bits, two-sided and one-sided.
    #[test]
    fn a_random_series_equals_the_exact_profiles_on_per_band_anchors(
        (series, m) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
        seed in 0u64..1_000_000,
    ) {
        let (band, chunk, tile) = shape;
        let quantized = samples(&series, scale());
        let exact = sides_bits(&naive_two_sided(&quantized[..], m));
        let exact_left = profile_bits(&naive_left_profile(&quantized[..], m));
        let _guard = forced_route(per_band(OneLoop::On, seed));
        let (profile, _) = two_sided_profile_tiled(
            &series, nz(m), scale(), nz(threads), nz(band), nz(chunk), nz(tile),
        );
        prop_assert_eq!(sides_bits(&profile), exact, "{} threads {:?} seed {}", threads, shape, seed);
        prop_assert_eq!(
            walk(Join::Left, &series, m, threads, shape),
            exact_left,
            "left {} threads {:?} seed {}",
            threads,
            shape,
            seed
        );
    }
}

/// The route forced on with every tile on the exact i64 ticks (`wide`), with the anchors' seed.
fn wide_forced(seed: u64) -> RouteForce {
    RouteForce {
        wide: true,
        ..per_band(OneLoop::On, seed)
    }
}

/// A walk with a spike of `height` ticks at every `every`th sample: a window that holds a spike
/// spans more ticks than one anchor covers at the `m` the tests pick, and one that holds none
/// does not, so the series mixes tiles of both widths.
fn spiky(seed: u64, n: usize, every: usize, height: f64) -> Vec<f64> {
    lcg(seed, n, None)
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if i % every == every / 2 {
                x + height * MARKET_TICK
            } else {
                *x
            }
        })
        .collect()
}

/// A series the wide tests run, its window length and the force that runs it.
type WideCase = (&'static str, Vec<f64>, usize, RouteForce);

/// The series the wide tests run: walks forced wide, a ramp whose every window spans more ticks
/// than one anchor covers, and spiky series whose bands mix both widths (`m` 24).
fn wide_cases() -> Vec<WideCase> {
    vec![
        ("forced wide walk", lcg(91, 400, None), 12, wide_forced(0)),
        (
            "forced wide ties",
            lcg(92, 300, Some(3)),
            8,
            wide_forced(0x5EED),
        ),
        ("ramp", ramp(93, 500, 1.0e7), 24, per_band(OneLoop::On, 0)),
        (
            "spiky",
            spiky(94, 700, 97, 2.5e8),
            24,
            per_band(OneLoop::On, 0),
        ),
        (
            "spiky dense",
            spiky(95, 600, 41, 2.5e8),
            24,
            per_band(OneLoop::On, 0x5EED),
        ),
    ]
}

/// Tiles on the exact i64 ticks run on the loop: forced on, at 1, 2 and 8 threads, in every family
/// the route takes, every run equals the tiled path's bits and the loop ran wide tiles; a series
/// that mixes widths runs both, and the lanes handed down between its bands change no bit.
#[test]
fn wide_tiles_run_on_the_loop_and_equal_the_tiled_path() {
    for (what, series, m, forced) in wide_cases() {
        for join in JOINS {
            for shape in [(16, 16, 64), (64, 64, 128)] {
                let reference = {
                    let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                    walk(join, &series, m, 1, shape)
                };
                for threads in THREADS {
                    let guard = forced_route(forced);
                    let got = walk(join, &series, m, threads, shape);
                    let work = guard.work();
                    assert_eq!(
                        got, reference,
                        "{what} {join:?} {shape:?} {threads} threads"
                    );
                    assert!(work.wide_bands > 0, "{what} {join:?} {shape:?}: {work:?}");
                }
            }
        }
    }
    // The spiky series runs some bands narrow: fewer wide ones than the same shape forced wide.
    let series = spiky(94, 700, 97, 2.5e8);
    let mixed = {
        let guard = forced_route(per_band(OneLoop::On, 0));
        let _ = walk(Join::TwoSided, &series, 24, 1, (16, 16, 64));
        guard.work()
    };
    let all_wide = {
        let guard = forced_route(wide_forced(0));
        let _ = walk(Join::TwoSided, &series, 24, 1, (16, 16, 64));
        guard.work()
    };
    assert!(
        0 < mixed.wide_bands && mixed.wide_bands < all_wide.wide_bands,
        "{mixed:?} against {all_wide:?}"
    );
    assert!(mixed.lanes_handed_down > 0, "{mixed:?}");
}

/// The best pair and a handoff that always misses on wide tiles: no bit changes.
#[test]
fn wide_tiles_keep_the_bits_for_the_best_pair_and_with_missed_handoffs() {
    for (what, series, m, forced) in wide_cases() {
        let off = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            best_pair(&series, m, 1)
        };
        for threads in [1, 4] {
            let _guard = forced_route(forced);
            assert_eq!(
                best_pair(&series, m, threads),
                off,
                "{what} best pair {threads}"
            );
        }
        for join in JOINS {
            let reference = {
                let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                walk(join, &series, m, 1, (16, 16, 64))
            };
            let guard = forced_route(RouteForce {
                one_loop_no_handoff: true,
                ..forced
            });
            assert_eq!(
                walk(join, &series, m, 4, (16, 16, 64)),
                reference,
                "{what} {join:?} with missed handoffs"
            );
            assert!(guard.work().wide_bands > 0, "{what} {join:?}");
        }
    }
}

/// I1 on wide tiles: every cell of the triangle is skipped or computed, no pair is evaluated twice
/// or outside the triangle, and the pairs evaluated are the pairs counted.
#[test]
fn wide_tiles_cover_every_cell_once() {
    let m = 8;
    let zone = ladde::exclusion::zone(m);
    for windows in [2, 9, 17, 33, 65, 129, 200] {
        for (what, series) in area_series(windows, m) {
            for join in JOINS {
                for shape in SHAPES {
                    for threads in [1, 4] {
                        let guard = forced_route(RouteForce {
                            log_pairs: true,
                            no_constant_blocks: true,
                            ..wide_forced(0x5EED)
                        });
                        let _ = walk(join, &series, m, threads, shape);
                        let work = guard.work();
                        let at = format!(
                            "{join:?} {what} windows {windows} {threads} threads {shape:?}"
                        );
                        assert_eq!(
                            work.skipped_area + work.computed_area,
                            triangle_area(windows, zone),
                            "{at}: {work:?}"
                        );
                        let (pairs, total) = evaluated_pairs(&guard.pair_log());
                        assert_eq!(pairs.len(), total, "a pair evaluated twice: {at}");
                        assert_eq!(pairs.len(), work.computed_area, "{at}: {work:?}");
                        assert!(
                            pairs
                                .iter()
                                .all(|&(row, column)| row < windows && column + zone < row),
                            "a pair outside the triangle: {at}"
                        );
                    }
                }
            }
        }
    }
}

/// I5 on wide tiles: a repeating series with flat runs, pruned hard, evaluates every pair with a
/// constant window and returns the tiled path's bits.
#[test]
fn wide_tiles_never_skip_a_constant_window() {
    let m = 32;
    let zone = ladde::exclusion::zone(m);
    let base = lcg(61, 700, None);
    let mut series: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    for start in [100, 333, 1000] {
        series = flat(series, start, 3 * m);
    }
    let windows = series.len() - m + 1;
    let constant = |w: usize| series[w..w + m].iter().all(|&x| x == series[w]);
    for (join, shape) in JOINS
        .into_iter()
        .flat_map(|j| [(16, 16, 64), (64, 64, 128)].map(|s| (j, s)))
    {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, &series, m, 1, shape)
        };
        let guard = forced_route(RouteForce {
            log_pairs: true,
            no_constant_blocks: true,
            ..wide_forced(0x5EED)
        });
        assert_eq!(
            walk(join, &series, m, 1, shape),
            reference,
            "{join:?} {shape:?}"
        );
        let work = guard.work();
        assert!(
            work.skipped_area > 0,
            "{join:?} the series prunes: {work:?}"
        );
        let (pairs, _) = evaluated_pairs(&guard.pair_log());
        let seen: std::collections::BTreeSet<(usize, usize)> = pairs.into_iter().collect();
        for row in 0..windows {
            for column in 0..row.saturating_sub(zone) {
                if constant(row) || constant(column) {
                    assert!(
                        seen.contains(&(row, column)),
                        "{join:?} {shape:?}: pair ({row}, {column}) with a constant window was skipped"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(seeded(30))]
    /// I7 on wide tiles: a random series, shape and thread count equal the exact reference on
    /// bits, two-sided and one-sided.
    #[test]
    fn a_random_series_equals_the_exact_profiles_on_wide_tiles(
        (series, m) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
        seed in 0u64..1_000_000,
    ) {
        let (band, chunk, tile) = shape;
        let quantized = samples(&series, scale());
        let exact = sides_bits(&naive_two_sided(&quantized[..], m));
        let exact_left = profile_bits(&naive_left_profile(&quantized[..], m));
        let _guard = forced_route(wide_forced(seed));
        let (profile, _) = two_sided_profile_tiled(
            &series, nz(m), scale(), nz(threads), nz(band), nz(chunk), nz(tile),
        );
        prop_assert_eq!(sides_bits(&profile), exact, "{} threads {:?} seed {}", threads, shape, seed);
        prop_assert_eq!(
            walk(Join::Left, &series, m, threads, shape),
            exact_left,
            "left {} threads {:?} seed {}",
            threads,
            shape,
            seed
        );
    }
}
