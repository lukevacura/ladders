#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The one-loop route of the AB-join: forced on, it returns every output bit the tiled path
//! returns, with the PAA filter on in both.
//!
//! Every bit is equal at 1, 2 and 8 threads, for one neighbor and for `k` of 2, 3 and 8, on walks,
//! ties, repeats, constant runs, NaN gaps and a far best pair, with window counts around the band,
//! block and range boundaries on either side, rectangles that are wide, tall and tiny, and in
//! every run the route does not take (some diagonals only); the handoff forced to miss changes no
//! bit; a random pair of series equals the exact reference. The route's own counters are tested
//! too: every cell of the rectangle is skipped or computed, no pair is evaluated twice or missed,
//! and no pair with a constant window is skipped by the ladder (those tests force the constant
//! class's block skip off, which leaves such pairs out by rule). By default the route is taken at and above the
//! gate's window length and not below it.

#[path = "common/route.rs"]
mod route;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    ab_join, ab_join_report_tiled, ab_join_single_threaded, ab_join_topk, ab_join_topk_over_tiled,
    ab_join_topk_tiled,
};
use ladde::entry::Entry;
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

/// A rough series whose first stretch repeats at its end, far apart.
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

/// One named pair of series and their window length.
type Case = (String, Vec<f64>, Vec<f64>, usize);

/// The rectangles the tests sweep: window counts of `a` by those of `b` around the band (16, 64),
/// block (16, 64, 256) and tile boundaries on each side, with every kind of series; tiny n and m;
/// and longer series with a constant run at band and block edges.
fn cases() -> Vec<Case> {
    let mut out: Vec<Case> = Vec::new();
    let m = 8;
    for (wa, wb) in [
        (2, 9),
        (9, 2),
        (1, 1),
        (16, 17),
        (17, 16),
        (24, 24),
        (31, 33),
        (33, 31),
        (64, 65),
        (65, 64),
        (40, 100),
        (100, 40),
        (73, 129),
        (129, 73),
        (128, 128),
    ] {
        let (na, nb) = (wa + m - 1, wb + m - 1);
        let mut add = |name: &str, a: Vec<f64>, b: Vec<f64>| {
            out.push((format!("{name} wa={wa} wb={wb}"), a, b, m));
        };
        add("walk", lcg(1, na, None), lcg(101, nb, None));
        add("ties", lcg(2, na, Some(3)), lcg(102, nb, Some(3)));
        add("repeat", lcg(3, na, None), {
            // `b` holds a copy of `a`'s first stretch: an exact repeat across the series.
            let mut b = lcg(103, nb, None);
            let len = (m + 2).min(na).min(nb);
            let head: Vec<f64> = lcg(3, na, None)[..len].to_vec();
            b[nb - len..].copy_from_slice(&head);
            b
        });
        add(
            "flat",
            flat(lcg(4, na, None), na / 3, m + 3),
            flat(lcg(104, nb, None), nb / 2, m + 2),
        );
        add(
            "gaps",
            gaps(lcg(5, na, None), 5, 17),
            gaps(lcg(105, nb, None), 3, 13),
        );
        add("far", far_pair(na, m), far_pair(nb, m));
    }
    for (m, na, nb) in [
        (8, 8, 8),
        (8, 9, 10),
        (8, 10, 9),
        (2, 4, 9),
        (2, 9, 4),
        (3, 5, 12),
        (4, 13, 5),
    ] {
        out.push((
            format!("tiny walk {na}x{nb} m={m}"),
            lcg(6, na, None),
            lcg(106, nb, None),
            m,
        ));
        out.push((
            format!("tiny ties {na}x{nb} m={m}"),
            lcg(7, na, Some(2)),
            lcg(107, nb, Some(2)),
            m,
        ));
        out.push((
            format!("tiny gaps {na}x{nb} m={m}"),
            gaps(lcg(9, na, None), 1, 4),
            gaps(lcg(109, nb, None), 2, 3),
            m,
        ));
    }
    let (na, nb, m) = (230, 190, 12);
    for start in [7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128] {
        out.push((
            format!("flat at {start}"),
            flat(lcg(10, na, None), start, m + 6),
            flat(lcg(110, nb, None), start, m + 6),
            m,
        ));
    }
    out.push((
        "long walk".into(),
        lcg(11, 300, None),
        lcg(111, 260, None),
        24,
    ));
    out.push(("long far".into(), far_pair(300, 24), far_pair(270, 24), 24));
    out.push((
        "long ties".into(),
        lcg(12, 300, Some(4)),
        lcg(112, 280, Some(4)),
        24,
    ));
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
type Family = fn(&[f64], &[f64], usize, usize) -> Vec<Bits>;

fn nearest(a: &[f64], b: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    let mut out: Vec<Bits> = SHAPES
        .iter()
        .flat_map(|&(band, chunk, tile)| {
            let (profile, _) = ab_join_report_tiled(
                a,
                b,
                nz(m),
                scale(),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            profile_bits(&profile)
        })
        .collect();
    out.extend(profile_bits(&ab_join(
        a,
        b,
        nz(m),
        scale(),
        scale(),
        nz(threads),
    )));
    out.extend(profile_bits(&ab_join_single_threaded(
        a,
        b,
        nz(m),
        scale(),
        scale(),
    )));
    out
}

fn topk_of_shape(
    k: usize,
    (a, b): (&[f64], &[f64]),
    m: usize,
    threads: usize,
    (band, chunk, tile): (usize, usize, usize),
) -> Vec<Bits> {
    let (profile, _) = ab_join_topk_tiled(
        a,
        b,
        nz(m),
        scale(),
        scale(),
        nz(threads),
        nz(band),
        nz(chunk),
        nz(tile),
        nz(k),
    )
    .expect("small series fit");
    profile
        .iter()
        .flat_map(|(row, entries)| topk_bits(*row, entries))
        .collect()
}

fn topk_of(k: usize, a: &[f64], b: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    let mut out: Vec<Bits> = SHAPES
        .iter()
        .flat_map(|&shape| topk_of_shape(k, (a, b), m, threads, shape))
        .collect();
    let default =
        ab_join_topk(a, b, nz(m), scale(), scale(), nz(threads), nz(k)).expect("small series fit");
    out.extend(
        default
            .iter()
            .flat_map(|(row, entries)| topk_bits(*row, entries)),
    );
    out
}

fn topk2(a: &[f64], b: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    topk_of(2, a, b, m, threads)
}

fn topk3(a: &[f64], b: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    topk_of(3, a, b, m, threads)
}

fn topk8(a: &[f64], b: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    topk_of(8, a, b, m, threads)
}

/// One neighbor per row, but only some diagonals walked: the route does not take it.
fn some_diagonals(a: &[f64], b: &[f64], m: usize, threads: usize) -> Vec<Bits> {
    let (wa, wb) = (a.len() as i64 - m as i64 + 1, b.len() as i64 - m as i64 + 1);
    let diagonals: Vec<i64> = (1 - wa..wb).filter(|d| d % 3 != 0).collect();
    SHAPES
        .iter()
        .flat_map(|&(band, chunk, tile)| {
            let (profile, _) = ab_join_topk_over_tiled(
                a,
                b,
                nz(m),
                scale(),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(1),
                &diagonals,
            )
            .expect("small series fit");
            profile
                .iter()
                .flat_map(|(row, entries)| topk_bits(*row, entries))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `family` under `force` at `threads`.
fn run(
    family: Family,
    (a, b): (&[f64], &[f64]),
    m: usize,
    threads: usize,
    force: RouteForce,
) -> Vec<Bits> {
    let _guard = forced_route(force);
    family(a, b, m, threads)
}

/// Forced on, at 1, 2 and 8 threads, every run equals the tiled path's at one thread.
fn assert_family(name: &str, family: Family, filter: SurvivorForce, cases: &[&Case]) {
    for (what, a, b, m) in cases {
        let reference = run(family, (a, b), *m, 1, force(OneLoop::Off, filter));
        for threads in THREADS {
            assert_eq!(
                run(family, (a, b), *m, threads, force(OneLoop::On, filter)),
                reference,
                "{name} {filter:?} on {threads} threads, {what}"
            );
            if threads > 1 {
                assert_eq!(
                    run(family, (a, b), *m, threads, force(OneLoop::Off, filter)),
                    reference,
                    "{name} {filter:?} off {threads} threads, {what}"
                );
            }
        }
    }
}

#[test]
fn the_route_equals_the_tiled_path_in_the_join() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().collect();
    assert_family("join", nearest, SurvivorForce::Paa, &refs);
}

#[test]
fn the_route_equals_the_tiled_path_without_the_paa_filter() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(3).collect();
    for filter in [SurvivorForce::Gate, SurvivorForce::Off] {
        assert_family("join", nearest, filter, &refs);
    }
}

/// More than one neighbor per row: each row's `k`-th floor is the floor the ladder reads.
#[test]
fn the_route_equals_the_tiled_path_with_k_neighbors() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(2).collect();
    for (name, family) in [
        ("top-2", topk2 as Family),
        ("top-3", topk3),
        ("top-8", topk8),
    ] {
        assert_family(name, family, SurvivorForce::Paa, &refs);
    }
}

#[test]
fn runs_the_route_does_not_take_keep_todays_bits_with_it_forced() {
    let all = cases();
    let refs: Vec<&Case> = all.iter().step_by(4).collect();
    assert_family("some diagonals", some_diagonals, SurvivorForce::Paa, &refs);
}

/// A handoff that always misses starts every lane lazily and changes no bit.
#[test]
fn a_handoff_that_always_misses_changes_no_bit() {
    let all = cases();
    for (name, family) in [("join", nearest as Family), ("top-3", topk3)] {
        for (what, a, b, m) in all.iter().step_by(2) {
            let reference = run(
                family,
                (a, b),
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
                        run(family, (a, b), *m, threads, forced),
                        reference,
                        "{name} no_handoff {no_handoff} {threads} threads, {what}"
                    );
                }
            }
        }
    }
}

/// `k` neighbors per row (`None` for the one-neighbor join) over `(a, b)` in the shape
/// `(band, chunk, tile)`, as bits.
fn walk(
    k: Option<usize>,
    (a, b): (&[f64], &[f64]),
    m: usize,
    threads: usize,
    (band, chunk, tile): (usize, usize, usize),
) -> Vec<Bits> {
    match k {
        None => profile_bits(
            &ab_join_report_tiled(
                a,
                b,
                nz(m),
                scale(),
                scale(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .0,
        ),
        Some(k) => topk_of_shape(k, (a, b), m, threads, (band, chunk, tile)),
    }
}

/// The joins the counter tests run over: one neighbor and three.
const JOINS: [Option<usize>; 2] = [None, Some(3)];

/// Every pair of the rectangle is skipped on a proof or computed: the two areas add up to the
/// rectangle's, from the run's own counters. A pair that touches a non-finite window is a cell
/// like any other.
fn assert_area(
    join: Option<usize>,
    (a, b): (&[f64], &[f64]),
    m: usize,
    threads: usize,
    shape: (usize, usize, usize),
) {
    let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
    let _ = walk(join, (a, b), m, threads, shape);
    let work = guard.work();
    let area = (a.len() - m + 1) * (b.len() - m + 1);
    assert_eq!(
        work.skipped_area + work.computed_area,
        area,
        "{join:?} {}x{} m {m} {threads} threads, shape {shape:?}: {work:?}",
        a.len(),
        b.len()
    );
}

/// The pairs of one window count the area and evaluation tests sweep: a walk, one with NaN gaps,
/// one with a constant run, and one with ties everywhere.
fn area_series(wa: usize, wb: usize, m: usize) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    let (na, nb) = (wa + m - 1, wb + m - 1);
    vec![
        ("walk", lcg(21, na, None), lcg(121, nb, None)),
        (
            "gaps",
            gaps(lcg(23, na, None), 5, 11),
            gaps(lcg(123, nb, None), 4, 9),
        ),
        (
            "flat",
            flat(lcg(24, na, None), na / 3, m + 3),
            flat(lcg(124, nb, None), nb / 4, m + 3),
        ),
        ("ties", lcg(25, na, Some(3)), lcg(125, nb, Some(3))),
    ]
}

#[test]
fn every_cell_of_the_rectangle_is_skipped_or_computed() {
    let m = 8;
    for (wa, wb) in [
        (2, 9),
        (9, 2),
        (16, 17),
        (33, 64),
        (65, 40),
        (129, 200),
        (200, 129),
    ] {
        for (_, a, b) in area_series(wa, wb, m) {
            for join in JOINS {
                for threads in THREADS {
                    for shape in SHAPES {
                        assert_area(join, (&a, &b), m, threads, shape);
                    }
                }
            }
        }
    }
}

#[test]
fn the_ladder_evaluates_and_computes_where_it_applies() {
    let (a, b) = (lcg(22, 600, None), lcg(122, 700, None));
    for join in JOINS {
        let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let _ = walk(join, (&a, &b), 24, 1, (64, 64, 128));
        let work = guard.work();
        assert!(work.ladder_entries[0] > 0, "{join:?} {work:?}");
        assert!(
            work.survivor_runs > 0 && work.computed_area > 0,
            "{join:?} {work:?}"
        );
    }
}

/// A run the route does not take (some diagonals only) reports nothing of the ladder, and a join
/// with the route off reports none either.
#[test]
fn a_run_the_route_does_not_take_counts_no_ladder() {
    let (a, b) = (lcg(22, 600, None), lcg(122, 700, None));
    for route in [OneLoop::Off, OneLoop::On] {
        let guard = forced_route(force(route, SurvivorForce::Paa));
        let _ = some_diagonals(&a, &b, 24, 1);
        let work = guard.work();
        assert_eq!(work.ladder_entries, [0; 3], "{route:?}");
        assert_eq!(work.skipped_area + work.computed_area, 0, "{route:?}");
    }
    let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
    let _ = nearest(&a, &b, 24, 1);
    let work = guard.work();
    assert_eq!(work.ladder_entries, [0; 3]);
    assert_eq!(work.skipped_area + work.computed_area, 0);
}

/// By default ([`OneLoop::Auto`]) the join takes the route: the ladder runs, and the bits are the
/// same as with the route forced off.
#[test]
fn the_default_takes_the_route_for_the_join() {
    let (a, b) = (lcg(22, 600, None), lcg(122, 700, None));
    for (name, family) in [("join", nearest as Family), ("top-3", topk3)] {
        for threads in [1, 4] {
            let guard = forced_route(force(OneLoop::Auto, SurvivorForce::Paa));
            let got = family(&a, &b, 24, threads);
            let work = guard.work();
            drop(guard);
            assert!(work.ladder_entries[0] > 0, "{name} {threads} threads");
            assert!(work.skipped_area + work.computed_area > 0, "{name}");
            let reference = run(
                family,
                (&a, &b),
                24,
                threads,
                force(OneLoop::Off, SurvivorForce::Paa),
            );
            assert_eq!(got, reference, "{name} {threads} threads");
        }
    }
}

/// The production AB-join, with nothing forced: the gate's filter and the route's choice only.
fn production(a: &[f64], b: &[f64], m: usize, threads: usize, k: usize) -> Vec<Bits> {
    if k == 1 {
        let mut out = profile_bits(&ab_join(a, b, nz(m), scale(), scale(), nz(threads)));
        out.extend(profile_bits(&ab_join_single_threaded(
            a,
            b,
            nz(m),
            scale(),
            scale(),
        )));
        return out;
    }
    ab_join_topk(a, b, nz(m), scale(), scale(), nz(threads), nz(k))
        .expect("small series fit")
        .iter()
        .flat_map(|(row, entries)| topk_bits(*row, entries))
        .collect()
}

/// Production takes the route with no force at the gate's window length (the two-sided profile's:
/// 512 on one thread, 768 on more), for one neighbor and for `k`, and stays plain below it; the
/// bits equal the plain pass (the filter forced off) at 1, 2 and 8 threads either way.
#[test]
fn production_joins_take_the_route_from_the_gates_window_length() {
    for (m, taken) in [(400, false), (511, false), (1024, true)] {
        let (a, b) = (lcg(31, m + 1500, None), lcg(131, m + 1800, None));
        for k in [1, 3] {
            let plain = {
                let _guard = forced_route(force(OneLoop::Off, SurvivorForce::Off));
                production(&a, &b, m, 1, k)
            };
            for threads in THREADS {
                let guard = forced_route(RouteForce::default());
                let got = production(&a, &b, m, threads, k);
                let work = guard.work();
                drop(guard);
                let context = format!("m {m}, k {k}, {threads} threads: {work:?}");
                assert_eq!(work.ladder_entries[0] > 0, taken, "{context}");
                assert_eq!(
                    work.skipped_area + work.computed_area > 0,
                    taken,
                    "{context}"
                );
                assert_eq!(got, plain, "{context}");
            }
        }
    }
}

/// `b` holding the shape of `a` again, twice: the ladder skips a share of the rectangle.
fn repeating(m: usize) -> (Vec<f64>, Vec<f64>) {
    let base = lcg(41, 600, None);
    let a: Vec<f64> = base
        .iter()
        .enumerate()
        .map(|(i, v)| v + (i % 7) as f64 * 1e-6)
        .collect();
    let b: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 5) as f64 * 1e-6)
        .collect();
    let _ = m;
    (a, b)
}

/// A series with a repeating shape prunes: the ladder skips a share of the rectangle.
#[test]
fn a_repeating_series_is_pruned() {
    let (a, b) = repeating(32);
    for join in JOINS {
        let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let _ = walk(join, (&a, &b), 32, 1, (64, 64, 128));
        let work = guard.work();
        assert!(work.skipped_area > 0, "{join:?} skipped nothing: {work:?}");
        assert!(
            work.ladder_entries.iter().all(|&e| e > 0),
            "{join:?} {work:?}"
        );
    }
}

/// More windows than one range of slots (4096) holds: the floors are read, the ladder passed and
/// the floors raised more than once per band, at every shape. No bit changes, every cell is
/// covered once, and the ladder skips a share of it.
#[test]
fn a_rectangle_longer_than_one_range_changes_no_bit_and_covers_every_cell() {
    let m = 64;
    let base = lcg(51, 3000, None);
    let a: Vec<f64> = base
        .iter()
        .enumerate()
        .map(|(i, v)| v + (i % 5) as f64 * 1e-6)
        .collect();
    let b: Vec<f64> = base
        .iter()
        .chain(base.iter())
        .enumerate()
        .map(|(i, v)| v + (i % 3) as f64 * 1e-6)
        .collect();
    let area = (a.len() - m + 1) * (b.len() - m + 1);
    for shape in [(64, 64, 4096), (256, 512, 8192), (1024, 256, 1024)] {
        for join in JOINS {
            let reference = {
                let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                walk(join, (&a, &b), m, 1, shape)
            };
            for threads in [1, 4] {
                let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
                let got = walk(join, (&a, &b), m, threads, shape);
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

/// No pair is evaluated twice, each evaluated pair is a pair of the rectangle, and the pairs
/// evaluated are the pairs the run counts as computed; on a series that cannot be pruned (every
/// window constant) they are the whole rectangle, so none is missed either.
#[test]
fn no_pair_is_evaluated_twice_or_missed() {
    let m = 4;
    for (wa, wb) in [
        (2, 5),
        (5, 2),
        (9, 16),
        (17, 18),
        (33, 25),
        (40, 64),
        (65, 40),
    ] {
        let (na, nb) = (wa + m - 1, wb + m - 1);
        let constant = (vec![100.0; na], vec![101.0; nb]);
        for (what, a, b) in area_series(wa, wb, m)
            .into_iter()
            .chain([("constant", constant.0, constant.1)])
        {
            for (join, shape) in JOINS.into_iter().flat_map(|j| SHAPES.map(|s| (j, s))) {
                let guard = forced_route(RouteForce {
                    log_pairs: true,
                    no_constant_blocks: true,
                    ..force(OneLoop::On, SurvivorForce::Paa)
                });
                let _ = walk(join, (&a, &b), m, 1, shape);
                let work = guard.work();
                let (pairs, total) = evaluated_pairs(&guard.pair_log());
                let at = format!("{join:?} {what} {wa}x{wb} shape {shape:?}");
                assert_eq!(pairs.len(), total, "a pair evaluated twice: {at}");
                assert!(
                    pairs.iter().all(|&(row, column)| row < wa && column < wb),
                    "a pair outside the rectangle: {at}"
                );
                assert_eq!(pairs.len(), work.computed_area, "{at}: {work:?}");
                if what == "constant" {
                    assert_eq!(pairs.len(), wa * wb, "{at}");
                }
            }
        }
    }
}

/// I5 at the route: a constant window is never skipped. On a repeating series (which the ladder
/// prunes hard) with flat runs at several places on both sides, every pair with a constant window
/// is among the pairs the run evaluated, and the profile equals the tiled path's; on series of
/// one level the whole rectangle is evaluated, at windows that span every layer of the ladder.
#[test]
fn a_constant_window_is_never_skipped() {
    let m = 32;
    let (mut a, mut b) = repeating(m);
    for start in [100, 333] {
        a = flat(a, start, 3 * m);
    }
    for start in [50, 400, 900] {
        b = flat(b, start, 3 * m);
    }
    let (wa, wb) = (a.len() - m + 1, b.len() - m + 1);
    let constant = |s: &[f64], w: usize| s[w..w + m].iter().all(|&x| x == s[w]);
    assert!((0..wa).filter(|&w| constant(&a, w)).count() > 30);
    assert!((0..wb).filter(|&w| constant(&b, w)).count() > 60);
    for (join, shape) in JOINS
        .into_iter()
        .flat_map(|j| [(16, 16, 64), (64, 64, 128), (256, 256, 512)].map(|s| (j, s)))
    {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, (&a, &b), m, 1, shape)
        };
        let guard = forced_route(RouteForce {
            log_pairs: true,
            no_constant_blocks: true,
            ..force(OneLoop::On, SurvivorForce::Paa)
        });
        let got = walk(join, (&a, &b), m, 1, shape);
        assert_eq!(got, reference, "{join:?} {shape:?}");
        let work = guard.work();
        assert!(
            work.skipped_area > 0,
            "{join:?} the series prunes: {work:?}"
        );
        let (pairs, _) = evaluated_pairs(&guard.pair_log());
        let seen: std::collections::BTreeSet<(usize, usize)> = pairs.into_iter().collect();
        for row in 0..wa {
            for column in 0..wb {
                if constant(&a, row) || constant(&b, column) {
                    assert!(
                        seen.contains(&(row, column)),
                        "{join:?} {shape:?}: pair ({row}, {column}) with a constant window was skipped"
                    );
                }
            }
        }
    }
    // One level on each side: nothing can be skipped.
    let (level_a, level_b) = (vec![100.0; 500 + m - 1], vec![101.0; 600 + m - 1]);
    for join in JOINS {
        let guard = forced_route(RouteForce {
            log_pairs: true,
            no_constant_blocks: true,
            ..force(OneLoop::On, SurvivorForce::Paa)
        });
        let _ = walk(join, (&level_a, &level_b), m, 1, (64, 64, 256));
        let (pairs, total) = evaluated_pairs(&guard.pair_log());
        assert_eq!(pairs.len(), total, "{join:?}");
        assert_eq!(pairs.len(), 500 * 600, "{join:?}");
        assert_eq!(guard.work().skipped_area, 0, "{join:?}");
    }
}

// Joins no one anchor covers per side: each tile narrows to anchors of its own where a pair of
// anchors covers it and runs on the exact i64 ticks where none does, always on the route.

/// Every tile takes anchors of its own (placed from `seed`, `0` keeping them centered).
fn per_tile(seed: u64) -> RouteForce {
    RouteForce {
        per_tile: true,
        anchor_seed: seed,
        ..force(OneLoop::On, SurvivorForce::Paa)
    }
}

/// Every tile on the exact i64 ticks.
fn wide_forced(seed: u64) -> RouteForce {
    RouteForce {
        wide: true,
        ..per_tile(seed)
    }
}

/// A walk of `n` samples that climbs `slope` ticks a sample, so its ticks span more than one
/// anchor covers at `m` while a band and a tile span little.
fn ramp(seed: u64, n: usize, slope: f64) -> Vec<f64> {
    lcg(seed, n, None)
        .iter()
        .enumerate()
        .map(|(i, x)| x + i as f64 * slope * MARKET_TICK)
        .collect()
}

/// A walk with a spike of `height` ticks at every `every`th sample: a window that holds a spike
/// spans more ticks than one anchor covers, one that holds none does not.
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

/// Forced per-tile and forced wide, at 1, 2 and 8 threads and with the anchors centered and
/// placed anywhere, every join equals the tiled path's at one thread, for one neighbor and for
/// `k` of 3 and 8; the route ran wide tiles where it was forced wide.
#[test]
fn joins_with_no_whole_side_anchor_equal_the_tiled_path() {
    let all = cases();
    for (what, a, b, m) in all.iter().step_by(3) {
        for join in [None, Some(3), Some(8)] {
            for shape in [(16, 16, 64), (64, 64, 128)] {
                let reference = {
                    let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                    walk(join, (a, b), *m, 1, shape)
                };
                for threads in THREADS {
                    for (name, forced) in [
                        ("per tile", per_tile(0)),
                        ("per tile placed", per_tile(0x5EED)),
                        ("wide", wide_forced(0x5EED)),
                    ] {
                        let guard = forced_route(forced);
                        let got = walk(join, (a, b), *m, threads, shape);
                        let work = guard.work();
                        drop(guard);
                        assert_eq!(
                            got, reference,
                            "{name} {join:?} {shape:?} {threads} threads, {what}"
                        );
                        // A window length below the ladder's segments never takes the route.
                        if name == "wide" && *m >= 8 {
                            assert!(work.wide_bands > 0, "{what} {join:?} {shape:?}: {work:?}");
                        }
                    }
                }
            }
        }
    }
}

/// A join whose sides no one anchor covers takes the route by itself, with the ladder running, at
/// the default and forced on, and returns the plain pass's bits; narrow tiles hand lanes down
/// between anchors (moving them), and wide tiles run where the ticks span more than an anchor.
#[test]
fn a_join_no_whole_anchor_covers_runs_on_the_loop() {
    for (m, slope, na, nb) in [(24usize, 2.0e6, 700usize, 800usize), (8, 5.0e6, 500, 450)] {
        let (a, b) = (ramp(81, na, slope), ramp(181, nb, slope));
        for join in JOINS {
            for shape in [(16, 16, 64), (64, 64, 128)] {
                let off = {
                    let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
                    let got = walk(join, (&a, &b), m, 1, shape);
                    assert_eq!(guard.work().ladder_entries, [0; 3]);
                    got
                };
                for (name, route) in [("on", OneLoop::On), ("auto", OneLoop::Auto)] {
                    for threads in THREADS {
                        let guard = forced_route(force(route, SurvivorForce::Paa));
                        let got = walk(join, (&a, &b), m, threads, shape);
                        let work = guard.work();
                        drop(guard);
                        assert_eq!(
                            got, off,
                            "{name} {join:?} {shape:?} {threads} threads m {m}"
                        );
                        assert!(
                            work.ladder_entries[0] > 0 && work.computed_area > 0,
                            "{name} {join:?} {shape:?} m {m}: {work:?}"
                        );
                    }
                }
            }
        }
    }
    // Narrow tiles on anchors of their own hand lanes down and move them.
    let (a, b) = (ramp(82, 900, 3.0e6), ramp(182, 800, 3.0e6));
    for join in JOINS {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, (&a, &b), 24, 1, (16, 16, 64))
        };
        for seed in [0, 0x5EED] {
            let guard = forced_route(per_tile(seed));
            let got = walk(join, (&a, &b), 24, 1, (16, 16, 64));
            let work = guard.work();
            assert_eq!(got, reference, "{join:?} seed {seed}");
            assert!(
                work.lanes_handed_down > 0 && work.lanes_moved > 0,
                "{join:?} seed {seed}: {work:?}"
            );
            assert!(work.lanes_moved <= work.lanes_handed_down);
        }
    }
    // With one anchor per side no lane is moved.
    let (a, b) = (lcg(83, 900, None), lcg(183, 800, None));
    let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
    let _ = walk(None, (&a, &b), 24, 1, (16, 16, 64));
    let work = guard.work();
    assert!(work.lanes_handed_down > 0, "{work:?}");
    assert_eq!(work.lanes_moved, 0, "{work:?}");
}

/// Sides no one anchor covers, cut by a plan whose band is too tall for one anchor per side (a band
/// of 4096 rows), run in bands one grain tall: each tile takes anchors of its own, none runs on the
/// exact ticks, lanes are moved between the tiles' anchors, and no bit differs from the
/// plain pass's, at 1, 2 and 8 threads, for one neighbor and for `k` of 3.
#[test]
fn join_bands_cut_a_grain_tall_take_anchors_where_the_plans_band_has_none() {
    let (m, shape) = (24, (4096, 64, 4096));
    let (a, b) = (ramp(85, 7000, 2.6e4), ramp(185, 6500, 2.6e4));
    for join in JOINS {
        let reference = {
            let guard = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            let got = walk(join, (&a, &b), m, 1, shape);
            assert_eq!(guard.work().ladder_entries, [0; 3]);
            got
        };
        for threads in THREADS {
            let guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
            let got = walk(join, (&a, &b), m, threads, shape);
            let work = guard.work();
            drop(guard);
            assert_eq!(got, reference, "{join:?} {threads} threads");
            assert!(work.ladder_entries[0] > 0, "{join:?}: {work:?}");
            assert_eq!(
                work.skipped_area + work.computed_area,
                (a.len() - m + 1) * (b.len() - m + 1),
                "{join:?}: {work:?}"
            );
            assert_eq!(work.wide_bands, 0, "{join:?}: {work:?}");
            // A handoff is made only when the band above has finished first: certain on one thread.
            if threads == 1 {
                assert!(work.lanes_moved > 0, "{join:?}: {work:?}");
            }
        }
    }
}

/// A series that mixes widths runs both kinds of tile in one join, and the lanes handed down
/// between them (restarted at a change of width) change no bit, with the handoff missing or not.
#[test]
fn a_join_mixing_narrow_and_wide_tiles_keeps_every_bit() {
    let (a, b) = (spiky(94, 700, 97, 2.5e8), spiky(194, 650, 89, 2.5e8));
    let m = 24;
    for join in JOINS {
        let reference = {
            let _g = forced_route(force(OneLoop::Off, SurvivorForce::Paa));
            walk(join, (&a, &b), m, 1, (16, 16, 64))
        };
        let mixed = {
            let guard = forced_route(per_tile(0));
            let got = walk(join, (&a, &b), m, 1, (16, 16, 64));
            assert_eq!(got, reference, "{join:?}");
            guard.work()
        };
        let all_wide = {
            let guard = forced_route(wide_forced(0));
            let _ = walk(join, (&a, &b), m, 1, (16, 16, 64));
            guard.work()
        };
        assert!(
            0 < mixed.wide_bands && mixed.wide_bands < all_wide.wide_bands,
            "{join:?}: {mixed:?} against {all_wide:?}"
        );
        assert!(mixed.lanes_handed_down > 0, "{join:?}: {mixed:?}");
        for forced in [per_tile(0), per_tile(0x5EED), wide_forced(0)] {
            for threads in [1, 4] {
                let guard = forced_route(RouteForce {
                    one_loop_no_handoff: true,
                    ..forced
                });
                assert_eq!(
                    walk(join, (&a, &b), m, threads, (16, 16, 64)),
                    reference,
                    "{join:?} with missed handoffs, {threads} threads"
                );
                drop(guard);
            }
        }
    }
}

/// I1 and I2 where no whole anchor covers a side: every cell is skipped or computed once, no pair
/// is evaluated twice, and no pair with a constant window is skipped.
#[test]
fn joins_with_no_whole_side_anchor_cover_every_cell_once() {
    let m = 8;
    for (wa, wb) in [(2, 9), (17, 16), (33, 64), (129, 200)] {
        for (what, a, b) in area_series(wa, wb, m) {
            for join in JOINS {
                for shape in SHAPES {
                    for forced in [per_tile(0x5EED), wide_forced(0x5EED)] {
                        let guard = forced_route(RouteForce {
                            log_pairs: true,
                            no_constant_blocks: true,
                            ..forced
                        });
                        let _ = walk(join, (&a, &b), m, 1, shape);
                        let work = guard.work();
                        let (pairs, total) = evaluated_pairs(&guard.pair_log());
                        let at = format!("{join:?} {what} {wa}x{wb} {shape:?}");
                        assert_eq!(
                            work.skipped_area + work.computed_area,
                            wa * wb,
                            "{at}: {work:?}"
                        );
                        assert_eq!(pairs.len(), total, "a pair evaluated twice: {at}");
                        assert_eq!(pairs.len(), work.computed_area, "{at}: {work:?}");
                        if what == "flat" {
                            let constant =
                                |s: &[f64], w: usize| s[w..w + m].iter().all(|&x| x == s[w]);
                            let seen: std::collections::BTreeSet<(usize, usize)> =
                                pairs.into_iter().collect();
                            for row in 0..wa {
                                for column in 0..wb {
                                    if constant(&a, row) || constant(&b, column) {
                                        assert!(
                                            seen.contains(&(row, column)),
                                            "{at}: ({row}, {column}) skipped"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(seeded(40))]
    /// A random pair of series, a shape and a thread count: the join equals the exact reference
    /// on bits.
    #[test]
    fn a_random_pair_equals_the_exact_join(
        (a, m) in market_series_and_m(8..140),
        (b, _) in market_series_and_m(8..140),
        threads in prop::sample::select(THREADS.to_vec()),
        shape in prop::sample::select(SHAPES.to_vec()),
        k in prop_oneof![Just(1usize), Just(2), Just(3), Just(8)],
    ) {
        let expected_k = spill_run::exact_join_topk(&a, scale(), &b, scale(), m, k);
        let _guard = forced_route(force(OneLoop::On, SurvivorForce::Paa));
        let got: Vec<Vec<Bits>> = if k == 1 {
            let expected = spill_run::exact_join(&a, scale(), &b, scale(), m);
            let (profile, _) = ab_join_report_tiled(
                &a, &b, nz(m), scale(), scale(), nz(threads), nz(shape.0), nz(shape.1), nz(shape.2),
            );
            prop_assert_eq!(profile_bits(&profile), expected, "{} threads {:?}", threads, shape);
            return Ok(());
        } else {
            let (profile, _) = ab_join_topk_tiled(
                &a, &b, nz(m), scale(), scale(), nz(threads), nz(shape.0), nz(shape.1),
                nz(shape.2), nz(k),
            ).unwrap();
            profile
                .iter()
                .map(|(row, entries): &(u64, Vec<Entry>)| topk_bits(*row, entries))
                .collect()
        };
        prop_assert_eq!(got, expected_k, "k {} {} threads {:?}", k, threads, shape);
    }
}
