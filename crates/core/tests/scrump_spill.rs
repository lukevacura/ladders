#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The anytime profiles on series with windows only the cold route ranks: every pair that touches
//! a window with no distance in the hot passes is ranked when the profile is made, so after any
//! number of updates the profile is the exact one over the diagonals visited and those pairs, and
//! once every diagonal is visited it is the full exact profile, on bits, against an independent
//! arbitrary-precision reference, at any k, at any thread count and under every forced route.

#[path = "common/route.rs"]
mod route;
use route::spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    ab_join_spill_report, ab_join_topk, two_sided_spill_report, two_sided_topk_profile,
};
use ladde::entry::{Entry, TwoSidedTopk};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use ladde::scrump::{Scrump, ScrumpJoin, Shuffled};
use ladde::spill::lone_deviation_limit;
use spill_run::{Bits, anytime_join_topk, anytime_two_sided_topk, nz, unit, walk, with_spikes};

const THREADS: [usize; 3] = [1, 2, 7];
const KS: [usize; 3] = [1, 3, 1000];
const PERCENTAGE: f64 = 0.3;

fn entry_bits(row: u64, e: Entry) -> Bits {
    (
        row,
        e.rho().to_bits(),
        e.squared_distance().to_bits(),
        e.index().get().unwrap_or(u64::MAX),
    )
}

fn sides_bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<(Bits, Bits, Vec<Bits>)> {
    profile
        .iter()
        .map(|(row, s)| {
            (
                entry_bits(*row, s.left),
                entry_bits(*row, s.right),
                s.nearest.iter().map(|&e| entry_bits(*row, e)).collect(),
            )
        })
        .collect()
}

fn join_bits(profile: &[(u64, Vec<Entry>)]) -> Vec<Vec<Bits>> {
    profile
        .iter()
        .map(|(row, kept)| kept.iter().map(|&e| entry_bits(*row, e)).collect())
        .collect()
}

/// A shock `factor` times the deviation at which a lone tick takes its window past the exact
/// domain.
fn shock(m: usize, factor: f64) -> f64 {
    lone_deviation_limit(m) as f64 * factor
}

/// The self-join after every update, and once complete, against the reference, under every forced
/// route, at every thread count and every k of [`KS`]. The visiting order depends only on the
/// seed, so the reference is computed once per update.
fn check_self(series: &[f64], m: usize, scale: Scale) {
    let Ok(_) = Scrump::diagonals(series.len(), nz(m)) else {
        panic!("the series has diagonals");
    };
    for k in KS {
        let mut expected = Vec::new();
        for force in route::forces(0x5EED) {
            let _guard = forced_route(force);
            for threads in THREADS {
                let mut scrump = Scrump::new(
                    series,
                    nz(m),
                    scale,
                    nz(threads),
                    nz(k),
                    PERCENTAGE,
                    Shuffled::seeded(11),
                )
                .unwrap();
                let mut at = 0;
                loop {
                    if expected.len() == at {
                        let visited = scrump.visited();
                        expected.push(anytime_two_sided_topk(series, scale, m, k, visited));
                    }
                    let got = sides_bits(&scrump.profile().unwrap());
                    assert_eq!(got, expected[at], "k {k}, {force:?}, {threads} after {at}");
                    if scrump.is_complete() {
                        break;
                    }
                    scrump.update().unwrap();
                    at += 1;
                }
                let full = two_sided_topk_profile(series, nz(m), scale, nz(threads), nz(k));
                let done = sides_bits(&scrump.profile().unwrap());
                assert_eq!(
                    done,
                    sides_bits(&full.unwrap()),
                    "k {k}, {force:?}, {threads}"
                );
            }
        }
    }
}

/// [`check_self`] for the join of `a` against `b`.
fn check_join(a: &[f64], b: &[f64], m: usize) {
    for k in KS {
        let mut expected = Vec::new();
        for force in route::forces(0x5EED) {
            let _guard = forced_route(force);
            for threads in THREADS {
                let mut join = ScrumpJoin::new(
                    a,
                    b,
                    nz(m),
                    (unit(), unit()),
                    nz(threads),
                    nz(k),
                    PERCENTAGE,
                    Shuffled::seeded(5),
                )
                .unwrap();
                let mut at = 0;
                loop {
                    if expected.len() == at {
                        let sides = ((a, unit()), (b, unit()));
                        let visited = join.visited();
                        expected.push(anytime_join_topk(sides.0, sides.1, m, k, visited));
                    }
                    let got = join_bits(&join.profile().unwrap());
                    assert_eq!(got, expected[at], "k {k}, {force:?}, {threads} after {at}");
                    if join.is_complete() {
                        break;
                    }
                    join.update().unwrap();
                    at += 1;
                }
                let full = ab_join_topk(a, b, nz(m), unit(), unit(), nz(threads), nz(k)).unwrap();
                let done = join_bits(&join.profile().unwrap());
                assert_eq!(done, join_bits(&full), "k {k}, {force:?}, {threads}");
            }
        }
    }
}

/// How many windows the batch two-sided profile ranks cold, and how many ticks are outliers.
fn spilled(series: &[f64], m: usize, scale: Scale) -> (usize, usize) {
    let _guard = forced_route(RouteForce::default());
    let (_, _, report) = two_sided_spill_report(series, nz(m), scale, nz(1));
    (report.spilled_windows, report.outlier_ticks)
}

/// How many windows of either series the batch join ranks cold.
fn join_spilled(a: &[f64], b: &[f64], m: usize) -> usize {
    let _guard = forced_route(RouteForce::default());
    let (_, _, report) = ab_join_spill_report(a, b, nz(m), unit(), unit(), nz(1));
    report.spilled_windows
}

#[test]
fn shocks_from_just_past_the_exact_domain_to_a_thousand_times_it() {
    let m = 8;
    for factor in [1.5, 4.0, 30.0, 1000.0] {
        let size = shock(m, factor);
        for (case, spikes) in [
            vec![(40, size)],
            vec![(40, -size), (41, size)],
            vec![(0, size), (89, -size)],
        ]
        .into_iter()
        .enumerate()
        {
            let series = with_spikes(walk(90, 3 + case as u64), &spikes);
            assert!(spilled(&series, m, unit()).0 > 0, "×{factor}");
            check_self(&series, m, unit());
        }
    }
}

/// `parts` joined, each preceded by a stretch of `gap` samples of a walk.
fn spaced(parts: &[Vec<f64>], gap: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for (at, part) in parts.iter().enumerate() {
        out.extend(walk(gap, 40 + at as u64));
        out.extend(part);
    }
    out.extend(walk(gap, 99));
    out
}

#[test]
fn exact_ties_between_a_pair_ranked_up_front_and_a_visited_one_go_to_the_smaller_index() {
    // The pattern scaled far past the exact domain and shifted inside it correlates exactly with
    // the pattern itself: a tie between a window ranked up front and one visited with its
    // diagonal, on either side of a row and across a join.
    let m = 8;
    let pattern = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, -2.0, 6.0];
    let shifted = |by: f64| pattern.iter().map(|x| x + by).collect::<Vec<_>>();
    let scaled = pattern.iter().map(|x| x * 1e9).collect::<Vec<_>>();
    let series = spaced(
        &[
            shifted(1050.0),
            scaled.clone(),
            shifted(1000.0),
            shifted(1070.0),
        ],
        12,
    );
    check_self(&series, m, unit());
    let a = spaced(&[shifted(1050.0)], 12);
    let b = spaced(&[scaled, shifted(1000.0)], 12);
    check_join(&a, &b, m);
}

#[test]
fn an_excursion_ranks_only_its_edges_up_front_and_visits_its_inner_windows() {
    // The first `m` ticks of the excursion and the first `m` after it are outliers, but a window
    // inside it has a distance in the hot passes, so only the windows across its edges are ranked
    // up front.
    let m = 8;
    let mut series = walk(90, 6);
    for x in &mut series[40..60] {
        *x += shock(m, 4.0);
    }
    let (windows, outliers) = spilled(&series, m, unit());
    assert_eq!(outliers, 2 * m);
    assert!(windows > 0);
    check_self(&series, m, unit());
}

#[test]
fn windows_past_the_exact_domain_without_any_outlier() {
    // A walk folded into a triangle of steps of 1.5e8 ticks: no tick is further than the limit
    // from the one before it, but most windows' variance numerators are past the exact domain.
    let m = 16;
    let series: Vec<f64> = walk(70, 8)
        .iter()
        .map(|x| ((x.rem_euclid(46.0) - 23.0).abs() - 11.5) * 1.5e8)
        .collect();
    let (windows, outliers) = spilled(&series, m, unit());
    assert_eq!(outliers, 0);
    assert!(windows > 20, "{windows}");
    check_self(&series, m, unit());
}

#[test]
fn a_join_ranks_the_windows_of_either_series_up_front() {
    let m = 6;
    for factor in [1.5, 1000.0] {
        let size = shock(m, factor);
        let (calm_a, calm_b) = (walk(50, 21), walk(40, 22));
        let (shocked_a, shocked_b) = (
            with_spikes(calm_a.clone(), &[(20, size)]),
            with_spikes(calm_b.clone(), &[(0, -size), (30, size)]),
        );
        for (a, b) in [
            (&shocked_a, &calm_b),
            (&calm_a, &shocked_b),
            (&shocked_a, &shocked_b),
        ] {
            assert!(join_spilled(a, b, m) > 0, "×{factor}");
            check_join(a, b, m);
        }
    }
}

#[test]
fn before_the_first_update_the_profile_holds_the_pairs_ranked_up_front() {
    let m = 8;
    let series = with_spikes(walk(60, 4), &[(30, shock(m, 2.0))]);
    let _guard = forced_route(RouteForce::default());
    let scrump = Scrump::new(
        &series,
        nz(m),
        unit(),
        nz(1),
        nz(2),
        PERCENTAGE,
        Shuffled::seeded(1),
    )
    .unwrap();
    assert!(scrump.visited().is_empty());
    let got = sides_bits(&scrump.profile().unwrap());
    assert_eq!(got, anytime_two_sided_topk(&series, unit(), m, 2, &[]));
    // Every window reaches one that holds the shock outside its exclusion zone.
    for (row, (_, _, nearest)) in got.iter().enumerate() {
        assert!(!nearest.is_empty(), "{row}");
    }
}

/// The fuzz target's reproducer: a subnormal sample then zeros, at the smallest scale, so the
/// windows that hold the first sample have no distance in the hot passes.
#[test]
fn a_subnormal_then_zeros_converges_to_the_exact_profile() {
    let scale = Scale::new(5e-324).unwrap();
    let mut series = vec![0.0; 60];
    series[0] = 1e-310;
    let m = 26;
    assert!(spilled(&series, m, scale).0 > 0);
    check_self(&series, m, scale);
}
