#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The two-sided profile, the top-k profiles and the AB-join with the cold exact route against the
//! exact reference, on bits: outliers from just past the exact domain to the edge of the tick
//! range, in either series of a join, adjacent, opposite-sign and edge outliers, gaps, excursions
//! whose inner windows are finite today, exact ties, under every forced spill route and at every
//! thread count and tiling.

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    ab_join, ab_join_report_tiled, ab_join_single_threaded, ab_join_spill_report, ab_join_topk,
    ab_join_topk_tiled, two_sided_profile, two_sided_profile_tiled, two_sided_spill_report,
    two_sided_topk_profile, two_sided_topk_profile_tiled,
};
use ladde::entry::{Entry, TwoSided, TwoSidedTopk};
use ladde::route::{RouteForce, SpillForce, forced_route};
use ladde::stats::Series;
use spill_run::*;

fn force(spill: SpillForce) -> RouteForce {
    RouteForce {
        spill,
        ..RouteForce::default()
    }
}

const ROUTES: [SpillForce; 3] = [SpillForce::Auto, SpillForce::On, SpillForce::Off];
const TILINGS: [(usize, usize, usize); 2] = [(3, 2, 5), (64, 8, 64)];

fn entry_bits(row: u64, e: Entry) -> Bits {
    (
        row,
        e.rho().to_bits(),
        e.squared_distance().to_bits(),
        e.index().get().unwrap_or(u64::MAX),
    )
}

fn sides_bits(profile: &[(u64, TwoSided)]) -> Vec<[Bits; 3]> {
    profile
        .iter()
        .map(|&(row, s)| [s.left, s.right, s.nearest].map(|e| entry_bits(row, e)))
        .collect()
}

fn topk_sides_bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<(Bits, Bits, Vec<Bits>)> {
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

fn join_topk_bits(profile: &[(u64, Vec<Entry>)]) -> Vec<Vec<Bits>> {
    profile
        .iter()
        .map(|(row, kept)| kept.iter().map(|&e| entry_bits(*row, e)).collect())
        .collect()
}

/// The two-sided profile and its top-k twin, under every spill route, thread count and tiling.
fn check_two_sided(series: &[f64], m: usize, ks: &[usize]) {
    let expected = exact_two_sided(series, unit(), m);
    for spill in ROUTES {
        let _guard = forced_route(force(spill));
        for threads in [1usize, 3] {
            let got = two_sided_profile(series, nz(m), unit(), nz(threads));
            assert_eq!(sides_bits(&got), expected, "m {m}, {spill:?}, {threads}");
            for (band, chunk, tile) in TILINGS {
                let (got, _) = two_sided_profile_tiled(
                    series,
                    nz(m),
                    unit(),
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                );
                assert_eq!(sides_bits(&got), expected, "m {m}, {spill:?}, {threads}");
            }
        }
    }
    for &k in ks {
        let expected = exact_two_sided_topk(series, unit(), m, k);
        for spill in ROUTES {
            let _guard = forced_route(force(spill));
            for threads in [1usize, 3] {
                let got = two_sided_topk_profile(series, nz(m), unit(), nz(threads), nz(k));
                assert_eq!(
                    topk_sides_bits(&got.unwrap()),
                    expected,
                    "m {m}, k {k}, {spill:?}, {threads}"
                );
                for (band, chunk, tile) in TILINGS {
                    let (got, _) = two_sided_topk_profile_tiled(
                        series,
                        nz(m),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                        nz(k),
                    )
                    .unwrap();
                    assert_eq!(
                        topk_sides_bits(&got),
                        expected,
                        "m {m}, k {k}, {spill:?}, {threads}"
                    );
                }
            }
        }
    }
}

/// The join of `a` against `b` and its top-k twin, under every spill route, thread count, tiling
/// and the single pass.
fn check_join(a: &[f64], b: &[f64], m: usize, ks: &[usize]) {
    let expected = exact_join(a, unit(), b, unit(), m);
    for spill in ROUTES {
        let _guard = forced_route(force(spill));
        assert_eq!(
            bits(&ab_join_single_threaded(a, b, nz(m), unit(), unit())),
            expected,
            "m {m}, {spill:?}, one pass"
        );
        for threads in [1usize, 3] {
            let got = ab_join(a, b, nz(m), unit(), unit(), nz(threads));
            assert_eq!(bits(&got), expected, "m {m}, {spill:?}, {threads}");
            for (band, chunk, tile) in TILINGS {
                let (got, _) = ab_join_report_tiled(
                    a,
                    b,
                    nz(m),
                    unit(),
                    unit(),
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                );
                assert_eq!(bits(&got), expected, "m {m}, {spill:?}, {threads}");
            }
        }
    }
    for &k in ks {
        let expected = exact_join_topk(a, unit(), b, unit(), m, k);
        for spill in ROUTES {
            let _guard = forced_route(force(spill));
            for threads in [1usize, 3] {
                let got = ab_join_topk(a, b, nz(m), unit(), unit(), nz(threads), nz(k)).unwrap();
                assert_eq!(
                    join_topk_bits(&got),
                    expected,
                    "m {m}, k {k}, {spill:?}, {threads}"
                );
                for (band, chunk, tile) in TILINGS {
                    let (got, _) = ab_join_topk_tiled(
                        a,
                        b,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                        nz(k),
                    )
                    .unwrap();
                    assert_eq!(
                        join_topk_bits(&got),
                        expected,
                        "m {m}, k {k}, {spill:?}, {threads}"
                    );
                }
            }
        }
    }
}

/// Whether all of `results` are equal.
fn same<T: PartialEq>(results: &[T]) -> bool {
    results.windows(2).all(|pair| pair[0] == pair[1])
}

fn excursion(mut series: Vec<f64>, at: usize, len: usize, shift: f64) -> Vec<f64> {
    for x in &mut series[at..at + len] {
        *x += shift;
    }
    series
}

#[test]
fn two_sided_profiles_match_the_reference_for_outliers_of_every_size_and_place() {
    let m = 10;
    for size in [1.0e9, 3.0e11, 4.0e13, 8.0e15] {
        for (case, spikes) in [
            vec![(120, size)],
            vec![(60, size), (61, -size), (170, size)],
            vec![(0, size), (199, -size)],
            vec![(40, size), (40 + m, size), (160, -size)],
        ]
        .into_iter()
        .enumerate()
        {
            let series = with_spikes(walk(200, 11 + case as u64), &spikes);
            check_two_sided(&series, m, &[1, 3, 9]);
        }
    }
}

#[test]
fn joins_match_the_reference_for_outliers_in_either_series_or_both() {
    let m = 8;
    let clean = |len, seed| walk(len, seed);
    for size in [2.0e9, 5.0e11, 6.0e13] {
        let spiked = |len, seed, at| with_spikes(walk(len, seed), &[(at, size), (at + 17, -size)]);
        check_join(&spiked(150, 3, 70), &clean(120, 4), m, &[1, 3, 8, 1000]);
        check_join(&clean(120, 5), &spiked(150, 6, 30), m, &[1, 3, 8, 1000]);
        check_join(&spiked(110, 7, 20), &spiked(140, 8, 100), m, &[1, 4]);
        let same = spiked(130, 9, 50);
        check_join(&same, &same, m, &[1, 3]);
    }
    let edges = with_spikes(walk(90, 12), &[(0, 9.0e11), (89, -9.0e11)]);
    check_join(&edges, &walk(80, 13), m, &[2]);
    check_join(&walk(80, 14), &edges, m, &[2]);
}

#[test]
fn windows_too_wide_without_an_outlier_get_neighbors_in_both_profiles_and_joins() {
    let m = 12;
    let noise = |seed| -> Vec<f64> {
        walk(150, seed)
            .iter()
            .map(|x| (x % 23.0 - 11.0) * 5.0e7)
            .collect()
    };
    check_two_sided(&noise(3), m, &[2]);
    check_join(&noise(4), &walk(100, 5), m, &[2]);
    check_join(&walk(100, 6), &noise(7), m, &[2]);
}

#[test]
fn excursions_whose_inner_windows_are_finite_today_match_the_reference() {
    let m = 14;
    for (at, len, shift) in [
        (70usize, 50usize, 4.0e11),
        (0, 30, -2.0e12),
        (150, 40, 7.0e13),
    ] {
        let series = excursion(walk(200, at as u64 + 1), at, len, shift);
        check_two_sided(&series, m, &[1, 4]);
        check_join(&series, &walk(120, 2), m, &[1, 4]);
        check_join(&walk(120, 3), &series, m, &[1, 4]);
    }
}

#[test]
fn gaps_around_outliers_match_the_reference() {
    let m = 8;
    for (gaps, spikes) in [
        (vec![0usize, 1, 2], vec![(60, 4.0e11)]),
        (vec![117, 118, 119], vec![(60, 4.0e11)]),
        (vec![62], vec![(60, 4.0e11)]),
        (vec![60, 61, 90], vec![(59, 4.0e11), (95, -2.0e12)]),
        (vec![0, 119], vec![(0, 5.0e11), (1, 5.0e11), (119, -5.0e11)]),
    ] {
        let mut series = with_spikes(walk(120, 12), &spikes);
        for &at in &gaps {
            series[at] = f64::NAN;
        }
        check_two_sided(&series, m, &[1, 4]);
        check_join(&series, &walk(90, 15), m, &[1, 4]);
        check_join(&walk(90, 16), &series, m, &[1, 4]);
    }
}

/// A series of few levels with runs, so that exact ties abound.
fn few_levels(len: usize, seed: u64, levels: &[f64], run: usize) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|i| {
            if i % run == 0 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
            }
            levels[(state % levels.len() as u64) as usize]
        })
        .collect()
}

#[test]
fn tie_heavy_series_with_outliers_match_the_reference() {
    let m = 6;
    let spikes = [(50, 6.0e11), (51, 6.0e11), (110, -9.0e11)];
    let series: Vec<Vec<f64>> = vec![
        vec![40.0; 140],
        (0..140).map(|i| [3.0, 8.0][i % 2]).collect(),
        (0..140).map(|i| [0.0, 2.0, 1.0, 3.0][i % 4]).collect(),
        (0..140)
            .map(|i| [1.0, 5.0, 2.0, 2.0, 9.0, 4.0, 4.0][i % 7])
            .collect(),
        few_levels(140, 3, &[5.0, 5.0, 9.0, 1.0], 1),
        few_levels(140, 4, &[5.0, 9.0, 1.0], 9),
    ];
    for series in series {
        let spiked = with_spikes(series.clone(), &spikes);
        check_two_sided(&spiked, m, &[1, 3, 9]);
        check_join(&spiked, &series, m, &[1, 3]);
        check_join(&series, &spiked, m, &[1, 3]);
        check_join(&spiked, &spiked, m, &[2]);
    }
}

#[test]
fn the_reports_say_whether_the_cold_route_took_every_window_it_could() {
    let m = 20;
    let series = with_spikes(walk(1200, 2), &[(600, 1.0e12)]);
    let other = walk(900, 3);
    for (spill, engaged) in [(SpillForce::On, true), (SpillForce::Off, false)] {
        let _guard = forced_route(force(spill));
        let (_, batch, report) = two_sided_spill_report(&series, nz(m), unit(), nz(1));
        assert_eq!(report.engaged, engaged, "two-sided, {spill:?}");
        assert!(report.walked.cells > 0, "two-sided, {spill:?}");
        assert_eq!(batch.wide_tiles == 0, engaged, "two-sided, {spill:?}");
        let (_, batch, report) =
            ab_join_spill_report(&series, &other, nz(m), unit(), unit(), nz(1));
        assert_eq!(report.engaged, engaged, "join, {spill:?}");
        assert!(report.walked.cells > 0, "join, {spill:?}");
        assert_eq!(batch.wide_tiles == 0, engaged, "join, {spill:?}");
        let (_, _, report) = ab_join_spill_report(&other, &series, nz(m), unit(), unit(), nz(1));
        assert_eq!(report.engaged, engaged, "join, b spilled, {spill:?}");
        assert!(report.walked.cells > 0, "join, b spilled, {spill:?}");
    }
}

#[test]
fn clean_series_never_reach_the_cold_route() {
    let _guard = forced_route(force(SpillForce::On));
    let series = walk(500, 9);
    let (_, _, report) = two_sided_spill_report(&series, nz(20), unit(), nz(2));
    assert_eq!(report, ladde::batch::SpillReport::default());
    let (_, _, report) =
        ab_join_spill_report(&series, &walk(400, 8), nz(20), unit(), unit(), nz(2));
    assert_eq!(report, ladde::batch::SpillReport::default());
}

#[test]
fn the_callers_constant_flags_rank_the_same_on_every_route() {
    let m = 12;
    let base = walk(240, 3);
    let series = excursion(
        (0..240)
            .map(|i| if (i / 9) % 3 == 0 { 40.0 } else { base[i] })
            .collect(),
        100,
        30,
        8.0e11,
    );
    let series = with_spikes(series, &[(40, 5.0e12), (190, -5.0e12)]);
    let windows = series.len() + 1 - m;
    for round in 0..3u64 {
        let flags: Vec<bool> = (0..windows)
            .map(|i| {
                (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .rotate_left(round as u32 * 7)
                    % 5
                    < round + 1
            })
            .collect();
        let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
        let other = walk(160, 21);
        let mut two_sided = Vec::new();
        let mut joins = Vec::new();
        let mut joins_b = Vec::new();
        for spill in ROUTES {
            let _guard = forced_route(force(spill));
            for threads in [1, 3] {
                let profile = two_sided_profile(flagged, nz(m), unit(), nz(threads));
                two_sided.push(sides_bits(&profile));
                let join = ab_join(flagged, &other, nz(m), unit(), unit(), nz(threads));
                joins.push(bits(&join));
                let join = ab_join(&other, flagged, nz(m), unit(), unit(), nz(threads));
                joins_b.push(bits(&join));
            }
        }
        assert!(same(&two_sided), "two-sided, round {round}");
        assert!(same(&joins), "join, round {round}");
        assert!(same(&joins_b), "join, b flagged, round {round}");
    }
}

/// A walk of whole cents around 5000.00, as ticks at scale one.
fn cent_walk(len: usize, seed: u64) -> Vec<f64> {
    walk(len, seed)
        .iter()
        .map(|x| 500_000.0 + (x - 1000.0))
        .collect()
}

#[test]
fn prints_of_one_and_a_half_to_a_thousand_times_the_price_match_the_reference() {
    let m = 50;
    for factor in [1.5, 2.0, 10.0, 100.0, 1000.0] {
        let print = |series: &mut Vec<f64>, at: usize| series[at] *= factor;
        let mut series = cent_walk(260, 4);
        for at in [70, 71, 200] {
            print(&mut series, at);
        }
        let mut other = cent_walk(180, 5);
        print(&mut other, 90);
        check_two_sided(&series, m, &[1, 4]);
        check_join(&series, &other, m, &[1, 3]);
        check_join(&other, &series, m, &[1, 3]);
    }
}

#[test]
fn the_first_window_of_one_series_pairs_with_the_last_window_of_the_other() {
    // `a` opens with a copy of `b`'s last window, spike and all: the pair is on the join's last
    // diagonal, and an exact match.
    let m = 8;
    let mut b = walk(120, 41);
    b[117] += 6.0e11;
    let mut a = walk(100, 42);
    a[..m].copy_from_slice(&b[b.len() - m..]);
    check_join(&a, &b, m, &[1, 3]);
    check_join(&b, &a, m, &[1, 3]);
}

#[test]
fn small_random_series_with_outliers_match_the_reference_at_every_edge() {
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let levels = [0.0, 1.0, 2.0, 5.0, 5.0, 7.0];
    for case in 0..120 {
        let m = 2 + (next() % 4) as usize;
        let len = 24 + (next() % 30) as usize;
        let mut random = |len: usize, spikes: u64| {
            let mut series: Vec<f64> = (0..len)
                .map(|_| levels[(next() % levels.len() as u64) as usize])
                .collect();
            for _ in 0..spikes {
                let at = (next() % len as u64) as usize;
                let size = [3.0e11, -7.0e11, 2.0e12][(next() % 3) as usize];
                series[at] += size;
            }
            series
        };
        let spikes = 1 + case % 4;
        let (series, other) = (random(len, spikes), random(len - 5, case % 3));
        check_two_sided(&series, m, &[2]);
        check_join(&series, &other, m, &[2]);
        check_join(&other, &series, m, &[2]);
    }
}
