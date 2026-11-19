#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The batch left profile with the cold exact route against the exact reference, on bits:
//! outliers from just past the exact domain to the edge of the tick range, adjacent, opposite-sign
//! and edge outliers, gaps, under every forced spill route and at every thread count.

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    left_profile, left_profile_report, left_profile_report_tiled, left_profile_spill_report,
    left_topk_profile,
};
use ladde::entry::Entry;
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

/// Every spill route and thread count against the reference on `series`.
fn check(series: &[f64], m: usize, threads: &[usize]) {
    let expected = reference(series, m, unit());
    for spill in ROUTES {
        let _guard = forced_route(force(spill));
        for &count in threads {
            let (got, _) = left_profile_report(series, nz(m), unit(), nz(count));
            assert_eq!(bits(&got), expected, "m {m}, {spill:?}, {count} threads");
        }
        assert_eq!(bits(&left_profile(series, nz(m), unit())), expected);
    }
}

#[test]
fn outliers_of_every_size_match_the_reference_under_every_route() {
    let m = 20;
    for size in [1.0e9, 1.0e11, 1.0e13, 4.0e15, 8.0e15] {
        for (case, spikes) in [
            vec![(150, size)],
            vec![(150, -size)],
            vec![(150, size), (300, -size)],
            vec![(150, size), (151, size)],
            vec![(150, size), (151, -size)],
            vec![(0, size)],
            vec![(399, size)],
            vec![(0, size), (399, -size)],
            vec![(100, size), (260, size), (333, -size)],
        ]
        .into_iter()
        .enumerate()
        {
            let series = with_spikes(walk(400, 7 + case as u64), &spikes);
            check(&series, m, &[1, 3]);
        }
    }
}

#[test]
fn long_series_split_the_cold_pass_into_bands_and_match_the_reference() {
    for (m, spikes) in [
        (4usize, vec![(700, 3.0e11), (1500, -4.0e12), (1501, 9.0e12)]),
        (20, vec![(100, 2.0e12), (1400, -7.0e11), (2400, 5.0e13)]),
    ] {
        let series = with_spikes(walk(3000, m as u64), &spikes);
        check(&series, m, &[1, 2, 8]);
    }
}

#[test]
fn the_report_says_whether_the_cold_route_took_every_window_it_could() {
    let series = with_spikes(walk(1200, 2), &[(600, 1.0e12)]);
    let _guard = forced_route(force(SpillForce::On));
    let (_, _, spill) = left_profile_spill_report(&series, nz(20), unit(), nz(1));
    assert!(spill.engaged);
    assert_eq!(spill.outlier_ticks, 1);
    assert!(spill.spilled_windows >= 20 && spill.walked.cells > 0);
    drop(_guard);
    let _guard = forced_route(force(SpillForce::Off));
    let (_, _, spill) = left_profile_spill_report(&series, nz(20), unit(), nz(1));
    assert!(!spill.engaged);
}

#[test]
fn a_tiled_run_matches_too() {
    let series = with_spikes(walk(500, 3), &[(250, 6.0e11), (251, -6.0e11)]);
    let expected = reference(&series, 12, unit());
    for spill in ROUTES {
        let _guard = forced_route(force(spill));
        let (got, _) =
            left_profile_report_tiled(&series, nz(12), unit(), nz(3), nz(16), nz(2), nz(16));
        assert_eq!(bits(&got), expected, "{spill:?}");
    }
}

fn topk_bits(rows: &[(u64, Vec<Entry>)]) -> Vec<Vec<Bits>> {
    rows.iter()
        .map(|(row, entries)| {
            entries
                .iter()
                .map(|e| {
                    (
                        *row,
                        e.rho().to_bits(),
                        e.squared_distance().to_bits(),
                        e.index().get().unwrap_or(u64::MAX),
                    )
                })
                .collect()
        })
        .collect()
}

/// Every spill route, thread count and `k` of the top-k left profile against the reference.
fn check_topk(series: &[f64], m: usize, ks: &[usize], threads: &[usize]) {
    let (ticks, had) = ticks_of(series, unit());
    for &k in ks {
        let expected = spill_ref::left_topk(&ticks, &had, m, k);
        for spill in ROUTES {
            let _guard = forced_route(force(spill));
            for &count in threads {
                let got = left_topk_profile(series, nz(m), unit(), nz(count), nz(k));
                assert_eq!(
                    topk_bits(&got),
                    expected,
                    "m {m}, k {k}, {spill:?}, {count} threads"
                );
            }
        }
    }
}

#[test]
fn the_top_k_left_profile_matches_the_reference_for_any_k_on_every_route() {
    let m = 10;
    for spikes in [
        vec![(120, 5.0e11)],
        vec![(60, 2.0e12), (61, -2.0e12), (200, 1.0e13)],
        vec![(0, 3.0e11), (239, -3.0e11)],
    ] {
        let series = with_spikes(walk(240, 9), &spikes);
        check_topk(&series, m, &[1, 2, 3, 7, 50, 1000, usize::MAX], &[1, 3]);
    }
}

#[test]
fn a_print_in_the_first_window_alone_reaches_every_later_row_up_to_the_last() {
    // The first window is the only one the cold route ranks as a neighbor, and every row keeps
    // all its earlier windows, so the last row's list ends with it.
    let m = 10;
    for spike in [3.0e11, -3.0e11, 8.0e15] {
        let series = with_spikes(walk(240, 9), &[(0, spike)]);
        check_topk(&series, m, &[1000], &[1, 3]);
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
fn tie_heavy_series_with_outliers_match_the_reference_on_every_route() {
    let m = 6;
    let spikes = [(70, 6.0e11), (71, 6.0e11), (150, -9.0e11)];
    let series: Vec<Vec<f64>> = vec![
        vec![40.0; 220],
        (0..220).map(|i| [3.0, 8.0][i % 2]).collect(),
        (0..220).map(|i| [0.0, 2.0, 1.0, 3.0][i % 4]).collect(),
        (0..220)
            .map(|i| [1.0, 5.0, 2.0, 2.0, 9.0, 4.0, 4.0][i % 7])
            .collect(),
        few_levels(220, 3, &[5.0, 5.0, 9.0, 1.0], 1),
        few_levels(220, 4, &[5.0, 9.0, 1.0], 9),
        (0..220)
            .map(|i| [2.0, 7.0, 3.0, 9.0, 4.0][i % 5] + (i / 55) as f64 * 100.0)
            .collect(),
    ];
    for series in series {
        let series = with_spikes(series, &spikes);
        check(&series, m, &[1, 4]);
        check_topk(&series, m, &[1, 3, 9], &[1, 4]);
    }
}

#[test]
fn gaps_at_the_ends_and_inside_windows_that_hold_outliers_match_the_reference() {
    let m = 8;
    for (gaps, spikes) in [
        (vec![0usize, 1, 2], vec![(60, 4.0e11)]),
        (vec![117, 118, 119], vec![(60, 4.0e11)]),
        (vec![62], vec![(60, 4.0e11)]),
        (vec![60, 61, 90], vec![(59, 4.0e11), (95, -2.0e12)]),
        (vec![0, 119], vec![(0, 5.0e11), (1, 5.0e11), (119, -5.0e11)]),
        (vec![40, 41, 42, 43, 44, 45, 46, 47, 48], vec![(44, 7.0e11)]),
    ] {
        let mut series = with_spikes(walk(120, 12), &spikes);
        for &at in &gaps {
            series[at] = f64::NAN;
        }
        check(&series, m, &[1, 3]);
        check_topk(&series, m, &[1, 4], &[1, 3]);
    }
}

/// `series` with `shift` added to the `len` samples from `at`: an excursion whose inner windows
/// are finite today although they hold outliers.
fn with_excursion(mut series: Vec<f64>, at: usize, len: usize, shift: f64) -> Vec<f64> {
    for x in &mut series[at..at + len] {
        *x += shift;
    }
    series
}

#[test]
fn excursions_whose_inner_windows_are_finite_today_match_the_reference_on_every_route() {
    let m = 20;
    for (at, len, shift) in [
        (200usize, 60usize, 5.0e11),
        (200, 20, -5.0e11),
        (0, 40, 3.0e12),
        (360, 40, 9.0e13),
        (150, 25, 2.0e11),
    ] {
        let series = with_excursion(walk(400, at as u64), at, len, shift);
        check(&series, m, &[1, 3]);
        check_topk(&series, m, &[1, 4], &[1, 3]);
    }
    let series = with_excursion(with_spikes(walk(400, 5), &[(50, 6.0e11)]), 250, 70, -4.0e12);
    check(&series, m, &[1, 3]);
}

#[test]
fn the_callers_constant_flags_rank_the_same_on_every_route() {
    let m = 12;
    let base = walk(300, 3);
    let series = with_excursion(
        (0..300)
            .map(|i| if (i / 9) % 3 == 0 { 40.0 } else { base[i] })
            .collect(),
        120,
        30,
        8.0e11,
    );
    let series = with_spikes(series, &[(40, 5.0e12), (230, -5.0e12)]);
    let windows = series.len() + 1 - m;
    for round in 0..4u64 {
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
        let mut results = Vec::new();
        for spill in ROUTES {
            let _guard = forced_route(force(spill));
            for threads in [1, 3] {
                results.push(bits(
                    &left_profile_report(flagged, nz(m), unit(), nz(threads)).0,
                ));
            }
        }
        assert!(
            results.windows(2).all(|pair| pair[0] == pair[1]),
            "round {round}"
        );
    }
}

#[test]
fn once_the_cold_route_takes_the_outliers_the_hot_passes_run_on_one_anchor() {
    let m = 20;
    for (outlier, gaps) in [
        (600usize, vec![601usize, 602]),
        (600, vec![599]),
        (1, vec![25]),
        (3, vec![30, 31]),
        (1199, vec![]),
    ] {
        let mut series = with_spikes(walk(1200, 21), &[(outlier, 1.0e12)]);
        for &at in &gaps {
            series[at] = f64::NAN;
        }
        let guard = forced_route(force(SpillForce::On));
        let (on, batch, spill) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
        assert!(spill.engaged, "outlier {outlier}");
        assert_eq!((batch.tiles, batch.wide_tiles), (1, 0), "outlier {outlier}");
        drop(guard);
        let _guard = forced_route(force(SpillForce::Off));
        let (off, batch, spill) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
        assert!(!spill.engaged && batch.wide_tiles > 0, "outlier {outlier}");
        assert_eq!(bits(&on), bits(&off), "outlier {outlier}");
    }
}

#[test]
fn an_outlier_that_only_missing_windows_hold_leaves_the_cold_route_with_nothing_to_rank() {
    let m = 20;
    let mut series = with_spikes(walk(1200, 22), &[(3, 1.0e12)]);
    series[4] = f64::NAN;
    let _guard = forced_route(force(SpillForce::On));
    let (got, _, spill) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
    assert!(!spill.engaged);
    assert_eq!(spill.spilled_windows, 0);
    assert_eq!(bits(&got), reference(&series, m, unit()));
}

#[test]
fn an_excursion_fenced_by_gaps_is_taken_by_the_cold_route_though_no_window_is_too_wide() {
    // Every window across the excursion's edges has a gap, so the one window inside it is the only
    // one that holds outliers, and it is finite today. No longer than `m`, every tick of it is an
    // outlier.
    let m = 20;
    let mut series = with_excursion(walk(900, 31), 300, m, 2.0e12);
    series[299] = f64::NAN;
    series[300 + m] = f64::NAN;
    let guard = forced_route(force(SpillForce::On));
    let (on, batch, spill) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
    assert!(spill.engaged);
    assert_eq!(spill.outlier_ticks, m);
    assert_eq!(spill.spilled_windows, 1);
    assert_eq!((batch.tiles, batch.wide_tiles), (1, 0));
    assert_eq!(bits(&on), reference(&series, m, unit()));
    drop(guard);
    let _guard = forced_route(force(SpillForce::Off));
    let (off, _, spill) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
    assert!(!spill.engaged);
    assert_eq!(bits(&off), bits(&on));
}

#[test]
fn windows_too_wide_without_any_outlier_get_distances_on_every_route() {
    // A walk folded into a triangle of steps of 1.5e8 ticks: no tick is further than the limit
    // from the one before it, but most windows' variance numerators are past the exact domain.
    let m = 16;
    let series: Vec<f64> = walk(300, 8)
        .iter()
        .map(|x| ((x.rem_euclid(46.0) - 23.0).abs() - 11.5) * 1.5e8)
        .collect();
    check(&series, m, &[1, 3]);
    check_topk(&series, m, &[1, 3], &[1, 2]);
    for spill in ROUTES {
        let _guard = forced_route(force(spill));
        let (_, _, report) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
        assert_eq!(report.outlier_ticks, 0, "{spill:?}");
        assert!(report.spilled_windows > 100, "{spill:?}");
    }
}
