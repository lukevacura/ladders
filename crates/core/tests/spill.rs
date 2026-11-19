#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The cold exact route against the exact reference with no width cutoff, on bits: outliers from
//! just past the exact domain to the edge of the tick range, adjacent, opposite-sign and edge
//! outliers, gaps, constant runs and exact repeats, and series built so that candidates differ by
//! less than the ranking estimate's own error.

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::route::{RouteForce, SpillForce, forced_route};
use ladde::spill::{Survey, Width, lone_deviation_limit};
use spill_run::*;

fn on() -> RouteForce {
    RouteForce {
        spill: SpillForce::On,
        ..RouteForce::default()
    }
}

/// The cold route on `series` against the reference, on bits; returns the run.
fn check(series: &[f64], m: usize) -> Outcome {
    let _guard = forced_route(on());
    let outcome = left_spill(series, m, unit());
    let expected = reference(series, m, unit());
    assert_eq!(bits(&outcome.entries), expected, "m {m}");
    outcome
}

#[test]
fn outliers_of_every_size_match_the_reference() {
    let m = 20;
    for size in [
        1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15, 4.0e15, 8.0e15,
    ] {
        for (case, spikes) in [
            vec![(150, size)],
            vec![(150, -size)],
            vec![(150, size), (300, -size)],
            vec![(150, size), (151, size)],
            vec![(150, size), (151, -size)],
            vec![(150, size), (150 + m, size)],
            vec![(0, size)],
            vec![(399, size)],
            vec![(0, size), (399, -size)],
            vec![(100, size), (260, size), (333, -size)],
        ]
        .into_iter()
        .enumerate()
        {
            let series = with_spikes(walk(400, 7 + case as u64), &spikes);
            let outcome = check(&series, m);
            assert!(outcome.outlier_ticks >= 1, "size {size} case {case}");
            assert!(outcome.walked.cells > 0, "size {size} case {case}");
            assert_eq!(outcome.width, Some(Width::Narrow128));
        }
    }
}

#[test]
fn a_tick_is_an_outlier_exactly_when_a_lone_window_holding_it_leaves_the_exact_domain() {
    for m in [2usize, 3, 8, 20, 64, 1000] {
        let limit = lone_deviation_limit(m);
        let variance =
            |deviation: u64| (m as u128 - 1) * u128::from(deviation) * u128::from(deviation);
        assert!(variance(limit) < 1 << 63, "m {m}");
        assert!(variance(limit + 1) >= 1 << 63, "m {m}");
        for (shock, flagged) in [(limit, false), (limit + 1, true)] {
            let mut series = vec![1000.0; 300];
            series[150] += shock as f64;
            let (ticks, had) = ticks_of(&series, unit());
            let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
            assert_eq!(survey.outliers()[150], flagged, "m {m}, shock {shock}");
            assert_eq!(
                survey.outliers().iter().filter(|&&f| f).count(),
                usize::from(flagged)
            );
        }
    }
}

#[test]
fn a_window_holding_a_huge_tick_is_spilled_only_when_it_holds_an_outlier_or_is_too_wide() {
    let m = 20;
    let series = with_spikes(walk(400, 3), &[(200, 1.0e12)]);
    let (ticks, had) = ticks_of(&series, unit());
    let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
    let expected: Vec<bool> = (0..=400 - m).map(|i| (181..=200).contains(&i)).collect();
    assert_eq!(survey.spilled(), expected.as_slice());
}

#[test]
fn adjacent_opposite_sign_and_edge_outliers_at_every_position_near_the_ends() {
    let m = 12;
    for at in (0..6).chain(94..100) {
        for sign in [1.0, -1.0] {
            let series = with_spikes(walk(100, at as u64 + 1), &[(at, sign * 5.0e11)]);
            check(&series, m);
        }
    }
    for at in 0..8 {
        let series = with_spikes(
            walk(100, 31 + at as u64),
            &[(at, 2.0e12), (at + 1, -2.0e12), (99 - at, 9.0e14)],
        );
        check(&series, m);
    }
}

#[test]
fn gaps_at_tile_boundaries_and_at_both_ends_match_the_reference() {
    // Tiles are 16 lanes wide, chunks 2: put gaps and outliers on and around those boundaries.
    let m = 8;
    for boundary in [14usize, 15, 16, 17, 31, 32, 33, 47, 48] {
        for (gap_at, spike_at) in [(boundary, 70), (60, boundary), (boundary, boundary + 1)] {
            let mut series = with_spikes(walk(120, boundary as u64), &[(spike_at, 7.0e11)]);
            series[gap_at] = f64::NAN;
            check(&series, m);
        }
    }
    let mut series = with_spikes(walk(120, 77), &[(60, 3.0e12), (90, -3.0e12)]);
    for at in [0usize, 1, 2, 117, 118, 119] {
        series[at] = f64::NAN;
    }
    check(&series, m);
    let mut series = with_spikes(walk(120, 78), &[(0, 3.0e12)]);
    series[119] = f64::NAN;
    series[1] = f64::NAN;
    check(&series, m);
    let all_gaps_but_outlier: Vec<f64> = (0..60)
        .map(|i| if i == 30 { 1.0e13 } else { f64::NAN })
        .collect();
    check(&all_gaps_but_outlier, 6);
}

#[test]
fn a_gap_inside_a_window_that_holds_an_outlier_leaves_it_without_a_distance() {
    let m = 10;
    let mut series = with_spikes(walk(200, 4), &[(100, 1.0e12)]);
    series[103] = f64::NAN;
    let outcome = check(&series, m);
    for row in 94..=103usize {
        assert_eq!(outcome.entries[row].1.index().get(), None, "row {row}");
    }
}

fn pattern(seed: u64, len: usize) -> Vec<i64> {
    walk(len, seed).iter().map(|&x| x as i64 - 1000).collect()
}

/// `(scale, level, perturbation)` of one copy of the pattern.
type Copy = (i64, i64, Option<(usize, i64)>);

/// `copies` of one pattern with a filler walk between them: `(scale, level, perturbation)` per
/// copy, the perturbation `(position, delta)` added after scaling.
fn copies_series(seed: u64, m: usize, copies: &[Copy]) -> (Vec<f64>, usize) {
    let base = pattern(seed, m);
    let filler = walk(copies.len() * (m + 4), seed ^ 0xABCD);
    let mut series = Vec::new();
    let mut starts = Vec::new();
    for (index, &(scale, level, perturb)) in copies.iter().enumerate() {
        series.extend(
            filler[index * (m + 4)..index * (m + 4) + m / 2 + 2]
                .iter()
                .copied(),
        );
        starts.push(series.len());
        let mut copy: Vec<i64> = base.iter().map(|&x| x * scale + level).collect();
        if let Some((at, delta)) = perturb {
            copy[at % m] += delta;
        }
        series.extend(copy.iter().map(|&x| x as f64));
    }
    series.extend(filler[..m / 2 + 2].iter().copied());
    (series, starts.len())
}

#[test]
fn candidates_within_the_estimate_band_are_ordered_exactly() {
    let mut in_band = 0;
    let mut exact_ties = 0;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in 0..160u64 {
        let m = 5 + (next() % 12) as usize;
        let count = 3 + (next() % 5) as usize;
        let copies: Vec<Copy> = (0..count)
            .map(|_| {
                let scale = match next() % 5 {
                    0 => 1,
                    1 => 1 << 20,
                    2 => 1 << 34,
                    3 => 1 << 42,
                    _ => 1 << 46,
                };
                let scale = if next() % 7 == 0 { -scale } else { scale };
                let level = match next() % 3 {
                    0 => 0,
                    1 => (next() % 1_000_000) as i64,
                    _ => (next() % (1 << 50)) as i64,
                };
                let perturb = match next() % 3 {
                    0 => None,
                    1 => Some(((next() % 64) as usize, 1)),
                    _ => Some(((next() % 64) as usize, -1)),
                };
                (scale, level, perturb)
            })
            .collect();
        let (series, _) = copies_series(case, m, &copies);
        let outcome = check(&series, m);
        assert!(outcome.walked.cells > 0);
        let (ticks, had) = ticks_of(&series, unit());
        in_band += spill_ref::rows_decided_by_the_exact_order(&ticks, &had, m);
        exact_ties += spill_ref::rows_with_an_exact_tie(&ticks, &had, m);
    }
    assert!(
        in_band > 200,
        "the generator reached the band {in_band} times"
    );
    assert!(exact_ties > 20, "exact ties {exact_ties}");
}

#[test]
fn a_hot_row_between_an_exact_copy_and_a_spilled_near_copy() {
    // The row's pattern at scale one is hot; its exact copy is hot and decides by estimate
    // alone, while the huge near copies are cold and sit within the band of it.
    let mut crossed = 0;
    for seed in 0..40u64 {
        let m = 6 + (seed % 9) as usize;
        let copies = [
            (1, 0, None),
            (1 << 46, 3, Some((2, 1))),
            (1 << 46, 0, Some((1, -1))),
            (1 << 46, 5, None),
            (1, 7, None),
        ];
        let (series, _) = copies_series(seed + 500, m, &copies);
        check(&series, m);
        let (ticks, had) = ticks_of(&series, unit());
        crossed += spill_ref::rows_decided_by_the_exact_order(&ticks, &had, m);
    }
    assert!(crossed > 40, "{crossed}");
}

fn few_levels(len: usize, seed: u64, levels: &[f64]) -> Vec<f64> {
    walk(len, seed)
        .iter()
        .map(|&x| levels[(x as usize / 3) % levels.len()])
        .collect()
}

#[test]
fn tie_heavy_series_with_outliers_match_the_reference() {
    let m = 8;
    let period =
        |p: usize| -> Vec<f64> { (0..200).map(|i| ((i % p) * (i % p) + 1) as f64).collect() };
    let mut cases: Vec<Vec<f64>> = vec![
        vec![5.0; 200],
        (0..200).map(|i| if i < 100 { 3.0 } else { 9.0 }).collect(),
        period(2),
        period(4),
        period(7),
        few_levels(200, 9, &[1.0, 4.0, 9.0]),
        few_levels(200, 10, &[0.0, 1.0]),
    ];
    for series in cases.clone() {
        for spikes in [
            vec![(100, 5.0e11)],
            vec![(60, 5.0e11), (130, 5.0e11)],
            vec![(60, 5.0e11), (130, -5.0e11), (131, 5.0e11)],
        ] {
            cases.push(with_spikes(series.clone(), &spikes));
        }
    }
    let mut with_ties = 0;
    for series in &cases {
        check(series, m);
        let (ticks, had) = ticks_of(series, unit());
        with_ties += spill_ref::rows_with_an_exact_tie(&ticks, &had, m);
    }
    assert!(with_ties > 1000, "{with_ties}");
}

#[test]
fn an_outlier_repeated_at_an_offset_matches_its_own_earlier_copy() {
    // Two identical shocks: each later window ties exactly with the earlier one's.
    let m = 10;
    let mut series = walk(400, 21);
    for (at, shock) in [(80usize, 6.0e12), (230, 6.0e12), (320, 6.0e12)] {
        for k in 0..m {
            series[at + k] += shock * (k + 1) as f64;
        }
    }
    check(&series, m);
}

#[test]
fn the_256_bit_tier_matches_the_reference() {
    // m·R passes 2^64 with an outlier near the end of the tick range at a window of 2048.
    let m = 2048;
    let series = with_spikes(walk(m + 700, 13), &[(2300, 8.0e15), (2400, -8.0e15)]);
    let outcome = check(&series, m);
    assert_eq!(outcome.width, Some(Width::Wide256));
    let near = with_spikes(walk(m + 700, 14), &[(2500, 1.0e13)]);
    let outcome = check(&near, m);
    assert_eq!(outcome.width, Some(Width::Narrow128));
}

#[test]
fn a_series_beyond_the_exact_domain_has_no_cold_route() {
    // R near 2^54 with m past 2^32 is out of reach for any real series, so test the boundary of
    // the tier choice directly.
    let ticks = [-(1i64 << 53), 1 << 53];
    assert_eq!(
        ladde::spill::width_for(&ticks, 1 << 31),
        Some(Width::Wide256)
    );
    assert_eq!(ladde::spill::width_for(&ticks, 1 << 33), None);
    assert_eq!(ladde::spill::width_for(&[], 5), Some(Width::Narrow128));
}

#[test]
fn clean_series_never_engage_and_equal_todays_routes() {
    let _guard = forced_route(RouteForce::default());
    for seed in 0..4u64 {
        let series = walk(300, 100 + seed);
        let outcome = left_spill(&series, 12, unit());
        assert_eq!(outcome.spilled_windows, 0);
        assert_eq!(outcome.walked.cells, 0);
        assert_eq!(
            bits(&outcome.entries),
            bits(&left_today(&series, 12, unit()))
        );
    }
}

/// The windows of `series` holding any of `positions`.
fn holding(len: usize, m: usize, positions: &[usize]) -> Vec<bool> {
    (0..=len - m)
        .map(|i| positions.iter().any(|&p| (i..i + m).contains(&p)))
        .collect()
}

#[test]
fn shocks_under_the_outlier_limit_cold_equal_todays_bits_where_todays_routes_rank_them_too() {
    // A 3e8 shock leaves every window's variance under 2^63 (and under the outlier limit at
    // m = 20, about 7e8): today's routes rank every pair on the exact i64 pass, and the cold
    // route, asked for the windows holding the shock, must reproduce their bits.
    let m = 20;
    let _guard = forced_route(RouteForce::default());
    for (seed, at) in [(1u64, 90usize), (2, 233), (3, 10)] {
        let series = with_spikes(walk(400, seed), &[(at, 3.0e8), (at + 111, -2.8e8)]);
        let (ticks, had) = ticks_of(&series, unit());
        let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
        assert!(survey.outliers().iter().all(|&flag| !flag));
        assert!(survey.spilled().iter().all(|&flag| !flag));
        let spilled = holding(400, m, &[at, at + 111]);
        let survey = Survey::with_spilled(&ticks, &had, nz(m), spilled.clone())
            .unwrap()
            .unwrap();
        let today = left_today(&series, m, unit());
        let mut entries = today.clone();
        let walked = survey.merge_left(&mut entries).unwrap();
        assert!(walked.cells > 0);
        assert_eq!(bits(&entries), bits(&today));
        // And from the pairs among the other windows alone.
        let hot =
            spill_ref::left_profile_where(&ticks, &had, m, &|i, j| !spilled[i] && !spilled[j]);
        let mut entries: Vec<_> = hot.iter().map(|&b| (b.0, entry_of(b))).collect();
        survey.merge_left(&mut entries).unwrap();
        assert_eq!(bits(&entries), bits(&today));
    }
}

#[test]
fn the_walk_counts_are_what_the_pass_walks() {
    let _guard = forced_route(on());
    for (seed, spikes) in [
        (1u64, vec![(150usize, 1.0e12)]),
        (2, vec![(10, 1.0e12), (11, 1.0e12), (300, -1.0e12)]),
        (3, vec![(0, 1.0e12), (399, 1.0e12)]),
        (4, (0..30).map(|k| (13 * k + 5, 4.0e11)).collect()),
    ] {
        let mut series = with_spikes(walk(400, seed), &spikes);
        series[200] = f64::NAN;
        let outcome = left_spill(&series, 20, unit());
        let (ticks, had) = ticks_of(&series, unit());
        let survey = Survey::new(&ticks, &had, nz(20)).unwrap().unwrap();
        assert_eq!(outcome.walked, survey.walk().unwrap(), "seed {seed}");
    }
}

#[test]
fn every_window_cold_equals_the_reference_on_clean_data() {
    let m = 12;
    for seed in 0..4u64 {
        let series = walk(260, 40 + seed);
        let (ticks, had) = ticks_of(&series, unit());
        let windows = ticks.len() - m + 1;
        let survey = Survey::with_spilled(&ticks, &had, nz(m), vec![true; windows])
            .unwrap()
            .unwrap();
        let mut entries: Vec<_> = (0..windows as u64)
            .map(|row| (row, ladde::entry::Entry::NONE))
            .collect();
        survey.merge_left(&mut entries).unwrap();
        assert_eq!(bits(&entries), reference(&series, m, unit()));
        assert_eq!(bits(&entries), bits(&left_today(&series, m, unit())));
    }
}

#[test]
fn any_split_between_hot_and_cold_windows_equals_the_reference() {
    let m = 9;
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in 0..60u64 {
        let mut series = if case % 3 == 0 {
            few_levels(180, case, &[1.0, 2.0, 5.0])
        } else {
            walk(180, case)
        };
        if case % 4 == 0 {
            series[(next() % 180) as usize] = f64::NAN;
        }
        if case % 5 == 0 {
            series[(next() % 180) as usize] += 3.0e12;
        }
        let (ticks, had) = ticks_of(&series, unit());
        let windows = ticks.len() - m + 1;
        let spilled: Vec<bool> = (0..windows).map(|_| next() % 4 == 0).collect();
        let hot =
            spill_ref::left_profile_where(&ticks, &had, m, &|i, j| !spilled[i] && !spilled[j]);
        let mut entries: Vec<_> = hot.iter().map(|&b| (b.0, entry_of(b))).collect();
        let survey = Survey::with_spilled(&ticks, &had, nz(m), spilled.clone())
            .unwrap()
            .unwrap();
        survey.merge_left(&mut entries).unwrap();
        assert_eq!(bits(&entries), reference(&series, m, unit()), "case {case}");
    }
}

#[test]
fn a_constant_plateau_far_from_the_level_ranks_exactly_among_constant_windows() {
    // Windows inside the plateau are constant and hold outliers (cold); the rest are constant and
    // hot: constant pairs of both kinds, and constant against valid, in one series.
    let m = 8;
    let mut series = vec![5.0; 200];
    for x in &mut series[60..110] {
        *x = 4.0e12;
    }
    for (at, x) in series.iter_mut().enumerate().skip(150) {
        *x = 5.0 + (at % 3) as f64;
    }
    let outcome = check(&series, m);
    assert!(outcome.walked.cells > 0);
}
