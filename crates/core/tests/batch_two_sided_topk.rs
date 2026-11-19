#![cfg(feature = "oracle")]

//! The two-sided batch profile with `k` neighbors equals `naive_two_sided_topk` on bits, at every
//! thread count and tiling, on series with NaN gaps, flat runs, exact repeats and heavy ties, for
//! `k` below, at and above the candidate count; with `k` of one it is `two_sided_profile`.

use core::num::NonZeroUsize;

use ladde::batch::{
    two_sided_profile, two_sided_profile_tiled, two_sided_topk_profile,
    two_sided_topk_profile_tiled,
};
use ladde::entry::{Entry, TwoSidedTopk};
use ladde::naive::naive_two_sided_topk;
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series_and_m};

type EntryBits = (u64, u64, u64);

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn entry_bits(entry: Entry) -> EntryBits {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

type RowBits = (u64, [EntryBits; 2], Vec<EntryBits>);

fn bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<RowBits> {
    profile
        .iter()
        .map(|(row, sides)| {
            (
                *row,
                [sides.left, sides.right].map(entry_bits),
                sides.nearest.iter().copied().map(entry_bits).collect(),
            )
        })
        .collect()
}

fn naive(series: &[f64], m: usize, scale: Scale, k: usize) -> Vec<(u64, TwoSidedTopk)> {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    naive_two_sided_topk(&samples, m, nz(k))
}

/// Tile shapes `(band_rows, chunk_lanes, tile_lanes)`, down to one lane and past the series.
fn shapes(n: usize) -> [(usize, usize, usize); 6] {
    [
        (1, 1, 1),
        (2, 3, 6),
        (3, 4, 8),
        (8, 5, 5),
        (n + 5, 2, 4),
        (4, n + 5, n + 5),
    ]
}

const KS: [usize; 5] = [1, 2, 3, 8, 200];

fn assert_matches_naive_at(series: &[f64], m: usize, scale: Scale, ks: &[usize]) {
    for &k in ks {
        let expected = bits(&naive(series, m, scale, k));
        for threads in [1, 2, 7] {
            for (band_rows, chunk_lanes, tile_lanes) in shapes(series.len()) {
                let (actual, _) = two_sided_topk_profile_tiled(
                    series,
                    nz(m),
                    scale,
                    nz(threads),
                    nz(band_rows),
                    nz(chunk_lanes),
                    nz(tile_lanes),
                    nz(k),
                )
                .unwrap();
                assert_eq!(
                    bits(&actual),
                    expected,
                    "k {k}, threads {threads}, shape {:?}",
                    (band_rows, chunk_lanes, tile_lanes)
                );
            }
            assert_eq!(
                bits(&two_sided_topk_profile(series, nz(m), scale, nz(threads), nz(k)).unwrap()),
                expected,
                "k {k}, default tiling, threads {threads}"
            );
        }
    }
}

fn assert_matches_naive(series: &[f64], m: usize, scale: Scale) {
    assert_matches_naive_at(series, m, scale, &KS);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(16))]

    #[test]
    fn topk_equals_naive_on_market_series((series, m) in market_series_and_m(0..70)) {
        assert_matches_naive(&series, m, Scale::new(MARKET_TICK).unwrap());
    }
}

#[test]
fn a_repeating_series_ties_to_the_smaller_index() {
    let series: Vec<f64> = (0..60)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    assert_matches_naive(&series, 4, Scale::new(1.0).unwrap());
}

#[test]
fn a_constant_series_equals_naive() {
    assert_matches_naive(&[7.0; 50], 4, Scale::new(1.0).unwrap());
}

#[test]
fn constant_runs_between_levels_equal_naive() {
    let series: Vec<f64> = (0..150).map(|k| f64::from(((k / 13) % 3) as u8)).collect();
    assert_matches_naive(&series, 5, Scale::new(1.0).unwrap());
}

#[test]
fn period_k_repeats_equal_naive() {
    for period in [2, 4, 7] {
        let series: Vec<f64> = (0..90)
            .map(|k| f64::from(((k % period) * 5 % 7) as u8))
            .collect();
        assert_matches_naive(&series, 6, Scale::new(1.0).unwrap());
    }
}

#[test]
fn few_level_series_equal_naive() {
    let mut state: u64 = 0xD1B5_4A32_D192_ED03;
    for (levels, m) in [(2u64, 3usize), (2, 8), (3, 4)] {
        let series: Vec<f64> = (0..120)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) % levels) as f64
            })
            .collect();
        assert_matches_naive(&series, m, Scale::new(1.0).unwrap());
    }
}

#[test]
fn exact_repeats_with_offsets_equal_naive() {
    let piece: Vec<f64> = (0..11).map(|k| f64::from((k * k % 7) as u8)).collect();
    let series: Vec<f64> = (0..9)
        .flat_map(|copy| piece.iter().map(move |x| x + f64::from(copy % 3)))
        .collect();
    assert_matches_naive(&series, 4, Scale::new(1.0).unwrap());
}

#[test]
fn flat_runs_and_nan_gaps_equal_naive() {
    let mut series: Vec<f64> = (0..80).map(|k| f64::from((k * 7 % 11) as u8)).collect();
    series[10..30].fill(5.0);
    series[40] = f64::NAN;
    series[55..58].fill(f64::NAN);
    assert_matches_naive(&series, 5, Scale::new(1.0).unwrap());
}

#[test]
fn gaps_at_the_very_start_and_end_equal_naive() {
    let mut series: Vec<f64> = (0..70).map(|k| f64::from((k * 5 % 13) as u8)).collect();
    series[..3].fill(f64::NAN);
    series[66..].fill(f64::NAN);
    assert_matches_naive(&series, 4, Scale::new(1.0).unwrap());
}

/// A deterministic pseudo-random walk, wider than the right side's lock groups.
fn walk(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut level = 0.0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 56) % 3) as f64 - 1.0;
            level
        })
        .collect()
}

#[test]
fn a_series_across_several_column_groups_equals_naive() {
    assert_matches_naive_at(&walk(1300), 4, Scale::new(1.0).unwrap(), &[1, 3, 8]);
}

#[test]
fn a_series_too_wide_for_one_anchor_equals_naive() {
    let series: Vec<f64> = walk(120)
        .iter()
        .enumerate()
        .map(|(k, x)| x + if k % 2 == 0 { 0.0 } else { 1.5e9 })
        .collect();
    assert_matches_naive(&series, 4, Scale::new(1.0).unwrap());
}

#[test]
fn k_of_one_is_the_two_sided_profile_on_bits() {
    let series = walk(300);
    let scale = Scale::new(1.0).unwrap();
    let single = two_sided_profile(&series, nz(6), scale, nz(2));
    let topk = two_sided_topk_profile(&series, nz(6), scale, nz(2), nz(1)).unwrap();
    assert_eq!(single.len(), topk.len());
    for ((row, sides), (topk_row, topk_sides)) in single.iter().zip(&topk) {
        assert_eq!(row, topk_row);
        assert_eq!(entry_bits(sides.left), entry_bits(topk_sides.left));
        assert_eq!(entry_bits(sides.right), entry_bits(topk_sides.right));
        match sides.nearest.index().get() {
            None => assert!(topk_sides.nearest.is_empty()),
            Some(_) => assert_eq!(
                topk_sides
                    .nearest
                    .iter()
                    .copied()
                    .map(entry_bits)
                    .collect::<Vec<_>>(),
                [entry_bits(sides.nearest)]
            ),
        }
    }
}

#[test]
fn the_tiled_k_of_one_is_the_tiled_two_sided_profile() {
    let series = walk(200);
    let scale = Scale::new(1.0).unwrap();
    let (single, _) = two_sided_profile_tiled(&series, nz(5), scale, nz(3), nz(4), nz(3), nz(9));
    let (topk, _) =
        two_sided_topk_profile_tiled(&series, nz(5), scale, nz(3), nz(4), nz(3), nz(9), nz(1))
            .unwrap();
    for ((_, sides), (_, topk_sides)) in single.iter().zip(&topk) {
        assert_eq!(
            topk_sides.nearest.first().copied().map(entry_bits),
            sides
                .nearest
                .index()
                .get()
                .map(|_| entry_bits(sides.nearest))
        );
    }
}

/// Window 5 of `0 1 0 1 ...` at `m = 2`: the exact ties at distance zero are windows 1, 3, 7 and 9
/// (smaller index first, across both sides), then the tied nearest of the rest.
#[test]
fn a_fixed_tie_case_has_literal_bits() {
    let series: Vec<f64> = (0..12).map(|k| f64::from((k % 2) as u8)).collect();
    let topk =
        two_sided_topk_profile(&series, nz(2), Scale::new(1.0).unwrap(), nz(1), nz(4)).unwrap();
    let (row, sides) = &topk[5];
    assert_eq!(*row, 5);
    let got: Vec<(u64, u64)> = sides
        .nearest
        .iter()
        .map(|e| (e.index().get().unwrap(), e.squared_distance().to_bits()))
        .collect();
    assert_eq!(got, [(1, 0), (3, 0), (7, 0), (9, 0)]);
    assert_eq!(sides.left.index().get(), Some(1));
    assert_eq!(sides.right.index().get(), Some(7));
}

/// A window with fewer than `k` neighbors keeps fewer entries, and one with none keeps none.
#[test]
fn a_window_with_fewer_than_k_neighbors_keeps_fewer() {
    let series: Vec<f64> = (0..9).map(|k| f64::from((k * 3 % 5) as u8)).collect();
    let topk =
        two_sided_topk_profile(&series, nz(4), Scale::new(1.0).unwrap(), nz(1), nz(50)).unwrap();
    let counts: Vec<usize> = topk.iter().map(|(_, s)| s.nearest.len()).collect();
    let expected: Vec<usize> = naive(&series, 4, Scale::new(1.0).unwrap(), 50)
        .iter()
        .map(|(_, s)| s.nearest.len())
        .collect();
    assert_eq!(counts, expected);
    assert!(counts.iter().all(|&c| c < 6));
}

/// Past the accumulators' candidate caps, where exact ties are resolved down to the nearest `k`
/// while tiles merge: every pair ties, or a handful of distances repeat many times.
#[test]
fn ties_beyond_the_candidate_caps_resolve_to_the_exact_nearest_k() {
    let scale = Scale::new(1.0).unwrap();
    let constant = vec![7.0; 420];
    let period_four: Vec<f64> = (0..420).map(|k| f64::from([0, 2, 1, 3][k % 4])).collect();
    for series in [&constant, &period_four] {
        let m = 4;
        for k in [1, 3, 8] {
            let expected = bits(&naive(series, m, scale, k));
            for threads in [1, 3] {
                for (band_rows, chunk_lanes, tile_lanes) in
                    [(16, 8, 32), (64, 64, 64), (430, 430, 430)]
                {
                    let (actual, _) = two_sided_topk_profile_tiled(
                        series,
                        nz(m),
                        scale,
                        nz(threads),
                        nz(band_rows),
                        nz(chunk_lanes),
                        nz(tile_lanes),
                        nz(k),
                    )
                    .unwrap();
                    assert_eq!(
                        bits(&actual),
                        expected,
                        "k {k}, threads {threads}, shape {:?}",
                        (band_rows, chunk_lanes, tile_lanes)
                    );
                }
            }
        }
    }
}

#[test]
fn a_long_period_series_across_several_column_groups_equals_naive() {
    let series: Vec<f64> = (0..1500)
        .map(|k| f64::from([0, 3, 1, 4][k % 4] as u8))
        .collect();
    let scale = Scale::new(1.0).unwrap();
    for k in [1, 3] {
        let expected = bits(&naive(&series, 8, scale, k));
        for threads in [1, 2, 7] {
            assert_eq!(
                bits(&two_sided_topk_profile(&series, nz(8), scale, nz(threads), nz(k)).unwrap()),
                expected,
                "k {k}, threads {threads}"
            );
        }
    }
}

#[test]
fn a_series_shorter_than_the_window_is_empty() {
    let scale = Scale::new(1.0).unwrap();
    assert!(
        two_sided_topk_profile(&[1.0, 2.0], nz(5), scale, nz(2), nz(3))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_huge_k_is_clamped_to_the_windows() {
    let series = walk(80);
    let scale = Scale::new(1.0).unwrap();
    let huge = two_sided_topk_profile(&series, nz(4), scale, nz(2), nz(usize::MAX)).unwrap();
    let all = two_sided_topk_profile(&series, nz(4), scale, nz(2), nz(77)).unwrap();
    assert_eq!(bits(&huge), bits(&all));
}
