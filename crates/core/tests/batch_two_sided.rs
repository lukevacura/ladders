#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The two-sided batch profile (left, right and nearer neighbor per window) equals
//! `naive_two_sided` on bits, at every thread count and tiling, on series with NaN gaps, flat
//! runs, exact repeats and ties, and on series too wide for one anchor.

use core::num::NonZeroUsize;

use ladde::batch::{fluss, left_profile, two_sided_profile, two_sided_profile_tiled};
use ladde::entry::{Entry, TwoSided};
use ladde::naive::naive_two_sided;
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series_and_m};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn entry_bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

type SideBits = [(u64, u64, u64); 3];

fn bits(profile: &[(u64, TwoSided)]) -> Vec<(u64, SideBits)> {
    profile
        .iter()
        .map(|&(row, sides)| {
            (
                row,
                [sides.left, sides.right, sides.nearest].map(entry_bits),
            )
        })
        .collect()
}

fn naive(series: &[f64], m: usize, scale: Scale) -> Vec<(u64, TwoSided)> {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    naive_two_sided(&samples, m)
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

fn assert_matches_naive(series: &[f64], m: usize, scale: Scale) {
    let naive = naive(series, m, scale);
    let expected = bits(&naive);
    for threads in [1, 2, 7] {
        for (band_rows, chunk_lanes, tile_lanes) in shapes(series.len()) {
            let (actual, _) = two_sided_profile_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(band_rows),
                nz(chunk_lanes),
                nz(tile_lanes),
            );
            assert_eq!(
                bits(&actual),
                expected,
                "threads {threads}, shape {:?}",
                (band_rows, chunk_lanes, tile_lanes)
            );
        }
        assert_eq!(
            bits(&two_sided_profile(series, nz(m), scale, nz(threads))),
            expected,
            "default tiling, threads {threads}"
        );
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]

    #[test]
    fn two_sided_equals_naive_on_market_series((series, m) in market_series_and_m(0..70)) {
        assert_matches_naive(&series, m, Scale::new(MARKET_TICK).unwrap());
    }
}

#[test]
fn a_repeating_series_ties_to_the_smaller_index_on_both_sides() {
    let series: Vec<f64> = (0..60)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
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
    assert_matches_naive(&walk(1300), 4, Scale::new(1.0).unwrap());
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
fn a_drifting_series_equals_naive() {
    let series: Vec<f64> = walk(100)
        .iter()
        .enumerate()
        .map(|(k, x)| x + k as f64 * 2.5e7)
        .collect();
    assert_matches_naive(&series, 4, Scale::new(1.0).unwrap());
}

#[test]
fn the_left_side_is_the_left_profile_and_threads_change_no_bit() {
    let series = walk(2700);
    let scale = Scale::new(1.0).unwrap();
    let m = nz(30);
    let one = two_sided_profile(&series, m, scale, nz(1));
    let left = left_profile(&series, m, scale);
    for (&(row, sides), &(left_row, left_entry)) in one.iter().zip(&left) {
        assert_eq!(row, left_row);
        assert_eq!(entry_bits(sides.left), entry_bits(left_entry), "row {row}");
    }
    assert!(one.iter().any(|(_, s)| s.right.index().get().is_some()));
    let expected = bits(&one);
    for threads in [2, 7] {
        assert_eq!(
            bits(&two_sided_profile(&series, m, scale, nz(threads))),
            expected,
            "threads {threads}"
        );
    }
}

#[test]
fn fluss_is_the_curve_of_the_nearest_indices_at_every_thread_count() {
    let series = walk(1500);
    let scale = Scale::new(1.0).unwrap();
    let m = nz(20);
    let indices: Vec<_> = two_sided_profile(&series, m, scale, nz(1))
        .iter()
        .map(|(_, sides)| sides.nearest.index())
        .collect();
    let want = ladde::floss::fluss(&indices, 5, 20, 4, 5, None).unwrap();
    assert_eq!(want.1.len(), 3);
    for threads in [1, 2, 7] {
        let got = fluss(&series, m, scale, nz(threads), 20, 4, 5, None).unwrap();
        assert_eq!(got.1, want.1, "threads {threads}");
        let curve_bits = |curve: &[f64]| curve.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(curve_bits(&got.0), curve_bits(&want.0), "threads {threads}");
    }
    assert!(fluss(&series, m, scale, nz(1), 20, 4, 5, Some(&[1.0])).is_none());
}

#[test]
fn a_series_shorter_than_the_window_is_empty() {
    let scale = Scale::new(1.0).unwrap();
    assert!(two_sided_profile(&[1.0, 2.0], nz(5), scale, nz(2)).is_empty());
}

#[test]
fn a_constant_series_equals_naive() {
    assert_matches_naive(&[2.0; 90], 6, Scale::new(1.0).unwrap());
    assert_matches_naive(&[2.0; 40], 4, Scale::new(1.0).unwrap());
}

#[test]
fn constant_runs_between_levels_equal_naive() {
    let series: Vec<f64> = (0..150).map(|k| f64::from(((k / 13) % 3) as u8)).collect();
    assert_matches_naive(&series, 5, Scale::new(1.0).unwrap());
}

#[test]
fn period_k_repeats_equal_naive() {
    for period in [1, 2, 3, 4, 7] {
        let series: Vec<f64> = (0..130)
            .map(|k| f64::from(((k % period) * 5 % 7) as u8))
            .collect();
        assert_matches_naive(&series, 6, Scale::new(1.0).unwrap());
    }
}

#[test]
fn few_level_series_equal_naive() {
    let mut state: u64 = 0xD1B5_4A32_D192_ED03;
    for (levels, m) in [(2u64, 3usize), (2, 8), (3, 4), (4, 5)] {
        let series: Vec<f64> = (0..160)
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
fn a_long_period_series_across_several_column_groups_equals_naive() {
    let series: Vec<f64> = (0..1500)
        .map(|k| f64::from([0, 3, 1, 4][k % 4] as u8))
        .collect();
    let expected = bits(&naive(&series, 8, Scale::new(1.0).unwrap()));
    for threads in [1, 2, 7] {
        assert_eq!(
            bits(&two_sided_profile(
                &series,
                nz(8),
                Scale::new(1.0).unwrap(),
                nz(threads)
            )),
            expected,
            "threads {threads}"
        );
    }
}
