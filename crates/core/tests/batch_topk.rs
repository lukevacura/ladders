//! The tiled batch left profile with `k` neighbors equals `naive_left_topk` on bits, at every
//! thread count and tiling, on series with NaN gaps, flat runs, exact repeats and heavy ties, for
//! `k` below, at and above the candidate count; with `k` of one it is `left_profile`.

use core::num::NonZeroUsize;

use ladde::batch::{left_profile, left_topk_profile, left_topk_profile_tiled};
use ladde::entry::Entry;
use ladde::naive::naive_left_topk;
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

fn bits(profile: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<EntryBits>)> {
    profile
        .iter()
        .map(|(row, entries)| (*row, entries.iter().copied().map(entry_bits).collect()))
        .collect()
}

fn naive(series: &[f64], m: usize, scale: Scale, k: usize) -> Vec<(u64, Vec<Entry>)> {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    naive_left_topk(&samples, m, nz(k))
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

fn assert_matches_naive(series: &[f64], m: usize, scale: Scale) {
    for k in KS {
        let expected = bits(&naive(series, m, scale, k));
        for threads in [1, 2, 7] {
            for (band_rows, chunk_lanes, tile_lanes) in shapes(series.len()) {
                let (actual, _) = left_topk_profile_tiled(
                    series,
                    nz(m),
                    scale,
                    nz(threads),
                    nz(band_rows),
                    nz(chunk_lanes),
                    nz(tile_lanes),
                    nz(k),
                );
                assert_eq!(
                    bits(&actual),
                    expected,
                    "k {k}, threads {threads}, shape {:?}",
                    (band_rows, chunk_lanes, tile_lanes)
                );
            }
            assert_eq!(
                bits(&left_topk_profile(series, nz(m), scale, nz(threads), nz(k))),
                expected,
                "k {k}, default tiling, threads {threads}"
            );
        }
    }
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
fn a_period_four_series_equals_naive() {
    let series: Vec<f64> = (0..70).map(|k| f64::from([0, 2, 1, 3][k % 4])).collect();
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
fn a_series_too_wide_for_one_anchor_equals_naive() {
    let series: Vec<f64> = (0..120)
        .map(|k| f64::from(((k * 7 % 13) as u8) % 5) + if k % 2 == 0 { 0.0 } else { 1.5e9 })
        .collect();
    assert_matches_naive(&series, 4, Scale::new(1.0).unwrap());
}

#[test]
fn k_of_one_is_the_left_profile_on_bits() {
    let series: Vec<f64> = (0..90).map(|k| f64::from((k * 5 % 17) as u8)).collect();
    let scale = Scale::new(1.0).unwrap();
    let single = left_profile(&series, nz(5), scale);
    let topk = left_topk_profile(&series, nz(5), scale, nz(2), nz(1));
    assert_eq!(single.len(), topk.len());
    for ((row, entry), (topk_row, entries)) in single.iter().zip(&topk) {
        assert_eq!(row, topk_row);
        match entry.index().get() {
            None => assert!(entries.is_empty()),
            Some(_) => assert_eq!(
                entries.iter().copied().map(entry_bits).collect::<Vec<_>>(),
                [entry_bits(*entry)]
            ),
        }
    }
}

/// Window 5 of `0 1 0 1 ...` at `m = 2`: the exact ties at distance zero are windows 1 and 3
/// (smaller index first), then the tied nearest of the rest, the smaller index.
#[test]
fn a_fixed_tie_case_has_literal_bits() {
    let series: Vec<f64> = (0..12).map(|k| f64::from((k % 2) as u8)).collect();
    let topk = left_topk_profile(&series, nz(2), Scale::new(1.0).unwrap(), nz(1), nz(3));
    let (row, entries) = &topk[5];
    assert_eq!(*row, 5);
    let got: Vec<(u64, u64)> = entries
        .iter()
        .map(|e| (e.index().get().unwrap(), e.squared_distance().to_bits()))
        .collect();
    assert_eq!(got, [(1, 0), (3, 0), (0, 8.0_f64.to_bits())]);
}

/// Past the row accumulator's candidate cap, where exact ties are resolved down to the nearest
/// `k` while tiles merge: every pair ties, or a handful of distances repeat many times.
#[test]
fn ties_beyond_the_candidate_cap_resolve_to_the_exact_nearest_k() {
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
                    let (actual, _) = left_topk_profile_tiled(
                        series,
                        nz(m),
                        scale,
                        nz(threads),
                        nz(band_rows),
                        nz(chunk_lanes),
                        nz(tile_lanes),
                        nz(k),
                    );
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
