#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Batch on a series too wide for one anchor: independent tiles each take their own anchor for
//! their rows and for their columns, only a tile whose own side is still too wide runs the exact
//! i64 pass, and the entries equal the naive oracle on bits for any thread count and tiling.
//! The report counts the tiles and the wide ones.

use core::num::NonZeroUsize;

use ladde::batch::{BatchReport, left_profile, left_profile_report, left_profile_report_tiled};
use ladde::entry::Entry;
use ladde::naive::naive_left_profile;
use ladde::quantize::{Scale, to_ticks};

const M: usize = 4;
/// Twice the largest offset one anchor covers at `M` (the span a single anchor can hold).
const TWICE_BOUND: i64 = 2 * (i32::MAX as i64 / M as i64);

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

fn bits(profile: &[(u64, Entry)]) -> Vec<(u64, u64, u64, u64)> {
    profile
        .iter()
        .map(|&(row, e)| {
            (
                row,
                e.rho().to_bits(),
                e.squared_distance().to_bits(),
                e.index().get().unwrap_or(u64::MAX),
            )
        })
        .collect()
}

/// A small deterministic noise in `0..bound`.
fn noise(k: u64, bound: u64) -> f64 {
    let mut x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % bound) as f64
}

fn span(series: &[f64]) -> i64 {
    let ticks: Vec<i64> = series
        .iter()
        .filter_map(|&x| to_ticks(x, scale()))
        .collect();
    ticks.iter().max().unwrap() - ticks.iter().min().unwrap()
}

/// A steady rise of `step` per sample over a small noise: the whole series is wider than one
/// anchor, a short stretch of it is not.
fn drift(n: usize, step: f64) -> Vec<f64> {
    (0..n)
        .map(|k| k as f64 * step + noise(k as u64, 1000))
        .collect()
}

/// Small noise with a few spikes wide enough that a tile touching one needs the exact pass.
fn spikes(n: usize, at: &[usize]) -> Vec<f64> {
    (0..n)
        .map(|k| noise(k as u64, 1000) + if at.contains(&k) { 1.2e9 } else { 0.0 })
        .collect()
}

/// Every pair of neighbors is a whole anchor apart: no tile can narrow either side.
fn alternating(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| noise(k as u64, 1000) + if k % 2 == 0 { 0.0 } else { 1.5e9 })
        .collect()
}

fn naive(series: &[f64]) -> Vec<(u64, Entry)> {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale())).collect();
    naive_left_profile(&samples, M)
}

fn shapes(n: usize) -> Vec<(usize, usize, usize)> {
    // (band_rows, chunk_lanes, tile_lanes), including one and more than the series.
    vec![
        (1, 1, 1),
        (2, 3, 6),
        (3, 4, 8),
        (8, 5, 5),
        (n + 5, 2, 4),
        (4, n + 5, n + 5),
    ]
}

fn run(
    series: &[f64],
    threads: usize,
    shape: (usize, usize, usize),
) -> (Vec<(u64, Entry)>, BatchReport) {
    left_profile_report_tiled(
        series,
        nz(M),
        scale(),
        nz(threads),
        nz(shape.0),
        nz(shape.1),
        nz(shape.2),
    )
}

/// Every shape and thread count gives the naive profile, with `check` run on each report.
fn assert_equals_naive(series: &[f64], mut check: impl FnMut(BatchReport, (usize, usize, usize))) {
    let expected = naive(series);
    assert!(
        expected.iter().any(|&(_, e)| e.index().get().is_some()),
        "the expected profile is not trivial"
    );
    let expected = bits(&expected);
    for threads in [1, 2, 3, 7] {
        for shape in shapes(series.len()) {
            let (got, report) = run(series, threads, shape);
            assert_eq!(bits(&got), expected, "threads {threads}, shape {shape:?}");
            check(report, shape);
        }
    }
}

#[test]
fn a_drift_past_the_bound_with_local_ranges_inside_it_runs_no_wide_tile() {
    let series = drift(90, 2.5e7);
    assert!(
        span(&series) > TWICE_BOUND,
        "no one anchor covers the series"
    );
    // Tiles narrow enough that rows and columns each span well under the bound.
    for threads in [1, 2, 3, 7] {
        for shape in [(1, 1, 1), (2, 3, 6), (3, 4, 8), (8, 5, 5)] {
            let (got, report) = run(&series, threads, shape);
            assert!(report.tiles > 1);
            assert_eq!(report.wide_tiles, 0, "threads {threads}, shape {shape:?}");
            assert_eq!(
                bits(&got),
                bits(&naive(&series)),
                "threads {threads}, shape {shape:?}"
            );
        }
    }
    assert_equals_naive(&series, |report, _| {
        assert!(report.wide_tiles <= report.tiles);
    });
}

#[test]
fn a_tile_wider_than_the_bound_alone_runs_the_exact_pass() {
    let series = drift(90, 2.5e7);
    // One band, one tile: it spans the whole series.
    let (_, report) = run(&series, 1, (200, 200, 200));
    assert_eq!(report, BatchReport::of(1, 1));
}

#[test]
fn local_spikes_run_some_wide_tiles_and_still_equal_naive() {
    let series = spikes(80, &[23, 57]);
    assert!(span(&series) > TWICE_BOUND);
    let mut saw = (false, false);
    assert_equals_naive(&series, |report, shape| {
        if shape == (2, 3, 6) || shape == (3, 4, 8) {
            assert!(
                report.wide_tiles > 0 && report.wide_tiles < report.tiles,
                "{report:?}"
            );
        }
        saw.0 |= report.wide_tiles > 0;
        saw.1 |= report.wide_tiles < report.tiles;
    });
    assert!(saw.0 && saw.1);
}

#[test]
fn an_all_wide_series_equals_naive() {
    let series = alternating(70);
    assert_equals_naive(&series, |report, shape| {
        assert!(report.tiles > 0);
        assert_eq!(report.wide_tiles, report.tiles, "{shape:?}");
    });
}

#[test]
fn a_series_that_fits_one_anchor_runs_no_wide_tile() {
    let series = drift(90, 1e3);
    assert!(span(&series) <= TWICE_BOUND);
    for threads in [1, 2, 3] {
        let (got, report) = left_profile_report(&series, nz(M), scale(), nz(threads));
        assert_eq!(report.wide_tiles, 0);
        assert!(report.tiles >= 1);
        assert_eq!(bits(&got), bits(&left_profile(&series, nz(M), scale())));
        assert_eq!(bits(&got), bits(&naive(&series)));
    }
    assert_equals_naive(&series, |report, _| assert_eq!(report.wide_tiles, 0));
}

#[test]
fn the_default_shape_equals_naive_on_a_wide_series() {
    for series in [drift(120, 2.5e7), spikes(120, &[40, 90]), alternating(100)] {
        let expected = bits(&naive(&series));
        for threads in [1, 2, 3, 7] {
            let (got, report) = left_profile_report(&series, nz(M), scale(), nz(threads));
            assert_eq!(bits(&got), expected, "threads {threads}");
            assert!(report.tiles > 0);
        }
    }
}

#[test]
fn a_moderate_wide_series_gives_the_same_bits_at_every_thread_count_and_shape() {
    let n = 600;
    let m = 8;
    let series: Vec<f64> = (0..n)
        .map(|k| k as f64 * 4e6 + noise(k as u64, 5000) + if k % 150 == 77 { 4e8 } else { 0.0 })
        .collect();
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale())).collect();
    let expected = bits(&naive_left_profile(&samples, m));
    for threads in [1, 2, 3, 7] {
        for shape in [
            (1, 1, 1),
            (16, 8, 40),
            (7, 64, 128),
            (n + 1, 3, 9),
            (40, n + 1, n + 1),
        ] {
            let (got, _) = left_profile_report_tiled(
                &series,
                nz(m),
                scale(),
                nz(threads),
                nz(shape.0),
                nz(shape.1),
                nz(shape.2),
            );
            assert_eq!(bits(&got), expected, "threads {threads}, shape {shape:?}");
        }
    }
}

/// Four moderate wide series whose profiles hash to what the whole-series exact pass produced
/// before tiles took their own anchors (recorded once from that engine).
#[test]
fn moderate_wide_series_match_the_profile_the_whole_series_exact_pass_gave() {
    const EXPECTED: u64 = 0xc7de_72a9_c046_ef99;
    let cases = [
        (1500usize, 16usize, 3e6f64, 0.0f64),
        (1200, 8, 5e6, 3e8),
        (900, 32, 1e6, 0.0),
        (700, 5, 4e7, 1e8),
    ];
    for (threads, shape) in [
        (1, None),
        (2, None),
        (3, Some((5, 7, 20))),
        (7, Some((1, 1, 1))),
    ] {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &(n, m, step, spike) in &cases {
            let series: Vec<f64> = (0..n)
                .map(|k| {
                    k as f64 * step
                        + noise(k as u64, 4000)
                        + if k % 211 == 100 { spike } else { 0.0 }
                })
                .collect();
            let profile = match shape {
                None => left_profile_report(&series, nz(m), scale(), nz(threads)).0,
                Some((b, c, t)) => {
                    left_profile_report_tiled(
                        &series,
                        nz(m),
                        scale(),
                        nz(threads),
                        nz(b),
                        nz(c),
                        nz(t),
                    )
                    .0
                }
            };
            assert!(
                profile
                    .iter()
                    .filter(|(_, e)| e.index().get().is_some())
                    .count()
                    > n / 2
            );
            for (row, e) in profile {
                for v in [
                    row,
                    e.rho().to_bits(),
                    e.squared_distance().to_bits(),
                    e.index().get().unwrap_or(u64::MAX),
                ] {
                    hash ^= v;
                    hash = hash.wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        assert_eq!(hash, EXPECTED, "threads {threads}, shape {shape:?}");
    }
}
