#![cfg(all(shuttle, feature = "parallel"))]

//! The threaded batch scheduler over randomized Shuttle schedules on a small series: every
//! schedule gives `left_profile`'s profile on bits.

use core::num::NonZeroUsize;

use ladde::batch::{
    left_profile, left_profile_report_tiled, left_profile_threads_tiled, left_profile_threads_with,
};
use ladde::entry::Entry;
use ladde::quantize::Scale;

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

#[test]
fn shuttle_threaded_batch_equals_single_threaded() {
    let series: Vec<f64> = (0..48u64)
        .map(|k| 1e10 + ((k * 7 + k * k * 3) % 13) as f64)
        .collect();
    let m = NonZeroUsize::new(4).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let expected = bits(&left_profile(&series, m, scale));
    assert!(expected.iter().any(|&(_, _, _, i)| i != u64::MAX));
    shuttle::check_random_with_seed(
        move || {
            // Tiles wider than one L1 chunk (the default shape), with bands that finish and free
            // their accumulators while other bands' tiles are still running.
            for (threads, band_rows, chunk_lanes, tile_lanes) in
                [(2, 3, 4, 8), (3, 4, 3, 9), (4, 2, 2, 6)]
            {
                let got = left_profile_threads_tiled(
                    &series,
                    m,
                    scale,
                    NonZeroUsize::new(threads).unwrap(),
                    NonZeroUsize::new(band_rows).unwrap(),
                    NonZeroUsize::new(chunk_lanes).unwrap(),
                    NonZeroUsize::new(tile_lanes).unwrap(),
                );
                assert_eq!(
                    bits(&got),
                    expected,
                    "{threads} {band_rows} {chunk_lanes} {tile_lanes}"
                );
            }
            for (threads, band_rows, chunk_lanes) in [(2, 3, 4), (3, 1, 5), (4, 7, 2)] {
                let got = left_profile_threads_with(
                    &series,
                    m,
                    scale,
                    NonZeroUsize::new(threads).unwrap(),
                    NonZeroUsize::new(band_rows).unwrap(),
                    NonZeroUsize::new(chunk_lanes).unwrap(),
                );
                assert_eq!(bits(&got), expected, "{threads} {band_rows} {chunk_lanes}");
            }
        },
        testkit::generators::TEST_SEED,
        200,
    );
}

/// A series too wide for one anchor, so every schedule runs the per-tile anchors: a drift with a
/// few spikes, so some tiles narrow both sides and some need the exact pass. The profile is the
/// naive-checked one the single-threaded run gives, and the wide tiles are counted the same.
#[test]
fn shuttle_per_tile_anchors_equal_single_threaded_and_count_wide_tiles() {
    let series: Vec<f64> = (0..40u64)
        .map(|k| {
            k as f64 * 2.5e7 + ((k * 7 + k * k * 3) % 13) as f64 + if k == 20 { 1.2e9 } else { 0.0 }
        })
        .collect();
    let m = NonZeroUsize::new(4).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let nz = |x| NonZeroUsize::new(x).unwrap();
    shuttle::check_random_with_seed(
        move || {
            // Under the model, one thread is still the threaded engine, so it is the reference.
            let (expected, expected_report) =
                left_profile_report_tiled(&series, m, scale, nz(1), nz(3), nz(2), nz(4));
            assert!(
                expected_report.wide_tiles > 0
                    && expected_report.wide_tiles < expected_report.tiles
            );
            let expected = bits(&expected);
            assert!(expected.iter().any(|&(_, _, _, i)| i != u64::MAX));
            for (threads, band_rows, chunk_lanes, tile_lanes) in [(2, 3, 2, 4), (3, 2, 2, 6)] {
                let (got, report) = left_profile_report_tiled(
                    &series,
                    m,
                    scale,
                    nz(threads),
                    nz(band_rows),
                    nz(chunk_lanes),
                    nz(tile_lanes),
                );
                assert_eq!(bits(&got), expected);
                assert!(report.wide_tiles > 0 && report.wide_tiles <= report.tiles);
            }
            let (got, report) =
                left_profile_report_tiled(&series, m, scale, nz(3), nz(3), nz(2), nz(4));
            assert_eq!(bits(&got), expected);
            assert_eq!(report, expected_report);
        },
        testkit::generators::TEST_SEED,
        100,
    );
}
