#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! A band in the middle of a long series whose rows get no candidate from some of the band's
//! chunks (every neighbor there is non-finite) merges to the single-threaded result, and a row
//! with no candidate anywhere stays `Entry::NONE`.

use core::num::NonZeroUsize;

use ladde::batch::{left_profile, left_profile_threads_with};
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

/// 200 missing samples (every window over them is non-finite), then a short finite stretch,
/// a second gap, and a finite tail that repeats the stretch: the first rows of the stretch see
/// only non-finite neighbors in their low chunks, and the rows in the second gap have no
/// candidate at all.
fn gapped_series() -> Vec<f64> {
    let stretch: Vec<f64> = (0..30)
        .map(|k| 1e10 + ((k * 7 + k * k) % 11) as f64)
        .collect();
    let mut series = vec![f64::NAN; 200];
    series.extend_from_slice(&stretch);
    series.extend(std::iter::repeat_n(f64::NAN, 40));
    series.extend_from_slice(&stretch);
    series
}

#[test]
fn bands_with_empty_chunks_merge_to_the_single_threaded_profile() {
    let m = NonZeroUsize::new(4).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let series = gapped_series();
    let expected = left_profile(&series, m, scale);

    let exclusion = ladde::exclusion::zone(m.get());
    let none_past_exclusion = expected
        .iter()
        .filter(|&&(row, e)| row as usize > exclusion && e.index().get().is_none())
        .count();
    let real = expected
        .iter()
        .filter(|&&(_, e)| e.index().get().is_some())
        .count();
    assert!(none_past_exclusion > 100, "rows with no candidate anywhere");
    assert!(real > 20, "rows with a real neighbor");

    let want = bits(&expected);
    for threads in [1usize, 2, 3, 8] {
        for (band_rows, chunk_lanes) in [(7usize, 5usize), (1, 3), (16, 37), (1000, 9)] {
            let got = left_profile_threads_with(
                &series,
                m,
                scale,
                NonZeroUsize::new(threads).unwrap(),
                NonZeroUsize::new(band_rows).unwrap(),
                NonZeroUsize::new(chunk_lanes).unwrap(),
            );
            assert_eq!(
                bits(&got),
                want,
                "threads {threads}, band_rows {band_rows}, chunk_lanes {chunk_lanes}"
            );
        }
    }
}

/// The default entry point past its single-threaded threshold (so it really runs tiles) with its
/// own default band and chunk sizes.
#[test]
fn the_default_entry_point_runs_tiles_and_matches() {
    let m = NonZeroUsize::new(50).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut level = 1e10;
    let series: Vec<f64> = (0..3_000)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 56) as i64 % 5 - 2) as f64;
            level
        })
        .collect();
    let want = bits(&left_profile(&series, m, scale));
    for threads in [2usize, 3, 8] {
        let got = ladde::batch::left_profile_threads(
            &series,
            m,
            scale,
            NonZeroUsize::new(threads).unwrap(),
        );
        assert_eq!(bits(&got), want, "threads {threads}");
    }
}
