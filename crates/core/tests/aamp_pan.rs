#![cfg(not(any(loom, shuttle)))]

//! The non-normalized pan matrix profile (STUMPY's `aamp_stimp`) holds, for each visited size,
//! the exact nearest window on integer ticks (ties to the smaller start, a window with a gap
//! having none) at every thread count; normalizes the image by `|max - min| * sqrt(m)` over the
//! finite samples; errors exactly where the z-normalized one does; and leaves the z-normalized
//! one alone.

use core::num::NonZeroUsize;

#[path = "common/nonnorm_derived.rs"]
mod reference;

use ladde::index::SampleIndex;
use ladde::pan::{PanProfile, PanTransform};
use ladde::quantize::Scale;
use proptest::prelude::*;
use reference::{distance, nearest, ticks};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn finished(
    series: &[f64],
    (min_m, max_m, step): (usize, Option<usize>, usize),
    scale: Scale,
    threads: usize,
) -> PanProfile {
    let mut pan =
        PanProfile::new_nonnorm(series, min_m, max_m, step, Some(scale), nz(threads)).unwrap();
    while pan.update() {}
    pan
}

fn levels(n: usize, k: u64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % k) as f64
        })
        .collect()
}

fn tie_heavy_series() -> Vec<(&'static str, Vec<f64>)> {
    let periodic = |n: usize, p: usize| -> Vec<f64> {
        (0..n).map(|i| ((i % p) * (i % p) % 7) as f64).collect()
    };
    let mut runs = vec![0.0; 40];
    runs.extend(vec![3.0; 30]);
    runs.extend(levels(40, 3, 5));
    runs.extend(vec![3.0; 20]);
    let mut gaps = levels(100, 5, 7);
    for at in [0, 1, 2, 40, 41, 42, 43, 97, 98, 99] {
        gaps[at] = f64::NAN;
    }
    let mut infinite = levels(90, 4, 9);
    infinite[10] = f64::INFINITY;
    infinite[50] = f64::NEG_INFINITY;
    infinite[51] = f64::NAN;
    let mut repeats = levels(30, 50, 11);
    let tail = repeats.clone();
    repeats.extend(tail.iter().map(|x| x + 100.0));
    repeats.extend(tail);
    vec![
        ("constant", vec![2.0; 90]),
        ("runs", runs),
        ("period 2", periodic(90, 2)),
        ("period 4", periodic(90, 4)),
        ("period 7", periodic(90, 7)),
        ("few levels", levels(100, 3, 3)),
        ("repeats", repeats),
        ("gaps", gaps),
        ("infinite samples", infinite),
    ]
}

#[test]
fn every_row_is_the_exact_nearest_window_on_ticks_whatever_the_threads() {
    let unit = scale(0.5);
    for (name, series) in tie_heavy_series() {
        let range = (3, Some(30), 4);
        let base = finished(&series, range, unit, 1);
        assert_eq!(base.processed(), base.sizes());
        for threads in [2, 3, 8] {
            let other = finished(&series, range, unit, threads);
            for (a, b) in base.rows().iter().zip(other.rows()) {
                assert_eq!(bits(&a.distances), bits(&b.distances), "{name}");
                assert_eq!(a.indices, b.indices, "{name}");
            }
        }
        let t = ticks(&series, unit);
        for row in base.rows() {
            let want = nearest(&t, row.window);
            assert_eq!(row.distances.len(), want.len(), "{name} m={}", row.window);
            for (i, near) in want.iter().enumerate() {
                let (distance, index) = near
                    .map_or((f64::INFINITY, SampleIndex::NONE), |(s, j)| {
                        (distance(s, unit), SampleIndex::new(j as u64).unwrap())
                    });
                assert_eq!(
                    row.distances[i].to_bits(),
                    distance.to_bits(),
                    "{name} m={} window {i}",
                    row.window
                );
                assert_eq!(row.indices[i], index, "{name} m={} window {i}", row.window);
            }
        }
    }
}

#[test]
fn threaded_rows_on_a_series_long_enough_to_split_equal_the_single_thread() {
    let series: Vec<f64> = (0..3200).map(|i| ((i * i + i / 7) % 5) as f64).collect();
    let single = finished(&series, (8, Some(14), 3), scale(1.0), 1);
    let threaded = finished(&series, (8, Some(14), 3), scale(1.0), 4);
    for (a, b) in single.rows().iter().zip(threaded.rows()) {
        assert_eq!(bits(&a.distances), bits(&b.distances));
        assert_eq!(a.indices, b.indices);
    }
}

fn normalized_only() -> PanTransform {
    PanTransform {
        threshold: 0.2,
        normalize: true,
        contrast: false,
        binary: false,
        clip: false,
    }
}

/// The normalized cell of `distance` in a row of window size `m`, restated.
fn normalized(distance: f64, span: f64, m: usize) -> f64 {
    let scaled = distance * (1.0 / (span * (m as f64).sqrt()));
    if scaled.is_nan() {
        scaled
    } else {
        scaled.min(1.0)
    }
}

#[test]
fn the_image_divides_by_the_range_of_the_finite_samples_times_the_root_of_the_size() {
    for (name, series) in tie_heavy_series() {
        let finite: Vec<f64> = series.iter().copied().filter(|x| x.is_finite()).collect();
        let high = finite.iter().copied().fold(f64::MIN, f64::max);
        let low = finite.iter().copied().fold(f64::MAX, f64::min);
        let span = (high - low).abs();
        let pan = finished(&series, (3, Some(20), 5), scale(1.0), 2);
        let image = pan.pan(normalized_only());
        let sizes: Vec<usize> = {
            let mut all: Vec<usize> = pan.window_sizes().collect();
            all.sort();
            all
        };
        for (row, &m) in sizes.iter().enumerate() {
            let computed = pan.rows().iter().find(|r| r.window == m).unwrap();
            for (col, &d) in computed.distances.iter().enumerate() {
                let cell = image[row * pan.len() + col];
                if d == f64::INFINITY {
                    continue;
                }
                let want = normalized(d, span, m);
                assert!(
                    cell.to_bits() == want.to_bits() || (cell.is_nan() && want.is_nan()),
                    "{name} m={m} column {col}: {cell} vs {want}"
                );
            }
        }
    }
}

#[test]
fn the_two_metrics_normalize_differently_and_report_different_distances() {
    let series = levels(80, 40, 2);
    let mut norm = PanProfile::new(&series, 6, Some(6), 1, Some(scale(1.0)), nz(1)).unwrap();
    let mut raw = PanProfile::new_nonnorm(&series, 6, Some(6), 1, Some(scale(1.0)), nz(1)).unwrap();
    norm.update();
    raw.update();
    assert_ne!(
        bits(&norm.rows()[0].distances),
        bits(&raw.rows()[0].distances)
    );
    let raw_distance = raw.rows()[0].distances[0];
    let high = series.iter().copied().fold(f64::MIN, f64::max);
    let low = series.iter().copied().fold(f64::MAX, f64::min);
    let want = (raw_distance * (1.0 / ((high - low) * 6f64.sqrt()))).min(1.0);
    assert_eq!(raw.pan(normalized_only())[0].to_bits(), want.to_bits());
}

#[test]
fn a_series_without_a_finite_sample_has_no_neighbors_and_an_all_nan_image() {
    let series = vec![f64::NAN; 30];
    let mut pan = PanProfile::new_nonnorm(&series, 3, Some(6), 1, None, nz(1)).unwrap();
    while pan.update() {}
    for row in pan.rows() {
        assert!(row.distances.iter().all(|d| *d == f64::INFINITY));
        assert!(row.indices.iter().all(|i| i.get().is_none()));
    }
    assert!(pan.pan(normalized_only()).iter().all(|x| x.is_nan()));
}

#[test]
fn a_constant_series_has_zero_span_and_a_nan_image() {
    let series = vec![4.0; 40];
    let pan = finished(&series, (3, Some(8), 1), scale(1.0), 1);
    let image = pan.pan(normalized_only());
    assert_eq!(image.len(), pan.sizes() * 40);
    // Zero distance over zero span is NaN, as in STUMPY, and nothing is left to fill it from.
    assert!(image.iter().all(|x| x.is_nan()));
}

#[test]
fn ranges_are_errors_exactly_as_for_the_z_normalized_profile() {
    let series = levels(60, 5, 1);
    let unit = Some(scale(1.0));
    for min_m in [0, 1, 2, 3, 4, 10, 47, 59, 60, 61, usize::MAX] {
        for max_m in [
            None,
            Some(0),
            Some(3),
            Some(8),
            Some(40),
            Some(200),
            Some(usize::MAX),
        ] {
            for step in [0, 1, 2, 7, 1000, usize::MAX] {
                let a = PanProfile::new(&series, min_m, max_m, step, unit, nz(1));
                let b = PanProfile::new_nonnorm(&series, min_m, max_m, step, unit, nz(1));
                match (a, b) {
                    (Ok(a), Ok(b)) => assert!(a.window_sizes().eq(b.window_sizes())),
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    other => panic!("{min_m} {max_m:?} {step}: {:?}", other.0.is_ok()),
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(30))]

    #[test]
    fn rows_equal_the_tick_brute_force_on_random_series(
        series in prop::collection::vec(
            prop_oneof![
                1 => Just(f64::NAN),
                1 => Just(f64::INFINITY),
                12 => (0u8..6).prop_map(|x| f64::from(x) * 0.25),
            ],
            12..70,
        ),
        threads in 1usize..4,
    ) {
        let unit = scale(0.25);
        let pan = finished(&series, (3, Some(9), 2), unit, threads);
        let t = ticks(&series, unit);
        for row in pan.rows() {
            let want = nearest(&t, row.window);
            for (i, near) in want.iter().enumerate() {
                let (d, index) = near.map_or((f64::INFINITY, None), |(s, j)| (distance(s, unit), Some(j as u64)));
                prop_assert_eq!(row.distances[i].to_bits(), d.to_bits());
                prop_assert_eq!(row.indices[i].get(), index);
            }
        }
    }
}
