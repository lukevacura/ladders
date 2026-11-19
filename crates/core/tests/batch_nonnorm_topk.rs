#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized two-sided profile and AB-join with `k` neighbors equal an exact integer
//! reference on bits at every thread count and tile shape, on series of heavy ties, gaps at every
//! boundary and ticks past the `u64` bound, for `k` below, at and above the candidate count; with
//! `k` of one they are the one-neighbor profiles.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/nonnorm_topk.rs"]
mod nonnorm_topk;

use ladde::batch::nonnorm::{
    ab_join, ab_join_topk, ab_join_topk_tiled, two_sided_profile, two_sided_topk_profile,
    two_sided_topk_profile_tiled,
};
use nonnorm::{LEVELS, nz, planted, series, shapes, tie_series, unit};
use nonnorm_topk::{batch_join, batch_sides, exact_join, exact_two_sided};
use proptest::prelude::*;

const THREADS: [usize; 3] = [1, 2, 7];
const KS: [usize; 5] = [1, 2, 3, 8, 200];

fn assert_two_sided(series: &[f64], m: usize, ks: &[usize]) {
    for &k in ks {
        let expected = exact_two_sided(series, unit(), m, nz(k));
        for threads in THREADS {
            for (band, chunk, tile) in shapes(series.len()) {
                let (rows, _) = two_sided_topk_profile_tiled(
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
                    batch_sides(&rows),
                    expected,
                    "two-sided, m {m}, k {k}, threads {threads}, shape {:?}",
                    (band, chunk, tile)
                );
            }
            let rows = two_sided_topk_profile(series, nz(m), unit(), nz(threads), nz(k)).unwrap();
            assert_eq!(
                batch_sides(&rows),
                expected,
                "two-sided, m {m}, k {k}, default tiling"
            );
        }
    }
}

fn assert_join(a: &[f64], b: &[f64], m: usize, ks: &[usize]) {
    for &k in ks {
        let expected = exact_join(a, b, unit(), m, nz(k));
        for threads in THREADS {
            for (band, chunk, tile) in shapes(a.len().max(b.len())) {
                let (rows, _) = ab_join_topk_tiled(
                    a,
                    b,
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
                    batch_join(&rows),
                    expected,
                    "join, m {m}, k {k}, threads {threads}, shape {:?}",
                    (band, chunk, tile)
                );
            }
            let rows = ab_join_topk(a, b, nz(m), unit(), nz(threads), nz(k)).unwrap();
            assert_eq!(batch_join(&rows), expected, "join, m {m}, k {k}, default");
        }
    }
}

fn assert_all(series: &[f64], m: usize, ks: &[usize]) {
    assert_two_sided(series, m, ks);
    assert_join(series, series, m, ks);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]

    #[test]
    fn random_series_equal_the_reference(series in series(), m in 1usize..=6) {
        assert_all(&series, m, &[2, 5]);
    }

    #[test]
    fn tie_series_equal_the_reference(series in tie_series(), m in 1usize..=6) {
        assert_all(&series, m, &[2, 5]);
    }

    #[test]
    fn planted_repeats_and_gaps_equal_the_reference((series, m) in planted()) {
        assert_all(&series, m, &[3]);
    }

    #[test]
    fn two_series_join_equal_the_reference(
        a in prop_oneof![series(), tie_series()],
        b in prop_oneof![series(), tie_series()],
        m in 1usize..=6,
    ) {
        assert_join(&a, &b, m, &[2, 4]);
    }
}

/// Series whose pairs tie in long runs: a constant, flat runs between levels, short periods, few
/// levels, and exact repeats at an offset.
fn tie_heavy() -> Vec<Vec<f64>> {
    let runs: Vec<f64> = (0..60)
        .map(|k| {
            if (k / 7) % 2 == 0 {
                0.0
            } else {
                (1u64 << 33) as f64
            }
        })
        .collect();
    let few_levels: Vec<f64> = (0..60).map(|k| f64::from((k * 5 % 3) as u8)).collect();
    let mut all = vec![vec![7.0; 40], runs, few_levels];
    for period in [2usize, 4, 7] {
        all.push(
            (0..70)
                .map(|k| [0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0][k % period])
                .collect(),
        );
    }
    all.push(
        (0..70)
            .map(|k| (k % 9) as f64 + if k >= 35 { 1e6 } else { 0.0 })
            .collect(),
    );
    all
}

#[test]
fn constant_runs_periods_and_few_levels_tie_to_the_smaller_index() {
    for series in &tie_heavy() {
        for m in [1, 3, 5, 8] {
            assert_all(series, m, &KS);
        }
    }
    // Every pair of a constant series ties at zero, so each row keeps the smallest indices.
    let flat = two_sided_topk_profile(&[7.0; 40], nz(4), unit(), nz(1), nz(3)).unwrap();
    let kept: Vec<u64> = flat[20]
        .1
        .nearest
        .iter()
        .map(|n| n.index().get().unwrap())
        .collect();
    assert_eq!(kept, [0, 1, 2]);
}

#[test]
fn gaps_at_the_start_the_end_and_every_tile_boundary() {
    let walk = || -> Vec<f64> { (0..48).map(|k| f64::from((k * 7 % 11) as u8)).collect() };
    let gap_sets: [&[usize]; 5] = [
        &[0, 1, 2],
        &[45, 46, 47],
        &[0, 47],
        &[5, 6, 11, 12, 17, 18, 23, 24],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ],
    ];
    for gaps in gap_sets {
        let mut series = walk();
        for &g in gaps {
            series[g] = f64::NAN;
        }
        for m in [2, 4, 6] {
            assert_all(&series, m, &[1, 3, 200]);
        }
    }
    assert_all(&vec![f64::NAN; 30], 4, &[1, 3]);
}

#[test]
fn ticks_past_the_u64_bound_keep_the_bits() {
    for level in [(1i64 << 31) - 1, 1 << 31, 1 << 40, LEVELS[5]] {
        let series: Vec<f64> = (0..30)
            .map(|k| {
                let wobble = (k * 7 % 5) as i64;
                let tick = if (k / 6) % 2 == 0 {
                    wobble
                } else {
                    level + wobble
                };
                tick.min(1 << 53) as f64
            })
            .collect();
        assert_all(&series, 4, &[2, 6]);
    }
}

#[test]
fn k_of_one_is_the_one_neighbor_profile_and_the_sides_stay_one_nearest() {
    for series in &tie_heavy() {
        let m = 4;
        let one = two_sided_profile(series, nz(m), unit(), nz(2));
        for k in [1usize, 4] {
            let rows = two_sided_topk_profile(series, nz(m), unit(), nz(2), nz(k)).unwrap();
            assert_eq!(rows.len(), one.len());
            for ((row, sides), (_, single)) in rows.iter().zip(&one) {
                assert_eq!(sides.left, single.left, "row {row}");
                assert_eq!(sides.right, single.right, "row {row}");
                assert_eq!(
                    sides.nearest.first().copied().unwrap_or(single.nearest),
                    single.nearest
                );
                if k == 1 {
                    assert_eq!(
                        sides.nearest.len(),
                        usize::from(single.nearest.index().get().is_some())
                    );
                }
            }
        }
        let join = ab_join(series, series, nz(m), unit(), nz(2));
        let rows = ab_join_topk(series, series, nz(m), unit(), nz(2), nz(1)).unwrap();
        for ((_, kept), (_, single)) in rows.iter().zip(&join) {
            assert_eq!(kept.first().copied().unwrap_or(*single), *single);
        }
    }
}

#[test]
fn a_k_beyond_the_windows_is_padding_not_an_error() {
    let series: Vec<f64> = (0..40).map(|k| f64::from((k * 3 % 7) as u8)).collect();
    let huge = nz(usize::MAX);
    let all = two_sided_topk_profile(&series, nz(4), unit(), nz(2), huge).unwrap();
    assert_eq!(
        batch_sides(&all),
        exact_two_sided(&series, unit(), 4, nz(series.len()))
    );
    let join = ab_join_topk(&series, &series[3..], nz(4), unit(), nz(2), huge).unwrap();
    assert_eq!(
        batch_join(&join),
        exact_join(&series, &series[3..], unit(), 4, nz(series.len()))
    );
}

#[test]
fn empty_short_and_oversized_windows_return_no_rows_or_no_neighbors() {
    let empty: [f64; 0] = [];
    for threads in THREADS {
        let k = nz(3);
        let t = nz(threads);
        assert!(
            two_sided_topk_profile(&empty, nz(3), unit(), t, k)
                .unwrap()
                .is_empty()
        );
        assert!(
            two_sided_topk_profile(&[1.0, 2.0], nz(3), unit(), t, k)
                .unwrap()
                .is_empty()
        );
        assert!(
            two_sided_topk_profile(&[1.0; 5], nz(usize::MAX), unit(), t, k)
                .unwrap()
                .is_empty()
        );
        assert!(
            ab_join_topk(&[1.0, 2.0], &[1.0; 5], nz(3), unit(), t, k)
                .unwrap()
                .is_empty()
        );
        let none = ab_join_topk(&[1.0; 5], &[1.0, 2.0], nz(3), unit(), t, k).unwrap();
        assert_eq!(none.len(), 3);
        assert!(none.iter().all(|(_, kept)| kept.is_empty()));
    }
}

/// Past the size where the default plan spreads tiles over threads, ties included: every thread
/// count gives the reference.
#[test]
fn default_tiling_at_threaded_size_is_thread_invariant() {
    let n = 2_700;
    let series: Vec<f64> = (0..n)
        .map(|i| match i / 400 % 3 {
            0 => (i % 5) as f64,
            1 => ((i * i / 7) % 4) as f64,
            _ => 3.0,
        })
        .collect();
    let m = 8;
    let k = nz(5);
    let two_sided = exact_two_sided(&series, unit(), m, k);
    let join = exact_join(&series, &series[100..], unit(), m, k);
    for threads in THREADS {
        let rows = two_sided_topk_profile(&series, nz(m), unit(), nz(threads), k).unwrap();
        assert_eq!(
            batch_sides(&rows),
            two_sided,
            "two-sided, threads {threads}"
        );
        let rows = ab_join_topk(&series, &series[100..], nz(m), unit(), nz(threads), k).unwrap();
        assert_eq!(batch_join(&rows), join, "join, threads {threads}");
    }
}
