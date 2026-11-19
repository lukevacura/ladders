#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The batch cold route's memory stays within a fixed number of bytes per window and its allocation
//! count does not grow with the series, also where every candidate ties exactly: its allocations
//! are those of the same run with the outliers as gaps (the hot passes' own) plus a number that
//! depends on neither the length nor the tie structure. Its own test binary, with a process-wide allocation tracker, and
//! a single test so nothing else allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use ladde::batch::{
    ab_join, ab_join_topk, left_profile, left_topk_profile, two_sided_profile,
    two_sided_topk_profile,
};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, SpillForce, forced_route};
use peak_alloc::{large_allocations_of, peak_of};
use std::hint::black_box;
use std::num::NonZeroUsize;

type Make = fn(usize) -> Vec<f64>;

const M: usize = 8;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// Which batch profile a run computes.
#[derive(Clone, Copy, Debug)]
enum Profile {
    Left,
    TwoSided,
    /// The series joined with itself.
    Join,
}

/// `series` with two outliers, or with those two samples missing.
fn with_outliers(mut series: Vec<f64>, as_gaps: bool) -> Vec<f64> {
    let n = series.len();
    series[n / 3] = if as_gaps { f64::NAN } else { 7.0e11 };
    series[2 * n / 3] = if as_gaps { f64::NAN } else { -9.0e11 };
    series
}

fn force(spill: SpillForce) -> RouteForce {
    RouteForce {
        spill,
        ..RouteForce::default()
    }
}

/// One run of `profile` at `k` neighbors over `series`.
fn run(profile: Profile, series: &[f64], k: usize) {
    let scale = Scale::new(1.0).unwrap();
    let (m, one, k) = (nz(M), nz(1), nz(k));
    match (profile, k.get()) {
        (Profile::Left, 1) => drop(black_box(left_profile(series, m, scale))),
        (Profile::Left, _) => drop(black_box(left_topk_profile(series, m, scale, one, k))),
        (Profile::TwoSided, 1) => drop(black_box(two_sided_profile(series, m, scale, one))),
        (Profile::TwoSided, _) => {
            drop(black_box(two_sided_topk_profile(series, m, scale, one, k)));
        }
        (Profile::Join, 1) => drop(black_box(ab_join(series, series, m, scale, scale, one))),
        (Profile::Join, _) => {
            drop(black_box(ab_join_topk(
                series, series, m, scale, scale, one, k,
            )));
        }
    }
}

/// (peak bytes, allocations) of one run with the cold route on.
fn cold(profile: Profile, make: Make, n: usize, k: usize) -> (usize, usize) {
    let series = with_outliers(make(n), false);
    let _guard = forced_route(force(SpillForce::On));
    let mut allocations = 0;
    let peak = peak_of(|| {
        allocations = large_allocations_of(1, || run(profile, &series, k));
    });
    (peak, allocations)
}

/// (peak bytes, allocations) of the same run with the outliers as gaps.
fn hot(profile: Profile, make: Make, n: usize, k: usize) -> (usize, usize) {
    let series = with_outliers(make(n), true);
    let _guard = forced_route(force(SpillForce::Auto));
    let mut allocations = 0;
    let peak = peak_of(|| {
        allocations = large_allocations_of(1, || run(profile, &series, k));
    });
    (peak, allocations)
}

#[test]
fn memory_is_linear_and_allocations_do_not_grow_with_the_series() {
    let series: [(&str, Make); 2] = [
        ("constant", |n| vec![3.0; n]),
        ("few levels", |n| {
            (0..n)
                .map(|i| [5.0, 5.0, 9.0, 9.0, 9.0, 1.0][(i / 7) % 6])
                .collect()
        }),
    ];
    let (small_n, large_n) = (1_000, 4_000);
    for profile in [Profile::Left, Profile::TwoSided, Profile::Join] {
        for (name, make) in series {
            for k in [1, 4] {
                // What the cold route adds to the hot passes' own memory and allocations.
                let extra = |n: usize| {
                    let ((peak, count), (hot_peak, hot_count)) =
                        (cold(profile, make, n, k), hot(profile, make, n, k));
                    (
                        peak.saturating_sub(hot_peak),
                        count as i64 - hot_count as i64,
                    )
                };
                let ((small, small_count), (large, large_count)) = (extra(small_n), extra(large_n));
                for (extra, n) in [(small, small_n), (large, large_n)] {
                    assert!(
                        extra < n * 1024,
                        "{profile:?}, {name}, k {k}: {extra} bytes at {n}"
                    );
                }
                // A two-sided top-k profile builds its rows' lists differently with the cold route
                // (one allocation per row fewer), so its count is only bounded.
                let per_row_lists = matches!(profile, Profile::TwoSided) && k > 1;
                assert!(
                    small_count <= 64 && large_count <= 64,
                    "{profile:?}, {name}, k {k}: {small_count} and {large_count} allocations"
                );
                if !per_row_lists {
                    assert_eq!(
                        small_count, large_count,
                        "{profile:?}, {name}, k {k}: the cold route's allocations grew"
                    );
                }
            }
        }
    }
}
