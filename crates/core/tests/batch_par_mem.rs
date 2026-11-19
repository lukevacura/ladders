#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! The memory of `left_profile_threads_with` does not grow with the thread count: the per-worker
//! state is a tile scratch and a band's accumulators, none of it of the series' length. Its own
//! test binary, with a process-wide allocator (testkit's counts one thread only, and the work here
//! allocates on worker threads), and a single test so nothing else allocates meanwhile.
//!
//! A peak depends on how the threads interleave, so no peak bound is both exact and tight enough to
//! see a per-worker copy of one length-n column. What is exact is the allocations: everything of
//! the series' length is allocated once by the caller, and each worker's own allocations are
//! bounded by the tile plan. So the number of allocations as large as half a column is the same
//! for any thread count, and a worker that copied a column would add one per worker.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::batch::left_profile_threads_with;
use ladde::quantize::Scale;
use peak_alloc::{
    assert_equal_large_allocations, assert_workers_hold_less_than_half_a_column, peak_of,
};

fn series_of(n: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut level = 1e10;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 56) as i64 % 5 - 2) as f64;
            level
        })
        .collect()
}

fn run(series: &[f64], threads: usize, band_rows: usize, chunk_lanes: usize) {
    let m = NonZeroUsize::new(50).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let profile = left_profile_threads_with(
        series,
        m,
        scale,
        NonZeroUsize::new(threads).unwrap(),
        NonZeroUsize::new(band_rows).unwrap(),
        NonZeroUsize::new(chunk_lanes).unwrap(),
    );
    assert_eq!(profile.len(), series.len() - 49);
}

#[test]
fn memory_does_not_grow_with_threads() {
    let n = 16_000;
    let (band_rows, chunk_lanes) = (64, 2_048);
    let series = series_of(n);

    // O(n) at all: about 220 bytes per sample (columns, output, per-row accumulators, tile list).
    let two = peak_of(|| run(&series, 2, band_rows, chunk_lanes));
    assert!(two < n * 1024, "two-thread peak {two} bytes");

    // Half a column of ticks. Each worker's own allocations are well below it: its tile scratch
    // holds `chunk_lanes` words of lag state and of rho and `band_rows` row accumulators, and the
    // list of finished tiles it keeps is one small record per tile it claimed.
    let half_column = 4 * n;
    let tiles = n.div_ceil(band_rows) * n.div_ceil(chunk_lanes);
    assert_workers_hold_less_than_half_a_column(
        half_column,
        (band_rows, chunk_lanes, chunk_lanes),
        tiles,
    );

    assert_equal_large_allocations(half_column, &[2, 3, 8, 16], "left profile", |threads| {
        run(&series, threads, band_rows, chunk_lanes)
    });
}
