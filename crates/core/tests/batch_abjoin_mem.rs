#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! Peak heap use of the AB-join grows linearly with the two series, even when every window of A
//! ties every window of B exactly, and the threaded join adds no copy of a column per worker: the
//! peaks are measured single-threaded and the threaded runs are compared by their allocations of
//! half a column or more, which no schedule changes. The threaded series and their tiles are sized
//! by the model of a worker's buffers (`CountedRun`).
//! Its own test binary, with a process-wide peak-tracking allocator (testkit's counts one thread
//! only, and the work here allocates on worker threads), and a single test so nothing else
//! allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::batch::ab_join_report_tiled;
use ladde::quantize::Scale;
use peak_alloc::{
    CountedRun, assert_equal_large_allocations, assert_workers_hold_less_than_half_a_column,
    peak_of,
};

const BAND_ROWS: usize = 64;
const CHUNK_LANES: usize = 128;
const TILE_LANES: usize = 1536;

/// A series every window of which repeats exactly with period `period`.
fn periodic(n: usize, period: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + (i % period) as f64).collect()
}

#[test]
fn peak_heap_is_linear_in_the_series_on_exact_ties() {
    let m = NonZeroUsize::new(40).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let nz = |x: usize| NonZeroUsize::new(x).unwrap();
    let join = |a: &[f64], b: &[f64], threads: usize, tile_lanes: usize| {
        let (profile, report) = ab_join_report_tiled(
            a,
            b,
            m,
            scale,
            scale,
            nz(threads),
            nz(BAND_ROWS),
            nz(CHUNK_LANES),
            nz(tile_lanes),
        );
        assert_eq!(profile.len(), a.len() - 39);
        assert!(profile.iter().all(|(_, e)| e.index().get().is_some()));
        report.tiles
    };
    let ties = |n: usize| (periodic(n, 7), periodic(n + n / 3, 7));
    let (a, b) = ties(4000);
    let (a2, b2) = ties(8000);
    // Single-threaded, so the peaks do not depend on a schedule.
    let peak = |a: &[f64], b: &[f64]| {
        peak_of(|| {
            join(a, b, 1, TILE_LANES);
        })
    };
    let (small, large) = (peak(&a, &b), peak(&a2, &b2));
    // Twice the series, not four times the pairs.
    assert!(
        large * 10 < small * 25,
        "peak {small} bytes at 4000 and {large} at 8000"
    );

    // A worker's own buffers are bounded by the tile plan and are below half a column of the
    // shorter series; so the allocations of half a column or more are the caller's, whatever the
    // thread count, and a worker that copied a column would add one per worker.
    let counted = CountedRun::new(BAND_ROWS, CHUNK_LANES);
    let (a, b) = ties(counted.samples);
    let half_column = counted.half_column();
    assert_workers_hold_less_than_half_a_column(
        half_column,
        counted.shape(),
        join(&a, &b, 2, counted.tile_lanes),
    );
    assert_equal_large_allocations(half_column, &[2, 3, 8, 16], "ab join", |threads| {
        join(&a, &b, threads, counted.tile_lanes);
    });
}
