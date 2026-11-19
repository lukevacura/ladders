#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! Peak heap use of the multi-dimensional batch profile grows linearly with the series and with
//! the number of dimensions, on series of exact ties as on a walk: a constant series offers every
//! pair the same distance, and no candidate list may grow with the pairs. The peaks are measured
//! single-threaded, and the threaded runs are compared by their allocations of half a column or
//! more, which no schedule changes: a worker that copied a column would add one per worker. The
//! threaded series and its tiles are sized by the model of a worker's buffers (`CountedRun`). Its
//! own test binary, with a process-wide allocator (the work allocates on worker threads), and a
//! single test so nothing else allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::batch::multi::{maamp, maamp_report_tiled, mstump, mstump_report_tiled};
use ladde::quantize::Scale;
use peak_alloc::{
    CountedRun, assert_equal_large_allocations, assert_workers_hold_less_than_half_a_column,
    peak_of,
};

const BAND_ROWS: usize = 64;
const CHUNK_LANES: usize = 128;

fn walk(n: usize, salt: u64) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ salt;
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

type Series = dyn Fn(usize, u64) -> Vec<f64>;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn dimensions_of(make: &Series, n: usize, dimensions: usize) -> Vec<Vec<f64>> {
    (0..dimensions).map(|a| make(n, a as u64)).collect()
}

fn peak(make: &Series, n: usize, dimensions: usize, normalized: bool) -> usize {
    let series = dimensions_of(make, n, dimensions);
    let refs: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let scales = vec![Scale::new(1.0).unwrap(); dimensions];
    peak_of(|| {
        let windows = if normalized {
            mstump(&refs, nz(16), &scales, &[], false, nz(1))
                .unwrap()
                .distance[0]
                .len()
        } else {
            maamp(&refs, nz(16), scales[0], &[], false, nz(1))
                .unwrap()
                .distance[0]
                .len()
        };
        assert_eq!(windows, n - 15);
    })
}

fn assert_threads_add_no_column(name: &str, make: &Series, normalized: bool) {
    let dimensions = 2;
    let counted = CountedRun::new(BAND_ROWS, CHUNK_LANES);
    let n = counted.samples;
    let series = dimensions_of(make, n, dimensions);
    let refs: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let scales = vec![Scale::new(1.0).unwrap(); dimensions];
    let (band, chunk, tile) = (
        nz(counted.band_rows),
        nz(counted.chunk_lanes),
        nz(counted.tile_lanes),
    );
    let run = |threads: usize| {
        if normalized {
            let (profile, _) = mstump_report_tiled(
                &refs,
                nz(16),
                &scales,
                &[],
                false,
                nz(threads),
                band,
                chunk,
                tile,
            )
            .unwrap();
            profile.distance[0].len()
        } else {
            let (profile, _) = maamp_report_tiled(
                &refs,
                nz(16),
                scales[0],
                &[],
                false,
                nz(threads),
                band,
                chunk,
                tile,
            )
            .unwrap();
            profile.distance[0].len()
        }
    };
    assert_workers_hold_less_than_half_a_column(
        counted.half_column(),
        counted.shape(),
        n.div_ceil(counted.band_rows) * n.div_ceil(counted.tile_lanes),
    );
    let label = format!("{name}, {dimensions} dimensions, normalized {normalized}");
    assert_equal_large_allocations(counted.half_column(), &[2, 8], &label, |threads| {
        assert_eq!(run(threads), n - 15);
    });
}

#[test]
fn peak_heap_is_linear_in_the_series_and_the_dimensions() {
    let (small, big) = (400, 1_600);
    let constant = |n: usize, _: u64| vec![5.0; n];
    let period_two = |n: usize, _: u64| (0..n).map(|k| (k % 2) as f64).collect::<Vec<_>>();
    let kinds: [(&str, &Series); 3] = [
        ("constant", &constant),
        ("period two", &period_two),
        ("walk", &walk),
    ];
    for (name, make) in kinds {
        for (dimensions, normalized) in [(2, true), (4, true), (2, false), (4, false)] {
            let (few, many) = (
                peak(make, small, dimensions, normalized),
                peak(make, big, dimensions, normalized),
            );
            // Ticks, columns, a slot per dimension and window, and the output: a few hundred bytes
            // per window and dimension at most; the bands' scratch is flat in the series.
            let per_window = 400 * dimensions;
            assert!(
                many < big * per_window,
                "{name} d {dimensions} {normalized}: {many} bytes at n = {big}"
            );
            assert!(
                many.saturating_sub(few) < (big - small) * per_window,
                "{name} d {dimensions} {normalized}: {few} bytes at n = {small}, {many} at n = {big}"
            );
        }
        for normalized in [true, false] {
            assert_threads_add_no_column(name, make, normalized);
        }
    }
}
