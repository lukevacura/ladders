#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! Peak heap use of the non-normalized batch profiles grows linearly with the series, on series
//! of exact ties as on a walk: a constant series offers every pair the same distance, and no
//! candidate list may grow with the pairs. The peaks are measured single-threaded, and the
//! threaded runs are compared by their allocations of half a column or more, which no schedule
//! changes: a worker that copied a column would add one per worker. The threaded series and its
//! tiles are sized by the model of a worker's buffers (`CountedRun`). Its own test binary, with a
//! process-wide allocator (the work allocates on worker threads), and a single test so nothing
//! else allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::batch::nonnorm::{
    ab_join, ab_join_report_tiled, left_profile, left_profile_report_tiled, two_sided_profile,
};
use ladde::quantize::Scale;
use peak_alloc::{
    CountedRun, assert_equal_large_allocations, assert_workers_hold_less_than_half_a_column,
    peak_of,
};

const BAND_ROWS: usize = 64;
const CHUNK_LANES: usize = 128;

fn walk(n: usize) -> Vec<f64> {
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

type Series = dyn Fn(usize) -> Vec<f64>;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn m() -> NonZeroUsize {
    nz(16)
}

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

fn peaks(series: &[f64]) -> [usize; 3] {
    let windows = series.len() - 15;
    [
        peak_of(|| assert_eq!(left_profile(series, m(), scale(), nz(1)).len(), windows)),
        peak_of(|| {
            assert_eq!(
                two_sided_profile(series, m(), scale(), nz(1)).len(),
                windows
            )
        }),
        peak_of(|| {
            assert_eq!(
                ab_join(series, &series[..windows], m(), scale(), nz(1)).len(),
                windows
            )
        }),
    ]
}

/// Left profile and join, each at several thread counts. (Not the two-sided profile: its shared
/// right-neighbor candidate lists grow in the order tiles merge, so the number of large
/// reallocations there follows the schedule although the result does not. Its ties are covered by
/// the single-threaded peaks.)
fn assert_threads_add_no_column(name: &str, make: &Series) {
    let counted = CountedRun::new(BAND_ROWS, CHUNK_LANES);
    let series = make(counted.samples);
    let series = series.as_slice();
    let windows = series.len() - 15;
    let (band, chunk, lanes) = (
        nz(counted.band_rows),
        nz(counted.chunk_lanes),
        nz(counted.tile_lanes),
    );
    let half_column = counted.half_column();
    let left = |threads: usize| {
        left_profile_report_tiled(series, m(), scale(), nz(threads), band, chunk, lanes)
    };
    let join = |threads: usize| {
        ab_join_report_tiled(
            series,
            &series[..windows],
            m(),
            scale(),
            nz(threads),
            band,
            chunk,
            lanes,
        )
    };
    let tiles = left(2).1.tiles.max(join(2).1.tiles);
    assert_workers_hold_less_than_half_a_column(half_column, counted.shape(), tiles);
    let threads = [2, 3, 8];
    assert_equal_large_allocations(half_column, &threads, &format!("{name} left"), |t| {
        assert_eq!(left(t).0.len(), windows);
    });
    assert_equal_large_allocations(half_column, &threads, &format!("{name} join"), |t| {
        assert_eq!(join(t).0.len(), windows);
    });
}

#[test]
fn peak_heap_is_linear_in_the_series_on_ties_and_walks() {
    let (small, big) = (3_000, 12_000);
    let constant = |n: usize| vec![5.0; n];
    let period_two = |n: usize| (0..n).map(|k| (k % 2) as f64).collect::<Vec<_>>();
    let kinds: [(&str, &Series); 3] = [
        ("constant", &constant),
        ("period two", &period_two),
        ("walk", &walk),
    ];
    for (name, make) in kinds {
        let (few, many) = (peaks(&make(small)), peaks(&make(big)));
        for (which, (few, many)) in ["left", "two-sided", "join"]
            .iter()
            .zip(few.iter().zip(&many))
        {
            // Output entries, ticks and right-side minima: a few hundred bytes per window at
            // most; the bands' scratch is flat in the series.
            assert!(
                *many < big * 256,
                "{name} {which}: {many} bytes at n = {big}"
            );
            // Three extra copies of the added samples' share of a pass would be quadratic; the
            // growth stays within a linear allowance for the added 9,000 samples.
            assert!(
                many.saturating_sub(*few) < (big - small) * 256,
                "{name} {which}: {few} bytes at n = {small}, {many} at n = {big}"
            );
        }
        assert_threads_add_no_column(name, make);
    }
}
