#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! Peak heap use of the top-k batch profile on series where every pair ties exactly stays linear
//! in the series length: doubling it at most about doubles the peak. The peaks are measured
//! single-threaded, and the threaded runs are compared by their allocations of half a column or
//! more, which no schedule changes: a worker that copied a column would add one per worker. The
//! threaded series and its tiles are sized by the model of a worker's buffers (`CountedRun`). Its
//! own test binary, with a process-wide allocator (the work allocates on worker threads), and a
//! single test so nothing else allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::batch::{left_topk_profile, left_topk_profile_tiled};
use ladde::quantize::Scale;
use peak_alloc::{
    CountedRun, assert_equal_large_allocations, assert_workers_hold_less_than_half_a_column,
    peak_of,
};

type Make = fn(usize) -> Vec<f64>;

const BAND_ROWS: usize = 64;
const CHUNK_LANES: usize = 128;

#[test]
fn peak_heap_is_linear_in_the_series_under_exact_ties() {
    let m = NonZeroUsize::new(8).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let k = NonZeroUsize::new(4).unwrap();
    let nz = |x: usize| NonZeroUsize::new(x).unwrap();
    let series: [(&str, Make); 2] = [
        ("constant", |n| vec![3.0; n]),
        ("period four", |n| {
            (0..n).map(|i| [0.0, 2.0, 1.0, 3.0][i % 4]).collect()
        }),
    ];
    let counted = CountedRun::new(BAND_ROWS, CHUNK_LANES);
    for (name, make) in series {
        let run = |n: usize| {
            let series = make(n);
            peak_of(|| {
                let profile = left_topk_profile(&series, m, scale, nz(1), k);
                assert_eq!(profile.len(), n - 7);
            })
        };
        let (small, large) = (run(3_000), run(6_000));
        assert!(
            large <= small * 9 / 4,
            "{name}: peak {large} bytes at 6000 vs {small} at 3000"
        );
        assert!(large < 6_000 * 2048, "{name}: peak {large} bytes");

        let series = make(counted.samples);
        let run = |threads: usize| {
            left_topk_profile_tiled(
                &series,
                m,
                scale,
                nz(threads),
                nz(counted.band_rows),
                nz(counted.chunk_lanes),
                nz(counted.tile_lanes),
                k,
            )
        };
        let half_column = counted.half_column();
        assert_workers_hold_less_than_half_a_column(half_column, counted.shape(), run(2).1.tiles);
        assert_equal_large_allocations(half_column, &[2, 3, 8], name, |threads| {
            assert_eq!(run(threads).0.len(), series.len() - 7);
        });
    }
}
