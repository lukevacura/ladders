#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Keeping the right side costs the streams no allocation once warm: the bounded streams allocate
//! nothing after their first rows, and the growing ones only the doublings of the arrays that
//! grow with the windows, on ties and gaps, at one and several neighbors, on both arithmetics.
//! (A stream whose windows the cold route ranks keeps their lists in a map, which allocates.)

use core::num::NonZeroUsize;

use ladde::bounded::BoundedStream;
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use ladde::stream::GrowingStream;
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

const SAMPLES: usize = 5_000;

/// The arrays that grow with the windows (the lists and their spans, the windows, the floors, the
/// champions) double at most once between 2 500 and 5 000 windows: 4 096 is the only power of two
/// between.
const RIGHT_ARRAYS: usize = 7;

fn series(n: usize) -> Vec<Vec<f64>> {
    vec![
        vec![5.0; n],
        (0..n).map(|k| (k % 4) as f64).collect(),
        (0..n).map(|k| (k % 7) as f64).collect(),
        (0..n)
            .map(|k| {
                if k % 11 == 0 {
                    f64::NAN
                } else {
                    (k % 5) as f64
                }
            })
            .collect(),
        {
            let (mut x, mut p) = (7u64, 0.0);
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    p += (x % 9) as f64 - 4.0;
                    p
                })
                .collect()
        },
    ]
}

fn forces() -> [RouteForce; 2] {
    [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ]
}

fn measured(series: &[f64], mut update: impl FnMut(f64), warm: usize) -> usize {
    let (head, tail) = series.split_at(warm);
    head.iter().for_each(|&x| update(x));
    allocations_in(|| tail.iter().for_each(|&x| update(x))).1
}

#[test]
fn the_growing_stream_allocates_only_the_doublings_for_the_right_side() {
    for force in forces() {
        let _forced = forced_route(force);
        for (which, series) in series(SAMPLES).iter().enumerate() {
            for k in [1, 4] {
                let mut plain = GrowingStream::with_k(nz(16), unit(), nz(k));
                let mut right = GrowingStream::with_right_side(nz(16), unit(), nz(k));
                let base = measured(
                    series,
                    |x| {
                        let _ = plain.update_topk(x);
                    },
                    SAMPLES / 2,
                );
                let with = measured(
                    series,
                    |x| {
                        let _ = right.update_topk(x);
                    },
                    SAMPLES / 2,
                );
                assert!(
                    with <= base + RIGHT_ARRAYS,
                    "series {which} k {k} {force:?}: {with} against {base}"
                );
            }
        }
    }
}

#[test]
fn the_bounded_stream_allocates_nothing_for_the_right_side() {
    for force in forces() {
        let _forced = forced_route(force);
        for (which, series) in series(SAMPLES).iter().enumerate() {
            for k in [1, 4] {
                let mut right =
                    BoundedStream::with_right_side(nz(16), nz(300), unit(), nz(k)).unwrap();
                let count = measured(
                    series,
                    |x| {
                        let _ = right.update_topk(x);
                    },
                    400,
                );
                assert_eq!(count, 0, "series {which} k {k} {force:?}");
            }
        }
    }
}

#[test]
fn the_non_normalized_streams_allocate_only_the_doublings_for_the_right_side() {
    for (which, series) in series(SAMPLES).iter().enumerate() {
        for k in [1, 4] {
            let mut plain = GrowingNonNormStream::with_k(nz(16), unit(), nz(k));
            let mut right = GrowingNonNormStream::with_right_side(nz(16), unit(), nz(k));
            let base = measured(
                series,
                |x| {
                    let _ = plain.update_topk(x);
                },
                SAMPLES / 2,
            );
            let with = measured(
                series,
                |x| {
                    let _ = right.update_topk(x);
                },
                SAMPLES / 2,
            );
            assert!(
                with <= base + RIGHT_ARRAYS,
                "series {which} k {k}: {with} against {base}"
            );
            let mut bounded =
                BoundedNonNormStream::with_right_side(nz(16), nz(300), unit(), nz(k)).unwrap();
            let mut plain = BoundedNonNormStream::with_k(nz(16), nz(300), unit(), nz(k)).unwrap();
            let base = measured(
                series,
                |x| {
                    let _ = plain.update_topk(x);
                },
                400,
            );
            let with = measured(
                series,
                |x| {
                    let _ = bounded.update_topk(x);
                },
                400,
            );
            assert_eq!(with, base, "bounded series {which} k {k}");
        }
    }
}
