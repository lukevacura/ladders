#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Keeping `k` neighbors per row costs the streams no allocation once warm: the bounded stream
//! allocates nothing after its first row with neighbors, and the growing stream only what it
//! allocates at `k` of one (the amortized doubling of the arrays that grow with the samples), on
//! ties, gaps and both arithmetics.

use core::num::NonZeroUsize;

use ladde::bounded::BoundedStream;
use ladde::exclusion::zone;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use ladde::stream::GrowingStream;
use testkit::alloc_count::{CountingAllocator, allocations_in, large_requests_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

/// The growing stream warms up on the first half of a series and is measured on the second. A
/// row's lists of tied candidates grow with the ties it has seen until they hold `k`; on the walk
/// that takes past 2 000 samples, so the warm-up cannot be shorter.
const GROWING_SAMPLES: usize = 5_000;

/// The arrays that grow with the samples (ticks, window sums, ranking columns, presence flags,
/// narrowed copies, the diagonal buffer, the wide scratch, and the cold route's spilled windows,
/// their bitmap and its dot products by lag) double at most once between 2 500 and 5 000
/// samples, as 4 096 is the only power of two between.
const GROWING_ARRAYS: usize = 13;

/// How long a bounded stream runs once its window is full: long enough for every feature of the
/// tie-heavy series (plateaus of 50, gaps every 11, a drift step every 400) to pass through it.
const PAST_FULL: usize = 450;

fn tie_heavy(n: usize) -> Vec<Vec<f64>> {
    let level = (1u64 << 33) as f64;
    vec![
        vec![5.0; n],
        // A walk whose near-tie records keep being set long after any fixed warm-up.
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
        (0..n).map(|k| (k % 2) as f64).collect(),
        (0..n).map(|k| (k % 4) as f64).collect(),
        (0..n).map(|k| (k % 7) as f64).collect(),
        (0..n)
            .map(|k| if (k / 50) % 2 == 0 { 0.0 } else { level })
            .collect(),
        (0..n)
            .map(|k| {
                if k % 11 == 0 {
                    f64::NAN
                } else {
                    (k % 5) as f64
                }
            })
            .collect(),
        drift(n),
    ]
}

/// A drifting few-level series: it keeps setting new near-tie records long after warm-up.
fn drift(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((k * 7 + 13) % 17) as f64 + (k / 400) as f64)
        .collect()
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

/// The allocations of the updates in the second half of `series`, with `k` neighbors kept.
fn growing_allocations(series: &[f64], m: usize, k: usize) -> usize {
    let (warm, measured) = series.split_at(series.len() / 2);
    let mut stream = GrowingStream::with_k(nz(m), unit(), nz(k));
    for &x in warm {
        let _ = stream.update_topk(x);
    }
    let (_, count) = allocations_in(|| {
        for &x in measured {
            let _ = stream.update_topk(x);
        }
    });
    count
}

/// The first `m + zone(m) + 1` samples: the bounded stream's first row with neighbors.
fn first_with_neighbors(m: usize) -> usize {
    m + zone(m) + 1
}

/// The allocations of a bounded stream's updates after its first row with neighbors, with `k`
/// neighbors kept.
fn bounded_allocations(series: &[f64], m: usize, w: usize, k: usize) -> usize {
    let (warm, measured) = series.split_at(first_with_neighbors(m));
    let mut stream = BoundedStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
    for &x in warm {
        let _ = stream.update_topk(x);
    }
    let (_, count) = allocations_in(|| {
        for &x in measured {
            let _ = stream.update_topk(x);
        }
    });
    count
}

#[test]
fn the_growing_stream_allocates_no_more_for_k_than_for_one() {
    for force in forces() {
        let _forced = forced_route(force);
        for (which, series) in tie_heavy(GROWING_SAMPLES).iter().enumerate() {
            let single = growing_allocations(series, 16, 1);
            // Every `k` from 2 to the pruning threshold keeps its neighbors the same way; a `k` of
            // 16 runs in `a_late_near_tie_record_costs_the_growing_stream_only_doublings`.
            let k = 3;
            let count = growing_allocations(series, 16, k);
            assert!(
                count <= single,
                "series {which} k {k} {force:?}: {count} allocations against {single}"
            );
            assert!(
                count <= GROWING_ARRAYS,
                "series {which} k {k} {force:?}: {count} allocations, more than one doubling of \
                 each array that grows with the samples"
            );
        }
    }
}

/// Every shape runs from its first row with neighbors until its window has been full for
/// [`PAST_FULL`] samples, so the measured rows include the ring's ramp-up, its wrap and rows at
/// the most neighbors the window holds.
#[test]
fn the_bounded_stream_allocates_nothing_after_warm_up() {
    for force in forces() {
        let _forced = forced_route(force);
        // No row with neighbors; fewer neighbors than the largest `k`; a few more; more than the
        // candidates a row resolves at once, at every `k`; one full tile; two tiles, the
        // candidates capped by the tile rather than by the neighbors. Past the resolve point every
        // `k` from 2 to the pruning threshold takes the same path, so the tile shapes keep one.
        let every_k: &[usize] = &[1, 3, 16, 40];
        for (m, w, ks) in [
            (4, 4, every_k),
            (8, 33, every_k),
            (16, 64, every_k),
            (3, 200, every_k),
            (16, 1024, &[1, 3]),
            (5, 1200, &[1, 3]),
        ] {
            let n = first_with_neighbors(m) + w + PAST_FULL;
            for (which, series) in tie_heavy(n).iter().enumerate() {
                for &k in ks {
                    let count = bounded_allocations(series, m, w, k);
                    assert_eq!(count, 0, "series {which} m {m} w {w} k {k} {force:?}");
                }
            }
        }
    }
}

/// A `k` beyond every window the stream can hold reserves what the windows can fill, not `k`.
#[test]
fn a_huge_k_reserves_only_what_the_windows_can_fill() {
    let _serial = forced_route(RouteForce::default());
    let series = &tie_heavy(2_500)[6];
    let k = nz(1 << 40);
    let (mut bounded, large) = large_requests_in(1 << 20, || {
        BoundedStream::with_k(nz(16), nz(1024), unit(), k).unwrap()
    });
    assert!(large.is_empty(), "{large:?}");
    let (_, large) = large_requests_in(1 << 20, || {
        for &x in series {
            let _ = bounded.update_topk(x);
        }
    });
    assert!(large.is_empty(), "{large:?}");
    let (_, large) = large_requests_in(1 << 20, || {
        let mut growing = GrowingStream::with_k(nz(16), unit(), k);
        for &x in &series[..600] {
            let _ = growing.update_topk(x);
        }
    });
    assert!(large.is_empty(), "{large:?}");
}

/// A drifting few-level series keeps setting new near-tie records, so a row long after warm-up
/// can pile up more tied candidates than any row before it: still no allocation, at `k` of 16.
#[test]
fn a_late_near_tie_record_allocates_nothing_at_k_16() {
    let _serial = forced_route(RouteForce::default());
    let series = drift(3_500);
    let mut stream = BoundedStream::with_k(nz(5), nz(2400), unit(), nz(16)).unwrap();
    for &x in &series[..2_500] {
        let _ = stream.update_topk(x);
    }
    let (_, count) = allocations_in(|| {
        for &x in &series[2_500..] {
            let _ = stream.update_topk(x);
        }
    });
    assert_eq!(count, 0);
}

/// Everything a row can need is reserved by the first row that has neighbors: from the next row
/// on a bounded stream allocates nothing, whatever `k` is.
#[test]
fn a_bounded_stream_allocates_nothing_after_its_first_row_with_neighbors() {
    let _serial = forced_route(RouteForce::default());
    let (m, w) = (5, 400);
    for series in tie_heavy(first_with_neighbors(m) + w + PAST_FULL) {
        for k in [1, 3, 16, 40] {
            assert_eq!(bounded_allocations(&series, m, w, k), 0, "k {k}");
        }
    }
}

/// The same late near-tie records in the growing stream cost it only the doubling of its arrays.
#[test]
fn a_late_near_tie_record_costs_the_growing_stream_only_doublings() {
    let _serial = forced_route(RouteForce::default());
    let series = drift(5_000);
    for m in [5, 16] {
        let count = growing_allocations(&series, m, 16);
        assert!(count <= GROWING_ARRAYS, "m {m}: {count} allocations");
    }
}
