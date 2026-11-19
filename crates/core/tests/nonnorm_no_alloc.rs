#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized streams allocate nothing per row: the growing stream only the amortized
//! doubling of its arrays, on ties and on the wide arithmetic alike.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::route::{RouteForce, forced_route};
use nonnorm::*;
use testkit::alloc_count::{CountingAllocator, allocations_in, large_requests_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const UPDATES: usize = 3_000;

/// Each of the four arrays doubles at most once per power of two.
const DOUBLINGS: usize = 4 * 14;

fn tie_heavy() -> Vec<Vec<f64>> {
    let level = (1u64 << 33) as f64;
    vec![
        vec![5.0; UPDATES],
        (0..UPDATES).map(|k| (k % 2) as f64).collect(),
        (0..UPDATES).map(|k| (k % 7) as f64).collect(),
        (0..UPDATES)
            .map(|k| if (k / 50) % 2 == 0 { 0.0 } else { level })
            .collect(),
        (0..UPDATES).map(|k| level * (k % 3) as f64).collect(),
        (0..UPDATES)
            .map(|k| {
                if k % 11 == 0 {
                    f64::NAN
                } else {
                    (k % 5) as f64
                }
            })
            .collect(),
    ]
}

#[test]
fn the_growing_stream_allocates_only_to_grow() {
    for force in [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ] {
        let _forced = forced_route(force);
        for series in tie_heavy() {
            let mut stream = GrowingNonNormStream::new(nz(16), unit());
            let (_, count) = allocations_in(|| {
                for &x in &series {
                    let _ = stream.update(x);
                }
            });
            assert!(
                count <= DOUBLINGS,
                "{count} allocations for {UPDATES} updates"
            );
        }
    }
}

/// Everything the bounded stream owns is sized in `new`, so no update allocates, whatever the
/// ties, the arithmetic or the shocks crossing the bound in both directions.
#[test]
fn the_bounded_stream_allocates_nothing_after_new() {
    let mut shocks = tie_heavy();
    shocks.push(
        (0..UPDATES)
            .map(|k| match k % 400 {
                100..=103 => (1u64 << 40) as f64,
                _ => (k % 11) as f64,
            })
            .collect(),
    );
    for force in [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ] {
        let _forced = forced_route(force);
        for series in &shocks {
            for (m, w) in [(16, 64), (4, 4), (3, 200)] {
                let mut stream = BoundedNonNormStream::new(nz(m), nz(w), unit()).unwrap();
                let (_, count) = allocations_in(|| {
                    for &x in series {
                        let _ = stream.update(x);
                    }
                });
                assert_eq!(count, 0, "m {m} w {w}");
            }
        }
    }
}

const WARM: usize = 2_000;

fn forces() -> [RouteForce; 2] {
    [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ]
}

/// The allocations of the updates after the first `WARM`, keeping `k` neighbors.
fn growing_topk_allocations(series: &[f64], k: usize) -> usize {
    let mut stream = GrowingNonNormStream::with_k(nz(16), unit(), nz(k));
    for &x in &series[..WARM] {
        let _ = stream.update_topk(x);
    }
    let (_, count) = allocations_in(|| {
        for &x in &series[WARM..] {
            let _ = stream.update_topk(x);
        }
    });
    count
}

/// Keeping `k` neighbors costs the growing stream nothing beyond what it allocates for one: its
/// reserved lists stop growing once they hold `k`, long before the warm-up ends.
#[test]
fn the_growing_stream_allocates_no_more_for_k_than_for_one() {
    for force in forces() {
        let _forced = forced_route(force);
        for (which, series) in tie_heavy().iter().enumerate() {
            let single = growing_topk_allocations(series, 1);
            // A `k` of 16 keeps its neighbors the way both of these do.
            for k in [3, 40] {
                let count = growing_topk_allocations(series, k);
                assert!(
                    count <= single,
                    "series {which} k {k} {force:?}: {count} allocations against {single}"
                );
            }
        }
    }
}

/// Everything a row keeps is reserved by the first row with neighbors, so from the next row on a
/// bounded stream allocates nothing, whatever `k` is.
#[test]
fn the_bounded_stream_allocates_nothing_after_its_first_row_with_neighbors() {
    let mut shocks = tie_heavy();
    shocks.push(
        (0..UPDATES)
            .map(|k| match k % 400 {
                100..=103 => (1u64 << 40) as f64,
                _ => (k % 11) as f64,
            })
            .collect(),
    );
    for force in forces() {
        let _forced = forced_route(force);
        for series in &shocks {
            for (m, w) in [(16, 64), (4, 4), (3, 200), (5, 400)] {
                let first = m + ladde::exclusion::zone(m) + 1;
                for k in [1, 3, 16, 40] {
                    let mut stream =
                        BoundedNonNormStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
                    for &x in &series[..first] {
                        let _ = stream.update_topk(x);
                    }
                    let (_, count) = allocations_in(|| {
                        for &x in &series[first..] {
                            let _ = stream.update_topk(x);
                        }
                    });
                    assert_eq!(count, 0, "m {m} w {w} k {k} {force:?}");
                }
            }
        }
    }
}

/// A `k` beyond every window the stream can hold reserves what the windows can fill, not `k`.
#[test]
fn a_huge_k_reserves_only_what_the_windows_can_fill() {
    let _serial = forced_route(RouteForce::default());
    let series = &tie_heavy()[2];
    let k = nz(1 << 40);
    let (mut bounded, large) = large_requests_in(1 << 20, || {
        BoundedNonNormStream::with_k(nz(16), nz(1024), unit(), k).unwrap()
    });
    assert!(large.is_empty(), "{large:?}");
    let (_, large) = large_requests_in(1 << 20, || {
        for &x in &series[..WARM] {
            let _ = bounded.update_topk(x);
        }
    });
    assert!(large.is_empty(), "{large:?}");
    let (_, large) = large_requests_in(1 << 20, || {
        let mut growing = GrowingNonNormStream::with_k(nz(16), unit(), k);
        for &x in &series[..600] {
            let _ = growing.update_topk(x);
        }
    });
    assert!(large.is_empty(), "{large:?}");
}
