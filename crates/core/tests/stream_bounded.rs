#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! `BoundedStream`'s collected output always satisfies `testkit::validate::left_profile` and
//! `complete_rows`, checked on market-shaped series quantized to ticks; and its memory stays
//! fixed once `w` samples have arrived, however much longer the stream runs. `testkit` is a
//! dev-dependency of `ladde`, and `ladde` is a dependency of `testkit`; that
//! cycle is fine here because this integration test links against the compiled library rather
//! than its source, but it is why the first check can't live in `bounded.rs`'s own unit tests
//! (see `naive_left_profile.rs`), and why the second runs as its own test binary rather than
//! alongside them (see `stats_no_alloc.rs`).

use core::num::NonZeroUsize;
use ladde::bounded::BoundedStream;
use ladde::quantize::Scale;
use proptest::prelude::*;
use testkit::alloc_count::{CountingAllocator, allocations_in};
use testkit::generators::{MARKET_TICK, market_series_and_m};
use testkit::validate::{complete_rows, left_profile};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn bounded_output_satisfies_left_profile_invariants(
        (series, m) in market_series_and_m(0..48),
        extra in 0usize..32,
    ) {
        let _serial = ladde::route::forced_route(ladde::route::RouteForce::default());
        let scale = Scale::new(MARKET_TICK).unwrap();
        let w = m + extra;
        let mut stream = BoundedStream::new(
            NonZeroUsize::new(m).unwrap(),
            NonZeroUsize::new(w).unwrap(),
            scale,
        )
        .expect("w is always m or more here");
        let profile: Vec<_> = series
            .iter()
            .filter_map(|&x| stream.update(x))
            .map(|(row, entry)| (row.get().expect("row is never NONE"), entry))
            .collect();

        prop_assert_eq!(left_profile(&profile, m), Ok(()));
        prop_assert_eq!(complete_rows(&profile, series.len(), m), Ok(()));
    }
}

/// Every allocation `BoundedStream` ever makes happens in `new`: pushing samples past `w`
/// forever after must allocate nothing, since nothing it owns is sized by sample count.
#[test]
fn capacity_stays_fixed_after_w_samples() {
    let _serial = ladde::route::forced_route(ladde::route::RouteForce::default());
    let m = NonZeroUsize::new(16).unwrap();
    let w = NonZeroUsize::new(64).unwrap();
    let scale = Scale::new(MARKET_TICK).unwrap();
    let mut stream = BoundedStream::new(m, w, scale).expect("w is always m or more here");

    for k in 0..w.get() {
        stream.update(k as f64 * MARKET_TICK);
    }

    let (_, count) = allocations_in(|| {
        for k in 0..10_000 {
            stream.update((k as f64).sin() * 100.0);
        }
    });
    assert_eq!(count, 0);
}

/// Neither a re-anchor nor the switch to Wide owns anything sized by sample count either: both
/// are O(w) *work*, not O(w) *allocation* (`reanchor_live_diagonals` and `switch_to_wide` only
/// ever write into `diagonals`, already sized once in `new`). Checked on the one tick that
/// triggers each, and on the 100 ticks after, all on the stream `new` already allocated for.
#[test]
fn re_anchor_and_the_switch_to_wide_allocate_nothing() {
    let _serial = ladde::route::forced_route(ladde::route::RouteForce::default());
    let m = NonZeroUsize::new(4).unwrap();
    let w = NonZeroUsize::new(14).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let mut stream = BoundedStream::new(m, w, scale).expect("w is always m or more here");

    let bound = i64::from(i32::MAX) / m.get() as i64;
    let flush = w.get() + 2;

    // Flush the initial (anchor = 0) plateau fully through the ring first; not measured.
    for _ in 0..flush {
        stream.update(0.0);
    }

    // The re-anchor tick itself: `2 * bound` is exactly the widest span a re-anchor can still
    // cover from a plateau at `0`.
    let level_1 = 2 * bound;
    let (_, re_anchor_count) = allocations_in(|| {
        stream.update(level_1 as f64);
    });
    assert_eq!(re_anchor_count, 0);

    // Flush the new plateau fully through the ring before the next jump; not measured.
    for _ in 0..flush {
        stream.update(level_1 as f64);
    }

    // The switch-to-wide tick itself: `2 * bound + 1` past the current plateau is wider than
    // any anchor could cover.
    let level_2 = level_1 + 2 * bound + 1;
    let (_, switch_count) = allocations_in(|| {
        stream.update(level_2 as f64);
    });
    assert_eq!(switch_count, 0);

    // 100 ticks after, permanently on the Wide path now.
    let (_, after_count) = allocations_in(|| {
        for k in 0..100 {
            stream.update(level_2 as f64 + (k % 7) as f64 - 3.0);
        }
    });
    assert_eq!(after_count, 0);
}

/// A series of exact ties, which pile up candidates, still costs the stream no allocation once
/// `w` samples have arrived.
#[test]
fn exact_ties_allocate_nothing_after_w_samples() {
    let _serial = ladde::route::forced_route(ladde::route::RouteForce::default());
    let m = NonZeroUsize::new(16).unwrap();
    let w = NonZeroUsize::new(1024).unwrap();
    let scale = Scale::new(1.0).unwrap();
    for sample in [|_: usize| 3.0, |k: usize| (k % 4) as f64] {
        let mut stream = BoundedStream::new(m, w, scale).expect("w is m or more");
        for k in 0..2 * w.get() {
            stream.update(sample(k));
        }
        let (_, count) = allocations_in(|| {
            for k in 0..5000 {
                stream.update(sample(k));
            }
        });
        assert_eq!(count, 0);
    }
}

/// The same ties on the exact i64 path cost no allocation after `w` samples either, whether the
/// path was forced from the first sample or reached by a mid-stream jump.
#[test]
fn exact_ties_allocate_nothing_on_the_wide_path() {
    use ladde::narrow::Path;
    use ladde::route::{RouteForce, forced_route};

    let m = NonZeroUsize::new(16).unwrap();
    let w = NonZeroUsize::new(1024).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let samples: [fn(usize) -> f64; 3] = [|_| 3.0, |k| (k % 4) as f64, |k| (k % 7) as f64 * 1e8];
    for forced in [true, false] {
        for (which, sample) in samples.into_iter().enumerate() {
            // Unforced runs hold the guard too, so no other test's forced choice reaches them.
            let _guard = forced_route(RouteForce {
                wide: forced,
                ..RouteForce::default()
            });
            let mut stream = BoundedStream::new(m, w, scale).expect("w is m or more");
            if !forced {
                for k in 0..w.get() {
                    stream.update((k % 3) as f64);
                }
                stream.update(1e15);
            }
            for k in 0..2 * w.get() {
                stream.update(sample(k));
            }
            // The small-span series let an unforced stream return to narrow, which allocates
            // nothing either.
            if forced || which == 2 {
                assert_eq!(stream.path(), Path::Wide);
            }
            let (_, count) = allocations_in(|| {
                for k in 0..5000 {
                    stream.update(sample(k));
                }
            });
            assert_eq!(count, 0);
        }
    }
}

/// A shock that leaves the buffer (the switch back to the narrow path, O(w) work on its row) and
/// a series that keeps shocking and returning allocate nothing once the stream is warm.
#[test]
fn the_switch_back_to_narrow_allocates_nothing() {
    let _unforced = ladde::route::forced_route(ladde::route::RouteForce::default());
    let m = NonZeroUsize::new(4).unwrap();
    let w = NonZeroUsize::new(14).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let shock = 3.0 * (i64::from(i32::MAX) / m.get() as i64) as f64;
    let small = |k: usize| ((k * 5) % 11) as f64;
    let mut stream = BoundedStream::new(m, w, scale).expect("w is always m or more here");
    for k in 0..w.get() + 2 {
        stream.update(small(k));
    }
    let (_, count) = allocations_in(|| {
        for round in 0..20 {
            for k in 0..3 {
                stream.update(shock + k as f64);
            }
            for k in 0..30 + 13 * round {
                stream.update(small(k));
            }
        }
    });
    assert_eq!(count, 0);
    let counts = stream.route_counts();
    assert!(
        counts.switches_to_narrow > 0 && counts.switches_to_narrow <= counts.switches_to_wide,
        "{counts:?}"
    );
}
