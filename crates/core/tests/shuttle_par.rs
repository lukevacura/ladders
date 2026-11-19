#![cfg(all(shuttle, feature = "parallel"))]

use core::num::NonZeroUsize;

use ladde::par::for_each_chunk;
use ladde::sync::{AtomicUsize, Ordering};

/// Random schedules: each index runs exactly once and lands at its own position.
#[test]
fn shuttle_every_index_once_in_order() {
    shuttle::check_random_with_seed(
        || {
            for (threads, chunks) in [(1, 5), (2, 0), (2, 1), (2, 7), (3, 2), (4, 9)] {
                let hits: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
                let out = for_each_chunk(NonZeroUsize::new(threads).unwrap(), chunks, |i| {
                    hits[i].fetch_add(1, Ordering::SeqCst);
                    i + 100
                });
                let expected: Vec<usize> = (0..chunks).map(|i| i + 100).collect();
                assert_eq!(out, expected);
                assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
            }
        },
        testkit::generators::TEST_SEED,
        300,
    );
}
