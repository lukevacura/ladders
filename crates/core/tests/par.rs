#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

use core::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
// timing-ok: a hang-guard bound only; correct code never waits on the clock
use std::time::{Duration, Instant};

use ladde::par::{for_each_chunk, threads_available};

/// Spin until `cond` holds; false after `HANG_GUARD`, so a missing second thread fails instead
/// of hanging. Correct code never waits on the clock: `cond` holds by ordering.
const HANG_GUARD: Duration = Duration::from_secs(60);

fn spin_until(cond: impl Fn() -> bool) -> bool {
    // timing-ok: a hang-guard bound only; correct code never waits on the clock
    let start = Instant::now();
    while !cond() {
        // timing-ok: a hang-guard bound only; correct code never waits on the clock
        if start.elapsed() > HANG_GUARD {
            return false;
        }
        // timing-ok: the loop is bounded by the hang guard; correct code never waits on the clock
        std::hint::spin_loop();
    }
    true
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[test]
fn every_index_once_in_order() {
    for threads in [1, 2, 7, 64] {
        for chunks in [0, 1, threads - 1, threads + 1, 10_000] {
            let hits: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
            let out = for_each_chunk(nz(threads), chunks, |i| {
                hits[i].fetch_add(1, Ordering::SeqCst);
                i * 3 + 1
            });
            let expected: Vec<usize> = (0..chunks).map(|i| i * 3 + 1).collect();
            assert_eq!(out, expected, "threads {threads} chunks {chunks}");
            assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
        }
    }
}

#[test]
fn one_thread_runs_inline() {
    // timing-ok: compares thread ids for equality; the assertion holds under every schedule
    let caller = std::thread::current().id();
    // timing-ok: compares thread ids for equality; the assertion holds under every schedule
    let ids = for_each_chunk(nz(1), 50, |_| std::thread::current().id());
    assert!(ids.iter().all(|id| *id == caller));
}

#[test]
fn threads_share_the_work() {
    // Two chunks that each wait for the other can only finish on two threads.
    let arrived = AtomicUsize::new(0);
    let out = for_each_chunk(nz(2), 2, |i| {
        arrived.fetch_add(1, Ordering::SeqCst);
        let met = spin_until(|| arrived.load(Ordering::SeqCst) >= 2);
        assert!(met, "no second thread ran");
        i
    });
    assert_eq!(out, vec![0, 1]);
}

#[test]
fn results_may_be_non_copy_and_borrow() {
    let words = ["a", "bb", "ccc"];
    let out = for_each_chunk(nz(3), 3, |i| words[i].repeat(2));
    assert_eq!(out, vec!["aa", "bbbb", "cccccc"]);
}

#[test]
fn closure_panic_propagates() {
    let seen = Mutex::new(0);
    let r = std::panic::catch_unwind(|| {
        for_each_chunk(nz(4), 100, |i| {
            *seen.lock().unwrap() += 1;
            assert_ne!(i, 37, "boom");
        })
    });
    assert!(r.is_err());
}

#[test]
fn available_threads_is_at_least_one() {
    // timing-ok: compared with the core crate's own count; no outcome depends on it
    let std_count = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    assert_eq!(threads_available().get(), std_count);
}

#[test]
fn worker_panic_payload_reaches_the_caller() {
    // The calling thread's chunk waits until a worker has started, and every worker chunk
    // panics, so the only panic is on a worker and must come back with its own payload.
    // timing-ok: compares thread ids for equality; the assertion holds under every schedule
    let caller = std::thread::current().id();
    let worker_started = AtomicUsize::new(0);
    let r = std::panic::catch_unwind(|| {
        for_each_chunk(nz(2), 2, |_| {
            // timing-ok: compares thread ids for equality; the assertion holds under every schedule
            if std::thread::current().id() == caller {
                let met = spin_until(|| worker_started.load(Ordering::SeqCst) != 0);
                assert!(met, "no second thread ran");
            } else {
                worker_started.store(1, Ordering::SeqCst);
                panic!("worker boom");
            }
        })
    });
    let payload = r.unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"worker boom"));
}
