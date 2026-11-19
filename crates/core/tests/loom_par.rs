#![cfg(all(loom, feature = "parallel"))]

use core::num::NonZeroUsize;

use ladde::par::for_each_chunk;
use ladde::sync::{Arc, AtomicUsize, Ordering};

/// Two threads, three chunks, every interleaving: each index runs once, results in order.
#[test]
fn loom_two_threads_three_chunks() {
    loom::model(|| {
        let hits: Arc<Vec<AtomicUsize>> = Arc::new((0..3).map(|_| AtomicUsize::new(0)).collect());
        let seen = Arc::clone(&hits);
        let out = for_each_chunk(NonZeroUsize::new(2).unwrap(), 3, move |i| {
            seen[i].fetch_add(1, Ordering::SeqCst);
            i * 10
        });
        assert_eq!(out, vec![0, 10, 20]);
        for hit in hits.iter() {
            assert_eq!(hit.load(Ordering::SeqCst), 1);
        }
    });
}

/// The closure may borrow from the caller's stack: workers are joined before the call returns.
#[test]
fn loom_workers_borrow_the_callers_data() {
    loom::model(|| {
        let hits: Vec<AtomicUsize> = (0..3).map(|_| AtomicUsize::new(0)).collect();
        let words = ["a", "b", "c"];
        let out = for_each_chunk(NonZeroUsize::new(2).unwrap(), 3, |i| {
            hits[i].fetch_add(1, Ordering::SeqCst);
            words[i].repeat(2)
        });
        assert_eq!(out, vec!["aa", "bb", "cc"]);
        for hit in &hits {
            assert_eq!(hit.load(Ordering::SeqCst), 1);
        }
    });
}

/// A chunk that panics reaches the caller, and every chunk still ran: the other thread keeps
/// claiming, and no worker is left unjoined whichever thread hit the panic.
#[test]
fn loom_a_panicking_chunk_propagates_and_every_chunk_runs() {
    loom::model(|| {
        let hits: Vec<AtomicUsize> = (0..3).map(|_| AtomicUsize::new(0)).collect();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for_each_chunk(NonZeroUsize::new(2).unwrap(), 3, |i| {
                hits[i].fetch_add(1, Ordering::SeqCst);
                assert_ne!(i, 1, "chunk one fails");
                i
            })
        }));
        assert!(outcome.is_err());
        for hit in &hits {
            assert_eq!(hit.load(Ordering::SeqCst), 1);
        }
    });
}

/// A panic in the scope's own closure still joins every spawned worker before it unwinds.
#[test]
fn loom_scope_joins_its_workers_when_its_closure_panics() {
    loom::model(|| {
        let done = AtomicUsize::new(0);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ladde::sync::thread::scope(|scope| {
                scope.spawn(|| {
                    done.fetch_add(1, Ordering::SeqCst);
                });
                scope.spawn(|| {
                    done.fetch_add(1, Ordering::SeqCst);
                });
                panic!("the scope's closure fails");
            })
        }));
        assert!(outcome.is_err());
        assert_eq!(done.load(Ordering::SeqCst), 2);
    });
}

/// A worker that panics is reported by its own handle, and the scope still returns normally.
#[test]
fn loom_a_panicking_worker_is_reported_by_its_handle() {
    loom::model(|| {
        let ran = AtomicUsize::new(0);
        let reported = ladde::sync::thread::scope(|scope| {
            let failing = scope.spawn(|| {
                ran.fetch_add(1, Ordering::SeqCst);
                panic!("the worker fails");
            });
            let fine = scope.spawn(|| ran.fetch_add(1, Ordering::SeqCst));
            (failing.join().is_err(), fine.join().is_ok())
        });
        assert_eq!(reported, (true, true));
        assert_eq!(ran.load(Ordering::SeqCst), 2);
    });
}
