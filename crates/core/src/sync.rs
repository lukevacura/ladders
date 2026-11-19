//! The one door for atomics, interior mutability, `Arc`, and thread spawning.
//!
//! Everything that needs them imports from here, never straight from `core`, `alloc`,
//! `std`, `loom`, or `shuttle`, so a `--cfg loom` or `--cfg shuttle` build swaps in a
//! model checker underneath the same call sites.

// Test-only per-thread counters (batch's instrumentation) and the route harness's force scope
// need `thread_local!`; the crate is `no_std`, so test and oracle builds reach `std` from here,
// with or without the `parallel` feature.
#[cfg(any(test, feature = "oracle"))]
extern crate std;
#[cfg(any(test, feature = "oracle"))]
pub use std::thread_local;

#[cfg(loom)]
pub use loom::cell::UnsafeCell;
#[cfg(loom)]
pub use loom::sync::Arc;
#[cfg(loom)]
pub use loom::sync::Mutex;
#[cfg(loom)]
pub use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Loom's threads, plus (with `parallel`) the scoped threads loom 0.7 lacks, so `par` runs the
/// same code under loom as in a plain build.
#[cfg(loom)]
pub mod thread {
    pub use loom::thread::*;

    #[cfg(feature = "parallel")]
    pub use self::scoped::{Scope, ScopedJoinHandle, scope};

    #[cfg(feature = "parallel")]
    mod scoped {
        use alloc::boxed::Box;
        use alloc::sync::Arc;
        use alloc::vec::Vec;
        use core::marker::PhantomData;
        use std::any::Any;
        use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
        use std::sync::{Mutex, MutexGuard, PoisonError};

        type Outcome<T> = Result<T, Box<dyn Any + Send>>;
        type Handles = Mutex<Vec<Option<loom::thread::JoinHandle<()>>>>;

        /// A poisoned lock still holds valid data here; refusing it would skip the joins.
        fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
            mutex.lock().unwrap_or_else(PoisonError::into_inner)
        }

        /// Threads that may borrow from the enclosing function; [`scope`] joins them all.
        ///
        /// `spawn` is called only from the thread that called `scope`. The `std` locks and
        /// `Arc`s in this module are touched by a job only before it ends, and by the scoping
        /// thread only after loom's `join` of that job, so they never race and loom needs no
        /// model of them. A job's panic is returned by its handle's `join`; the panic of a job
        /// never joined is dropped.
        pub struct Scope<'scope, 'env: 'scope> {
            handles: Handles,
            scope: PhantomData<&'scope mut &'scope ()>,
            env: PhantomData<&'env mut &'env ()>,
        }

        pub struct ScopedJoinHandle<'scope, T> {
            handles: &'scope Handles,
            index: usize,
            outcome: Arc<Mutex<Option<Outcome<T>>>>,
        }

        impl<'scope, 'env> Scope<'scope, 'env> {
            pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
            where
                F: FnOnce() -> T + Send + 'scope,
                T: Send + 'scope,
            {
                let outcome = Arc::new(Mutex::new(None));
                let theirs = Arc::clone(&outcome);
                let job: Box<dyn FnOnce() + Send + 'scope> = Box::new(move || {
                    let result = catch_unwind(AssertUnwindSafe(f));
                    *locked(&theirs) = Some(result);
                });
                // SAFETY: loom's `spawn` wants `'static`, but the job borrows for `'scope`. The
                // job is joined before `scope` returns (`join`, or the sweep at the end of
                // `scope`, also on a panic in the scope's closure), and `'scope` ends no
                // earlier, so nothing the job borrows is gone while it runs.
                let job: Box<dyn FnOnce() + Send + 'static> = unsafe { core::mem::transmute(job) };
                let handle = loom::thread::spawn(job);
                let mut handles = locked(&self.handles);
                handles.push(Some(handle));
                ScopedJoinHandle {
                    handles: &self.handles,
                    index: handles.len() - 1,
                    outcome,
                }
            }
        }

        impl<T> ScopedJoinHandle<'_, T> {
            pub fn join(self) -> Outcome<T> {
                let handle = locked(self.handles)[self.index].take();
                if let Some(handle) = handle {
                    handle.join().expect("scoped job catches its own panic");
                }
                locked(&self.outcome)
                    .take()
                    .expect("a joined scoped job leaves its outcome")
            }
        }

        pub fn scope<'env, F, T>(f: F) -> T
        where
            F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
        {
            let scope = Scope {
                handles: Mutex::new(Vec::new()),
                scope: PhantomData,
                env: PhantomData,
            };
            let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));
            // Join every job before anything unwinds: a job still running would outlive its borrows.
            let pending: Vec<_> = locked(&scope.handles).drain(..).flatten().collect();
            for handle in pending {
                // A job catches its own panic, so this join never returns an error.
                let _ = handle.join();
            }
            result.unwrap_or_else(|payload| resume_unwind(payload))
        }
    }
}

#[cfg(shuttle)]
pub use shuttle::sync::Arc;
#[cfg(shuttle)]
pub use shuttle::sync::Mutex;
#[cfg(shuttle)]
pub use shuttle::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(shuttle)]
pub use shuttle::thread;

#[cfg(not(any(loom, shuttle)))]
pub use alloc::sync::Arc;
#[cfg(not(any(loom, shuttle)))]
pub use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(all(any(feature = "parallel", feature = "oracle"), not(any(loom, shuttle))))]
pub use std::sync::{Mutex, MutexGuard};

// Shuttle has no `UnsafeCell` of its own (its model runs real threads over real
// memory), so it shares the plain wrapper below with ordinary builds. Only loom's
// cell is special: it tracks each access to simulate weak-memory visibility.
#[cfg(not(loom))]
pub use self::plain_cell::UnsafeCell;

#[cfg(not(loom))]
mod plain_cell {
    /// Same size and alignment as `T`: `#[repr(transparent)]` all the way down.
    #[repr(transparent)]
    pub struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

    impl<T> UnsafeCell<T> {
        pub const fn new(value: T) -> Self {
            Self(core::cell::UnsafeCell::new(value))
        }

        pub fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
            f(self.0.get())
        }

        pub fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            f(self.0.get())
        }
    }
}

/// Scoped threads and the core count, for `par` (feature `parallel`). Shuttle brings its
/// own `thread::scope` (through `thread` above); loom 0.7 has none, so the loom `thread` module
/// above builds one on its `spawn`.
#[cfg(all(feature = "parallel", not(any(loom, shuttle))))]
pub mod thread {
    pub use std::thread::{Scope, ScopedJoinHandle, scope};
}

/// How many threads the machine offers: at least 1. Model checkers run one real core's
/// worth, so they report 1.
#[cfg(feature = "parallel")]
pub(crate) fn available_threads() -> usize {
    #[cfg(any(loom, shuttle))]
    {
        1
    }
    #[cfg(not(any(loom, shuttle)))]
    {
        std::thread::available_parallelism().map_or(1, core::num::NonZeroUsize::get)
    }
}
