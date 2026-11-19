//! Deterministic work sharing over chunk indices (feature `parallel`).
//!
//! `for_each_chunk` runs a closure on every index `0..chunks` exactly once, on up to
//! `threads` scoped threads (the caller's thread is one of them). Threads claim the next
//! index from one shared counter, so which thread runs which chunk depends on the
//! schedule, but each result is placed by its chunk index: the returned `Vec` is the same
//! for any thread count and any schedule. No call site spawns threads itself.
//!
//! Each thread keeps its `(index, result)` pairs in a private `Vec` and hands them back
//! when it finishes, so there is no shared mutable output and no `unsafe`. The cost is one
//! small allocation per thread plus the result `Vec`.
//!
//! A panic in the closure propagates to the caller of `for_each_chunk` (the other threads
//! keep claiming and running every remaining chunk, since there is no cancel flag); the results are dropped.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::sync::{AtomicUsize, Ordering};

/// The machine's core count, or 1 if it cannot be determined.
pub fn threads_available() -> NonZeroUsize {
    NonZeroUsize::new(crate::sync::available_threads()).unwrap_or(NonZeroUsize::MIN)
}

/// Claim indices from `next` until `chunks` is exhausted, keeping each result with its index.
fn drain<R>(next: &AtomicUsize, chunks: usize, f: &impl Fn(usize) -> R) -> Vec<(usize, R)> {
    let mut done = Vec::new();
    loop {
        let index = next.fetch_add(1, Ordering::Relaxed);
        if index >= chunks {
            return done;
        }
        // A thread claims each index at most once, so it can never hold more than `chunks`.
        debug_assert!(done.len() < chunks, "claimed more chunks than exist");
        done.push((index, f(index)));
    }
}

/// Put every `(index, result)` pair at its index. Panics if an index is missing or repeated.
fn assemble<R>(parts: impl IntoIterator<Item = Vec<(usize, R)>>, chunks: usize) -> Vec<R> {
    let mut slots: Vec<Option<R>> = (0..chunks).map(|_| None).collect();
    for (index, result) in parts.into_iter().flatten() {
        assert!(
            slots[index].replace(result).is_none(),
            "chunk {index} ran twice"
        );
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| slot.unwrap_or_else(|| panic!("chunk {index} never ran")))
        .collect()
}

/// Where a list of independent tasks runs: a task's input is pure and its result is placed by
/// its index, so the result is the same on every executor.
pub trait Executor: Sync {
    /// How many tasks may run at once, which schedulers size their windows by.
    fn width(&self) -> NonZeroUsize;

    /// Runs `f(0)`, ..., `f(tasks - 1)` each exactly once and returns the results in index order.
    fn run<R: Send>(&self, tasks: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R>;
}

/// The CPU's threads: up to `threads` of them, through [`for_each_chunk`].
#[derive(Clone, Copy, Debug)]
pub struct Cpu {
    pub threads: NonZeroUsize,
}

impl Executor for Cpu {
    fn width(&self) -> NonZeroUsize {
        self.threads
    }

    fn run<R: Send>(&self, tasks: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
        for_each_chunk(self.threads, tasks, f)
    }
}

/// Run `f(0)`, ..., `f(chunks - 1)` each exactly once on up to `threads` threads and return
/// the results in index order. With one thread (or one chunk) nothing is spawned.
pub fn for_each_chunk<R: Send>(
    threads: NonZeroUsize,
    chunks: usize,
    f: impl Fn(usize) -> R + Sync,
) -> Vec<R> {
    use crate::sync::thread;

    let workers = threads.get().min(chunks);
    let next = AtomicUsize::new(0);
    if workers <= 1 {
        return assemble([drain(&next, chunks, &f)], chunks);
    }
    // A worker runs in its caller's route-force scope (test and oracle builds; nothing
    // elsewhere), so a forced route reaches every thread of the run it was forced for.
    let force = crate::route::Scope::current();
    let (next_ref, f_ref) = (&next, &f);
    let parts = thread::scope(|scope| {
        let handles: Vec<_> = (1..workers)
            .map(|_| {
                scope.spawn(move || {
                    let _scope = force.enter();
                    drain(next_ref, chunks, f_ref)
                })
            })
            .collect();
        let mut parts = Vec::with_capacity(workers);
        parts.push(drain(&next, chunks, &f));
        for handle in handles {
            match handle.join() {
                Ok(part) => parts.push(part),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        parts
    });
    assemble(parts, chunks)
}
