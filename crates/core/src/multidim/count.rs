//! How many orders between means went to the exact distances, and how many cells the estimate
//! screen settled without them, for tests: process-wide counts
//! that only tests and the `oracle` feature keep (and not under a model checker).

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
mod on {
    use crate::sync::{AtomicUsize, Ordering::Relaxed};

    static EXACT: AtomicUsize = AtomicUsize::new(0);
    static FALLBACKS: AtomicUsize = AtomicUsize::new(0);
    static SCREENED: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn exact_decision() {
        EXACT.fetch_add(1, Relaxed);
    }

    pub(crate) fn cells_screened(count: usize) {
        SCREENED.fetch_add(count, Relaxed);
    }

    /// Cells of a multi-dimensional walk settled by the estimate screen so far, in this process.
    pub fn screened_cells() -> usize {
        SCREENED.load(Relaxed)
    }

    pub(crate) fn budget_fallback() {
        FALLBACKS.fetch_add(1, Relaxed);
    }

    /// Orders decided by the exact distances so far, in this process.
    pub fn exact_decisions() -> usize {
        EXACT.load(Relaxed)
    }

    /// Sums of nested roots that ended at the precision budget as ties so far.
    pub fn budget_fallbacks() -> usize {
        FALLBACKS.load(Relaxed)
    }
}

#[cfg(not(all(any(test, feature = "oracle"), not(any(loom, shuttle)))))]
mod on {
    pub(crate) fn exact_decision() {}
    pub(crate) fn budget_fallback() {}
    pub(crate) fn cells_screened(_count: usize) {}
}

pub(crate) use on::{budget_fallback, cells_screened, exact_decision};
#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
pub use on::{budget_fallbacks, exact_decisions, screened_cells};
