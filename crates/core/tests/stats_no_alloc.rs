//! `CountingAllocator` only counts once it is the process's global allocator, so this runs as
//! its own test binary rather than alongside `stats`'s unit tests.

use core::num::NonZeroUsize;
use ladde::stats::WindowStats;
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn push_after_new_allocates_nothing() {
    let mut stats = WindowStats::new(NonZeroUsize::new(64).unwrap());
    let (_, count) = allocations_in(|| {
        for tick in 0..10_000i64 {
            stats.push(tick, true);
        }
    });
    assert_eq!(count, 0);
}
