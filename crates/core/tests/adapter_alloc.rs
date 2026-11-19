//! A normal capacity is allocated once, in full, and pushing allocates nothing more (its own
//! test binary: the counting allocator is global, and counts this thread only).
#![cfg(not(any(loom, shuttle)))]

use core::mem::size_of;
use ladde::adapter::published;
use testkit::alloc_count::{CountingAllocator, large_requests_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const CAPACITY: usize = 100_000;

#[test]
fn a_capacity_is_one_allocation_of_exactly_that_many_slots() {
    let bytes = CAPACITY * size_of::<u64>();
    let (_, requests) = large_requests_in(bytes / 2, || {
        let (mut publisher, reader) = published::<u64>(CAPACITY).unwrap();
        for value in 0..CAPACITY as u64 {
            publisher.push(value).unwrap();
        }
        assert_eq!(reader.snapshot().len(), CAPACITY);
    });
    let sizes: Vec<_> = requests.iter().map(|r| (r.bytes, r.realloc)).collect();
    assert_eq!(sizes, [(bytes, false)]);
}
