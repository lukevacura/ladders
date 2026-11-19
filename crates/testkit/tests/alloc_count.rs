//! `CountingAllocator` only counts once it is the process's global allocator, so these run
//! as their own test binary rather than alongside `alloc_count`'s unit tests.

use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn counts_vec_allocations() {
    let (_, count) = allocations_in(|| {
        let vec = Vec::<u8>::with_capacity(10);
        drop(vec);
    });
    assert_eq!(count, 1);
}

#[test]
fn arithmetic_is_zero() {
    let (total, count) = allocations_in(|| (1u64..1000).sum::<u64>());
    assert_eq!(total, 499_500);
    assert_eq!(count, 0);
}

#[test]
fn realloc_counts() {
    let mut vec = Vec::<u8>::with_capacity(4);
    vec.extend_from_slice(&[0, 1, 2, 3]);
    let (_, count) = allocations_in(|| vec.push(4));
    assert_eq!(count, 1);
}

#[test]
fn other_threads_not_counted() {
    let handle = std::thread::spawn(|| {
        let allocated = Vec::<u8>::with_capacity(64);
        allocated.len()
    });
    let (_, count) = allocations_in(|| handle.join().unwrap());
    assert_eq!(count, 0);
}

#[test]
fn nested_counts_are_separate() {
    let (inner_count, outer_count) = allocations_in(|| {
        let _ = Vec::<u8>::with_capacity(1);
        let (_, inner) = allocations_in(|| {
            let _ = Vec::<u8>::with_capacity(1);
        });
        let _ = Vec::<u8>::with_capacity(1);
        inner
    });
    assert_eq!(inner_count, 1);
    assert_eq!(outer_count, 3);
}
