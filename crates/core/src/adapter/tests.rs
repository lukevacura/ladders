use super::{CapacityError, Full, published};
use core::mem::size_of;
use proptest::prelude::*;
use testkit::generators::seeded;

#[test]
fn empty_until_pushed() {
    let (_publisher, reader) = published::<u64>(4).unwrap();
    assert_eq!(reader.snapshot(), &[] as &[u64]);
}

#[test]
fn snapshot_is_the_published_prefix() {
    let (mut publisher, reader) = published::<u64>(4).unwrap();
    publisher.push(10).unwrap();
    assert_eq!(reader.snapshot(), &[10]);
    publisher.push(20).unwrap();
    assert_eq!(reader.snapshot(), &[10, 20]);
}

#[test]
fn full_once_capacity_is_written() {
    let (mut publisher, _reader) = published::<u64>(2).unwrap();
    assert_eq!(publisher.push(1), Ok(()));
    assert_eq!(publisher.push(2), Ok(()));
    assert_eq!(publisher.push(3), Err(Full));
}

#[test]
fn zero_capacity_is_full_immediately() {
    let (mut publisher, reader) = published::<u64>(0).unwrap();
    assert_eq!(publisher.push(1), Err(Full));
    assert_eq!(reader.snapshot(), &[] as &[u64]);
}

#[test]
fn cloned_reader_sees_the_same_prefix() {
    let (mut publisher, reader) = published::<u64>(4).unwrap();
    publisher.push(1).unwrap();
    let cloned = reader.clone();
    publisher.push(2).unwrap();
    assert_eq!(reader.snapshot(), cloned.snapshot());
    assert_eq!(cloned.snapshot(), &[1, 2]);
}

fn builds_or_refuses<T: Copy>(capacity: usize) -> bool {
    match published::<T>(capacity) {
        Ok((_publisher, reader)) => {
            assert_eq!(reader.snapshot().len(), 0);
            true
        }
        Err(error) => {
            assert_eq!(error, CapacityError);
            false
        }
    }
}

#[test]
fn a_capacity_beyond_the_address_space_is_an_error() {
    assert_eq!(published::<u8>(usize::MAX).err(), Some(CapacityError));
    assert_eq!(published::<u64>(usize::MAX).err(), Some(CapacityError));
    assert_eq!(published::<[u8; 3]>(usize::MAX).err(), Some(CapacityError));
}

#[test]
fn a_zero_sized_element_has_any_capacity() {
    let (mut publisher, reader) = published::<()>(usize::MAX).unwrap();
    assert_eq!(publisher.push(()), Ok(()));
    assert_eq!(reader.snapshot().len(), 1);
}

fn edge_capacities<T: Copy>() -> [usize; 7] {
    let limit = isize::MAX as usize / size_of::<T>();
    [0, 1, usize::MAX, limit - 1, limit, limit + 1, limit + 2]
}

fn over_size_limit<T>(capacity: usize) -> bool {
    capacity
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
}

fn edge_capacities_of<T: Copy>() {
    for capacity in edge_capacities::<T>() {
        // Miri cannot model an allocator refusing a representable request, so under it only the
        // capacities the size arithmetic rejects (and small ones) are tried.
        if cfg!(miri) && capacity > 1 && !over_size_limit::<T>(capacity) {
            continue;
        }
        let built = builds_or_refuses::<T>(capacity);
        assert!(built || capacity > 1);
        assert!(!over_size_limit::<T>(capacity) || !built);
    }
}

#[test]
fn edge_capacities_never_panic() {
    edge_capacities_of::<u8>();
    edge_capacities_of::<u64>();
    edge_capacities_of::<(u64, u64, u64)>();
    assert!(builds_or_refuses::<u64>(0));
    assert!(builds_or_refuses::<u64>(1));
    assert!(!builds_or_refuses::<u64>(usize::MAX));
    assert!(!builds_or_refuses::<u64>(isize::MAX as usize / 8 + 1));
}

proptest! {
    #![proptest_config(seeded(64))]

    #[test]
    fn small_capacities_hold_exactly_that_many(capacity in 0usize..200) {
        let (mut publisher, reader) = published::<u32>(capacity).unwrap();
        for value in 0..capacity as u32 {
            prop_assert_eq!(publisher.push(value), Ok(()));
        }
        prop_assert_eq!(publisher.push(0), Err(Full));
        prop_assert_eq!(reader.snapshot().len(), capacity);
    }

    #[test]
    fn huge_capacities_are_an_error(
        capacity in prop_oneof![
            Just(usize::MAX),
            (0usize..1000).prop_map(|d| usize::MAX - d),
            (0usize..1000).prop_map(|d| isize::MAX as usize / 8 - 500 + d),
            (0usize..1000).prop_map(|d| isize::MAX as usize / 2 - 500 + d),
        ],
    ) {
        // Miri cannot model an allocator refusing a representable request.
        #[cfg(miri)]
        prop_assume!(over_size_limit::<u64>(capacity));
        prop_assert!(!builds_or_refuses::<u64>(capacity));
    }
}
