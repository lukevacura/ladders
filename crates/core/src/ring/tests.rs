use super::MirroredRing;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Op {
    Push(i64),
    Update(i64),
    Stamp(i64),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        any::<i64>().prop_map(Op::Push),
        any::<i64>().prop_map(Op::Update),
        any::<i64>().prop_map(Op::Stamp),
    ]
}

/// Every slot's two copies hold the same value, live or not.
fn assert_mirrors_equal(ring: &MirroredRing<i64>) {
    for slot in 0..ring.capacity {
        assert_eq!(
            ring.storage[slot],
            ring.storage[slot + ring.capacity],
            "mirrored copies diverged at slot {slot}"
        );
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn matches_vecdeque_model(
        capacity in 1usize..=64,
        ops in proptest::collection::vec(op_strategy(), 0..256),
    ) {
        let mut ring = MirroredRing::new(NonZeroUsize::new(capacity).unwrap(), 0i64);
        let mut model: VecDeque<i64> = VecDeque::new();
        prop_assert_eq!(ring.len(), 0);
        for op in &ops {
            match *op {
                Op::Push(value) => {
                    ring.push(value);
                    model.push_back(value);
                    if model.len() > capacity {
                        model.pop_front();
                    }
                }
                Op::Update(delta) => {
                    ring.update_live(|v| *v = v.wrapping_add(delta));
                    for v in model.iter_mut() {
                        *v = v.wrapping_add(delta);
                    }
                }
                Op::Stamp(seed) => {
                    // Each call overwrites with the next counter value, so the
                    // resulting slice reveals the order `f` was applied in.
                    let mut next_stamp = seed;
                    ring.update_live(|v| {
                        *v = next_stamp;
                        next_stamp = next_stamp.wrapping_add(1);
                    });
                    let mut next_stamp = seed;
                    for v in model.iter_mut() {
                        *v = next_stamp;
                        next_stamp = next_stamp.wrapping_add(1);
                    }
                }
            }
            let expected: Vec<i64> = model.iter().copied().collect();
            prop_assert_eq!(ring.len(), expected.len());
            prop_assert_eq!(ring.as_slice(), expected.as_slice());
            assert_mirrors_equal(&ring);
        }
    }
}

#[test]
fn storage_never_reallocates() {
    let mut ring = MirroredRing::new(NonZeroUsize::new(7).unwrap(), 0u32);
    let address = ring.storage.as_ptr();
    let storage_len = ring.storage.len();
    for value in 0..100 {
        ring.push(value);
        assert_eq!(ring.storage.as_ptr(), address);
        assert_eq!(ring.storage.len(), storage_len);
    }
    assert_eq!(storage_len, 2 * ring.capacity());
}
