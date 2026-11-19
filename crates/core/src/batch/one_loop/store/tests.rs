//! The lane store's rule: a handoff only on an equal row, and the refused handoff leaves the cell as
//! it was. The rule is tested on the generic cells with values that can be told apart; the lane
//! store is tested for its range check and the no-handoff force. Under `loom` and `shuttle`, the
//! cells race under every schedule.

#[cfg(not(any(loom, shuttle)))]
mod plain {
    use crate::batch::one_loop::store::{Carry, Cells, LaneStore};
    use crate::route::{RouteForce, forced_route};

    #[test]
    fn a_matching_row_returns_what_was_put_and_empties_the_cell() {
        let cells = Cells::<u32>::try_new(3).unwrap();
        cells.put(1, 77, 40);
        assert_eq!(cells.take(1, 40), Some(77));
        assert_eq!(cells.take(1, 40), None, "taken once");
        assert_eq!(cells.take(0, 40), None, "other cells are untouched");
    }

    #[test]
    fn a_row_off_by_one_misses_and_leaves_the_cell_for_a_later_match() {
        let cells = Cells::<u32>::try_new(1).unwrap();
        cells.put(0, 5, 40);
        assert_eq!(cells.take(0, 39), None);
        assert_eq!(cells.take(0, 41), None);
        assert_eq!(cells.take(0, 0), None);
        assert_eq!(cells.take(0, 40), Some(5), "still there");
    }

    #[test]
    fn a_put_replaces_what_was_there() {
        let cells = Cells::<u32>::try_new(1).unwrap();
        cells.put(0, 1, 10);
        cells.put(0, 2, 20);
        assert_eq!(cells.take(0, 10), None);
        assert_eq!(cells.take(0, 20), Some(2));
    }

    #[test]
    fn a_range_past_the_store_drops_on_put_and_misses_on_take() {
        let cells = Cells::<u32>::try_new(2).unwrap();
        cells.put(2, 9, 0);
        cells.put(usize::MAX, 9, 0);
        assert_eq!(cells.take(2, 0), None);
        assert_eq!(cells.take(usize::MAX, 0), None);
        assert_eq!(cells.take(0, 0), None);
        assert_eq!(cells.take(1, 0), None);
        let none = Cells::<u32>::try_new(0).unwrap();
        none.put(0, 1, 0);
        assert_eq!(none.take(0, 0), None);
    }

    #[test]
    fn lanes_hand_down_on_an_equal_row_only() {
        let _route = forced_route(RouteForce::default());
        let store = LaneStore::try_new(2).unwrap();
        store.put(0, Carry::try_new(8).unwrap(), 64);
        assert!(store.take(0, 63).is_none());
        assert!(store.take(0, 65).is_none());
        assert!(store.take(1, 64).is_none());
        assert!(store.take(0, 64).is_some());
        assert!(store.take(0, 64).is_none(), "taken once");
        store.put(5, Carry::try_new(8).unwrap(), 64);
        assert!(store.take(5, 64).is_none(), "past the store");
    }

    #[test]
    fn the_no_handoff_force_makes_every_take_miss() {
        let store = LaneStore::try_new(1).unwrap();
        {
            let _route = forced_route(RouteForce {
                one_loop_no_handoff: true,
                ..RouteForce::default()
            });
            store.put(0, Carry::try_new(8).unwrap(), 8);
            assert!(store.take(0, 8).is_none());
        }
        let _route = forced_route(RouteForce::default());
        assert!(store.take(0, 8).is_some(), "the miss left the cell alone");
    }

    #[cfg(any(feature = "parallel", feature = "oracle"))]
    #[test]
    fn a_poisoned_cell_is_read_as_it_stands() {
        let cells = Cells::<u32>::try_new(1).unwrap();
        cells.put(0, 3, 12);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cells.with(0, |_| panic!("poison the lock"));
        }));
        assert!(poisoned.is_err());
        assert_eq!(cells.take(0, 11), None);
        assert_eq!(cells.take(0, 12), Some(3));
        cells.put(0, 4, 13);
        assert_eq!(cells.take(0, 13), Some(4));
    }
}

/// Tasks that put and take on shared cells: a take returns only what was put at the row it asked
/// for, and every put is taken at most once.
#[cfg(any(loom, shuttle))]
fn race(tasks: usize, rounds: usize) {
    use crate::batch::one_loop::store::Cells;
    use crate::sync::{Arc, Mutex, thread};
    use alloc::vec::Vec;
    // A value is (through_row, unique id); the row of range `r` in round `i` is `r + i`.
    let cells = Arc::new(Cells::<(usize, usize)>::try_new(2).unwrap());
    let taken = Arc::new(Mutex::new(Vec::new()));
    let put = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<_> = (0..tasks)
        .map(|t| {
            let (cells, taken, put) = (Arc::clone(&cells), Arc::clone(&taken), Arc::clone(&put));
            thread::spawn(move || {
                for i in 0..rounds {
                    for r in 0..2 {
                        let id = (t * rounds + i) * 2 + r;
                        // Recorded before the value can be seen by a taker.
                        put.lock().unwrap().push(id);
                        cells.put(r, (r + i, id), r + i);
                        let want = r + (i + t) % rounds;
                        if let Some((row, got)) = cells.take(r, want) {
                            assert_eq!(row, want, "lanes of another row handed down");
                            taken.lock().unwrap().push(got);
                        }
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let mut seen = taken.lock().unwrap().clone();
    for r in 0..2 {
        for row in 0..tasks * rounds + 2 {
            if let Some((_, id)) = cells.take(r, row) {
                seen.push(id);
            }
        }
    }
    let before = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), before, "a put was handed out twice");
    let put = put.lock().unwrap();
    assert!(seen.iter().all(|id| put.contains(id)), "a value nobody put");
}

#[cfg(shuttle)]
#[test]
fn shuttle_takes_return_only_the_asked_row_and_each_put_once() {
    shuttle::check_random_with_seed(|| race(3, 3), testkit::generators::TEST_SEED, 500);
}

#[cfg(loom)]
#[test]
fn loom_takes_return_only_the_asked_row_and_each_put_once() {
    loom::model(|| race(2, 1));
}
