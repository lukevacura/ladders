#![cfg(loom)]
//! The survivor filter's shared floors under every interleaving and every reordering the relaxed
//! atomics permit: concurrent raises leave the highest finite floor, a raise never lowers one, a
//! non-finite floor is never stored, and a racing reader sees only floors some raise stored, never
//! one lower than it saw before, and never below its own.

use ladde::batch::SharedFloors;
use ladde::sync::{Arc, thread};

const OPEN: f64 = f64::NEG_INFINITY;

/// Two raisers of one row race, one reading the row's floor (with its own as the lower bound)
/// before it raises and one closing a tile: whichever lands last, the higher floor stays, and the
/// reader's floor is never below its own.
#[test]
fn loom_two_raisers_leave_the_higher_floor() {
    for (gate, close) in [(0.75, 0.5), (0.5, 0.75)] {
        loom::model(move || {
            let table = Arc::new(SharedFloors::try_new(1, false).unwrap());
            let closer = {
                let table = Arc::clone(&table);
                thread::spawn(move || {
                    table.raise_row(0, close);
                })
            };
            let gater = {
                let table = Arc::clone(&table);
                thread::spawn(move || {
                    let floor = table.row(0, gate);
                    table.raise_row(0, gate);
                    assert!(
                        floor >= gate && (floor == gate || floor == close),
                        "{floor}"
                    );
                })
            };
            closer.join().unwrap();
            gater.join().unwrap();
            assert_eq!(table.row(0, OPEN), 0.75);
        });
    }
}

/// A raiser of a column races a raiser of `∞` and a reader: the reader sees the open floor or the
/// raised one, never `∞`, the second of its reads is never lower than the first, and its own
/// higher floor always wins.
#[test]
fn loom_a_reader_sees_only_stored_floors_in_order() {
    loom::model(|| {
        let table = Arc::new(SharedFloors::try_new(2, true).unwrap());
        let raiser = {
            let table = Arc::clone(&table);
            thread::spawn(move || {
                table.raise_column(1, 0.5);
            })
        };
        let settled = {
            let table = Arc::clone(&table);
            thread::spawn(move || {
                table.raise_column(1, f64::INFINITY);
                table.raise_column(1, f64::NAN);
            })
        };
        let first = table.column(1, OPEN);
        let second = table.column(1, OPEN);
        assert!(first == OPEN || first == 0.5, "{first}");
        assert!(second >= first, "{first} then {second}");
        assert_eq!(table.column(1, 0.9), 0.9);
        raiser.join().unwrap();
        settled.join().unwrap();
        assert_eq!(table.column(1, OPEN), 0.5);
        assert_eq!(table.column(0, OPEN), OPEN, "another column is untouched");
    });
}
