#![cfg(shuttle)]
//! The survivor filter's shared floors under random schedules of three threads: every row ends at
//! the highest finite floor any thread raised it to, whatever the order; a raise never lowers a
//! floor; a non-finite floor is never stored; and each thread's reads of a cell never fall and
//! never return less than its own floor or a value nobody raised.

use ladde::batch::SharedFloors;
use ladde::sync::{Arc, thread};

const OPEN: f64 = f64::NEG_INFINITY;

/// Floors each thread raises rows `0..3` and columns `0..3` to, in its own order, with `∞` and
/// NaN among them.
const RAISES: [[f64; 4]; 3] = [
    [0.25, 0.75, f64::INFINITY, 0.5],
    [0.5, f64::NAN, 0.125, 0.875],
    [0.875, 0.25, 0.625, f64::INFINITY],
];

fn highest(cell: usize) -> f64 {
    RAISES
        .iter()
        .flat_map(|t| t.iter().copied().skip(cell % 2))
        .filter(|f| f.is_finite())
        .fold(OPEN, f64::max)
}

#[test]
fn shuttle_three_threads_leave_the_highest_finite_floor() {
    shuttle::check_random_with_seed(
        || {
            let table = Arc::new(SharedFloors::try_new(3, true).unwrap());
            let threads: Vec<_> = RAISES
                .into_iter()
                .map(|floors| {
                    let table = Arc::clone(&table);
                    thread::spawn(move || {
                        let mut seen = [OPEN; 3];
                        for cell in 0..3 {
                            for &floor in floors.iter().skip(cell % 2) {
                                // Rows alternate between a read before the raise and a plain raise.
                                if cell == 1 {
                                    let met = table.row(cell, floor);
                                    assert!(met.is_nan() || met >= floor, "{met} below {floor}");
                                }
                                table.raise_row(cell, floor);
                                table.raise_column(cell, floor);
                                let now = table.row(cell, OPEN);
                                assert!(now >= seen[cell], "a read fell");
                                assert!(now < f64::INFINITY, "{now}: never ∞ or NaN");
                                if floor.is_finite() {
                                    assert!(now >= floor, "{now} below its own raise {floor}");
                                }
                                assert_eq!(table.column(cell, 2.0), 2.0);
                                seen[cell] = now;
                            }
                        }
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            for cell in 0..3 {
                assert_eq!(table.row(cell, OPEN), highest(cell), "row {cell}");
                assert_eq!(table.column(cell, OPEN), highest(cell), "column {cell}");
            }
        },
        testkit::generators::TEST_SEED,
        500,
    );
}
