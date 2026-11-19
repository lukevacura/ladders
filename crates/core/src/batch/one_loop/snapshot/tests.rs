//! The snapshot's refusal to give a floor to what the floors say nothing about, its blocks as the
//! ladder reads them, and its never being above the live table.

use super::{FloorSnapshot, FloorSources};
use crate::batch::survivor::SharedFloors;
use crate::row::RowAcc;
use crate::stats::{Window, WindowKind};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

fn window(kind: WindowKind) -> Window {
    Window { sum: 0, kind }
}

fn valid() -> WindowKind {
    WindowKind::Valid {
        variance_numerator: 1,
        inv_sqrt: 1.0,
    }
}

/// Rows `0..n`, each begun on a window of `kind(row)`.
fn accs(n: usize, kind: impl Fn(usize) -> WindowKind) -> Vec<RowAcc> {
    (0..n)
        .map(|row| {
            let mut acc = RowAcc::new();
            acc.begin(&window(kind(row)), NonZeroUsize::MIN, n);
            acc
        })
        .collect()
}

#[test]
fn a_snapshot_of_floors_nothing_holds_reads_nan_everywhere() {
    let sources = FloorSources {
        accs: &[],
        first_row: 0,
        right: None,
        shared: None,
    };
    let mut snapshot = FloorSnapshot::try_new(4, 6).unwrap();
    snapshot.refresh(&sources, 0..32, 0..32);
    assert!(snapshot.rows().lowest(0..32).is_nan());
    // No column partial and no table: a column's floor is open (-inf), which every bound is above.
    assert_eq!(
        snapshot.columns().expect("column blocks").lowest(0..32),
        f64::NEG_INFINITY
    );
    // A range outside the blocks held reads NaN.
    assert!(snapshot.rows().lowest(64..72).is_nan());
    assert!(
        snapshot
            .columns()
            .expect("column blocks")
            .lowest(1000..1008)
            .is_nan()
    );
}

/// The blocks start where the refresh says (rows on their first row, columns on the multiple of
/// eight at or below the first column) and each is the lowest floor of its eight windows.
#[test]
fn a_blocks_floor_is_the_lowest_of_its_windows_and_starts_where_the_refresh_says() {
    let rows = accs(32, |_| valid());
    let table = SharedFloors::try_new(64, true).unwrap();
    for w in 0..64 {
        table.raise_row(w % 32, 0.2 + (w % 7) as f64 * 0.1);
        table.raise_column(w, 0.1 + (w % 9) as f64 * 0.05);
    }
    let sources = FloorSources {
        accs: &rows,
        first_row: 0,
        right: None,
        shared: Some(&table),
    };
    let mut snapshot = FloorSnapshot::try_new(2, 6).unwrap();
    snapshot.refresh(&sources, 16..32, 13..45);
    assert_eq!(snapshot.rows().first, 16);
    assert_eq!(snapshot.columns().expect("column blocks").first, 8);
    for w in 16..32 {
        assert!(snapshot.rows().lowest(w..w + 1) <= table.row(w, f64::NEG_INFINITY));
    }
    for w in 13..45 {
        assert!(
            snapshot.columns().expect("column blocks").lowest(w..w + 1)
                <= table.column(w, f64::NEG_INFINITY)
        );
    }
    let eight = |first: usize| {
        (first..first + 8)
            .map(|w| table.column(w, f64::NEG_INFINITY))
            .fold(f64::INFINITY, f64::min)
    };
    assert_eq!(
        snapshot.columns().expect("column blocks").lowest(16..24),
        eight(16)
    );
    // The range below the first block is outside it.
    assert!(
        snapshot
            .columns()
            .expect("column blocks")
            .lowest(0..8)
            .is_nan()
    );
}

/// I4: a block's floor in the snapshot is the lowest of its rows' and columns' floors, so it is
/// never above any window's floor, and never above what the table holds later, since the table
/// only rises.
#[test]
fn a_snapshot_is_never_above_a_floor_the_table_holds() {
    let rows = accs(16, |_| valid());
    let table = SharedFloors::try_new(16, true).unwrap();
    for window in 0..16 {
        let floor = 0.5 + (window * 7 % 5) as f64 * 0.1;
        table.raise_row(window, floor);
        table.raise_column(window, floor + 0.05);
    }
    let sources = FloorSources {
        accs: &rows,
        first_row: 0,
        right: None,
        shared: Some(&table),
    };
    let mut snapshot = FloorSnapshot::try_new(2, 2).unwrap();
    snapshot.refresh(&sources, 0..16, 0..16);
    for window in 0..16 {
        assert!(snapshot.rows().lowest(window..window + 1) <= table.row(window, f64::NEG_INFINITY));
        assert!(
            snapshot
                .columns()
                .expect("column blocks")
                .lowest(window..window + 1)
                <= table.column(window, f64::NEG_INFINITY)
        );
    }
    snapshot.assert_not_above(&sources, &(0..16), &(0..16));
    // The table rises after the snapshot: the snapshot is still not above it.
    for window in 0..16 {
        table.raise_row(window, 0.95);
        table.raise_column(window, 0.95);
    }
    snapshot.assert_not_above(&sources, &(0..16), &(0..16));
}

/// The assertion fires on a snapshot above the live table (a snapshot taken after raises the table
/// does not hold, or a block's floor taken as the highest of its rows).
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "above the live floor")]
fn a_snapshot_above_the_live_floor_is_caught() {
    let rows = accs(8, |_| valid());
    let table = SharedFloors::try_new(8, true).unwrap();
    let sources = FloorSources {
        accs: &rows,
        first_row: 0,
        right: None,
        shared: Some(&table),
    };
    let mut snapshot = FloorSnapshot::try_new(1, 1).unwrap();
    snapshot.refresh(&sources, 0..8, 0..8);
    snapshot.rows_shared[0] = 0.9;
    snapshot.assert_not_above(&sources, &(0..8), &(0..8));
}

/// I5, the snapshot's layer: a block of rows holding a constant row reads `NaN` however high the
/// other rows' floors are, so no bound is below it and the block is never skipped.
#[test]
fn a_block_holding_a_constant_row_reads_nan() {
    let rows = accs(16, |row| {
        if row == 3 {
            WindowKind::Constant
        } else {
            valid()
        }
    });
    let table = SharedFloors::try_new(16, true).unwrap();
    for window in 0..16 {
        table.raise_row(window, 0.9);
        table.raise_column(window, 0.9);
    }
    let sources = FloorSources {
        accs: &rows,
        first_row: 0,
        right: None,
        shared: Some(&table),
    };
    let mut snapshot = FloorSnapshot::try_new(2, 2).unwrap();
    snapshot.refresh(&sources, 0..16, 0..16);
    assert!(snapshot.rows().lowest(0..8).is_nan());
    assert!(snapshot.rows().lowest(3..4).is_nan());
    // The block without a constant row holds a floor to skip below.
    assert!(snapshot.rows().lowest(8..16) >= 0.0);
    assert!(!snapshot.rows().lowest(8..16).is_nan());
}

/// A run's global floor is a floor of every block that has one: the higher of it and the block's
/// own, and still `NaN` for a block holding a constant row, as the one-loop route never skips such a
/// block by the global floor.
#[test]
fn the_global_floor_lifts_every_block_but_a_constant_one() {
    let rows = accs(16, |row| {
        if row == 3 {
            WindowKind::Constant
        } else {
            valid()
        }
    });
    let table = SharedFloors::try_new(16, true).unwrap().with_global(0.5);
    for window in 8..16 {
        table.raise_row(window, 0.7);
    }
    let sources = FloorSources {
        accs: &rows,
        first_row: 0,
        right: None,
        shared: Some(&table),
    };
    let mut snapshot = FloorSnapshot::try_new(2, 2).unwrap();
    snapshot.refresh(&sources, 0..16, 0..16);
    assert!(snapshot.rows().lowest(0..8).is_nan());
    // Own floors above the global one stay; open ones read the global one.
    assert_eq!(snapshot.rows().lowest(8..16), 0.7);
    assert_eq!(snapshot.columns().expect("column blocks").lowest(0..8), 0.5);
    table.raise_global(0.9);
    snapshot.refresh(&sources, 0..16, 0..16);
    assert_eq!(snapshot.rows().lowest(8..16), 0.9);
    assert_eq!(
        snapshot.columns().expect("column blocks").lowest(8..16),
        0.9
    );
}

/// A one-sided run asks for no column blocks: the snapshot reads no column and hands the ladder
/// none (its columns hold no floor a pair must beat), while its rows read as before.
#[test]
fn a_snapshot_without_column_blocks_hands_the_ladder_no_column_floors() {
    let accs = accs(8, |_| valid());
    let table = SharedFloors::try_new(32, false).unwrap();
    table.raise_row(3, 0.25);
    let sources = FloorSources {
        accs: &accs,
        first_row: 0,
        right: None,
        shared: Some(&table),
    };
    let mut snapshot = FloorSnapshot::try_new(1, 0).unwrap();
    snapshot.refresh(&sources, 0..8, 0..32);
    assert!(snapshot.columns().is_none());
    assert!(snapshot.rows().lowest(0..8) <= 0.25);
    snapshot.assert_not_above(&sources, &(0..8), &(0..32));
}
