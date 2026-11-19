use super::{SharedFloors, share_closed};
use crate::kernel_neon::{Conversion, RowScalars};
use crate::row::{ChunkCols, ColumnAcc, RowAcc};
use crate::stats::{Window, WindowKind};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

const VALID: Window = Window {
    sum: 0,
    kind: WindowKind::Valid {
        variance_numerator: 100,
        inv_sqrt: 0.1,
    },
};

/// A row's accumulator after one lane at estimate `0.9` (for a valid row): its floor is just
/// below `0.9`.
fn row_at(window: Window) -> RowAcc {
    let mut acc = RowAcc::new();
    acc.begin(&window, NonZeroUsize::MIN, 10);
    let neighbor = |_: usize| VALID;
    acc.start_lane(
        90,
        &ChunkCols {
            enter: &[],
            leave: &[],
            sum: &[0],
            r: &[0.1],
            bias: &[0.0],
            first: 0,
            neighbor: &neighbor,
        },
        RowScalars {
            me: 0,
            ml: 0,
            si: 0,
            ri: 0.1,
            conversion: Conversion::FULL_RANGE,
        },
    );
    acc
}

/// Row 10 of a left profile with `exclusion` 0: what the shared-floor tests close.
const ROWS: core::ops::Range<usize> = 10..11;

/// A column partial over columns `0..20` of valid windows keeping `k` each, `reach` rows able to
/// reach each, every column's floor set to `floor`.
fn columns_at(k: usize, reach: usize, floor: f64) -> ColumnAcc {
    let windows = [VALID; 20];
    let mut part = ColumnAcc::new();
    part.begin(0..20, &windows, NonZeroUsize::new(k).unwrap(), |_| reach);
    for column in 0..20 {
        part.set_floor_for_test(column, floor);
    }
    part
}

/// A fresh row partial: valid, with no lane yet, so its own floor is open.
fn open_row() -> RowAcc {
    let mut acc = RowAcc::new();
    acc.begin(&VALID, NonZeroUsize::MIN, 10);
    acc
}

fn column_floors(table: &SharedFloors) -> Vec<f64> {
    (0..20)
        .map(|c| table.column(c, f64::NEG_INFINITY))
        .collect()
}

/// A row floor one tile's partial reached is what another tile reads: a closing tile raises its
/// rows' shared floors to the floor its partials reached. Dropping the raise loses it.
#[test]
fn a_closing_tile_raises_its_rows_shared_floors() {
    let table = SharedFloors::try_new(20, false).unwrap();
    assert_eq!(table.row(10, f64::NEG_INFINITY), f64::NEG_INFINITY);
    share_closed(&table, (&ROWS, &[row_at(VALID)]), None);
    assert!(table.row(10, f64::NEG_INFINITY) > 0.8);
    // A row with no lane yet has an open floor and raises nothing.
    let other = SharedFloors::try_new(20, false).unwrap();
    share_closed(&other, (&ROWS, &[open_row()]), None);
    assert_eq!(other.row(10, f64::NEG_INFINITY), f64::NEG_INFINITY);
}

/// At tile close every column the partial holds is raised, to its partial's floor: the columns
/// along the tile's top edge are never left by a range's raise.
#[test]
fn a_closing_tile_raises_every_column_it_holds() {
    let table = SharedFloors::try_new(20, true).unwrap();
    let part = columns_at(1, 5, 0.9);
    share_closed(&table, (&ROWS, &[open_row()]), Some(&part));
    assert!(column_floors(&table).iter().all(|&f| f == 0.9));
}

/// The column pitfall: a partial whose columns hold room for fewer than `k` (a tile where only
/// two rows reach them, at `k` 4) has a floor of only those rows' pairs, above the run's `k`-th
/// best, so it is never shared; a skip on it would drop a pair of the column's true top `k`.
#[test]
fn a_column_with_room_for_fewer_than_k_is_not_shared() {
    let table = SharedFloors::try_new(20, true).unwrap();
    let few = columns_at(4, 2, 0.9);
    assert!(!few.keeps_k(3));
    share_closed(&table, (&ROWS, &[open_row()]), Some(&few));
    assert!(
        column_floors(&table)
            .iter()
            .all(|&f| f == f64::NEG_INFINITY)
    );
}

/// The settled pitfall: a column settled in one tile (floor `∞`: `k` pairs of correlation one
/// from that tile's rows) may still tie a smaller row in another tile, so its `∞` is never
/// shared, while the other columns' finite floors are.
#[test]
fn a_settled_column_is_not_shared() {
    let table = SharedFloors::try_new(20, true).unwrap();
    let mut part = columns_at(1, 5, 0.5);
    part.set_floor_for_test(3, f64::INFINITY);
    share_closed(&table, (&ROWS, &[open_row()]), Some(&part));
    let floors = column_floors(&table);
    assert_eq!(floors[3], f64::NEG_INFINITY, "{floors:?}");
    assert!(floors.iter().enumerate().all(|(c, &f)| c == 3 || f == 0.5));
}

/// A row's floor raises the global floor to exactly the near-tie band below that floor, and a
/// lower one does not lower it.
#[test]
fn rows_raise_the_global_floor_to_the_band_below_their_floor() {
    let table = SharedFloors::try_new(20, false)
        .unwrap()
        .with_global(f64::NEG_INFINITY);
    share_closed(&table, (&ROWS, &[row_at(VALID)]), None);
    let floor = table.row(10, f64::NEG_INFINITY);
    assert!(floor > 0.8);
    let banded = crate::estimate::band_floor(
        floor,
        <crate::estimate::F64 as crate::estimate::Producer>::DELTA,
    );
    assert!(banded < floor);
    assert_eq!(table.global().map(f64::to_bits), Some(banded.to_bits()));
    share_closed(&table, (&ROWS, &[open_row()]), None);
    assert_eq!(table.global().map(f64::to_bits), Some(banded.to_bits()));
}

/// Where a run's global floor starts: only a best-pair run that offers every pair to a row keeping
/// one neighbor keeps one; it starts open, or just below a correlation of one when the constant
/// class holds two windows outside each other's zone.
#[test]
fn a_best_pair_runs_global_floor_starts_open_or_just_below_the_constant_pair() {
    use crate::batch::plan::Engine;
    use crate::batch::quantized::Quantized;
    let m = NonZeroUsize::new(8).unwrap();
    let scale = crate::quantize::Scale::new(1.0).unwrap();
    let series = |flat: usize| -> Vec<f64> {
        let mut series: Vec<f64> = (0..80)
            .map(|k| f64::from([3, 7, 1, 9, 4, 2][k % 6]))
            .collect();
        series[10..10 + flat].fill(5.0);
        series
    };
    let built =
        |flat, engine| Quantized::build_on(&series(flat), m, scale, NonZeroUsize::MIN, engine);
    // A flat run of m + 4 samples has windows up to four starts apart, outside a zone of two.
    let known = built(12, Engine::BestPair);
    let seed = known.global_floor(true).expect("a best-pair run keeps one");
    assert!(seed < 1.0 && seed > 1.0 - 1e-12, "{seed}");
    assert_eq!(known.global_floor(false), None);
    assert_eq!(built(12, Engine::TwoSided).global_floor(true), None);
    assert_eq!(built(12, Engine::Left).global_floor(true), None);
    // A flat run of m + 2 has windows within each other's zone: no pair is known.
    let none = built(10, Engine::BestPair);
    assert!(none.constant_class(None).is_some());
    assert_eq!(none.global_floor(true), Some(f64::NEG_INFINITY));
    assert_eq!(
        built(0, Engine::BestPair).global_floor(true),
        Some(f64::NEG_INFINITY)
    );
}
