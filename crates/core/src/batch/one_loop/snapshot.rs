//! The floor snapshot of a band: the lowest effective floor per block of eight windows of the
//! band's rows and of the columns a range of its slots meets, read once per range from the tile's
//! own partials and the run's shared table, and handed to the dense ladder
//! ([`super::dense::Floors`]) as the floors its thresholds come from.
//!
//! A floor read here is a floor of the final result at the time it is read (a partial's floor
//! comes from pairs already offered, a shared one from a partial's), and a block's value is the
//! lowest of its windows', so a skip against it is a proof, and a floor that has risen since is
//! only higher. A block the floors say nothing about (a constant window, no window that ranks, a
//! window the tile does not hold) reads `NaN`, which nothing is below: the ladder reads a `NaN`
//! floor as a threshold nothing exceeds.

use super::dense::FloorBlocks;
use crate::batch::survivor::{RowSurvey, SharedFloors};
use crate::capacity::{CapacityError, reserved_vec};
use crate::kbest::lower_floor;
use crate::row::{ColumnAcc, RowAcc};
use alloc::vec::Vec;
use core::ops::Range;

/// Windows per block of the floor snapshot: the PAA group.
const BLOCK: usize = 8;

/// What the snapshot reads the floors from: the tile's own partials and the run's shared table.
pub(super) struct FloorSources<'a> {
    /// The tile's row accumulators, `accs[i]` for row `first_row + i`.
    pub(super) accs: &'a [RowAcc],
    pub(super) first_row: usize,
    /// The tile's column partial.
    pub(super) right: Option<&'a ColumnAcc>,
    /// The run's shared floors, when it shares them.
    pub(super) shared: Option<&'a SharedFloors>,
}

/// The lowest effective floors per 8-window block of a band's rows and of a range's columns.
#[derive(Debug)]
pub(super) struct FloorSnapshot {
    /// The first row of the first block (a multiple of [`BLOCK`]) and the first column of the
    /// first column block.
    row0: usize,
    column0: usize,
    rows: Vec<f64>,
    columns: Vec<f64>,
    /// The shared table's part of each block, read at the same time (checked against the live
    /// table by [`Self::assert_not_above`]).
    rows_shared: Vec<f64>,
    columns_shared: Vec<f64>,
}

/// The higher of an own floor and a shared one as a window's floor is, `NaN` when the own is.
#[inline]
fn meet(own: f64, shared: f64) -> f64 {
    if own.is_nan() || own >= shared {
        own
    } else {
        shared
    }
}

/// A window block's floor with the run's global `floor` applied: the higher of the two, and `NaN`
/// (nothing is below it, a constant window or no ranking row) stays `NaN`, as the one-loop route never
/// skips such a block by the global floor either. `-inf` when the run keeps none.
#[inline]
fn with_global(block: f64, global: f64) -> f64 {
    if block.is_nan() || block >= global {
        block
    } else {
        global
    }
}

impl FloorSnapshot {
    /// A snapshot with room for `row_blocks` and `column_blocks` blocks, all `NaN` (nothing is
    /// below it), or an error when it cannot be allocated. A one-sided run asks for no column
    /// blocks and reads no column.
    pub(super) fn try_new(row_blocks: usize, column_blocks: usize) -> Result<Self, CapacityError> {
        let make = |len: usize| -> Result<Vec<f64>, CapacityError> {
            let mut v = reserved_vec(len as u128)?;
            v.resize(len, f64::NAN);
            Ok(v)
        };
        Ok(Self {
            row0: 0,
            column0: 0,
            rows: make(row_blocks)?,
            columns: make(column_blocks)?,
            rows_shared: make(row_blocks)?,
            columns_shared: make(column_blocks)?,
        })
    }

    /// The rows' floors, as the ladder reads them.
    pub(super) fn rows(&self) -> FloorBlocks<'_> {
        FloorBlocks {
            first: self.row0,
            floors: &self.rows,
        }
    }

    /// The columns' floors, as the ladder reads them; `None` for a snapshot with no column blocks
    /// (a one-sided run, whose columns hold no floor a pair must beat).
    pub(super) fn columns(&self) -> Option<FloorBlocks<'_>> {
        (!self.columns.is_empty()).then_some(FloorBlocks {
            first: self.column0,
            floors: &self.columns,
        })
    }

    /// Reads the floors of `rows` and `columns` (absolute window indexes; `rows` starts at a
    /// multiple of 8) from `sources`, one minimum per 8-window block, as many blocks as the
    /// snapshot has room for (the rest read `NaN`).
    pub(super) fn refresh(
        &mut self,
        sources: &FloorSources,
        rows: Range<usize>,
        columns: Range<usize>,
    ) {
        debug_assert!(rows.start.is_multiple_of(BLOCK), "rows start on a block");
        self.row0 = rows.start;
        self.column0 = columns.start - columns.start % BLOCK;
        // A pair below the global floor is below the best pair's: every block's floor is at least
        // it, read here before any pair of the range is computed.
        let global = sources
            .shared
            .and_then(SharedFloors::global)
            .unwrap_or(f64::NEG_INFINITY);
        for b in 0..self.rows.len() {
            let from = rows.start + b * BLOCK;
            let (own, shared) = block_of_rows(sources, from..from + BLOCK, rows.end);
            self.rows[b] = with_global(meet(own, shared), global);
            self.rows_shared[b] = shared;
        }
        for b in 0..self.columns.len() {
            let from = self.column0 + b * BLOCK;
            let (own, shared) = block_of_columns(sources, from..from + BLOCK, columns.end);
            self.columns[b] = with_global(meet(own, shared), global);
            self.columns_shared[b] = shared;
        }
    }

    /// `debug_assert!`s that no block's shared floor is above what the table holds now: the table
    /// only rises, so a snapshot read earlier is never above the live floor.
    pub(super) fn assert_not_above(
        &self,
        sources: &FloorSources,
        rows: &Range<usize>,
        columns: &Range<usize>,
    ) {
        #[cfg(debug_assertions)]
        {
            for (b, &was) in self.rows_shared.iter().enumerate() {
                let from = self.row0 + b * BLOCK;
                let (_, live) = block_of_rows(sources, from..from + BLOCK, rows.end);
                debug_assert!(
                    was.partial_cmp(&live) != Some(core::cmp::Ordering::Greater),
                    "a row snapshot {was} is above the live floor {live}"
                );
            }
            for (b, &was) in self.columns_shared.iter().enumerate() {
                let from = self.column0 + b * BLOCK;
                let (_, live) = block_of_columns(sources, from..from + BLOCK, columns.end);
                debug_assert!(
                    was.partial_cmp(&live) != Some(core::cmp::Ordering::Greater),
                    "a column snapshot {was} is above the live floor {live}"
                );
            }
        }
        #[cfg(not(debug_assertions))]
        let _ = (sources, rows, columns);
    }
}

/// The own and the shared floor of a block of rows `block` (clipped at `end`): the lowest of the
/// ranking rows' partial floors and of their shared floors. `NaN` for the own one when a row is
/// constant or none ranks (nothing is below it, so the block is never skipped).
fn block_of_rows(sources: &FloorSources, block: Range<usize>, end: usize) -> (f64, f64) {
    let block = block.start..block.end.min(end);
    let acc_of = |row: usize| {
        row.checked_sub(sources.first_row)
            .and_then(|i| sources.accs.get(i))
    };
    let mut own = f64::INFINITY;
    let mut ranking = 0;
    for row in block.clone() {
        match acc_of(row).map(RowAcc::survey) {
            Some(RowSurvey::Constant) => return (f64::NAN, f64::NEG_INFINITY),
            Some(RowSurvey::Valid { floor, .. }) => {
                ranking += 1;
                own = lower_floor(own, floor);
            }
            Some(RowSurvey::Silent) | None => {}
        }
    }
    if ranking == 0 {
        return (f64::NAN, f64::NEG_INFINITY);
    }
    let shared = match sources.shared {
        None => f64::NEG_INFINITY,
        // Every row of the block ranks: one read of the table's lowest over the block.
        Some(table) if ranking == block.len() => table.lowest_rows(block),
        // A row that ranks nothing sets no floor and is left out of the minimum, so the rows are
        // read one by one.
        Some(table) => block
            .filter(|&row| {
                matches!(
                    acc_of(row).map(RowAcc::survey),
                    Some(RowSurvey::Valid { .. })
                )
            })
            .map(|row| table.lowest_rows(row..row + 1))
            .fold(f64::INFINITY, f64::min),
    };
    (own, shared)
}

/// The own and the shared floor of a block of columns `block` (clipped at `end`): own partial
/// floors from the tile's column partial (`-inf` for a column it does not hold) and the shared
/// table's lowest.
fn block_of_columns(sources: &FloorSources, block: Range<usize>, end: usize) -> (f64, f64) {
    let block = block.start..block.end.min(end);
    if block.start >= block.end {
        return (f64::NAN, f64::NEG_INFINITY);
    }
    let mut own = f64::INFINITY;
    match sources.right {
        Some(right) => {
            let held = right.columns();
            for column in block.clone() {
                let floor = if held.contains(&column) {
                    right.floors(column, 1)[0]
                } else {
                    f64::NEG_INFINITY
                };
                own = lower_floor(own, floor);
            }
        }
        None => own = f64::NEG_INFINITY,
    }
    let shared = sources
        .shared
        .map_or(f64::NEG_INFINITY, |table| table.lowest_columns(block));
    (own, shared)
}

#[cfg(test)]
mod tests;
