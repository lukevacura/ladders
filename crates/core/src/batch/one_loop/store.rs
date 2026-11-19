//! The lane store: where a band leaves the lanes of one slot range for the band below. The tiles
//! of a run are the executor's units and nothing orders them, so a handoff is an offer: a band
//! takes the stored lanes only if they are current through the row before its own first row, and
//! otherwise starts its lanes lazily, which is exact. Bits therefore never depend on the order the
//! tiles ran in; only how many lanes restart does.
//!
//! What a band hands down is a [`Carry`]: the lanes' states, and the row each is current to. A
//! lane that a skip left behind is not marked when it is skipped (the walk records survivors and
//! computes them later, so a mark would run ahead of the compute); its row says how far it is
//! behind, and the band that computes it next catches it up through that gap.

use super::anchored::Frame;
use crate::capacity::{CapacityError, reserved_vec};
use crate::cross_term::UNSTARTED;
use alloc::vec::Vec;

#[cfg(any(feature = "parallel", feature = "oracle", loom, shuttle))]
use crate::sync::Mutex;
#[cfg(not(any(feature = "parallel", feature = "oracle", loom, shuttle)))]
use core::cell::RefCell;

/// The lanes of one slot range as a band leaves them: the exact state of each lane in the layout the
/// pass reads (slot `s` of the range at index `len - 1 - (s - first slot)`), and the row each lane is
/// current to.
pub(super) struct Carry {
    /// The lane states, one per slot of the range, as the pass lays a chunk out.
    pub(super) qt: Vec<i64>,
    /// Per slot of the range, the row its lane is current to: its state holds the pair of the row
    /// before it. [`UNSTARTED`] for a lane that holds no state yet.
    pub(super) through: Vec<usize>,
    /// The anchors the states are measured from (`Frame::default()` for a run with one anchor for
    /// the whole series, where every band's frame is the same).
    pub(super) frame: Frame,
    /// Whether the states are the wide path's (unscaled, over the exact i64 ticks) and not the
    /// narrow path's `m·QT` in `frame`.
    pub(super) wide: bool,
}

impl Carry {
    /// Lanes for `slots` slots, none started, or an error when they cannot be allocated
    /// (`try_reserve`, no panic).
    pub(super) fn try_new(slots: usize) -> Result<Self, CapacityError> {
        let mut qt = reserved_vec::<i64>(slots as u128)?;
        qt.resize(slots, 0);
        let mut through = reserved_vec::<usize>(slots as u128)?;
        through.resize(slots, UNSTARTED);
        Ok(Self {
            qt,
            through,
            frame: Frame::default(),
            wide: false,
        })
    }
}

/// What a band left in a cell: the value, and the row it is current up to, exclusive (its states
/// are those before row `through_row`, the band's end), so the band whose first row is
/// `through_row` continues it.
struct Handoff<T> {
    value: T,
    through_row: usize,
}

/// A cell: a lock with threads (`parallel`, `oracle`), a plain cell without, where the tiles run in
/// order on one thread and nothing can race.
#[cfg(any(feature = "parallel", feature = "oracle", loom, shuttle))]
type Cell<T> = Mutex<Option<Handoff<T>>>;
#[cfg(not(any(feature = "parallel", feature = "oracle", loom, shuttle)))]
type Cell<T> = RefCell<Option<Handoff<T>>>;

/// One cell per slot range, each holding what the last band that finished there left, if no band
/// has taken it. The rule is the store's and does not depend on what is stored.
struct Cells<T> {
    cells: Vec<Cell<T>>,
}

impl<T> Cells<T> {
    fn try_new(ranges: usize) -> Result<Self, CapacityError> {
        let mut cells = reserved_vec::<Cell<T>>(ranges as u128)?;
        #[cfg(any(feature = "parallel", feature = "oracle", loom, shuttle))]
        cells.resize_with(ranges, || Mutex::new(None));
        #[cfg(not(any(feature = "parallel", feature = "oracle", loom, shuttle)))]
        cells.resize_with(ranges, || RefCell::new(None));
        Ok(Self { cells })
    }

    /// Runs `f` on cell `range`; `None` when there is no such cell, or (without threads) when it is
    /// already borrowed, which a single-threaded caller never does. A poisoned lock is read as it
    /// stands: a cell holds a whole value or nothing, since `f` only swaps it.
    fn with<R>(&self, range: usize, f: impl FnOnce(&mut Option<Handoff<T>>) -> R) -> Option<R> {
        let cell = self.cells.get(range)?;
        #[cfg(any(feature = "parallel", feature = "oracle", loom, shuttle))]
        {
            let mut guard = match cell.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            Some(f(&mut guard))
        }
        #[cfg(not(any(feature = "parallel", feature = "oracle", loom, shuttle)))]
        {
            let mut guard = cell.try_borrow_mut().ok()?;
            Some(f(&mut guard))
        }
    }

    fn put(&self, range: usize, value: T, through_row: usize) {
        // A range past the store (or a borrowed cell) drops the value: the next band starts lazily.
        let _ = self.with(range, |cell| {
            *cell = Some(Handoff { value, through_row });
        });
    }

    fn take(&self, range: usize, first_row: usize) -> Option<T> {
        self.with(range, |cell| match cell {
            Some(held) if held.through_row == first_row => cell.take().map(|h| h.value),
            _ => None,
        })
        .flatten()
    }
}

/// The lanes a band left, per slot range of the run, for the band below.
pub(super) struct LaneStore {
    cells: Cells<Carry>,
}

impl core::fmt::Debug for LaneStore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LaneStore")
            .field("ranges", &self.cells.cells.len())
            .finish_non_exhaustive()
    }
}

impl LaneStore {
    /// A store of `ranges` empty cells, or an error when they cannot be allocated (`try_reserve`,
    /// no panic); allocated once per run.
    pub(super) fn try_new(ranges: usize) -> Result<Self, CapacityError> {
        Ok(Self {
            cells: Cells::try_new(ranges)?,
        })
    }

    /// Leaves `lanes` in cell `range`, current up to row `through_row` (exclusive), replacing what was
    /// there. A range past the store drops them (the next band starts lazily). A poisoned lock is
    /// taken as it stands: the cell holds a whole value or nothing.
    pub(super) fn put(&self, range: usize, lanes: Carry, through_row: usize) {
        self.cells.put(range, lanes, through_row);
    }

    /// The lanes of cell `range` if their `through_row == first_row`, leaving the cell empty; `None` (the cell untouched) when
    /// the cell is empty, holds other rows, or the route's no-handoff force is on. A band that
    /// gets `None` starts its lanes lazily. Never returns lanes one row stale or ahead.
    pub(super) fn take(&self, range: usize, first_row: usize) -> Option<Carry> {
        if crate::route::one_loop_no_handoff() {
            return None;
        }
        self.cells.take(range, first_row)
    }
}

#[cfg(test)]
mod tests;
