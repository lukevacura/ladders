//! The floors one self-join run's threads share: one monotone floor per window as a row (its left
//! neighbors' `k`-th best, band-floored) and, when the run offers pairs to columns, one as a
//! column (its right neighbors'). Each is the largest some partial accumulator of that window has
//! reached, so it is at most the window's final floor on that side: a block whose bound is below
//! it holds no pair of the window's result and no tie, the same as below the partial's own floor.
//!
//! **Why it exists.** A tile's partials see only the tile's pairs, so a tiled or threaded pass
//! gates its blocks against weaker floors than one walk over the whole row would. The table hands
//! every tile the best floor any tile has reached so far.
//!
//! **Rules** (every one is what keeps a shared value a floor of the whole run):
//! - only a finite floor is raised: `−∞` says nothing, a NaN floor is nobody's, and a column's
//!   `+∞` (settled: `k` pairs of correlation one from smaller rows) holds only for the rows its
//!   own tile offers, since another tile may still bring a smaller row that ties;
//! - a raise is an atomic max, so the stored floor never falls;
//! - a read returns the higher of the caller's own floor and the stored one, and the caller's own
//!   when that is NaN (nothing is below NaN, so it stays the stricter answer). A read may be stale:
//!   a stale floor is lower and only skips less.
//!
//! What a column partial may raise (only columns holding room for all `k` of their nearest) is its
//! caller's rule ([`super::share_closed`]); this owner takes only finite floors and keeps them monotone.
//!
//! **The global floor.** A run that reports only its best pair keeps one more monotone floor
//! beside the table's ([`SharedFloors::with_global`]): the highest floor any row has reached, or
//! the floor of a pair the run knows before it starts. Each of those stands for a real pair of
//! the run, outside the exclusion zone, so the best pair's correlation is at least the global
//! floor's, and a block whose bound is below it (by the declared band, as for any floor) holds no
//! pair that ties or beats the best. A window whose best pair is at least the final global floor
//! therefore has its exact entry, and every other window's best is below the best pair's and
//! cannot be picked. It follows the rules above; it is a floor like the others, an estimate
//! lowered by the band, never part of a result, and it belongs to one run of one series.
//!
//! The table is a rebuildable cache: it lives for one run, is made only when the survivor filter
//! is on, and is never in a checkpoint or a result. The amount of work it saves depends on the
//! order threads raise and read it (determinism, ladder amendment); results never do.
//!
//! **Memory order.** Every access is `Relaxed`. A stored floor is a self-contained claim (some
//! partial of that window reached it), nothing else is published with it, and atomic max on one
//! location is totally ordered, so any value a read returns is a floor some raise stored and a
//! later read of the same cell on the same thread never returns a lower one.

use alloc::vec::Vec;
use core::ops::Range;

use crate::capacity::CapacityError;
use crate::sync::{AtomicU64, Ordering::Relaxed};

/// One floor per window as a row, and (when the run has columns) one as a column, each stored as
/// an order-preserving integer key, so an atomic max on the integer is a max on the floor.
pub struct SharedFloors {
    rows: Vec<AtomicU64>,
    columns: Vec<AtomicU64>,
    /// The global floor, in a run that keeps one.
    global: Option<AtomicU64>,
}

/// `x`'s order as an unsigned integer: `key(a) < key(b)` exactly when `a < b`, for every non-NaN
/// `a` and `b`, with `−0` just below `+0` (they compare equal, so either is the same floor).
#[inline(always)]
fn key(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

/// The floor whose [`key`] is `k`.
#[inline(always)]
fn unkey(k: u64) -> f64 {
    if k >> 63 == 1 {
        f64::from_bits(k & !(1 << 63))
    } else {
        f64::from_bits(!k)
    }
}

/// A table side of `len` cells, every one open (`−∞`), reserved fallibly.
fn open(len: usize) -> Result<Vec<AtomicU64>, CapacityError> {
    let mut cells = Vec::new();
    cells.try_reserve_exact(len).map_err(|_| CapacityError)?;
    cells.extend((0..len).map(|_| AtomicU64::new(key(f64::NEG_INFINITY))));
    Ok(cells)
}

/// The higher of `own` and the floor `cell` stores; `own` when it is NaN or there is no cell.
#[inline(always)]
fn read(cell: Option<&AtomicU64>, own: f64) -> f64 {
    let Some(cell) = cell else {
        return own;
    };
    let shared = unkey(cell.load(Relaxed));
    if own.is_nan() || own >= shared {
        own
    } else {
        shared
    }
}

/// The lowest floor `cells` store over `range`: one `Relaxed` load per cell. `+∞` for an empty
/// range (nothing to be below), and `−∞` when the table does not hold every cell of the range: a
/// window it holds no cell for has no shared floor, so its floor is its caller's own and nothing
/// shared can say it is above `−∞`. Every stored floor is at most that window's final floor, and
/// a window's floor is the higher of its own and its stored one, so the lowest stored floor is at
/// most the lowest floor of the range: a value below it is below every window of the range, and a
/// stale read is lower still, so it only answers "not below" more often.
#[inline]
fn lowest(cells: &[AtomicU64], range: Range<usize>) -> f64 {
    if range.is_empty() {
        return f64::INFINITY;
    }
    let Some(held) = cells.get(range) else {
        return f64::NEG_INFINITY;
    };
    held.iter()
        .map(|cell| cell.load(Relaxed))
        .min()
        .map_or(f64::INFINITY, unkey)
}

/// The table's one write: stores the floor whose key is `k` in `cell` unless a floor at least as
/// high is there (an atomic max, so a racing raise never lowers it); whether it rose.
#[inline(always)]
fn store_max(cell: &AtomicU64, k: u64) -> bool {
    cell.fetch_max(k, Relaxed) < k
}

/// Raises `cell` to `floor` when `floor` is finite and above what it stores; whether it did.
#[inline(always)]
fn raise(cell: Option<&AtomicU64>, floor: f64) -> bool {
    let Some(cell) = cell.filter(|_| floor.is_finite()) else {
        return false;
    };
    let k = key(floor);
    // The load first: most raises find the floor already there, and a load does not take the
    // cache line from the other threads that read it.
    if cell.load(Relaxed) >= k {
        return false;
    }
    store_max(cell, k)
}

impl SharedFloors {
    /// A table for a run over `windows` windows, every floor open: rows always, columns only when
    /// `columns` (a run that offers pairs to its columns). A capacity error when it cannot be
    /// allocated; the run then gates on its partials alone.
    pub fn try_new(windows: usize, columns: bool) -> Result<Self, CapacityError> {
        Ok(Self {
            rows: open(windows)?,
            columns: open(if columns { windows } else { 0 })?,
            global: None,
        })
    }

    /// The table keeping a global floor that starts at `seed`: a floor some pair of the run
    /// stands for, or `−∞` (as any `seed` that is not finite) to start it open.
    pub(crate) fn with_global(mut self, seed: f64) -> Self {
        let seed = if seed.is_finite() {
            seed
        } else {
            f64::NEG_INFINITY
        };
        self.global = Some(AtomicU64::new(key(seed)));
        self
    }

    /// The global floor, `None` in a run that keeps none.
    #[inline(always)]
    pub(crate) fn global(&self) -> Option<f64> {
        self.global.as_ref().map(|cell| unkey(cell.load(Relaxed)))
    }

    /// Raises the global floor to `floor`, a floor a row of the run reached: only a finite one,
    /// and only upward. Returns whether it rose; never in a run that keeps none.
    #[inline(always)]
    pub(crate) fn raise_global(&self, floor: f64) -> bool {
        raise(self.global.as_ref(), floor)
    }

    /// Row `row`'s floor: the higher of `own` (its partial's) and the shared one. `own` for a row
    /// the table does not hold.
    #[inline(always)]
    pub fn row(&self, row: usize, own: f64) -> f64 {
        read(self.rows.get(row), own)
    }

    /// The lowest floor the table stores for the rows `range` (`+∞` for none, `−∞` when the table
    /// holds fewer rows than `range` reaches, as [`lowest`] explains), at most the lowest floor
    /// any row of `range` reads back from [`Self::row`] whatever its own floor is.
    #[inline]
    pub(in crate::batch) fn lowest_rows(&self, range: Range<usize>) -> f64 {
        lowest(&self.rows, range)
    }

    /// [`Self::lowest_rows`] for the columns `range`; `−∞` for any column when the run offers no
    /// pairs to its columns (a table without column cells).
    #[inline]
    pub(in crate::batch) fn lowest_columns(&self, range: Range<usize>) -> f64 {
        lowest(&self.columns, range)
    }

    /// [`Self::row`] for column `column`.
    #[inline(always)]
    pub fn column(&self, column: usize, own: f64) -> f64 {
        read(self.columns.get(column), own)
    }

    /// Raises row `row`'s shared floor to `floor`, a floor one of its partials reached: only a
    /// finite one, and only upward. Returns whether the stored floor rose.
    #[inline(always)]
    pub fn raise_row(&self, row: usize, floor: f64) -> bool {
        raise(self.rows.get(row), floor)
    }

    /// [`Self::raise_row`] for column `column`, a floor of a column partial that holds room for
    /// all `k` of the column's nearest.
    #[inline(always)]
    pub fn raise_column(&self, column: usize, floor: f64) -> bool {
        raise(self.columns.get(column), floor)
    }
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod tests {
    use super::{SharedFloors, key, unkey};

    const ORDERED: [f64; 9] = [
        f64::NEG_INFINITY,
        -1.0,
        -f64::MIN_POSITIVE,
        -0.0,
        0.0,
        f64::MIN_POSITIVE,
        0.5,
        1.0,
        f64::INFINITY,
    ];

    #[test]
    fn the_key_orders_as_the_floors_do_and_round_trips() {
        for (i, &a) in ORDERED.iter().enumerate() {
            assert_eq!(unkey(key(a)).to_bits(), a.to_bits(), "{a}");
            for &b in &ORDERED[i + 1..] {
                assert!(key(a) < key(b), "{a} {b}");
            }
        }
    }

    #[test]
    fn a_raise_never_lowers_and_a_read_takes_the_higher() {
        let t = SharedFloors::try_new(4, true).unwrap();
        assert_eq!(t.row(1, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(t.raise_row(1, 0.5));
        assert!(!t.raise_row(1, 0.25), "a lower floor is not stored");
        assert!(!t.raise_row(1, 0.5), "an equal one does not rise");
        assert_eq!(t.row(1, 0.25), 0.5);
        assert_eq!(t.row(1, 0.75), 0.75);
        assert_eq!(t.row(0, 0.1), 0.1, "another row is untouched");
        assert!(t.raise_row(1, 0.625));
        assert_eq!(t.row(1, f64::NEG_INFINITY), 0.625);
        assert_eq!(
            t.column(1, f64::NEG_INFINITY),
            f64::NEG_INFINITY,
            "rows and columns are apart"
        );
    }

    #[test]
    fn the_global_floor_is_kept_only_when_asked_and_only_rises() {
        let t = SharedFloors::try_new(2, true).unwrap();
        assert_eq!(t.global(), None);
        assert!(!t.raise_global(0.5), "a table without one takes none");
        assert_eq!(t.global(), None);
        let t = t.with_global(f64::NEG_INFINITY);
        assert_eq!(t.global(), Some(f64::NEG_INFINITY));
        for floor in [f64::INFINITY, f64::NAN, f64::NEG_INFINITY] {
            assert!(!t.raise_global(floor), "{floor}");
        }
        assert!(t.raise_global(0.5));
        assert!(!t.raise_global(0.25));
        assert!(!t.raise_global(0.5));
        assert_eq!(t.global(), Some(0.5));
        assert_eq!(t.row(0, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(t.column(0, f64::NEG_INFINITY), f64::NEG_INFINITY);
        let seeded = SharedFloors::try_new(1, false).unwrap().with_global(0.75);
        assert_eq!(seeded.global(), Some(0.75));
        let open = SharedFloors::try_new(1, false)
            .unwrap()
            .with_global(f64::NAN);
        assert_eq!(open.global(), Some(f64::NEG_INFINITY));
    }

    /// The lowest of `(0..len).map(read)` over `range`, one cell at a time.
    fn direct(range: core::ops::Range<usize>, len: usize, read: impl Fn(usize) -> f64) -> f64 {
        if range.is_empty() {
            return f64::INFINITY;
        }
        if range.end > len {
            return f64::NEG_INFINITY;
        }
        range.map(read).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn the_lowest_floor_of_a_range_is_the_direct_minimum() {
        let len = 11;
        let t = SharedFloors::try_new(len, true).unwrap();
        // Rows and columns raised to different, unordered floors; some cells stay open.
        let floors = [
            0.5,
            -0.25,
            0.75,
            f64::NAN,
            0.125,
            0.0,
            -0.0,
            0.875,
            0.625,
            -1.0,
            0.25,
        ];
        for (i, &floor) in floors.iter().enumerate() {
            if i % 4 != 3 {
                t.raise_row(i, floor);
            }
            if i % 3 != 1 {
                t.raise_column(i, floors[len - 1 - i]);
            }
        }
        for start in 0..=len + 1 {
            for end in 0..=len + 2 {
                let range = start..end;
                let rows = direct(range.clone(), len, |i| t.row(i, f64::NEG_INFINITY));
                let columns = direct(range.clone(), len, |i| t.column(i, f64::NEG_INFINITY));
                // Equal as values: `-0` and `+0` are the same floor.
                assert_eq!(t.lowest_rows(range.clone()), rows, "rows {range:?}");
                assert_eq!(
                    t.lowest_columns(range.clone()),
                    columns,
                    "columns {range:?}"
                );
            }
        }
    }

    #[test]
    fn the_lowest_floor_of_nothing_is_open_above_and_of_unheld_cells_open_below() {
        let t = SharedFloors::try_new(4, false).unwrap();
        for i in 0..4 {
            t.raise_row(i, 0.5);
        }
        assert_eq!(t.lowest_rows(2..2), f64::INFINITY, "an empty range");
        assert_eq!(t.lowest_rows(1..4), 0.5);
        assert_eq!(t.lowest_rows(3..5), f64::NEG_INFINITY, "row 4 is not held");
        assert_eq!(t.lowest_rows(9..10), f64::NEG_INFINITY);
        assert_eq!(
            t.lowest_columns(0..1),
            f64::NEG_INFINITY,
            "a table without columns says nothing of them"
        );
        assert_eq!(t.lowest_columns(0..0), f64::INFINITY);
        t.raise_row(2, 0.25);
        t.raise_row(2, 0.125);
        assert_eq!(
            t.lowest_rows(0..4),
            0.5,
            "a lower raise never lowers a cell"
        );
        let u = SharedFloors::try_new(3, false).unwrap();
        u.raise_row(1, 0.5);
        assert_eq!(
            u.lowest_rows(0..3),
            f64::NEG_INFINITY,
            "an open cell is -inf"
        );
        assert_eq!(u.lowest_rows(1..2), 0.5);
    }

    #[test]
    fn only_finite_floors_are_shared() {
        let t = SharedFloors::try_new(2, true).unwrap();
        for floor in [f64::INFINITY, f64::NAN, f64::NEG_INFINITY] {
            assert!(!t.raise_column(0, floor), "{floor}");
            assert!(!t.raise_row(0, floor), "{floor}");
        }
        assert_eq!(t.column(0, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(t.row(0, f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn a_nan_own_floor_stays_and_cells_out_of_range_change_nothing() {
        let t = SharedFloors::try_new(2, false).unwrap();
        assert!(t.raise_row(0, 0.5));
        assert!(t.row(0, f64::NAN).is_nan());
        assert!(
            !t.raise_column(0, 0.5),
            "a table without columns takes none"
        );
        assert_eq!(t.column(0, 0.1), 0.1);
        assert!(!t.raise_row(9, 0.5));
        assert_eq!(t.row(9, 0.1), 0.1);
    }
}
