//! The survivor filter's shared state: the table of floors a self-join run's tiles share
//! ([`SharedFloors`]) and the seed pre-pass that raises it before the tiles run ([`seeds`]). The
//! one-loop route's ladder ([`crate::batch::one_loop`]) reads those floors to prove blocks of
//! pairs skippable from a PAA bound ([`asked`]): a block is skipped only when its bound is below
//! the floor of every row and every column in it ([`crate::kbest::below_floor`], the rule the
//! keeper's lane blocks use too), so a skipped pair is strictly worse than `k` pairs its row or
//! column already holds, and it is in no result and in no tie. A block holding a constant window
//! is never skipped (the constant-row route owns it). Skipping never changes a result, only the
//! work.
//!
//! With the filter on, how many blocks are skipped can differ between thread counts and tile
//! shapes, because floors arrive in a different interleaving; results never differ (determinism,
//! ladder amendment). The counters are reported beside the result, never in it.
//!
//! A two-sided run that reports only its best pair keeps a global floor
//! ([`SharedFloors::with_global`]): a block whose bound is below it is skipped whatever its own
//! rows' and columns' floors are, so the run's best pair, and every window that ties it, is the
//! plain pass's, and the other windows' entries are not.

use core::ops::Range;

use crate::estimate::{F64, Producer};
use crate::row::{ColumnAcc, RowAcc};
use crate::sync::Arc;

pub(in crate::batch) mod asked;
mod floors;
pub use floors::SharedFloors;
mod seeds;
pub(super) use seeds::{SeedInput, seed_floors};

/// Which filter a run installs: the PAA bound of the one-loop route's ladder, or none, which is the
/// plain pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Filter {
    /// Every pair is computed: the plain pass.
    #[default]
    Off,
    /// The one-loop route's ladder proves blocks skippable from the PAA coefficients its column
    /// set was built with.
    Paa,
}

impl Filter {
    /// Whether the run reads PAA coefficients.
    pub(crate) fn reads_paa(self) -> bool {
        self == Self::Paa
    }
}

/// The shared floors a tiled self-join run over `windows` windows installs: a fresh table when
/// the filter is on and floors are shared (the route harness may force each tile onto its own
/// partials instead), with column floors when the run offers pairs to its columns. `None` when the
/// filter is off, or when the table cannot be allocated, and the run gates on its partials alone.
/// A run whose plan names a `global` floor keeps one, starting there.
pub(super) fn shared_floors(
    filter: Filter,
    windows: usize,
    columns: bool,
    global: Option<f64>,
) -> Option<Arc<SharedFloors>> {
    if filter == Filter::Off || !crate::route::floors_shared(true) {
        return None;
    }
    let table = SharedFloors::try_new(windows, columns).ok()?;
    crate::route::count_floor_table();
    let table = match global {
        Some(seed) => table.with_global(seed),
        None => table,
    };
    Some(Arc::new(table))
}

/// Raises `table`'s global floor, in a run that keeps one, to the near-tie band below `floor`, a
/// floor one of its rows reached, and tells the route harness when it rose. The band holds even
/// for a row whose floor has none yet (a row the constant class seeded, before any estimate).
#[inline(always)]
pub(super) fn raise_global(table: &SharedFloors, floor: f64) {
    if table.raise_global(crate::estimate::band_floor(floor, F64::DELTA)) {
        crate::route::count_global_rise();
    }
}

/// A row as the filter judges a block of it.
#[derive(Clone, Copy)]
pub(crate) enum RowSurvey {
    /// A non-finite row: it ranks nothing, so it sets no floor.
    Silent,
    /// A constant row: its block is the constant-row route's, never skipped.
    Constant,
    /// A valid row and its current floor.
    Valid { floor: f64 },
}

/// When a self-join tile of `rows` closes, raises the shared floors of its rows to what their
/// partials `accs` reached over its last chunk, and with the column partial `right`, of every
/// column it holds: the columns along the tile's top edge are never left by a block inside the
/// tile, so a range's raise (the one-loop route's `raise`) does not reach them. Counts the floors
/// that rose for the route harness.
pub(super) fn share_closed(
    table: &SharedFloors,
    (rows, accs): (&Range<usize>, &[RowAcc]),
    right: Option<&ColumnAcc>,
) {
    let mut raises = 0;
    let mut highest = f64::NEG_INFINITY;
    for (acc, row) in accs.iter().zip(rows.clone()) {
        if let RowSurvey::Valid { floor, .. } = acc.survey() {
            raises += usize::from(table.raise_row(row, floor));
            highest = highest.max(floor);
        }
    }
    raise_global(table, highest);
    if let Some(right) = right {
        raises += raise_columns(table, right, right.columns());
    }
    crate::route::count_floor_raises(raises);
}

/// Raises the shared floors of `columns` (all held by `right`) to their partial's, for each column
/// that holds room for all `k` of its nearest ([`ColumnAcc::keeps_k`]): a column with fewer slots
/// (in a tile where fewer than `k` rows reach it) ranks only the rows that tile offers, so its
/// floor is not a floor of the whole run. Returns how many rose.
fn raise_columns(table: &SharedFloors, right: &ColumnAcc, columns: Range<usize>) -> usize {
    let floors = right.floors(columns.start, columns.len());
    floors
        .iter()
        .zip(columns)
        .filter(|&(&floor, column)| right.keeps_k(column) && table.raise_column(column, floor))
        .count()
}

/// The streams' hook, the place a stream asks whether to run a block of its diagonals (`row`
/// against neighbors `lanes`): a stream visits every block until its sleeping diagonals are built.
#[inline(always)]
pub(crate) fn stream_visits(_row: usize, _lanes: Range<usize>) -> bool {
    true
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod seeds_tests;
#[cfg(test)]
mod tests;
