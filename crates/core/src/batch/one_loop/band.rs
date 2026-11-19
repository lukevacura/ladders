//! The tile driver of the one-loop route: a tile's slots go ascending in ranges of
//! [`RANGE_SLOTS`]. For each range the driver reads the floors of the band's rows and of the
//! range's columns once ([`super::snapshot`]), lets the dense ladder prove blocks of the range
//! skippable ([`super::dense`]), computes the survivors, and raises the floors from the pairs it
//! computed, so the next range reads them risen. The lanes still live at the band's bottom edge
//! are handed to the next band through the store ([`super::store`]). A tile still opens and closes
//! its row and column partials per band, as the plain tiles do, so memory stays linear in the
//! series.
//!
//! **Computing a range.** The range is cut into chunks of the plan's `chunk_lanes` slots, counted
//! from the tile's first slot (so a lane's state stays in the L1 cache while a chunk's row blocks
//! go by). Inside a chunk the finest row blocks go in order, and each block's survivors are the
//! maximal runs of surviving 16-slot blocks, one call of the pass's leaf compute per run. The
//! rows of one lane ascend (a lane belongs to one chunk, and row blocks ascend), and so do the
//! rows of one column (a column's later pairs lie in higher slots, so in the same or a later chunk
//! and range).
//!
//! **Floors.** A range's skips rest on floors some partial had reached before the range began,
//! which are at most final; floors rise only after the range, from its computed pairs. The range
//! is a geometry choice, not a fitted one: its lanes' states (8 bytes a slot) fill about one L1
//! data cache of 32 KiB, and the snapshot costs a read per eight windows of the band's rows and
//! the range's columns, against `side x range` pairs it serves.
//!
//! **Lane rows.** A lane (a slot's diagonal) holds its exact state through some row, which this
//! driver keeps per slot (`Carry::through`): the pair of the row before it is the state. A skip
//! touches nothing, so a lane's row says how far behind it is, and the run that reaches it next
//! catches it up through exactly that gap by the cross-term owner's step-or-restart rule (a lane
//! that holds no state restarts from the exact dot product). A run computes only lanes current
//! through the row before its first row (`debug_assert!`ed before every compute).

use super::Counts;
use super::anchored::{Frame, Lanes, move_lanes};
use super::dense::{Band, Boxes, Floors, GRAIN, Ladder, SLOTS, valid_cells_over};
use super::snapshot::{FloorSnapshot, FloorSources};
use super::store::{Carry, LaneStore};
use crate::batch::pass::{
    Block, Shape, WideBlock, catch_up_lanes, compute_visits, compute_visits_wide,
};
use crate::batch::plan::Tile;
use crate::batch::survivor::{RowSurvey, SharedFloors, raise_global};
use crate::batch::tile::{NarrowTile, WideTile, close_tile, open_tile};
use crate::batch::two_sided::RightBook;
use crate::bound_paa::PaaSides;
use crate::capacity::reserved_vec;
use crate::cross_term::{CrossTerms, Side, UNSTARTED};
use crate::row::{ColumnAcc, RowAcc};
use crate::stats::{ConstantClass, JoinClass, RowsClass, Window};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Range;

/// The side of the finest block, in rows and in slots: one call of the leaf compute covers a row
/// block of this many rows.
pub(super) const LEAF: usize = SLOTS[SLOTS.len() - 1];

/// The slots of one range: the floors are read, the ladder passed and the floors raised once per
/// range. A multiple of the ladder's grain; a cache geometry (see the module's note), not a
/// tuned value.
pub(super) const RANGE_SLOTS: usize = 4096;
/// The coarse bits of a range.
pub(super) const RANGE_BITS: usize = RANGE_SLOTS / SLOTS[0];
const _: () = assert!(RANGE_SLOTS.is_multiple_of(SLOTS[0] * GRAIN));
/// The rows of the pieces a band is cut into on a run whose tiles each take anchors of their own,
/// where the plan's band has a tile no anchors cover: the slots one grain of the ladder covers. A
/// band so tall spans few enough ticks (rows, and the columns of its ranges, which grow with the
/// rows) for one anchor per side to cover a tile wherever the series' own windows allow.
pub(super) const ANCHORED_SIDE: usize = SLOTS[0] * GRAIN;

/// The pairs of the tiles `fits` rejects, to compare two cuts of the same rows by.
pub(super) fn uncovered_pairs(shape: Shape, tiles: &[Tile], fits: &dyn Fn(&Tile) -> bool) -> usize {
    tiles
        .iter()
        .filter(|tile| !fits(tile))
        .map(|tile| valid_cells_over(shape, &tile.rows, &(tile.qa..tile.qb)))
        .fold(0, usize::saturating_add)
}

/// The pieces of `rows` of [`ANCHORED_SIDE`] rows, cut at its multiples, in order.
pub(super) fn pieces(rows: Range<usize>) -> impl Iterator<Item = Range<usize>> + Clone {
    let end = rows.end;
    core::iter::successors(Some(rows.start), move |&at| {
        Some((at / ANCHORED_SIDE + 1) * ANCHORED_SIDE).filter(|&to| to < end)
    })
    .map(move |at| at..((at / ANCHORED_SIDE + 1) * ANCHORED_SIDE).min(end))
}

/// The sizes a run's bands, tiles and chunks take, derived from the plan's shape so that every
/// band is a power of two of blocks of [`LEAF`] rows, whatever shape the plan names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    /// The band's rows: a power of two, at least [`LEAF`], and the unit the ladder's row blocks
    /// divide.
    pub(super) side: usize,
    /// The slots of a tile, a multiple of `side`.
    pub(super) tile_lanes: usize,
    /// The slots of a chunk, a multiple of [`LEAF`].
    pub(super) chunk: usize,
}

impl Layout {
    /// The layout for a plan of `band_rows` rows, `chunk_lanes` slots per chunk and `tile_lanes`
    /// slots per tile, for a series of `windows` windows: the band is the largest power of two of
    /// leaf rows the plan's band holds (at least one), a tile a whole number of bands, and a
    /// chunk a whole number of leaves.
    pub(super) fn new(
        (band_rows, chunk_lanes, tile_lanes): (usize, usize, usize),
        windows: usize,
    ) -> Self {
        let squares = (band_rows / LEAF).max(1);
        let side = LEAF << squares.ilog2();
        let tile_lanes = tile_lanes
            .max(1)
            .div_ceil(side)
            .saturating_mul(side)
            .min(windows.max(1).div_ceil(side).saturating_mul(side));
        let chunk = chunk_lanes
            .max(1)
            .div_ceil(LEAF)
            .saturating_mul(LEAF)
            .min(tile_lanes);
        Self {
            side,
            tile_lanes,
            chunk,
        }
    }
}

/// One leaf of the compute: the rows of a row block and the slots of a run of surviving blocks,
/// with the first row of the tile and one past the last slot the carry holds.
pub(super) struct Leaf {
    pub(super) rows: Range<usize>,
    pub(super) slots: Range<usize>,
    pub(super) first_row: usize,
    pub(super) top: usize,
}

/// The arithmetic a tile runs in, which is all the band driver asks of it: the narrow tile's
/// (`m·QT` lanes over i32 offsets from the tile's anchors) or the wide tile's (unscaled `QT` lanes
/// over the exact i64 ticks). The driver is the same code at either width; only the leaf compute
/// and the cross-term side its catch-up reads differ.
pub(super) trait TileKernel {
    /// The side of the pairs' windows the lanes' states are measured over.
    type Side: Side;
    /// Whether the lanes' states are the wide path's. A carry's states are only read by a band of
    /// the same width; a band of the other one restarts them from the exact dot products.
    const WIDE: bool;
    fn shape(&self) -> Shape;
    fn paa(&self) -> Option<PaaSides<'_>>;
    fn m(&self) -> usize;
    fn k(&self) -> NonZeroUsize;
    fn class(&self) -> Option<&ConstantClass>;
    /// The AB-join's constant class, when it has one.
    fn joined(&self) -> Option<&JoinClass>;
    fn pairs(&self) -> CrossTerms<Self::Side>;
    /// Computes the pairs of `at`'s rows over its slots into `qt` (the carry's states, ending at
    /// `at.top`, so slot `s` is at `top - 1 - s`), `accs` (the leaf's rows' partials), `right` and
    /// `rho`; returns the pairs computed. The lanes must be current through the leaf's first row.
    fn compute(
        &self,
        at: &Leaf,
        qt: &mut [i64],
        accs: &mut [RowAcc],
        rho: &mut [f64],
        right: Option<&mut ColumnAcc>,
    ) -> usize;
}

impl<'a> TileKernel for NarrowTile<'a> {
    type Side = &'a [i32];
    const WIDE: bool = false;
    fn shape(&self) -> Shape {
        self.shape
    }
    fn paa(&self) -> Option<PaaSides<'_>> {
        self.paa
    }
    fn m(&self) -> usize {
        self.m
    }
    fn k(&self) -> NonZeroUsize {
        self.k
    }
    fn class(&self) -> Option<&ConstantClass> {
        self.class
    }
    fn joined(&self) -> Option<&JoinClass> {
        self.joined
    }
    fn pairs(&self) -> CrossTerms<Self::Side> {
        self.pairs
    }
    fn compute(
        &self,
        at: &Leaf,
        qt: &mut [i64],
        accs: &mut [RowAcc],
        rho: &mut [f64],
        right: Option<&mut ColumnAcc>,
    ) -> usize {
        let block = Block {
            cols: self.cols,
            base: self.base,
            pairs: self.pairs,
            rows: at.rows.clone(),
            scalars: &self.scalars[at.rows.start - at.first_row..at.rows.end - at.first_row],
            m: self.m,
            shape: self.shape,
            slots: at.top,
        };
        let (a, b) = (at.slots.start, at.slots.end);
        compute_visits(
            &block,
            (a, b),
            &mut qt[at.top - b..at.top - a],
            accs,
            rho,
            right,
            core::slice::from_ref(&at.slots),
        )
    }
}

impl<'a> TileKernel for WideTile<'a> {
    type Side = &'a [i64];
    const WIDE: bool = true;
    fn shape(&self) -> Shape {
        self.shape
    }
    fn paa(&self) -> Option<PaaSides<'_>> {
        self.paa
    }
    fn m(&self) -> usize {
        self.m
    }
    fn k(&self) -> NonZeroUsize {
        self.k
    }
    fn class(&self) -> Option<&ConstantClass> {
        self.class
    }
    fn joined(&self) -> Option<&JoinClass> {
        self.joined
    }
    fn pairs(&self) -> CrossTerms<Self::Side> {
        self.pairs
    }
    fn compute(
        &self,
        at: &Leaf,
        qt: &mut [i64],
        accs: &mut [RowAcc],
        rho: &mut [f64],
        right: Option<&mut ColumnAcc>,
    ) -> usize {
        let block = WideBlock {
            cols: self.cols,
            row_scalars: self.row_scalars,
            pairs: self.pairs,
            rows: at.rows.clone(),
            m: self.m,
            shape: self.shape,
        };
        let (a, b) = (at.slots.start, at.slots.end);
        compute_visits_wide(
            &block,
            (a, b),
            &mut qt[at.top - b..at.top - a],
            accs,
            rho,
            right,
            core::slice::from_ref(&at.slots),
        )
    }
}

/// What a tile's driver reads beyond the tile's own rows and slots.
pub(super) struct BandInput<'a, T, const K: usize> {
    /// The tile's columns, scalars and filters, in the tile's arithmetic.
    pub(super) kernel: &'a T,
    pub(super) layout: Layout,
    /// The run's boxes, built once.
    pub(super) boxes: &'a Boxes<K>,
    /// The run's shared floors, read for snapshots and raised after each range.
    pub(super) floors: Option<&'a SharedFloors>,
    /// The lanes handed between bands; a tile's slot range is its `column`.
    pub(super) store: &'a LaneStore,
    pub(super) windows: &'a [Window],
    /// The anchors the band's rows and columns are narrowed to; `Frame::default()` when one anchor
    /// covers the whole series.
    pub(super) frame: Frame,
    /// Each window's own sum of absolute ticks, of the row side and of the column side, to move
    /// lanes between frames; `None` for a run with one frame.
    pub(super) sums: Option<(&'a [i64], &'a [i64])>,
}

/// What the driver needs of the thread's scratch.
pub(super) struct TileScratch<'a> {
    /// The chunk's `ρ̂` buffer; grown to a chunk of the layout when it is shorter.
    pub(super) rho: &'a mut Vec<f64>,
    pub(super) accs: &'a mut [RowAcc],
}

/// What a run of survivors reads and writes, bundled for the compute.
struct Flush<'a, T> {
    kernel: &'a T,
    tile: &'a Tile,
    exclusion: usize,
    /// One past the last slot the carry holds: slot `s` is at `top - 1 - s` of the carry's states.
    top: usize,
}

/// The pairs of `rows` by `slots` that exist: the cells whose lag is at most the row's.
pub(super) fn valid_cells(exclusion: usize, rows: &Range<usize>, slots: &Range<usize>) -> usize {
    valid_cells_over(Shape::Triangle { exclusion }, rows, slots)
}

/// The columns the pairs of `rows` by `slots` fall in, clipped to a series of `windows` windows:
/// pair `(row, slot)` meets column `row - exclusion - 1 - slot`.
pub(super) fn columns_of(
    exclusion: usize,
    rows: &Range<usize>,
    slots: &Range<usize>,
    windows: usize,
) -> Range<usize> {
    let lag = exclusion + 1;
    let hi = rows.end.saturating_sub(lag + slots.start).min(windows);
    let lo = (rows.start + 1).saturating_sub(lag + slots.end).min(hi);
    lo..hi
}

/// Runs one tile: every range of its slots proved, computed and raised, lanes taken from and
/// handed to the store, and the tile's partials left as today's narrow tile leaves them. Returns
/// the tile's counts, or `None` when the tile is not one this driver runs (not a triangle, no
/// coefficients) or its working memory cannot be allocated, with nothing touched, so the caller
/// runs today's tile. Every valid cell of the tile's triangle is skipped on a proof or computed
/// exactly once, so `skipped_area + computed_area` is the tile's pairs.
pub(super) fn run_tile<T: TileKernel, const K: usize>(
    input: &BandInput<'_, T, K>,
    tile: &Tile,
    scratch: TileScratch<'_>,
    right: Option<(&RightBook, &mut ColumnAcc)>,
) -> Option<Counts> {
    #[cfg(feature = "std")]
    let started = std::time::Instant::now();
    let kernel = input.kernel;
    let Shape::Triangle { exclusion } = kernel.shape() else {
        return None;
    };
    let sides = kernel.paa()?;
    let layout = input.layout;
    let n = input.windows.len();
    let lag = exclusion + 1;
    let (qa, qb) = (tile.qa, tile.qb);
    let top = qa + layout.tile_lanes;
    if qb > top || qa >= qb || tile.rows.start < lag || tile.rows.end > n {
        return None;
    }
    let TileScratch { rho, accs } = scratch;

    // Everything that can fail to allocate is allocated before anything is touched.
    if rho.len() < layout.chunk {
        rho.try_reserve(layout.chunk - rho.len()).ok()?;
        rho.resize(layout.chunk, 0.0);
    }
    let mut ladder = Ladder::try_new(
        input.boxes,
        &Band {
            rows: tile.rows.clone(),
            slots: qa..qb,
        },
    )?;
    // A one-sided run (no column partial) offers no pair to a column, so no column has a floor to
    // read: its snapshot holds no column blocks, and the ladder's columns constrain nothing.
    let column_blocks = if right.is_some() {
        layout.side.saturating_add(RANGE_SLOTS) / 8 + 2
    } else {
        0
    };
    let mut snapshot = FloorSnapshot::try_new(layout.side / 8, column_blocks).ok()?;
    let row0 = tile.rows.start / layout.side * layout.side;
    let unit_rows = row0..row0 + layout.side;
    let mut counts = Counts {
        wide_bands: usize::from(T::WIDE),
        ..Counts::default()
    };
    let mut carry = match input.store.take(tile.column, tile.rows.start) {
        Some(carry)
            if carry.qt.len() == layout.tile_lanes && carry.through.len() == layout.tile_lanes =>
        {
            carry
        }
        _ => Carry::try_new(layout.tile_lanes).ok()?,
    };
    // Lanes a band of the other width left are in the other arithmetic: they restart from the
    // exact dot products, as lanes no band started do.
    if carry.wide != T::WIDE {
        carry.through.fill(UNSTARTED);
        carry.wide = T::WIDE;
    }
    counts.lanes_handed_down += (qa..qb.min(tile.rows.start.saturating_sub(lag)))
        .filter(|&s| carry.through[s - qa] != UNSTARTED)
        .count();
    // Lanes the band above left in other anchors are moved to this band's (a no-op for one frame).
    counts.lanes_moved += move_lanes(
        &mut carry,
        input.frame,
        &Lanes {
            m: kernel.m(),
            shape: kernel.shape(),
            first_row: tile.rows.start,
            slots: qa..qb.min(tile.rows.start.saturating_sub(lag)),
            span: (qa, top),
            sums: input.sums,
        },
    );

    let mut right = open_tile(
        kernel.shape(),
        tile,
        (input.windows, kernel.class()),
        kernel.k(),
        accs,
        right,
    );
    let flush = Flush {
        kernel,
        tile,
        exclusion,
        top,
    };
    let tile_rows = tile.rows.start..tile.rows.end.min(n);
    let settled = kernel.class().and_then(|class| class.blocks(kernel.k()));
    // What the class settles for each row of the tile, read once; without room for it the tile
    // computes what the ladder leaves, which only costs time.
    let (settled, row_classes) = match settled {
        Some(blocks) => match reserved_vec::<RowsClass>(tile_rows.len() as u128) {
            Ok(mut classes) => {
                classes.extend(tile_rows.clone().map(|row| blocks.row_class(row)));
                (Some(blocks), classes)
            }
            Err(_) => (None, Vec::new()),
        },
        None => (None, Vec::new()),
    };

    let coarse_bits = ladder.coarse_bits();
    let mut at = 0;
    while at < coarse_bits {
        let coarse = at..(at + RANGE_BITS).min(coarse_bits);
        debug_assert!(coarse.end > at, "the range loop advances");
        at = coarse.end;
        let slots = ladder.coarse_slots(coarse.clone());
        // The slots of the tile this range holds (the ladder's bits are padded to whole grains).
        let span = slots.start.max(qa)..slots.end.min(qb);
        if span.is_empty() {
            continue;
        }

        // Read the floors, then prove: both before any pair of the range is computed.
        let columns = columns_of(exclusion, &unit_rows, &slots, n);
        snapshot.refresh(
            &FloorSources {
                accs,
                first_row: tile.rows.start,
                right: right.as_ref().map(|(_, part)| &**part),
                shared: input.floors,
            },
            unit_rows.clone(),
            columns.clone(),
        );
        let floors = Floors {
            rows: snapshot.rows(),
            cols: snapshot.columns(),
        };
        ladder.refresh(input.boxes, sides, &floors, coarse.clone());
        let work = ladder.pass(input.boxes, coarse);
        for (count, evaluated) in counts.entries.iter_mut().zip(work.evaluated) {
            *count += evaluated;
        }

        // Compute the survivors, chunk-outer and row block inner.
        let (mut computed, mut survivor_area) = (0, 0);
        let first = (span.start - qa) / layout.chunk;
        let last = (span.end - 1 - qa) / layout.chunk;
        for chunk in first..=last {
            let cut = (qa + chunk * layout.chunk).max(span.start)
                ..(qa + (chunk + 1) * layout.chunk).min(span.end);
            let bits = ladder.fine_bits(cut.clone());
            for block in 0..ladder.finest_rows() {
                let rows = ladder.rows_of(block);
                if rows.end <= tile_rows.start || rows.start >= tile_rows.end {
                    continue;
                }
                let rows_in = rows.start.max(tile_rows.start)..rows.end.min(tile_rows.end);
                let block_class = settled.map(|_| {
                    let at = rows_in.start - tile_rows.start;
                    row_classes[at..at + rows_in.len()]
                        .iter()
                        .fold(row_classes[at], |all, &row| all.join(row))
                });
                for run in ladder.survivors_in(block, bits.clone()) {
                    let slots = ladder.slots_of(run);
                    let slots = slots.start.max(cut.start)..slots.end.min(cut.end);
                    // The class leaves out the parts of the run it settles for the block's rows.
                    let mut compute = |slots: Range<usize>| {
                        let area = valid_cells(exclusion, &rows_in, &slots);
                        let pairs = compute_run(
                            &flush,
                            rows.start,
                            slots,
                            (&mut carry, accs, rho),
                            right.as_mut().map(|(_, part)| &mut **part),
                            input.windows,
                            &mut counts,
                        );
                        debug_assert_eq!(pairs, area, "a survivor run computes exactly its pairs");
                        computed += pairs;
                        survivor_area += area;
                        counts.survivor_runs += 1;
                    };
                    match (&settled, block_class) {
                        (Some(settled), Some(class)) => {
                            settled.unsettled_in(class, &rows_in, &slots, LEAF, compute)
                        }
                        _ => compute(slots),
                    }
                }
            }
        }
        counts.computed_area += computed;
        counts.skipped_area +=
            valid_cells(exclusion, &tile_rows, &span).saturating_sub(survivor_area);

        // The floors the next range reads are raised from what this one computed; the snapshot
        // just used was read before, and the table only rises.
        let sources = FloorSources {
            accs,
            first_row: tile.rows.start,
            right: right.as_ref().map(|(_, part)| &**part),
            shared: input.floors,
        };
        snapshot.assert_not_above(&sources, &unit_rows, &columns);
        if let Some(table) = input.floors {
            raise(
                table,
                (accs, tile.rows.start),
                right.as_ref().map(|(_, part)| &**part),
                (
                    tile_rows.clone(),
                    &columns_of(exclusion, &tile_rows, &span, n),
                ),
            );
        }
    }
    close_tile(right, input.windows, input.floors, (&tile.rows, accs));
    // The band below continues these lanes where it starts.
    if tile.rows.end < n {
        input.store.put(tile.column, carry, tile.rows.end);
    }
    #[cfg(feature = "std")]
    {
        counts.band_nanos += usize::try_from(started.elapsed().as_nanos()).unwrap_or(usize::MAX);
    }
    crate::route::count_one_loop(&counts);
    Some(counts)
}

/// Raises the shared floors of `rows` (partials `accs`, whose first row is `first_row`) and of
/// the columns `columns` (the tile's column partial `right`, for each column that holds room for
/// all `k` of its nearest) to what the partials reached: a partial's floor comes from pairs
/// already offered, so it is at most the window's final floor, as a tile's close has it
/// ([`crate::batch::tile::close_tile`]).
pub(super) fn raise(
    table: &SharedFloors,
    (accs, first_row): (&[RowAcc], usize),
    right: Option<&ColumnAcc>,
    (rows, columns): (Range<usize>, &Range<usize>),
) {
    let mut highest = f64::NEG_INFINITY;
    for row in rows {
        let Some(acc) = row.checked_sub(first_row).and_then(|i| accs.get(i)) else {
            continue;
        };
        if let RowSurvey::Valid { floor, .. } = acc.survey() {
            table.raise_row(row, floor);
            highest = highest.max(floor);
        }
    }
    // The highest row floor raises the run's global floor, when it keeps one.
    raise_global(table, highest);
    if let Some(right) = right {
        let held = right.columns();
        for column in columns.start.max(held.start)..columns.end.min(held.end) {
            if right.keeps_k(column) {
                table.raise_column(column, right.floors(column, 1)[0]);
            }
        }
    }
}

/// Catches the lanes of `slots` (slots of a tile that starts at `qa`, the carry's states ending at
/// slot `top`, a lag of `lag`) up to the state before row `row`, a run of equal rows at a time:
/// stepped through the gap or restarted from the exact dot product, whichever the cross-term
/// owner's rule picks, and exactly the same bits either way.
pub(super) fn catch_up<S: Side>(
    pairs: &CrossTerms<S>,
    carry: &mut Carry,
    (qa, top, lag): (usize, usize, usize),
    (slots, row): (Range<usize>, usize),
    counts: &mut Counts,
) {
    let shape = Shape::Triangle { exclusion: lag - 1 };
    catch_up_over(pairs, carry, (qa, top, shape), (slots, row), counts);
}

/// [`catch_up`] for a run of any `shape`: slot `s` of row `row` meets column
/// `row + shape.offset(s)`. The step is the pass's ([`catch_up_lanes`]), shared with the plain
/// pass's class skip.
pub(super) fn catch_up_over<S: Side>(
    pairs: &CrossTerms<S>,
    carry: &mut Carry,
    (qa, top, shape): (usize, usize, Shape),
    (slots, row): (Range<usize>, usize),
    counts: &mut Counts,
) {
    let caught = catch_up_lanes(
        pairs,
        (&mut carry.through, &mut carry.qt),
        (qa, top, shape),
        (slots, row),
    );
    counts.lanes_restarted += caught.resumed;
    counts.lanes_stepped += caught.stepped;
    counts.lane_steps += caught.steps;
}

/// Computes the row block starting at `row0` (a multiple of [`LEAF`]) over the adjacent `slots`, a
/// run of surviving blocks, and returns the pairs computed.
#[allow(clippy::too_many_arguments)]
fn compute_run<T: TileKernel>(
    flush: &Flush<'_, T>,
    row0: usize,
    slots: Range<usize>,
    (carry, accs, rho): (&mut Carry, &mut [RowAcc], &mut [f64]),
    mut right: Option<&mut ColumnAcc>,
    windows: &[Window],
    counts: &mut Counts,
) -> usize {
    let kernel = flush.kernel;
    let Flush {
        tile,
        exclusion,
        top,
        ..
    } = *flush;
    let lag = exclusion + 1;
    let (qa, first_row) = (tile.qa, tile.rows.start);
    let rows = row0.max(tile.rows.start)..(row0 + LEAF).min(tile.rows.end);
    if rows.start >= rows.end || slots.end > top {
        return 0;
    }
    let (a, b) = (slots.start, slots.end);

    // Catch up the lanes that continue into the first row (slots below `rows.start - lag`), a run
    // of equal rows at a time: stepped through the gap or restarted, whichever is cheaper, and
    // exactly the same bits either way.
    let live_end = b.min(rows.start.saturating_sub(lag));
    catch_up(
        &kernel.pairs(),
        carry,
        (qa, top, lag),
        (a..live_end, rows.start),
        counts,
    );
    debug_assert!(
        (a..live_end).all(|s| carry.through[s - qa] == rows.start),
        "every lane is current before its run computes"
    );

    let computed = kernel.compute(
        &Leaf {
            rows: rows.clone(),
            slots: slots.clone(),
            first_row,
            top,
        },
        &mut carry.qt,
        &mut accs[rows.start - first_row..rows.end - first_row],
        rho,
        right.as_deref_mut(),
    );
    if let Some(part) = right {
        part.compact(windows);
    }
    // The lanes that exist at the last row are current through it.
    for s in a..b.min(rows.end.saturating_sub(lag)) {
        carry.through[s - qa] = rows.end;
    }
    computed
}

#[cfg(test)]
mod tests;
