use super::pass::{
    BLOCK_ROWS, Block, ClassSkip, Shape, WideBlock, advance_chunk, advance_chunk_skipping,
    advance_chunk_wide, advance_chunk_wide_skipping,
};
use super::plan::{SlotRuns, Tile, TilePlan};
use super::quantized::Quantized;
use super::reanchor::{TileSides, drifting_tile_partials, tile_fits_cells};
use super::survivor::{SharedFloors, share_closed};
use super::two_sided::RightBook;
use crate::bound_paa::PaaSides;
use crate::cross_term::CrossTerms;
use crate::kernel_neon::RowScalars;
use crate::narrow::{self, Anchor};
use crate::row::{ColumnAcc, Columns, NarrowColumns, RowAcc, WideRow, narrow_row_scalars};
use crate::stats::{ConstantClass, JoinClass, Window};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// One thread's reusable tile scratch: the tile's `qt` slots (up to the tile width), one L1
/// chunk's `ρ̂` tile, a partial [`RowAcc`] per band row, and the narrowed copies of the tile's two
/// sides (rows, columns) when each takes its own anchor.
pub(super) struct Scratch {
    pub(super) qt: Vec<i64>,
    pub(super) tile: Vec<f64>,
    pub(super) accs: Vec<RowAcc>,
    pub(super) scalars: Vec<RowScalars>,
    pub(super) row_ticks: Vec<i32>,
    pub(super) row_sums: Vec<i32>,
    pub(super) col_ticks: Vec<i32>,
    pub(super) col_sums: Vec<i32>,
    pub(super) right: ColumnAcc,
    pub(super) ends: Vec<usize>,
    pub(super) lane_anchors: Vec<i64>,
}

impl Scratch {
    /// A thread's scratch for `plan`'s tiles.
    pub(super) fn new(plan: TilePlan) -> Self {
        Self {
            qt: vec![0; plan.tile_lanes],
            tile: vec![0.0; plan.chunk_lanes],
            accs: (0..plan.band_rows).map(|_| RowAcc::new()).collect(),
            scalars: Vec::with_capacity(plan.band_rows),
            row_ticks: Vec::new(),
            row_sums: Vec::new(),
            col_ticks: Vec::new(),
            col_sums: Vec::new(),
            right: ColumnAcc::new(),
            ends: Vec::new(),
            lane_anchors: Vec::new(),
        }
    }
}

/// What a tile's body reports: whether it ran the exact i64 pass.
pub(super) type RanWide = bool;

/// What one narrow tile reads: the columns its neighbors come from (window `base` is `cols`'
/// first), its rows' scalars, the exact `m·QT` of its pairs, and the state a slot carries into the
/// band's first row (`None` for a slot that holds none), each side measured from its own anchor.
pub(super) struct NarrowTile<'a> {
    pub(super) cols: &'a NarrowColumns<'a>,
    pub(super) base: usize,
    pub(super) scalars: &'a [RowScalars],
    pub(super) pairs: CrossTerms<&'a [i32]>,
    pub(super) start_qt: &'a dyn Fn(usize) -> Option<i64>,
    pub(super) m: usize,
    pub(super) shape: Shape,
    pub(super) chunk_lanes: usize,
    pub(super) k: NonZeroUsize,
    /// The slots to walk, `None` for all of the tile's.
    pub(super) runs: Option<&'a SlotRuns>,
    /// The PAA coefficients of the rows' and columns' column sets, when they have them.
    pub(super) paa: Option<PaaSides<'a>>,
    /// The self-join's constant class, when it has one.
    pub(super) class: Option<&'a ConstantClass>,
    /// The AB-join's constant class, when it has one.
    pub(super) joined: Option<&'a JoinClass>,
}

/// The pair one row before the band's first row `r0` that triangle slot `slot` carries.
pub(super) fn triangle_start_pair(
    exclusion: usize,
    r0: usize,
    slot: usize,
) -> Option<(usize, usize)> {
    let lag = exclusion + 1 + slot;
    (lag < r0).then(|| (r0 - 1, r0 - 1 - lag))
}

/// The neighbor `j` that join slot `slot` pairs with row `r0`, when it is a continuing diagonal
/// (`j ≥ 1`): the slot carries the pair `(r0 − 1, j − 1)` into that row.
pub(super) fn join_start_column(n_b: usize, r0: usize, slot: usize) -> Option<usize> {
    let j = (r0 + n_b - 1).checked_sub(slot)?;
    (slot >= r0 && j >= 1).then_some(j)
}

/// The columns a triangle tile's pairs fall in, and its starting states are read from: the
/// neighbors from the one whose diagonal the tile starts furthest back to the last neighbor of its
/// last row.
pub(super) fn tile_columns(tile: &Tile, exclusion: usize) -> core::ops::Range<usize> {
    let (r0, r1) = (tile.rows.start, tile.rows.end);
    (r0 - 1).saturating_sub(exclusion + tile.qb)..r1 - 1 - exclusion - tile.qa
}

/// The columns a join tile's pairs fall in, and its starting states are read from: one before the
/// first neighbor it pairs (window 0 when it holds a fresh diagonal) to the last.
fn join_columns(tile: &Tile, n_b: usize) -> core::ops::Range<usize> {
    let (r0, r1) = (tile.rows.start, tile.rows.end);
    (r0 + n_b).saturating_sub(tile.qb).saturating_sub(1)..(r1 + n_b - 1 - tile.qa).min(n_b)
}

impl Shape {
    /// `column − row` along slot `slot`'s diagonal.
    #[inline]
    pub(super) fn offset(self, slot: usize) -> i64 {
        match self {
            Self::Triangle { exclusion } => -((exclusion + 1 + slot) as i64),
            Self::Join { n_b } => n_b as i64 - 1 - slot as i64,
        }
    }

    pub(super) fn columns(self, tile: &Tile) -> core::ops::Range<usize> {
        match self {
            Self::Triangle { exclusion } => tile_columns(tile, exclusion),
            Self::Join { n_b } => join_columns(tile, n_b),
        }
    }
}

/// Starts the lanes `qa..qb` of `tile_qt` (slice index `i` is slot `qb − 1 − i`) from
/// `start_qt`: every one, or with `runs` those in the set. Integer arithmetic is exact, so that
/// state is bit for bit the one the recurrence reaches.
fn start_lanes(
    (qa, qb): (usize, usize),
    tile_qt: &mut [i64],
    start_qt: &dyn Fn(usize) -> Option<i64>,
    runs: Option<&SlotRuns>,
) {
    let mut started = 0;
    match runs {
        None => {
            for (i, state) in tile_qt.iter_mut().enumerate() {
                if let Some(start) = start_qt(qb - 1 - i) {
                    *state = start;
                    started += 1;
                }
            }
        }
        Some(runs) => {
            for slot in runs.within(qa, qb).flatten() {
                if let Some(start) = start_qt(slot) {
                    tile_qt[qb - 1 - slot] = start;
                    started += 1;
                }
            }
        }
    }
    crate::route::count_lane_starts(started);
}

/// Calls `walk(sa, sb)` for the L1 sub-chunks of `chunk_lanes` of the slots `qa..qb`, in ascending
/// slots: all of them, or with `runs` those in the set (a chunk never spans two runs).
pub(super) fn for_each_chunk(
    (qa, qb): (usize, usize),
    chunk_lanes: usize,
    runs: Option<&SlotRuns>,
    mut walk: impl FnMut(usize, usize),
) {
    let mut span = |lo: usize, hi: usize| {
        for sa in (lo..hi).step_by(chunk_lanes) {
            walk(sa, (sa + chunk_lanes).min(hi));
        }
    };
    match runs {
        None => span(qa, qb),
        Some(runs) => runs.within(qa, qb).for_each(|run| span(run.start, run.end)),
    }
}

/// Opens `tile`'s partials: begins each band row's accumulator and, with `right`, the column
/// accumulator seeded from the book, each taking what the constant `class` settles for it.
/// Returns `right` for the tile's walk and [`close_tile`].
pub(super) fn open_tile<'a>(
    shape: Shape,
    tile: &Tile,
    (windows, class): (&[Window], Option<&ConstantClass>),
    k: NonZeroUsize,
    accs: &mut [RowAcc],
    right: Option<(&'a RightBook, &'a mut ColumnAcc)>,
) -> Option<(&'a RightBook, &'a mut ColumnAcc)> {
    let right = right.map(|(book, part)| {
        part.begin(shape.columns(tile), windows, k, |_| tile.rows.len());
        book.seed(part);
        if let Some(class) = class {
            part.cut_constant_columns(class);
        }
        (book, part)
    });
    for (acc, row) in accs.iter_mut().zip(tile.rows.clone()) {
        acc.begin(&windows[row], k, windows.len());
        if let Some(class) = class {
            acc.take_class(class.row(row, k));
        }
    }
    right
}

/// Closes a finished tile of `rows`: raises the shared `floors` of its rows from their partials
/// `accs` and of its columns ([`share_closed`]) when the run keeps them, then merges its column
/// partial into the book.
pub(super) fn close_tile(
    right: Option<(&RightBook, &mut ColumnAcc)>,
    windows: &[Window],
    floors: Option<&SharedFloors>,
    (rows, accs): (&core::ops::Range<usize>, &[RowAcc]),
) {
    if let Some(table) = floors {
        let part = right.as_ref().map(|(_, part)| &**part);
        share_closed(table, (rows, accs), part);
    }
    if let Some((book, part)) = right {
        book.merge(part, windows);
    }
}

/// The constant class's block skip for a tile of slots `qa..qb` and rows `rows`: the self-join's
/// for a triangle, the AB-join's for a join; `None` without the class that shape asks.
fn tile_skip<'a>(
    (class, joined): (Option<&'a ConstantClass>, Option<&'a JoinClass>),
    k: NonZeroUsize,
    shape: Shape,
    (qa, qb): (usize, usize),
    (rows, chunk_lanes): (core::ops::Range<usize>, usize),
) -> Option<ClassSkip<'a>> {
    match shape {
        Shape::Triangle { exclusion } => {
            ClassSkip::new(class, k, exclusion, (qa..qb, rows), chunk_lanes)
        }
        Shape::Join { n_b } => ClassSkip::for_join(joined, n_b, (qa..qb, rows), chunk_lanes),
    }
}

/// Starts every lane of `tile` from its exact windowed dot product ([`NarrowTile::start_qt`]), `m`
/// multiplies per lane, which is `m / band_rows` of the tile's own work per chunk. Integer
/// arithmetic is exact, so that state is bit for bit the one the recurrence reaches. Then walks
/// the lanes in L1 sub-chunks of `chunk_lanes`, each across the whole band in blocks of
/// [`BLOCK_ROWS`] through [`advance_chunk`], into `accs`, one lock-free partial per row: the plain
/// pass, every pair computed.
pub(super) fn narrow_tile_partials(
    t: &NarrowTile,
    tile: &Tile,
    qt: &mut [i64],
    rho: &mut [f64],
    accs: &mut [RowAcc],
    windows: &[Window],
    right: Option<(&RightBook, &mut ColumnAcc)>,
) {
    let NarrowTile {
        cols,
        base,
        scalars,
        pairs,
        start_qt,
        m,
        shape,
        chunk_lanes,
        k,
        runs,
        class,
        joined,
        ..
    } = *t;
    let Tile { rows, qa, qb, .. } = tile;
    let (qa, qb) = (*qa, *qb);
    let tile_qt = &mut qt[..qb - qa];
    let mut right = open_tile(shape, tile, (windows, class), k, accs, right);

    // Starting state: before the band's first row `r0`, a slot holds the pair one row back; a
    // diagonal that only starts inside the band is started by `advance_chunk` and needs none.
    // Slice index `i` is slot `qb − 1 − i`.
    let r0 = rows.start;
    start_lanes((qa, qb), tile_qt, start_qt, runs);
    let mut skip = tile_skip(
        (class, joined),
        k,
        shape,
        (qa, qb),
        (rows.clone(), chunk_lanes),
    );
    // L1 sub-chunks in ascending slots, as the single-threaded pass runs them; each walks the
    // whole band, so its `qt` stays hot, into the band's one accumulator per row.
    for_each_chunk((qa, qb), chunk_lanes, runs, |sa, sb| {
        let sub_qt = &mut tile_qt[qb - sb..qb - sa];
        for off in (0..rows.len()).step_by(BLOCK_ROWS) {
            let end = (off + BLOCK_ROWS).min(rows.len());
            let block = Block {
                cols,
                base,
                pairs,
                rows: r0 + off..r0 + end,
                scalars: &scalars[off..end],
                m,
                shape,
                slots: qb,
            };
            let part = right.as_mut().map(|(_, part)| &mut **part);
            let accs = &mut accs[off..end];
            match skip.as_mut() {
                Some(skip) => {
                    advance_chunk_skipping(&block, (sa, sb), sub_qt, accs, rho, part, skip)
                }
                None => advance_chunk(&block, (sa, sb), sub_qt, accs, rho, part),
            }
            if let Some((_, part)) = right.as_mut() {
                part.compact(windows);
            }
        }
    });
    close_tile(right, windows, None, (rows, accs));
}

/// What one wide tile reads: [`NarrowTile`]'s on the exact i64 ticks, with unscaled `qt` slots.
pub(super) struct WideTile<'a> {
    pub(super) cols: &'a Columns<'a>,
    pub(super) windows: &'a [Window],
    pub(super) row_scalars: &'a dyn Fn(usize) -> WideRow,
    pub(super) pairs: CrossTerms<&'a [i64]>,
    pub(super) start_qt: &'a dyn Fn(usize) -> Option<i64>,
    pub(super) m: usize,
    pub(super) shape: Shape,
    pub(super) chunk_lanes: usize,
    pub(super) k: NonZeroUsize,
    /// The slots to walk, `None` for all of the tile's.
    pub(super) runs: Option<&'a SlotRuns>,
    /// The PAA coefficients of the rows' and columns' column sets, when they have them.
    pub(super) paa: Option<PaaSides<'a>>,
    /// The self-join's constant class, when it has one.
    pub(super) class: Option<&'a ConstantClass>,
    /// The AB-join's constant class, when it has one.
    pub(super) joined: Option<&'a JoinClass>,
}

/// [`narrow_tile_partials`] on the exact i64 ticks, for a tile whose rows or columns span more
/// than any one anchor covers: the same lanes, chunks and partials, with unscaled `qt` slots.
pub(super) fn wide_tile_partials(
    t: &WideTile,
    tile: &Tile,
    scratch: &mut Scratch,
    right: Option<&RightBook>,
) {
    let WideTile {
        cols,
        windows,
        row_scalars,
        pairs,
        start_qt,
        m,
        shape,
        chunk_lanes,
        k,
        runs,
        class,
        joined,
        ..
    } = *t;
    let Scratch {
        qt,
        tile: rho,
        accs,
        right: part,
        ..
    } = scratch;
    let mut right = open_tile(
        shape,
        tile,
        (windows, class),
        k,
        accs,
        right.map(|book| (book, part)),
    );
    let Tile { rows, qa, qb, .. } = tile;
    let (qa, qb) = (*qa, *qb);
    let tile_qt = &mut qt[..qb - qa];
    let r0 = rows.start;
    start_lanes((qa, qb), tile_qt, start_qt, runs);
    let mut skip = tile_skip(
        (class, joined),
        k,
        shape,
        (qa, qb),
        (rows.clone(), chunk_lanes),
    );
    for_each_chunk((qa, qb), chunk_lanes, runs, |sa, sb| {
        let sub_qt = &mut tile_qt[qb - sb..qb - sa];
        for off in (0..rows.len()).step_by(BLOCK_ROWS) {
            let end = (off + BLOCK_ROWS).min(rows.len());
            let block = WideBlock {
                cols,
                row_scalars,
                pairs,
                rows: r0 + off..r0 + end,
                m,
                shape,
            };
            let part = right.as_mut().map(|(_, part)| &mut **part);
            let accs = &mut accs[off..end];
            match skip.as_mut() {
                Some(skip) => {
                    advance_chunk_wide_skipping(&block, (sa, sb), sub_qt, accs, rho, part, skip)
                }
                None => advance_chunk_wide(&block, (sa, sb), sub_qt, accs, rho, part),
            }
            if let Some((_, part)) = right.as_mut() {
                part.compact(windows);
            }
        }
    });
    close_tile(right, windows, None, (rows, accs));
}

/// The anchors a tile's rows and columns take, `None` when either side is wider than one anchor
/// covers (the tile runs the exact pass).
pub(super) fn anchors_of(
    row_ticks: &[i64],
    col_ticks: &[i64],
    m: NonZeroUsize,
) -> Option<(Anchor, Anchor)> {
    Some((
        Anchor::covering(row_ticks, m)?,
        Anchor::covering(col_ticks, m)?,
    ))
}

/// Whether a triangle tile of `ticks` needs the exact pass: no one anchor per side covers it, and
/// it does not fit cells either.
pub(super) fn triangle_tile_is_wide(
    ticks: &[i64],
    m: NonZeroUsize,
    tile: &Tile,
    chunk_lanes: usize,
) -> bool {
    let exclusion = crate::exclusion::zone(m.get());
    let columns = tile_columns(tile, exclusion);
    let (r0, r1) = (tile.rows.start, tile.rows.end);
    anchors_of(
        &ticks[r0 - 1..r1 - 1 + m.get()],
        &ticks[columns.start..columns.end - 1 + m.get()],
        m,
    )
    .is_none()
        && !tile_fits_cells(
            ticks,
            ticks,
            m,
            Shape::Triangle { exclusion },
            tile,
            chunk_lanes,
        )
}

/// Whether a join tile of `a`'s and `b`'s ticks needs the exact pass.
pub(super) fn join_tile_is_wide(
    a: &[i64],
    b: &[i64],
    m: NonZeroUsize,
    tile: &Tile,
    chunk_lanes: usize,
) -> bool {
    let n_b = (b.len() + 1).saturating_sub(m.get());
    let columns = join_columns(tile, n_b);
    let (r0, r1) = (tile.rows.start, tile.rows.end);
    anchors_of(
        &a[r0.saturating_sub(1)..r1 - 1 + m.get()],
        &b[columns.start..columns.end - 1 + m.get()],
        m,
    )
    .is_none()
        && !tile_fits_cells(a, b, m, Shape::Join { n_b }, tile, chunk_lanes)
}

/// One tile's partials when the rows and the columns each take their own anchor. The rows' windows
/// cover ticks `[r0 − 1, r1 + m − 1)`; the columns are the windows from the one whose diagonal the
/// tile starts furthest back to the last neighbor of its last row, a contiguous range. `N` and `V`
/// are unchanged by shifting either side's ticks by a whole tick count, so the two sides narrow
/// independently. A side wider than one anchor covers sends the tile to the exact pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn tile_partials(
    q: &Quantized,
    m: NonZeroUsize,
    chunk_lanes: usize,
    tile: &Tile,
    scratch: &mut Scratch,
    k: NonZeroUsize,
    right: Option<&RightBook>,
    runs: Option<&SlotRuns>,
) -> RanWide {
    let mu = m.get();
    let exclusion = crate::exclusion::zone(mu);
    let shape = Shape::Triangle { exclusion };
    let (r0, r1) = (tile.rows.start, tile.rows.end);
    // Columns: windows `cb..col_end`. The furthest-back start state is pair
    // `(r0 − 1, r0 − 1 − lag)` at the widest lag, or window 0 when a fresh diagonal is in the tile.
    let columns = shape.columns(tile);
    let (cb, col_end) = (columns.start, columns.end);
    let row_ticks = &q.ticks[r0 - 1..r1 - 1 + mu];
    let col_ticks = &q.ticks[cb..col_end - 1 + mu];
    let paa = q.paa.as_deref().map(PaaSides::of);
    let class = q.constant_class(runs);
    let anchors = anchors_of(row_ticks, col_ticks, m).filter(|_| crate::route::forced_cell() == 0);
    let Some((row_anchor, col_anchor)) = anchors else {
        let cols = q.columns();
        let sides = TileSides {
            rows: &cols,
            row_windows: &q.windows,
            cols: &cols,
            conversion: q.range.conversion(),
            class,
        };
        if drifting_tile_partials(&sides, m, shape, chunk_lanes, tile, scratch, k, right, runs) {
            return false;
        }
        let pairs = CrossTerms::of(cols.ticks, mu);
        let row_scalars = |row: usize| WideRow {
            enter: cols.ticks[row + mu - 1],
            leave: cols.ticks[row - 1],
            sum: q.windows[row].sum,
        };
        wide_tile_partials(
            &WideTile {
                cols: &cols,
                windows: &q.windows,
                row_scalars: &row_scalars,
                pairs,
                start_qt: &|slot| {
                    triangle_start_pair(exclusion, r0, slot).map(|(i, j)| pairs.at(i, j))
                },
                m: mu,
                shape,
                chunk_lanes,
                k,
                runs,
                paa,
                class,
                joined: None,
            },
            tile,
            scratch,
            right,
        );
        return true;
    };
    let Scratch {
        qt,
        tile: rho,
        accs,
        scalars,
        row_ticks: rt,
        row_sums: rs,
        col_ticks: ct,
        col_sums: cs,
        right: part,
        ..
    } = scratch;
    narrow::narrow_ticks_into(row_anchor, m, row_ticks, rt);
    narrow::narrow_sums_into(row_anchor, m, &q.sum[r0..r1], rs);
    narrow::narrow_ticks_into(col_anchor, m, col_ticks, ct);
    narrow::narrow_sums_into(col_anchor, m, &q.sum[cb..col_end], cs);
    let conversion = q.range.conversion();
    let row_cols = NarrowColumns {
        ticks: rt,
        sum: rs,
        r: &q.r,
        bias: &q.bias,
        variance: &q.variance,
        conversion,
    };
    scalars.clear();
    scalars.extend(tile.rows.clone().map(|row| {
        narrow_row_scalars(&row_cols, &q.windows[row], row - (r0 - 1), mu, rs[row - r0])
    }));
    let col_cols = NarrowColumns {
        ticks: ct,
        sum: cs,
        r: &q.r[cb..col_end],
        bias: &q.bias[cb..col_end],
        variance: &q.variance[cb..col_end],
        conversion,
    };
    let pairs = CrossTerms::new(&rt[..], r0 - 1, &ct[..], cb, mu);
    narrow_tile_partials(
        &NarrowTile {
            cols: &col_cols,
            base: cb,
            scalars,
            pairs,
            start_qt: &|slot| triangle_start_pair(exclusion, r0, slot).map(|(i, j)| pairs.at(i, j)),
            m: mu,
            shape,
            chunk_lanes,
            k,
            runs,
            paa,
            class,
            joined: None,
        },
        tile,
        qt,
        rho,
        accs,
        &q.windows,
        right.map(|book| (book, part)),
    );
    false
}
