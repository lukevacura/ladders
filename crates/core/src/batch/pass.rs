#[cfg(test)]
use super::VISITS;
use crate::capacity::reserved_vec;
use crate::cross_term::{CatchUp, CrossTerms, Side, UNSTARTED};
use crate::entry::Entry;
use crate::kernel_neon::RowScalars;
use crate::nearest::Nearest;
use crate::row::{
    ChunkCols, ColumnAcc, Columns, FrontGrowBuffer, NarrowColumns, RankRow, RowAcc, WideChunkCols,
    WideRank, WideRow, centered_cross_fast, narrow_row_scalars, pass_chunk_offering,
    pass_chunk_wide, window_at_narrow,
};
use crate::stats::{ClassBlocks, ConstantClass, JoinClass, JoinRows, RowsClass, Window};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Range;

/// Rows advanced per diagonal chunk before it is stored back: enough to amortize a chunk's own
/// load over several rows, small enough that every row's own scalars fit comfortably in
/// registers alongside it.
pub(super) const BLOCK_ROWS: usize = 8;

/// Which pairs a row ranks, as slots: a slot is one diagonal, and slot `q` of a row with slots
/// `first..first + count` pairs it with neighbor `first + count − 1 − q`, so ascending slots are
/// descending neighbors.
#[derive(Clone, Copy, Debug)]
pub(super) enum Shape {
    /// The left profile: row `i` has the lags past the exclusion zone, slot `q` the lag
    /// `exclusion + 1 + q`.
    Triangle { exclusion: usize },
    /// The AB-join: row `i` meets every one of the `n_b` windows of `b`, slot `q` the diagonal
    /// of neighbor `i + n_b − 1 − q`.
    Join { n_b: usize },
}

impl Shape {
    /// `row`'s first slot and how many it has.
    pub(super) fn row_slots(self, row: usize) -> (usize, usize) {
        match self {
            Self::Triangle { exclusion } => (0, row - exclusion),
            Self::Join { n_b } => (row, n_b),
        }
    }
}

/// What one block of rows shares across its slot chunks: the columns the neighbors are read from
/// (window `base` is `cols`' first), the exact `m·QT` of its pairs (a row's fresh diagonal, paired
/// with window 0, is only read where window 0 is in `cols`), the block's rows and their scalars,
/// and the number of live `qt` slots (counted from the buffer's far end).
pub(super) struct Block<'a> {
    pub(super) cols: &'a NarrowColumns<'a>,
    pub(super) base: usize,
    pub(super) pairs: CrossTerms<&'a [i32]>,
    pub(super) rows: core::ops::Range<usize>,
    pub(super) scalars: &'a [RowScalars],
    pub(super) m: usize,
    pub(super) shape: Shape,
    pub(super) slots: usize,
}

/// What one row does in slot chunk `[qa, qb)`: its fresh diagonal (paired with window 0) is its
/// last slot, started in the chunk that holds it and nowhere else.
struct RowChunk {
    /// The slot of the row's fresh diagonal, when this chunk holds it.
    fresh: Option<usize>,
    /// Continuing diagonals are slots `from..hi` (none when `hi == from`); slot `q` pairs the row
    /// with neighbor `end − 1 − q`, so ascending slot is descending neighbor.
    from: usize,
    hi: usize,
    end: usize,
}

impl RowChunk {
    /// The neighbor ranges of the continuing diagonals inside `visits` (slot ranges), one per
    /// visit that holds some, ascending slots first.
    fn neighbors<'a>(
        &'a self,
        visits: &'a [core::ops::Range<usize>],
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        let end = self.end;
        visits.iter().filter_map(move |visit| {
            let (qs, qe) = (visit.start.max(self.from), visit.end.min(self.hi));
            (qs < qe).then(|| (end - qe, end - qs))
        })
    }

    /// One past the slot of the diagonal pairing neighbor `j`, the first a chunk's state slice from
    /// `j` ends before.
    fn slot_of(&self, j: usize) -> usize {
        self.end - j
    }
}

/// A row with slots `first..first + count` in slot chunk `[qa, qb)`, or `None` when it has no
/// slot in it.
fn row_chunk(first: usize, count: usize, qa: usize, qb: usize) -> Option<RowChunk> {
    let end = first + count;
    if qb <= first || qa >= end {
        return None;
    }
    let from = qa.max(first);
    let fresh = end - 1;
    let (fresh, hi) = if fresh < qb {
        (Some(fresh), fresh)
    } else {
        (None, qb)
    };
    Some(RowChunk {
        fresh,
        from,
        hi,
        end,
    })
}

/// Catches the lanes of `slots` up to the state before row `row` after a skipped run of `run`
/// rows ([`UNSTARTED`] for lanes that hold no state), `qt` holding the states ending at slot
/// `top` (slot `s` at `top - 1 - s`): stepped or restarted by the cross-term owner's rule.
fn catch_up_run<S: Side>(
    pairs: &CrossTerms<S>,
    qt: &mut [i64],
    (top, shape): (usize, Shape),
    (slots, row): (Range<usize>, usize),
    run: usize,
) -> CatchUp {
    // The lanes' states in the pass's layout; the first lane's pair is the lowest column's.
    let states = &mut qt[top - slots.end..top - slots.start];
    let first_column = (row as i64 + shape.offset(slots.end - 1)) as usize;
    pairs.advance_or_resume(run, (row, first_column), states)
}

/// Catches the lanes of `slots` (slots of a run that starts at `qa`, the states in `qt` ending at
/// slot `top`, so slot `s` is at `top - 1 - s`, and `through_of` holding each slot's current row
/// from `qa`) up to the state before row `row`, a run of equal rows at a time: stepped through the
/// gap or restarted from the exact dot product, whichever the cross-term owner's rule picks, and
/// exactly the same bits either way. Returns what the catch-up did.
pub(super) fn catch_up_lanes<S: Side>(
    pairs: &CrossTerms<S>,
    (through_of, qt): (&mut [usize], &mut [i64]),
    (qa, top, shape): (usize, usize, Shape),
    (slots, row): (Range<usize>, usize),
) -> CatchUp {
    let mut caught = CatchUp::default();
    if slots.start >= slots.end {
        return caught;
    }
    // Most blocks find every lane current: one vector scan says so.
    if through_of[slots.start - qa..slots.end - qa]
        .iter()
        .all(|&through| through == row)
    {
        return caught;
    }
    let mut s = slots.start;
    while s < slots.end {
        let through = through_of[s - qa];
        let mut e = s + 1;
        while e < slots.end && through_of[e - qa] == through {
            e += 1;
        }
        if through != row {
            debug_assert!(
                through == UNSTARTED || through < row,
                "a lane is never ahead of the first row of the block that reaches it"
            );
            let run = if through == UNSTARTED {
                UNSTARTED
            } else {
                row.saturating_sub(through)
            };
            let done = catch_up_run(pairs, qt, (top, shape), (s..e, row), run);
            caught.resumed += done.resumed;
            caught.stepped += done.stepped;
            caught.steps += done.steps;
            through_of[s - qa..e - qa].fill(row);
        }
        s = e;
    }
    caught
}

/// The slots a part of the constant class's block skip holds: the unit the class settles whole.
const CLASS_PART: usize = crate::kbest::BLOCK_LANES;

/// A part every block that reaches it finds current: it has not been left out since it was last
/// computed (or started).
const CURRENT: usize = usize::MAX;

/// What a [`ClassSkip`] asks of the class, and what the class says of each row, read once.
enum Source<'a> {
    /// A self-join's: the class's blocks, and the lag of slot 0.
    Triangle {
        blocks: ClassBlocks<'a>,
        lag: usize,
        rows: Vec<RowsClass>,
    },
    /// An AB-join's, with the windows of `b`.
    Join {
        class: &'a JoinClass,
        n_b: usize,
        rows: Vec<JoinRows>,
    },
}

/// The plain pass's constant-class block skip: what the pass of one self-join or AB-join asks the
/// class ([`ClassBlocks::unsettled_in`], [`JoinClass::unsettled_in`]) before it advances a block of
/// rows over a chunk of slots, and the lane state the answer leaves behind. The class settles
/// whole parts of [`CLASS_PART`] slots, so the state is one word per part of a chunk:
/// [`CURRENT`], or the row the part's lanes are current to once a block has left it out. The block
/// that next computes the part catches its lanes up first, so every state the pass reads is
/// current; a part never left out costs nothing. Only a skip proven by the class is taken, so
/// every output bit is the plain pass's.
pub(super) struct ClassSkip<'a> {
    source: Source<'a>,
    /// The first slot of the pass, and the slots of one chunk: chunk `c` holds the slots from
    /// `first + c * chunk_lanes`, and its parts are numbered from the part of its first slot.
    first: usize,
    chunk_lanes: usize,
    /// The parts one chunk can touch.
    stride: usize,
    /// Per part of every chunk, [`CURRENT`] or the row its lanes are current to.
    through: Vec<usize>,
    /// How many parts are not [`CURRENT`].
    marked: usize,
    /// The first row of `rows`.
    first_row: usize,
    /// The slot runs the last block computes; sized for a chunk, never grown.
    visits: Vec<Range<usize>>,
}

impl<'a> ClassSkip<'a> {
    /// The skip for a pass over `slots` of the rows `rows` in chunks of `chunk_lanes` from the
    /// first slot, of a self-join that keeps `k` neighbors per row with exclusion zone
    /// `exclusion`. `None` when there is no class or it is forced off, or when its small tables
    /// cannot be allocated: the pass then computes every pair, which only costs time.
    pub(super) fn new(
        class: Option<&'a ConstantClass>,
        k: NonZeroUsize,
        exclusion: usize,
        (slots, rows): (Range<usize>, Range<usize>),
        chunk_lanes: usize,
    ) -> Option<Self> {
        let blocks = class?.blocks(k)?;
        let mut classes = reserved_vec::<RowsClass>(rows.len() as u128).ok()?;
        classes.extend(rows.clone().map(|row| blocks.row_class(row)));
        let source = Source::Triangle {
            blocks,
            lag: exclusion + 1,
            rows: classes,
        };
        Self::over(source, (slots, rows.start), chunk_lanes)
    }

    /// [`Self::new`] for a pass of an AB-join of `n_b` windows of `b`, with the class of its
    /// sides: the same skip, over the join's slots.
    pub(super) fn for_join(
        class: Option<&'a JoinClass>,
        n_b: usize,
        (slots, rows): (Range<usize>, Range<usize>),
        chunk_lanes: usize,
    ) -> Option<Self> {
        let class = class?;
        let mut classes = reserved_vec::<JoinRows>(rows.len() as u128).ok()?;
        classes.extend(rows.clone().map(|row| class.row_class(row)));
        let source = Source::Join {
            class,
            n_b,
            rows: classes,
        };
        Self::over(source, (slots, rows.start), chunk_lanes)
    }

    fn over(
        source: Source<'a>,
        (slots, first_row): (Range<usize>, usize),
        chunk_lanes: usize,
    ) -> Option<Self> {
        let chunk_lanes = chunk_lanes.max(1);
        let stride = chunk_lanes / CLASS_PART + 2;
        let parts = slots.len().div_ceil(chunk_lanes).checked_mul(stride)?;
        let mut through = reserved_vec::<usize>(parts as u128).ok()?;
        through.resize(parts, CURRENT);
        let visits = reserved_vec::<Range<usize>>(stride as u128).ok()?;
        Some(Self {
            source,
            first: slots.start,
            chunk_lanes,
            stride,
            through,
            marked: 0,
            first_row,
            visits,
        })
    }

    /// The word of the part holding `slot` in the chunk `qa..`.
    fn part(&self, qa: usize, slot: usize) -> usize {
        (qa - self.first) / self.chunk_lanes * self.stride + slot / CLASS_PART - qa / CLASS_PART
    }

    /// The slots whose lanes continue into `row`, from the pair one row before it: a self-join's
    /// lags past the exclusion zone of the row before, a join's diagonals that still have a
    /// column in `b` and a column before it.
    fn continuing(&self, row: usize) -> Range<usize> {
        match self.source {
            Source::Triangle { lag, .. } => 0..row.saturating_sub(lag),
            Source::Join { n_b, .. } => row..row + n_b - 1,
        }
    }

    /// The shape of the pass, for the catch-up.
    fn shape(&self) -> Shape {
        match self.source {
            Source::Triangle { lag, .. } => Shape::Triangle { exclusion: lag - 1 },
            Source::Join { n_b, .. } => Shape::Join { n_b },
        }
    }

    /// The slot runs of chunk `qa..qb` that `rows` compute, with every lane in them current to
    /// `rows.start` (`chunk_qt` holds the chunk's states, slot `s` at `qb - 1 - s`; `pairs` is the
    /// block's cross terms); the parts left out are marked as behind.
    pub(super) fn visits<S: Side>(
        &mut self,
        rows: &Range<usize>,
        (qa, qb): (usize, usize),
        chunk_qt: &mut [i64],
        pairs: &CrossTerms<S>,
    ) -> &[Range<usize>] {
        let at = rows.start - self.first_row;
        self.visits.clear();
        let visits = &mut self.visits;
        let mut push = |run: Range<usize>| {
            debug_assert!(
                visits.len() < visits.capacity(),
                "the runs fit their reservation"
            );
            visits.push(run);
        };
        match &self.source {
            Source::Triangle {
                blocks,
                rows: classes,
                ..
            } => {
                let class = classes[at..at + rows.len()]
                    .iter()
                    .fold(classes[at], |all, &row| all.join(row));
                blocks.unsettled_in(class, rows, &(qa..qb), CLASS_PART, &mut push);
            }
            Source::Join {
                class,
                rows: classes,
                ..
            } => {
                let rows_class = classes[at..at + rows.len()]
                    .iter()
                    .fold(classes[at], |all, &row| all.join(row));
                class.unsettled_in(rows_class, rows, &(qa..qb), CLASS_PART, &mut push);
            }
        }
        // The lanes that continue into the block's first row; later lanes start inside it.
        let alive = self.continuing(rows.start);
        let shape = self.shape();
        if self.marked > 0 {
            let mut caught = CatchUp::default();
            for run in &self.visits {
                let mut s = run.start;
                while s < run.end {
                    let end = ((s / CLASS_PART + 1) * CLASS_PART).min(run.end);
                    let word = (qa - self.first) / self.chunk_lanes * self.stride + s / CLASS_PART
                        - qa / CLASS_PART;
                    let behind = self.through[word];
                    if behind != CURRENT {
                        self.through[word] = CURRENT;
                        self.marked -= 1;
                        let continuing = s.max(alive.start)..end.min(alive.end);
                        if continuing.start < continuing.end {
                            // Lanes that held a state when the part was left out step through the
                            // rows since; lanes that started inside the skipped block restart.
                            let held = continuing.start..continuing.end;
                            let until = match shape {
                                Shape::Triangle { exclusion } => {
                                    behind.saturating_sub(exclusion + 1)
                                }
                                Shape::Join { n_b } => behind + n_b - 1,
                            };
                            let held = until.clamp(held.start, held.end);
                            let mut catch = |slots: Range<usize>, run: usize| {
                                if slots.start < slots.end {
                                    let done = catch_up_run(
                                        pairs,
                                        chunk_qt,
                                        (qb, shape),
                                        (slots, rows.start),
                                        run,
                                    );
                                    caught.resumed += done.resumed;
                                    caught.stepped += done.stepped;
                                    caught.steps += done.steps;
                                }
                            };
                            catch(continuing.start..held, rows.start - behind);
                            catch(held..continuing.end, UNSTARTED);
                        }
                    }
                    s = end;
                }
            }
            if caught != CatchUp::default() {
                crate::route::count_catch_up(&caught);
            }
        }
        // What the class settles is left behind from this block's first row.
        let mut from = qa;
        for i in 0..=self.visits.len() {
            let to = self.visits.get(i).map_or(qb, |run| run.start);
            let mut s = from;
            while s < to {
                let word = self.part(qa, s);
                if self.through[word] == CURRENT {
                    self.through[word] = rows.start;
                    self.marked += 1;
                }
                s = (s / CLASS_PART + 1) * CLASS_PART;
            }
            from = self.visits.get(i).map_or(qb, |run| run.end);
        }
        &self.visits
    }
}

/// Advances lag chunk `[qa, qb)` for every row of `block` into `accs[b]` (row `rows.start + b`),
/// every pair of it: the plain pass. `chunk_qt` is exactly this chunk's slots,
/// `live[slots − qb .. slots − qa]`, and the only `qt` it touches: chunks are independent, so this
/// is the whole of what a thread runs.
///
/// Inside a row's chunk, ascending slot index is ascending neighbor `j`, the order
/// [`pass_chunk_offering`] reads its columns in.
pub(super) fn advance_chunk(
    block: &Block,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    right: Option<&mut ColumnAcc>,
) {
    debug_assert_eq!(chunk_qt.len(), qb - qa);
    debug_assert!(qb <= block.slots);
    let computed = advance_rows(
        block,
        (qa, qb),
        chunk_qt,
        accs,
        tile,
        right,
        core::slice::from_ref(&(qa..qb)),
    );
    crate::route::count_pairs(computed);
}

/// [`advance_chunk`] leaving out the slots the constant class settles for the block (`skip`), and
/// catching up the lanes it left behind before a later block computes them.
pub(super) fn advance_chunk_skipping(
    block: &Block,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    right: Option<&mut ColumnAcc>,
    skip: &mut ClassSkip,
) {
    debug_assert_eq!(chunk_qt.len(), qb - qa);
    debug_assert!(qb <= block.slots);
    let visits = skip.visits(&block.rows, (qa, qb), chunk_qt, &block.pairs);
    let computed = compute_visits(block, (qa, qb), chunk_qt, accs, tile, right, visits);
    crate::route::count_pairs(computed);
}

/// [`advance_chunk`]'s rows over the slots of `qa..qb` that `visits` holds: inlined once for the
/// whole chunk, the plain pass, and once for the visits the one-loop route's ladder leaves.
#[inline(always)]
fn advance_rows(
    block: &Block,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    mut right: Option<&mut ColumnAcc>,
    visits: &[core::ops::Range<usize>],
) -> usize {
    let Block {
        cols,
        base,
        m,
        shape,
        ..
    } = *block;
    let mut computed = 0;
    let resume_period = crate::route::resume_period();
    for (b, row) in block.rows.clone().enumerate() {
        let (first, count) = shape.row_slots(row);
        let Some(span) = row_chunk(first, count, qa, qb) else {
            continue;
        };
        let acc = &mut accs[b];
        let scalars = block.scalars[b];
        let resume_forced = resume_period != 0 && row.is_multiple_of(resume_period);
        if let Some(fresh) = span.fresh {
            computed += 1;
            crate::route::record_pairs(row, 0..1);
            debug_assert_eq!(base, 0, "window 0 is in the columns");
            let qt0 = block.pairs.at(row, 0);
            chunk_qt[qb - 1 - fresh] = qt0;
            let neighbor = |_: usize| window_at_narrow(cols, 0);
            let started = acc.start_lane(
                qt0,
                &ChunkCols {
                    enter: &[],
                    leave: &[],
                    sum: &cols.sum[..1],
                    r: &cols.r[..1],
                    bias: &cols.bias[..1],
                    first: 0,
                    neighbor: &neighbor,
                },
                scalars,
            );
            if let (Some(part), Some(lane)) = (right.as_deref_mut(), started) {
                part.offer_lane::<RankRow>(0, acc.id(row), lane, || {
                    crate::columns::variance(&neighbor(0))
                });
            }
        }
        for (lo, hi_j) in span.neighbors(visits) {
            computed += hi_j - lo;
            crate::route::record_pairs(row, lo..hi_j);
            let (l, h) = (lo - base, hi_j - base);
            let neighbor = |lane: usize| window_at_narrow(cols, l + lane);
            let states = &mut chunk_qt[qb - span.slot_of(lo)..][..hi_j - lo];
            if resume_forced {
                block.pairs.force_resume(row, (row, lo), states);
            }
            pass_chunk_offering(
                states,
                ChunkCols {
                    enter: &cols.ticks[l + m - 1..h + m - 1],
                    leave: &cols.ticks[l - 1..h - 1],
                    sum: &cols.sum[l..h],
                    r: &cols.r[l..h],
                    bias: &cols.bias[l..h],
                    first: lo,
                    neighbor: &neighbor,
                },
                scalars,
                acc,
                tile,
                right.as_deref_mut().map(|part| (part, row)),
            );
        }
    }
    computed
}

/// One block's pairs over exactly the slots `visits` holds inside the chunk `qa..qb`, as
/// [`advance_chunk`] computes them once its gate has chosen the visits, and the pairs computed:
/// the leaf compute of a walk that chooses its visits itself. `visits` are ascending, disjoint
/// slot ranges inside the chunk; `chunk_qt` holds the chunk's slots as for [`advance_chunk`], and
/// the states of the slots the visits hold must be current at the block's first row (a caller that
/// left a slot's lane behind resumes it first). Nothing is gated, resumed or counted here.
#[inline]
pub(super) fn compute_visits(
    block: &Block,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    right: Option<&mut ColumnAcc>,
    visits: &[core::ops::Range<usize>],
) -> usize {
    debug_assert_eq!(chunk_qt.len(), qb - qa);
    debug_assert!(qb <= block.slots);
    debug_assert!(
        visits
            .iter()
            .all(|v| qa <= v.start && v.start <= v.end && v.end <= qb)
            && visits.windows(2).all(|w| w[0].end <= w[1].start),
        "visits are ascending, disjoint ranges inside the chunk"
    );
    advance_rows(block, (qa, qb), chunk_qt, accs, tile, right, visits)
}

/// [`compute_visits`] on the exact i64 arithmetic, with unscaled `qt` slots: the leaf compute of
/// a walk that chooses its visits itself, on a wide tile.
#[inline]
pub(super) fn compute_visits_wide(
    block: &WideBlock,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    right: Option<&mut ColumnAcc>,
    visits: &[core::ops::Range<usize>],
) -> usize {
    debug_assert_eq!(chunk_qt.len(), qb - qa);
    debug_assert!(
        visits
            .iter()
            .all(|v| qa <= v.start && v.start <= v.end && v.end <= qb)
            && visits.windows(2).all(|w| w[0].end <= w[1].start),
        "visits are ascending, disjoint ranges inside the chunk"
    );
    advance_rows_wide(block, (qa, qb), chunk_qt, accs, tile, right, visits)
}

/// The exact i64 columns a wide tile reads its neighbors from, each row's scalars and the `QT`
/// of its pairs: absolute, nothing narrowed.
pub(super) struct WideBlock<'a> {
    pub(super) cols: &'a Columns<'a>,
    pub(super) row_scalars: &'a dyn Fn(usize) -> WideRow,
    pub(super) pairs: CrossTerms<&'a [i64]>,
    pub(super) rows: core::ops::Range<usize>,
    pub(super) m: usize,
    pub(super) shape: Shape,
}

/// [`advance_chunk`] on the exact i64 arithmetic, with unscaled `qt` slots.
pub(super) fn advance_chunk_wide(
    block: &WideBlock,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    right: Option<&mut ColumnAcc>,
) {
    debug_assert_eq!(chunk_qt.len(), qb - qa);
    let computed = advance_rows_wide(
        block,
        (qa, qb),
        chunk_qt,
        accs,
        tile,
        right,
        core::slice::from_ref(&(qa..qb)),
    );
    crate::route::count_pairs(computed);
}

/// [`advance_chunk_skipping`] on the exact i64 arithmetic.
pub(super) fn advance_chunk_wide_skipping(
    block: &WideBlock,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    right: Option<&mut ColumnAcc>,
    skip: &mut ClassSkip,
) {
    debug_assert_eq!(chunk_qt.len(), qb - qa);
    let visits = skip.visits(&block.rows, (qa, qb), chunk_qt, &block.pairs);
    let computed = compute_visits_wide(block, (qa, qb), chunk_qt, accs, tile, right, visits);
    crate::route::count_pairs(computed);
}

/// [`advance_rows`] on the exact i64 arithmetic.
#[inline(always)]
fn advance_rows_wide(
    block: &WideBlock,
    (qa, qb): (usize, usize),
    chunk_qt: &mut [i64],
    accs: &mut [RowAcc],
    tile: &mut [f64],
    mut right: Option<&mut ColumnAcc>,
    visits: &[core::ops::Range<usize>],
) -> usize {
    let WideBlock { cols, m, shape, .. } = *block;
    let mut computed = 0;
    let window_of = |j: usize| {
        crate::columns::window_of(cols.sum[j], cols.r[j], cols.bias[j], cols.variance[j])
    };
    let resume_period = crate::route::resume_period();
    for (b, row) in block.rows.clone().enumerate() {
        let (first, count) = shape.row_slots(row);
        let Some(span) = row_chunk(first, count, qa, qb) else {
            continue;
        };
        let acc = &mut accs[b];
        let scalars = (block.row_scalars)(row);
        let resume_forced = resume_period != 0 && row.is_multiple_of(resume_period);
        if let Some(fresh) = span.fresh {
            computed += 1;
            crate::route::record_pairs(row, 0..1);
            let qt0 = block.pairs.at(row, 0);
            chunk_qt[qb - 1 - fresh] = qt0;
            let neighbor = |_: usize| window_of(0);
            let started = acc.start_lane_wide(
                qt0,
                &WideChunkCols {
                    enter: &[],
                    leave: &[],
                    sum: &cols.sum[..1],
                    r: &cols.r[..1],
                    bias: &cols.bias[..1],
                    first: 0,
                    neighbor: &neighbor,
                },
                scalars,
                m,
            );
            if let (Some(part), Some(lane)) = (right.as_deref_mut(), started) {
                part.offer_lane::<WideRank>(0, acc.id(row), lane, || {
                    crate::columns::variance(&neighbor(0))
                });
            }
        }
        for (lo, hi_j) in span.neighbors(visits) {
            computed += hi_j - lo;
            crate::route::record_pairs(row, lo..hi_j);
            let neighbor = |lane: usize| window_of(lo + lane);
            let states = &mut chunk_qt[qb - span.slot_of(lo)..][..hi_j - lo];
            if resume_forced {
                block.pairs.force_resume(row, (row, lo), states);
            }
            pass_chunk_wide(
                states,
                WideChunkCols {
                    enter: &cols.ticks[lo + m - 1..hi_j + m - 1],
                    leave: &cols.ticks[lo - 1..hi_j - 1],
                    sum: &cols.sum[lo..hi_j],
                    r: &cols.r[lo..hi_j],
                    bias: &cols.bias[lo..hi_j],
                    first: lo,
                    neighbor: &neighbor,
                },
                scalars,
                m,
                acc,
                tile,
            );
            if let Some(part) = right.as_deref_mut().filter(|_| acc.ranks()) {
                let len = hi_j - lo;
                let qt = &chunk_qt[qb - span.slot_of(lo)..][..len];
                part.offer_chunk::<WideRank>(
                    lo,
                    acc.id(row),
                    &tile[..len],
                    |lane| {
                        centered_cross_fast(qt[lane], m as i64, scalars.sum, cols.sum[lo + lane])
                    },
                    |lane| crate::columns::variance(&window_of(lo + lane)),
                );
            }
        }
    }
    computed
}

/// Starts `rows`' accumulators and returns their scalars: shared by every chunk of the block.
fn block_scalars(
    cols: &NarrowColumns,
    windows: &[Window],
    rows: &core::ops::Range<usize>,
    m: usize,
) -> Vec<RowScalars> {
    rows.clone()
        .map(|row| narrow_row_scalars(cols, &windows[row], row, m, cols.sum[row]))
        .collect()
}

/// The first `none_rows` rows have no neighbor (a row at or inside the exclusion zone); each is
/// `none()`.
pub(super) fn push_none_rows<R>(
    none_rows: usize,
    results: &mut Vec<(u64, R)>,
    mut none: impl FnMut() -> R,
) {
    results.extend((0..none_rows).map(|row| (row as u64, none())));
}

/// The narrow path's tiled engine over the lag-ordered `qt` (the growing stream's own
/// [`FrontGrowBuffer`], one slot per lag, never shifted): rows go in blocks of `block_rows`; per
/// block, every lag chunk of `chunk_lanes` slots is advanced for every row of the block through
/// [`advance_chunk`] into that row's [`RowAcc`] (the chunk's `qt` stays hot across the block), and
/// each row is finished into a [`Nearest`] once every chunk has run.
///
/// **Chunks are independent.** Chunk `[qa, qb)` holds the same lags for every block, reads only
/// its own slots, and touches no neighbor chunk's, so chunks may run in any order (or in
/// parallel, [`left_profile_threads_with`](super::left::left_profile_threads_with)) with identical results.
///
/// A Constant or non-finite row goes through the same pass ([`RowAcc::begin`] reads its kind), so
/// there is no separate path for it. Appends one `(row, Entry)` per window, in order.
#[allow(clippy::too_many_arguments)]
pub(super) fn tiled_narrow_profile(
    cols: &NarrowColumns,
    windows: &[Window],
    m: usize,
    exclusion: usize,
    chunk_lanes: usize,
    block_rows: usize,
    k: NonZeroUsize,
    class: Option<&ConstantClass>,
    results: &mut Vec<(u64, Entry)>,
) {
    let mut qt = FrontGrowBuffer::new();
    let mut tile = vec![0.0; chunk_lanes.min(windows.len())];
    let mut accs: Vec<RowAcc> = (0..block_rows).map(|_| RowAcc::new()).collect();

    push_none_rows(windows.len().min(exclusion + 1), results, || Entry::NONE);

    // The slots of the whole series: lag `exclusion + 1 + s` for `s` below the window count.
    let mut skip = ClassSkip::new(
        class,
        k,
        exclusion,
        (0..windows.len(), exclusion + 1..windows.len()),
        chunk_lanes,
    );
    let mut first = exclusion + 1;
    while first < windows.len() {
        let rows = first..(first + block_rows).min(windows.len());
        for _ in rows.clone() {
            qt.grow_front();
        }
        let live = qt.as_mut_slice();
        let slots = live.len();

        let scalars = block_scalars(cols, windows, &rows, m);
        for (acc, row) in accs.iter_mut().zip(rows.clone()) {
            acc.begin(&windows[row], k, windows.len());
            if let Some(class) = class {
                acc.take_class(class.row(row, k));
            }
        }
        let block = Block {
            cols,
            base: 0,
            pairs: CrossTerms::of(cols.ticks, m),
            rows: rows.clone(),
            scalars: &scalars,
            m,
            shape: Shape::Triangle { exclusion },
            slots,
        };
        for qa in (0..slots).step_by(chunk_lanes) {
            let qb = (qa + chunk_lanes).min(slots);
            let chunk_qt = &mut live[slots - qb..slots - qa];
            match skip.as_mut() {
                Some(skip) => advance_chunk_skipping(
                    &block,
                    (qa, qb),
                    chunk_qt,
                    &mut accs,
                    &mut tile,
                    None,
                    skip,
                ),
                None => advance_chunk(&block, (qa, qb), chunk_qt, &mut accs, &mut tile, None),
            }
        }

        for (acc, row) in accs.iter_mut().zip(rows.clone()) {
            let mut nearest = Nearest::with_k(&windows[row], m, k, windows.len());
            #[cfg(test)]
            VISITS.with(|v| v.borrow_mut().push((row, acc.visits.clone())));
            acc.finish(&mut nearest);
            nearest.push_kept(row as u64, k, results);
        }
        first = rows.end;
    }
}

/// What `rows` of a band rank into once every tile has merged: each row's entry, in order.
pub(super) fn finish_rows(
    windows: &[Window],
    m: usize,
    first: usize,
    k: NonZeroUsize,
    accs: &mut [RowAcc],
    results: &mut Vec<(u64, Entry)>,
) {
    for (r, acc) in accs.iter_mut().enumerate() {
        let row = first + r;
        let mut nearest = Nearest::with_k(&windows[row], m, k, windows.len());
        #[cfg(test)]
        VISITS.with(|v| v.borrow_mut().push((row, acc.visits.clone())));
        acc.finish(&mut nearest);
        nearest.push_kept(row as u64, k, results);
    }
}

#[cfg(test)]
mod tests;
