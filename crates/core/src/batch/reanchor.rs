//! Tiles whose rows or columns drift past what one anchor covers, run narrow anyway: the tile's
//! rows are cut into segments, each segment's lanes into groups, and every cell (a segment by a
//! group) takes one anchor per side. A diagonal's carried `m·QT` moves from one cell's anchors to
//! the next's by [`narrow::reanchor_pair`], a constant number of operations from the two windows'
//! sums, instead of being restarted from `m` products.
//! Every cell fits one anchor on each side, so every intermediate stays within
//! [`narrow::Narrow::fits`]'s bound; a tile with a cell that cannot fit even at the smallest cut
//! takes the exact i64 pass.

use super::pass::{BLOCK_ROWS, Block, Shape, advance_chunk};
use super::plan::{SlotRuns, Tile};
use super::tile::{
    Scratch, close_tile, for_each_chunk, join_start_column, open_tile, triangle_start_pair,
};
use super::two_sided::RightBook;
use crate::cross_term::CrossTerms;
use crate::kernel_neon::{Conversion, GROUP_LANES};
use crate::narrow::{self, Anchor};
use crate::row::{Columns, NarrowColumns, narrow_row_scalars};
use crate::stats::{ConstantClass, Window};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Range;

/// The most rows a segment spans: a cell's column ticks cover its lanes plus its rows, so short
/// segments leave room for wide lane groups, and a group is the unit a row pass is called on.
const SEGMENT_ROWS: usize = 256;

/// What a tile reads, per side: the rows' ticks and window sums with their windows, and the
/// columns the neighbors come from.
pub(super) struct TileSides<'a> {
    pub(super) rows: &'a Columns<'a>,
    pub(super) row_windows: &'a [Window],
    pub(super) cols: &'a Columns<'a>,
    /// How the row kernel converts the cross terms between the two sides.
    pub(super) conversion: Conversion,
    /// The self-join's constant class, when it has one.
    pub(super) class: Option<&'a ConstantClass>,
}

/// The column windows a cell's lane group reads when any of its lanes is live in its rows, and the
/// lowest and highest tick of them with the ticks that follow.
struct ColSpan {
    windows: Range<usize>,
    span: (i64, i64),
}

/// How a tile is cut: the ticks it reads and the span one anchor covers.
struct Planner<'a> {
    row_ticks: &'a [i64],
    col_ticks: &'a [i64],
    m: usize,
    shape: Shape,
    n_cols: usize,
    rows: Range<usize>,
    span_limit: i64,
    /// The rows per segment, and with `forced` the lanes per group.
    cap: usize,
    forced: bool,
}

fn span_of(ticks: &[i64]) -> (i64, i64) {
    ticks
        .iter()
        .fold((i64::MAX, i64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)))
}

impl<'a> Planner<'a> {
    fn new(
        row_ticks: &'a [i64],
        col_ticks: &'a [i64],
        m: NonZeroUsize,
        shape: Shape,
        tile: &Tile,
    ) -> Self {
        let mu = m.get();
        let forced = crate::route::forced_cell();
        Self {
            row_ticks,
            col_ticks,
            m: mu,
            shape,
            n_cols: (col_ticks.len() + 1).saturating_sub(mu),
            rows: tile.rows.clone(),
            span_limit: 2 * (i64::from(i32::MAX) / mu as i64),
            cap: if forced == 0 { SEGMENT_ROWS } else { forced },
            forced: forced != 0,
        }
    }

    fn fits(&self, (lo, hi): (i64, i64)) -> bool {
        hi.saturating_sub(lo) <= self.span_limit
    }

    /// The span of the row ticks segment `a..b` reads, when one anchor covers it.
    fn row_span(&self, a: usize, b: usize) -> Option<(i64, i64)> {
        let span = span_of(&self.row_ticks[a.saturating_sub(1)..b - 1 + self.m]);
        self.fits(span).then_some(span)
    }

    /// The column windows lane group `group` reads over rows `a..b`, `None` when none of its
    /// lanes is live in them. Slot `s` pairs row `r` with window `r + offset(s)`; its state and
    /// its leaving tick sit one window before the first.
    fn live_windows(&self, group: (usize, usize), a: usize, b: usize) -> Option<Range<usize>> {
        let (off_lo, off_hi) = (self.shape.offset(group.1 - 1), self.shape.offset(group.0));
        let j_min = a as i64 + off_lo;
        let j_max = b as i64 - 1 + off_hi;
        (j_max >= 0 && j_min < self.n_cols as i64)
            .then(|| (j_min - 1).max(0) as usize..((j_max + 1) as usize).min(self.n_cols))
    }

    /// Cuts lane group `group` over rows `a..b` into the widest groups, by repeated halving, whose
    /// column ticks each fit one anchor, handing them to `keep` in ascending slots; `false` when
    /// some part cannot fit even at the narrowest group.
    fn plan_group(
        &self,
        group: (usize, usize),
        (a, b): (usize, usize),
        keep: &mut impl FnMut((usize, usize), Option<ColSpan>),
    ) -> bool {
        let len = group.1 - group.0;
        let (forced_split, quantum) = if self.forced {
            (len > self.cap, 1)
        } else {
            (false, GROUP_LANES)
        };
        if !forced_split {
            match self.live_windows(group, a, b) {
                None => {
                    keep(group, None);
                    return true;
                }
                Some(windows) => {
                    let span = span_of(&self.col_ticks[windows.start..windows.end + self.m - 1]);
                    if self.fits(span) {
                        keep(group, Some(ColSpan { windows, span }));
                        return true;
                    }
                }
            }
        }
        let mid = group.0 + (len / 2).div_ceil(quantum) * quantum;
        if mid >= group.1 || mid == group.0 {
            return false;
        }
        self.plan_group((group.0, mid), (a, b), keep)
            && self.plan_group((mid, group.1), (a, b), keep)
    }

    /// The end of the segment from row `a`: at most [`Self::cap`] rows, fewer when the rows' ticks
    /// or some lane chunk's cells would not fit one anchor each; `None` when not even one row
    /// does.
    fn segment_end(&self, a: usize, chunks: &Chunks) -> Option<usize> {
        let mut b = (a + self.cap).min(self.rows.end);
        loop {
            if self.row_span(a, b).is_some()
                && chunks.all(|chunk| self.plan_group(chunk, (a, b), &mut |_, _| {}))
            {
                return Some(b);
            }
            if b - a == 1 {
                return None;
            }
            b = a + (b - a).div_ceil(2);
        }
    }
}

/// `chunk_lanes` as whole kernel groups when it holds at least one, so no chunk but a tile's last
/// ends in a part group.
fn whole_groups(chunk_lanes: usize) -> usize {
    match chunk_lanes / GROUP_LANES {
        0 => chunk_lanes,
        groups => groups * GROUP_LANES,
    }
}

/// The lane chunks a tile walks.
struct Chunks<'a> {
    span: (usize, usize),
    lanes: usize,
    runs: Option<&'a SlotRuns>,
}

impl<'a> Chunks<'a> {
    fn new(tile: &Tile, chunk_lanes: usize, runs: Option<&'a SlotRuns>) -> Self {
        Chunks {
            span: (tile.qa, tile.qb),
            lanes: whole_groups(chunk_lanes),
            runs,
        }
    }

    /// Whether `test` holds for every chunk, in ascending slots.
    fn all(&self, mut test: impl FnMut((usize, usize)) -> bool) -> bool {
        let mut holds = true;
        for_each_chunk(self.span, self.lanes, self.runs, |sa, sb| {
            holds = holds && test((sa, sb));
        });
        holds
    }
}

/// Whether every cell of the tile fits one anchor per side, pushing the end of each segment to
/// `ends` when given.
fn plan_tile(planner: &Planner, chunks: &Chunks, mut ends: Option<&mut Vec<usize>>) -> bool {
    if let Some(ends) = ends.as_deref_mut() {
        ends.clear();
    }
    let mut a = planner.rows.start;
    while a < planner.rows.end {
        let Some(b) = planner.segment_end(a, chunks) else {
            return false;
        };
        if let Some(ends) = ends.as_deref_mut() {
            ends.push(b);
        }
        a = b;
    }
    true
}

/// Whether a triangle or join tile fits cells over `row_ticks` and `col_ticks`: the exact pass is
/// not needed. Mirrors [`drifting_tile_partials`]'s own cut.
pub(super) fn tile_fits_cells(
    row_ticks: &[i64],
    col_ticks: &[i64],
    m: NonZeroUsize,
    shape: Shape,
    tile: &Tile,
    chunk_lanes: usize,
) -> bool {
    if crate::route::forced_wide() {
        return false;
    }
    let planner = Planner::new(row_ticks, col_ticks, m, shape, tile);
    plan_tile(&planner, &Chunks::new(tile, chunk_lanes, None), None)
}

/// The state a slot starts from, as its `m·QT` over the first cell's narrowed windows: before the
/// tile's first row, the pair one row back that the slot carries into it.
fn start_state(shape: Shape, r0: usize, slot: usize, pairs: &CrossTerms<&[i32]>) -> Option<i64> {
    match shape {
        Shape::Triangle { exclusion } => {
            triangle_start_pair(exclusion, r0, slot).map(|(i, j)| pairs.at(i, j))
        }
        Shape::Join { n_b } => join_start_column(n_b, r0, slot).map(|j| match r0 {
            0 => pairs.before_first_row(j),
            _ => pairs.at(r0 - 1, j - 1),
        }),
    }
}

/// Moves the carried `m·QT` of every live lane of `group` from the anchors of the cells before row
/// `a` (`old_row`, and each lane's own column anchor in `lane_anchors`) to this cell's: slot `s`
/// carries the pair `(a − 1, a − 1 + offset(s))`.
#[allow(clippy::too_many_arguments)]
fn reanchor_lanes(
    group_qt: &mut [i64],
    lane_anchors: &[i64],
    group: (usize, usize),
    a: usize,
    sides: &TileSides,
    m: NonZeroUsize,
    shape: Shape,
    old_row: Anchor,
    (new_row, new_col): (Anchor, Anchor),
) {
    let mu = m.get() as i64;
    let delta_row = new_row.tick - old_row.tick;
    let row_sum = sides.rows.sum[a - 1].wrapping_sub(mu.wrapping_mul(old_row.tick));
    let n_cols = sides.cols.sum.len() as i64;
    for (lane, slot) in (group.0..group.1).enumerate() {
        let j = (a - 1) as i64 + shape.offset(slot);
        if j < 0 || j >= n_cols {
            continue;
        }
        let old_col = lane_anchors[lane];
        let col_sum = sides.cols.sum[j as usize].wrapping_sub(mu.wrapping_mul(old_col));
        let state = &mut group_qt[group.1 - 1 - slot];
        *state = narrow::reanchor_pair(
            m,
            delta_row,
            new_col.tick - old_col,
            *state,
            row_sum,
            col_sum,
        );
    }
}

/// One tile's partials on the narrow arithmetic when its rows or columns drift past one anchor:
/// the rows are cut into segments, each segment's lane chunks into groups, and every cell takes an
/// anchor per side, the carried `m·QT` re-anchored between segments. Returns `false`, having
/// touched nothing, when some cell cannot fit even at the finest cut: the caller runs the exact
/// pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn drifting_tile_partials(
    sides: &TileSides,
    m: NonZeroUsize,
    shape: Shape,
    chunk_lanes: usize,
    tile: &Tile,
    scratch: &mut Scratch,
    k: NonZeroUsize,
    right: Option<&RightBook>,
    runs: Option<&SlotRuns>,
) -> bool {
    if crate::route::forced_wide() {
        return false;
    }
    let planner = Planner::new(sides.rows.ticks, sides.cols.ticks, m, shape, tile);
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
        ends,
        lane_anchors,
    } = scratch;
    let chunks = Chunks::new(tile, chunk_lanes, runs);
    if !plan_tile(&planner, &chunks, Some(ends)) {
        return false;
    }

    let mu = m.get();
    let (qa, qb) = (tile.qa, tile.qb);
    let tile_qt = &mut qt[..qb - qa];
    lane_anchors.clear();
    lane_anchors.resize(qb - qa, 0);
    let windows = sides.row_windows;
    let mut right = open_tile(
        shape,
        tile,
        (windows, sides.class),
        k,
        accs,
        right.map(|book| (book, part)),
    );
    let r0 = tile.rows.start;
    let mut previous_row: Option<Anchor> = None;
    let mut a = r0;
    for &b in ends.iter() {
        let row_span = planner.row_span(a, b).expect("planned segments fit");
        let row_anchor = Anchor::fitting(row_span.0, row_span.1, m, (a as u64) << 1)
            .expect("a planned span fits one anchor");
        narrow::narrow_ticks_into(
            row_anchor,
            m,
            &sides.rows.ticks[a.saturating_sub(1)..b - 1 + mu],
            rt,
        );
        if a == 0 {
            rt.insert(0, 0);
        }
        narrow::narrow_sums_into(row_anchor, m, &sides.rows.sum[a..b], rs);
        let row_cols = NarrowColumns {
            ticks: rt,
            sum: rs,
            r: &[],
            bias: &[],
            variance: &[],
            conversion: sides.conversion,
        };
        scalars.clear();
        scalars.extend(
            (a..b).map(|row| {
                narrow_row_scalars(&row_cols, &windows[row], row + 1 - a, mu, rs[row - a])
            }),
        );
        let rt: &[i32] = rt;
        let scalars: &[_] = scalars;
        chunks.all(|chunk| {
            planner.plan_group(chunk, (a, b), &mut |group, cols| {
                let Some(ColSpan {
                    windows: col_windows,
                    span,
                }) = cols
                else {
                    return;
                };
                let col_anchor = Anchor::fitting(
                    span.0,
                    span.1,
                    m,
                    (a as u64) << 1 | 1 | (group.0 as u64) << 32,
                )
                .expect("a planned span fits one anchor");
                let col_base = col_windows.start;
                narrow::narrow_ticks_into(
                    col_anchor,
                    m,
                    &sides.cols.ticks[col_base..col_windows.end + mu - 1],
                    ct,
                );
                narrow::narrow_sums_into(col_anchor, m, &sides.cols.sum[col_windows.clone()], cs);
                let group_qt = &mut tile_qt[qb - group.1..qb - group.0];
                let lanes = &mut lane_anchors[group.0 - qa..group.1 - qa];
                let row_frame = if a == 0 { &rt[1..] } else { rt };
                let pairs = CrossTerms::new(row_frame, a.saturating_sub(1), &ct[..], col_base, mu);
                match previous_row {
                    Some(old_row) => reanchor_lanes(
                        group_qt,
                        lanes,
                        group,
                        a,
                        sides,
                        m,
                        shape,
                        old_row,
                        (row_anchor, col_anchor),
                    ),
                    None => {
                        for slot in group.0..group.1 {
                            if let Some(state) = start_state(shape, r0, slot, &pairs) {
                                group_qt[group.1 - 1 - slot] = state;
                            }
                        }
                    }
                }
                lanes.fill(col_anchor.tick);
                let col_cols = NarrowColumns {
                    ticks: ct,
                    sum: cs,
                    r: &sides.cols.r[col_windows.clone()],
                    bias: &sides.cols.bias[col_windows.clone()],
                    variance: &sides.cols.variance[col_windows],
                    conversion: sides.conversion,
                };
                for off in (0..b - a).step_by(BLOCK_ROWS) {
                    let end = (off + BLOCK_ROWS).min(b - a);
                    let block = Block {
                        cols: &col_cols,
                        base: col_base,
                        pairs,
                        rows: a + off..a + end,
                        scalars: &scalars[off..end],
                        m: mu,
                        shape,
                        slots: qb,
                    };
                    let part = right.as_mut().map(|(_, part)| &mut **part);
                    advance_chunk(
                        &block,
                        group,
                        group_qt,
                        &mut accs[a - r0 + off..a - r0 + end],
                        rho,
                        part,
                    );
                    if let Some((_, part)) = right.as_mut() {
                        part.compact(windows);
                    }
                }
            })
        });
        if b < tile.rows.end {
            crate::route::count_cell_boundary(planner.forced);
        }
        previous_row = Some(row_anchor);
        a = b;
    }
    close_tile(right, windows, None, (&tile.rows, accs));
    true
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod tests;
