//! Per-band anchors for the one-loop route. A series whose ticks span more than one anchor covers
//! cannot be narrowed whole, but a band's rows and a tile's columns usually span much less: the
//! band narrows its own rows to an anchor of their own and the tile's columns to another, as the
//! plain tile does, and runs the same band driver over them. The ladder reads the windows'
//! PAA coefficients, which no anchor touches, so it is reused as it is.
//!
//! **Lanes between anchors.** A lane's state is an `m·QT` measured from the anchors of the band
//! that left it, so the band below, whose anchors may differ, moves each lane that is current
//! through its first row to its own anchors, exactly ([`crate::narrow::reanchor_pair`], the move
//! the plain tiled pass's re-anchored cells make), before anything reads it. A lane a skip left behind
//! needs no move: the band below cannot step it through rows its narrowed ticks do not hold, so
//! its catch-up resumes it from the exact dot product in the band's own anchors, whatever the
//! state it carried. A tile some anchor cannot cover (its rows or columns span more ticks than one
//! anchor holds, whether one window does or the tile's whole run of windows) runs the same driver
//! on the exact i64 ticks ([`super::wide`]), its lanes handed between wide bands as they are and
//! restarted at a change of width, which is exact. No tile the route takes runs the plain cell pass
//! for want of an anchor.

use super::band::Layout;
use super::band::{self, BandInput, TileScratch};
use super::dense::Boxes;
use super::store::{Carry, LaneStore};
use super::wide;
use crate::batch::pass::Shape;
use crate::batch::plan::Tile;
use crate::batch::quantized::Quantized;
use crate::batch::survivor::{SeedInput, SharedFloors, seed_floors};
use crate::batch::tile::{
    NarrowTile, RanWide, Scratch, anchors_of, narrow_tile_partials, tile_partials,
    triangle_start_pair,
};
use crate::batch::two_sided::RightBook;
use crate::bound_paa::PaaSides;
use crate::cross_term::{CrossTerms, UNSTARTED};
use crate::narrow;
use crate::row::{NarrowColumns, centered_cross_fast, narrow_row_scalars};
use core::num::NonZeroUsize;

/// The anchors a band's lanes are measured from: the row side's tick and the column side's. A run
/// with one anchor for the whole series has one frame for every band, `Frame::default()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Frame {
    pub(super) row: i64,
    pub(super) col: i64,
}

/// Where a band's lanes sit when it takes them from the band above.
#[derive(Clone, Debug)]
pub(super) struct Lanes<'a> {
    /// The window length.
    pub(super) m: usize,
    /// The shape of the run: slot `s` of row `r` meets column `r + shape.offset(s)`.
    pub(super) shape: Shape,
    /// The band's first row.
    pub(super) first_row: usize,
    /// The slots of the band's tile that are live in its first row, ascending.
    pub(super) slots: core::ops::Range<usize>,
    /// The tile's first slot, and one past the last slot the carry holds.
    pub(super) span: (usize, usize),
    /// Each window's own sum of ticks, absolute, of the row side and of the column side (one
    /// slice twice for a self-join); `None` when the run has no anchors to move.
    pub(super) sums: Option<(&'a [i64], &'a [i64])>,
}

/// Moves the lanes of `carry` that are current through the band's first row from the anchors of
/// the carry's frame to `to`, and labels the carry with `to`; returns how many it moved. Every
/// other lane is left as it is (see the module note). A lane whose windows' sums are not at hand
/// is reset to a lane that holds no state, which the catch-up starts from the exact dot product.
pub(super) fn move_lanes(carry: &mut Carry, to: Frame, at: &Lanes<'_>) -> usize {
    let from = carry.frame;
    carry.frame = to;
    if from == to {
        return 0;
    }
    let (qa, top) = at.span;
    let m = at.m as i64;
    let mut moved = 0;
    for slot in at.slots.clone() {
        let Some(through) = carry.through.get_mut(slot - qa) else {
            continue;
        };
        if *through != at.first_row {
            continue;
        }
        // The pair the lane carries into the band's first row: row `first_row - 1` with its
        // column on the lane's diagonal.
        let row = at.first_row - 1;
        let sums = at.sums.and_then(|(row_sums, col_sums)| {
            let column = usize::try_from(row as i64 + at.shape.offset(slot)).ok()?;
            Some((*row_sums.get(row)?, *col_sums.get(column)?))
        });
        let (Some((row_sum, col_sum)), Some(state)) = (sums, carry.qt.get_mut(top - 1 - slot))
        else {
            *through = UNSTARTED;
            continue;
        };
        *state = narrow::reanchor_pair(
            NonZeroUsize::new(at.m).unwrap_or(NonZeroUsize::MIN),
            to.row.wrapping_sub(from.row),
            to.col.wrapping_sub(from.col),
            *state,
            row_sum.wrapping_sub(m.wrapping_mul(from.row)),
            col_sum.wrapping_sub(m.wrapping_mul(from.col)),
        );
        moved += 1;
    }
    moved
}

/// What a run with per-band anchors reads, beside the tile.
pub(super) struct Anchored<'a, const K: usize> {
    pub(super) quantized: &'a Quantized,
    pub(super) m: NonZeroUsize,
    pub(super) k: NonZeroUsize,
    pub(super) layout: Layout,
    pub(super) boxes: &'a Boxes<K>,
    pub(super) floors: Option<&'a SharedFloors>,
    pub(super) store: &'a LaneStore,
    pub(super) book: Option<&'a RightBook>,
    pub(super) sides: PaaSides<'a>,
}

impl<const K: usize> Anchored<'_, K> {
    /// One tile: its rows and its columns narrowed to anchors of their own, then the band driver
    /// over them. A tile that no pair of anchors covers, or that the driver declines, runs as the
    /// tiled path runs it. Returns whether the tile ran the exact i64 pass.
    pub(super) fn tile(&self, tile: &Tile, scratch: &mut Scratch, chunk_lanes: usize) -> RanWide {
        let q = self.quantized;
        let mu = self.m.get();
        let exclusion = crate::exclusion::zone(mu);
        let shape = Shape::Triangle { exclusion };
        let (r0, r1) = (tile.rows.start, tile.rows.end);
        let lag = exclusion + 1;
        // A tile the cut makes is always valid; one that is not runs plain, never through a
        // bound this driver would trip over.
        let valid = tile.qa < tile.qb
            && r0 > exclusion
            && r1 > lag + tile.qa
            && r1 <= q.windows.len()
            && q.sum.len() >= r1;
        let columns = if valid { shape.columns(tile) } else { 0..0 };
        let (cb, col_end) = (columns.start, columns.end);
        let ticks = (|| {
            let rows = q.ticks.get(r0.checked_sub(1)?..(r1 - 1).checked_add(mu)?)?;
            let cols = q
                .ticks
                .get(cb..(col_end.checked_sub(1)?).checked_add(mu)?)?;
            Some((rows, cols))
        })();
        let anchors = ticks
            .filter(|_| valid && cb < col_end)
            .and_then(|(rows, cols)| anchors_of(rows, cols, self.m))
            .filter(|_| crate::route::forced_cell() == 0);
        let Some((row_anchor, col_anchor)) = anchors else {
            // A tile that no anchor covers runs the band driver over the exact i64 ticks; only a
            // tile the cut did not make (never one the planner cuts) runs plain.
            if valid && cb < col_end {
                return wide::tile(self, tile, scratch);
            }
            return tile_partials(
                q,
                self.m,
                chunk_lanes,
                tile,
                scratch,
                self.k,
                self.book,
                None,
            );
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
        narrow::narrow_ticks_into(row_anchor, self.m, &q.ticks[r0 - 1..r1 - 1 + mu], rt);
        narrow::narrow_sums_into(row_anchor, self.m, &q.sum[r0..r1], rs);
        narrow::narrow_ticks_into(col_anchor, self.m, &q.ticks[cb..col_end - 1 + mu], ct);
        narrow::narrow_sums_into(col_anchor, self.m, &q.sum[cb..col_end], cs);
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
        let narrow = NarrowTile {
            cols: &col_cols,
            base: cb,
            scalars,
            pairs,
            start_qt: &|slot| triangle_start_pair(exclusion, r0, slot).map(|(i, j)| pairs.at(i, j)),
            m: mu,
            shape,
            chunk_lanes: self.layout.chunk,
            k: self.k,
            runs: None,
            paa: Some(self.sides),
            class: q.constant_class(None),
            joined: None,
        };
        let input = BandInput {
            kernel: &narrow,
            layout: self.layout,
            boxes: self.boxes,
            floors: self.floors,
            store: self.store,
            windows: &q.windows,
            frame: Frame {
                row: row_anchor.tick,
                col: col_anchor.tick,
            },
            sums: Some((&q.sum, &q.sum)),
        };
        let walked = band::run_tile::<_, K>(
            &input,
            tile,
            TileScratch {
                rho: &mut *rho,
                accs: &mut *accs,
            },
            self.book.map(|book| (book, &mut *part)),
        );
        if walked.is_none() {
            // A tile the driver declines (nothing touched) runs plain, every pair computed.
            narrow_tile_partials(
                &narrow,
                tile,
                qt,
                rho,
                accs,
                &q.windows,
                self.book.map(|book| (book, part)),
            );
        }
        false
    }
}

/// The seed pre-pass over exact cross terms of the whole series, when no
/// anchor covers the series: the same seeds, read from the series' own ticks.
pub(super) fn seed(
    q: &Quantized,
    (m, k): (NonZeroUsize, NonZeroUsize),
    paa: PaaSides<'_>,
    floors: Option<&SharedFloors>,
    (global, threads): (bool, NonZeroUsize),
) {
    let cols = q.columns();
    seed_floors(
        floors,
        &SeedInput {
            pairs: CrossTerms::of(cols.ticks, m.get()),
            centered: &|i, j, qt| centered_cross_fast(qt, m.get() as i64, cols.sum[i], cols.sum[j]),
            r: cols.r,
            bias: cols.bias,
            paa: paa.cols,
            exclusion: crate::exclusion::zone(m.get()),
            k: k.get(),
        },
        (false, global),
        threads,
    );
}

#[cfg(test)]
mod tests;
