//! A tile of the AB-join on the one-loop route for a pair of series no one anchor covers per side:
//! each tile narrows its rows of `a` and its columns of `b` to anchors of their own, as the plain
//! join tile does, and runs the join's band driver ([`super::join_band`]) over them; a tile no
//! pair of anchors covers runs the same driver on the exact i64 ticks ([`WideTile`]). Lanes between
//! bands are moved between the bands' anchors ([`super::anchored::move_lanes`]) or restarted at a
//! change of width, exactly as the self-join's are ([`super::anchored`], [`super::wide`]), so bits
//! never depend on which tiles were narrow.

use super::anchored::Frame;
use super::band::{BandInput, Layout, TileScratch};
use super::dense::Boxes;
use super::join_band;
use super::store::LaneStore;
use crate::batch::pass::Shape;
use crate::batch::plan::Tile;
use crate::batch::quantized::Quantized;
use crate::batch::survivor::SharedFloors;
use crate::batch::tile::{
    NarrowTile, RanWide, Scratch, WideTile, anchors_of, join_start_column, narrow_tile_partials,
    wide_tile_partials,
};
use crate::bound_paa::PaaSides;
use crate::cross_term::CrossTerms;
use crate::narrow;
use crate::row::{NarrowColumns, WideRow, narrow_row_scalars};
use crate::stats::JoinClass;
use core::num::NonZeroUsize;

/// What a run with per-tile anchors reads, beside the tile.
pub(super) struct PerTile<'a, const K: usize> {
    pub(super) a: &'a Quantized,
    pub(super) b: &'a Quantized,
    pub(super) m: NonZeroUsize,
    pub(super) k: NonZeroUsize,
    pub(super) layout: Layout,
    pub(super) boxes: &'a Boxes<K>,
    pub(super) floors: Option<&'a SharedFloors>,
    pub(super) store: &'a LaneStore,
    pub(super) sides: PaaSides<'a>,
    /// The join's constant class, when it has one.
    pub(super) class: Option<&'a JoinClass>,
}

impl<const K: usize> PerTile<'_, K> {
    /// One tile: the band driver over anchors of the tile's own where a pair of anchors covers its
    /// rows and its columns, over the exact ticks where none does. Returns whether the tile ran
    /// the exact i64 pass.
    pub(super) fn tile(&self, tile: &Tile, scratch: &mut Scratch) -> RanWide {
        let (a, b) = (self.a, self.b);
        let mu = self.m.get();
        let n_b = b.windows.len();
        let shape = Shape::Join { n_b };
        let (r0, r1) = (tile.rows.start, tile.rows.end);
        let columns = shape.columns(tile);
        let (cb, col_end) = (columns.start, columns.end);
        // A tile the cut makes always has ticks to anchor; one that does not runs the exact
        // pass, which declines it.
        let ticks = (|| {
            let rows = a
                .ticks
                .get(r0.saturating_sub(1)..(r1.checked_sub(1)?).checked_add(mu)?)?;
            let cols = b
                .ticks
                .get(cb..(col_end.checked_sub(1)?).checked_add(mu)?)?;
            Some((rows, cols))
        })();
        let anchors = ticks
            .filter(|_| cb < col_end && r0 < r1 && r1 <= a.windows.len() && a.sum.len() >= r1)
            .and_then(|(rows, cols)| anchors_of(rows, cols, self.m))
            .filter(|_| crate::route::forced_cell() == 0);
        let Some((row_anchor, col_anchor)) = anchors else {
            return self.wide(tile, scratch);
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
            ..
        } = scratch;
        narrow::narrow_ticks_into(
            row_anchor,
            self.m,
            &a.ticks[r0.saturating_sub(1)..r1 - 1 + mu],
            rt,
        );
        // Row 0's slots hold the pair one row before it: its leaving tick is a zero in front of
        // the row ticks, so the same pass advances it.
        if r0 == 0 {
            rt.insert(0, 0);
        }
        narrow::narrow_sums_into(row_anchor, self.m, &a.sum[r0..r1], rs);
        narrow::narrow_ticks_into(col_anchor, self.m, &b.ticks[cb..col_end - 1 + mu], ct);
        narrow::narrow_sums_into(col_anchor, self.m, &b.sum[cb..col_end], cs);
        let conversion = a.range.joint(b.range).conversion();
        let row_cols = NarrowColumns {
            ticks: rt,
            sum: rs,
            r: &a.r,
            bias: &a.bias,
            variance: &a.variance,
            conversion,
        };
        scalars.clear();
        scalars.extend(tile.rows.clone().map(|row| {
            narrow_row_scalars(&row_cols, &a.windows[row], row + 1 - r0, mu, rs[row - r0])
        }));
        let col_cols = NarrowColumns {
            ticks: ct,
            sum: cs,
            r: &b.r[cb..col_end],
            bias: &b.bias[cb..col_end],
            variance: &b.variance[cb..col_end],
            conversion,
        };
        let row_frame = if r0 == 0 { &rt[1..] } else { &rt[..] };
        let pairs = CrossTerms::new(row_frame, r0.saturating_sub(1), &ct[..], cb, mu);
        let narrow = NarrowTile {
            cols: &col_cols,
            base: cb,
            scalars,
            pairs,
            start_qt: &|slot| {
                join_start_column(n_b, r0, slot).map(|j| match r0.checked_sub(1) {
                    Some(before) => pairs.at(before, j - 1),
                    None => pairs.before_first_row(j),
                })
            },
            m: mu,
            shape,
            chunk_lanes: self.layout.chunk,
            k: self.k,
            runs: None,
            paa: Some(self.sides),
            class: None,
            joined: self.class,
        };
        let input = BandInput {
            kernel: &narrow,
            layout: self.layout,
            boxes: self.boxes,
            floors: self.floors,
            store: self.store,
            windows: &a.windows,
            frame: Frame {
                row: row_anchor.tick,
                col: col_anchor.tick,
            },
            sums: Some((&a.sum, &b.sum)),
        };
        let walked = join_band::run_tile::<_, K>(
            &input,
            tile,
            TileScratch {
                rho: &mut *rho,
                accs: &mut *accs,
            },
        );
        if walked.is_none() {
            // A tile the driver declines (nothing touched) runs plain, every pair computed.
            narrow_tile_partials(&narrow, tile, qt, rho, accs, &a.windows, None);
        }
        false
    }

    /// One tile on the exact i64 ticks: the band driver over [`WideTile`], or the plain wide tile
    /// where the driver declines it. Returns that the tile ran the exact i64 pass.
    fn wide(&self, tile: &Tile, scratch: &mut Scratch) -> RanWide {
        let (a, b) = (self.a, self.b);
        let mu = self.m.get();
        let n_b = b.windows.len();
        let r0 = tile.rows.start;
        let b_cols = b.columns();
        let pairs = CrossTerms::new(&a.ticks[..], 0, &b.ticks[..], 0, mu);
        let row_scalars = |row: usize| WideRow {
            enter: a.ticks[row + mu - 1],
            leave: row.checked_sub(1).map_or(0, |before| a.ticks[before]),
            sum: a.sum[row],
        };
        let wide = WideTile {
            cols: &b_cols,
            windows: &a.windows,
            row_scalars: &row_scalars,
            pairs,
            start_qt: &|slot| {
                join_start_column(n_b, r0, slot).map(|j| match r0.checked_sub(1) {
                    Some(before) => pairs.at(before, j - 1),
                    None => pairs.before_first_row(j),
                })
            },
            m: mu,
            shape: Shape::Join { n_b },
            chunk_lanes: self.layout.chunk,
            k: self.k,
            runs: None,
            paa: Some(self.sides),
            class: None,
            joined: self.class,
        };
        let walked = {
            let Scratch {
                tile: rho, accs, ..
            } = &mut *scratch;
            let input = BandInput {
                kernel: &wide,
                layout: self.layout,
                boxes: self.boxes,
                floors: self.floors,
                store: self.store,
                windows: &a.windows,
                frame: Frame::default(),
                sums: None,
            };
            join_band::run_tile::<_, K>(&input, tile, TileScratch { rho, accs })
        };
        if walked.is_none() {
            // A tile the driver declines (nothing touched) runs plain, every pair computed.
            wide_tile_partials(&wide, tile, scratch, None);
        }
        true
    }
}
