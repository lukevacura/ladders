//! A tile on the exact i64 ticks, for the tiles no anchor covers (its rows or columns spanning
//! more ticks than one i32 anchor holds): the band driver of [`super::band`]
//! over [`WideTile`], the arithmetic the plain tiled pass's exact pass runs. The driver is the same code
//! as the narrow tile's; only the leaf compute and the cross-term side differ
//! ([`super::band::TileKernel`]), so the ladder, the floors, the lane store, the catch-up and the
//! constant-window rule are not repeated here.
//!
//! **Lanes.** A wide band's lane states are unscaled `QT` over the exact ticks, so a handoff
//! between two wide bands needs no move at all (their frames are the same: none). A band that
//! meets lanes left in the other width restarts them from the exact dot products
//! ([`super::store::Carry::wide`]), as lanes no band has started are, so bits never depend on which
//! bands were wide.

use super::anchored::{Anchored, Frame};
use super::band::{self, BandInput, TileScratch};
use crate::batch::pass::Shape;
use crate::batch::plan::Tile;
use crate::batch::tile::{RanWide, Scratch, WideTile, triangle_start_pair, wide_tile_partials};
use crate::cross_term::CrossTerms;
use crate::row::WideRow;

/// One wide tile: the band driver over the series' exact ticks, or the plain wide tile where the
/// driver declines it. Returns that the tile ran the exact i64 pass.
pub(super) fn tile<const K: usize>(
    run: &Anchored<'_, K>,
    tile: &Tile,
    scratch: &mut Scratch,
) -> RanWide {
    let q = run.quantized;
    let mu = run.m.get();
    let exclusion = crate::exclusion::zone(mu);
    let r0 = tile.rows.start;
    let cols = q.columns();
    let pairs = CrossTerms::of(cols.ticks, mu);
    let row_scalars = |row: usize| WideRow {
        enter: cols.ticks[row + mu - 1],
        leave: cols.ticks[row - 1],
        sum: q.windows[row].sum,
    };
    let wide = WideTile {
        cols: &cols,
        windows: &q.windows,
        row_scalars: &row_scalars,
        pairs,
        start_qt: &|slot| triangle_start_pair(exclusion, r0, slot).map(|(i, j)| pairs.at(i, j)),
        m: mu,
        shape: Shape::Triangle { exclusion },
        chunk_lanes: run.layout.chunk,
        k: run.k,
        runs: None,
        paa: Some(run.sides),
        class: q.constant_class(None),
        joined: None,
    };
    let walked = {
        let Scratch {
            tile: rho,
            accs,
            right: part,
            ..
        } = &mut *scratch;
        let input = BandInput {
            kernel: &wide,
            layout: run.layout,
            boxes: run.boxes,
            floors: run.floors,
            store: run.store,
            windows: &q.windows,
            frame: Frame::default(),
            sums: None,
        };
        band::run_tile::<_, K>(
            &input,
            tile,
            TileScratch { rho, accs },
            run.book.map(|book| (book, part)),
        )
    };
    if walked.is_none() {
        // A tile the driver declines (nothing touched) runs plain, every pair computed.
        wide_tile_partials(&wide, tile, scratch, run.book);
    }
    true
}
