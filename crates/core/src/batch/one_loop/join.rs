//! The entry of the AB-join to the one-loop route: every window of `a` against every window of
//! `b`, no exclusion zone, one list of neighbors per `a` window. The ladder, the floor snapshot
//! and the lane store are the self-join's; what the join adds is its own cut of the rectangle into
//! tiles and its tile driver ([`super::join_band`]).
//!
//! **Geometry.** A band is a unit of the layout's `side` rows of `a`, and slot `s` of row `i` meets
//! column `i + n_b - 1 - s` of `b`. The columns of row block `a` (rows `a·rb..(a + 1)·rb`) by slot
//! block `b` (slots `b·S..(b + 1)·S`) are `c0..c0 + rb + S - 1` with
//! `c0 = rb·(a - q·(b + 1)) + n_b` and `q = S/rb`, depending on `a - q·(b + 1)` alone, exactly as
//! in the self-join with the signed lag `1 - n_b`: one table of column boxes per
//! layer, built once from `b`'s coefficients, serves every band ([`super::dense::Boxes`]), and the
//! row boxes come from `a`'s.
//!
//! **Tiles.** A band's slots run from its first row to its last row plus `n_b - 1`. Tiles are the
//! slot ranges of the layout's width aligned to multiples of that width, the same in every band
//! (the self-join's tiles all start at slot 0, so its ranges align by construction; here the band's
//! own first slot moves with the band), so the store's cell of a range is the same for the band
//! above and the band below and a lane handed down finds its lanes again.
//!
//! **Floors.** The join reports `a`'s profile, so only the rows have floors: the shared table holds
//! one per window of `a`, no column floor, no global floor, and no seeds (a self-join's seeds are
//! pairs of its own diagonals; the join has none worth the pass). A row's floor is at most its
//! final one, so a block below the lowest floor of its rows is below the `k`-th best of every row in
//! it, for any `k`, and no row would keep a pair in it.
//!
//! **What is reused.** The ladder, its thresholds, the snapshot, the store, the compute and the
//! catch-up are the self-join's, with the shape of the run ([`crate::batch::pass::Shape`]) in
//! place of its lag; what is new is the cut, the tile driver and the start of the lanes at row 0.

use super::band::{ANCHORED_SIDE, BandInput, Layout, TileScratch};
use super::dense::Boxes;
use super::join_anchored;
use super::join_band;
use super::store::LaneStore;
use crate::batch::pass::Shape;
use crate::batch::plan::{Tile, TileCut, TilePlan};
use crate::batch::quantized::Quantized;
use crate::batch::schedule::{BatchReport, run_tiles_top_down};
use crate::batch::survivor::shared_floors;
use crate::batch::tile::{NarrowTile, Scratch, join_start_column, narrow_tile_partials};
use crate::bound_paa::{PaaSides, Segments};
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::narrow;
use crate::route::OneLoop;
use crate::row::{NarrowColumns, narrow_row_scalars};
use crate::stats::JoinClass;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Whether the route runs an AB-join over every diagonal (`restricted` when only some are walked),
/// with the PAA coefficients of both sides (`paa`). The join takes the route wherever its
/// conditions hold (over one narrow anchor per side where they cover the series, per-tile anchors
/// or exact ticks otherwise), and nothing takes it where it is forced off ([`OneLoop::Off`], the
/// plain pass).
pub(in crate::batch) fn taken(restricted: bool, paa: bool) -> bool {
    !restricted && paa && crate::route::one_loop() != OneLoop::Off
}

/// The AB-join of `qa` against `qb` as [`ab_join`](mod@crate::batch::ab_join)'s tile engine runs it, run by the
/// one-loop route: every row's `plan.k` nearest windows of `qb`, appended to `results` row by row.
/// `None`, with nothing touched, when the run cannot take the route (no coefficients, `m` below
/// the segments, memory that cannot be allocated): the caller then runs
/// the plain pass. The same bits as the plain pass either way.
pub(in crate::batch) fn profile(
    (qa, qb): (&Quantized, &Quantized),
    m: NonZeroUsize,
    plan: TilePlan,
    threads: NonZeroUsize,
    class: Option<&JoinClass>,
    results: &mut Vec<(u64, Entry)>,
) -> Option<BatchReport> {
    let sides = PaaSides::join(qa.paa.as_deref(), qb.paa.as_deref())?;
    let run = Run {
        qa,
        qb,
        m,
        plan,
        threads,
        sides,
        class,
    };
    match sides.segments() {
        Segments::Four => run.walk::<4>(results),
        Segments::Eight => run.walk::<8>(results),
        Segments::Sixteen => run.walk::<16>(results),
    }
}

/// The arguments of a run the route takes.
struct Run<'a> {
    qa: &'a Quantized,
    qb: &'a Quantized,
    m: NonZeroUsize,
    plan: TilePlan,
    threads: NonZeroUsize,
    sides: PaaSides<'a>,
    class: Option<&'a JoinClass>,
}

impl Run<'_> {
    /// The run's tiles, with the ladder over `K` segments. `None` when the run's working memory
    /// cannot be allocated, before anything is touched.
    fn walk<const K: usize>(&self, results: &mut Vec<(u64, Entry)>) -> Option<BatchReport> {
        let Self {
            qa,
            qb,
            m,
            plan,
            threads,
            sides,
            class,
        } = *self;
        let mu = m.get();
        let (n_a, n_b) = (qa.windows.len(), qb.windows.len());
        let slots = n_a.checked_add(n_b)?.checked_sub(1)?;
        // One narrow anchor per side, where each covers its series (and no cell cut is forced):
        // each side narrowed once, with the plan's bands. Otherwise each tile takes anchors of its
        // own and the bands are cut one grain tall ([`ANCHORED_SIDE`]).
        let whole = match (qa.narrow_anchor(m), qb.narrow_anchor(m)) {
            (Some(a), Some(b)) if crate::route::forced_cell() == 0 => Some((a, b)),
            _ => None,
        };
        let band_rows = if whole.is_some() {
            plan.band_rows
        } else {
            plan.band_rows.min(ANCHORED_SIDE)
        };
        let layout = Layout::new((band_rows, plan.chunk_lanes, plan.tile_lanes), slots);
        let boxes = Boxes::<K>::try_new_join(sides, mu, ((n_a, n_b), layout.side))?;
        let tiles = cut_tiles((n_a, n_b), &layout)?;
        let store = LaneStore::try_new(slots.div_ceil(layout.tile_lanes)).ok()?;
        let cut = TileCut {
            plan: TilePlan {
                band_rows: layout.side,
                chunk_lanes: layout.chunk,
                tile_lanes: layout.tile_lanes,
                k: plan.k,
                survivor: plan.survivor,
            },
            none_rows: 0,
            tiles,
        };

        let floors = shared_floors(plan.survivor, n_a, false, None);
        let shared = floors.as_deref();
        let Some(anchors) = whole else {
            let per_tile = join_anchored::PerTile {
                a: qa,
                b: qb,
                m,
                k: plan.k,
                layout,
                boxes: &boxes,
                floors: shared,
                store: &store,
                sides,
                class,
            };
            let body = |tile: &Tile, scratch: &mut Scratch, _chunk_lanes: usize| {
                per_tile.tile(tile, scratch)
            };
            return Some(run_tiles_top_down(
                &qa.windows,
                mu,
                cut,
                threads,
                results,
                &body,
            ));
        };

        // Each side narrowed once for the whole series, as the one-pass join narrows it: the rows'
        // ticks lead with the zero a first row leaves, so the same pass advances it.
        let (mut a_ticks, mut a_sums, mut b_ticks, mut b_sums) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        narrow::narrow_ticks_into(anchors.0, m, &qa.ticks, &mut a_ticks);
        a_ticks.insert(0, 0);
        narrow::narrow_sums_into(anchors.0, m, &qa.sum, &mut a_sums);
        narrow::narrow_ticks_into(anchors.1, m, &qb.ticks, &mut b_ticks);
        narrow::narrow_sums_into(anchors.1, m, &qb.sum, &mut b_sums);
        let conversion = qa.range.joint(qb.range).conversion();
        let a_cols = NarrowColumns {
            ticks: &a_ticks,
            sum: &a_sums,
            r: &qa.r,
            bias: &qa.bias,
            variance: &qa.variance,
            conversion,
        };
        let b_cols = NarrowColumns {
            ticks: &b_ticks,
            sum: &b_sums,
            r: &qb.r,
            bias: &qb.bias,
            variance: &qb.variance,
            conversion,
        };
        let pairs = CrossTerms::new(&a_ticks[1..], 0, &b_ticks[..], 0, mu);
        let shape = Shape::Join { n_b };
        let body = |tile: &Tile, scratch: &mut Scratch, _chunk_lanes: usize| {
            let Scratch {
                qt,
                tile: rho,
                accs,
                scalars,
                ..
            } = scratch;
            scalars.clear();
            scalars.extend(tile.rows.clone().map(|row| {
                narrow_row_scalars(&a_cols, &qa.windows[row], row + 1, mu, a_sums[row])
            }));
            let first = tile.rows.start;
            let narrow = NarrowTile {
                cols: &b_cols,
                base: 0,
                scalars,
                pairs,
                start_qt: &|slot| {
                    join_start_column(n_b, first, slot).map(|j| match first.checked_sub(1) {
                        Some(before) => pairs.at(before, j - 1),
                        None => pairs.before_first_row(j),
                    })
                },
                m: mu,
                shape,
                chunk_lanes: layout.chunk,
                k: plan.k,
                runs: None,
                paa: Some(sides),
                class: None,
                joined: class,
            };
            let input = BandInput {
                kernel: &narrow,
                layout,
                boxes: &boxes,
                floors: shared,
                store: &store,
                windows: &qa.windows,
                frame: super::anchored::Frame::default(),
                sums: None,
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
                narrow_tile_partials(&narrow, tile, qt, rho, accs, &qa.windows, None);
            }
            false
        };
        Some(run_tiles_top_down(
            &qa.windows,
            mu,
            cut,
            threads,
            results,
            &body,
        ))
    }
}

/// The tiles of the rectangle of `n_a` rows by `n_b` columns: bands of the layout's side rows
/// (the first at row 0) and, in each, the slot ranges of the layout's tile width, aligned to
/// multiples of it, that hold a slot of the band (its first row to its last row plus `n_b - 1`),
/// top band first and ascending slots inside a band; `None` when they cannot be allocated.
fn cut_tiles((n_a, n_b): (usize, usize), layout: &Layout) -> Option<Vec<Tile>> {
    let mut tiles = Vec::new();
    let (mut start, mut band) = (0, 0);
    while start < n_a {
        let end = (start + layout.side).min(n_a);
        let lanes = start..end + n_b - 1;
        let first = lanes.start / layout.tile_lanes;
        let last = (lanes.end - 1) / layout.tile_lanes;
        for (column, range) in (first..=last).enumerate() {
            let qa = range * layout.tile_lanes;
            tiles.try_reserve(1).ok()?;
            tiles.push(Tile {
                band,
                rows: start..end,
                column,
                qa,
                qb: (qa + layout.tile_lanes).min(lanes.end),
            });
        }
        band += 1;
        start = end;
    }
    Some(tiles)
}

#[cfg(test)]
mod tests;
