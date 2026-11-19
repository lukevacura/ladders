//! The one-loop route of a tiled self-join, two-sided or one-sided (the left profile, whose columns
//! hold no floor, so only its rows' floors prove a skip): a dense ladder of bulk bounds replaces the
//! group, block and sub-block checks of the old tiled path. For a band's rows and a range of its slots, the
//! ladder sets one bit per block of 256, 64 and 16 slots, a set bit meaning the block needs no
//! compute: its PAA bound is below every floor of its rows and columns, or it holds no pair. The
//! zero bits of the finest layer are the survivors, as runs, and each run is computed lane by lane
//! through the pass's leaf compute; the floors rise from the computed pairs after each range of
//! slots, so the next range's bounds read them. The lanes still live at a band's bottom edge are
//! handed down to the next band through a store, so no band boundary restarts a lane.
//!
//! The AB-join runs the same ladder, snapshot and store (`join` is its entry and cut of the
//! rectangle, `join_band` its tile driver; only the rows have floors, as in the left profile).
//!
//! The owners, one per file: `dense` is the bulk bound pass and its f32 soundness rule; `snapshot`
//! reads the floors the ladder's thresholds come from; `band` drives a tile (range by range:
//! snapshot, ladder, compute, raise, handoff); `store` holds the lanes between bands; `anchored`
//! narrows each band to anchors of its own when no one anchor covers the series, and moves lanes
//! between bands' anchors. This file owns the entry, the conditions under which the route runs,
//! and the counters.
//!
//! Every output bit equals the plain pass's (every pair computed, no pruning): a skip is a proof
//! against floors that are at most final, and the compute is the exact kernel. This route is the
//! only pruning path: where its conditions do not hold, or it is forced off, the plain pass runs.

use super::left::tiled_narrow_profile_par;
use super::pass::Shape;
use super::plan::{SlotRuns, Tile, TileCut, TilePlan};
use super::quantized::Quantized;
use super::schedule::{BatchReport, run_tiles_top_down};
use super::survivor::{SeedInput, SharedFloors, seed_floors, shared_floors};
use super::tile::{NarrowTile, Scratch, narrow_tile_partials, triangle_start_pair};
use super::two_sided::RightBook;
use crate::bound_paa::{PaaSides, Segments};
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::narrow;
use crate::route::OneLoop;
use crate::row::{NarrowColumns, narrow_row_scalars};
use crate::stats::{ConstantClass, Window};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

mod anchored;
mod band;
mod dense;
pub(super) mod join;
mod join_anchored;
mod join_band;
mod snapshot;
mod store;
mod wide;

/// What a run of the route did, as the route harness counts it ([`crate::route::Work`]); never part
/// of a result. Each tile's counts are added into its run's, and a count may depend on the
/// interleaving of floors, a result never does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Entries the ladder's passes evaluated, per layer (blocks of 256, 64 and 16 slots): a
    /// block under a skipped parent is not evaluated.
    pub(crate) entries: [usize; 3],
    /// Runs of surviving blocks of the finest layer computed, one call of the leaf compute each.
    pub(crate) survivor_runs: usize,
    /// Lanes stepped through a gap by the exact recurrence.
    pub(crate) lanes_stepped: usize,
    /// Lanes started or restarted from an exact dot product (a lane no band has started, or one
    /// whose gap since its last row the restart crosses cheaper than stepping does).
    pub(crate) lanes_restarted: usize,
    /// The steps the stepped lanes took in all.
    pub(crate) lane_steps: usize,
    /// Lanes a band took from the band above with no restart.
    pub(crate) lanes_handed_down: usize,
    /// Lanes a band took from the band above whose states it moved to its own anchors (per-band
    /// anchors only).
    pub(crate) lanes_moved: usize,
    /// Tiles the driver ran on the exact i64 ticks.
    pub(crate) wide_bands: usize,
    /// Nanoseconds the bands took, summed over threads.
    pub(crate) band_nanos: usize,
    /// Cells of the triangle skipped on a proof: the pairs of every range that no survivor run
    /// holds, a pair that touches a non-finite window counted like any other (the area is the
    /// triangle's cells, the pairs outside the exclusion zone, whatever their windows hold).
    pub(crate) skipped_area: usize,
    /// Cells of the triangle computed: the pairs the pass evaluated for the survivor runs (a pair
    /// that touches a non-finite window is evaluated, to advance its lane, and counted, though it
    /// ranks nothing). The two areas add up to the triangle's.
    pub(crate) computed_area: usize,
}

/// What a run has, as far as the route's conditions read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Conditions {
    /// Only some diagonals are walked.
    pub(super) runs: bool,
    /// The run reports only its best pair, and keeps a global floor.
    pub(super) global: bool,
    /// The PAA coefficients exist.
    pub(super) paa: bool,
}

/// Whether the route runs for a run with `conditions`: any number of neighbors per row, every
/// diagonal and the PAA coefficients, two-sided (pairs offered to the columns' neighbors
/// too) or one-sided (the left profile, whose columns hold no floor, see
/// [`dense::Floors`]); a global floor, when the run keeps one (the best-pair search), is one more
/// floor the ladder reads. The route narrows the whole series to one anchor where one covers it
/// and each band to its own anchors where none does ([`anchored`]). Anything else (the cold
/// route, a filter that is off, some diagonals) runs plain.
pub(super) fn applies(conditions: &Conditions) -> bool {
    !conditions.runs && conditions.paa
}

/// Whether the route runs a self-join walking only some diagonals if `runs`, with the PAA
/// coefficients if `paa` and keeping a global floor if `global`: the callers that choose between
/// the tile engine and the single-threaded one ask this, so a run the route does not take keeps
/// the plain engine. Every run of every family takes the route wherever its conditions hold (over
/// one anchor for the whole series, per-band anchors or exact ticks alike), and nothing does
/// where it is forced off ([`OneLoop::Off`], the plain pass, the bits' reference).
pub(in crate::batch) fn taken(runs: bool, paa: bool, global: bool) -> bool {
    applies(&Conditions { runs, global, paa }) && crate::route::one_loop() != OneLoop::Off
}

/// The self-join (two-sided with `right`, one-sided without) over one narrow anchor for the whole
/// series, as [`tiled_narrow_profile_par`] runs it plain, run by the one-loop route where
/// [`taken`] holds, and plain otherwise. The same arguments, the same bits.
#[allow(clippy::too_many_arguments)]
pub(super) fn profile(
    cols: &NarrowColumns,
    windows: &[Window],
    m: usize,
    plan: TilePlan,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    right: Option<&RightBook>,
    (runs, paa, class): (Option<&SlotRuns>, Option<PaaSides>, Option<&ConstantClass>),
    global: Option<f64>,
) -> BatchReport {
    if let (true, Some(sides)) = (taken(runs.is_some(), paa.is_some(), global.is_some()), paa) {
        let run = Run {
            source: Source::Whole(cols),
            windows,
            m,
            plan,
            threads,
            book: right,
            sides,
            class,
            global,
        };
        let report = match sides.segments() {
            Segments::Four => run.walk::<4>(results),
            Segments::Eight => run.walk::<8>(results),
            Segments::Sixteen => run.walk::<16>(results),
        };
        if let Some(report) = report {
            return report;
        }
    }
    tiled_narrow_profile_par(
        cols,
        (windows, m),
        plan,
        threads,
        results,
        right,
        (runs, class),
    )
}

/// The self-join over per-band anchors, for a series no one anchor covers: each band narrows its
/// rows and each tile its columns to anchors of their own ([`anchored`]), and a tile no anchors
/// cover runs on the exact i64 ticks ([`wide`]). The arguments of [`profile`], with the quantized series in place of its
/// narrowed columns; `None` when the route is not taken (or its working memory cannot be
/// allocated), before anything is touched, and the caller runs the plain pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn profile_per_band(
    quantized: &Quantized,
    m: NonZeroUsize,
    plan: TilePlan,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    right: Option<&RightBook>,
    global: Option<f64>,
) -> Option<BatchReport> {
    let sides = quantized.paa.as_deref().map(PaaSides::of)?;
    if !taken(false, true, global.is_some()) {
        return None;
    }
    let run = Run {
        source: Source::PerBand(quantized),
        windows: &quantized.windows,
        m: m.get(),
        plan,
        threads,
        book: right,
        sides,
        class: quantized.constant_class(None),
        global,
    };
    match sides.segments() {
        Segments::Four => run.walk::<4>(results),
        Segments::Eight => run.walk::<8>(results),
        Segments::Sixteen => run.walk::<16>(results),
    }
}

/// Where a run's columns come from.
#[derive(Clone, Copy)]
enum Source<'a> {
    /// One narrow anchor covers the whole series.
    Whole(&'a NarrowColumns<'a>),
    /// Each band takes anchors of its own over the series' ticks.
    PerBand(&'a Quantized),
}

/// The arguments of a run the route takes.
struct Run<'a> {
    source: Source<'a>,
    windows: &'a [Window],
    m: usize,
    plan: TilePlan,
    threads: NonZeroUsize,
    /// The columns' neighbors' book; `None` for a one-sided run, which offers nothing to columns.
    book: Option<&'a RightBook>,
    sides: PaaSides<'a>,
    class: Option<&'a ConstantClass>,
    /// Where the run's global floor starts, when it keeps one.
    global: Option<f64>,
}

impl Run<'_> {
    /// The run's tiles, with the ladder over `K` segments. `None` when the run's working memory
    /// cannot be allocated, before anything is touched; the caller then runs the plain pass.
    fn walk<const K: usize>(&self, results: &mut Vec<(u64, Entry)>) -> Option<BatchReport> {
        let Self {
            source,
            windows,
            m,
            plan,
            threads,
            book,
            sides,
            class,
            global,
        } = *self;
        let n = windows.len();
        let exclusion = crate::exclusion::zone(m);
        let lag = exclusion + 1;
        let layout = band::Layout::new((plan.band_rows, plan.chunk_lanes, plan.tile_lanes), n);
        let boxes = dense::Boxes::<K>::try_new(sides, m, (n, lag, layout.side))?;
        // A band whose rows no one anchor covers is cut into bands one grain tall, whose rows (and
        // tiles' columns) span fewer ticks, where that leaves fewer pairs in tiles no anchor
        // covers; any other band, and every band of a run with one anchor for the series, keeps
        // the plan's rows.
        let tiles = match source {
            Source::Whole(_) => cut_tiles(n, exclusion, &layout, &|_| true)?,
            Source::PerBand(q) => {
                let mu = NonZeroUsize::new(m)?;
                let free = crate::route::forced_cell() != 0 || crate::route::forced_wide();
                let fits = |tile: &Tile| {
                    let (r0, r1) = (tile.rows.start, tile.rows.end);
                    free || q
                        .ticks
                        .get(r0 - 1..r1 - 1 + m)
                        .is_some_and(|t| narrow::Anchor::can_cover(t, mu))
                };
                cut_tiles(n, exclusion, &layout, &fits)?
            }
        };
        let ranges = n.saturating_sub(lag).div_ceil(layout.tile_lanes);
        let store = store::LaneStore::try_new(ranges).ok()?;
        let cut = TileCut {
            plan: TilePlan {
                band_rows: layout.side,
                chunk_lanes: layout.chunk,
                tile_lanes: layout.tile_lanes,
                k: plan.k,
                survivor: plan.survivor,
            },
            none_rows: n.min(lag),
            tiles,
        };
        let floors = shared_floors(plan.survivor, n, book.is_some(), global);
        let shared = floors.as_deref();
        // The seeds stay today's pre-pass: the ladder reads the floors they raise.
        match source {
            Source::Whole(cols) => seed_floors(
                shared,
                &SeedInput {
                    pairs: CrossTerms::of(cols.ticks, m),
                    centered: &|i, j, qt| narrow::cross(qt, cols.sum[i], cols.sum[j]),
                    r: cols.r,
                    bias: cols.bias,
                    paa: sides.cols,
                    exclusion,
                    k: plan.k.get(),
                },
                (false, global.is_some()),
                threads,
            ),
            Source::PerBand(quantized) => anchored::seed(
                quantized,
                (NonZeroUsize::new(m)?, plan.k),
                sides,
                shared,
                (global.is_some(), threads),
            ),
        }
        let report = match source {
            Source::Whole(cols) => {
                let run = WholeRun {
                    cols,
                    windows,
                    m,
                    k: plan.k,
                    layout,
                    boxes: &boxes,
                    floors: shared,
                    store: &store,
                    book,
                    sides,
                    class,
                };
                let body = |tile: &Tile, scratch: &mut Scratch, _chunk_lanes: usize| {
                    whole_tile(&run, tile, scratch)
                };
                run_tiles_top_down(windows, m, cut, threads, results, &body)
            }
            Source::PerBand(quantized) => {
                let anchored = anchored::Anchored {
                    quantized,
                    m: NonZeroUsize::new(m)?,
                    k: plan.k,
                    layout,
                    boxes: &boxes,
                    floors: shared,
                    store: &store,
                    book,
                    sides,
                };
                let body = |tile: &Tile, scratch: &mut Scratch, chunk_lanes: usize| {
                    anchored.tile(tile, scratch, chunk_lanes)
                };
                run_tiles_top_down(windows, m, cut, threads, results, &body)
            }
        };
        Some(report)
    }
}

/// What a tile of a whole-anchor run reads.
struct WholeRun<'a, const K: usize> {
    cols: &'a NarrowColumns<'a>,
    windows: &'a [Window],
    m: usize,
    k: NonZeroUsize,
    layout: band::Layout,
    boxes: &'a dense::Boxes<K>,
    floors: Option<&'a SharedFloors>,
    store: &'a store::LaneStore,
    book: Option<&'a RightBook>,
    sides: PaaSides<'a>,
    class: Option<&'a ConstantClass>,
}

/// One tile of a whole-anchor run: the band driver over the series' own narrowed columns, or the
/// plain tile where the driver declines it. Never the exact i64 pass.
fn whole_tile<const K: usize>(run: &WholeRun<'_, K>, tile: &Tile, scratch: &mut Scratch) -> bool {
    let WholeRun {
        cols,
        windows,
        m,
        k,
        layout,
        boxes,
        floors,
        store,
        book,
        sides,
        class,
    } = *run;
    let exclusion = crate::exclusion::zone(m);
    let shape = Shape::Triangle { exclusion };
    let pairs = CrossTerms::of(cols.ticks, m);
    let Scratch {
        qt,
        tile: rho,
        accs,
        scalars,
        right: part,
        ..
    } = scratch;
    scalars.clear();
    scalars.extend(
        tile.rows
            .clone()
            .map(|row| narrow_row_scalars(cols, &windows[row], row, m, cols.sum[row])),
    );
    let narrow = NarrowTile {
        cols,
        base: 0,
        scalars,
        pairs,
        start_qt: &|slot| {
            triangle_start_pair(exclusion, tile.rows.start, slot).map(|(i, j)| pairs.at(i, j))
        },
        m,
        shape,
        chunk_lanes: layout.chunk,
        k,
        runs: None,
        paa: Some(sides),
        class,
        joined: None,
    };
    let input = band::BandInput {
        kernel: &narrow,
        layout,
        boxes,
        floors,
        store,
        windows,
        frame: anchored::Frame::default(),
        sums: None,
    };
    let walked = band::run_tile::<_, K>(
        &input,
        tile,
        band::TileScratch {
            rho: &mut *rho,
            accs: &mut *accs,
        },
        book.map(|book| (book, &mut *part)),
    );
    if walked.is_none() {
        // A tile the driver declines (nothing touched) runs plain, every pair computed.
        narrow_tile_partials(
            &narrow,
            tile,
            qt,
            rho,
            accs,
            windows,
            book.map(|book| (book, part)),
        );
    }
    false
}

/// The tiles of the triangle of `n` windows with bands that start at multiples of the layout's
/// side (the first band at the first row with a lag, `exclusion + 1`) and tiles of the layout's
/// width, top band first and ascending slots inside a band; `None` when they cannot be allocated.
/// A band whose tiles `fits` rejects is cut into bands of [`band::ANCHORED_SIDE`] rows, each its
/// own band of tiles over the same slot cells, so the lanes pass from one to the next as between
/// bands, where that leaves fewer pairs in tiles `fits` rejects.
fn cut_tiles(
    n: usize,
    exclusion: usize,
    layout: &band::Layout,
    fits: &dyn Fn(&Tile) -> bool,
) -> Option<Vec<Tile>> {
    let lag = exclusion + 1;
    let push_band = |tiles: &mut Vec<Tile>, rows: core::ops::Range<usize>, band: usize| {
        // The lanes of the band's last row: slots `0..end - lag`.
        let lanes = rows.end - lag;
        for (column, qa) in (0..lanes).step_by(layout.tile_lanes).enumerate() {
            tiles.try_reserve(1).ok()?;
            tiles.push(Tile {
                band,
                rows: rows.clone(),
                column,
                qa,
                qb: (qa + layout.tile_lanes).min(lanes),
            });
        }
        Some(())
    };
    let mut tiles = Vec::new();
    let (mut start, mut band) = (lag, 0);
    while start < n {
        let end = ((start / layout.side + 1) * layout.side).min(n);
        let mut whole = Vec::new();
        push_band(&mut whole, start..end, band)?;
        let mut chosen = whole;
        let mut used = 1;
        let uncovered = band::uncovered_pairs(Shape::Triangle { exclusion }, &chosen, fits);
        if uncovered > 0 {
            let mut cut = Vec::new();
            let mut pieces = 0;
            for rows in band::pieces(start..end) {
                push_band(&mut cut, rows, band + pieces)?;
                pieces += 1;
            }
            if band::uncovered_pairs(Shape::Triangle { exclusion }, &cut, fits) < uncovered {
                chosen = cut;
                used = pieces;
            }
        }
        tiles.try_reserve(chosen.len()).ok()?;
        tiles.extend(chosen);
        band += used;
        start = end;
    }
    Some(tiles)
}

#[cfg(test)]
mod tests;
