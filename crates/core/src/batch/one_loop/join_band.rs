//! The tile driver of the one-loop route for the AB-join: [`super::band`]'s driver over a
//! rectangle of rows of `a` by columns of `b`, with no triangle and no exclusion zone.
//!
//! **Geometry.** Slot `s` of row `i` meets column `i + n_b - 1 - s` (the pass's layout), so row `i`
//! has the slots `i..i + n_b` and a lane (a slot) lives from the row whose column is 0 (row
//! `s + 1 - n_b`, or row 0) to row `s`, where its column is `n_b - 1`: it is born and it ends inside
//! the sweep, where a self-join's lane is only born. A block of rows by slots has two edges of
//! cells that are not pairs, so a block is skippable only if it holds no such cell; the rest is the
//! self-join's: slots go ascending in ranges of [`super::band::RANGE_SLOTS`], each range reads the floors, lets
//! the dense ladder prove blocks skippable, computes the surviving runs and raises the floors, and
//! the lanes still live at a band's bottom edge go to the band below through the store.
//!
//! **Floors.** The join reports `a`'s profile only: a pair of row `i` and column `j` is offered to
//! row `i` alone and no partial of column `j` takes it, so a column has no floor a pair must beat,
//! and a block is skippable iff every correlation in it is below the floors of its rows (the
//! left profile's rule, see [`super::dense::Floors`]). A row's floor is the near-tie band of its
//! `k`-th best correlation so far over every pair offered to it, by this tile or any other (the
//! shared table), so it is at most the row's final floor and a skip against it is a proof.
//!
//! **Lane rows.** As in [`super::band`], a lane holds its exact state through a row, kept per slot
//! (`Carry::through`), and a run catches the live lanes up to the row before its first row. Lanes
//! live at a run's first row `r` are the slots `r..r + n_b - 1` (their column is at least 1 there);
//! a lane whose column is 0 at the first row, or later, is started by the pass itself. Row 0 has
//! no pair before it: a lane not yet started starts from the exact state before the first row
//! ([`CrossTerms::before_first_row`](crate::cross_term::CrossTerms)), the state the plain tiled pass
//! starts it from.

use super::Counts;
use super::anchored::{Lanes, move_lanes};
use super::band::{
    BandInput, LEAF, Leaf, RANGE_BITS, TileKernel, TileScratch, catch_up_over, raise,
};
use super::dense::{Band, Floors, Ladder, valid_cells_over};
use super::snapshot::{FloorSnapshot, FloorSources};
use super::store::Carry;
use crate::batch::pass::Shape;
use crate::batch::plan::Tile;
use crate::batch::tile::{close_tile, open_tile};
use crate::capacity::reserved_vec;
use crate::cross_term::UNSTARTED;
use crate::row::RowAcc;
use crate::stats::JoinRows;
use alloc::vec::Vec;
use core::ops::Range;

/// What a run of survivors reads and writes, bundled for the compute.
struct Flush<'a, T> {
    kernel: &'a T,
    tile: &'a Tile,
    /// The columns of `b`.
    n_b: usize,
    /// One past the last slot the carry holds: slot `s` is at `top - 1 - s` of the carry's states.
    top: usize,
}

/// Runs one tile of the AB-join: every range of its slots proved, computed and raised, lanes taken
/// from and handed to the store, and the tile's row partials left as today's narrow tile leaves
/// them. Returns the tile's counts, or `None` when the tile is not one this driver runs (not a
/// join tile, no coefficients, a slot range that is not aligned to the layout's tile width) or its
/// working memory cannot be allocated, with nothing touched, so the caller runs today's tile.
/// Every valid cell of the tile's rectangle is skipped on a proof or computed exactly once, so
/// `skipped_area + computed_area` is the tile's pairs.
pub(super) fn run_tile<T: TileKernel, const K: usize>(
    input: &BandInput<'_, T, K>,
    tile: &Tile,
    scratch: TileScratch<'_>,
) -> Option<Counts> {
    #[cfg(feature = "std")]
    let started = std::time::Instant::now();
    let kernel = input.kernel;
    let Shape::Join { n_b } = kernel.shape() else {
        return None;
    };
    let sides = kernel.paa()?;
    let layout = input.layout;
    let n_a = input.windows.len();
    let (qa, qb) = (tile.qa, tile.qb);
    let top = qa + layout.tile_lanes;
    if qb > top
        || qa >= qb
        || !qa.is_multiple_of(layout.tile_lanes)
        || tile.rows.start >= tile.rows.end
        || tile.rows.end > n_a
    {
        return None;
    }
    // The store's cell of the tile: slot ranges are aligned to the tile width in every band.
    let cell = qa / layout.tile_lanes;
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
    // The join offers no pair to a column: the snapshot holds no column blocks.
    let mut snapshot = FloorSnapshot::try_new(layout.side / 8, 0).ok()?;
    let row0 = tile.rows.start / layout.side * layout.side;
    let unit_rows = row0..row0 + layout.side;
    let mut counts = Counts {
        wide_bands: usize::from(T::WIDE),
        ..Counts::default()
    };
    let mut carry = match input.store.take(cell, tile.rows.start) {
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
    // The lanes live at the band's first row, moved from the anchors the band above measured
    // them from to this band's (a no-op for one frame).
    let live = qa.max(tile.rows.start)..qb.min(tile.rows.start + n_b - 1);
    counts.lanes_handed_down += live
        .clone()
        .filter(|&s| carry.through[s - qa] != UNSTARTED)
        .count();
    counts.lanes_moved += move_lanes(
        &mut carry,
        input.frame,
        &Lanes {
            m: kernel.m(),
            shape: kernel.shape(),
            first_row: tile.rows.start,
            slots: live,
            span: (qa, top),
            sums: input.sums,
        },
    );

    let shape = kernel.shape();
    let right = open_tile(
        shape,
        tile,
        (input.windows, kernel.class()),
        kernel.k(),
        accs,
        None,
    );
    let flush = Flush {
        kernel,
        tile,
        n_b,
        top,
    };
    let tile_rows = tile.rows.clone();
    let no_columns = 0..0;
    // What the class settles for each row of the tile, read once; without room for it the tile
    // computes what the ladder leaves, which only costs time.
    let (settled, row_classes) = match kernel.joined() {
        Some(class) => match reserved_vec::<JoinRows>(tile_rows.len() as u128) {
            Ok(mut classes) => {
                classes.extend(tile_rows.clone().map(|row| class.row_class(row)));
                (Some(class), classes)
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
        snapshot.refresh(
            &FloorSources {
                accs,
                first_row: tile.rows.start,
                right: None,
                shared: input.floors,
            },
            unit_rows.clone(),
            no_columns.clone(),
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
                        let area = valid_cells_over(shape, &rows_in, &slots);
                        let pairs = compute_run(
                            &flush,
                            rows.start,
                            slots,
                            (&mut carry, accs, rho),
                            &mut counts,
                        );
                        debug_assert_eq!(pairs, area, "a survivor run computes exactly its pairs");
                        computed += pairs;
                        survivor_area += area;
                        counts.survivor_runs += 1;
                    };
                    match (settled, block_class) {
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
            valid_cells_over(shape, &tile_rows, &span).saturating_sub(survivor_area);

        // The floors the next range reads are raised from what this one computed; the snapshot
        // just used was read before, and the table only rises.
        let sources = FloorSources {
            accs,
            first_row: tile.rows.start,
            right: None,
            shared: input.floors,
        };
        snapshot.assert_not_above(&sources, &unit_rows, &no_columns);
        if let Some(table) = input.floors {
            raise(
                table,
                (accs, tile.rows.start),
                None,
                (tile_rows.clone(), &no_columns),
            );
        }
    }
    close_tile(right, input.windows, input.floors, (&tile.rows, accs));
    // The band below continues these lanes where it starts.
    if tile.rows.end < n_a {
        input.store.put(cell, carry, tile.rows.end);
    }
    #[cfg(feature = "std")]
    {
        counts.band_nanos += usize::try_from(started.elapsed().as_nanos()).unwrap_or(usize::MAX);
    }
    crate::route::count_one_loop(&counts);
    Some(counts)
}

/// Computes the row block starting at `row0` (a multiple of [`LEAF`]) over the adjacent `slots`, a
/// run of surviving blocks, and returns the pairs computed.
fn compute_run<T: TileKernel>(
    flush: &Flush<'_, T>,
    row0: usize,
    slots: Range<usize>,
    (carry, accs, rho): (&mut Carry, &mut [RowAcc], &mut [f64]),
    counts: &mut Counts,
) -> usize {
    let kernel = flush.kernel;
    let Flush { tile, n_b, top, .. } = *flush;
    let (qa, first_row) = (tile.qa, tile.rows.start);
    let rows = row0.max(tile.rows.start)..(row0 + LEAF).min(tile.rows.end);
    if rows.start >= rows.end || slots.end > top {
        return 0;
    }
    let (a, b) = (slots.start, slots.end);

    // The lanes live at the first row (column at least 1, slot not yet past): current through it,
    // by the cross-term owner's step-or-restart rule, or from the state before row 0.
    let live = a.max(rows.start)..b.min(rows.start + n_b - 1);
    if live.start < live.end {
        if rows.start == 0 {
            for s in live.clone() {
                if carry.through[s - qa] == UNSTARTED {
                    carry.qt[top - 1 - s] = kernel.pairs().before_first_row(n_b - 1 - s);
                    carry.through[s - qa] = 0;
                    counts.lanes_restarted += 1;
                }
            }
        } else {
            catch_up_over(
                &kernel.pairs(),
                carry,
                (qa, top, kernel.shape()),
                (live.clone(), rows.start),
                counts,
            );
        }
        debug_assert!(
            live.clone().all(|s| carry.through[s - qa] == rows.start),
            "every lane is current before its run computes"
        );
    }

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
        None,
    );
    // The lanes that exist at the last row (its column is 0 or more, its slot not past) are
    // current through it.
    for s in a.max(rows.end - 1)..b.min(rows.end - 1 + n_b) {
        carry.through[s - qa] = rows.end;
    }
    computed
}

#[cfg(test)]
mod tests;
