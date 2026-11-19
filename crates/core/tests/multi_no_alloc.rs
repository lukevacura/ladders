#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]
//! The multi-dimensional batch profile allocates a fixed number of times per tile, however long
//! the series: a cell orders its dimensions and sums their means in buffers its tile owns, so the
//! allocations of a run grow with its tiles (set-up, a tile's scratch, a band's partials), never
//! with its cells. `CountingAllocator` only counts once it is the process's global allocator, so
//! this is its own test binary (see `nearest_no_alloc.rs`); on one thread the tiles run on the
//! counting thread.

use core::num::NonZeroUsize;

use ladde::batch::multi::{maamp_report_tiled, mstump_report_tiled};
use ladde::quantize::Scale;
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// Ties and a walk, so neither a constant column nor a lucky tie-free one hides a path.
fn dimension(n: usize, salt: usize) -> Vec<f64> {
    let mut level = 0.0;
    (0..n)
        .map(|t| {
            level += ((t * 7 + salt * 3) % 5) as f64 - 2.0;
            if t % 9 < 3 { 1.0 } else { level }
        })
        .collect()
}

/// Allocations and tiles of one run on one thread, `normalized` or not, over `n` samples.
fn run(n: usize, dimensions: usize, normalized: bool) -> (usize, usize) {
    let series: Vec<Vec<f64>> = (0..dimensions).map(|a| dimension(n, a)).collect();
    let refs: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let scale = Scale::new(1.0).unwrap();
    let (report, allocations) = allocations_in(|| {
        if normalized {
            let scales = vec![scale; dimensions];
            mstump_report_tiled(
                &refs,
                nz(8),
                &scales,
                &[1],
                true,
                nz(1),
                nz(16),
                nz(8),
                nz(32),
            )
            .unwrap()
            .1
        } else {
            maamp_report_tiled(
                &refs,
                nz(8),
                scale,
                &[1],
                true,
                nz(1),
                nz(16),
                nz(8),
                nz(32),
            )
            .unwrap()
            .1
        }
    });
    (allocations, report.tiles)
}

#[test]
fn allocations_per_tile_do_not_grow_with_the_series() {
    for (dimensions, normalized) in [(3, true), (6, true), (24, true), (3, false), (24, false)] {
        let (few, tiles_few) = run(120, dimensions, normalized);
        let (many, tiles_many) = run(360, dimensions, normalized);
        assert!(
            tiles_many > 4 * tiles_few,
            "{tiles_few} tiles, then {tiles_many}"
        );
        let (per_few, per_many) = (few / tiles_few, many / tiles_many);
        assert!(
            per_many <= 24 && per_many <= per_few + 2,
            "d {dimensions} {normalized}: {few} allocations in {tiles_few} tiles, \
             {many} in {tiles_many}"
        );
    }
}
