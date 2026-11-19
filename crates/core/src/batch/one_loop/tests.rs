//! The self-join's cut of the triangle into bands: a band the plan cuts keeps its rows where its
//! own anchor covers them, is cut into bands one grain tall where it does not and a piece is
//! covered, and stays whole where no piece is covered either.

use super::band::{ANCHORED_SIDE, Layout};
use super::cut_tiles;
use crate::batch::plan::Tile;
use alloc::vec::Vec;
use core::ops::Range;

/// The distinct bands of the tiles, as the rows each holds, in order.
fn bands(tiles: &[crate::batch::plan::Tile]) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for tile in tiles {
        if out.last() != Some(&tile.rows) {
            out.push(tile.rows.clone());
        }
    }
    out
}

#[test]
fn a_band_whose_own_anchor_covers_it_keeps_its_rows() {
    let (n, exclusion) = (20_000, 3);
    let layout = Layout::new((4096, 64, 4096), n);
    assert_eq!(layout.side, 4096);
    let whole = cut_tiles(n, exclusion, &layout, &|_| true).expect("small cut");
    let kept = bands(&whole);
    assert!(kept.iter().all(|rows| rows.len() <= 4096));
    assert_eq!(kept.len(), n.div_ceil(4096));
    // Every band is covered: the same tiles as the cut with no test at all.
    let by_cover = cut_tiles(n, exclusion, &layout, &|tile| tile.rows.len() <= 4096).expect("cut");
    assert_eq!(bands(&by_cover), kept);
}

#[test]
fn a_band_no_anchor_covers_is_cut_a_grain_tall_where_a_piece_is_covered() {
    let (n, exclusion) = (20_000, 3);
    let layout = Layout::new((4096, 64, 4096), n);
    let covered = |tile: &Tile| tile.rows.len() <= ANCHORED_SIDE;
    let tiles = cut_tiles(n, exclusion, &layout, &covered).expect("small cut");
    let cut = bands(&tiles);
    assert!(
        cut.iter().all(|rows| rows.len() <= ANCHORED_SIDE),
        "{cut:?}"
    );
    // The pieces tile the rows the plan's bands hold, in order, with no gap and no overlap.
    let plan = bands(&cut_tiles(n, exclusion, &layout, &|_| true).expect("cut"));
    assert_eq!(cut.first().map(|r| r.start), plan.first().map(|r| r.start));
    assert_eq!(cut.last().map(|r| r.end), plan.last().map(|r| r.end));
    assert!(cut.windows(2).all(|pair| pair[0].end == pair[1].start));
    // Band numbers follow the pieces, and a piece holds the same slot cells as the band it came
    // from, so lanes pass from one to the next through the store.
    let numbers: Vec<usize> = tiles.iter().map(|tile| tile.band).collect();
    assert!(numbers.windows(2).all(|pair| pair[1] - pair[0] <= 1));
    assert_eq!(numbers.last().copied(), Some(cut.len() - 1));
    assert!(
        tiles
            .iter()
            .all(|tile| tile.qa.is_multiple_of(layout.tile_lanes))
    );
}

#[test]
fn a_band_stays_whole_where_no_piece_is_covered_either() {
    let (n, exclusion) = (20_000, 3);
    let layout = Layout::new((4096, 64, 4096), n);
    let plan = bands(&cut_tiles(n, exclusion, &layout, &|_| true).expect("cut"));
    let none = bands(&cut_tiles(n, exclusion, &layout, &|_| false).expect("cut"));
    assert_eq!(none, plan);
}
