extern crate std;

use super::super::pass::Shape;
use super::super::plan::{SlotRuns, Tile};
use super::*;
use crate::naive::{
    NonNormDistance, NonNormEntry, naive_ab_join_nonnorm, naive_left_profile_nonnorm,
};
use alloc::vec::Vec;
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// `(squared distance, neighbor)`, `(0, u64::MAX)` for none.
fn plain(entry: NonNormEntry) -> (u128, u64) {
    match entry.index().get() {
        Some(j) => {
            let squared = match entry.distance() {
                NonNormDistance::Narrow(d) => u128::from(d),
                NonNormDistance::Wide(d) => u128::try_from(d).expect("a distance fits u128 here"),
            };
            (squared, j)
        }
        None => (0, u64::MAX),
    }
}

fn plain_neighbor(neighbor: NonNormNeighbor) -> (u128, u64) {
    match neighbor.index().get() {
        Some(j) => (neighbor.squared_ticks(), j),
        None => (0, u64::MAX),
    }
}

/// A missing sample, then ticks near `2⁶²` that differ by a few: the series' present range is
/// tiny, but the forward-fill's `0` before the first sample would span `2⁶²`.
fn leading_gap_samples(n: usize) -> Vec<Option<i64>> {
    (0..n)
        .map(|t| (t > 0).then(|| (1i64 << 62) + (t * t % 7) as i64))
        .collect()
}

#[test]
fn a_missing_leading_sample_is_not_a_tick_for_the_unrepresentable_rule() {
    let (n, m) = (50, 16);
    let samples = leading_gap_samples(n);
    let series = TickSeries::from_samples(&samples, m);
    assert_eq!(
        first_unrepresentable_row(&series, None, m),
        series.windows()
    );

    let expected: Vec<_> = naive_left_profile_nonnorm(&samples, m)
        .into_iter()
        .map(|(_, entry)| plain(entry))
        .collect();
    assert!(expected.iter().any(|&(_, j)| j != u64::MAX));
    for threads in [1, 3] {
        let (rows, _) = left_rows(
            &series,
            nz(m),
            nz(threads),
            None,
            None,
            Single { right: None },
            Vec::new(),
        );
        let rows: Vec<_> = rows
            .into_iter()
            .map(|(_, neighbor)| plain_neighbor(neighbor))
            .collect();
        assert_eq!(rows, expected, "left, threads {threads}");
    }
}

#[test]
fn a_missing_leading_sample_of_the_other_series_is_not_a_tick_in_a_join() {
    let (n, m) = (50, 16);
    let a_samples: Vec<Option<i64>> = (0..n)
        .map(|t| Some((1i64 << 62) + (t * 3 % 5) as i64))
        .collect();
    let b_samples = leading_gap_samples(n);
    let (a, b) = (
        TickSeries::from_samples(&a_samples, m),
        TickSeries::from_samples(&b_samples, m),
    );
    let expected: Vec<_> = naive_ab_join_nonnorm(&a_samples, &b_samples, m)
        .into_iter()
        .map(|(_, entry)| plain(entry))
        .collect();
    assert!(expected.iter().any(|&(_, j)| j != u64::MAX));
    let (rows, _) = join_rows(&a, &b, nz(m), nz(2), None, None);
    let rows: Vec<_> = rows
        .into_iter()
        .map(|(_, neighbor)| plain_neighbor(neighbor))
        .collect();
    assert_eq!(rows, expected);
}

#[test]
fn the_rule_counts_present_ticks_only() {
    // Present ticks span `2⁶³` from the third sample on: `4·R² = 2¹²⁸` for `m = 4`.
    let samples: Vec<Option<i64>> = [None, None, Some(-(1i64 << 62)), Some(1 << 62)]
        .into_iter()
        .chain((0..10).map(|t| Some(t % 3)))
        .collect();
    let series = TickSeries::from_samples(&samples, 4);
    assert_eq!(series.present_range(), Some((-(1i64 << 62), 1 << 62)));
    assert_eq!(first_unrepresentable_row(&series, None, 4), 0);
    let all_missing = TickSeries::from_samples(&[None; 6], 4);
    assert_eq!(all_missing.present_range(), None);
    assert_eq!(first_unrepresentable_row(&all_missing, None, 4), 3);
}

/// Which walk a `u64` tile runs in [`walk_tile`].
#[derive(Clone, Copy, Debug)]
enum Walk {
    Lanes(usize),
    Diagonals,
}

type Nearests = Vec<(u128, u64)>;

/// Runs one tile of `tiles` as [`Tiles::run`] does, through `walk`: each band row's nearest and
/// each column's nearest later row. `None` when the tile is not a `u64` one.
fn walk_tile(tiles: &Tiles<Single>, tile: &Tile, walk: Walk) -> Option<(Nearests, Nearests)> {
    let columns = tiles.shape.columns(tile);
    let rows = tile.rows.clone();
    if rows.is_empty() || columns.is_empty() {
        return None;
    }
    let (width, fit) = tiles.width(rows.clone(), &columns);
    if width != RowWidth::Narrow {
        return None;
    }
    let mut scratch = Scratch::<Single> {
        rows: (0..rows.len()).map(|_| tiles.keep.new_row()).collect(),
        first_column: columns.start,
        columns: tiles.keep.new_columns(),
        residues: Vec::new(),
    };
    tiles
        .keep
        .reset_columns(&mut scratch.columns, columns.len(), rows.len());
    match walk {
        Walk::Lanes(chunk) => tiles.walk_lanes(tile, rows, &mut scratch, chunk, fit),
        Walk::Diagonals => tiles.walk::<u64>(tile, rows, &mut scratch),
    }
    let book = RightMins::new(tiles.cols.windows());
    Single { right: Some(&book) }.merge_columns(columns.start, &mut scratch.columns);
    let finished = scratch
        .rows
        .into_iter()
        .map(|row| plain_neighbor(row.finish()))
        .collect();
    let right = book.finish().into_iter().map(plain_neighbor).collect();
    Some((finished, right))
}

/// Ticks around one level, a few apart or at the ends of the widest range a `u64` tile can hold
/// at its `m`; some missing.
fn narrow_samples() -> impl Strategy<Value = (Vec<Option<i64>>, usize)> {
    let level = prop_oneof![Just(0i64), Just(-(1i64 << 62)), Just((1i64 << 62) - 3)];
    let spread = prop_oneof![Just(4i64), Just(i64::from(i32::MAX)), Just(1i64 << 31)];
    (level, spread, 1usize..=6).prop_flat_map(|(level, spread, m)| {
        let spread = if m > 3 {
            spread.min(i64::from(i32::MAX))
        } else {
            spread
        };
        let tick = prop_oneof![
            1 => Just(None),
            2 => Just(Some(level)),
            2 => Just(Some(level + spread)),
            4 => (0..=spread).prop_map(move |t| Some(level + t)),
        ];
        (prop::collection::vec(tick, 1..48), Just(m))
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    /// Any tile of a left profile or a join, in any chunk length, over all of its slots or a
    /// chosen set: the lane walk leaves every row's and every column's nearest exactly as the
    /// diagonal walk does.
    #[test]
    fn the_lane_walk_equals_the_diagonal_walk(
        (samples, m) in narrow_samples(),
        other in prop::collection::vec(prop::option::of(0i64..9), 1..30),
        join in any::<bool>(),
        cut in (0usize..64, 0usize..64, 0usize..64, 0usize..64),
        chunk in 1usize..10,
        chosen in prop::option::of(prop::collection::vec(0usize..64, 0..12)),
    ) {
        let series = TickSeries::from_samples(&samples, m);
        let level = samples.iter().flatten().next().copied().unwrap_or(0);
        let other: Vec<Option<i64>> = other.iter().map(|t| t.map(|t| level + t)).collect();
        let other = TickSeries::from_samples(&other, m);
        let windows = series.windows();
        let exclusion = crate::exclusion::zone(m);
        // A band's rows, and the slots the cut gives its tiles (see `TileCut`).
        let (cols, shape, first_row) = if join {
            (&other, Shape::Join { n_b: other.windows() }, 0)
        } else {
            (&series, Shape::Triangle { exclusion }, exclusion + 1)
        };
        prop_assume!(windows > first_row && cols.windows() > 0);
        let span = windows - first_row;
        let (r0, r1) = (first_row + cut.0 % span, first_row + cut.1 % span);
        let rows = r0.min(r1)..r0.max(r1) + 1;
        let lanes = if join {
            rows.start..rows.end + cols.windows() - 1
        } else {
            0..rows.end - 1 - exclusion
        };
        let (qa, qb) = (cut.2 % lanes.len(), cut.3 % lanes.len());
        let (qa, qb) = (lanes.start + qa.min(qb), lanes.start + qa.max(qb) + 1);
        let runs = chosen.map(SlotRuns::from_slots);
        let book = RightMins::new(cols.windows());
        let tiles = Tiles {
            rows: &series,
            cols,
            m,
            shape,
            unrepresentable_from: windows,
            keep: Single { right: Some(&book) },
            runs: runs.as_ref(),
        };
        let tile = Tile { band: 0, rows, column: 0, qa, qb };
        let expected = walk_tile(&tiles, &tile, Walk::Diagonals);
        prop_assume!(expected.is_some());
        prop_assert_eq!(walk_tile(&tiles, &tile, Walk::Lanes(chunk)), expected);
    }
}
