#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The tile engines over a subset of the diagonals equal the exact reference over exactly those
//! pairs, on bits, at every thread count and tile shape, whichever diagonals the subset holds:
//! none, one, runs, every other one, all of them, and numbers that are no diagonal at all.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/route.rs"]
mod route;

use ladde::batch::nonnorm::two_sided_profile_over_tiled;
use ladde::batch::two_sided_topk_profile_over_tiled;
use ladde::entry::{Entry, TwoSidedTopk};
use ladde::naive::{naive_two_sided_nonnorm_on, naive_two_sided_topk_on};
use ladde::nonnorm::NonNormTwoSided;
use ladde::quantize::Scale;
use proptest::prelude::*;
use route::{cases, nz, samples};

type EntryBits = (u64, u64, u64);

fn entry_bits(entry: Entry) -> EntryBits {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<(u64, [EntryBits; 2], Vec<EntryBits>)> {
    profile
        .iter()
        .map(|(row, sides)| {
            (
                *row,
                [sides.left, sides.right].map(entry_bits),
                sides.nearest.iter().copied().map(entry_bits).collect(),
            )
        })
        .collect()
}

/// A neighbor as `(squared ticks, index)`, none as `(0, u64::MAX)`.
fn plain(neighbor: ladde::nonnorm::NonNormNeighbor) -> (u128, u64) {
    match neighbor.index().get() {
        Some(j) => (neighbor.squared_ticks(), j),
        None => (0, u64::MAX),
    }
}

fn nonnorm_bits(profile: &[(u64, NonNormTwoSided)]) -> Vec<(u64, [(u128, u64); 3])> {
    profile
        .iter()
        .map(|&(row, s)| (row, [s.left, s.right, s.nearest].map(plain)))
        .collect()
}

/// The diagonals of a self-join of `windows` windows at window length `m`, as the subsets the
/// tests pick from: none, one, a run, every other, a block, all, and numbers that are no diagonal.
fn subsets(windows: usize, m: usize, salt: u64) -> Vec<Vec<i64>> {
    let first = ladde::exclusion::zone(m) as i64 + 1;
    let all: Vec<i64> = (first..windows as i64).collect();
    let pick = |mut keep: Box<dyn FnMut(usize, i64) -> bool>| -> Vec<i64> {
        all.iter()
            .enumerate()
            .filter(|&(at, &d)| keep(at, d))
            .map(|(_, &d)| d)
            .collect()
    };
    let random = |per_mille: u64| {
        let mut state = salt | 1;
        pick(Box::new(move |_, _| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % 1000 < per_mille
        }))
    };
    let mut subsets = vec![
        vec![],
        all.first().copied().into_iter().collect(),
        all.last().copied().into_iter().collect(),
        pick(Box::new(|at, _| at % 2 == 0)),
        pick(Box::new(|at, _| at % 3 != 1)),
        pick(Box::new(|at, _| (3..9).contains(&at))),
        random(100),
        random(500),
        random(900),
        all.clone(),
        vec![
            -3,
            0,
            1,
            first - 1,
            windows as i64,
            windows as i64 + 5,
            i64::MIN,
            i64::MAX,
        ],
    ];
    subsets.push(
        all.iter()
            .rev()
            .copied()
            .chain(all.iter().copied())
            .collect(),
    );
    subsets
}

const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)];
const THREADS: [usize; 3] = [1, 2, 7];

fn assert_znorm(series: &[f64], m: usize, scale: Scale, k: usize, salt: u64, tag: &str) {
    let windows = (series.len() + 1).saturating_sub(m);
    let samples = samples(series, scale);
    for diagonals in subsets(windows, m, salt) {
        let expected = bits(&naive_two_sided_topk_on(&samples, m, nz(k), &diagonals));
        for threads in THREADS {
            for (band, chunk, tile) in SHAPES {
                let (profile, _) = two_sided_topk_profile_over_tiled(
                    series,
                    nz(m),
                    scale,
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                    nz(k),
                    &diagonals,
                )
                .unwrap();
                assert_eq!(
                    bits(&profile),
                    expected,
                    "{tag}: k {k}, threads {threads}, shape {:?}, {} diagonals",
                    (band, chunk, tile),
                    diagonals.len()
                );
            }
        }
    }
}

fn assert_nonnorm(series: &[f64], m: usize, salt: u64, tag: &str) {
    let scale = Scale::new(1.0).unwrap();
    let windows = (series.len() + 1).saturating_sub(m);
    let samples = samples(series, scale);
    for diagonals in subsets(windows, m, salt) {
        let expected: Vec<_> = naive_two_sided_nonnorm_on(&samples, m, &diagonals)
            .into_iter()
            .map(|(row, s)| {
                let plain = |e| {
                    let (_, squared, index) = nonnorm::entry_row(row, e);
                    (squared, index)
                };
                (row, [s.left, s.right, s.nearest].map(plain))
            })
            .collect();
        for threads in THREADS {
            for (band, chunk, tile) in SHAPES {
                let (profile, _) = two_sided_profile_over_tiled(
                    series,
                    nz(m),
                    scale,
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                    &diagonals,
                );
                assert_eq!(
                    nonnorm_bits(&profile),
                    expected,
                    "{tag}: threads {threads}, shape {:?}, {} diagonals",
                    (band, chunk, tile),
                    diagonals.len()
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(10))]

    #[test]
    fn znorm_subsets_equal_the_reference((series, m, scale) in cases(), salt in any::<u64>()) {
        for k in [1, 3] {
            assert_znorm(&series, m, scale, k, salt, "cases");
        }
    }
}

#[test]
fn znorm_subsets_equal_the_reference_on_ties_and_gaps() {
    let scale = Scale::new(1.0).unwrap();
    let flat = vec![7.0; 70];
    let runs: Vec<f64> = (0..90).map(|k| f64::from(((k / 11) % 3) as u8)).collect();
    let period: Vec<f64> = (0..80)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    let mut gapped: Vec<f64> = (0..80)
        .map(|k| f64::from(((k * k / 5 + k) % 11) as u8))
        .collect();
    for k in [0, 1, 2, 39, 40, 41, 77, 78, 79] {
        gapped[k] = f64::NAN;
    }
    for (tag, series, m) in [
        ("flat", &flat, 4),
        ("runs", &runs, 5),
        ("period", &period, 6),
        ("gapped", &gapped, 6),
    ] {
        for k in [1, 2, 200] {
            assert_znorm(series, m, scale, k, 7, tag);
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(16))]

    #[test]
    fn nonnorm_subsets_equal_the_reference(
        series in prop_oneof![nonnorm::series(), nonnorm::tie_series()],
        m in 1usize..=6,
        salt in any::<u64>(),
    ) {
        assert_nonnorm(&series, m, salt, "series");
    }
}

/// The tiles a subset run cuts, worked out from the definition: per band of `band_rows` rows from
/// the first row with a pair, the tiles of `tile_lanes` slots that hold a visited slot, and one
/// tile with no slot for a band none of them reaches.
fn expected_tiles(
    windows: usize,
    m: usize,
    (band_rows, tile_lanes): (usize, usize),
    diagonals: &[i64],
) -> usize {
    let exclusion = ladde::exclusion::zone(m);
    let slots: Vec<usize> = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(d).ok()?.checked_sub(exclusion + 1))
        .collect();
    let mut tiles = 0;
    let mut start = exclusion + 1;
    while start < windows {
        let end = (start + band_rows).min(windows);
        let lanes = end - 1 - exclusion;
        let kept = (0..lanes)
            .step_by(tile_lanes)
            .filter(|&qa| {
                slots
                    .iter()
                    .any(|&s| (qa..(qa + tile_lanes).min(lanes)).contains(&s))
            })
            .count();
        tiles += kept.max(1);
        start = end;
    }
    tiles
}

#[test]
fn only_the_tiles_holding_a_visited_diagonal_are_cut() {
    let series: Vec<f64> = (0..120)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let (m, scale) = (6, Scale::new(1.0).unwrap());
    let windows = series.len() + 1 - m;
    for diagonals in subsets(windows, m, 3) {
        for (band, chunk, tile) in [(4, 2, 4), (7, 3, 6), (200, 5, 9)] {
            let (_, znorm) = two_sided_topk_profile_over_tiled(
                &series,
                nz(m),
                scale,
                nz(1),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(1),
                &diagonals,
            )
            .unwrap();
            let (_, nonnorm) = two_sided_profile_over_tiled(
                &series,
                nz(m),
                scale,
                nz(1),
                nz(band),
                nz(chunk),
                nz(tile),
                &diagonals,
            );
            // A tile is a whole number of chunks wide.
            let tile = tile.div_ceil(chunk) * chunk;
            let want = expected_tiles(windows, m, (band, tile), &diagonals);
            assert_eq!(
                znorm.tiles,
                want,
                "z-normalized {diagonals:?} {:?}",
                (band, tile)
            );
            assert_eq!(nonnorm.tiles, want, "non-normalized {:?}", (band, tile));
        }
    }
}
