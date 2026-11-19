//! Shared pieces of the non-normalized stream tests: series that straddle the `u64` bound, the
//! rows of a stream and of the exact reference as comparable plain values.
#![allow(dead_code)]

use core::num::NonZeroUsize;

use ladde::index::SampleIndex;
use ladde::naive::{
    NonNormDistance, NonNormEntry, naive_ab_join_nonnorm, naive_left_profile_nonnorm,
    naive_two_sided_nonnorm, windowed_nonnorm_entry,
};
use ladde::nonnorm::{NonNormNeighbor, NonNormTwoSided};
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;

/// `(row, squared distance in ticks, neighbor)`; no neighbor is `(row, 0, u64::MAX)`.
pub type Row = (u64, u128, u64);

pub fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

pub fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

pub fn stream_row((row, neighbor): (SampleIndex, NonNormNeighbor)) -> Row {
    let row = row.get().expect("a row is never NONE");
    match neighbor.index().get() {
        Some(j) => (row, neighbor.squared_ticks(), j),
        None => (row, 0, u64::MAX),
    }
}

pub fn entry_row(row: u64, entry: NonNormEntry) -> Row {
    match entry.index().get() {
        Some(j) => {
            let squared = match entry.distance() {
                NonNormDistance::Narrow(d) => u128::from(d),
                NonNormDistance::Wide(d) => u128::try_from(d).expect("a distance fits u128 here"),
            };
            (row, squared, j)
        }
        None => (row, 0, u64::MAX),
    }
}

pub fn samples(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

pub fn naive_rows(series: &[f64], scale: Scale, m: usize) -> Vec<Row> {
    naive_left_profile_nonnorm(&samples(series, scale), m)
        .into_iter()
        .map(|(row, entry)| entry_row(row, entry))
        .collect()
}

pub fn naive_window_rows(series: &[f64], scale: Scale, m: usize, w: usize) -> Vec<Row> {
    let samples = samples(series, scale);
    let rows = (samples.len() + 1).saturating_sub(m);
    (0..rows)
        .map(|row| entry_row(row as u64, windowed_nonnorm_entry(&samples, m, w, row)))
        .collect()
}

/// A batch profile's rows as comparable plain values.
pub fn batch_rows(profile: &[(u64, NonNormNeighbor)]) -> Vec<Row> {
    profile
        .iter()
        .map(|&(row, neighbor)| match neighbor.index().get() {
            Some(j) => (row, neighbor.squared_ticks(), j),
            None => (row, 0, u64::MAX),
        })
        .collect()
}

/// A row's left, right and nearest neighbor as `(squared distance, index)`, none as
/// `(0, u64::MAX)`.
pub type Sides = (u64, [(u128, u64); 3]);

fn plain(row: Row) -> (u128, u64) {
    (row.1, row.2)
}

pub fn batch_sides(profile: &[(u64, NonNormTwoSided)]) -> Vec<Sides> {
    profile
        .iter()
        .map(|&(row, sides)| {
            let rows = [sides.left, sides.right, sides.nearest]
                .map(|neighbor| batch_rows(&[(row, neighbor)])[0]);
            (row, rows.map(plain))
        })
        .collect()
}

pub fn naive_sides(series: &[f64], scale: Scale, m: usize) -> Vec<Sides> {
    naive_two_sided_nonnorm(&samples(series, scale), m)
        .into_iter()
        .map(|(row, sides)| {
            let rows = [sides.left, sides.right, sides.nearest].map(|e| entry_row(row, e));
            (row, rows.map(plain))
        })
        .collect()
}

pub fn naive_join_rows(a: &[f64], b: &[f64], scale: Scale, m: usize) -> Vec<Row> {
    naive_ab_join_nonnorm(&samples(a, scale), &samples(b, scale), m)
        .into_iter()
        .map(|(row, entry)| entry_row(row, entry))
        .collect()
}

/// Tile shapes `(band_rows, chunk_lanes, tile_lanes)`, down to one lane and past the series.
pub fn shapes(n: usize) -> [(usize, usize, usize); 6] {
    [
        (1, 1, 1),
        (2, 3, 6),
        (3, 4, 8),
        (8, 5, 5),
        (n + 5, 2, 4),
        (4, n + 5, n + 5),
    ]
}

pub const LEVELS: [i64; 6] = [0, 1 << 20, 1 << 31, 1 << 33, -(1 << 40), 1 << 53];

/// Walks that stay, repeat, step by a few ticks, jump between levels (some past the `u64` bound
/// for small `m`) and drop samples; scale 1 reads them back exactly.
pub fn series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec((0u8..10, -3i64..=3, 0usize..LEVELS.len()), 0..60).prop_map(|ops| {
        let mut current = 0i64;
        ops.into_iter()
            .map(|(op, noise, level)| match op {
                0 | 1 => f64::NAN,
                2..=4 => current as f64,
                5 => {
                    current = LEVELS[level];
                    current as f64
                }
                _ => {
                    current = (current + noise).clamp(-(1 << 53), 1 << 53);
                    current as f64
                }
            })
            .collect()
    })
}

/// Series made of ties: constants, flat runs between levels, short periods at a level, and exact
/// repeats at offsets.
pub fn tie_series() -> impl Strategy<Value = Vec<f64>> {
    let periodic = (
        1usize..8,
        prop::collection::vec(-5i64..=5, 8),
        0usize..LEVELS.len(),
        0usize..60,
    )
        .prop_map(|(period, pattern, level, len)| {
            (0..len)
                .map(|k| (LEVELS[level] + pattern[k % period]) as f64)
                .collect::<Vec<_>>()
        });
    let runs = (
        1usize..9,
        0usize..LEVELS.len(),
        0usize..LEVELS.len(),
        0usize..60,
    )
        .prop_map(|(run, a, b, len)| {
            (0..len)
                .map(|k| LEVELS[if (k / run) % 2 == 0 { a } else { b }] as f64)
                .collect::<Vec<_>>()
        });
    let few_levels = prop::collection::vec(0usize..3, 0..60)
        .prop_map(|picks| picks.into_iter().map(|p| (p as i64 * 7) as f64).collect());
    prop_oneof![periodic, runs, few_levels]
}

/// Plateaus at levels around the `u64` bound, each long enough that a bounded stream sees the
/// shock leave its window and the stream return to the narrow arithmetic.
pub fn shocked() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        (0usize..LEVELS.len(), 1usize..30, -3i64..=3, 0u8..12),
        3..14,
    )
    .prop_map(|segments| {
        let mut series = Vec::new();
        for (level, len, noise, kind) in segments {
            for k in 0..len {
                let wobble = (k as i64 * 7 + noise) % 5 - 2;
                series.push(if kind == 0 && k % 5 == 3 {
                    f64::NAN
                } else {
                    (LEVELS[level] + wobble).clamp(-(1 << 53), 1 << 53) as f64
                });
            }
        }
        series
    })
}

/// Small integers with exact copies planted and gaps of both kinds.
pub fn planted() -> impl Strategy<Value = (Vec<f64>, usize)> {
    (
        proptest::collection::vec(-9i32..9, 12..48),
        proptest::collection::vec(
            (any::<prop::sample::Index>(), any::<prop::sample::Index>()),
            0..3,
        ),
        proptest::collection::vec((any::<prop::sample::Index>(), any::<bool>()), 0..3),
        2usize..7,
    )
        .prop_map(|(base, plants, gaps, m)| {
            let mut series: Vec<f64> = base.iter().map(|&x| f64::from(x)).collect();
            let n = series.len();
            for (from, to) in plants {
                let (from, to) = (from.index(n - m), to.index(n - m));
                for k in 0..m {
                    series[to + k] = series[from + k];
                }
            }
            for (gap, infinite) in gaps {
                series[gap.index(n)] = if infinite { f64::INFINITY } else { f64::NAN };
            }
            (series, m)
        })
}
