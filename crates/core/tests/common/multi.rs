//! Shared pieces of the multi-dimensional batch tests: seeded multi-dimensional inputs full of
//! exact ties and gaps, the tile shapes, and profiles as comparable plain values.
#![allow(dead_code)]

use core::num::NonZeroUsize;

use ladde::multidim::{DimensionDistance, MultiProfile};
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;

pub const THREADS: [usize; 3] = [1, 2, 7];

pub fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

pub fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

/// Tile shapes `(band_rows, chunk_lanes, tile_lanes)`, down to one lane and past the series.
pub fn shapes(n: usize) -> [(usize, usize, usize); 5] {
    [
        (1, 1, 1),
        (2, 3, 6),
        (3, 4, 8),
        (n + 5, 2, 4),
        (4, n + 5, n + 5),
    ]
}

/// A multi-dimensional input: its dimensions, window length, `include` and `discords`.
#[derive(Clone, Debug)]
pub struct Input {
    pub dimensions: Vec<Vec<f64>>,
    pub m: usize,
    pub include: Vec<usize>,
    pub discords: bool,
}

pub fn ticks(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

pub fn slices(series: &[Vec<f64>]) -> Vec<&[f64]> {
    series.iter().map(Vec::as_slice).collect()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % below
    }
}

/// One dimension of a kind, `n` samples long, from `seed`: walks, constants, flat runs between
/// levels, short periods, few levels, exact repeats at offsets, a high offset.
pub fn kind_series(kind: u8, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03);
    let motif: Vec<i64> = (0..8).map(|_| rng.next(50) as i64).collect();
    let pattern: Vec<i64> = (0..7).map(|_| rng.next(40) as i64).collect();
    let period = 2 + (seed % 6) as usize;
    let mut level = 0i64;
    (0..n)
        .map(|t| {
            (match kind % 9 {
                0 => {
                    level += rng.next(21) as i64 - 10;
                    level
                }
                1 => 5,
                2 => (t / 7 % 3) as i64 * 4,
                3 => pattern[t % period],
                4 => rng.next(3) as i64,
                5 => motif[t % 8] + 100 * (t / 8 % 3) as i64,
                6 => {
                    level += rng.next(5) as i64 - 2;
                    (1i64 << 40) + level
                }
                7 => {
                    level += rng.next(21) as i64 - 10;
                    level
                }
                _ => {
                    // Runs at levels around the `u64` bound of a non-normalized distance.
                    const LEVELS: [i64; 6] = [0, 1 << 20, 1 << 31, 1 << 33, -(1 << 40), 1 << 53];
                    LEVELS[(seed as usize + t / 5) % 6] + rng.next(5) as i64
                }
            }) as f64
        })
        .collect()
}

/// Inputs of one to four dimensions, some an exact scaled copy of an earlier one (tied
/// distances in every cell), with gaps anywhere including the first and last samples.
pub fn input(kinds: u8) -> impl Strategy<Value = Input> {
    (
        1usize..=4,
        0usize..48,
        3usize..=7,
        prop::collection::vec((0u8..kinds, any::<u64>(), 0u8..4, 1u8..=3), 4),
        prop::collection::vec((0u8..4, any::<u16>()), 0..6),
        prop::collection::vec(0usize..4, 0..4),
        any::<bool>(),
    )
        .prop_map(|(d, n, m, specs, gaps, include, discords)| {
            let mut dimensions: Vec<Vec<f64>> = Vec::new();
            for (a, &(kind, seed, copy, factor)) in specs.iter().take(d).enumerate() {
                let series = if copy == 0 && a > 0 {
                    let source = &dimensions[(seed % a as u64) as usize];
                    source
                        .iter()
                        .map(|&x| x * f64::from(factor) + 3.0)
                        .collect()
                } else {
                    kind_series(kind, n, seed)
                };
                dimensions.push(series);
            }
            for (a, at) in gaps {
                if n > 0 {
                    let series = &mut dimensions[usize::from(a) % d];
                    let position = match at % 8 {
                        0 => 0,
                        1 => n - 1,
                        _ => usize::from(at) % n,
                    };
                    series[position] = f64::NAN;
                }
            }
            Input {
                dimensions,
                m,
                include: include.into_iter().map(|a| a % d).collect(),
                discords,
            }
        })
}

/// Exact repeats, constants and periods across all dimensions of one input, and a gap at each
/// end: the hand-made tie cases.
pub fn tie_inputs() -> Vec<Input> {
    let mut all = Vec::new();
    for (n, m) in [(40usize, 3usize), (60, 5), (45, 8)] {
        for &discords in &[false, true] {
            for include in [vec![], vec![1], vec![2, 0]] {
                let mut dimensions: Vec<Vec<f64>> = (0..3u8)
                    .map(|a| {
                        kind_series(
                            [1, 2, 3, 4, 5, 3][usize::from(a) + usize::from(discords) * 3],
                            n,
                            11 + u64::from(a),
                        )
                    })
                    .collect();
                let copy: Vec<f64> = dimensions[0].iter().map(|&x| 2.0 * x + 1.0).collect();
                dimensions.push(copy);
                all.push(Input {
                    dimensions: dimensions.clone(),
                    m,
                    include: include.clone(),
                    discords,
                });
                dimensions[1][0] = f64::NAN;
                dimensions[2][n - 1] = f64::NAN;
                all.push(Input {
                    dimensions,
                    m,
                    include,
                    discords,
                });
            }
        }
    }
    all
}

/// A profile as plain values: per `k` and window, the mean's bits and the neighbor's index
/// (`u64::MAX` for none).
pub fn bits<D: DimensionDistance>(profile: &MultiProfile<D>) -> Vec<Vec<(u64, u64)>> {
    profile
        .distance
        .iter()
        .zip(&profile.index)
        .map(|(distances, indices)| {
            distances
                .iter()
                .zip(indices)
                .map(|(distance, index)| {
                    (distance.to_f64().to_bits(), index.get().unwrap_or(u64::MAX))
                })
                .collect()
        })
        .collect()
}

/// More than twenty dimensions (past where an in-place sort is an insertion sort) that are
/// copies of three bases, so almost every dimension ties with several others in every cell: each
/// copy is the base itself (`scaled` false, tied in either metric) or a scaled and shifted copy
/// (`scaled` true, tied under z-normalization). Some copies lose a sample, `include` and
/// `discords` as given.
pub fn tied_copies(count: usize, scaled: bool, include: Vec<usize>, discords: bool) -> Input {
    let n = 36;
    let bases = [
        kind_series(3, n, 7),
        kind_series(4, n, 8),
        kind_series(0, n, 9),
    ];
    let dimensions = (0..count)
        .map(|a| {
            let mut copy: Vec<f64> = bases[a % 3]
                .iter()
                .map(|&x| {
                    if scaled {
                        x * (1 + a % 3) as f64 + 5.0
                    } else {
                        x
                    }
                })
                .collect();
            if a % 7 == 5 {
                copy[(a * 5) % n] = f64::NAN;
            }
            copy
        })
        .collect();
    Input {
        dimensions,
        m: 4,
        include,
        discords,
    }
}
