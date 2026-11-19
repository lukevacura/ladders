#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Every z-normalized batch engine equals its exact reference on bits on series made of exact
//! ties: constants, constant runs between levels, periodic repeats, few-level noise, exact repeats
//! at an offset, and scaled and negated copies (whose correlation is exactly one and minus one),
//! with NaN gaps at the ends and in the middle. Ties go to the smaller index at every thread
//! count, tiling and `k`: the engines skip a pair that ties pairs they already hold, and these
//! are the series where almost every pair does.

use core::num::NonZeroUsize;

use ladde::batch::{
    ab_join_report_tiled, ab_join_topk_tiled, left_profile_report_tiled, left_topk_profile_tiled,
    two_sided_profile_tiled, two_sided_topk_profile_tiled,
};
use ladde::entry::{Entry, TwoSided, TwoSidedTopk};
use ladde::naive::{
    naive_ab_join, naive_ab_join_topk, naive_left_profile, naive_left_topk, naive_two_sided,
    naive_two_sided_topk,
};
use ladde::quantize::{Scale, to_ticks};

type EntryBits = (u64, u64, u64);

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn entry_bits(entry: Entry) -> EntryBits {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn bits(profile: &[(u64, Entry)]) -> Vec<(u64, EntryBits)> {
    profile
        .iter()
        .map(|&(row, e)| (row, entry_bits(e)))
        .collect()
}

fn topk_bits(profile: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<EntryBits>)> {
    profile
        .iter()
        .map(|(row, es)| (*row, es.iter().copied().map(entry_bits).collect()))
        .collect()
}

fn sides_bits(profile: &[(u64, TwoSided)]) -> Vec<(u64, [EntryBits; 3])> {
    profile
        .iter()
        .map(|&(row, s)| (row, [s.left, s.right, s.nearest].map(entry_bits)))
        .collect()
}

type SidesTopk = (u64, [EntryBits; 2], Vec<EntryBits>);

fn sides_topk_bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<SidesTopk> {
    profile
        .iter()
        .map(|(row, s)| {
            (
                *row,
                [s.left, s.right].map(entry_bits),
                s.nearest.iter().copied().map(entry_bits).collect(),
            )
        })
        .collect()
}

/// A small deterministic generator, so that the series are fixed.
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

fn levels(len: usize, seed: u64, count: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.next(count) as f64).collect()
}

fn runs(len: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    while out.len() < len {
        let level = rng.next(3) as f64;
        let run = 1 + rng.next(9) as usize;
        out.extend(std::iter::repeat_n(level, run));
    }
    out.truncate(len);
    out
}

fn gaps(mut series: Vec<f64>) -> Vec<f64> {
    let n = series.len();
    for at in [0, 1, n / 3, n / 3 + 1, n - 2, n - 1] {
        series[at] = f64::NAN;
    }
    series
}

/// The tie-heavy corpus: each series by name.
fn corpus() -> Vec<(&'static str, Vec<f64>)> {
    let walk = levels(40, 7, 6);
    let affine: Vec<f64> = walk
        .iter()
        .copied()
        .chain(walk.iter().map(|x| 2.0 * x + 1.0))
        .chain(walk.iter().map(|x| -x))
        .chain(walk.iter().copied())
        .collect();
    let offset: Vec<f64> = walk
        .iter()
        .chain(walk.iter())
        .chain(walk.iter())
        .map(|x| x + 5.0)
        .collect();
    let mut out = vec![
        ("constant", vec![7.0; 90]),
        (
            "runs between levels",
            (0..120).map(|k| ((k / 13) % 3) as f64).collect(),
        ),
        ("random runs", runs(130, 11)),
        ("period 2", (0..90).map(|k| (k % 2) as f64).collect()),
        (
            "period 4",
            (0..90).map(|k| (k % 4 * 5 % 7) as f64).collect(),
        ),
        (
            "period 7",
            (0..100).map(|k| (k % 7 * 3 % 5) as f64).collect(),
        ),
        ("two levels", levels(110, 3, 2)),
        ("few levels", levels(120, 5, 4)),
        ("scaled and negated copies", affine),
        ("repeats at an offset", offset),
    ];
    let with_gaps: Vec<_> = out
        .iter()
        .map(|(name, series)| (*name, gaps(series.clone())))
        .collect();
    out.extend(with_gaps);
    out
}

fn ticks(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

/// Tile shapes `(band_rows, chunk_lanes, tile_lanes)`, down to one lane and past the series.
fn shapes(n: usize) -> [(usize, usize, usize); 4] {
    [(1, 1, 1), (3, 4, 8), (8, 5, 5), (n + 5, n + 5, n + 5)]
}

const THREADS: [usize; 2] = [1, 3];
const KS: [usize; 3] = [1, 2, 5];
const MS: [usize; 2] = [3, 8];

#[test]
fn the_left_and_two_sided_profiles_equal_the_reference() {
    let scale = Scale::new(1.0).unwrap();
    for (name, series) in corpus() {
        let samples = ticks(&series, scale);
        for m in MS {
            let left = bits(&naive_left_profile(&samples[..], m));
            let both = sides_bits(&naive_two_sided(&samples[..], m));
            for threads in THREADS {
                for (band, chunk, tile) in shapes(series.len()) {
                    let at = format!(
                        "{name}, m {m}, threads {threads}, shape {:?}",
                        (band, chunk, tile)
                    );
                    let (actual, _) = left_profile_report_tiled(
                        &series[..],
                        nz(m),
                        scale,
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                    );
                    assert_eq!(bits(&actual), left, "left, {at}");
                    let (actual, _) = two_sided_profile_tiled(
                        &series[..],
                        nz(m),
                        scale,
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                    );
                    assert_eq!(sides_bits(&actual), both, "two-sided, {at}");
                }
            }
        }
    }
}

#[test]
fn the_top_k_profiles_equal_the_reference() {
    let scale = Scale::new(1.0).unwrap();
    for (name, series) in corpus() {
        let samples = ticks(&series, scale);
        for m in MS {
            for k in KS {
                let left = topk_bits(&naive_left_topk(&samples[..], m, nz(k)));
                let both = sides_topk_bits(&naive_two_sided_topk(&samples[..], m, nz(k)));
                for threads in THREADS {
                    for (band, chunk, tile) in shapes(series.len()) {
                        let at = format!(
                            "{name}, m {m}, k {k}, threads {threads}, shape {:?}",
                            (band, chunk, tile)
                        );
                        let (actual, _) = left_topk_profile_tiled(
                            &series[..],
                            nz(m),
                            scale,
                            nz(threads),
                            nz(band),
                            nz(chunk),
                            nz(tile),
                            nz(k),
                        );
                        assert_eq!(topk_bits(&actual), left, "left top-k, {at}");
                        let (actual, _) = two_sided_topk_profile_tiled(
                            &series[..],
                            nz(m),
                            scale,
                            nz(threads),
                            nz(band),
                            nz(chunk),
                            nz(tile),
                            nz(k),
                        )
                        .unwrap();
                        assert_eq!(sides_topk_bits(&actual), both, "two-sided top-k, {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn the_joins_equal_the_reference() {
    let scale = Scale::new(1.0).unwrap();
    let all = corpus();
    for (i, (name, a)) in all.iter().enumerate() {
        // Each series against the next one and against itself shifted: repeats across the sides.
        let (other, b) = &all[(i + 3) % all.len()];
        let mut shifted = a.clone();
        shifted.rotate_left(a.len() / 3);
        for (side, b) in [(*other, b), ("itself, shifted", &shifted)] {
            let (samples_a, samples_b) = (ticks(a, scale), ticks(b, scale));
            for m in MS {
                let join = bits(&naive_ab_join(&samples_a[..], &samples_b[..], m));
                for k in [1, 3] {
                    let topk = topk_bits(&naive_ab_join_topk(
                        &samples_a[..],
                        &samples_b[..],
                        m,
                        nz(k),
                    ));
                    for threads in THREADS {
                        for (band, chunk, tile) in shapes(a.len().min(b.len())) {
                            let at = format!(
                                "{name} against {side}, m {m}, threads {threads}, shape {:?}",
                                (band, chunk, tile)
                            );
                            if k == 1 {
                                let (actual, _) = ab_join_report_tiled(
                                    &a[..],
                                    &b[..],
                                    nz(m),
                                    scale,
                                    scale,
                                    nz(threads),
                                    nz(band),
                                    nz(chunk),
                                    nz(tile),
                                );
                                assert_eq!(bits(&actual), join, "join, {at}");
                            }
                            let (actual, _) = ab_join_topk_tiled(
                                &a[..],
                                &b[..],
                                nz(m),
                                scale,
                                scale,
                                nz(threads),
                                nz(band),
                                nz(chunk),
                                nz(tile),
                                nz(k),
                            )
                            .unwrap();
                            assert_eq!(topk_bits(&actual), topk, "join top-{k}, {at}");
                        }
                    }
                }
            }
        }
    }
}
