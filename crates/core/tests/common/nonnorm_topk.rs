//! An exact reference for the non-normalized top-k profiles that shares nothing with the engines
//! or the oracle: every eligible pair's squared distance summed in `u128` from the ticks, sorted
//! by `(squared distance, index)`, and cut at `k`. Ticks stay within `±2^53` and windows within
//! a few hundred samples, so no sum reaches `2^128`.
#![allow(dead_code)]

use core::num::NonZeroUsize;

use ladde::nonnorm::{NonNormNeighbor, NonNormTwoSidedTopk};
use ladde::quantize::{Scale, to_ticks};

/// `(squared distance in ticks, neighbor start)`.
pub type Pair = (u128, u64);

/// One row of a two-sided top-k profile: the left and right nearest (`None` for no neighbor on
/// that side) and the `k` nearest over both sides.
pub type Sides = (u64, Option<Pair>, Option<Pair>, Vec<Pair>);

pub fn plain(neighbor: NonNormNeighbor) -> Option<Pair> {
    neighbor
        .index()
        .get()
        .map(|index| (neighbor.squared_ticks(), index))
}

pub fn batch_sides(profile: &[(u64, NonNormTwoSidedTopk)]) -> Vec<Sides> {
    profile
        .iter()
        .map(|(row, sides)| {
            (
                *row,
                plain(sides.left),
                plain(sides.right),
                sides.nearest.iter().copied().filter_map(plain).collect(),
            )
        })
        .collect()
}

pub fn batch_join(profile: &[(u64, Vec<NonNormNeighbor>)]) -> Vec<(u64, Vec<Pair>)> {
    profile
        .iter()
        .map(|(row, kept)| (*row, kept.iter().copied().filter_map(plain).collect()))
        .collect()
}

struct Ticked {
    ticks: Vec<Option<i64>>,
    m: usize,
}

impl Ticked {
    fn new(series: &[f64], scale: Scale, m: usize) -> Self {
        Self {
            ticks: series.iter().map(|&x| to_ticks(x, scale)).collect(),
            m,
        }
    }

    fn windows(&self) -> usize {
        (self.ticks.len() + 1).saturating_sub(self.m)
    }

    fn window(&self, start: usize) -> Option<Vec<i64>> {
        self.ticks[start..start + self.m].iter().copied().collect()
    }
}

fn squared(a: &[i64], b: &[i64]) -> u128 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let difference = (i128::from(x) - i128::from(y)).unsigned_abs();
            difference * difference
        })
        .sum()
}

fn nearest(mut pairs: Vec<Pair>, k: usize) -> Vec<Pair> {
    pairs.sort_unstable();
    pairs.truncate(k);
    pairs
}

/// The exact two-sided top-k profile of `series` at tick size `scale`.
pub fn exact_two_sided(series: &[f64], scale: Scale, m: usize, k: NonZeroUsize) -> Vec<Sides> {
    let ticked = Ticked::new(series, scale, m);
    let exclusion = ladde::exclusion::zone(m);
    let windows: Vec<_> = (0..ticked.windows()).map(|i| ticked.window(i)).collect();
    (0..windows.len())
        .map(|row| {
            let Some(own) = &windows[row] else {
                return (row as u64, None, None, Vec::new());
            };
            let pairs = |range: core::ops::Range<usize>| -> Vec<Pair> {
                range
                    .filter_map(|j| Some((squared(own, windows.get(j)?.as_ref()?), j as u64)))
                    .collect()
            };
            let left = pairs(0..row.saturating_sub(exclusion));
            let right = pairs((row + exclusion + 1).min(windows.len())..windows.len());
            let both = nearest([left.clone(), right.clone()].concat(), k.get());
            (
                row as u64,
                nearest(left, 1).first().copied(),
                nearest(right, 1).first().copied(),
                both,
            )
        })
        .collect()
}

/// The exact top-k AB-join of `a` against `b` (the same tick size), no exclusion zone.
pub fn exact_join(
    a: &[f64],
    b: &[f64],
    scale: Scale,
    m: usize,
    k: NonZeroUsize,
) -> Vec<(u64, Vec<Pair>)> {
    let (a, b) = (Ticked::new(a, scale, m), Ticked::new(b, scale, m));
    let b_windows: Vec<_> = (0..b.windows()).map(|j| b.window(j)).collect();
    (0..a.windows())
        .map(|row| {
            let pairs = match a.window(row) {
                Some(own) => b_windows
                    .iter()
                    .enumerate()
                    .filter_map(|(j, window)| Some((squared(&own, window.as_ref()?), j as u64)))
                    .collect(),
                None => Vec::new(),
            };
            (row as u64, nearest(pairs, k.get()))
        })
        .collect()
}

/// The exact left top-k of `series`: each window's `k` nearest earlier windows outside the
/// exclusion zone, and, with `horizon`, only those starting inside the last `horizon` samples up
/// to the window's own end.
pub fn exact_left(
    series: &[f64],
    scale: Scale,
    m: usize,
    k: NonZeroUsize,
    horizon: Option<usize>,
) -> Vec<(u64, Vec<Pair>)> {
    let ticked = Ticked::new(series, scale, m);
    let exclusion = ladde::exclusion::zone(m);
    let windows: Vec<_> = (0..ticked.windows()).map(|i| ticked.window(i)).collect();
    (0..windows.len())
        .map(|row| {
            let Some(own) = &windows[row] else {
                return (row as u64, Vec::new());
            };
            let first = horizon.map_or(0, |w| (row + m).saturating_sub(w));
            let pairs = (first..row.saturating_sub(exclusion))
                .filter_map(|j| Some((squared(own, windows[j].as_ref()?), j as u64)))
                .collect();
            (row as u64, nearest(pairs, k.get()))
        })
        .collect()
}

/// A stream row's kept neighbors as pairs.
pub fn kept_pairs(kept: &[NonNormNeighbor]) -> Vec<Pair> {
    kept.iter().copied().filter_map(plain).collect()
}
