//! The estimate screen of the multi-dimensional walk: a cell whose every mean is provably farther
//! than its row's best is settled from a lower bound on each dimension's distance, without the
//! exact distances, their order or their means. Every other cell takes the exact path, so the
//! screen decides only how fast a cell is settled, never what any slot holds.
//!
//! **Why it is sound.** Each dimension's bound `L ≤ d` is below its exact distance (see
//! [`znorm_lower`] and [`root_lower`]). The `k + 1` dimensions a mean is of are the `k + 1`
//! nearest (farthest for discords) after the `include` ones, in the exact order; whichever they
//! are, their exact sum is at least the sum of the `k + 1` smallest bounds (the largest, for
//! discords) taken the same way, so the bounds need no exact order of their own and a near-tie in
//! the dimensions' order changes nothing here. A slot's `farther` is at least its best's exact
//! mean (it is the value times the order's error ratio). A cell is settled only if, for every
//! `k`, the rounded sum of bounds exceeds `farther · scale(k)`, which covers the rounding of the
//! sum and of that product: its exact mean is then strictly farther than every slot's, so the
//! exact path would take nothing from it. A missing distance bounds as infinite: a mean that
//! holds one is missing and never taken.

use super::super::quantized::Quantized;
use crate::estimate::{F64, Producer, rank_estimate};
use crate::row::centered_cross_fast;
use alloc::vec::Vec;
use core::ops::Range;

/// `1 − 2^-50`: what a bound computed through a few correctly rounded steps is multiplied by to
/// stay below the exact value.
const SHRINK: f64 = 1.0 - 1.0 / (1u64 << 50) as f64;

/// A lower bound on the exact z-normalized distance `√(2m(1 − ρ))` of a pair whose ranking
/// estimate is `rho_hat` (made by [`F64`], so within `2^-50` of `ρ`), `two_m` being `2m`.
///
/// With `u = 2^-53`: `1 − ρ̂` rounds within `2u·1.0001`, so less [`F64`]'s declared bound
/// `2^-49` it is below `1 − ρ` by more than that; the subtraction, the product with `2m`, the
/// root and the final product each round up by at most `u`, and `(1 + u)^3 (1 − 2^-50) < 1`. A difference at or below zero
/// bounds the distance by zero.
#[inline(always)]
pub(super) fn znorm_lower(rho_hat: f64, two_m: f64) -> f64 {
    let gap = (1.0 - rho_hat) - F64::DELTA;
    libm::sqrt(two_m * gap.max(0.0)) * SHRINK
}

/// Writes [`znorm_lower`] for each cell `(row + c, column + c)` of a diagonal whose `QT` is
/// `states[c]` to `out`, in one dimension of windows of length `m`. The estimate is the ranking
/// expression plus both windows' biases (zero for a window with a variance), so that a constant
/// window's fixed correlation (`1` for two, `1/2` for one) and a gap's `−∞` (an infinite bound)
/// need no branch.
pub(super) fn znorm_bounds<'a>(
    dimension: &Quantized,
    m: usize,
    row: usize,
    column: usize,
    states: &[i64],
    out: impl Iterator<Item = &'a mut f64>,
) {
    let two_m = 2.0 * m as f64;
    let sums = dimension.sum[row..].iter().zip(&dimension.sum[column..]);
    let roots = dimension.r[row..].iter().zip(&dimension.r[column..]);
    let biases = dimension.bias[row..].iter().zip(&dimension.bias[column..]);
    let cells = states.iter().zip(sums.zip(roots).zip(biases));
    for (
        bound,
        (&qt, (((&row_sum, &column_sum), (&row_r, &column_r)), (&row_bias, &column_bias))),
    ) in out.zip(cells)
    {
        let n = centered_cross_fast(qt, m as i64, row_sum, column_sum);
        *bound = znorm_lower(
            rank_estimate(n, row_r, column_r) + row_bias + column_bias,
            two_m,
        );
    }
}

/// A lower bound on `√squared`: the conversion to `f64`, the root and the product round up by at
/// most `u` each, and `(1 + u)^{5/2} (1 − 2^-50) < 1`.
#[inline(always)]
pub(super) fn root_lower(squared: u128) -> f64 {
    libm::sqrt(squared as f64) * SHRINK
}

/// What a sum of `k + 1` non-negative bounds is compared with, as a multiple of a slot's
/// `farther`: `(k + 1)(1 + (k + 2)·2^-51)`, which exceeds `k + 1` by more than the `k` additions
/// and the product can round; infinite (nothing settles) once that rounding is no longer small.
fn sum_scale(k: usize) -> f64 {
    let unit = 1.0 / (1u64 << 53) as f64;
    let operations = (k + 2) as f64 * unit;
    if operations >= 0.01 {
        return f64::INFINITY;
    }
    (k + 1) as f64 * (1.0 + 4.0 * operations)
}

/// What settles a cell from its bounds: how the dimensions are ordered, what a sum of bounds is
/// compared with, and each row's ceilings.
///
/// A row's ceiling for `k` is the least `farther` of any slot `k` the row has held in the tiles
/// of its band this screen has walked, one after another, or of the window it starts from: the
/// exact mean of a window the row is offered, so the row's final best is at most it, wherever
/// that best comes from. It outlives the tile, so a tile's first cells are screened against what
/// the band's earlier tiles found. It is kept multiplied by `sum_scale(k)`, as a threshold.
pub(super) struct Screen {
    scale: Vec<f64>,
    include: Vec<usize>,
    rest: Vec<usize>,
    discords: bool,
    keys: Vec<f64>,
    /// `thresholds[(row - rows.start) * count + k]`: the ceiling times `scale[k]`.
    thresholds: Vec<f64>,
    rows: Range<usize>,
}

impl Screen {
    /// The screen for cells of `count` dimensions, `include` (no repeats, in range) first in the
    /// order given, the others nearest first, or farthest first if `discords`, in bands of up to
    /// `band_rows` rows.
    pub(super) fn new(count: usize, include: &[usize], discords: bool, band_rows: usize) -> Self {
        Self {
            scale: (0..count).map(sum_scale).collect(),
            include: include.to_vec(),
            rest: (0..count).filter(|a| !include.contains(a)).collect(),
            discords,
            keys: Vec::with_capacity(count),
            thresholds: Vec::with_capacity(band_rows.saturating_mul(count)),
            rows: 0..0,
        }
    }

    /// Starts a tile of `rows`: the ceilings stay if they are of these rows, and are none
    /// otherwise, which it returns `true` for.
    pub(super) fn begin(&mut self, rows: &Range<usize>) -> bool {
        if self.rows == *rows {
            return false;
        }
        self.thresholds.clear();
        self.thresholds
            .resize(rows.len() * self.scale.len(), f64::INFINITY);
        self.rows = rows.clone();
        true
    }

    /// `row`'s thresholds.
    fn thresholds(&mut self, row: usize) -> &mut [f64] {
        let count = self.scale.len();
        &mut self.thresholds[(row - self.rows.start) * count..][..count]
    }

    /// Lowers `row`'s ceilings to its slots' `farther(k)`.
    #[inline(always)]
    pub(super) fn lower(&mut self, row: usize, farther: impl Fn(usize) -> f64) {
        let Self {
            scale,
            thresholds,
            rows,
            ..
        } = self;
        let count = scale.len();
        let thresholds = &mut thresholds[(row - rows.start) * count..][..count];
        for (k, (threshold, &scale)) in thresholds.iter_mut().zip(scale.iter()).enumerate() {
            *threshold = threshold.min(farther(k) * scale);
        }
    }

    /// Lowers `row`'s ceiling for `k` to `farther`, at least the exact mean of a window the row
    /// is offered.
    pub(super) fn lower_one(&mut self, row: usize, k: usize, farther: f64) {
        let scale = self.scale[k];
        let threshold = &mut self.thresholds(row)[k];
        *threshold = threshold.min(farther * scale);
    }

    /// Whether a cell of `row` whose bound on dimension `a`'s distance is `bounds[a]` (infinite
    /// for a missing one) is farther, in every `k`, than the row's ceiling.
    #[inline(always)]
    pub(super) fn settles(&mut self, row: usize, bounds: &[f64]) -> bool {
        let count = self.scale.len();
        let Self {
            include,
            rest,
            discords,
            keys,
            thresholds,
            rows,
            ..
        } = self;
        let thresholds = &thresholds[(row - rows.start) * count..][..count];
        match count {
            0..=4 => return settles_in::<4>(include, rest, *discords, bounds, thresholds, &FOUR),
            5..=8 => return settles_in::<8>(include, rest, *discords, bounds, thresholds, &EIGHT),
            _ => {}
        }
        let fixed = include.len();
        keys.clear();
        keys.extend(include.iter().map(|&a| bounds[a]));
        let discords = *discords;
        keys.extend(rest.iter().map(|&a| flip(bounds[a], discords)));
        keys[fixed..].sort_unstable_by(f64::total_cmp);
        keys[fixed..]
            .iter_mut()
            .for_each(|key| *key = flip(*key, discords));
        past_every_ceiling(keys, thresholds)
    }
}

/// With `discords`, the bound negated: farthest first is then nearest first, a missing distance
/// (infinite) still last.
#[inline(always)]
fn flip(bound: f64, discords: bool) -> f64 {
    if !discords || bound == f64::INFINITY {
        bound
    } else {
        -bound
    }
}

/// Whether every running sum of `keys` exceeds its threshold. Every mean is checked, with no early
/// exit: a cell settles nearly always.
#[inline(always)]
fn past_every_ceiling(keys: &[f64], thresholds: &[f64]) -> bool {
    let mut sum = 0.0;
    keys.iter()
        .zip(thresholds)
        .fold(true, |settled, (&bound, &threshold)| {
            sum += bound;
            settled & (sum > threshold)
        })
}

/// [`Screen::settles`] for up to `N` dimensions: the keys go into a fixed array padded with
/// infinities and sort through the network `pairs`, and each loop runs at most `N` times, so a
/// cell makes no call and copies no slice of a length known only at run time.
#[inline(always)]
fn settles_in<const N: usize>(
    include: &[usize],
    rest: &[usize],
    discords: bool,
    bounds: &[f64],
    thresholds: &[f64],
    pairs: &[(usize, usize)],
) -> bool {
    let mut sorted = [f64::INFINITY; N];
    for (i, key) in sorted.iter_mut().enumerate() {
        if let Some(&a) = rest.get(i) {
            *key = flip(bounds[a], discords);
        }
    }
    sort_network(&mut sorted, pairs);
    let fixed = include.len();
    let mut sum = 0.0;
    let mut settled = true;
    for k in 0..N {
        let Some(&threshold) = thresholds.get(k) else {
            break;
        };
        sum += match include.get(k) {
            Some(&a) => bounds[a],
            None => flip(sorted[k - fixed], discords),
        };
        settled &= sum > threshold;
    }
    settled
}

/// A sorting network for four keys: pairs to put in order, in turn.
const FOUR: [(usize, usize); 5] = [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)];

/// A sorting network for eight keys, of depth six.
const EIGHT: [(usize, usize); 19] = [
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (2, 4),
    (3, 5),
    (1, 4),
    (3, 6),
    (1, 2),
    (3, 4),
    (5, 6),
];

/// Sorts `keys` ascending through the network `pairs` of `min` and `max`, in place; the keys are a
/// cell's dimensions and never NaN.
#[inline(always)]
fn sort_network<const N: usize>(keys: &mut [f64; N], pairs: &[(usize, usize)]) {
    for &(low, high) in pairs {
        let (a, b) = (keys[low], keys[high]);
        keys[low] = a.min(b);
        keys[high] = a.max(b);
    }
}

#[cfg(test)]
mod tests;
