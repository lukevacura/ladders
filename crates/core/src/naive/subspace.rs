//! The subspace reference: every sample's bin counted edge by edge from exact 256-bit sums,
//! the dimensions ordered by a plain sort on a key. Independent of the engine's binary search,
//! its wide comparison and its shared dimension order.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use ethnum::I256;

use super::{Window, window_stats};
use crate::multidim::MultiError;
use crate::quantize::{Scale, to_ticks};
use crate::stats::Series;
use crate::subspace::{Bits, Subspace};

/// A positive finite `edge` as `(mantissa, shift)` with `edge = mantissa / 2^shift`.
fn dyadic(edge: f64) -> (u64, u32) {
    let (mut value, mut shift) = (edge, 0);
    while value.fract() != 0.0 {
        value *= 2.0;
        shift += 1;
    }
    (value as u64, shift)
}

/// `edge < a / sqrt(variance)`.
pub fn naive_edge_below(edge: f64, a: i128, variance: u64) -> bool {
    let (mantissa, shift) = dyadic(edge.abs());
    let scaled = I256::from(a) << shift;
    let squared = scaled.checked_mul(scaled).expect("fits 256 bits");
    let bound = I256::from(mantissa)
        .checked_mul(I256::from(mantissa))
        .and_then(|square| square.checked_mul(I256::from(variance)))
        .expect("fits 256 bits");
    if edge == 0.0 {
        scaled > 0
    } else if edge > 0.0 {
        scaled > 0 && squared > bound
    } else {
        scaled >= 0 || squared < bound
    }
}

/// The bins of the z-normalized window `samples[start..start + m]`, `None` without a distance.
/// A window the caller flagged constant is all middle bins, and so is an unflagged one of equal
/// ticks, whose samples are all at the mean.
pub fn naive_z_bins<'a>(
    samples: impl Into<Series<'a>>,
    scale: Scale,
    start: usize,
    m: usize,
    bits: Bits,
) -> Option<Vec<u16>> {
    let samples: Series<'a> = samples.into();
    let ticks: Vec<Option<i64>> = samples
        .values()
        .iter()
        .map(|&x| to_ticks(x, scale))
        .collect();
    let flagged = samples.constant_windows().map(|flags| flags[start]);
    match (window_stats(&ticks, start, m), flagged) {
        (Window::NonFinite, _) => None,
        (Window::Constant, _) | (_, Some(true)) => Some(alloc::vec![bits.middle_bin(); m]),
        (
            Window::Valid {
                sum,
                variance_numerator,
            },
            _,
        ) => {
            let variance = variance_numerator as u64;
            let deviation = libm::sqrt(variance as f64) / m as f64 * scale.get();
            Some(
                ticks[start..start + m]
                    .iter()
                    .map(|tick| {
                        let a =
                            m as i128 * i128::from(tick.expect("a valid window has ticks")) - sum;
                        let below = (0..bits.edge_count())
                            .filter(|&i| {
                                let edge = bits.edge(i);
                                if deviation < 1e-7 {
                                    edge < a as f64 / m as f64 * scale.get()
                                } else {
                                    naive_edge_below(edge, a, variance)
                                }
                            })
                            .count();
                        below as u16
                    })
                    .collect(),
            )
        }
    }
}

/// The bins of the window `samples[start..start + m]` between `low` and `high`.
pub fn naive_linear_bins(
    samples: &[f64],
    low: f64,
    high: f64,
    start: usize,
    m: usize,
    bits: Bits,
) -> Option<Vec<u16>> {
    let window = &samples[start..start + m];
    window.iter().all(|x| x.is_finite()).then(|| {
        let steps = f64::from((1u32 << bits.get()) - 1);
        window
            .iter()
            .map(|&x| {
                let ratio = (x - low) / (high - low);
                let scaled = if ratio.is_nan() { 0.0 } else { ratio * steps };
                libm::rint(scaled) as u16 + 1
            })
            .collect()
    })
}

pub(super) type Bins = Vec<Option<(Vec<u16>, Vec<u16>)>>;

/// The first `k + 1` dimensions by (has no distance, signed squared distance, dimension).
pub fn naive_choose(bins: &Bins, k: usize, include: &[usize], discords: bool) -> Subspace {
    let key = |dimension: usize| -> (bool, i128, usize) {
        match &bins[dimension] {
            None => (true, 0, dimension),
            Some((first, second)) => {
                let squared: i128 = first
                    .iter()
                    .zip(second)
                    .map(|(&a, &b)| i128::from(a) - i128::from(b))
                    .map(|difference| difference * difference)
                    .sum();
                (false, if discords { -squared } else { squared }, dimension)
            }
        }
    };
    let mut ordered: Vec<usize> = (0..bins.len()).collect();
    ordered.sort_by_key(|&dimension| key(dimension));
    let mut front: Vec<usize> = Vec::new();
    let mut back: Vec<usize> = Vec::new();
    for dimension in ordered {
        if include.contains(&dimension) {
            front.push(dimension);
        } else {
            back.push(dimension);
        }
    }
    front.extend(back);
    front.truncate(k + 1);
    let complete = front.iter().all(|&dimension| bins[dimension].is_some());
    Subspace {
        dimensions: front,
        complete,
    }
}

pub(super) fn checked(
    lengths: &[usize],
    m: NonZeroUsize,
    pair: (usize, usize),
    k: usize,
    include: &[usize],
) -> Result<(), MultiError> {
    let Some(&n) = lengths.first() else {
        return Err(MultiError::NoDimensions);
    };
    if lengths.iter().any(|&other| other != n) {
        return Err(MultiError::UnequalLengths);
    }
    if m.get() > n {
        return Err(MultiError::WindowTooLong);
    }
    if include.iter().any(|&dimension| dimension >= lengths.len()) {
        return Err(MultiError::IncludeOutOfRange);
    }
    if k >= lengths.len() {
        return Err(MultiError::KOutOfRange);
    }
    if pair.0 > n - m.get() || pair.1 > n - m.get() {
        return Err(MultiError::WindowOutOfRange);
    }
    Ok(())
}

/// [`crate::subspace::subspace`], computed directly.
pub fn naive_subspace<'a, S: Into<Series<'a>> + Copy>(
    set: &[(S, Scale)],
    m: NonZeroUsize,
    pair: (usize, usize),
    k: usize,
    include: &[usize],
    discords: bool,
    bits: Bits,
) -> Result<Subspace, MultiError> {
    let set: Vec<(Series<'a>, Scale)> = set.iter().map(|&(s, scale)| (s.into(), scale)).collect();
    let lengths: Vec<usize> = set.iter().map(|(series, _)| series.len()).collect();
    checked(&lengths, m, pair, k, include)?;
    let bins: Bins = set
        .iter()
        .map(|&(series, scale)| {
            Some((
                naive_z_bins(series, scale, pair.0, m.get(), bits)?,
                naive_z_bins(series, scale, pair.1, m.get(), bits)?,
            ))
        })
        .collect();
    Ok(naive_choose(&bins, k, include, discords))
}

/// [`crate::subspace::maamp_subspace`], computed directly.
pub fn naive_maamp_subspace(
    set: &[&[f64]],
    m: NonZeroUsize,
    pair: (usize, usize),
    k: usize,
    include: &[usize],
    discords: bool,
    bits: Bits,
) -> Result<Subspace, MultiError> {
    let lengths: Vec<usize> = set.iter().map(|series| series.len()).collect();
    checked(&lengths, m, pair, k, include)?;
    let (low, high) = naive_linear_range(set);
    let bins: Bins = set
        .iter()
        .map(|series| {
            Some((
                naive_linear_bins(series, low, high, pair.0, m.get(), bits)?,
                naive_linear_bins(series, low, high, pair.1, m.get(), bits)?,
            ))
        })
        .collect();
    Ok(naive_choose(&bins, k, include, discords))
}

/// The smallest and largest finite sample of every dimension; `(0, 0)` without any.
pub fn naive_linear_range(set: &[&[f64]]) -> (f64, f64) {
    let finite: Vec<f64> = set
        .iter()
        .flat_map(|series| series.iter().copied())
        .filter(|x| x.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 0.0);
    }
    (
        finite.iter().copied().fold(f64::INFINITY, f64::min),
        finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}
