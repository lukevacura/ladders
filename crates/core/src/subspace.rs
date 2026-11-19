//! The subspace of a motif pair, following STUMPY 1.14.1's `subspace` and `maamp_subspace`:
//! which `k + 1` dimensions a pair of windows is closest in.
//!
//! Every sample of the two windows is put into one of `2^bits` bins. A z-normalized sample's bin is
//! the number of edges strictly below its z-value, the edges being the standard normal quantiles of
//! `i / 2^bits`, each correctly rounded, decided exactly on the quantized ticks (never on a rounded
//! z-value); a constant window is all zeros, the middle bin. A non-normalized sample's bin is
//! STUMPY's `round((x - low) / (high - low) * (2^bits - 1)) + 1` on the original samples, `low` and
//! `high` the extremes of every finite sample of every dimension. A dimension's distance is the
//! root of the sum of squared bin differences, an integer compared exactly; dimensions are ordered
//! by it, ties to the smaller dimension, the `include` dimensions first, ordered by their own
//! distances.
//!
//! Differences from STUMPY: its edges, scipy's `norm.ppf(i / 2^bits)`, are up to 2 ulps from the
//! correctly rounded quantiles and vary by platform, so a z-value between the two moves one bin;
//! `discords` puts the farthest dimensions first (STUMPY returns the wrong dimensions); a dimension
//! whose window has a non-finite sample has no distance and sorts last, instead of being
//! zero-filled; a z-normalized window whose standard deviation, in the samples' own units, is below
//! `1e-7` is not scaled, as in STUMPY, with that standard deviation computed in `f64`. When every
//! sample of every dimension is equal, every non-normalized sample is in bin 1 (STUMPY divides zero
//! by zero).

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::multidim::{
    IncludeOrder, MultiError, NonFinitePolicy, checked_include, order_dimensions,
};
use crate::nearest::cmp_wide_products;
use crate::pattern::TickedSeries;
use crate::quantize::{Scale, round_ties_even};
use crate::stats::{Series, WindowKind};

mod edges;
#[cfg(test)]
mod tests;

use edges::EDGES;

/// STUMPY's standard deviation below which a window is not divided by it.
const SMALL_DEVIATION: f64 = 1e-7;

/// The widest shift an edge's square needs (`edge² = mantissa² / 2^(2·shift)`), so that the
/// shifted square of a sample's offset stays within the exact comparison's range.
const MAX_EDGE_SHIFT: u32 = 62;

const _: () = {
    let mut i = 0;
    while i < EDGES.len() {
        let edge = EDGES[i];
        assert!(edge == 0.0 || edge_shift(edge) <= MAX_EDGE_SHIFT);
        i += 1;
    }
};

/// How many bits a sample is discretized to: `1..=8`, the edges for fewer bits being every
/// `2^(8 - bits)`-th edge of the eight-bit table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bits(u8);

impl Bits {
    pub const EIGHT: Self = Self(8);

    pub const fn new(bits: u8) -> Result<Self, MultiError> {
        if matches!(bits, 1..=8) {
            Ok(Self(bits))
        } else {
            Err(MultiError::BitsOutOfRange)
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    /// The number of edges, one less than the number of bins.
    pub(crate) const fn edge_count(self) -> usize {
        (1 << self.0) - 1
    }

    /// The `i`-th edge, smallest first.
    pub(crate) fn edge(self, i: usize) -> f64 {
        EDGES[((i + 1) << (8 - self.0)) - 1]
    }

    /// The bin of a sample at the mean.
    pub(crate) const fn middle_bin(self) -> u16 {
        (1 << (self.0 - 1)) - 1
    }
}

/// What the choice of a subspace is asked: which dimensions come first, in which direction, how
/// finely samples are binned.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a> {
    /// Dimensions that must be in the subspace, before all others (repeats are dropped).
    pub include: &'a [usize],
    /// The farthest dimensions first, instead of the nearest.
    pub discords: bool,
    pub bits: Bits,
    pub non_finite: NonFinitePolicy,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            include: &[],
            discords: false,
            bits: Bits::EIGHT,
            non_finite: NonFinitePolicy::Missing,
        }
    }
}

/// The first `k + 1` dimensions of the order of a pair of windows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subspace {
    pub dimensions: Vec<usize>,
    /// Every chosen dimension has a distance; false when fewer than `k + 1` do, the others
    /// following last.
    pub complete: bool,
}

/// The bins of a pair of windows in one dimension: the first window's and the second's.
pub(crate) type PairBins = (Vec<u16>, Vec<u16>);

/// A window's bins in each dimension, `None` where the dimension has no distance.
pub(crate) type Pair = Vec<Option<PairBins>>;

struct ZNormalized {
    ticked: TickedSeries,
    scale: Scale,
}

enum Kind<'a> {
    ZNormalized(Vec<ZNormalized>),
    Linear {
        series: &'a [&'a [f64]],
        low: f64,
        high: f64,
    },
}

/// A set of equally long series ready to bin windows of length `m`.
pub(crate) struct Discretizer<'a> {
    kind: Kind<'a>,
    m: usize,
    windows: usize,
    bits: Bits,
}

pub(crate) fn window_count(
    mut lengths: impl Iterator<Item = usize>,
    m: NonZeroUsize,
) -> Result<usize, MultiError> {
    let first = lengths.next().ok_or(MultiError::NoDimensions)?;
    if lengths.any(|other| other != first) {
        return Err(MultiError::UnequalLengths);
    }
    if m.get() > first {
        return Err(MultiError::WindowTooLong);
    }
    Ok(first - m.get() + 1)
}

impl<'a> Discretizer<'a> {
    pub(crate) fn z_normalized<'s, S: Into<Series<'s>> + Copy>(
        set: &[(S, Scale)],
        m: NonZeroUsize,
        bits: Bits,
    ) -> Result<Self, MultiError> {
        let set: Vec<(Series<'s>, Scale)> = set
            .iter()
            .map(|&(series, scale)| (series.into(), scale))
            .collect();
        let windows = window_count(set.iter().map(|(series, _)| series.len()), m)?;
        let dimensions = set
            .iter()
            .map(|&(series, scale)| ZNormalized {
                ticked: TickedSeries::new(series, m, scale),
                scale,
            })
            .collect();
        Ok(Self {
            kind: Kind::ZNormalized(dimensions),
            m: m.get(),
            windows,
            bits,
        })
    }

    pub(crate) fn linear(
        series: &'a [&'a [f64]],
        m: NonZeroUsize,
        bits: Bits,
    ) -> Result<Self, MultiError> {
        let windows = window_count(series.iter().map(|s| s.len()), m)?;
        let mut finite = series
            .iter()
            .flat_map(|s| s.iter())
            .filter(|x| x.is_finite());
        let (low, high) = match finite.next() {
            Some(&first) => {
                finite.fold((first, first), |(low, high), &x| (low.min(x), high.max(x)))
            }
            None => (0.0, 0.0),
        };
        Ok(Self {
            kind: Kind::Linear { series, low, high },
            m: m.get(),
            windows,
            bits,
        })
    }

    pub(crate) fn dimensions(&self) -> usize {
        match &self.kind {
            Kind::ZNormalized(dimensions) => dimensions.len(),
            Kind::Linear { series, .. } => series.len(),
        }
    }

    pub(crate) fn window_length(&self) -> usize {
        self.m
    }

    pub(crate) fn check_start(&self, start: usize) -> Result<(), MultiError> {
        if start < self.windows {
            Ok(())
        } else {
            Err(MultiError::WindowOutOfRange)
        }
    }

    /// The bins of the window at `start` in one dimension; `None` when it has a non-finite
    /// sample (or, z-normalized, a range the arithmetic cannot hold).
    fn window_bins(&self, dimension: usize, start: usize) -> Option<Vec<u16>> {
        match &self.kind {
            Kind::ZNormalized(dimensions) => {
                z_normalized_bins(&dimensions[dimension], start, self.m, self.bits)
            }
            Kind::Linear { series, low, high } => {
                let window = &series[dimension][start..start + self.m];
                window.iter().all(|x| x.is_finite()).then(|| {
                    window
                        .iter()
                        .map(|&x| linear_bin(x, *low, *high, self.bits))
                        .collect()
                })
            }
        }
    }

    /// The pair of windows at `first` and `second` in every dimension.
    pub(crate) fn pair(&self, first: usize, second: usize) -> Pair {
        (0..self.dimensions())
            .map(|dimension| {
                Some((
                    self.window_bins(dimension, first)?,
                    self.window_bins(dimension, second)?,
                ))
            })
            .collect()
    }
}

/// STUMPY's linear bin: `round((x - low) / (high - low) * (2^bits - 1)) + 1`, rounding ties to
/// even. A ratio that is not a number (the extremes are equal, or their difference overflows)
/// is zero.
fn linear_bin(x: f64, low: f64, high: f64, bits: Bits) -> u16 {
    let steps = f64::from((1u32 << bits.get()) - 1);
    let scaled = (x - low) / (high - low) * steps;
    round_ties_even(scaled) as u16 + 1
}

/// `edge` is `mantissa / 2^shift` for an integer `mantissa` below `2^53`: the shift.
const fn edge_shift(edge: f64) -> u32 {
    1075 - ((edge.to_bits() >> 52) & 0x7ff) as u32
}

/// `edge < a / sqrt(variance)`, exactly, where `a = m·x − S` is a sample's offset from the
/// window's sum and `variance` the window's variance numerator.
fn edge_below(edge: f64, a: i128, variance: u64) -> bool {
    if edge == 0.0 {
        return a > 0;
    }
    let shift = edge_shift(edge);
    let mantissa = u128::from((edge.to_bits() & ((1 << 52) - 1)) | (1 << 52));
    let offset = a.unsigned_abs();
    let order = cmp_wide_products(
        offset * offset,
        1 << (2 * shift),
        mantissa * mantissa,
        u128::from(variance),
    );
    if edge > 0.0 {
        a > 0 && order.is_gt()
    } else {
        a >= 0 || order.is_lt()
    }
}

/// The number of edges for which `below` holds, `below` holding for a prefix of them.
fn count_below(bits: Bits, below: impl Fn(f64) -> bool) -> u16 {
    let (mut low, mut high) = (0, bits.edge_count());
    while low < high {
        let middle = (low + high) / 2;
        if below(bits.edge(middle)) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low as u16
}

fn z_normalized_bins(
    dimension: &ZNormalized,
    start: usize,
    m: usize,
    bits: Bits,
) -> Option<Vec<u16>> {
    let kind = dimension.ticked.windows()[start].kind;
    let ticks = &dimension.ticked.ticks()[start..start + m];
    match kind {
        WindowKind::NonFinite => None,
        WindowKind::Constant => Some(alloc::vec![bits.middle_bin(); m]),
        WindowKind::Valid {
            variance_numerator, ..
        } => {
            let sum: i128 = ticks.iter().map(|&tick| i128::from(tick)).sum();
            let scale = dimension.scale.get();
            let small = libm::sqrt(variance_numerator as f64) / m as f64 * scale < SMALL_DEVIATION;
            Some(
                ticks
                    .iter()
                    .map(|&tick| {
                        let a = m as i128 * i128::from(tick) - sum;
                        if small {
                            let z = a as f64 / m as f64 * scale;
                            count_below(bits, |edge| edge < z)
                        } else {
                            count_below(bits, |edge| edge_below(edge, a, variance_numerator))
                        }
                    })
                    .collect(),
            )
        }
    }
}

/// The squared distance of one dimension: the sum of squared bin differences.
pub(crate) fn squared_distance((first, second): &PairBins) -> u128 {
    first
        .iter()
        .zip(second)
        .map(|(&a, &b)| u128::from(a.abs_diff(b)).pow(2))
        .sum()
}

/// The dimensions of `pair` in order, the first `k + 1` of them.
pub(crate) fn choose(pair: &Pair, k: usize, include: &[usize], discords: bool) -> Subspace {
    let keys: Vec<Option<u128>> = pair
        .iter()
        .map(|bins| bins.as_ref().map(squared_distance))
        .collect();
    let mut dimensions = order_dimensions(&keys, include, discords, IncludeOrder::ByDistance);
    dimensions.truncate(k + 1);
    let complete = dimensions
        .iter()
        .all(|&dimension| keys[dimension].is_some());
    Subspace {
        dimensions,
        complete,
    }
}

pub(crate) fn check_choice(
    discretizer: &Discretizer<'_>,
    k: usize,
    options: &Options<'_>,
) -> Result<Vec<usize>, MultiError> {
    let include = checked_include(options.include, discretizer.dimensions())?;
    if k >= discretizer.dimensions() {
        return Err(MultiError::KOutOfRange);
    }
    Ok(include)
}

fn subspace_of(
    discretizer: &Discretizer<'_>,
    (first, second): (usize, usize),
    k: usize,
    options: &Options<'_>,
) -> Result<Subspace, MultiError> {
    let include = check_choice(discretizer, k, options)?;
    discretizer.check_start(first)?;
    discretizer.check_start(second)?;
    let NonFinitePolicy::Missing = options.non_finite;
    Ok(choose(
        &discretizer.pair(first, second),
        k,
        &include,
        options.discords,
    ))
}

/// The `k + 1` dimensions in which the windows of length `m` at `pair.0` and `pair.1` of the
/// z-normalized series are closest (farthest, for `discords`); each series is quantized at its
/// own scale. `Err` for no series, series of different lengths, a window longer than them, an
/// `include` entry or `k` that is not a dimension, or a window start that is not a window.
pub fn subspace<'s, S: Into<Series<'s>> + Copy>(
    set: &[(S, Scale)],
    m: NonZeroUsize,
    pair: (usize, usize),
    k: usize,
    options: &Options<'_>,
) -> Result<Subspace, MultiError> {
    let discretizer = Discretizer::z_normalized(set, m, options.bits)?;
    subspace_of(&discretizer, pair, k, options)
}

/// [`subspace`] for series that are not z-normalized: STUMPY's `maamp_subspace` with `p = 2`.
pub fn maamp_subspace(
    set: &[&[f64]],
    m: NonZeroUsize,
    pair: (usize, usize),
    k: usize,
    options: &Options<'_>,
) -> Result<Subspace, MultiError> {
    let discretizer = Discretizer::linear(set, m, options.bits)?;
    subspace_of(&discretizer, pair, k, options)
}
