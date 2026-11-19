//! The multi-dimensional profile's vocabulary: the k-dimensional value, the policy for windows
//! without a distance, the input errors, and the profile a reference or engine returns.
//!
//! What an engine must reproduce from the reference to equal it on bits:
//! - Each column's dimensions are sorted by the exact comparator, stable, so ties go to the
//!   smaller dimension, in discords mode too; `include` dimensions come first in the order given.
//!   A dimension without a distance sorts last in both modes (STUMPY treats the gap as an infinite
//!   distance and so picks it first as a discord; here it is missing instead).
//! - A mean of `k >= 2` dimensions is summed in that order, left to right, with no fused or
//!   reassociated arithmetic, then divided by `k`; that is the value reported.
//! - Two means of the same `k` are ordered by `MeanOrder`, exactly: a tie between them goes to
//!   the smaller window start.
//! - Every cell is computed from the row's perspective, row `i` and candidate `j`, for both
//!   directions of every pair: a distance is not symmetric in its last bit.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::marker::PhantomData;

use crate::index::SampleIndex;
use crate::nearest::{ExactCorrelation, ExactDistance, Held};
use crate::spill::Spilled;

mod big;
mod count;

pub(crate) use count::{cells_screened, exact_decision};
mod roots;

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
#[doc(hidden)]
pub use count::{budget_fallbacks, exact_decisions, screened_cells};

pub use roots::Terms;

/// What one dimension's distance offers the multi-dimensional layer: the exact order across
/// windows, the distance itself, which the mean of `k >= 2` dimensions sums, and the distance as
/// an exact algebraic number, which decides the order between two such means.
pub trait DimensionDistance: Copy + Ord {
    /// The distance, a square root in the metric's own unit.
    fn root(self) -> f64;

    /// Bounds `|root(self) − distance| / distance` for the exact distance.
    const ROOT_RELATIVE_ERROR: f64;

    /// Adds this distance, negated if `negative`, to the exact sum `terms` holds.
    #[doc(hidden)]
    fn push_exact(self, negative: bool, terms: &mut Terms);

    /// `Some(false)` if the distance is exactly zero whatever the pair, `Some(true)` if it is
    /// exactly the square root of the window length whatever the pair (the same for every
    /// distance of one profile), `None` if it depends on the pair.
    #[doc(hidden)]
    fn fixed(self) -> Option<bool>;
}

impl DimensionDistance for ExactDistance {
    fn root(self) -> f64 {
        libm::sqrt(self.entry().squared_distance())
    }

    /// With `u = 2^-53`: the squared distance is computed from `ρ̂`, within `16u` of the exact
    /// correlation, through at most 19 roundings of `u` each, so it is within `35u` of the exact
    /// one relative to it; the root halves that and adds its own rounding, under `19u`. The bound
    /// is `64u = 2^-47`.
    const ROOT_RELATIVE_ERROR: f64 = 1.0 / (1u64 << 47) as f64;

    fn push_exact(self, negative: bool, terms: &mut Terms) {
        match self.correlation() {
            Held::Hot(correlation) => terms.push_znorm(negative, self.window(), correlation),
            Held::Cold(correlation) => terms.push_znorm_wide(negative, self.window(), correlation),
        }
    }

    fn fixed(self) -> Option<bool> {
        match self.correlation() {
            Held::Hot(ExactCorrelation::BothConstant) | Held::Cold(Spilled::BothConstant) => {
                Some(false)
            }
            Held::Hot(ExactCorrelation::OneConstant) | Held::Cold(Spilled::OneConstant) => {
                Some(true)
            }
            Held::Hot(ExactCorrelation::Valid { .. }) | Held::Cold(Spilled::Valid { .. }) => None,
        }
    }
}

/// The distance of a window pair over its `k` chosen dimensions: the mean of those dimensions'
/// distances (square roots, not squares), summed left to right in the order given. It is built
/// only from per-dimension distances; nothing converts in from a bare `f64`. Two means are
/// ordered only by `MeanOrder`, which decides on the `f64` value outside the error bound of
/// the arithmetic and on the exact distances inside it, so equality here is that of the reported
/// value.
#[derive(Debug)]
pub struct MeanDistance<D = ExactDistance> {
    value: f64,
    count: usize,
    dimension: PhantomData<fn() -> D>,
}

impl<D> MeanDistance<D> {
    /// No distance: a pair with fewer than `k` dimensions that have one. Orders after every
    /// real distance.
    pub const MISSING: Self = Self::new(f64::INFINITY, 0);

    pub const fn to_f64(self) -> f64 {
        self.value
    }

    pub const fn is_missing(self) -> bool {
        self.value.is_infinite()
    }

    pub(crate) const fn new(value: f64, count: usize) -> Self {
        Self {
            value,
            count,
            dimension: PhantomData,
        }
    }
}

impl<D: DimensionDistance> MeanDistance<D> {
    /// The mean of `dimensions`' distances in the order given; [`Self::MISSING`] for none, or if
    /// any is `None` (that dimension has no distance).
    pub fn of(dimensions: &[Option<D>]) -> Self {
        let mut sum = 0.0;
        for dimension in dimensions {
            let Some(distance) = dimension else {
                return Self::MISSING;
            };
            sum += distance.root();
        }
        match dimensions {
            [] => Self::MISSING,
            [_] => Self::new(sum, 1),
            _ => Self::new(sum / dimensions.len() as f64, dimensions.len()),
        }
    }

    /// Calls `visit(k, mean)` for every `k`, `mean` being [`Self::of`]`(&dimensions[..=k])`, in
    /// one left-to-right pass.
    pub fn each_prefix(dimensions: &[Option<D>], mut visit: impl FnMut(usize, Self)) {
        let mut sum = 0.0;
        for (k, dimension) in dimensions.iter().enumerate() {
            let Some(distance) = dimension else {
                for missing in k..dimensions.len() {
                    visit(missing, Self::MISSING);
                }
                return;
            };
            sum += distance.root();
            let mean = if k == 0 {
                Self::new(sum, 1)
            } else {
                Self::new(sum / (k + 1) as f64, k + 1)
            };
            visit(k, mean);
        }
    }
}

impl<D> Clone for MeanDistance<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for MeanDistance<D> {}

impl<D> PartialEq for MeanDistance<D> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.count == other.count
    }
}

impl<D> Eq for MeanDistance<D> {}

/// The order between means of the same number of dimensions.
///
/// **Bound.** A mean is `fl(fl(((t₁ + t₂) + …) + t_k) / k)` with `t_i` the rounded roots, each
/// within `τ = ROOT_RELATIVE_ERROR` of its exact distance `d_i`. The `k − 1` additions and the
/// division round `k` times in all, each by at most `u = 2^-53`, so the value is within
/// `e = (1 + τ)(1 + γ) − 1 ≤ 1.0001·(τ + γ)` of the exact mean, relative to it, `γ = k·u /
/// (1 − k·u)` (the code takes `k + 1` for `k`, and `τ + γ` times 1.25 for the error). Two values
/// `a` and `b` then order as their exact means when one exceeds the other times
/// `(1 + e)/(1 − e) ≤ 1 + 2.1·e`; the order uses `1 + 3·(1.25·(τ + γ))`, which is larger.
/// Anything closer goes to the exact decision.
///
/// **Exact decision.** The two means have the same count, so they order as the sums of their
/// distances. Equal distances on both sides cancel (the exact order of two distances is `Ord`);
/// what is left is decided by the sign of a signed sum of exact roots ([`Terms`]).
#[derive(Clone, Debug)]
pub(crate) struct MeanOrder {
    /// `(1 + e_k)/(1 − e_k)` rounded up, for `k = 1..=d`.
    ratios: Vec<f64>,
}

/// `(1 + e)/(1 − e)` rounded up, `e` bounding the relative error of a mean of `count`
/// distances whose roots are each within `root_error`: `(τ + γ) · 1.25`, with `γ` the rounding of
/// the `count` operations that make the mean (`count + 1`, to be safe); infinite, so that no
/// order is decided by value, once that rounding is no longer small.
fn error_ratio(count: usize, root_error: f64) -> f64 {
    let unit = 1.0 / (1u64 << 53) as f64;
    let operations = (count + 1) as f64 * unit;
    if operations >= 0.01 {
        return f64::INFINITY;
    }
    let error = (root_error + operations / (1.0 - operations)) * 1.25;
    1.0 + 3.0 * error
}

/// What an exact decision works in, reused from one to the next.
#[derive(Debug)]
pub(crate) struct MeanWork<D> {
    plus: Vec<D>,
    minus: Vec<D>,
    terms: Terms,
}

impl<D> MeanWork<D> {
    pub(crate) fn new() -> Self {
        Self {
            plus: Vec::new(),
            minus: Vec::new(),
            terms: Terms::new(),
        }
    }
}

impl MeanOrder {
    /// The order for means of up to `max_count` dimensions of distances with relative error
    /// `root_error`.
    pub(crate) fn new(max_count: usize, root_error: f64) -> Self {
        Self {
            ratios: (1..=max_count)
                .map(|count| error_ratio(count, root_error))
                .collect(),
        }
    }

    /// `(1 + e)/(1 − e)` rounded up for means of `k + 1` dimensions: a value that exceeds
    /// another's times this is decidedly farther.
    #[inline(always)]
    pub(crate) fn ratio(&self, k: usize) -> f64 {
        self.ratios[k]
    }

    /// The order of `a` against `b`, means of `k + 1` dimensions, if their values decide it:
    /// `None` when they are within the error bound of each other. A missing `a` is `Greater`
    /// than any value, and a missing `b` is `Less`; two missing means decide nothing.
    #[inline(always)]
    pub(crate) fn by_value<D>(
        &self,
        k: usize,
        a: &MeanDistance<D>,
        b: &MeanDistance<D>,
    ) -> Option<Ordering> {
        let ratio = self.ratios[k];
        if a.value > b.value * ratio {
            Some(Ordering::Greater)
        } else if a.value * ratio < b.value {
            Some(Ordering::Less)
        } else {
            None
        }
    }

    /// The order of `a` against `b`, two means of the same number of dimensions whose distances
    /// are `a_terms` and `b_terms`; a missing mean is the last.
    pub(crate) fn compare<D: DimensionDistance>(
        &self,
        work: &mut MeanWork<D>,
        a: &MeanDistance<D>,
        a_terms: &[Option<D>],
        b: &MeanDistance<D>,
        b_terms: &[Option<D>],
    ) -> Ordering {
        if a.is_missing() || b.is_missing() {
            return a.is_missing().cmp(&b.is_missing());
        }
        debug_assert_eq!(a.count, b.count);
        self.by_value(a.count - 1, a, b)
            .unwrap_or_else(|| exact_order(work, a_terms, b_terms))
    }
}

fn distances<D: Copy>(side: &[Option<D>]) -> impl Iterator<Item = D> + '_ {
    side.iter()
        .map(|distance| distance.expect("a mean that is not missing has every distance"))
}

/// The order of the sums of `a`'s and `b`'s distances, exactly; every entry is `Some`.
#[cold]
#[inline(never)]
pub(crate) fn exact_order<D: DimensionDistance>(
    work: &mut MeanWork<D>,
    a: &[Option<D>],
    b: &[Option<D>],
) -> Ordering {
    count::exact_decision();
    if let ([Some(only_a)], [Some(only_b)]) = (a, b) {
        return only_a.cmp(only_b);
    }
    if distances(a).zip(distances(b)).all(|(x, y)| x == y) {
        return Ordering::Equal;
    }
    let MeanWork { plus, minus, terms } = work;
    plus.clear();
    plus.extend(distances(a));
    minus.clear();
    minus.extend(distances(b));
    let mut at = 0;
    while at < plus.len() {
        match minus.iter().position(|other| *other == plus[at]) {
            Some(found) => {
                minus.swap_remove(found);
                plus.swap_remove(at);
            }
            None => at += 1,
        }
    }
    terms.clear();
    for &distance in plus.iter() {
        distance.push_exact(false, terms);
    }
    for &distance in minus.iter() {
        distance.push_exact(true, terms);
    }
    terms.sign()
}

/// What a window with a non-finite sample does to a dimension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NonFinitePolicy {
    /// The dimension has no distance for any pair involving the window, and sorts last.
    #[default]
    Missing,
}

/// Why a multi-dimensional profile, subspace or description length was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultiError {
    NoDimensions,
    UnequalLengths,
    /// An `include` entry is not a dimension of the input.
    IncludeOutOfRange,
    /// The window is longer than the series.
    WindowTooLong,
    /// The number of scales is not the number of dimensions.
    ScaleCount,
    /// The number of bits per sample is not in `1..=8`.
    BitsOutOfRange,
    /// A window start is not the start of a window of the series.
    WindowOutOfRange,
    /// `k` is not below the number of dimensions.
    KOutOfRange,
    /// The window starts do not come one per dimension.
    WrongPairCount,
    /// A profile's distances or indices are not one row of one entry per window for each
    /// dimension.
    ProfileShape,
    /// The cutoffs are not one per dimension.
    CutoffCount,
}

/// How the `include` dimensions are ordered among themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IncludeOrder {
    /// In the order the caller gave them.
    #[allow(
        dead_code,
        reason = "the reference and the engines use it; the subspace does not"
    )]
    AsGiven,
    /// By distance, like every other dimension.
    ByDistance,
}

/// `include` without repeats, in the order given; an entry that is not below `count` is an error.
pub(crate) fn checked_include(include: &[usize], count: usize) -> Result<Vec<usize>, MultiError> {
    let mut kept: Vec<usize> = Vec::new();
    for &dimension in include {
        if dimension >= count {
            return Err(MultiError::IncludeOutOfRange);
        }
        if !kept.contains(&dimension) {
            kept.push(dimension);
        }
    }
    Ok(kept)
}

fn cmp_keys<K: Ord>(a: &Option<K>, b: &Option<K>, farthest_first: bool) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) if farthest_first => b.cmp(a),
        (Some(a), Some(b)) => a.cmp(b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// The dimensions of one column in the order their distances are used: nearest first (farthest
/// first if `farthest_first`), a dimension without a distance last either way, ties to the
/// smaller dimension; the `include` dimensions (no repeats, in range) in front, ordered as
/// `include_order` says. [`order_dimensions_into`] without the allocation.
pub(crate) fn order_dimensions<K: Ord>(
    keys: &[Option<K>],
    include: &[usize],
    farthest_first: bool,
    include_order: IncludeOrder,
) -> Vec<usize> {
    let mut order = Vec::with_capacity(keys.len());
    order_dimensions_into(&mut order, keys, include, farthest_first, include_order);
    order
}

/// Fills `out` with the order [`order_dimensions`] returns. The sort is in place and breaks every
/// tie by dimension, so it is deterministic and allocates nothing once `out` holds
/// `keys.len()` entries.
pub(crate) fn order_dimensions_into<K: Ord>(
    out: &mut Vec<usize>,
    keys: &[Option<K>],
    include: &[usize],
    farthest_first: bool,
    include_order: IncludeOrder,
) {
    let by_key =
        |a: &usize, b: &usize| cmp_keys(&keys[*a], &keys[*b], farthest_first).then(a.cmp(b));
    out.clear();
    match include_order {
        IncludeOrder::AsGiven => {
            out.extend_from_slice(include);
            let start = out.len();
            out.extend((0..keys.len()).filter(|dimension| !include.contains(dimension)));
            out[start..].sort_unstable_by(by_key);
        }
        IncludeOrder::ByDistance => {
            out.extend(0..keys.len());
            out.sort_unstable_by(|a, b| {
                include
                    .contains(b)
                    .cmp(&include.contains(a))
                    .then_with(|| by_key(a, b))
            });
        }
    }
}

/// Per window start, for each `k` in `0..d`, the nearest eligible window over the mean of its
/// `k + 1` chosen dimensions: `distance[k][i]` and `index[k][i]`. A window without one has
/// [`MeanDistance::MISSING`] and [`SampleIndex::NONE`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiProfile<D: DimensionDistance = ExactDistance> {
    pub distance: Vec<Vec<MeanDistance<D>>>,
    pub index: Vec<Vec<SampleIndex>>,
}

#[cfg(test)]
mod tests;
