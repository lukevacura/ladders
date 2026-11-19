//! The multi-dimensional reference: for every window and every `k`, the eligible window whose
//! mean over the `k + 1` chosen dimensions is smallest, each dimension's distance exact and
//! each column's dimensions ordered by the exact comparator. O(d·n²·m), no recurrence. The
//! z-normalized and the non-normalized metric differ only in a dimension's distance.

use alloc::vec::Vec;
use core::cmp::Ordering;

use super::{Ticks, Window, candidate, comparable, cross_term, squared_distance_pair};
use crate::index::SampleIndex;
use crate::multidim::{
    DimensionDistance, IncludeOrder, MeanDistance, MeanOrder, MeanWork, MultiError, MultiProfile,
    NonFinitePolicy, checked_include, order_dimensions,
};
use crate::nearest::{ExactDistance, winner_entry};
use crate::nonnorm::SquaredTicks;

/// The distance between window `i` and window `j` of one dimension; `None` if either holds a
/// non-finite sample.
fn dimension_distance(
    samples: &[Option<i64>],
    windows: &[Window],
    m: usize,
    i: usize,
    j: usize,
) -> Option<ExactDistance> {
    let row = comparable(windows[i])?;
    let neighbor = comparable(windows[j])?;
    let offered = candidate(m, row, neighbor, cross_term(samples, i, j, m), j as u64);
    Some(ExactDistance::from_parts(
        winner_entry(m, offered),
        offered.correlation,
        m,
    ))
}

/// Whether the squared distances of a pair whose later window starts at `later` fit `u128`:
/// `m·R² < 2¹²⁸` over the range `R` of every present tick through that window.
fn representable(samples: &[Option<i64>], m: usize, later: usize) -> bool {
    let present = || samples[..later + m].iter().flatten().copied();
    let (Some(lowest), Some(highest)) = (present().min(), present().max()) else {
        return true;
    };
    let range = (i128::from(highest) - i128::from(lowest)) as u128;
    range
        .checked_mul(range)
        .and_then(|squared| squared.checked_mul(m as u128))
        .is_some()
}

/// The exact squared distance in ticks between window `i` and window `j` of one dimension;
/// `None` if either has a gap, or the pair is not `representable`.
fn squared_ticks(
    samples: &[Option<i64>],
    complete: &[bool],
    m: usize,
    i: usize,
    j: usize,
) -> Option<SquaredTicks> {
    if !complete[i] || !complete[j] || !representable(samples, m, i.max(j)) {
        return None;
    }
    let squared = squared_distance_pair(samples, i, samples, j, m);
    Some(SquaredTicks(
        u128::try_from(squared).expect("a representable pair's distance fits u128"),
    ))
}

/// The common length of `dimensions` and their `include` list without repeats.
fn checked(
    dimensions: &[&[Option<i64>]],
    include: &[usize],
) -> Result<(usize, Vec<usize>), MultiError> {
    let Some(first) = dimensions.first() else {
        return Err(MultiError::NoDimensions);
    };
    let n = first.len();
    if dimensions.iter().any(|samples| samples.len() != n) {
        return Err(MultiError::UnequalLengths);
    }
    Ok((n, checked_include(include, dimensions.len())?))
}

/// The profile over `windows` windows of `count` dimensions whose pair distance is
/// `distance(a, i, j)`, row `i` and candidate `j`, for every `j` with `|i − j| > exclusion`.
fn reduce<D: DimensionDistance>(
    count: usize,
    windows: usize,
    exclusion: usize,
    included: &[usize],
    discords: bool,
    distance: impl Fn(usize, usize, usize) -> Option<D>,
) -> MultiProfile<D> {
    let mut profile_distance = alloc::vec![Vec::with_capacity(windows); count];
    let mut profile_index = alloc::vec![Vec::with_capacity(windows); count];
    let order_of_means = MeanOrder::new(count, D::ROOT_RELATIVE_ERROR);
    let mut work = MeanWork::new();
    for i in 0..windows {
        let mut best = alloc::vec![(MeanDistance::<D>::MISSING, SampleIndex::NONE); count];
        let mut best_terms: Vec<Vec<Option<D>>> = alloc::vec![Vec::new(); count];
        for j in (0..windows).filter(|&j| i.abs_diff(j) > exclusion) {
            let distances: Vec<Option<D>> = (0..count).map(|a| distance(a, i, j)).collect();
            let order = order_dimensions(&distances, included, discords, IncludeOrder::AsGiven);
            let ordered: Vec<Option<D>> = order.iter().map(|&a| distances[a]).collect();
            for (k, slot) in best.iter_mut().enumerate() {
                let terms = &ordered[..=k];
                let value = MeanDistance::of(terms);
                let nearer =
                    order_of_means.compare(&mut work, &value, terms, &slot.0, &best_terms[k]);
                if nearer == Ordering::Less {
                    *slot = (
                        value,
                        SampleIndex::new(j as u64).expect("a window start is below NONE"),
                    );
                    best_terms[k] = terms.to_vec();
                }
            }
        }
        for (k, (value, neighbor)) in best.into_iter().enumerate() {
            profile_distance[k].push(value);
            profile_index[k].push(neighbor);
        }
    }
    MultiProfile {
        distance: profile_distance,
        index: profile_index,
    }
}

fn window_count(n: usize, m: usize) -> usize {
    if m == 0 || m > n { 0 } else { n - m + 1 }
}

/// `dimensions[a][k]` is `Some(tick)` or `None`; every dimension has the same length. Returns,
/// per window start and `k`, the minimum over windows `j` with `|i − j| > ceil(m/4)` of the
/// mean of the `k + 1` first dimensions of that column's order, ties to the smaller `j`.
/// `include` lists dimensions that come first in the given order (repeats dropped); `discords`
/// orders the present others farthest first. A dimension with a non-finite sample in either window has
/// no distance for the pair; `m == 0` or `m > n` gives no windows.
pub fn naive_multi_profile<'a, S: Into<Ticks<'a>> + Copy>(
    dimensions: &[S],
    m: usize,
    include: &[usize],
    discords: bool,
    policy: NonFinitePolicy,
) -> Result<MultiProfile, MultiError> {
    let NonFinitePolicy::Missing = policy;
    let ticks: Vec<Ticks<'a>> = dimensions.iter().map(|&series| series.into()).collect();
    let dimensions: Vec<&[Option<i64>]> = ticks.iter().map(Ticks::samples).collect();
    let (n, included) = checked(&dimensions, include)?;
    let windows = window_count(n, m);
    let stats: Vec<Vec<Window>> = ticks.iter().map(|series| series.windows(m)).collect();
    Ok(reduce(
        dimensions.len(),
        windows,
        crate::exclusion::zone(m),
        &included,
        discords,
        |a, i, j| dimension_distance(dimensions[a], &stats[a], m, i, j),
    ))
}

/// `naive_multi_profile` for the non-normalized metric: a dimension's distance is the exact
/// squared Euclidean distance of the two raw tick windows, missing for a window with a gap and
/// for a pair whose squared distance no `u128` is promised to hold (`m·R² ≥ 2¹²⁸` over every
/// present tick of the dimension through the later window).
pub fn naive_multi_profile_nonnorm(
    dimensions: &[&[Option<i64>]],
    m: usize,
    include: &[usize],
    discords: bool,
    policy: NonFinitePolicy,
) -> Result<MultiProfile<SquaredTicks>, MultiError> {
    let NonFinitePolicy::Missing = policy;
    let (n, included) = checked(dimensions, include)?;
    let windows = window_count(n, m);
    let complete: Vec<Vec<bool>> = dimensions
        .iter()
        .map(|samples| {
            (0..windows)
                .map(|start| samples[start..start + m].iter().all(Option::is_some))
                .collect()
        })
        .collect();
    Ok(reduce(
        dimensions.len(),
        windows,
        crate::exclusion::zone(m),
        &included,
        discords,
        |a, i, j| squared_ticks(dimensions[a], &complete[a], m, i, j),
    ))
}

#[cfg(test)]
mod tests;
