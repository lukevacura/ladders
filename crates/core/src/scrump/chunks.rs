//! How many diagonals each update visits: STUMPY's chunks, cut so each holds about
//! `percentage` of all the distances, by the same float arithmetic.

use super::ScrumpError;
use alloc::vec::Vec;
use core::ops::Range;

/// `Ok` for a percentage in `(0, 1]`, else [`ScrumpError::Percentage`] (NaN included).
pub(super) fn check_percentage(percentage: f64) -> Result<(), ScrumpError> {
    if percentage > 0.0 && percentage <= 1.0 {
        Ok(())
    } else {
        Err(ScrumpError::Percentage)
    }
}

/// The chunks of `order` (the diagonals in visiting order, never empty), as ranges into it.
///
/// Each diagonal weighs `distances(diagonal)`. With `chunks = ceil(1 / percentage)` and the
/// cumulative weights divided by their total, the boundary after chunk `c` (`1 <= c < chunks`)
/// is one past the first position whose share reaches `c / chunks` (`c` times `1 / chunks`, as
/// a product of two doubles). The first boundary equal to the one before it ends the cut: the
/// chunk that would have been empty runs to the end of `order` instead, and no later chunk
/// exists. `percentage` must be in `(0, 1]`.
pub(super) fn chunk_ranges(
    order: &[i64],
    distances: impl Fn(i64) -> u64,
    percentage: f64,
) -> Result<Vec<Range<usize>>, ScrumpError> {
    check_percentage(percentage)?;
    let mut cumulative: Vec<u64> = Vec::new();
    cumulative
        .try_reserve_exact(order.len())
        .map_err(|_| ScrumpError::Capacity)?;
    let mut total = 0u64;
    for &diagonal in order {
        total += distances(diagonal);
        cumulative.push(total);
    }

    let chunks = libm::ceil(1.0 / percentage);
    let step = 1.0 / chunks;
    let total = total as f64;
    let boundary = |c: u64| {
        // STUMPY's `cumsum / sum` is numba fastmath, which multiplies by the reciprocal; at an
        // exact tie the two roundings differ, so the boundary mirrors that (it depends on numba's lowering).
        1 + cumulative.partition_point(|&sum| (sum as f64 * (1.0 / total)) < c as f64 * step)
    };

    let mut ranges = Vec::new();
    let (mut start, mut previous) = (0, None);
    for c in 1..chunks as u64 {
        let end = boundary(c);
        ranges.try_reserve(1).map_err(|_| ScrumpError::Capacity)?;
        if previous == Some(end) {
            ranges.push(end..order.len());
            return Ok(ranges);
        }
        ranges.push(start..end);
        (start, previous) = (end, Some(end));
    }
    ranges.try_reserve(1).map_err(|_| ScrumpError::Capacity)?;
    ranges.push(start..order.len());
    Ok(ranges)
}
