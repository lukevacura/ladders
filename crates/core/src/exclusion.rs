//! The exclusion zone: a neighbor at most `ceil(m/4)` samples away is never a match (STUMPY's rule).

/// The zone's half-width for windows of length `m`.
#[inline(always)]
#[must_use]
pub const fn zone(m: usize) -> usize {
    m.div_ceil(4)
}
