//! A neighbor candidate: its correlation, its squared distance, and its index. Ranking one
//! candidate against another — exact nearest, ties to the smaller index — is
//! [`crate::nearest`]'s job, never a comparison on `Entry` itself.

use crate::index::SampleIndex;
use alloc::vec::Vec;

/// A candidate neighbor: its correlation, its squared distance, and its index.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    rho: f64,
    squared_distance: f64,
    index: SampleIndex,
}

impl Entry {
    /// No neighbor: the worst possible correlation and distance, and [`SampleIndex::NONE`].
    pub const NONE: Self = Self {
        rho: f64::NEG_INFINITY,
        squared_distance: f64::INFINITY,
        index: SampleIndex::NONE,
    };

    /// `None` if `rho` or `squared_distance` is NaN: a reported entry never carries one.
    pub fn new(rho: f64, squared_distance: f64, index: SampleIndex) -> Option<Self> {
        if rho.is_nan() || squared_distance.is_nan() {
            None
        } else {
            Some(Self {
                rho,
                squared_distance,
                index,
            })
        }
    }

    pub const fn rho(self) -> f64 {
        self.rho
    }

    pub const fn squared_distance(self) -> f64 {
        self.squared_distance
    }

    pub const fn index(self) -> SampleIndex {
        self.index
    }
}

/// The left and right nearest neighbor, and the better of the two. Looking right trades on
/// future information; this type's own name says so.
#[derive(Clone, Copy, Debug)]
pub struct TwoSided {
    pub left: Entry,
    pub right: Entry,
    pub nearest: Entry,
}

/// [`TwoSided`] with the `k` nearest neighbors over both sides: `left` and `right` stay each
/// side's one nearest, and `nearest` holds up to `k` entries, nearest first (none for a window
/// with no neighbor at all).
#[derive(Clone, Debug)]
pub struct TwoSidedTopk {
    pub left: Entry,
    pub right: Entry,
    pub nearest: Vec<Entry>,
}

#[cfg(test)]
mod tests;
