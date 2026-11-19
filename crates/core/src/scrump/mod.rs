//! Anytime matrix profiles: `scrump` (z-normalized) and `scraamp` (non-normalized), STUMPY's
//! SCRIMP, which visits a random subset of the diagonals of the distance matrix and refines the
//! profile with `update()`.
//!
//! The profile after any number of updates is the exact profile over the diagonals visited so
//! far, on bits (ties to the smaller index, whatever the order or the thread count); once every
//! diagonal is visited it is the full exact profile, and further updates change nothing.
//!
//! A z-normalized window with every sample present whose variance numerator passes `2⁶³` has no
//! distance in the hot passes; the batch cold route ranks it ([`crate::spill`]). The data alone
//! decides which windows these are (of either series of a join), so [`Scrump`] and [`ScrumpJoin`]
//! rank every pair that touches one when they are made, at a cost proportional to those pairs,
//! and hold them from then on, before the first update too: the profile is the exact one over
//! the diagonals visited and those pairs. Which pairs these are depends on neither a forced route
//! nor the cost model, so a partial profile is the same under every route and thread count. Every
//! other pair, including one with an outlier tick, is visited with its diagonal.
//!
//! The order the diagonals are visited in comes from a [`DiagonalSource`] the caller passes,
//! never from ambient state: [`Shuffled`] drives a fixed shuffle from a generator the caller owns
//! ([`SplitMix64`] by default), [`Explicit`] takes an order the caller made.
//! With the `rand` feature every `rand_core::RngCore` is a [`RandomSource`], so `Shuffled(rng)` draws
//! the order from it; the orders [`Shuffled::seeded`] gives never depend on that crate.
//!
//! The procedure is STUMPY 1.14.1's. A self-join visits the diagonals `ceil(m / 4) + 1 ..
//! windows`, an AB-join the diagonals `-(windows_a - 1) .. windows_b` (a diagonal is the `B`
//! window minus the `A` window of its pairs). The visiting order is cut into chunks holding about
//! `percentage` of all the distances each, by STUMPY's float arithmetic, and each `update()` visits
//! the next chunk. As in STUMPY nothing is visited until the first `update()`, so before it
//! the profile holds only the pairs ranked when it was made (none on most series).
//!
//! Differences from STUMPY:
//! - a `percentage` outside `(0, 1]` or NaN is an error (STUMPY clips a large one to one chunk and
//!   divides by a zero one); an order that is not a permutation of the diagonals is an error;
//! - exact ties go to the smaller index whatever the chunk, where STUMPY keeps the earlier
//!   chunk's neighbor; and the distances are exact where STUMPY's are rounded floats, so the two
//!   differ on near ties;
//! - `pre_scrump` is not available yet.
//!
//! One procedure serves every metric and both joins: it knows only the geometry of the diagonals
//! and an engine that visits a set of them and keeps the profile. The engines reuse the batch tile
//! scheduler over the long runs of visited diagonals (a tile walks only the visited diagonals of
//! its range), and the z-normalized engines walk the short runs one diagonal at a time with the
//! exact integer recurrence, because the tiled pass pays its set-up once per run per block of rows.
//! A visit therefore costs about its pairs, whatever the order; every run is exact either way, on
//! the same bits.

mod chunks;
mod nonnorm;
mod source;
mod znorm;

use crate::batch::SlotRuns;
use crate::capacity::CapacityError;
use alloc::vec::Vec;
use core::ops::Range;

pub use self::nonnorm::{Scraamp, ScraampJoin};
pub use self::source::{DiagonalSource, Explicit, RandomSource, Shuffled, SplitMix64};
pub use self::znorm::{Scrump, ScrumpJoin};

/// Why an anytime profile could not be made or updated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScrumpError {
    /// The percentage is not in `(0, 1]` (NaN included).
    Percentage,
    /// The series has no window, or, for a self-join, none with a pair outside its exclusion
    /// zone: there is no diagonal to visit.
    NoDiagonals,
    /// The explicit order is not a permutation of the diagonals.
    NotAPermutation,
    /// An allocation the profile needs is too large to make.
    Capacity,
}

impl From<CapacityError> for ScrumpError {
    fn from(_: CapacityError) -> Self {
        Self::Capacity
    }
}

/// Which diagonals exist and how a diagonal's number maps to the tile engine's slots and to the
/// number of distances on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Geometry {
    /// One series against itself: the diagonals past the exclusion zone.
    SelfJoin { windows: usize, exclusion: usize },
    /// Every window of `a` against every window of `b`: every diagonal.
    Join { windows_a: usize, windows_b: usize },
}

impl Geometry {
    /// The self-join of a series of `len` samples at window length `m`.
    pub(crate) fn self_join(len: usize, m: usize) -> Self {
        Self::SelfJoin {
            windows: (len + 1).saturating_sub(m),
            exclusion: crate::exclusion::zone(m),
        }
    }

    /// The join of series of `len_a` and `len_b` samples at window length `m`.
    pub(crate) fn join(len_a: usize, len_b: usize, m: usize) -> Self {
        Self::Join {
            windows_a: (len_a + 1).saturating_sub(m),
            windows_b: (len_b + 1).saturating_sub(m),
        }
    }

    /// The first diagonal and how many there are; `NoDiagonals` for none.
    pub(crate) fn range(self) -> Result<(i64, usize), ScrumpError> {
        match self {
            Self::SelfJoin { windows, exclusion } => {
                match windows
                    .checked_sub(exclusion + 1)
                    .filter(|&count| count > 0)
                {
                    Some(count) => Ok((exclusion as i64 + 1, count)),
                    None => Err(ScrumpError::NoDiagonals),
                }
            }
            Self::Join {
                windows_a,
                windows_b,
            } => {
                if windows_a == 0 || windows_b == 0 {
                    return Err(ScrumpError::NoDiagonals);
                }
                Ok((1 - windows_a as i64, windows_a + windows_b - 1))
            }
        }
    }

    /// The tile engine's slot of `diagonal`.
    pub(crate) fn slot(self, diagonal: i64) -> usize {
        match self {
            Self::SelfJoin { exclusion, .. } => diagonal as usize - exclusion - 1,
            Self::Join { windows_b, .. } => (windows_b as i64 - 1 - diagonal) as usize,
        }
    }

    /// The distances on `diagonal`.
    pub(crate) fn distances(self, diagonal: i64) -> u64 {
        match self {
            Self::SelfJoin { windows, .. } => windows as u64 - diagonal as u64,
            Self::Join {
                windows_a,
                windows_b,
            } => {
                let (a, b) = (windows_a as i64, windows_b as i64);
                (if diagonal >= 0 {
                    (b - diagonal).min(a)
                } else {
                    b.min(a + diagonal)
                }) as u64
            }
        }
    }
}

/// What one metric brings to the procedure: visiting a set of diagonals and keeping the profile
/// they give, merged into everything visited before.
pub(crate) trait Engine {
    /// Visits the diagonals of `runs` (tile engine slots). On an error the profile is unchanged.
    fn visit(&mut self, runs: &SlotRuns) -> Result<(), ScrumpError>;
}

/// The procedure over an engine: the order of the diagonals, its chunks and how many have run.
pub(crate) struct Anytime<E> {
    pub(crate) engine: E,
    geometry: Geometry,
    order: Vec<i64>,
    chunks: Vec<Range<usize>>,
    done: usize,
}

impl<E: Engine> Anytime<E> {
    /// `engine` over `geometry`'s diagonals in the order `source` gives, in chunks of
    /// `percentage` of the distances each.
    pub(crate) fn new(
        engine: E,
        geometry: Geometry,
        percentage: f64,
        source: impl DiagonalSource,
    ) -> Result<Self, ScrumpError> {
        chunks::check_percentage(percentage)?;
        let (first, count) = geometry.range()?;
        let order = source.order(first, count)?;
        let chunks = chunks::chunk_ranges(&order, |d| geometry.distances(d), percentage)?;
        Ok(Self {
            engine,
            geometry,
            order,
            chunks,
            done: 0,
        })
    }

    /// Visits the next chunk of diagonals; nothing once all are visited.
    pub(crate) fn update(&mut self) -> Result<(), ScrumpError> {
        let Some(chunk) = self.chunks.get(self.done) else {
            return Ok(());
        };
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(chunk.len())
            .map_err(|_| ScrumpError::Capacity)?;
        slots.extend(
            self.order[chunk.clone()]
                .iter()
                .map(|&diagonal| self.geometry.slot(diagonal)),
        );
        self.engine.visit(&SlotRuns::from_slots(slots))?;
        self.done += 1;
        Ok(())
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.done == self.chunks.len()
    }

    pub(crate) fn updates(&self) -> usize {
        self.done
    }

    pub(crate) fn chunks(&self) -> usize {
        self.chunks.len()
    }

    /// The diagonals visited so far, in visiting order.
    pub(crate) fn visited(&self) -> &[i64] {
        let end = self
            .done
            .checked_sub(1)
            .map_or(0, |last| self.chunks[last].end);
        &self.order[..end]
    }
}

#[cfg(test)]
mod tests;
