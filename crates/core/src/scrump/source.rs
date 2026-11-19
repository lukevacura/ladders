//! Where the order of the diagonals comes from: always the caller, never ambient state.

use super::ScrumpError;
use alloc::vec::Vec;

/// A stream of 64-bit words the caller owns. Implement it to draw the order of a [`Shuffled`]
/// source from your own generator; [`SplitMix64`] is the built-in one.
pub trait RandomSource {
    /// The next word of the stream.
    fn next_u64(&mut self) -> u64;

    /// A uniform value below `bound` (which must be positive), by Lemire's multiply-and-reject
    /// reduction: exactly unbiased, and the same words consumed on every platform. Zero for a
    /// `bound` of zero.
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u64) < threshold {
                product = u128::from(self.next_u64()) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }
}

/// With the `rand` feature every [`rand_core::RngCore`] drives a [`Shuffled`] order; its words
/// are its `next_u64`s, so the order is the generator's own and as stable as it is.
#[cfg(feature = "rand")]
impl<G: rand_core::RngCore> RandomSource for G {
    fn next_u64(&mut self) -> u64 {
        rand_core::RngCore::next_u64(self)
    }
}

/// Steele, Lea and Flood's SplitMix64: a 64-bit counter stepped by `0x9E37_79B9_7F4A_7C15` and
/// scrambled by two multiply-xorshift rounds. Its output for a seed is fixed, so the same seed
/// gives the same order on every platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomSource for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The source of the order in which an anytime profile visits its diagonals: `count` diagonals
/// numbered `first..first + count` (a diagonal is the column minus the row of its pairs), each
/// exactly once.
pub trait DiagonalSource {
    /// The visiting order, a permutation of `first..first + count`.
    fn order(self, first: i64, count: usize) -> Result<Vec<i64>, ScrumpError>;
}

/// A Fisher-Yates shuffle of the ascending diagonals drawn from `G`: for `i` from `count - 1`
/// down to 1, the diagonal at position `i` swaps with the one at position `below(i + 1)`. The
/// shuffle and the reduction of [`RandomSource::below`] are the whole contract: one seed, one
/// order, on every platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shuffled<G>(pub G);

impl Shuffled<SplitMix64> {
    /// The shuffle driven by [`SplitMix64`] from `seed`.
    pub const fn seeded(seed: u64) -> Self {
        Self(SplitMix64::new(seed))
    }
}

impl<G: RandomSource> DiagonalSource for Shuffled<G> {
    fn order(mut self, first: i64, count: usize) -> Result<Vec<i64>, ScrumpError> {
        let mut order = ascending(first, count)?;
        for at in (1..count).rev() {
            let with = self.0.below(at as u64 + 1) as usize;
            order.swap(at, with);
        }
        Ok(order)
    }
}

/// An order the caller made: it must hold every diagonal of `first..first + count` exactly once,
/// else [`ScrumpError::NotAPermutation`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Explicit(pub Vec<i64>);

impl DiagonalSource for Explicit {
    fn order(self, first: i64, count: usize) -> Result<Vec<i64>, ScrumpError> {
        if self.0.len() != count {
            return Err(ScrumpError::NotAPermutation);
        }
        let mut seen: Vec<bool> = Vec::new();
        seen.try_reserve_exact(count)
            .map_err(|_| ScrumpError::Capacity)?;
        seen.resize(count, false);
        for &diagonal in &self.0 {
            let at = diagonal
                .checked_sub(first)
                .and_then(|at| usize::try_from(at).ok())
                .filter(|&at| at < count)
                .ok_or(ScrumpError::NotAPermutation)?;
            if core::mem::replace(&mut seen[at], true) {
                return Err(ScrumpError::NotAPermutation);
            }
        }
        Ok(self.0)
    }
}

/// `first..first + count` as a vector.
fn ascending(first: i64, count: usize) -> Result<Vec<i64>, ScrumpError> {
    let mut order = Vec::new();
    order
        .try_reserve_exact(count)
        .map_err(|_| ScrumpError::Capacity)?;
    order.extend((0..count).map(|at| first + at as i64));
    Ok(order)
}
