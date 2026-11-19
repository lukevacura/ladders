//! Time-series chains over left and right nearest-neighbor indices (STUMPY's `atsc` and `allc`).
//!
//! Window `i` links to `right[i]` when that window's left neighbor is `i` again: the two are each
//! other's nearest in time order. Following links from a window gives its anchored chain.
//! Every window has at most one outgoing and one incoming link, so the links form disjoint
//! paths; [`all_chains`] lists every maximal path, and the longest is the unanchored chain.
//!
//! The index arrays must look like a two-sided profile's: a left neighbor before its window, a
//! right neighbor after it. That rules out cycles, so every walk is linear in the chain it
//! returns and the whole set is linear in the number of windows.

use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroUsize;

use crate::batch::two_sided_profile;
use crate::index::SampleIndex;
use crate::quantize::Scale;

/// Why a pair of index arrays cannot be read as left and right neighbors of one series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChainInputError {
    LengthMismatch,
    /// A neighbor or the anchor is not a window of the series.
    OutOfRange,
    /// A left neighbor at or after its window, or a right neighbor at or before it.
    NotForward,
}

impl fmt::Display for ChainInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LengthMismatch => "left and right indices must have the same length",
            Self::OutOfRange => "an index is not a window of the series",
            Self::NotForward => {
                "left neighbors must precede and right neighbors follow their window"
            }
        })
    }
}

/// Every maximal chain, ordered by first window, and the longest of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllChains {
    pub chains: Vec<Vec<usize>>,
    /// The first of the longest chains in `chains` (the one with the smallest first window);
    /// empty when there are no windows.
    pub longest: Vec<usize>,
}

fn checked(left: &[SampleIndex], right: &[SampleIndex]) -> Result<(), ChainInputError> {
    if left.len() != right.len() {
        return Err(ChainInputError::LengthMismatch);
    }
    let windows = left.len() as u64;
    for &side in left.iter().chain(right) {
        if side.get().is_some_and(|i| i >= windows) {
            return Err(ChainInputError::OutOfRange);
        }
    }
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        let i = i as u64;
        if l.get().is_some_and(|l| l >= i) || r.get().is_some_and(|r| r <= i) {
            return Err(ChainInputError::NotForward);
        }
    }
    Ok(())
}

fn link(left: &[SampleIndex], right: &[SampleIndex], window: usize) -> Option<usize> {
    let next = right[window].get()? as usize;
    (left[next].get() == Some(window as u64)).then_some(next)
}

fn walk(left: &[SampleIndex], right: &[SampleIndex], first: usize) -> Vec<usize> {
    let mut chain = alloc::vec![first];
    let mut window = first;
    while let Some(next) = link(left, right, window) {
        chain.push(next);
        window = next;
    }
    chain
}

/// The chain that starts at window `anchor` and follows links to the right.
pub fn anchored_chain(
    left: &[SampleIndex],
    right: &[SampleIndex],
    anchor: usize,
) -> Result<Vec<usize>, ChainInputError> {
    checked(left, right)?;
    if anchor >= left.len() {
        return Err(ChainInputError::OutOfRange);
    }
    Ok(walk(left, right, anchor))
}

/// Every maximal chain: one starts at each window no link enters, so every window is in exactly
/// one chain.
pub fn all_chains(
    left: &[SampleIndex],
    right: &[SampleIndex],
) -> Result<AllChains, ChainInputError> {
    checked(left, right)?;
    let mut entered = alloc::vec![false; left.len()];
    for window in 0..left.len() {
        if let Some(next) = link(left, right, window) {
            entered[next] = true;
        }
    }
    let chains: Vec<Vec<usize>> = (0..left.len())
        .filter(|&window| !entered[window])
        .map(|window| walk(left, right, window))
        .collect();
    let mut longest: &[usize] = &[];
    for chain in &chains {
        if chain.len() > longest.len() {
            longest = chain;
        }
    }
    let longest = longest.to_vec();
    Ok(AllChains { chains, longest })
}

/// [`all_chains`] over the left and right neighbors of `series`' exact two-sided profile.
/// Looks ahead, like [`two_sided_profile`]; the thread count never changes a bit.
pub fn chains_from_series(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> AllChains {
    let profile = two_sided_profile(series, m, scale, threads);
    let left: Vec<SampleIndex> = profile.iter().map(|(_, s)| s.left.index()).collect();
    let right: Vec<SampleIndex> = profile.iter().map(|(_, s)| s.right.index()).collect();
    all_chains(&left, &right).expect("a two-sided profile's neighbors lie on their own side")
}
