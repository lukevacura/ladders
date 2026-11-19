//! The description length of a motif pair at each subspace size, following STUMPY 1.14.1's `mdl`
//! and `maamp_mdl`: the bits needed to describe the pair's windows as one window plus the
//! differences of the other from it, over the dimensions of the subspace.
//!
//! For each `k` the pair is the window starts `subseq_idx[k]` and `nn_idx[k]`, normally the
//! motif pair of the `k + 1`-dimensional profile. Its subspace is [`crate::subspace`]'s, and with
//! `s` dimensions, `d` in all, `bits` per sample and `distinct` different values among the `s · m`
//! bin differences of the subspace, the size is
//! `bits · (2·d·m − s·m) + s·m · log2(distinct) + distinct · bits`. The integer parts are exact;
//! `log2` is a correctly rounded table, and the sum is formed in that order in `f64`.
//!
//! Difference from STUMPY: a pair whose subspace holds a dimension without a distance (a
//! non-finite sample in one of its windows) has no description: its size is infinity, so a
//! smallest size never picks it.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::multidim::{MultiError, NonFinitePolicy, checked_include};
use crate::quantize::Scale;
use crate::stats::Series;
use crate::subspace::{Discretizer, Options, Pair, Subspace, choose};

mod log2;
#[cfg(test)]
mod tests;

use log2::LOG2;

/// The sizes and subspaces of every `k`.
#[derive(Clone, Debug, PartialEq)]
pub struct Description {
    /// The size in bits for each `k`; infinity where the subspace is incomplete.
    pub bit_sizes: Vec<f64>,
    /// The subspace of each `k`: `k + 1` dimensions.
    pub subspaces: Vec<Subspace>,
}

/// The number of different values among the bin differences of the dimensions in `subspace`.
fn distinct_differences(pair: &Pair, subspace: &Subspace, bits: u32) -> usize {
    let reach = (1usize << bits) - 1;
    let mut seen = alloc::vec![false; 2 * reach + 1];
    for &dimension in &subspace.dimensions {
        let (first, second) = pair[dimension]
            .as_ref()
            .expect("a complete subspace has bins in every dimension");
        for (&a, &b) in first.iter().zip(second) {
            seen[reach + usize::from(a) - usize::from(b)] = true;
        }
    }
    seen.into_iter().filter(|&seen| seen).count()
}

/// STUMPY's size, evaluated as it evaluates it: `(whole + (cells · log2(distinct))) +
/// distinct · bits`, the integer terms converted to `f64` where STUMPY converts them.
fn bit_size(bits: u32, dimensions: usize, subspace: usize, m: usize, distinct: usize) -> f64 {
    let whole =
        u128::from(bits) * (2 * dimensions as u128 * m as u128 - subspace as u128 * m as u128);
    let cells = subspace as u128 * m as u128;
    (whole as f64 + cells as f64 * LOG2[distinct - 1])
        + (distinct as u128 * u128::from(bits)) as f64
}

fn describe(
    discretizer: &Discretizer<'_>,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    options: &Options<'_>,
) -> Result<Description, MultiError> {
    let dimensions = discretizer.dimensions();
    let include = checked_include(options.include, dimensions)?;
    if subseq_idx.len() != dimensions || nn_idx.len() != dimensions {
        return Err(MultiError::WrongPairCount);
    }
    for &start in subseq_idx.iter().chain(nn_idx) {
        discretizer.check_start(start)?;
    }
    let NonFinitePolicy::Missing = options.non_finite;
    let bits = u32::from(options.bits.get());
    let mut bit_sizes = Vec::with_capacity(dimensions);
    let mut subspaces = Vec::with_capacity(dimensions);
    for (k, (&first, &second)) in subseq_idx.iter().zip(nn_idx).enumerate() {
        let pair = discretizer.pair(first, second);
        let subspace = choose(&pair, k, &include, options.discords);
        bit_sizes.push(if subspace.complete {
            let distinct = distinct_differences(&pair, &subspace, bits);
            bit_size(
                bits,
                dimensions,
                subspace.dimensions.len(),
                discretizer.window_length(),
                distinct,
            )
        } else {
            f64::INFINITY
        });
        subspaces.push(subspace);
    }
    Ok(Description {
        bit_sizes,
        subspaces,
    })
}

/// The description length of the pair at every subspace size, over z-normalized series, each
/// quantized at its own scale: for each dimension `k`, the pair of windows of length `m` at
/// `subseq_idx[k]` and `nn_idx[k]`. `Err` for no series, series of different lengths, a window
/// longer than them, an `include` entry that is not a dimension, a window start that is not a
/// window, or starts that are not one per dimension.
pub fn mdl<'s, S: Into<Series<'s>> + Copy>(
    set: &[(S, Scale)],
    m: NonZeroUsize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    options: &Options<'_>,
) -> Result<Description, MultiError> {
    let discretizer = Discretizer::z_normalized(set, m, options.bits)?;
    describe(&discretizer, subseq_idx, nn_idx, options)
}

/// [`mdl`] for series that are not z-normalized: STUMPY's `maamp_mdl` with `p = 2`.
pub fn maamp_mdl(
    set: &[&[f64]],
    m: NonZeroUsize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    options: &Options<'_>,
) -> Result<Description, MultiError> {
    let discretizer = Discretizer::linear(set, m, options.bits)?;
    describe(&discretizer, subseq_idx, nn_idx, options)
}
