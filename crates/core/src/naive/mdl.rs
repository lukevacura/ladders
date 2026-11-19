//! The description-length reference: the subspace of each pair from the subspace reference, the
//! distinct differences counted in an ordered set, and `log2` computed bit by bit from exact
//! 256-bit squarings instead of read from the engine's table.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use ethnum::I256;

use super::subspace::{
    Bins, checked, naive_choose, naive_linear_bins, naive_linear_range, naive_z_bins,
};
use crate::multidim::MultiError;
use crate::quantize::Scale;
use crate::stats::Series;
use crate::subspace::{Bits, Subspace};

/// Fractional bits of the squarings; the error of the result stays below `2^-100`.
const FIXED_POINT: u32 = 120;
/// Fractional bits of the logarithm that are computed.
const LOG_BITS: u32 = 70;

/// `log2(n)` correctly rounded, for `n >= 1`.
pub fn naive_log2(n: u64) -> f64 {
    assert!(n >= 1, "the logarithm of zero");
    let whole = 63 - n.leading_zeros();
    if n == 1 << whole {
        return f64::from(whole);
    }
    let two = I256::ONE << (FIXED_POINT + 1);
    let mut x = (I256::from(n) << FIXED_POINT) >> whole;
    let mut fraction = 0u128;
    for _ in 0..LOG_BITS {
        x = (x * x) >> FIXED_POINT;
        fraction <<= 1;
        if x >= two {
            fraction |= 1;
            x >>= 1;
        }
    }
    let scaled = (u128::from(whole) << LOG_BITS) | fraction;
    let dropped = 128 - scaled.leading_zeros() - 53;
    let tail = scaled & ((1 << dropped) - 1);
    assert_ne!(tail, 1 << (dropped - 1), "a halfway case needs more bits");
    scaled as f64 / (1u128 << LOG_BITS) as f64
}

/// The reference sizes and subspaces; the result of [`crate::mdl::mdl`] computed directly.
#[allow(clippy::too_many_arguments)]
fn naive_describe(
    bins_of: impl Fn(usize, usize, usize) -> Option<Vec<u16>>,
    dimensions: usize,
    m: usize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    include: &[usize],
    discords: bool,
    bits: Bits,
) -> (Vec<f64>, Vec<Subspace>) {
    let mut sizes = Vec::new();
    let mut subspaces = Vec::new();
    for k in 0..dimensions {
        let pair: Bins = (0..dimensions)
            .map(|dimension| {
                Some((
                    bins_of(dimension, subseq_idx[k], m)?,
                    bins_of(dimension, nn_idx[k], m)?,
                ))
            })
            .collect();
        let subspace = naive_choose(&pair, k, include, discords);
        let size = if subspace.complete {
            let differences: BTreeSet<i32> = subspace
                .dimensions
                .iter()
                .flat_map(|&dimension| {
                    let (first, second) = pair[dimension].as_ref().unwrap();
                    first
                        .iter()
                        .zip(second)
                        .map(|(&a, &b)| i32::from(a) - i32::from(b))
                        .collect::<Vec<_>>()
                })
                .collect();
            let distinct = differences.len() as u64;
            let cells = (subspace.dimensions.len() * m) as f64;
            let whole =
                u64::from(bits.get()) * (2 * dimensions * m - subspace.dimensions.len() * m) as u64;
            whole as f64 + cells * naive_log2(distinct) + (distinct * u64::from(bits.get())) as f64
        } else {
            f64::INFINITY
        };
        sizes.push(size);
        subspaces.push(subspace);
    }
    (sizes, subspaces)
}

fn check_pairs(
    lengths: &[usize],
    m: NonZeroUsize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    include: &[usize],
) -> Result<(), MultiError> {
    checked(lengths, m, (0, 0), 0, include)?;
    if subseq_idx.len() != lengths.len() || nn_idx.len() != lengths.len() {
        return Err(MultiError::WrongPairCount);
    }
    let last = lengths[0] - m.get();
    if subseq_idx.iter().chain(nn_idx).any(|&start| start > last) {
        return Err(MultiError::WindowOutOfRange);
    }
    Ok(())
}

/// [`crate::mdl::mdl`], computed directly.
pub fn naive_mdl<'a, S: Into<Series<'a>> + Copy>(
    set: &[(S, Scale)],
    m: NonZeroUsize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    include: &[usize],
    discords: bool,
    bits: Bits,
) -> Result<(Vec<f64>, Vec<Subspace>), MultiError> {
    let set: Vec<(Series<'a>, Scale)> = set.iter().map(|&(s, scale)| (s.into(), scale)).collect();
    let lengths: Vec<usize> = set.iter().map(|(series, _)| series.len()).collect();
    check_pairs(&lengths, m, subseq_idx, nn_idx, include)?;
    Ok(naive_describe(
        |dimension, start, m| naive_z_bins(set[dimension].0, set[dimension].1, start, m, bits),
        set.len(),
        m.get(),
        subseq_idx,
        nn_idx,
        include,
        discords,
        bits,
    ))
}

/// [`crate::mdl::maamp_mdl`], computed directly.
pub fn naive_maamp_mdl(
    set: &[&[f64]],
    m: NonZeroUsize,
    subseq_idx: &[usize],
    nn_idx: &[usize],
    include: &[usize],
    discords: bool,
    bits: Bits,
) -> Result<(Vec<f64>, Vec<Subspace>), MultiError> {
    let lengths: Vec<usize> = set.iter().map(|series| series.len()).collect();
    check_pairs(&lengths, m, subseq_idx, nn_idx, include)?;
    let (low, high) = naive_linear_range(set);
    Ok(naive_describe(
        |dimension, start, m| naive_linear_bins(set[dimension], low, high, start, m, bits),
        set.len(),
        m.get(),
        subseq_idx,
        nn_idx,
        include,
        discords,
        bits,
    ))
}
