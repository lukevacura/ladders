//! The narrow chunk pass: one chunk of a row's diagonals advanced and ranked on i32 offsets.

use super::{ColumnAcc, RowAcc, RowFast};
use crate::columns;
use crate::estimate::{F64, Producer};
use crate::kbest::Blocks;
use crate::kernel_neon::{self, GROUP_LANES, RowScalars, SPAN_GROUPS, SPAN_LANES};
use crate::narrow;
use crate::right::ColumnSink;
use crate::stats::Window;

/// One contiguous, lag-ordered chunk of a row's neighbors: the five per-lane columns, the
/// absolute index of the chunk's first neighbor (so candidate indices are absolute), and how to
/// read a lane's whole [`Window`] back — called only for a candidate, which is rare.
pub struct ChunkCols<'a> {
    pub enter: &'a [i32],
    pub leave: &'a [i32],
    pub sum: &'a [i32],
    pub r: &'a [f64],
    pub bias: &'a [f64],
    pub first: usize,
    /// The neighbor window at chunk lane `lane` (`0..len`).
    pub neighbor: &'a dyn Fn(usize) -> Window,
}

/// Advances and ranks one contiguous chunk of continuing diagonals into `acc`. `tile` (at least
/// `qt.len()` long) holds this chunk's `ρ̂`; a non-finite row only advances.
pub fn pass_chunk(
    qt: &mut [i64],
    cols: ChunkCols<'_>,
    row: RowScalars,
    acc: &mut RowAcc,
    tile: &mut [f64],
) {
    pass_chunk_offering::<ColumnAcc>(qt, cols, row, acc, tile, None);
}

/// [`pass_chunk`] that also offers every ranked pair to `right`'s columns, as row `right.1`: the
/// kernel checks each lane against its column's floor as it ranks it, so only a group holding a
/// lane at or above its floor is read again.
pub(crate) fn pass_chunk_offering<S: ColumnSink>(
    qt: &mut [i64],
    cols: ChunkCols<'_>,
    row: RowScalars,
    acc: &mut RowAcc,
    tile: &mut [f64],
    mut right: Option<(&mut S, usize)>,
) {
    let len = qt.len();
    assert!(
        cols.enter.len() == len
            && cols.leave.len() == len
            && cols.sum.len() == len
            && cols.r.len() == len
            && cols.bias.len() == len
            && tile.len() >= len,
        "a chunk's columns and tile cover exactly its lanes"
    );
    #[cfg(test)]
    acc.visits.extend(cols.first..cols.first + len);
    let Some(kind) = acc.row() else {
        for (q, (&te, &tl)) in qt.iter_mut().zip(cols.enter.iter().zip(cols.leave)) {
            *q = narrow::advance(*q, row.me, row.ml, te, tl);
        }
        return;
    };
    let id = right.as_ref().map(|(_, row)| acc.id(*row));
    let mut maxima = [f64::NEG_INFINITY; SPAN_GROUPS];
    // A plain loop over the spans: a chunk is often one span, and a stepped range costs more to
    // start than the span's own bookkeeping.
    let mut lo = 0;
    while lo < len {
        let hi = (lo + SPAN_LANES).min(len);
        let span = &mut tile[lo..hi];
        let maxima = &mut maxima[..(hi - lo).div_ceil(GROUP_LANES)];
        let hits = match kind {
            RowFast::Constant => {
                for (j, rho_hat) in (lo..hi).zip(span.iter_mut()) {
                    qt[j] = narrow::advance(qt[j], row.me, row.ml, cols.enter[j], cols.leave[j]);
                    *rho_hat = columns::constant_row_rank(cols.r[j], cols.bias[j]);
                }
                group_maxima(span, maxima);
                None
            }
            RowFast::Valid { .. } => Some(kernel_neon::rank_row_groups(
                &mut qt[lo..hi],
                &cols.enter[lo..hi],
                &cols.leave[lo..hi],
                &cols.sum[lo..hi],
                &cols.r[lo..hi],
                &cols.bias[lo..hi],
                row,
                span,
                right
                    .as_ref()
                    .map(|(columns, _)| columns.floors(cols.first + lo, hi - lo)),
                maxima,
            )),
        };
        let qt = &qt[lo..hi];
        let n_of = |lane: usize| narrow::cross(qt[lane], row.si, cols.sum[lo + lane]);
        acc.absorb_with(
            Blocks::<RankRow>::new(span, maxima),
            cols.first + lo,
            |lane| (cols.neighbor)(lo + lane),
            n_of,
            |lane| columns::is_constant(cols.r[lo + lane], cols.bias[lo + lane]),
        );
        if let (Some((sink, _)), Some(row)) = (right.as_mut(), id) {
            let variance_of = |lane: usize| columns::variance(&(cols.neighbor)(lo + lane));
            match hits {
                Some(hits) => sink.offer_groups::<RankRow>(
                    cols.first + lo,
                    row,
                    span,
                    hits,
                    n_of,
                    variance_of,
                ),
                None => sink.offer_chunk::<RankRow>(cols.first + lo, row, span, n_of, variance_of),
            }
        }
        lo = hi;
    }
}

/// Each [`GROUP_LANES`]-lane group's largest `ρ̂` of `tile`, into `maxima` (one slot per group).
pub(super) fn group_maxima(tile: &[f64], maxima: &mut [f64]) {
    for (max, group) in maxima.iter_mut().zip(tile.chunks(GROUP_LANES)) {
        *max = group
            .iter()
            .fold(f64::NEG_INFINITY, |max, &rho_hat| max.max(rho_hat));
    }
}

/// The narrow kernel ([`kernel_neon::rank_row_groups`], NEON or its scalar body alike) as a
/// producer of estimates: it evaluates [`rank_estimate`](crate::estimate::rank_estimate), so it
/// declares [`F64`]'s bound.
pub(crate) struct RankRow;

impl Producer for RankRow {
    const DELTA: f64 = F64::DELTA;
}
