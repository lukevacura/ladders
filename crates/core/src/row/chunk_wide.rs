//! The wide chunk pass: one chunk of a row's diagonals advanced and ranked on exact i64 ticks.

use super::{RowAcc, RowFast, advance_lane, centered_cross_fast};
use crate::columns;
use crate::estimate::{F64, Producer};
use crate::kbest::Blocks;
use crate::kernel_neon::{GROUP_LANES, SPAN_GROUPS, SPAN_LANES};
use crate::stats::Window;

/// [`ChunkCols`](super::ChunkCols) on the exact i64 ticks and sums, for a tile whose window range is too wide for
/// any anchor.
pub struct WideChunkCols<'a> {
    pub enter: &'a [i64],
    pub leave: &'a [i64],
    pub sum: &'a [i64],
    pub r: &'a [f64],
    pub bias: &'a [f64],
    pub first: usize,
    pub neighbor: &'a dyn Fn(usize) -> Window,
}

/// The wide pass as a producer of estimates: the same ranking expression on exact i64 cross
/// terms, so the same bound as [`RankRow`](super::RankRow).
pub(crate) struct WideRank;

impl Producer for WideRank {
    const DELTA: f64 = F64::DELTA;
}

/// What a row contributes to a wide chunk: its entering and leaving tick and its window sum.
#[derive(Clone, Copy)]
pub struct WideRow {
    pub enter: i64,
    pub leave: i64,
    pub sum: i64,
}

/// [`pass_chunk`](super::pass_chunk) on the exact i64 arithmetic, with the same ranking expressions and the same
/// accumulator: a chunk whose offsets are too wide for any anchor yields the partial a narrow
/// chunk would.
pub fn pass_chunk_wide(
    qt: &mut [i64],
    cols: WideChunkCols<'_>,
    row: WideRow,
    m: usize,
    acc: &mut RowAcc,
    tile: &mut [f64],
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
    let m_i64 = m as i64;
    let Some(kind) = acc.row() else {
        for ((q, &te), &tl) in qt.iter_mut().zip(cols.enter).zip(cols.leave) {
            *q = advance_lane(*q, row.enter, row.leave, te, tl);
        }
        return;
    };
    let mut maxima = [f64::NEG_INFINITY; SPAN_GROUPS];
    for lo in (0..len).step_by(SPAN_LANES) {
        let hi = (lo + SPAN_LANES).min(len);
        let maxima = &mut maxima[..(hi - lo).div_ceil(GROUP_LANES)];
        let groups = (&mut *qt, &mut *tile, &mut *maxima, lo..hi);
        match kind {
            RowFast::Constant => rank_wide_groups(groups, &cols, row, |_, j| {
                columns::constant_row_rank(cols.r[j], cols.bias[j])
            }),
            RowFast::Valid { inv_sqrt } => rank_wide_groups(groups, &cols, row, |qt, j| {
                let n = centered_cross_fast(qt, m_i64, row.sum, cols.sum[j]);
                columns::valid_row_rank(n, inv_sqrt, cols.r[j], cols.bias[j])
            }),
        }
        acc.absorb_with(
            Blocks::<WideRank>::new(&tile[lo..hi], maxima),
            cols.first + lo,
            |lane| (cols.neighbor)(lo + lane),
            |lane| centered_cross_fast(qt[lo + lane], m_i64, row.sum, cols.sum[lo + lane]),
            |lane| columns::is_constant(cols.r[lo + lane], cols.bias[lo + lane]),
        );
    }
}

/// One wide span's lanes `lo..hi`, group by group: each advanced, ranked by `lane(qt, j)` into
/// `tile` and folded into its group's maximum.
#[inline(always)]
pub(super) fn rank_wide_groups(
    (qt, tile, maxima, span): (&mut [i64], &mut [f64], &mut [f64], core::ops::Range<usize>),
    cols: &WideChunkCols,
    row: WideRow,
    lane: impl Fn(i64, usize) -> f64,
) {
    for (g, max) in maxima.iter_mut().enumerate() {
        let mut group_max = f64::NEG_INFINITY;
        for j in span.start + g * GROUP_LANES..(span.start + (g + 1) * GROUP_LANES).min(span.end) {
            qt[j] = advance_lane(qt[j], row.enter, row.leave, cols.enter[j], cols.leave[j]);
            tile[j] = lane(qt[j], j);
            group_max = group_max.max(tile[j]);
        }
        *max = group_max;
    }
}
