//! The diagonal-major walk: each visited diagonal run down its rows with the exact integer
//! recurrence, for diagonals too scattered for the tiled pass, whose set-up is paid once per run
//! of consecutive diagonals per block of rows.
//!
//! A pair's window dot product is carried along its diagonal exactly (modulo `2⁶⁴`, as the exact
//! i64 pass carries it), its centered cross term comes from the two window sums, and the pair is
//! offered to its row's [`Nearest`], which settles the exact order and ties as everywhere. An offer
//! clearly worse than the kept set is dropped on the estimate alone, which is what the offer would
//! decide, so the kept neighbors are the exact nearest, equal on bits to the tiled pass's.
//!
//! Each row owns its reduction, so bands of rows run on separate threads with no lock. A pair
//! `(i, j)` is walked once from `i` and once from `j`, by the caller giving each side's offsets.

use super::quantized::Quantized;
use crate::capacity::CapacityError;
use crate::capacity::reserved_vec;
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::estimate::rank_estimate;
use crate::index::SampleIndex;
use crate::nearest::Nearest;
use crate::row::centered_cross_fast;
use crate::stats::WindowKind;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Rows per task: enough that a diagonal's start (`m` multiplies) is a small part of its walk.
fn band_rows(m: usize) -> usize {
    (8 * m).max(2048)
}

/// Rows per block of the walk: long enough to vectorize, short enough to stay in the L1.
const BLOCK: usize = 256;

const NON_FINITE: u8 = 0;
const CONSTANT: u8 = 1;
const VALID: u8 = 3;

/// One series' windows as parallel arrays: sum, reciprocal root of the variance (for a valid
/// window) and kind.
struct Columns {
    sum: Vec<i64>,
    inv: Vec<f64>,
    kind: Vec<u8>,
}

impl Columns {
    fn of(q: &Quantized) -> Result<Self, CapacityError> {
        let n = q.windows.len();
        let (mut sum, mut inv, mut kind) = (Vec::new(), Vec::new(), Vec::new());
        sum.try_reserve_exact(n).map_err(|_| CapacityError)?;
        inv.try_reserve_exact(n).map_err(|_| CapacityError)?;
        kind.try_reserve_exact(n).map_err(|_| CapacityError)?;
        for window in &q.windows {
            sum.push(window.sum);
            match window.kind {
                WindowKind::NonFinite => (inv.push(0.0), kind.push(NON_FINITE)),
                WindowKind::Constant => (inv.push(0.0), kind.push(CONSTANT)),
                WindowKind::Valid { inv_sqrt, .. } => (inv.push(inv_sqrt), kind.push(VALID)),
            };
        }
        Ok(Self { sum, inv, kind })
    }
}

/// Each window of `rows`' up to `k` nearest partners among the windows of `cols` at `offsets` (a
/// partner of window `r` is `r + offset`, where it exists), the way [`Nearest::push_kept`] lays
/// them out: per window, in order, a run of entries nearest first, or one empty entry.
pub(crate) fn walk(
    (rows, cols): (&Quantized, &Quantized),
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    offsets: &[i64],
) -> Result<Vec<(u64, Entry)>, CapacityError> {
    let n = rows.windows.len();
    let band = band_rows(m.get());
    let bands = n.div_ceil(band);
    let (row_columns, col_columns) = (Columns::of(rows)?, Columns::of(cols)?);
    let task = |at: usize| {
        let range = at * band..((at + 1) * band).min(n);
        walk_band(
            (rows, cols),
            (&row_columns, &col_columns),
            m,
            k,
            offsets,
            range,
        )
    };
    #[cfg(all(feature = "parallel", not(loom)))]
    let parts: Vec<Result<Vec<(u64, Entry)>, CapacityError>> =
        crate::par::for_each_chunk(threads, bands, task);
    #[cfg(not(all(feature = "parallel", not(loom))))]
    let parts: Vec<Result<Vec<(u64, Entry)>, CapacityError>> = {
        let _ = threads;
        (0..bands).map(task).collect()
    };
    let mut out = reserved_vec(n as u128)?;
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

fn walk_band(
    (rows, cols): (&Quantized, &Quantized),
    (row_columns, col_columns): (&Columns, &Columns),
    m: NonZeroUsize,
    k: NonZeroUsize,
    offsets: &[i64],
    band: core::ops::Range<usize>,
) -> Result<Vec<(u64, Entry)>, CapacityError> {
    let mu = m.get();
    let n = rows.windows.len() as i64;
    let columns = cols.windows.len() as i64;
    let mut nearest: Vec<Nearest> = Vec::new();
    nearest
        .try_reserve_exact(band.len())
        .map_err(|_| CapacityError)?;
    nearest.extend(
        band.clone()
            .map(|r| Nearest::with_k(&rows.windows[r], mu, k, offsets.len())),
    );
    let mut floor: Vec<f64> = Vec::new();
    floor
        .try_reserve_exact(band.len())
        .map_err(|_| CapacityError)?;
    floor.resize(band.len(), f64::NEG_INFINITY);
    let (a, b) = (&rows.ticks, &cols.ticks);
    let pairs = CrossTerms::new(&a[..], 0, &b[..], 0, mu);
    let (sum_a, inv_a, kind_a) = (&row_columns.sum, &row_columns.inv, &row_columns.kind);
    let (sum_b, inv_b, kind_b) = (&col_columns.sum, &col_columns.inv, &col_columns.kind);
    let mut qts = [0i64; BLOCK];
    let mut estimates = [0f64; BLOCK];
    for &offset in offsets {
        let lo = (band.start as i64).max(offset.saturating_neg());
        let hi = (band.end as i64).min(n).min(columns.saturating_sub(offset));
        if lo >= hi {
            continue;
        }
        let (lo, hi) = (lo as usize, hi as usize);
        let first = (lo as i64 + offset) as usize;
        let mut qt = pairs.at(lo, first);
        let mut start = lo;
        while start < hi {
            let len = (hi - start).min(BLOCK);
            let p0 = (start as i64 + offset) as usize;
            pairs.force_resume(start, (start + 1, p0 + 1), core::slice::from_mut(&mut qt));
            // The dot product of each pair of the block, then the next block's first.
            let steps = if start + len < hi { len } else { len - 1 };
            let (lead_a, lag_a) = (&a[start..start + steps], &a[start + mu..start + mu + steps]);
            let (lead_b, lag_b) = (&b[p0..p0 + steps], &b[p0 + mu..p0 + mu + steps]);
            for i in 0..steps {
                qts[i] = qt;
                qt = qt
                    .wrapping_sub(lead_a[i].wrapping_mul(lead_b[i]))
                    .wrapping_add(lag_a[i].wrapping_mul(lag_b[i]));
            }
            if steps < len {
                qts[len - 1] = qt;
            }
            // Every pair's estimate; a pair that is not valid with valid asks for the slow path.
            let (sa, sb) = (&sum_a[start..start + len], &sum_b[p0..p0 + len]);
            let (ia, ib) = (&inv_a[start..start + len], &inv_b[p0..p0 + len]);
            let (ka, kb) = (&kind_a[start..start + len], &kind_b[p0..p0 + len]);
            for i in 0..len {
                let cross = centered_cross_fast(qts[i], mu as i64, sa[i], sb[i]);
                let estimate = rank_estimate(cross, ia[i], ib[i]).clamp(-1.0, 1.0);
                estimates[i] = if ka[i] & kb[i] == VALID {
                    estimate
                } else {
                    f64::INFINITY
                };
            }
            let floors = &mut floor[start - band.start..start - band.start + len];
            for i in 0..len {
                if estimates[i] < floors[i] {
                    continue;
                }
                let (r, p) = (start + i, p0 + i);
                let reduction = &mut nearest[r - band.start];
                if ka[i] & kb[i] == VALID {
                    let cross = centered_cross_fast(qts[i], mu as i64, sa[i], sb[i]);
                    offer(reduction, cross, cols, p);
                } else if ka[i] != NON_FINITE && kb[i] != NON_FINITE {
                    offer(reduction, 0, cols, p);
                }
                floors[i] = reduction.admission_floor();
            }
            start += len;
        }
    }
    let mut out = reserved_vec(nearest.len() as u128)?;
    for (at, reduction) in nearest.into_iter().enumerate() {
        reduction.push_kept((band.start + at) as u64, k, &mut out);
    }
    Ok(out)
}

fn offer(reduction: &mut Nearest, cross: i64, cols: &Quantized, at: usize) {
    let index = SampleIndex::new(at as u64).expect("a window start is below the sentinel");
    reduction.offer(cross, &cols.windows[at], index);
}
