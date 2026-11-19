//! The exact profiles restricted to some diagonals: the pairs `(i, j)` with `j - i` in a given
//! set, and no others. A reference for the anytime profiles, which visit a subset of the diagonals
//! of the distance matrix; every pair is computed directly, as everywhere in this module.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use ethnum::I256;

use super::{
    Candidate, NonNormEntry, NonNormSides, TwoSidedTopk, Window, candidate, comparable, cross_term,
    cross_term_pair, fits_narrow_bound, nearest_k, nonnorm_entry, nonnorm_gap_windows,
    squared_distance_pair, window_stats,
};
use crate::entry::Entry;

/// `diagonals` as a set: sorted, each once.
fn as_set(diagonals: &[i64]) -> Vec<i64> {
    let mut set = diagonals.to_vec();
    set.sort_unstable();
    set.dedup();
    set
}

/// The window `sign * diagonal` away from `row`, when `diagonal` lies past the exclusion zone
/// (`e = ceil(m/4)`, so `diagonal > e`) and the window exists among `windows`.
fn partner(row: usize, sign: i64, diagonal: i64, m: usize, windows: usize) -> Option<usize> {
    if diagonal <= crate::exclusion::zone(m) as i64 {
        return None;
    }
    let at = (row as i64).checked_add(sign.checked_mul(diagonal)?)?;
    usize::try_from(at).ok().filter(|&j| j < windows)
}

/// The z-normalized two-sided top-`k` profile over the self-join pairs whose diagonal
/// `|j - i|` is in `diagonals` (a number inside the exclusion zone or beyond the series is no
/// diagonal and is ignored): [`naive_two_sided_topk`](super::naive_two_sided_topk)'s rows over
/// those pairs alone. `left` and `right` are each side's one nearest, `nearest` the `k` nearest
/// over both.
pub fn naive_two_sided_topk_on(
    samples: &[Option<i64>],
    m: usize,
    k: NonZeroUsize,
    diagonals: &[i64],
) -> Vec<(u64, TwoSidedTopk)> {
    if m == 0 || m > samples.len() {
        return Vec::new();
    }
    let diagonals = as_set(diagonals);
    let windows: Vec<Window> = (0..=samples.len() - m)
        .map(|start| window_stats(samples, start, m))
        .collect();
    let one = NonZeroUsize::MIN;
    (0..windows.len())
        .map(|i| {
            let sides = comparable(windows[i]).map(|row| {
                let pairs = |sign: i64| -> Vec<Candidate> {
                    diagonals
                        .iter()
                        .filter_map(|&d| partner(i, sign, d, m, windows.len()))
                        .filter_map(|j| {
                            let neighbor = comparable(windows[j])?;
                            let qt = cross_term(samples, i, j, m);
                            Some(candidate(m, row, neighbor, qt, j as u64))
                        })
                        .collect()
                };
                (pairs(-1), pairs(1))
            });
            let first = |candidates: Vec<Candidate>| {
                nearest_k(m, candidates, one)
                    .first()
                    .copied()
                    .unwrap_or(Entry::NONE)
            };
            let row = match sides {
                None => TwoSidedTopk {
                    left: Entry::NONE,
                    right: Entry::NONE,
                    nearest: Vec::new(),
                },
                Some((left, right)) => TwoSidedTopk {
                    left: first(left.clone()),
                    right: first(right.clone()),
                    nearest: nearest_k(m, left.into_iter().chain(right).collect(), k),
                },
            };
            (i as u64, row)
        })
        .collect()
}

/// The non-normalized two-sided profile over the self-join pairs whose diagonal is in
/// `diagonals` (as for [`naive_two_sided_topk_on`]):
/// [`naive_two_sided_nonnorm`](super::naive_two_sided_nonnorm)'s rows over those pairs alone.
pub fn naive_two_sided_nonnorm_on(
    samples: &[Option<i64>],
    m: usize,
    diagonals: &[i64],
) -> Vec<(u64, NonNormSides)> {
    if m == 0 || m > samples.len() {
        return Vec::new();
    }
    let diagonals = as_set(diagonals);
    let has_gap = nonnorm_gap_windows(samples, m);
    let narrow = fits_narrow_bound(samples, m);
    (0..has_gap.len())
        .map(|i| {
            let nearest_over = |sign: i64| -> Option<(u64, I256)> {
                if has_gap[i] {
                    return None;
                }
                let mut best: Option<(u64, I256)> = None;
                for &d in &diagonals {
                    let Some(j) = partner(i, sign, d, m, has_gap.len()).filter(|&j| !has_gap[j])
                    else {
                        continue;
                    };
                    let distance = squared_distance_pair(samples, i, samples, j, m);
                    let wins = best.is_none_or(|(at, current)| {
                        distance < current || (distance == current && (j as u64) < at)
                    });
                    if wins {
                        best = Some((j as u64, distance));
                    }
                }
                best
            };
            let (left, right) = (nearest_over(-1), nearest_over(1));
            let nearest = match (left, right) {
                (Some(l), Some(r)) => Some(if r.1 < l.1 { r } else { l }),
                (l, r) => l.or(r),
            };
            let sides = NonNormSides {
                left: nonnorm_entry(left, narrow),
                right: nonnorm_entry(right, narrow),
                nearest: nonnorm_entry(nearest, narrow),
            };
            (i as u64, sides)
        })
        .collect()
}

/// The `b` window `diagonal` past `row` of `a`, when it exists among `windows`.
fn column_of(row: usize, diagonal: i64, windows: usize) -> Option<usize> {
    let at = (row as i64).checked_add(diagonal)?;
    usize::try_from(at).ok().filter(|&j| j < windows)
}

/// The windows of `samples` at `m`, none when it is shorter.
fn windows_of(samples: &[Option<i64>], m: usize) -> Vec<Window> {
    if m > samples.len() {
        return Vec::new();
    }
    (0..=samples.len() - m)
        .map(|start| window_stats(samples, start, m))
        .collect()
}

/// The z-normalized top-`k` join over the pairs whose diagonal `j - i` (`j` in `b`, `i` in `a`)
/// is in `diagonals` (a number beyond both series is no diagonal and is ignored):
/// [`naive_ab_join_topk`](super::naive_ab_join_topk)'s rows over those pairs alone.
pub fn naive_ab_join_topk_on(
    a: &[Option<i64>],
    b: &[Option<i64>],
    m: usize,
    k: NonZeroUsize,
    diagonals: &[i64],
) -> Vec<(u64, Vec<Entry>)> {
    if m == 0 || m > a.len() {
        return Vec::new();
    }
    let diagonals = as_set(diagonals);
    let (a_windows, b_windows) = (windows_of(a, m), windows_of(b, m));
    a_windows
        .iter()
        .enumerate()
        .map(|(i, &a_window)| {
            let entries = comparable(a_window).map_or_else(Vec::new, |row| {
                let candidates = diagonals
                    .iter()
                    .filter_map(|&d| column_of(i, d, b_windows.len()))
                    .filter_map(|j| {
                        let neighbor = comparable(b_windows[j])?;
                        let qt = cross_term_pair(a, i, b, j, m);
                        Some(candidate(m, row, neighbor, qt, j as u64))
                    })
                    .collect();
                nearest_k(m, candidates, k)
            });
            (i as u64, entries)
        })
        .collect()
}

/// The non-normalized join over the pairs whose diagonal is in `diagonals` (as for
/// [`naive_ab_join_topk_on`]): [`naive_ab_join_nonnorm`](super::naive_ab_join_nonnorm)'s rows over
/// those pairs alone.
pub fn naive_ab_join_nonnorm_on(
    a: &[Option<i64>],
    b: &[Option<i64>],
    m: usize,
    diagonals: &[i64],
) -> Vec<(u64, NonNormEntry)> {
    if m == 0 || m > a.len() {
        return Vec::new();
    }
    let diagonals = as_set(diagonals);
    let a_gaps = nonnorm_gap_windows(a, m);
    let b_gaps = if m <= b.len() {
        nonnorm_gap_windows(b, m)
    } else {
        Vec::new()
    };
    let both: Vec<Option<i64>> = a.iter().chain(b).copied().collect();
    let narrow = fits_narrow_bound(&both, m);
    a_gaps
        .iter()
        .enumerate()
        .map(|(i, &gap)| {
            let mut best: Option<(u64, I256)> = None;
            if !gap {
                for &d in &diagonals {
                    let Some(j) = column_of(i, d, b_gaps.len()).filter(|&j| !b_gaps[j]) else {
                        continue;
                    };
                    let distance = squared_distance_pair(a, i, b, j, m);
                    let wins = best.is_none_or(|(at, current)| {
                        distance < current || (distance == current && (j as u64) < at)
                    });
                    if wins {
                        best = Some((j as u64, distance));
                    }
                }
            }
            (i as u64, nonnorm_entry(best, narrow))
        })
        .collect()
}
