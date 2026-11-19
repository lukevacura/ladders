//! What a top-`k` call keeps and how its flat lists are laid out, whatever the metric: how many
//! windows can be a neighbor, the entries each side's list holds, and the lists regrouped per row.
//! A metric supplies only which windows are finite and which of its entries hold a neighbor.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// `finite_from[i]`: the finite windows at `i` or later, of `len` windows of which `finite` says
/// which are.
pub(super) fn finite_from(len: usize, finite: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut finite_from = alloc::vec![0usize; len + 1];
    for i in (0..len).rev() {
        finite_from[i] = finite_from[i + 1] + usize::from(finite(i));
    }
    finite_from
}

/// What one side of a top-`k` call keeps: the neighbors in all, summed over every finite window
/// of `min(k, finite windows that can reach it)`, and the rows that keep none. Each is the exact
/// bound on that side's list, which holds one entry per kept neighbor and one empty entry per
/// row with none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SideCount {
    pub(super) kept: u128,
    empty_rows: u128,
}

impl SideCount {
    /// The entries of the side's flat list.
    pub(super) fn list_len(self) -> u128 {
        self.kept + self.empty_rows
    }
}

/// The left and the right side's counts for `len` windows at `k` and `exclusion`, summed without
/// overflow. A window's left neighbors are the finite windows at or before `i - exclusion - 1`,
/// its right neighbors the finite ones from `i + exclusion + 1`.
pub(super) fn side_counts(
    len: usize,
    finite: impl Fn(usize) -> bool,
    k: NonZeroUsize,
    exclusion: usize,
) -> (SideCount, SideCount) {
    let finite_from = finite_from(len, &finite);
    let total = finite_from[0];
    let (mut left, mut right) = (
        SideCount {
            kept: 0,
            empty_rows: 0,
        },
        SideCount {
            kept: 0,
            empty_rows: 0,
        },
    );
    for i in 0..len {
        let reaches = if finite(i) {
            [
                total - finite_from[i.saturating_sub(exclusion)],
                finite_from[(i + exclusion + 1).min(len)],
            ]
        } else {
            [0, 0]
        };
        for (side, reach) in [&mut left, &mut right].into_iter().zip(reaches) {
            let kept = k.get().min(reach);
            side.kept += kept as u128;
            side.empty_rows += u128::from(kept == 0);
        }
    }
    (left, right)
}

/// The entries of a join's flat list at `k` (already clamped to `b`'s finite windows): `k` per
/// finite `a` window, one empty entry for every other. Summed without overflow.
pub(super) fn join_list_len(windows_a: usize, finite_a: usize, k: NonZeroUsize) -> u128 {
    (windows_a - finite_a) as u128 + finite_a as u128 * k.get() as u128
}

/// `flat`, a run of kept entries per row, as one list per row with the empty entries (those
/// `holds_neighbor` refuses) dropped.
pub(crate) fn group_by_row<T>(
    flat: Vec<(u64, T)>,
    k: NonZeroUsize,
    rows: usize,
    holds_neighbor: impl Fn(&T) -> bool,
) -> Vec<(u64, Vec<T>)> {
    let mut grouped: Vec<(u64, Vec<T>)> = Vec::with_capacity(rows);
    for (row, entry) in flat {
        if grouped.last().is_none_or(|(last, _)| *last != row) {
            grouped.push((row, Vec::with_capacity(k.get().min(4))));
        }
        if let Some((_, kept)) = grouped.last_mut()
            && holds_neighbor(&entry)
        {
            kept.push(entry);
        }
    }
    grouped
}
