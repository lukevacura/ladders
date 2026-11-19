//! The right side of a non-normalized stream's windows, as [`crate::right`] keeps it for the
//! z-normalized ones: each held window's `k` nearest later neighbors and its `k` nearest over
//! both sides, by exact squared distance, ties to the smaller index. A pair is ranked once by
//! the row's pass; the row's reducer is wrapped so that each pair it is offered also goes to
//! the earlier window's lists.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::index::SampleIndex;
use crate::kernel::Reducer;
use crate::nonnorm::{LaneReducer, NonNormNeighbor};
use crate::right::Lists;

/// A pair's squared distance and the neighbor's window start.
type Pair = (u128, u64);

/// A held window's right list and its two-sided list.
pub(crate) type Column = (Vec<Pair>, Vec<Pair>);

const EMPTY: Pair = (u128::MAX, u64::MAX);

fn neighbor((squared, index): Pair) -> NonNormNeighbor {
    NonNormNeighbor {
        squared_ticks: squared,
        index: SampleIndex::new(index).expect("a window start is below NONE's sentinel"),
    }
}

/// The right and two-sided lists of every window a non-normalized stream holds.
pub(crate) struct RightSquared {
    right: Lists<Pair>,
    both: Lists<Pair>,
    /// The oldest window held, and one past the newest.
    first: u64,
    next: u64,
}

impl RightSquared {
    /// For a stream that keeps every window.
    pub(crate) fn growing(k: NonZeroUsize) -> Self {
        Self {
            right: Lists::growing(k.get(), EMPTY),
            both: Lists::growing(k.get(), EMPTY),
            first: 0,
            next: 0,
        }
    }

    /// For a stream that holds the latest `windows` windows, of which each can be reached by at
    /// most `reach` later ones; `None` when the lists cannot be allocated.
    pub(crate) fn held(k: NonZeroUsize, windows: usize, reach: usize) -> Option<Self> {
        Some(Self {
            right: Lists::held(k.get(), windows, k.get().min(reach), EMPTY)?,
            both: Lists::held(
                k.get(),
                windows,
                k.get().min(reach.saturating_mul(2)),
                EMPTY,
            )?,
            first: 0,
            next: 0,
        })
    }

    /// The windows held: the oldest to the newest.
    pub(crate) fn held_windows(&self) -> core::ops::Range<u64> {
        self.first..self.next
    }

    /// Adds window `row`, whose pass is next; a stream that holds a fixed number of windows drops
    /// the oldest.
    pub(crate) fn begin_row(&mut self, row: u64) {
        self.right.begin(row);
        self.both.begin(row);
        self.next = row + 1;
        if self.right.windows() > 0 {
            self.first = self.next.saturating_sub(self.right.windows());
        }
    }

    /// Offers the pair of the row `row` and its earlier neighbor `neighbor`, `squared` apart.
    fn offer(&mut self, row: u64, neighbor: u64, squared: u128) {
        let better = |a: &Pair, b: &Pair| a < b;
        let enters = |lists: &Lists<Pair>, window: u64, pair: Pair| {
            lists.last_if_full(window).is_none_or(|last| pair < *last)
        };
        let left = (squared, neighbor);
        if enters(&self.both, row, left) {
            self.both.keep(row, left, better);
        }
        if neighbor >= self.first {
            let later = (squared, row);
            if enters(&self.right, neighbor, later) {
                self.right.keep(neighbor, later, better);
                if enters(&self.both, neighbor, later) {
                    self.both.keep(neighbor, later, better);
                }
            }
        }
    }

    /// The held window's lists, as pairs of squared distance and neighbor.
    pub(crate) fn column(&self, window: u64) -> Column {
        (
            self.right.list(window).to_vec(),
            self.both.list(window).to_vec(),
        )
    }

    /// Sets the lists of the held window `window` (added by [`Self::begin_row`]); `false` when
    /// they do not fit what the stream keeps.
    pub(crate) fn load(
        &mut self,
        window: u64,
        right: &[(u128, u64)],
        both: &[(u128, u64)],
    ) -> bool {
        self.right.fill(window, right) && self.both.fill(window, both)
    }

    /// The window's `k` nearest later neighbors, nearest first.
    pub(crate) fn right_into(&self, window: u64, out: &mut Vec<NonNormNeighbor>) {
        out.clear();
        out.extend(self.right.list(window).iter().copied().map(neighbor));
    }

    /// The window's `k` nearest neighbors over both sides, nearest first.
    pub(crate) fn both_into(&self, window: u64, out: &mut Vec<NonNormNeighbor>) {
        out.clear();
        out.extend(self.both.list(window).iter().copied().map(neighbor));
    }
}

/// A row's reducer that also hands each pair it is offered to the right side.
pub(crate) struct Tee<'a, N> {
    inner: &'a mut N,
    right: &'a mut RightSquared,
    row: u64,
}

impl<'a, N> Tee<'a, N> {
    pub(crate) fn new(inner: &'a mut N, right: &'a mut RightSquared, row: u64) -> Self {
        Self { inner, right, row }
    }
}

impl<N: Reducer<Candidate = (u128, u64)>> Reducer for Tee<'_, N> {
    type Candidate = (u128, u64);
    type Output = ();

    #[inline]
    fn offer(&mut self, (squared, neighbor): (u128, u64)) {
        self.right.offer(self.row, neighbor, squared);
        self.inner.offer((squared, neighbor));
    }

    fn finish(self) {}
}

impl<N: Reducer<Candidate = (u128, u64)>> LaneReducer for Tee<'_, N> {}
