//! The cold route as the streams run it, one row at a time. A stream withholds each tick the
//! outlier rule flags as it arrives ([`withholds`]), so its hot passes see a gap there and keep
//! their narrow arithmetic; each window holding a withheld tick, and each window whose variance
//! numerator passes `2⁶³`, is *spilled*: a bit in a bitmap by row, with its exact statistics. A
//! row then ranks from the exact ticks every pair the hot passes cannot: a spilled row against
//! every neighbor, any other row against its spilled neighbors. A pair's dot product continues
//! from the previous row along its diagonal when that row ranked it too, or starts from `m`
//! multiplies. The candidates fold into the hot passes' kept neighbors by the shared exact order.

use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::{
    Arith, ColdWindow, OutlierScan, Pick, Spilled, cold_candidate, cold_nearer, dot, is_outlier,
    merge_nearest,
};
use crate::capacity::reserve_for;
use crate::entry::Entry;
use crate::nearest::{Candidate, ExactDistance, clearly_nearer, insert_topk, winner_entry};
use crate::right::Right;
use crate::route;
use crate::stats::{Window, WindowKind};
use crate::wide::Wide;

/// Whether a stream withholds a present `tick` from its hot passes: it is an outlier by
/// `is_outlier` against `level`, the newest present tick kept, with `run` the ticks
/// withheld since.
pub(crate) fn withholds(m: usize, tick: i64, level: i64, run: impl FnOnce() -> usize) -> bool {
    route::spill_engaged(is_outlier(m, tick, level, run))
}

/// The ticks a stream fed `ticks` (with presence `had`) from the first one withholds by
/// [`withholds`], as indices, and the rule's state after them: what a restored stream withholds
/// of the ticks it holds, and the level and run it continues from.
pub(crate) fn withheld_indices(m: usize, ticks: &[i64], had: &[bool]) -> (Vec<usize>, OutlierScan) {
    let mut scan = OutlierScan::default();
    let withheld = (0..ticks.len())
        .filter(|&at| had[at] && scan.step(m, ticks[at], route::spill_engaged))
        .collect();
    (withheld, scan)
}

/// Sets a stream's held samples to what its hot arithmetic holds once it withholds the ticks
/// at `withheld` (ascending, each after a present tick): the tick before each, as a gap.
pub(crate) fn hold(ticks: &mut [i64], had: &mut [bool], withheld: &[usize]) {
    for &at in withheld {
        ticks[at] = ticks[at - 1];
        had[at] = false;
    }
}

/// [`hold`]'s inverse, given each withheld index and its exact tick.
pub(crate) fn unhold(ticks: &mut [i64], had: &mut [bool], withheld: &[(usize, i64)]) {
    for &(at, tick) in withheld {
        ticks[at] = tick;
        had[at] = true;
    }
}

/// The caller's flag a window `canonical` was stored under, given `exact`, the window its ticks
/// make by the default rule: `None` when the two agree.
pub(crate) fn flag_of(canonical: Window, exact: Window) -> Option<bool> {
    if canonical == exact {
        None
    } else {
        Some(canonical == exact.under(Some(true)))
    }
}

/// Exact arithmetic a stream's cold route can run in.
pub(crate) trait Tier: Arith {
    /// The exact sum of a window the hot passes hold, from its wrapped sum or its ticks.
    fn exact_sum(hot_sum: i64, ticks: &[i64]) -> Self;
}

impl Tier for i128 {
    /// Below [`WIDE_FROM`] every window's sum is below `2⁶³`, so the wrapped sum is exact.
    fn exact_sum(hot_sum: i64, _ticks: &[i64]) -> Self {
        i128::from(hot_sum)
    }
}

impl Tier for Wide<4> {
    fn exact_sum(_hot_sum: i64, ticks: &[i64]) -> Self {
        ticks
            .iter()
            .fold(Self::ZERO, |sum, &tick| sum.add(Self::from_i64(tick)))
    }
}

/// The window length from which a stream's cold route runs in 256 bits: ticks stay within
/// `2⁵³`, so below it `m·R < 2⁶⁴` (the `i128` tier) whatever the data.
const WIDE_FROM: usize = 1 << 10;

/// The window length from which a stream has no cold route (`m·R` may reach `2⁸⁶`).
const UNAVAILABLE_FROM: u64 = 1 << 32;

/// Whether a stream of windows of `m` has a cold route.
pub(crate) fn available(m: usize) -> bool {
    (m as u64) < UNAVAILABLE_FROM
}

/// One row as the cold route sees it.
pub(crate) struct RowView<'a, H> {
    pub(crate) row: u64,
    /// The exact ticks the stream holds: position `p` at `exact[p − base]`.
    pub(crate) exact: &'a [i64],
    pub(crate) base: u64,
    /// The oldest window still a neighbor of this row.
    pub(crate) oldest: u64,
    /// The row's window as the hot passes hold it (non-finite when spilled).
    pub(crate) hot_window: Window,
    /// The caller's flag for the row's window.
    pub(crate) flag: Option<bool>,
    /// Whether the window holds a withheld tick.
    pub(crate) holds_withheld: bool,
    /// Whether position `p` had no sample (a gap, not a withheld tick).
    pub(crate) gap: &'a dyn Fn(u64) -> bool,
    /// A neighbor window as the hot passes hold it.
    pub(crate) hot: H,
}

impl<H> RowView<'_, H> {
    fn ticks(&self, window: u64, m: usize) -> &[i64] {
        let start = (window - self.base) as usize;
        &self.exact[start..start + m]
    }
}

/// What a row's cold pass gives back to the stream.
pub(crate) struct RowResult {
    /// The row's nearest neighbor.
    pub(crate) entry: Entry,
    /// It with its exact correlation, `None` when there is none.
    pub(crate) exact: Option<ExactDistance>,
    /// Whether the row's window is spilled.
    pub(crate) spilled: bool,
    /// Whether any pair was ranked cold.
    pub(crate) ranked: bool,
}

/// A stream's cold-route state in one tier.
pub(crate) struct Rows<A> {
    m: usize,
    exclusion: usize,
    k: NonZeroUsize,
    /// Rows per slot cycle of `windows` and `qt` when the stream holds only its latest windows;
    /// `None` when every window stays (slots indexed by row, grown as needed).
    held: Option<usize>,
    /// Spilled windows' exact statistics by slot, each tagged with its row.
    windows: Vec<(u64, ColdWindow<A>)>,
    /// One bit per window slot: spilled. Bounded: a power of two of bits, at least `held`.
    bits: Vec<u64>,
    /// By lag: the last row that ranked its pair at that lag cold, and the pair's dot product.
    qt: Vec<(u64, A)>,
    kept: Vec<Candidate<Spilled<A>>>,
    merged: Vec<Entry>,
}

impl<A: Tier> Rows<A> {
    /// `held` windows for a stream that keeps only its latest (every buffer reserved here), or
    /// `None` for one that keeps every window.
    fn new(m: usize, k: NonZeroUsize, held: Option<usize>) -> Self {
        let exclusion = crate::exclusion::zone(m);
        let (slots, words, neighbors) = match held {
            Some(held) => (
                held,
                held.next_power_of_two().max(64) / 64,
                held.saturating_sub(exclusion + 1),
            ),
            None => (0, 0, 0),
        };
        let kept = reserve_for(k.get(), neighbors);
        Self {
            m,
            exclusion,
            k,
            held,
            windows: vec![(u64::MAX, ColdWindow::MISSING); slots],
            bits: vec![0; words],
            qt: vec![(u64::MAX, A::ZERO); slots],
            kept: Vec::with_capacity(kept),
            merged: Vec::with_capacity(kept),
        }
    }

    fn window_slot(&self, row: u64) -> usize {
        match self.held {
            Some(held) => (row % held as u64) as usize,
            None => row as usize,
        }
    }

    fn bit_slot(&self, row: u64) -> usize {
        match self.held {
            Some(_) => (row & (self.bits.len() as u64 * 64 - 1)) as usize,
            None => row as usize,
        }
    }

    fn clear(&mut self) {
        self.bits.fill(0);
    }

    fn set_bit(&mut self, row: u64, spilled: bool) {
        let slot = self.bit_slot(row);
        if slot / 64 >= self.bits.len() {
            if !spilled {
                return;
            }
            self.bits.resize(slot / 64 + 1, 0);
        }
        let (word, bit) = (&mut self.bits[slot / 64], 1u64 << (slot % 64));
        if spilled {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }

    fn spilled_window(&self, row: u64) -> Option<ColdWindow<A>> {
        let &(tag, window) = self.windows.get(self.window_slot(row))?;
        (tag == row).then_some(window)
    }

    fn store(&mut self, row: u64, window: ColdWindow<A>) {
        let slot = self.window_slot(row);
        if slot >= self.windows.len() {
            self.windows
                .resize(slot + 1, (u64::MAX, ColdWindow::MISSING));
        }
        self.windows[slot] = (row, window);
    }

    /// Row `view.row`'s window as the cold route sees it, whether it is present, and whether it is
    /// spilled (a present window the hot arithmetic holds as non-finite: it holds a withheld tick
    /// or its variance numerator passes `2⁶³`), with its bit and statistics stored.
    fn classify<H>(&mut self, view: &RowView<'_, H>) -> (ColdWindow<A>, bool, bool) {
        let (m, row) = (self.m, view.row);
        let row_ticks = view.ticks(row, m);
        let non_finite = matches!(view.hot_window.kind, WindowKind::NonFinite);
        let present = !non_finite || !(row..row + m as u64).any(|p| (view.gap)(p));
        let spilled = present && non_finite;
        let window = if spilled {
            ColdWindow::of_ticks(row_ticks, view.flag)
        } else {
            ColdWindow::of_hot(
                &view.hot_window,
                A::exact_sum(view.hot_window.sum, row_ticks),
            )
        };
        self.set_bit(row, spilled);
        if spilled {
            self.store(row, window);
        }
        debug_assert!(!spilled || view.holds_withheld || window.too_wide());
        (window, present, spilled)
    }

    /// Classifies row `view.row` (spilled or not, its bit and statistics), ranks every pair the
    /// hot passes cannot, and folds the result into `hot`, the hot passes' kept neighbors
    /// (nearest first; `hot_exact` the first with its correlation). With `k` above one the merged
    /// neighbors replace `hot`'s contents.
    fn rank<H: Fn(u64) -> Window>(
        &mut self,
        view: &RowView<'_, H>,
        hot: &mut Vec<Entry>,
        hot_exact: Option<ExactDistance>,
        mut right: Option<&mut Right>,
    ) -> RowResult {
        let row = view.row;
        let (window, present, spilled) = self.classify(view);
        let mut ranked = RowResult {
            entry: hot.first().copied().unwrap_or(Entry::NONE),
            exact: hot_exact,
            spilled,
            ranked: false,
        };
        let Some(last) = row.checked_sub(self.exclusion as u64 + 1) else {
            return ranked;
        };
        if !present || view.oldest > last {
            return ranked;
        }
        self.kept.clear();
        if spilled {
            for neighbor in view.oldest..=last {
                self.offer(view, &window, neighbor, &mut right);
            }
            ranked.ranked = true;
        } else {
            let mut at = view.oldest;
            while let Some(neighbor) = self.next_spilled(at, last) {
                self.offer(view, &window, neighbor, &mut right);
                ranked.ranked = true;
                at = neighbor + 1;
            }
        }
        if !self.kept.is_empty() {
            self.merge(view, &window, hot, &mut ranked);
        }
        ranked
    }

    /// The first spilled row in `from..=last`.
    fn next_spilled(&self, from: u64, last: u64) -> Option<u64> {
        let mut found = None;
        let mut at = from;
        while at <= last && found.is_none() {
            let slot = self.bit_slot(at);
            let word = *self.bits.get(slot / 64)?;
            let span = (64 - (slot % 64) as u64).min(last - at + 1);
            let mask = if span == 64 {
                u64::MAX
            } else {
                (1u64 << span) - 1
            };
            let hits = (word >> (slot % 64)) & mask;
            if hits != 0 {
                found = Some(at + u64::from(hits.trailing_zeros()));
            }
            at += span;
        }
        found
    }

    /// Ranks the pair of row `view.row` (its window `window`) and `neighbor`; a stream that looks
    /// right also gets the pair for both windows.
    fn offer<H: Fn(u64) -> Window>(
        &mut self,
        view: &RowView<'_, H>,
        window: &ColdWindow<A>,
        neighbor: u64,
        right: &mut Option<&mut Right>,
    ) {
        let (m, row) = (self.m, view.row);
        let lag = (row - neighbor) as usize;
        let neighbor_window = self.neighbor_window(view, neighbor);
        if lag >= self.qt.len() {
            self.qt.resize(lag + 1, (u64::MAX, A::ZERO));
        }
        let (tagged, previous) = self.qt[lag];
        let qt = if tagged.checked_add(1) == Some(row) && neighbor > view.base {
            let at = |p: u64| view.exact[(p - view.base) as usize];
            let (last, before) = (m as u64 - 1, 1);
            previous
                .add(A::product(at(row + last), at(neighbor + last)))
                .sub(A::product(at(row - before), at(neighbor - before)))
        } else {
            dot(view.ticks(row, m), view.ticks(neighbor, m))
        };
        self.qt[lag] = (row, qt);
        let k = self.k.get();
        let full = (self.kept.len() == k).then(|| self.kept[k - 1].rho_hat);
        let skip = |rho_hat| full.is_some_and(|worst| clearly_nearer(worst, rho_hat));
        let pair = (window, &neighbor_window);
        let looks_right = right.is_some();
        let Some(candidate) = cold_candidate(m, pair, qt, neighbor, |rho_hat| {
            !looks_right && skip(rho_hat)
        }) else {
            return;
        };
        if let Some(right) = right.as_deref_mut()
            && let Some(column) = cold_candidate(m, (&neighbor_window, window), qt, row, |_| false)
        {
            right.offer_cold(row, neighbor, candidate, column);
        }
        if !skip(candidate.rho_hat) {
            insert_topk(&mut self.kept, k, candidate);
        }
    }

    fn neighbor_window<H: Fn(u64) -> Window>(
        &self,
        view: &RowView<'_, H>,
        neighbor: u64,
    ) -> ColdWindow<A> {
        self.spilled_window(neighbor).unwrap_or_else(|| {
            let hot = (view.hot)(neighbor);
            ColdWindow::of_hot(&hot, A::exact_sum(hot.sum, view.ticks(neighbor, self.m)))
        })
    }

    /// Merges the cold candidates (nearest first) with `hot`'s entries (nearest first, the two
    /// sets disjoint), keeping the `k` nearest.
    fn merge<H: Fn(u64) -> Window>(
        &mut self,
        view: &RowView<'_, H>,
        window: &ColdWindow<A>,
        hot: &mut Vec<Entry>,
        ranked: &mut RowResult,
    ) {
        let (m, row) = (self.m, view.row);
        let mut merged = core::mem::take(&mut self.merged);
        merged.clear();
        let exact_of = |entry: Entry| {
            let index = entry.index().get()?;
            let neighbor = self.neighbor_window(view, index);
            let qt = dot(view.ticks(row, m), view.ticks(index, m));
            cold_candidate(m, (window, &neighbor), qt, index, |_| false)
        };
        let cold_wins = |cold, at: usize| cold_nearer(cold, hot[at], || exact_of(hot[at]));
        let mut first = None;
        merge_nearest(
            hot.len(),
            &self.kept,
            self.k.get(),
            cold_wins,
            |pick| match pick {
                Pick::Cold(at) => {
                    first.get_or_insert(Err(self.kept[at]));
                    merged.push(winner_entry(m, self.kept[at]));
                }
                Pick::Hot(at) => {
                    first.get_or_insert(Ok(()));
                    merged.push(hot[at]);
                }
            },
        );
        if let Some(Err(cold)) = first {
            ranked.exact = Some(ExactDistance::held(merged[0], cold.correlation.held(), m));
        }
        ranked.entry = merged.first().copied().unwrap_or(Entry::NONE);
        core::mem::swap(hot, &mut merged);
        self.merged = merged;
    }
}

/// A stream's cold route in the tier its window length needs.
pub(crate) enum Cold {
    Narrow(Rows<i128>),
    Wide(Rows<Wide<4>>),
    /// Windows too long for the 256-bit tier: no tick is withheld, no row ranked cold.
    Unavailable,
}

impl Cold {
    /// For a stream of windows of `m` keeping `k` neighbors: `held` windows when it keeps only
    /// its latest (every buffer reserved here), or `None` when it keeps every window.
    pub(crate) fn new(m: NonZeroUsize, k: NonZeroUsize, held: Option<usize>) -> Self {
        let m = m.get();
        if m < WIDE_FROM {
            Self::Narrow(Rows::new(m, k, held))
        } else if available(m) {
            Self::Wide(Rows::new(m, k, held))
        } else {
            Self::Unavailable
        }
    }

    pub(crate) fn available(&self) -> bool {
        !matches!(self, Self::Unavailable)
    }

    /// Forgets which windows are spilled: every row from here on is classified afresh, and every
    /// earlier one is unspilled.
    pub(crate) fn clear(&mut self) {
        match self {
            Self::Narrow(rows) => rows.clear(),
            Self::Wide(rows) => rows.clear(),
            Self::Unavailable => {}
        }
    }

    /// [`Rows::rank`] in the stream's tier; `None` when it has no cold route.
    pub(crate) fn rank<H: Fn(u64) -> Window>(
        &mut self,
        view: &RowView<'_, H>,
        hot: &mut Vec<Entry>,
        hot_exact: Option<ExactDistance>,
        right: Option<&mut Right>,
    ) -> Option<RowResult> {
        match self {
            Self::Narrow(rows) => Some(rows.rank(view, hot, hot_exact, right)),
            Self::Wide(rows) => Some(rows.rank(view, hot, hot_exact, right)),
            Self::Unavailable => None,
        }
    }

    /// [`Rows::classify`] alone, for a restored stream's live windows: whether the row's window
    /// is spilled.
    pub(crate) fn classify<H>(&mut self, view: &RowView<'_, H>) -> bool {
        match self {
            Self::Narrow(rows) => rows.classify(view).2,
            Self::Wide(rows) => rows.classify(view).2,
            Self::Unavailable => false,
        }
    }

    /// Whether row `row`'s window is spilled, for a checkpoint.
    pub(crate) fn spilled(&self, row: u64) -> bool {
        match self {
            Self::Narrow(rows) => rows.spilled_window(row).is_some(),
            Self::Wide(rows) => rows.spilled_window(row).is_some(),
            Self::Unavailable => false,
        }
    }

    /// Whether row `row`'s window is spilled and constant, for a checkpoint.
    pub(crate) fn spilled_constant(&self, row: u64) -> bool {
        match self {
            Self::Narrow(rows) => rows.spilled_window(row).is_some_and(|w| w.is_constant()),
            Self::Wide(rows) => rows.spilled_window(row).is_some_and(|w| w.is_constant()),
            Self::Unavailable => false,
        }
    }

    /// Whether row `row`'s window is constant by its flag while its variance numerator passes
    /// `2⁶³`: what a checkpoint's arrays cannot hold.
    pub(crate) fn constant_too_wide(&self, row: u64) -> bool {
        let both = |constant: bool, too_wide: bool| constant && too_wide;
        match self {
            Self::Narrow(rows) => rows
                .spilled_window(row)
                .is_some_and(|w| both(w.is_constant(), w.too_wide())),
            Self::Wide(rows) => rows
                .spilled_window(row)
                .is_some_and(|w| both(w.is_constant(), w.too_wide())),
            Self::Unavailable => false,
        }
    }
}
