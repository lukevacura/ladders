//! The right side of a stream's windows: for each window the stream holds, its `k` nearest
//! *later* neighbors, kept as the later windows arrive, and its `k` nearest over both sides (the
//! nearest earlier ones it was ranked by when it arrived and the later ones). A stream that looks
//! right owns one [`Right`]; looking right trades on future information, so no stream builds
//! one unless asked.
//!
//! A row's pass offers each pair it ranks to the earlier window's lists as well, through
//! [`ColumnSink`]. A list keeps its items best first under the exact order the batch two-sided
//! profile uses (the estimates decide outside the near-tie band, the exact correlations inside
//! it, ties to the smaller index), so the lists are the exact `k` nearest whatever order the
//! pairs arrive in. A window's floor is the near-tie floor of its `k`-th right neighbor's
//! estimate: the kernel reads only the groups of lanes at or above it, and nothing below it is
//! looked at again.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZeroUsize;

use crate::columns;
use crate::entry::Entry;
use crate::estimate::{Producer, band_floor};
use crate::index::SampleIndex;
use crate::kbest::{self, Champion};
use crate::kernel_neon::GROUP_LANES;
use crate::nearest::{
    Candidate, ExactCorrelation, Held, candidate_of, clearly_nearer, nearer, winner_entry,
};
use crate::row::{ColumnAcc, RankedLane, RowId};
use crate::spill::{Arith, Spilled, cmp_held};
use crate::stats::Window;

/// Where a row's pass offers the pairs it ranked, as the earlier window's side of each pair.
pub(crate) trait ColumnSink {
    /// The floors of columns `first..first + len`, one per lane of a chunk starting there.
    fn floors(&self, first: usize, len: usize) -> &[f64];

    /// Offers a span of at most one span's lanes whose groups at or above a floor are known: bit
    /// `g` of `hits` is set when some lane of group `g` is, and only those groups are read.
    fn offer_groups<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        hits: u64,
        n_of: impl Fn(usize) -> i64,
        variance_of: impl Fn(usize) -> u64,
    );

    /// Offers every lane of `tile` (`tile[lane]` against column `first + lane`).
    fn offer_chunk<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        n_of: impl Fn(usize) -> i64,
        variance_of: impl Fn(usize) -> u64,
    );

    /// Offers one ranked pair; `variance` reads its column's variance.
    fn offer_lane<P: Producer>(
        &mut self,
        column: usize,
        row: RowId,
        lane: RankedLane,
        variance: impl FnOnce() -> u64,
    );
}

impl ColumnSink for ColumnAcc {
    fn floors(&self, first: usize, len: usize) -> &[f64] {
        ColumnAcc::floors(self, first, len)
    }

    fn offer_groups<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        hits: u64,
        n_of: impl Fn(usize) -> i64,
        variance_of: impl Fn(usize) -> u64,
    ) {
        ColumnAcc::offer_groups::<P>(self, first, row, tile, hits, n_of, variance_of);
    }

    fn offer_chunk<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        n_of: impl Fn(usize) -> i64,
        variance_of: impl Fn(usize) -> u64,
    ) {
        ColumnAcc::offer_chunk::<P>(self, first, row, tile, n_of, variance_of);
    }

    fn offer_lane<P: Producer>(
        &mut self,
        column: usize,
        row: RowId,
        lane: RankedLane,
        variance: impl FnOnce() -> u64,
    ) {
        ColumnAcc::offer_lane::<P>(self, column, row, lane, variance);
    }
}

/// Maps a window's absolute index to its slot: the index itself while the stream keeps every
/// window, else the index modulo the windows it holds.
#[derive(Clone, Copy)]
struct Wrap(usize);

impl Wrap {
    fn at(self, window: u64) -> usize {
        if self.0 == 0 {
            window as usize
        } else {
            (window % self.0 as u64) as usize
        }
    }
}

/// One window's slice of a shared array, and how many of its slots are filled.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    len: usize,
    cap: usize,
}

/// Every held window's best-first list of at most `k` items. A stream that keeps every window
/// grows a list in place up to `k`, by doubling its span at the end of the shared array; one
/// that holds a fixed number of windows gives each a span of its own for good.
pub(crate) struct Lists<T> {
    k: usize,
    wrap: Wrap,
    arena: Vec<T>,
    spans: Vec<Span>,
    fill: T,
}

impl<T: Copy> Lists<T> {
    pub(crate) fn growing(k: usize, fill: T) -> Self {
        Self {
            k,
            wrap: Wrap(0),
            arena: Vec::new(),
            spans: Vec::new(),
            fill,
        }
    }

    /// Lists for `windows` held windows of at most `per` items each; `None` when that many cannot
    /// be allocated.
    pub(crate) fn held(k: usize, windows: usize, per: usize, fill: T) -> Option<Self> {
        let slots = windows.checked_mul(per)?;
        let mut arena = Vec::new();
        arena.try_reserve_exact(slots).ok()?;
        arena.resize(slots, fill);
        let mut spans = Vec::new();
        spans.try_reserve_exact(windows).ok()?;
        spans.extend((0..windows).map(|slot| Span {
            start: slot * per,
            len: 0,
            cap: per,
        }));
        Some(Self {
            k,
            wrap: Wrap(windows),
            arena,
            spans,
            fill,
        })
    }

    /// Starts window `window`'s list, empty; a held stream's takes the place of the window that left.
    /// How many windows the lists hold at once; zero when they keep every window.
    pub(crate) fn windows(&self) -> u64 {
        self.wrap.0 as u64
    }

    pub(crate) fn begin(&mut self, window: u64) {
        if self.wrap.0 == 0 {
            self.spans.push(Span {
                start: self.arena.len(),
                len: 0,
                cap: 0,
            });
        } else {
            self.spans[self.wrap.at(window)].len = 0;
        }
    }

    pub(crate) fn list(&self, window: u64) -> &[T] {
        let span = self.spans[self.wrap.at(window)];
        &self.arena[span.start..span.start + span.len]
    }

    /// The window's last item once it holds `k`.
    pub(crate) fn last_if_full(&self, window: u64) -> Option<&T> {
        let span = self.spans[self.wrap.at(window)];
        (span.len == self.k)
            .then(|| self.arena.get(span.start + span.len - 1))
            .flatten()
    }

    /// Sets the window's list to `items` (best first, at most `k`), which `begin` has emptied;
    /// `false` when they do not fit.
    pub(crate) fn fill(&mut self, window: u64, items: &[T]) -> bool {
        if items.len() > self.k {
            return false;
        }
        let at = self.wrap.at(window);
        let mut span = self.spans[at];
        if self.wrap.0 == 0 {
            span = Span {
                start: self.arena.len(),
                len: items.len(),
                cap: items.len(),
            };
            self.arena.extend_from_slice(items);
        } else if items.len() <= span.cap {
            self.arena[span.start..span.start + items.len()].copy_from_slice(items);
            span.len = items.len();
        } else {
            return false;
        }
        self.spans[at] = span;
        true
    }

    /// Puts `item` in the window's list (`better(a, b)`: `a` strictly before `b`). Returns whether it was kept.
    pub(crate) fn keep(&mut self, window: u64, item: T, better: impl Fn(&T, &T) -> bool) -> bool {
        let at = self.wrap.at(window);
        let mut span = self.spans[at];
        if span.len == span.cap && span.cap < self.k && self.wrap.0 == 0 {
            let cap = span.cap.saturating_mul(2).clamp(1, self.k);
            let start = self.arena.len();
            self.arena
                .extend_from_within(span.start..span.start + span.len);
            self.arena.resize(start + cap, self.fill);
            span = Span {
                start,
                len: span.len,
                cap,
            };
        }
        let slots = &mut self.arena[span.start..span.start + span.cap];
        let kept = kbest::keep_within(slots, &mut span.len, item, better);
        self.spans[at] = span;
        kept
    }
}

/// The floors of the held windows, contiguous across any run of consecutive windows.
struct Floors {
    wrap: Wrap,
    values: Vec<f64>,
}

impl Floors {
    fn set(&mut self, window: u64, floor: f64) {
        let at = self.wrap.at(window);
        self.values[at] = floor;
        if self.wrap.0 != 0 {
            self.values[at + self.wrap.0] = floor;
        }
    }

    fn get(&self, window: u64) -> f64 {
        self.values[self.wrap.at(window)]
    }

    fn run(&self, first: u64, len: usize) -> &[f64] {
        let at = self.wrap.at(first);
        &self.values[at..at + len]
    }
}

/// One kept neighbor of a window: what it reports, and what the exact order needs of the pair.
#[derive(Clone, Copy)]
pub(crate) struct Item {
    pub(crate) entry: Entry,
    /// The pair's exact centered cross term.
    pub(crate) n: i64,
    /// The other window of the pair, as it was when the pair was ranked: it may since have left.
    pub(crate) neighbor: Window,
}

const EMPTY: Item = Item {
    entry: Entry::NONE,
    n: 0,
    neighbor: Window {
        sum: 0,
        kind: crate::stats::WindowKind::NonFinite,
    },
};

/// Whether `candidate` (against the window `own`) is exactly nearer than `item` is.
fn nearer_than(own: &Window, candidate: Candidate, item: &Item) -> bool {
    if clearly_nearer(candidate.rho_hat, item.entry.rho()) {
        return true;
    }
    if clearly_nearer(item.entry.rho(), candidate.rho_hat) {
        return false;
    }
    let index = item.entry.index();
    candidate_of(own, item.n, &item.neighbor, index).is_some_and(|other| nearer(candidate, other))
}

/// Whether item `a` is exactly nearer than item `b`, both neighbors of the window `own`.
fn item_nearer(own: &Window, a: &Item, b: &Item) -> bool {
    match candidate_of(own, a.n, &a.neighbor, a.entry.index()) {
        Some(candidate) => nearer_than(own, candidate, b),
        None => false,
    }
}

/// `n` of a candidate's pair, from its exact correlation.
fn cross_of(correlation: &ExactCorrelation) -> i64 {
    match *correlation {
        ExactCorrelation::Valid {
            sign,
            cross_squared,
            ..
        } => i64::from(sign) * i64::try_from(cross_squared.isqrt()).unwrap_or(i64::MAX),
        _ => 0,
    }
}

/// One kept neighbor of a window whose pair the hot arithmetic cannot rank (a window it holds as
/// non-finite is spilled to the cold route): what it reports, and the pair's exact correlation.
#[derive(Clone, Copy)]
pub(crate) struct ColdItem {
    pub(crate) entry: Entry,
    pub(crate) held: Held,
}

/// A window's cold lists: its right neighbors and its neighbors over both sides.
#[derive(Default)]
pub(crate) struct ColdLists {
    pub(crate) right: Vec<ColdItem>,
    pub(crate) both: Vec<ColdItem>,
}

/// One held window's lists, as a checkpoint carries them: each nearest first, every neighbor
/// with the exact correlation it was ranked by. Whether a pair was ranked by the hot arithmetic
/// or the cold route is not part of it, so the same lists are the same bytes whichever it was.
pub(crate) struct Column {
    pub(crate) right: Vec<ColdItem>,
    pub(crate) both: Vec<ColdItem>,
}

fn index_of(entry: Entry) -> u64 {
    entry.index().get().unwrap_or(u64::MAX)
}

/// The exact order of two neighbors of one window, whose exact correlations are `a` and `b`.
fn before(a: Entry, b: Entry, order: impl FnOnce() -> Ordering) -> bool {
    if clearly_nearer(a.rho(), b.rho()) {
        return true;
    }
    if clearly_nearer(b.rho(), a.rho()) {
        return false;
    }
    match order() {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => index_of(a) < index_of(b),
    }
}

fn cold_nearer(a: &ColdItem, b: &ColdItem) -> bool {
    before(a.entry, b.entry, || cmp_held(a.held, b.held))
}

/// Whether the hot neighbor `a` of the window `own` is exactly nearer than the cold one `b`.
fn hot_before_cold(own: &Window, a: &Item, b: &ColdItem) -> bool {
    before(a.entry, b.entry, || {
        candidate_of(own, a.n, &a.neighbor, a.entry.index()).map_or(Ordering::Less, |hot| {
            cmp_held(Held::Hot(hot.correlation), b.held)
        })
    })
}

/// The right and two-sided lists of every window a stream holds, and the floors the kernel
/// reads. One window is added per completed row ([`Self::begin_row`]); a stream that holds a
/// fixed number of windows drops the oldest as it adds one.
pub(crate) struct Right {
    m: usize,
    windows: Vec<Window>,
    wrap: Wrap,
    right: Lists<Item>,
    both: Lists<Item>,
    /// The lists of the pairs the cold route ranked, by window; most windows have none.
    cold: BTreeMap<u64, ColdLists>,
    floors: Floors,
    /// Each held window's last pair admitted to its right list, by slot.
    champions: Vec<Champion>,
    delta: f64,
    /// The oldest window held, and one past the newest.
    first: u64,
    next: u64,
    row: u64,
    row_window: Window,
    /// The newest row's variance numerator: `0` unless it is Valid.
    row_variance: u64,
}

impl Right {
    /// For a stream that keeps every window.
    pub(crate) fn growing(m: NonZeroUsize, k: NonZeroUsize) -> Self {
        Self {
            m: m.get(),
            windows: Vec::new(),
            wrap: Wrap(0),
            right: Lists::growing(k.get(), EMPTY),
            both: Lists::growing(k.get(), EMPTY),
            cold: BTreeMap::new(),
            floors: Floors {
                wrap: Wrap(0),
                values: Vec::new(),
            },
            champions: Vec::new(),
            delta: 0.0,
            first: 0,
            next: 0,
            row: 0,
            row_window: EMPTY.neighbor,
            row_variance: 0,
        }
    }

    /// For a stream that holds the latest `windows` windows, of which each can be reached by at
    /// most `reach` later ones; `None` when the lists cannot be allocated.
    pub(crate) fn held(
        m: NonZeroUsize,
        k: NonZeroUsize,
        windows: usize,
        reach: usize,
    ) -> Option<Self> {
        let per = k.get().min(reach);
        let both_per = k.get().min(reach.saturating_mul(2));
        let mut values = Vec::new();
        values.try_reserve_exact(windows.checked_mul(2)?).ok()?;
        values.resize(2 * windows, f64::NEG_INFINITY);
        let mut held = Vec::new();
        held.try_reserve_exact(windows).ok()?;
        held.resize(windows, EMPTY.neighbor);
        let mut champions = Vec::new();
        champions.try_reserve_exact(windows).ok()?;
        champions.resize(windows, Champion::NONE);
        Some(Self {
            m: m.get(),
            windows: held,
            wrap: Wrap(windows),
            right: Lists::held(k.get(), windows, per, EMPTY)?,
            both: Lists::held(k.get(), windows, both_per, EMPTY)?,
            cold: BTreeMap::new(),
            floors: Floors {
                wrap: Wrap(windows),
                values,
            },
            champions,
            delta: 0.0,
            first: 0,
            next: 0,
            row: 0,
            row_window: EMPTY.neighbor,
            row_variance: 0,
        })
    }

    /// The windows held: the oldest to the newest.
    pub(crate) fn held_windows(&self) -> core::ops::Range<u64> {
        self.first..self.next
    }

    /// Adds window `row`, whose pass is next; it has no later neighbor yet.
    pub(crate) fn begin_row(&mut self, row: u64, window: Window) {
        if self.wrap.0 == 0 {
            self.windows.push(window);
            self.floors.values.push(f64::NEG_INFINITY);
            self.champions.push(Champion::NONE);
        } else {
            let at = self.wrap.at(row);
            self.windows[at] = window;
            self.champions[at] = Champion::NONE;
            self.floors.set(row, f64::NEG_INFINITY);
            if row >= self.wrap.0 as u64 {
                self.first = row + 1 - self.wrap.0 as u64;
                self.cold.remove(&(self.first - 1));
            }
        }
        self.right.begin(row);
        self.both.begin(row);
        if matches!(window.kind, crate::stats::WindowKind::NonFinite) {
            self.floors.set(row, f64::INFINITY);
        }
        self.row = row;
        self.next = row + 1;
        self.row_window = window;
        self.row_variance = columns::variance(&window);
    }

    fn window(&self, window: u64) -> &Window {
        &self.windows[self.wrap.at(window)]
    }

    /// Offers the row's nearest earlier neighbors (what its pass kept) to its two-sided list.
    pub(crate) fn offer_left(&mut self, kept: impl Iterator<Item = Candidate>) {
        let (row, own) = (self.row, self.row_window);
        for candidate in kept {
            if candidate.index < self.first {
                continue;
            }
            let neighbor = *self.window(candidate.index);
            let item = Item {
                entry: winner_entry(self.m, candidate),
                n: cross_of(&candidate.correlation),
                neighbor,
            };
            self.both.keep(row, item, |a, b| item_nearer(&own, a, b));
        }
    }

    /// Offers the pair of the held window `column` and the newest row, whose cross term is `n` and
    /// whose estimate is `rho_hat`. Rows reach a column in ascending order, so a pair that ties
    /// the `k` pairs just admitted to the column's right list exactly, or follows `k` whose
    /// correlation is one, loses to them and is not offered; and once the column holds `k` of
    /// correlation one its floor rises to `∞`, since nothing later can beat them (a checked build
    /// leaves the floor and drops those pairs here instead, asserting the order).
    fn admit<P: Producer>(&mut self, column: u64, n: i64, rho_hat: f64) {
        let (slot, k) = (self.wrap.at(column), self.right.k);
        let champion = self.champions[slot];
        if cfg!(debug_assertions) && champion.is_settled(k) {
            debug_assert!(
                champion.outranks_index(self.row, k),
                "rows reach a column in ascending order"
            );
            return;
        }
        let row_variance = self.row_variance;
        let pair = Champion::of(self.row, n, row_variance, rho_hat);
        if champion.beats(&pair, k) {
            return;
        }
        let own = *self.window(column);
        let row = SampleIndex::new(self.row).expect("a row is below NONE's sentinel");
        let Some(candidate) = candidate_of(&own, n, &self.row_window, row) else {
            return;
        };
        let neighbor = self.row_window;
        let m = self.m;
        let make = || Item {
            entry: winner_entry(m, candidate),
            n,
            neighbor,
        };
        let enters = |lists: &Lists<Item>| {
            lists
                .last_if_full(column)
                .is_none_or(|last| nearer_than(&own, candidate, last))
        };
        if enters(&self.right) {
            self.right
                .keep(column, make(), |a, b| item_nearer(&own, a, b));
            if let Some(last) = self.right.last_if_full(column) {
                let floor = band_floor(last.entry.rho(), self.delta);
                self.floors.set(column, floor);
            }
            let champion = champion.then::<P>(pair, || columns::variance(&own), row_variance);
            self.champions[slot] = champion;
            if champion.is_settled(k) && !cfg!(debug_assertions) {
                self.floors.set(column, f64::INFINITY);
            }
            if enters(&self.both) {
                self.both
                    .keep(column, make(), |a, b| item_nearer(&own, a, b));
            }
        }
    }

    /// The window's `k` nearest later neighbors, nearest first.
    pub(crate) fn right_into(&self, window: u64, out: &mut Vec<Entry>) {
        out.clear();
        let hot = self.right.list(window);
        match self.cold.get(&window) {
            None => out.extend(hot.iter().map(|item| item.entry)),
            Some(cold) => self.merge_into(window, hot, &cold.right, out),
        }
    }

    /// The window's `k` nearest neighbors over both sides, nearest first.
    pub(crate) fn both_into(&self, window: u64, out: &mut Vec<Entry>) {
        out.clear();
        let hot = self.both.list(window);
        match self.cold.get(&window) {
            None => out.extend(hot.iter().map(|item| item.entry)),
            Some(cold) => self.merge_into(window, hot, &cold.both, out),
        }
    }

    /// The `k` nearest of the window's hot and cold neighbors (each nearest first, sharing none).
    fn merge_into(&self, window: u64, hot: &[Item], cold: &[ColdItem], out: &mut Vec<Entry>) {
        out.extend(self.merged(window, hot, cold).iter().map(|item| item.entry));
    }

    /// [`Self::merge_into`] with each neighbor's exact correlation.
    fn merged(&self, window: u64, hot: &[Item], cold: &[ColdItem]) -> Vec<ColdItem> {
        let own = self.window(window);
        let hot_item = |item: &Item| {
            candidate_of(own, item.n, &item.neighbor, item.entry.index()).map(|candidate| {
                ColdItem {
                    entry: item.entry,
                    held: Held::Hot(candidate.correlation),
                }
            })
        };
        let mut out = Vec::with_capacity(self.right.k.min(hot.len() + cold.len()));
        let (mut at_hot, mut at_cold) = (0, 0);
        while out.len() < self.right.k {
            let take_hot = match (hot.get(at_hot), cold.get(at_cold)) {
                (None, None) => break,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(a), Some(b)) => hot_before_cold(own, a, b),
            };
            if take_hot {
                out.extend(hot_item(&hot[at_hot]));
                at_hot += 1;
            } else {
                out.push(cold[at_cold]);
                at_cold += 1;
            }
        }
        out
    }

    /// The held window's lists.
    pub(crate) fn column(&self, window: u64) -> Column {
        let none = ColdLists::default();
        let cold = self.cold.get(&window).unwrap_or(&none);
        Column {
            right: self.merged(window, self.right.list(window), &cold.right),
            both: self.merged(window, self.both.list(window), &cold.both),
        }
    }

    /// Sets the lists of the held window `window` (added by [`Self::begin_row`]) to `column`'s,
    /// which stay apart from what the hot arithmetic keeps from here on; `false` when they do not
    /// fit what the stream keeps.
    pub(crate) fn load(&mut self, window: u64, column: Column) -> bool {
        let k = self.right.k;
        if column.right.len() > k || column.both.len() > k {
            return false;
        }
        if !column.right.is_empty() || !column.both.is_empty() {
            self.cold.insert(
                window,
                ColdLists {
                    right: column.right,
                    both: column.both,
                },
            );
        }
        true
    }

    /// Offers the pair of the newest row `row` and the window `neighbor` that the cold route
    /// ranked: `row_side` is the pair as the row ranks it, `column_side` as `neighbor` does.
    pub(crate) fn offer_cold<A: Arith>(
        &mut self,
        row: u64,
        neighbor: u64,
        row_side: Candidate<Spilled<A>>,
        column_side: Candidate<Spilled<A>>,
    ) {
        let k = self.right.k;
        let item = |side: Candidate<Spilled<A>>| ColdItem {
            entry: winner_entry(self.m, side),
            held: side.correlation.held(),
        };
        let (row_item, column_item) = (item(row_side), item(column_side));
        let lists = self.cold.entry(row).or_default();
        kbest::keep(&mut lists.both, k, row_item, cold_nearer);
        if neighbor >= self.first {
            let lists = self.cold.entry(neighbor).or_default();
            if kbest::keep(&mut lists.right, k, column_item, cold_nearer) {
                kbest::keep(&mut lists.both, k, column_item, cold_nearer);
            }
        }
    }
}

impl ColumnSink for Right {
    fn floors(&self, first: usize, len: usize) -> &[f64] {
        self.floors.run(first as u64, len)
    }

    fn offer_groups<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        mut hits: u64,
        n_of: impl Fn(usize) -> i64,
        _variance_of: impl Fn(usize) -> u64,
    ) {
        debug_assert_eq!(row.index as u64, self.row);
        self.delta = self.delta.max(P::DELTA);
        while hits != 0 {
            let lo = hits.trailing_zeros() as usize * GROUP_LANES;
            hits &= hits - 1;
            for (lane, &rho_hat) in tile.iter().enumerate().skip(lo).take(GROUP_LANES) {
                let column = (first + lane) as u64;
                if rho_hat >= self.floors.get(column) && rho_hat != f64::NEG_INFINITY {
                    self.admit::<P>(column, n_of(lane), rho_hat);
                }
            }
        }
    }

    fn offer_chunk<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        n_of: impl Fn(usize) -> i64,
        _variance_of: impl Fn(usize) -> u64,
    ) {
        debug_assert_eq!(row.index as u64, self.row);
        self.delta = self.delta.max(P::DELTA);
        for (lane, &rho_hat) in tile.iter().enumerate() {
            let column = (first + lane) as u64;
            if rho_hat >= self.floors.get(column) && rho_hat != f64::NEG_INFINITY {
                self.admit::<P>(column, n_of(lane), rho_hat);
            }
        }
    }

    fn offer_lane<P: Producer>(
        &mut self,
        column: usize,
        row: RowId,
        lane: RankedLane,
        _variance: impl FnOnce() -> u64,
    ) {
        debug_assert_eq!(row.index as u64, self.row);
        self.delta = self.delta.max(P::DELTA);
        let column = column as u64;
        if lane.rho_hat >= self.floors.get(column) && lane.rho_hat != f64::NEG_INFINITY {
            self.admit::<P>(column, lane.n, lane.rho_hat);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::stats::WindowKind;
    use std::vec;

    fn nz(x: usize) -> NonZeroUsize {
        NonZeroUsize::new(x).unwrap()
    }

    fn valid() -> Window {
        Window {
            sum: 0,
            kind: WindowKind::Valid {
                variance_numerator: 100,
                inv_sqrt: 0.1,
            },
        }
    }

    #[test]
    fn a_run_of_floors_reads_the_same_across_the_seam_of_the_ring() {
        let mut floors = Floors {
            wrap: Wrap(4),
            values: vec![f64::NEG_INFINITY; 8],
        };
        for window in 0..10u64 {
            floors.set(window, window as f64);
        }
        for first in 6..10u64 {
            for len in 1..=(10 - first) as usize {
                let run = floors.run(first, len);
                for (at, &floor) in run.iter().enumerate() {
                    assert_eq!(floor, (first as usize + at) as f64);
                }
            }
        }
    }

    #[test]
    fn cold_lists_leave_with_their_window() {
        let mut right = Right::held(nz(3), nz(1), 4, 2).unwrap();
        for row in 0..20u64 {
            right.begin_row(row, valid());
            if row > 0 {
                let side = Candidate {
                    index: row,
                    rho_hat: 0.5,
                    correlation: Spilled::<i128>::OneConstant,
                };
                right.offer_cold(row, row - 1, side, side);
            }
            assert!(right.cold.len() <= 4);
        }
    }

    #[test]
    fn loaded_lists_read_back_as_they_were_saved() {
        let mut right = Right::growing(nz(3), nz(2));
        right.begin_row(0, valid());
        let item = |index| ColdItem {
            entry: Entry::new(0.5, 1.0, SampleIndex::new(index).unwrap()).unwrap(),
            held: Held::Hot(ExactCorrelation::OneConstant),
        };
        let column = Column {
            right: vec![item(5), item(6)],
            both: vec![item(4)],
        };
        assert!(right.load(0, column));
        let mut out = Vec::new();
        right.right_into(0, &mut out);
        assert_eq!(out.len(), 2);
        right.both_into(0, &mut out);
        assert_eq!(out[0].index().get(), Some(4));
        let saved = right.column(0);
        assert_eq!(saved.right.len(), 2);
        assert!(!right.load(
            0,
            Column {
                right: vec![item(5); 3],
                both: vec![]
            }
        ));
    }
}
