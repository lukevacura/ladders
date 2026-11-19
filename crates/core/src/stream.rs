//! The first streaming engine: an unbounded, growing left nearest-neighbor profile over `f64`
//! samples. Each [`GrowingStream::update`] quantizes, forward-fills and folds one sample into
//! the running window statistics; the instant a window completes, one call to
//! `crate::row::pass_row_wide` (or, while on the narrow path, the narrow
//! `crate::row::pass_row_narrow`) advances every diagonal through it and the row's
//! exact entry is returned once, never to be revisited by a later sample.
//!
//! Unlike `BoundedStream`, nothing is ever evicted, so the anchor only has to re-center when a
//! new tick would leave its range *and* the stream's whole history still fits some anchor;
//! otherwise it switches, exactly and permanently, to the i64 pass. The running
//! `tick_min`/`tick_max` make that check O(1): rescanning the whole (ever-growing) history on
//! every tick, the way `BoundedStream` rescans its fixed-size ring, would cost O(n) extra work
//! every row, on top of the O(n) ranking work already inherent to a stream that never evicts.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::checkpoint::{self, CheckpointError};
use crate::columns;
use crate::cross_term::{self, CrossTerms};
use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::kernel_neon::Conversion;
use crate::narrow::{self, Anchor, Path, RouteCounts};
use crate::nearest::{Candidate, ExactDistance, Nearest, winner_entry};
use crate::quantize::{Scale, to_ticks};
use crate::right::Right;
use crate::row::{
    Columns, DEFAULT_TILE_LANES, FrontGrowBuffer, NarrowColumns, RowAcc, RowInputs,
    pass_row_narrow, pass_row_narrow_offering, pass_row_wide, pass_row_wide_offering, push_window,
};
use crate::sanitize::{ForwardFill, Sanitized};
use crate::spill;
use crate::spill::OutlierScan;
use crate::spill::stream::{Cold, RowView, flag_of, hold, unhold, withheld_indices, withholds};
use crate::stats::{Window, WindowKind, WindowStats};

/// Causal, exact left nearest-neighbor matrix profile over a growing stream: every array
/// extends as samples arrive, so a row once returned never changes.
pub struct GrowingStream {
    m: NonZeroUsize,
    /// `ceil(m/4)`: a neighbor this close or closer is never eligible, for any row.
    exclusion: usize,
    scale: Scale,
    /// How many nearest earlier neighbors each row keeps.
    k: NonZeroUsize,
    fill: ForwardFill<i64>,
    stats: WindowStats,
    /// Every sanitized tick seen so far, oldest first; `ticks[k..k + m]` is window `k`'s own.
    ticks: Vec<i64>,
    /// One entry per completed window, oldest first: [`Columns`]'s struct-of-arrays, owned.
    sum: Vec<i64>,
    r: Vec<f64>,
    bias: Vec<f64>,
    variance: Vec<u64>,
    /// The tick every narrowed column is anchored to; meaningless once `path` is `Wide`.
    anchor: Anchor,
    /// Which arithmetic this stream is currently running: `Narrow` through the verified
    /// NEON kernel, or `Wide`, the exact i64 fallback, permanently once switched.
    path: Path,
    /// The lowest and highest tick ever seen: tracked incrementally (never rescanned) so
    /// [`Self::maintain_narrow_range`] stays O(1) on a stream that never evicts.
    tick_min: i64,
    tick_max: i64,
    /// [`ticks`](Self::ticks), narrowed to `self.anchor`; frozen once `path` is `Wide` (never
    /// read again, so never filled again either, but never cleared).
    ticks_off: Vec<i32>,
    /// [`sum`](Self::sum), narrowed to `self.anchor`; same lifetime as [`ticks_off`](Self::ticks_off).
    sum_off: Vec<i32>,
    /// `qt[j]` is this row's state for the diagonal paired with neighbor `j`, for every
    /// eligible `j` (`0..current row − exclusion`): a diagonal's lag never changes, so grown
    /// one slot at the front each eligible row instead of shifted, nothing here ever moves on
    /// its own account. Its meaning depends on `path`: `m·QT` offsets from `anchor` while
    /// `Narrow`, plain absolute `QT` once `Wide`.
    qt: FrontGrowBuffer,
    /// The narrow path's L1-sized `ρ̂` tile and its per-row accumulator: scratch, not state.
    tile: Box<[f64]>,
    acc: RowAcc,
    /// The row's reduction and its newest result, kept between rows so a row allocates nothing
    /// once their capacity covers `k`.
    nearest: Nearest,
    entries: Vec<Entry>,
    /// Whether each tick in `ticks` had a real sample (`ticks` alone cannot say a filled tick
    /// from a present one). Neither `ticks` nor a completed window's kind keeps this once the
    /// window closes, but a checkpoint replays it through `WindowStats` (which exposes no state
    /// but `new`/`push`) to rebuild its hidden ring and to prove every stored window's kind.
    had_tick: Vec<bool>,
    /// Whether `fill` has ever turned a real sample into `Sanitized::Present` — equivalently,
    /// whether `fill`'s own (private) `last` is `Some`: a checkpoint restores `fill` through
    /// its own public `push`, which needs exactly this one bit plus the newest tick (already
    /// in `ticks`, since a present or filled tick is always written there as the same value
    /// `fill.last` holds right after it).
    seen_real_sample: bool,
    /// Route statistics since construction or restore; never part of a checkpoint.
    routes: RouteCounts,
    /// The exact cold route for the windows the hot arithmetic cannot rank: those holding a
    /// withheld tick, and those whose variance numerator passes `2⁶³`.
    cold: Cold,
    /// Whether rows may need the cold route: from the first withheld tick or the switch to the
    /// exact i64 arithmetic (the only one a variance past `2⁶³` fits in) on, since nothing
    /// leaves a growing stream.
    watching: bool,
    /// Withheld ticks, position and exact tick, oldest first.
    withheld: Vec<(u64, i64)>,
    /// The outlier rule's level and run.
    outliers: OutlierScan,
    /// `ticks` with every withheld tick exact, as far as the cold route has needed it.
    exact: Vec<i64>,
    /// The newest row's nearest and its exact distance, when [`Winner::Kept`] says they are here.
    kept_result: Box<Option<ExactDistance>>,
    /// Each window's nearest later neighbors, when the stream looks right.
    right: Option<Right>,
}

/// The window a reduction holds before its first row.
const UNSTARTED: Window = Window {
    sum: 0,
    kind: WindowKind::NonFinite,
};

impl GrowingStream {
    /// Keeps the nearest earlier neighbor of each window.
    pub fn new(m: NonZeroUsize, scale: Scale) -> Self {
        Self::with_k(m, scale, NonZeroUsize::MIN)
    }

    /// Keeps the `k` nearest earlier neighbors of each window ([`Self::update_topk`]); `k` of one
    /// is [`Self::new`]. Nothing is reserved for `k` here: each row reserves at most the
    /// neighbors it has, so a `k` beyond the windows that exist costs nothing.
    pub fn with_k(m: NonZeroUsize, scale: Scale, k: NonZeroUsize) -> Self {
        Self {
            m,
            exclusion: crate::exclusion::zone(m.get()),
            scale,
            k,
            fill: ForwardFill::default(),
            stats: WindowStats::new(m),
            ticks: Vec::new(),
            sum: Vec::new(),
            r: Vec::new(),
            bias: Vec::new(),
            variance: Vec::new(),
            anchor: Anchor { tick: 0 },
            path: Path::Narrow,
            tick_min: i64::MAX,
            tick_max: i64::MIN,
            ticks_off: Vec::new(),
            sum_off: Vec::new(),
            qt: FrontGrowBuffer::new(),
            tile: vec![0.0; DEFAULT_TILE_LANES].into_boxed_slice(),
            acc: RowAcc::new(),
            nearest: Nearest::with_k(&UNSTARTED, m.get(), k, 0),
            entries: Vec::new(),
            had_tick: Vec::new(),
            seen_real_sample: false,
            routes: RouteCounts::default(),
            cold: Cold::new(m, k, None),
            watching: false,
            withheld: Vec::new(),
            outliers: OutlierScan::default(),
            exact: Vec::new(),
            kept_result: Box::new(None),
            right: None,
        }
    }

    /// [`Self::with_k`] that also keeps, for each window, its `k` nearest *later* neighbors and
    /// its `k` nearest over both sides ([`Self::right_neighbors_into`],
    /// [`Self::two_sided_neighbors_into`]). This looks ahead of a window: an entry for a window
    /// can change when later samples arrive, so nothing causal may read it. Each row's pass
    /// offers every pair it ranks to the earlier window too; the stream's own entries are those
    /// of [`Self::with_k`], bit for bit.
    pub fn with_right_side(m: NonZeroUsize, scale: Scale, k: NonZeroUsize) -> Self {
        let mut stream = Self::with_k(m, scale, k);
        stream.right = Some(Right::growing(m, k));
        stream
    }

    /// Whether the stream keeps the right side ([`Self::with_right_side`]).
    pub fn looks_right(&self) -> bool {
        self.right.is_some()
    }

    /// The windows whose right and two-sided neighbors the stream holds: every window so far, or
    /// none when it does not look right.
    pub fn right_windows(&self) -> core::ops::Range<u64> {
        self.right.as_ref().map_or(0..0, Right::held_windows)
    }

    /// Window `window`'s `k` nearest later neighbors so far, nearest first under the exact order
    /// (ties to the smaller index), in `out` (cleared first); none for a window not held or a
    /// stream that does not look right. A window's entries are as the batch two-sided profile
    /// over the samples so far gives its right side.
    pub fn right_neighbors_into(&self, window: u64, out: &mut Vec<Entry>) {
        out.clear();
        if let Some(right) = self
            .right
            .as_ref()
            .filter(|_| self.right_windows().contains(&window))
        {
            right.right_into(window, out);
        }
    }

    /// Window `window`'s `k` nearest neighbors over both sides so far, as
    /// [`Self::right_neighbors_into`] but over the earlier ones too.
    pub fn two_sided_neighbors_into(&self, window: u64, out: &mut Vec<Entry>) {
        out.clear();
        if let Some(right) = self
            .right
            .as_ref()
            .filter(|_| self.right_windows().contains(&window))
        {
            right.both_into(window, out);
        }
    }

    /// The window length this stream was made with.
    pub fn window_length(&self) -> NonZeroUsize {
        self.m
    }

    /// How many nearest earlier neighbors each window keeps.
    pub fn k(&self) -> NonZeroUsize {
        self.k
    }

    /// The tick size each sample is quantized to.
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// Which arithmetic this stream is currently running: `Narrow`'s `m·QT` offsets
    /// through the verified NEON kernel, or `Wide`, the exact i64 fallback — permanently,
    /// once switched.
    #[doc(hidden)]
    pub fn path(&self) -> Path {
        self.path
    }

    /// Updates by route and route changes since this stream was created or restored.
    #[doc(hidden)]
    pub fn route_counts(&self) -> RouteCounts {
        self.routes
    }

    /// Every scalar and array this stream's future entries depend on, versioned and
    /// little-endian (`checkpoint.rs` owns the format). [`Self::restore`] on these bytes
    /// continues bitwise identically to a stream that never stopped. The bytes hold the
    /// samples as they came, whatever the stream withheld from its hot arithmetic, so they do
    /// not depend on the route.
    pub fn checkpoint(&self) -> Vec<u8> {
        let plain = self.checkpoint_plain();
        match &self.right {
            None => plain,
            Some(right) => checkpoint::wrap_right(checkpoint::KIND_GROWING, &plain, |buf| {
                checkpoint::put_columns(
                    buf,
                    right.held_windows().map(|w| right.column(w)).collect(),
                );
            }),
        }
    }

    fn checkpoint_plain(&self) -> Vec<u8> {
        let spill = checkpoint::Spill {
            constant: (0..self.sum.len() as u64)
                .map(|row| self.cold.constant_too_wide(row))
                .collect(),
        };
        if self.withheld.is_empty() {
            let qt = self.absolute_qt();
            return checkpoint::write_growing(checkpoint::GrowingSnapshot {
                m: self.m,
                k: self.k,
                scale: self.scale,
                seen_real_sample: self.seen_real_sample,
                ticks: &self.ticks,
                sum: &self.sum,
                r: &self.r,
                bias: &self.bias,
                variance: &self.variance,
                qt: &qt,
                had_tick: &self.had_tick,
                spill,
            });
        }
        let mut ticks = self.ticks.clone();
        let mut had = self.had_tick.clone();
        let withheld: Vec<(usize, i64)> = self
            .withheld
            .iter()
            .map(|&(position, tick)| (position as usize, tick))
            .collect();
        unhold(&mut ticks, &mut had, &withheld);
        let mut made = Vec::with_capacity(self.sum.len());
        checkpoint::replay_stats(self.m, &ticks, &had, |_, window| {
            made.push(window);
            true
        });
        let (mut sum, mut r, mut bias, mut variance) = (
            self.sum.clone(),
            self.r.clone(),
            self.bias.clone(),
            self.variance.clone(),
        );
        for (j, &made) in made.iter().enumerate() {
            let spilled = self.cold.spilled(j as u64);
            if spilled || made.kind == WindowKind::NonFinite {
                let window = made.under(spilled.then(|| self.cold.spilled_constant(j as u64)));
                sum[j] = window.sum;
                (r[j], bias[j], variance[j]) = columns::encode(&window);
            }
        }
        let m = self.m.get();
        let qt: Vec<i64> = match self.sum.len().checked_sub(1) {
            Some(row) => (0..row.saturating_sub(self.exclusion))
                .map(|j| cross_term::wide(&ticks[row..row + m], &ticks[j..j + m]))
                .collect(),
            None => Vec::new(),
        };
        checkpoint::write_growing(checkpoint::GrowingSnapshot {
            m: self.m,
            k: self.k,
            scale: self.scale,
            seen_real_sample: self.seen_real_sample,
            ticks: &ticks,
            sum: &sum,
            r: &r,
            bias: &bias,
            variance: &variance,
            qt: &qt,
            had_tick: &had,
            spill,
        })
    }

    /// Window `j` as the hot arithmetic holds it.
    fn window(&self, j: usize) -> Window {
        columns::window_of(self.sum[j], self.r[j], self.bias[j], self.variance[j])
    }

    /// Every live diagonal's absolute `QT`, whichever arithmetic the stream is running.
    fn absolute_qt(&self) -> Cow<'_, [i64]> {
        let qt = self.qt.as_slice();
        let Some(newest_row) = self.sum.len().checked_sub(1) else {
            return Cow::Borrowed(qt);
        };
        if self.path == Path::Wide {
            return Cow::Borrowed(qt);
        }
        let newest_sum = self.sum[newest_row];
        Cow::Owned(
            qt.iter()
                .zip(&self.sum)
                .map(|(&qt_m, &sum)| {
                    narrow::absolute_qt(self.anchor, self.m, qt_m, newest_sum, sum)
                })
                .collect(),
        )
    }

    /// Rebuilds a stream from [`Self::checkpoint`]'s own bytes: every ring, the front-growing
    /// `qt` buffer, and the hidden state behind `stats`/`fill` are all rebuilt fresh through
    /// their own public constructors, never by poking a private field directly. Any truncated
    /// or garbled input is rejected with a [`CheckpointError`] rather than panicking.
    pub fn restore(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let Some(mut wrapped) = checkpoint::open_right(bytes, checkpoint::KIND_GROWING)? else {
            return Self::restore_plain(bytes);
        };
        let mut stream = Self::restore_plain(wrapped.inner)?;
        let columns =
            checkpoint::get_columns(&mut wrapped.reader, stream.k.get(), stream.sum.len())?;
        wrapped.reader.finish()?;
        let mut right = Right::growing(stream.m, stream.k);
        for (row, column) in columns.into_iter().enumerate() {
            right.begin_row(row as u64, stream.window(row));
            if !right.load(row as u64, column) {
                return Err(CheckpointError::Inconsistent);
            }
        }
        stream.right = Some(right);
        Ok(stream)
    }

    fn restore_plain(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let mut parts: checkpoint::GrowingParts = checkpoint::read_growing(bytes)?;
        let samples = Self::withhold_restored(&mut parts)?;
        let exclusion = crate::exclusion::zone(parts.m.get());
        // Matches `update`'s own `count` exactly: the newest row's qt count is
        // `row.saturating_sub(exclusion)` where `row` is that row's own 0-indexed number, one
        // less than the number of windows completed so far (`parts.sum.len()`).
        let count = parts
            .sum
            .len()
            .checked_sub(1)
            .map_or(0, |newest_row| newest_row.saturating_sub(exclusion));

        let mut qt = FrontGrowBuffer::new();
        for _ in 0..count {
            qt.grow_front();
        }
        qt.as_mut_slice().copy_from_slice(&parts.qt);

        // The route is chosen the way a fresh stream chooses it: narrow when one anchor covers
        // every tick, otherwise the exact i64 arithmetic.
        let (tick_min, tick_max) = parts
            .ticks
            .iter()
            .fold((i64::MAX, i64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)));
        let covering = Anchor::covering(&parts.ticks, parts.m);
        let path = if covering.is_some() || parts.ticks.is_empty() {
            Path::Narrow
        } else {
            Path::Wide
        };
        let anchor = covering.unwrap_or(Anchor { tick: 0 });
        let (mut ticks_off, mut sum_off) = (Vec::new(), Vec::new());
        if path == Path::Narrow {
            narrow::narrow_ticks_into(anchor, parts.m, &parts.ticks, &mut ticks_off);
            narrow::narrow_sums_into(anchor, parts.m, &parts.sum, &mut sum_off);
            if let Some(newest_sum) = parts.sum.last() {
                for (slot, &sum) in qt.as_mut_slice().iter_mut().zip(&parts.sum) {
                    *slot = narrow::narrow_qt_m(anchor, parts.m, *slot, *newest_sum, sum);
                }
            }
        }

        // `WindowStats` exposes no state but `new`/`push`: replaying every tick with its
        // presence reconstructs its hidden ring and deques exactly, since their own future
        // behavior depends only on the trailing `m` ticks, never on the absolute position.
        let stats = checkpoint::replay_stats(parts.m, &parts.ticks, &parts.had_tick, |_, _| true)
            .ok_or(CheckpointError::Inconsistent)?;

        // `ForwardFill::push(Some(v))` sets its own hidden `last` to `Some(v)` unconditionally,
        // so one call reconstructs it exactly: the newest tick is always `fill.last`'s own
        // value, present or filled (sanitize.rs's `push` returns that same value either way).
        let mut fill = ForwardFill::default();
        if parts.seen_real_sample {
            let last = *samples.ticks.last().ok_or(CheckpointError::Inconsistent)?;
            let _ = fill.push(Some(last));
        }

        let mut stream = Self {
            m: parts.m,
            exclusion,
            scale: parts.scale,
            k: parts.k,
            fill,
            stats,
            ticks: parts.ticks,
            sum: parts.sum,
            r: parts.r,
            bias: parts.bias,
            variance: parts.variance,
            anchor,
            path,
            tick_min,
            tick_max,
            ticks_off,
            sum_off,
            qt,
            tile: vec![0.0; DEFAULT_TILE_LANES].into_boxed_slice(),
            acc: RowAcc::new(),
            nearest: Nearest::with_k(&UNSTARTED, parts.m.get(), parts.k, 0),
            entries: Vec::new(),
            had_tick: parts.had_tick,
            seen_real_sample: parts.seen_real_sample,
            routes: RouteCounts::default(),
            cold: Cold::new(parts.m, parts.k, None),
            watching: false,
            withheld: Vec::new(),
            outliers: OutlierScan::default(),
            exact: Vec::new(),
            kept_result: Box::new(None),
            right: None,
        };
        stream.restore_cold(samples)?;
        Ok(stream)
    }

    /// Turns `parts`, which hold the samples as they came, into what the hot arithmetic holds
    /// once the stream withholds what [`withheld_indices`] says of them; returns the samples as
    /// they came.
    fn withhold_restored(
        parts: &mut checkpoint::GrowingParts,
    ) -> Result<Restored, CheckpointError> {
        let bad = CheckpointError::Inconsistent;
        let m = parts.m.get();
        let stored = |j: usize| {
            columns::window_of(parts.sum[j], parts.r[j], parts.bias[j], parts.variance[j])
        };
        let mut made = Vec::with_capacity(parts.sum.len());
        checkpoint::replay_stats(parts.m, &parts.ticks, &parts.had_tick, |_, window| {
            made.push(window);
            true
        })
        .ok_or(bad)?;
        let flags: Vec<Option<bool>> = made
            .iter()
            .enumerate()
            .map(|(j, &made)| {
                let stored = stored(j);
                match stored.kind {
                    WindowKind::NonFinite => parts
                        .spill
                        .constant
                        .get(j)
                        .copied()
                        .unwrap_or(false)
                        .then_some(true),
                    _ => flag_of(stored, made),
                }
            })
            .collect();
        let ticks = parts.ticks.clone();
        let had = parts.had_tick.clone();
        let (withheld, outliers) = if spill::stream::available(m) {
            withheld_indices(m, &ticks, &had)
        } else {
            (Vec::new(), OutlierScan::default())
        };
        if !withheld.is_empty() {
            hold(&mut parts.ticks, &mut parts.had_tick, &withheld);
            let mut hot = Vec::with_capacity(parts.sum.len());
            checkpoint::replay_stats(parts.m, &parts.ticks, &parts.had_tick, |_, window| {
                hot.push(window);
                true
            })
            .ok_or(bad)?;
            for (j, window) in hot.iter().enumerate() {
                let window = window.under(flags[j]);
                parts.sum[j] = window.sum;
                (parts.r[j], parts.bias[j], parts.variance[j]) = columns::encode(&window);
            }
            if let Some(row) = parts.sum.len().checked_sub(1) {
                for (j, qt) in parts.qt.iter_mut().enumerate() {
                    *qt = cross_term::wide(&parts.ticks[row..row + m], &parts.ticks[j..j + m]);
                }
            }
        }
        Ok(Restored {
            withheld: withheld.iter().map(|&at| (at as u64, ticks[at])).collect(),
            outliers,
            ticks,
            had,
            flags,
        })
    }

    /// Rebuilds the cold route's state from the samples as they came: every window classified
    /// again, as when it was born.
    fn restore_cold(&mut self, samples: Restored) -> Result<(), CheckpointError> {
        self.withheld = samples.withheld;
        self.outliers = samples.outliers;
        let any_flagged = samples.flags.contains(&Some(true));
        if self.withheld.is_empty() && self.path == Path::Narrow && !any_flagged {
            return Ok(());
        }
        self.watch();
        let m = self.m.get() as u64;
        let had = &samples.had;
        let gap = |position: u64| !had[position as usize];
        for row in 0..self.sum.len() {
            let window = self.window(row);
            let row = row as u64;
            let view = RowView {
                row,
                exact: &samples.ticks,
                base: 0,
                oldest: 0,
                hot_window: window,
                flag: samples.flags[row as usize],
                holds_withheld: self
                    .withheld
                    .iter()
                    .any(|&(position, _)| (row..row + m).contains(&position)),
                gap: &gap,
                hot: |_: u64| window,
            };
            let spilled = self.cold.classify(&view);
            if samples.flags[row as usize] == Some(true)
                && window.kind == WindowKind::NonFinite
                && !spilled
            {
                return Err(CheckpointError::Inconsistent);
            }
        }
        if !self.withheld.is_empty() {
            self.exact = self.ticks.clone();
            for &(position, tick) in &self.withheld {
                self.exact[position as usize] = tick;
            }
        }
        Ok(())
    }

    /// `None` until the first window completes; then `(row, entry)` for the newest window: its
    /// nearest earlier neighbor, the first of what [`Self::update_topk`] keeps.
    pub fn update(&mut self, x: f64) -> Option<(SampleIndex, Entry)> {
        self.update_flagged(x, None)
    }

    /// [`Self::update`] with the caller's flag for the window this sample completes being
    /// constant (`None` for the default rule).
    pub fn update_flagged(
        &mut self,
        x: f64,
        constant: Option<bool>,
    ) -> Option<(SampleIndex, Entry)> {
        let (row, winner) = self.advance(x, constant)?;
        Some((row, self.entry_of(winner)))
    }

    /// The nearest `winner` names.
    fn entry_of(&self, winner: Winner) -> Entry {
        match winner {
            Winner::One(winner) => winner.map_or(Entry::NONE, |w| winner_entry(self.m.get(), w)),
            Winner::Kept => self.kept_result.map_or(Entry::NONE, |kept| kept.entry()),
        }
    }

    /// The nearest `winner` names, with its exact correlation.
    fn exact_of(&self, winner: Winner) -> Option<ExactDistance> {
        match winner {
            Winner::One(winner) => winner.map(|w| ExactDistance::of_winner(self.m.get(), w)),
            Winner::Kept => *self.kept_result,
        }
    }

    /// [`Self::update`], keeping the exact correlation the newest window's neighbor was ranked
    /// by (`None` where `update` reports [`Entry::NONE`]), with the same neighbor and distance bits.
    pub fn update_exact(&mut self, x: f64) -> Option<(SampleIndex, Option<ExactDistance>)> {
        self.update_exact_flagged(x, None)
    }

    /// [`Self::update_exact`] with the caller's flag for the window this sample completes.
    pub fn update_exact_flagged(
        &mut self,
        x: f64,
        constant: Option<bool>,
    ) -> Option<(SampleIndex, Option<ExactDistance>)> {
        let (row, winner) = self.advance(x, constant)?;
        Some((row, self.exact_of(winner)))
    }

    /// `None` until the first window completes; then the newest window's row and its `k` nearest
    /// earlier neighbors, nearest first under the exact order with ties to the smaller index:
    /// fewer when fewer exist, none for a window without an eligible neighbor. The slice holds
    /// until the next update.
    pub fn update_topk(&mut self, x: f64) -> Option<(SampleIndex, &[Entry])> {
        self.update_topk_flagged(x, None)
    }

    /// [`Self::update_topk`] with the caller's flag for the window this sample completes.
    pub fn update_topk_flagged(
        &mut self,
        x: f64,
        constant: Option<bool>,
    ) -> Option<(SampleIndex, &[Entry])> {
        let (row, winner) = self.advance(x, constant)?;
        if self.k == NonZeroUsize::MIN {
            let nearest = self.entry_of(winner);
            self.entries.clear();
            self.entries.extend(nearest.index().get().map(|_| nearest));
        }
        Some((row, &self.entries))
    }

    /// Folds in one sample and returns a completed window's row and its nearest neighbor; with
    /// `k` above one its kept neighbors are left in `self.entries` too.
    fn advance(&mut self, x: f64, constant: Option<bool>) -> Option<(SampleIndex, Winner)> {
        let (mut tick, mut had_tick) = match self.fill.push(to_ticks(x, self.scale)) {
            Sanitized::Present(t) => (t, true),
            Sanitized::Filled(t) => (t, false),
            // `0` here is a placeholder tick, pushed into `self.ticks` like any other (so
            // later windows still align), and never evicted (growing mode keeps everything):
            // unlike `BoundedStream`'s own "window 0 leaving" placeholder, which only a
            // single ring slot ever reads and which ages out once that slot is reused, this
            // one is read forever by any window whose own `leave`/`enter` lands on it. The
            // span check must see it like any other stored tick, so a series with a leading
            // gap followed by real data far from `0` switches to Wide instead of trying (and
            // failing) to narrow a tick that will always be live.
            Sanitized::Leading => (0, false),
        };
        self.had_tick.push(had_tick);
        // Checked (and, if needed, re-anchored or switched) before `tick` joins `self.ticks`:
        // so a re-anchor's own recompute always starts from exactly last row's narrowed
        // columns, undisturbed by this tick's own push.
        if let Some(level) = self.maintain_narrow_range(tick) {
            (tick, had_tick) = (level, false);
        }
        if had_tick {
            self.outliers.keep(tick);
        }
        self.seen_real_sample |= had_tick;
        self.routes.count_update(self.path);
        self.ticks.push(tick);
        if self.path == Path::Narrow {
            self.ticks_off
                .push(narrow::tick_offset(self.anchor, self.m, tick));
        }
        let window = self.stats.push(tick, had_tick)?.under(constant);

        let row = self.sum.len();
        push_window(
            window,
            &mut self.sum,
            &mut self.r,
            &mut self.bias,
            &mut self.variance,
        );
        let row_sum_offset = if self.path == Path::Narrow {
            let offset = narrow::sum_offset(self.anchor, self.m, window.sum);
            self.sum_off.push(offset);
            Some(offset)
        } else {
            None
        };

        if let Some(right) = &mut self.right {
            right.begin_row(row as u64, window);
        }
        let count = row.saturating_sub(self.exclusion);
        // `count` grows by exactly one on every call that reaches here (one completed window
        // at a time), so at most one fresh slot is ever needed; `qt` itself never shifts.
        if count > 0 {
            self.qt.grow_front();
        }

        if self.watching {
            return Some(self.rank_watched(row, window, row_sum_offset, count, constant));
        }
        let winner = self.rank_hot(row, window, row_sum_offset, count);
        let row_index = SampleIndex::new(row as u64).expect("row is below NONE's sentinel");
        Some((row_index, winner))
    }

    /// [`Self::rank_hot`] followed by the cold route ([`Self::rank_cold`]).
    #[cold]
    #[inline(never)]
    fn rank_watched(
        &mut self,
        row: usize,
        window: Window,
        row_sum_offset: Option<i32>,
        count: usize,
        constant: Option<bool>,
    ) -> (SampleIndex, Winner) {
        let winner = self.rank_hot(row, window, row_sum_offset, count);
        let hot = self.exact_of(winner);
        *self.kept_result = self.rank_cold(row as u64, window, constant, hot).1;
        let row_index = SampleIndex::new(row as u64).expect("row is below NONE's sentinel");
        (row_index, Winner::Kept)
    }

    /// The hot arithmetic's nearest neighbor of `row` (window `window`, its sum narrowed to
    /// `row_sum_offset` on the narrow arithmetic) over its `count` eligible neighbors, and with
    /// `k` above one its kept neighbors in `self.entries`.
    #[inline(always)]
    fn rank_hot(
        &mut self,
        row: usize,
        window: Window,
        row_sum_offset: Option<i32>,
        count: usize,
    ) -> Winner {
        let m = self.m.get();
        let row_inputs = RowInputs { row, window };
        let kept = match (self.path, row_sum_offset) {
            (Path::Narrow, Some(row_sum_offset)) => {
                let cols = NarrowColumns {
                    ticks: &self.ticks_off,
                    sum: &self.sum_off,
                    r: &self.r,
                    bias: &self.bias,
                    variance: &self.variance,
                    conversion: Conversion::FULL_RANGE,
                };
                self.acc.begin(&window, self.k, count);
                if self.k == NonZeroUsize::MIN {
                    let mut nearest = Nearest::new(&window, m);
                    match self.right.as_mut() {
                        None => pass_row_narrow(
                            self.qt.as_mut_slice(),
                            &cols,
                            &row_inputs,
                            row_sum_offset,
                            m,
                            self.exclusion,
                            &mut self.acc,
                            &mut self.tile,
                            &mut nearest,
                        ),
                        Some(right) => {
                            pass_row_narrow_offering(
                                self.qt.as_mut_slice(),
                                &cols,
                                &row_inputs,
                                row_sum_offset,
                                m,
                                self.exclusion,
                                (&mut self.acc, &mut self.tile, &mut nearest),
                                Some(&mut *right),
                            );
                            right.offer_left(nearest.kept());
                        }
                    }
                    return Winner::One(nearest.finish_winner());
                } else {
                    self.nearest.restart(&window, m, self.k, count);
                    match self.right.as_mut() {
                        None => pass_row_narrow(
                            self.qt.as_mut_slice(),
                            &cols,
                            &row_inputs,
                            row_sum_offset,
                            m,
                            self.exclusion,
                            &mut self.acc,
                            &mut self.tile,
                            &mut self.nearest,
                        ),
                        Some(right) => {
                            pass_row_narrow_offering(
                                self.qt.as_mut_slice(),
                                &cols,
                                &row_inputs,
                                row_sum_offset,
                                m,
                                self.exclusion,
                                (&mut self.acc, &mut self.tile, &mut self.nearest),
                                Some(&mut *right),
                            );
                            right.offer_left(self.nearest.kept());
                        }
                    }
                    self.nearest.drain_k(&mut self.entries)
                }
            }
            _ => {
                let cols = Columns {
                    ticks: &self.ticks,
                    sum: &self.sum,
                    r: &self.r,
                    bias: &self.bias,
                    variance: &self.variance,
                };
                self.acc.begin(&window, self.k, count);
                self.nearest.restart(&window, m, self.k, count);
                match self.right.as_mut() {
                    None => pass_row_wide(
                        self.qt.as_mut_slice(),
                        &cols,
                        &row_inputs,
                        m,
                        self.exclusion,
                        &mut self.acc,
                        &mut self.tile,
                        &mut self.nearest,
                    ),
                    Some(right) => {
                        pass_row_wide_offering(
                            self.qt.as_mut_slice(),
                            &cols,
                            &row_inputs,
                            m,
                            self.exclusion,
                            (&mut self.acc, &mut self.tile, &mut self.nearest),
                            Some(&mut *right),
                        );
                        right.offer_left(self.nearest.kept());
                    }
                }
                self.nearest.drain_k(&mut self.entries)
            }
        };
        *self.kept_result = kept;
        Winner::Kept
    }

    /// Whether `tick`, the newest sample, is withheld from the hot arithmetic ([`withholds`]): a
    /// present sample after an earlier present one.
    #[cold]
    #[inline(never)]
    fn withholds(&self, tick: i64) -> bool {
        if !self.newest_follows_a_sample() {
            return false;
        }
        let Some(level) = self.outliers.level() else {
            return false;
        };
        withholds(self.m.get(), tick, level, || self.outliers.run())
    }

    /// Whether the newest sample is withheld whatever the rule says: only under a forced route.
    fn forced_withhold(&self) -> bool {
        crate::route::spill_engaged(false) && self.newest_follows_a_sample()
    }

    /// Whether the newest sample is present and an earlier one was, with the cold route there.
    fn newest_follows_a_sample(&self) -> bool {
        self.cold.available() && self.seen_real_sample && self.had_tick.last() == Some(&true)
    }

    /// Withholds `tick` from the hot arithmetic, which holds the newest tick it has in its place
    /// (returned), as for a gap.
    #[cold]
    #[inline(never)]
    fn withhold(&mut self, tick: i64) -> i64 {
        let level = *self.ticks.last().expect("a tick precedes");
        *self.had_tick.last_mut().expect("the newest sample's flag") = false;
        self.withheld.push((self.ticks.len() as u64, tick));
        self.outliers.withhold();
        self.routes.withheld_ticks += 1;
        self.watch();
        level
    }

    /// Makes every row from here on run the cold route.
    fn watch(&mut self) {
        if !self.watching {
            self.cold.clear();
        }
        self.watching = true;
    }

    /// The cold route for `row` (hot window `window`, flag `flag`): classifies it, ranks the
    /// pairs the hot arithmetic cannot, and merges them with `hot`, the hot nearest (and with
    /// `k` above one, `self.entries`). Returns the nearest, and it as an
    /// [`ExactDistance`].
    #[cold]
    #[inline(never)]
    fn rank_cold(
        &mut self,
        row: u64,
        window: Window,
        flag: Option<bool>,
        hot: Option<ExactDistance>,
    ) -> (Entry, Option<ExactDistance>) {
        if self.k == NonZeroUsize::MIN {
            self.entries.clear();
            self.entries.extend(hot.map(|hot| hot.entry()));
        }
        let withheld = &self.withheld;
        let exact: &[i64] = if withheld.is_empty() {
            &self.ticks
        } else {
            for position in self.exact.len()..self.ticks.len() {
                let tick = match withheld.binary_search_by_key(&(position as u64), |&(p, _)| p) {
                    Ok(at) => withheld[at].1,
                    Err(_) => self.ticks[position],
                };
                self.exact.push(tick);
            }
            &self.exact
        };
        let had = &self.had_tick;
        let gap = |position: u64| {
            !had[position as usize]
                && withheld
                    .binary_search_by_key(&position, |&(withheld, _)| withheld)
                    .is_err()
        };
        let (sum, r, bias, variance) = (&self.sum, &self.r, &self.bias, &self.variance);
        let view = RowView {
            row,
            exact,
            base: 0,
            oldest: 0,
            hot_window: window,
            flag,
            holds_withheld: withheld
                .last()
                .is_some_and(|&(position, _)| position >= row),
            gap: &gap,
            hot: |neighbor: u64| {
                let j = neighbor as usize;
                columns::window_of(sum[j], r[j], bias[j], variance[j])
            },
        };
        let Some(result) = self
            .cold
            .rank(&view, &mut self.entries, hot, self.right.as_mut())
        else {
            return (hot.map_or(Entry::NONE, |hot| hot.entry()), hot);
        };
        self.routes.cold_rows += u64::from(result.ranked);
        (result.entry, result.exact)
    }

    /// Checked every tick. A no-op once `self.path` is `Wide`, or while `tick` still
    /// fits `self.anchor`. Otherwise tries to re-anchor every narrowed column exactly
    /// ([`narrow::tick_offset`]/[`narrow::sum_offset`], recomputed fresh for every tick and
    /// window the stream has ever seen) to a new anchor covering the stream's whole history
    /// so far, including `tick`; when even the best anchor could not cover that span, switches
    /// to `Wide` permanently instead, converting every live diagonal's state exactly.
    ///
    /// Returns the tick held in place of `tick` when `tick` is withheld instead
    /// ([`Self::withholds`]). Asked only where the tick does not fit the anchor or the stream is
    /// on the i64 arithmetic: a tick that fits lies within twice the anchor's bound of every
    /// tick held, which is inside the rule's limit.
    #[inline(always)]
    fn maintain_narrow_range(&mut self, tick: i64) -> Option<i64> {
        if self.path == Path::Wide {
            return self.withholds(tick).then(|| self.withhold(tick));
        }
        if crate::route::wide(false) {
            self.switch_to_wide();
            return self.withholds(tick).then(|| self.withhold(tick));
        }
        let needed = self.anchor.offset(tick, self.m).is_none();
        if !crate::route::reanchor_due(self.ticks.len() as u64, needed) {
            if self.forced_withhold() {
                return Some(self.withhold(tick));
            }
            self.tick_min = self.tick_min.min(tick);
            self.tick_max = self.tick_max.max(tick);
            return None;
        }
        if if needed {
            self.withholds(tick)
        } else {
            self.forced_withhold()
        } {
            return Some(self.withhold(tick));
        }
        self.tick_min = self.tick_min.min(tick);
        self.tick_max = self.tick_max.max(tick);

        // Centered, not pinned to `tick_min`'s own edge, for the same reason
        // `BoundedStream::maintain_narrow_range` centers its own anchor: a walk that keeps
        // setting new running extrema on both sides re-anchors far less often this way.
        let Some(new_anchor) = Anchor::spanning(
            self.tick_min,
            self.tick_max,
            self.m,
            self.ticks.len() as u64,
        ) else {
            self.switch_to_wide();
            return None;
        };
        // `self.qt` itself must move to the new anchor too, not only the per-window columns:
        // it is live state (every eligible diagonal's own `m·QT`), not something a fresh
        // narrowed read of `self.ticks`/`self.sum` could ever reconstruct. Done first, while
        // `self.anchor` still holds the *old* anchor `reanchor_live_qt` needs.
        self.reanchor_live_qt(new_anchor);
        self.reanchor_narrow_columns(new_anchor);
        self.anchor = new_anchor;
        self.routes.reanchors += 1;
        None
    }

    /// Re-anchors every live diagonal's `m·QT` (`self.qt`) from `self.anchor` to
    /// `new_anchor`, exactly ([`narrow::reanchor`]): `sa`/`sb` are each pair's two windows'
    /// own sum offsets under the *old* anchor, read fresh from their permanently stored
    /// absolute sums rather than kept incrementally, so this never accumulates its own
    /// rounding beyond `reanchor`'s one step. Must run before `self.anchor` is updated.
    fn reanchor_live_qt(&mut self, new_anchor: Anchor) {
        let completed = self.sum.len();
        if completed == 0 {
            return;
        }
        let newest_row = completed - 1;
        let count = newest_row.saturating_sub(self.exclusion);
        if count == 0 {
            return;
        }
        let delta = new_anchor.tick - self.anchor.tick;
        let sa = i64::from(narrow::sum_offset(
            self.anchor,
            self.m,
            self.sum[newest_row],
        ));
        // The logical slice (`j = 0..count`, ascending by neighbor), never the physical
        // buffer: grow-shift's own front-growing `qt` places neighbor `j` at a physical index
        // that has nothing to do with `j` itself.
        let qt = self.qt.as_mut_slice();
        for (j, qt_slot) in qt.iter_mut().enumerate().take(count) {
            let sb = i64::from(narrow::sum_offset(self.anchor, self.m, self.sum[j]));
            let (qt_m, _, _) = narrow::reanchor(delta, self.m, *qt_slot, sa, sb);
            *qt_slot = qt_m;
        }
    }

    /// Recomputes every narrowed column from its own absolute source (`self.ticks`,
    /// `self.sum`) at `new_anchor`: exact and independent of the old anchor, so this never
    /// accumulates its own rounding beyond one fresh conversion per value.
    fn reanchor_narrow_columns(&mut self, new_anchor: Anchor) {
        for (slot, &tick) in self.ticks.iter().enumerate() {
            self.ticks_off[slot] = narrow::tick_offset(new_anchor, self.m, tick);
        }
        for (slot, &sum) in self.sum.iter().enumerate() {
            self.sum_off[slot] = narrow::sum_offset(new_anchor, self.m, sum);
        }
    }

    /// Switches permanently to the exact i64 fallback, converting every live diagonal's
    /// state exactly: freshly computed absolute `QT` ([`cross_term::wide`]) for the pair `(newest
    /// completed row, j)`, the same pair `self.qt[j]` already represents, so the ordinary wide
    /// [`pass_row_wide`] can advance it normally on every row from here on. Runs before
    /// `tick` joins `self.ticks` (see the call site), so `self.ticks` here still ends exactly
    /// at the newest completed row's own last tick.
    fn switch_to_wide(&mut self) {
        self.routes.switches_to_wide += 1;
        self.watch();
        let completed = self.sum.len();
        if completed > 0 {
            let newest_row = completed - 1;
            let count = newest_row.saturating_sub(self.exclusion);
            if count > 0 {
                let pairs = CrossTerms::of(&self.ticks[..], self.m.get());
                // The logical slice (`j = 0..count`, ascending by neighbor), never the
                // physical buffer: see `reanchor_live_qt`'s own comment on this.
                let qt = self.qt.as_mut_slice();
                for (j, qt_slot) in qt.iter_mut().enumerate().take(count) {
                    *qt_slot = pairs.at(newest_row, j);
                }
            }
        }
        self.path = Path::Wide;
    }
}

/// The samples a checkpoint holds as they came, and what restoring withholds of them.
struct Restored {
    /// Positions and exact ticks withheld, oldest first.
    withheld: Vec<(u64, i64)>,
    /// The outlier rule's level and run after every tick.
    outliers: OutlierScan,
    /// Every tick with its presence.
    ticks: Vec<i64>,
    had: Vec<bool>,
    /// The caller's flag each window was stored under.
    flags: Vec<Option<bool>>,
}

/// A row's nearest as [`GrowingStream::advance`] leaves it: the one winner of the narrow
/// arithmetic, not yet reported, or a result kept in `kept_result`.
enum Winner {
    One(Option<Candidate>),
    Kept,
}

impl crate::stream_api::Stream for GrowingStream {
    fn update_flagged(&mut self, x: f64, constant: Option<bool>) -> Option<(SampleIndex, Entry)> {
        self.update_flagged(x, constant)
    }
}

#[cfg(test)]
mod tests;
