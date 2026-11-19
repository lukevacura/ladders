//! The second streaming engine: a fixed-window left nearest-neighbor profile over `f64`
//! samples that searches only the last `w` samples, in memory that never grows past `O(w)`.
//! Each [`BoundedStream::update`] quantizes, forward-fills and folds one sample into the
//! running window statistics, then offers every neighbor still inside the last `w` samples to
//! [`Nearest`] before returning the row's entry, never to be revisited later.
//!
//! Unlike the growing stream, a neighbor's diagonal cannot simply keep extending forever: once
//! its window would fall outside the last `w` samples, it is never read again. Rather than
//! shifting or evicting diagonals as they age, this engine gives every possible offset `row -
//! neighbor` a fixed slot, sized once from `w` and `m`, and advances each slot in place every
//! row; an offset whose neighbor would already be outside the last `w` samples is never given a
//! slot at all, so nothing to evict.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::capacity::reserve_for;
use crate::checkpoint::{self, CheckpointError};
use crate::columns;
use crate::cross_term::{self, CrossTerms, Offsets};
use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::kernel_neon::{Conversion, RowScalars};
use crate::narrow::{self, Anchor, Path, RouteCounts};
use crate::nearest::{Candidate, ExactDistance, Nearest, winner_entry};
use crate::quantize::{Scale, to_ticks};
use crate::right::{ColumnSink, Right};
use crate::ring::MirroredRing;
use crate::row::{
    ChunkCols, DEFAULT_TILE_LANES, RankRow, RowAcc, WideChunkCols, WideRank, WideRow,
    centered_cross_fast, pass_chunk, pass_chunk_offering, pass_chunk_wide,
};
use crate::sanitize::{ForwardFill, Sanitized};
use crate::spill;
use crate::spill::OutlierScan;
use crate::spill::stream::{Cold, RowView, flag_of, hold, unhold, withheld_indices, withholds};
use crate::stats::{Window, WindowKind, WindowStats};
use crate::znorm::TickWindow;

/// A placeholder window, never read: only the first [`MirroredRing::len`] records in
/// [`BoundedStream::windows`] are ever addressed.
const UNREAD_WINDOW: TickWindow = TickWindow {
    entering: 0,
    leaving: 0,
    stats: Window {
        sum: 0,
        kind: WindowKind::NonFinite,
    },
};

/// One chunk's exact enter, leave and sum columns for the wide pass, gathered from the window
/// ring each row: scratch, not state.
struct WideGather {
    enter: Box<[i64]>,
    leave: Box<[i64]>,
    sum: Box<[i64]>,
}

impl WideGather {
    fn new(lanes: usize) -> Self {
        Self {
            enter: vec![0; lanes].into_boxed_slice(),
            leave: vec![0; lanes].into_boxed_slice(),
            sum: vec![0; lanes].into_boxed_slice(),
        }
    }
}

/// What the newest window contributes to a row, in the arithmetic the stream is running.
#[derive(Clone, Copy)]
enum RowArithmetic {
    Narrow(RowScalars),
    Wide(WideRow),
}

/// Causal, exact left nearest-neighbor matrix profile over the last `w` samples: a window
/// older than that is never a candidate, however long the stream runs.
pub struct BoundedStream {
    m: NonZeroUsize,
    /// `ceil(m/4)`: a neighbor this close or closer is never eligible, for any row.
    exclusion: usize,
    /// `w - m`: the largest `row - neighbor` a window still inside the last `w` samples can
    /// have. Zero when `w == m`, so no row ever has a neighbor.
    max_offset: usize,
    scale: Scale,
    /// How many nearest earlier neighbors each row keeps.
    k: NonZeroUsize,
    fill: ForwardFill<i64>,
    stats: WindowStats,
    /// Windows completed so far; the next one returned is row [`Self::rows`].
    rows: u64,
    /// The last `w` sanitized ticks, oldest first.
    ticks: MirroredRing<i64>,
    /// The last `max_offset + 1` completed windows, oldest first.
    windows: MirroredRing<TickWindow>,
    /// `diagonals[d - 1]` is the pair `(current row, current row - d)`'s diagonal state, for
    /// every offset `d` in `exclusion + 1 ..= max_offset`; an offset at or below the exclusion
    /// zone keeps its slot allocated but never writes or reads it, since it is never eligible
    /// at any row. Its meaning depends on `path`: `m·QT` offsets from `anchor` while `Narrow`,
    /// plain absolute `QT` once `Wide`.
    diagonals: Box<[i64]>,
    /// The tick every live diagonal's narrow state is anchored to; meaningless once `path` is
    /// `Wide`.
    anchor: Anchor,
    /// Which arithmetic `diagonals` currently holds: `Narrow` through the fast kernel, or
    /// `Wide`, the exact i64 fallback, until the buffer fits an anchor again
    /// ([`Self::return_to_narrow_if_fits`]).
    path: Path,
    /// The narrow path's L1-sized `ρ̂` tile and its per-row accumulator: scratch, not state.
    tile: Box<[f64]>,
    acc: RowAcc,
    /// Whether `acc` has reserved everything a row can grow.
    acc_reserved: bool,
    /// The row's reduction, reserved by the first row with neighbors, and its newest result,
    /// reserved in `new`.
    nearest: Nearest,
    entries: Vec<Entry>,
    wide: WideGather,
    /// Every live window's own `(r, bias)` pair ([`columns::r_bias`]) and, while `Narrow`, its
    /// narrowed `enter`/`leave`/`sum` offset from `anchor`, each written once when the window is
    /// born ([`Self::push_window_columns`]) rather than refilled every row: pushed in lockstep
    /// with `windows`, so its ring position matches `windows.as_slice()`'s own position exactly
    /// once the ring is full, and differs only by the constant `windows.capacity() -
    /// windows.len()` during ramp-up. `enter`/`leave`/`sum` stay correct under a later re-anchor
    /// through `MirroredRing::update_live` ([`Self::reanchor_live_diagonals`]); `r`/`bias` never
    /// change and stay current on both paths. `enter`/`leave`/`sum` are meaningless once `path`
    /// is `Wide` (never read again).
    col_enter: MirroredRing<i32>,
    col_leave: MirroredRing<i32>,
    col_sum: MirroredRing<i32>,
    col_r: MirroredRing<f64>,
    col_bias: MirroredRing<f64>,
    /// Whether each tick in `ticks` had a real sample, pushed in lockstep with it (`ticks` alone
    /// cannot say a filled tick from a present one). Neither `ticks` nor a completed window's
    /// kind keeps this once the window closes, but a checkpoint replays it through `WindowStats`
    /// (which exposes no state but `new`/`push`) to rebuild its hidden ring and to prove every
    /// stored window's kind.
    had_tick: MirroredRing<bool>,
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
    cold: Box<Cold>,
    /// Rows below this one may need the cold route: `u64::MAX` on the exact i64 arithmetic (the
    /// only one a variance past `2⁶³` fits in), else [`Self::spill_end`].
    cold_end: u64,
    /// `k` while the next row needs only the hot arithmetic, `0` while it may need the cold
    /// route: one check per row serves both.
    hot_k: usize,
    /// One past the last row a spilled window or a withheld tick is still read on.
    spill_end: u64,
    /// Withheld ticks, position and exact tick, oldest first: those in `ticks` and the one just
    /// before it (the oldest window's leaving tick), for a checkpoint.
    withheld: MirroredRing<(u64, i64)>,
    /// The outlier rule's level and run, which the buffer may no longer hold.
    outliers: OutlierScan,
    /// `ticks` with every withheld tick exact, current as of row [`Self::exact_row`].
    exact: MirroredRing<i64>,
    exact_row: u64,
    /// The newest row's nearest and its exact distance, when [`Winner::Kept`] says they are here.
    kept_result: Box<Option<ExactDistance>>,
    /// Each held window's nearest later neighbors, when the stream looks right.
    right: Option<Right>,
}

impl BoundedStream {
    /// `w` is the number of trailing samples searched; `None` when `w` is less than `m`, since
    /// no window that long could ever stay inside it.
    pub fn new(m: NonZeroUsize, w: NonZeroUsize, scale: Scale) -> Option<Self> {
        Self::with_k(m, w, scale, NonZeroUsize::MIN)
    }

    /// [`Self::new`] keeping the `k` nearest earlier neighbors of each window
    /// ([`Self::update_topk`]); memory stays `O(w + k)`, and a `k` beyond the windows that fit
    /// in `w` reserves only those.
    pub fn with_k(m: NonZeroUsize, w: NonZeroUsize, scale: Scale, k: NonZeroUsize) -> Option<Self> {
        let max_offset = w.get().checked_sub(m.get())?;
        let capacity_windows =
            NonZeroUsize::new(max_offset + 1).expect("max_offset + 1 is never 0");
        let exclusion = crate::exclusion::zone(m.get());
        let neighbors = max_offset.saturating_sub(exclusion);
        Some(Self {
            m,
            exclusion,
            max_offset,
            scale,
            k,
            fill: ForwardFill::default(),
            stats: WindowStats::new(m),
            rows: 0,
            ticks: MirroredRing::new(w, 0),
            windows: MirroredRing::new(capacity_windows, UNREAD_WINDOW),
            diagonals: vec![0i64; max_offset].into_boxed_slice(),
            anchor: Anchor { tick: 0 },
            path: Path::Narrow,
            tile: vec![0.0f64; DEFAULT_TILE_LANES].into_boxed_slice(),
            acc: RowAcc::new(),
            acc_reserved: false,
            nearest: Nearest::with_k(&UNREAD_WINDOW.stats, m.get(), k, 0),
            entries: Vec::with_capacity(reserve_for(k.get(), neighbors)),
            wide: WideGather::new(capacity_windows.get().min(DEFAULT_TILE_LANES)),
            col_enter: MirroredRing::new(capacity_windows, 0i32),
            col_leave: MirroredRing::new(capacity_windows, 0i32),
            col_sum: MirroredRing::new(capacity_windows, 0i32),
            col_r: MirroredRing::new(capacity_windows, 0.0f64),
            col_bias: MirroredRing::new(capacity_windows, 0.0f64),
            had_tick: MirroredRing::new(w, false),
            seen_real_sample: false,
            routes: RouteCounts::default(),
            cold: Box::new(Cold::new(m, k, Some(capacity_windows.get()))),
            cold_end: 0,
            hot_k: k.get(),
            spill_end: 0,
            withheld: MirroredRing::new(w.saturating_add(1), (0, 0)),
            outliers: OutlierScan::default(),
            exact: MirroredRing::new(w, 0),
            exact_row: u64::MAX,
            kept_result: Box::new(None),
            right: None,
        })
    }

    /// [`Self::with_k`] that also keeps, for each window it holds (the last `w − m + 1`), its `k`
    /// nearest *later* neighbors and its `k` nearest over both sides, the earlier ones as the
    /// stream found them ([`Self::right_neighbors_into`], [`Self::two_sided_neighbors_into`]).
    /// This looks ahead of a window: an entry for a window can change when later samples arrive,
    /// so nothing causal may read it. The stream's own entries are those of [`Self::with_k`], bit
    /// for bit. `None` when `w` is less than `m`, or when the lists cannot be allocated.
    pub fn with_right_side(
        m: NonZeroUsize,
        w: NonZeroUsize,
        scale: Scale,
        k: NonZeroUsize,
    ) -> Option<Self> {
        let mut stream = Self::with_k(m, w, scale, k)?;
        let windows = stream.max_offset + 1;
        let reach = stream.max_offset.saturating_sub(stream.exclusion);
        stream.right = Some(Right::held(m, k, windows, reach)?);
        Some(stream)
    }

    /// Whether the stream keeps the right side ([`Self::with_right_side`]).
    pub fn looks_right(&self) -> bool {
        self.right.is_some()
    }

    /// The windows whose right and two-sided neighbors the stream holds: the last `w − m + 1`, or
    /// none when it does not look right.
    pub fn right_windows(&self) -> core::ops::Range<u64> {
        self.right.as_ref().map_or(0..0, Right::held_windows)
    }

    /// Window `window`'s `k` nearest later neighbors so far, nearest first under the exact order
    /// (ties to the smaller index), in `out` (cleared first); none for a window the stream no
    /// longer holds or one that does not look right.
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

    /// Window `window`'s `k` nearest neighbors over both sides, as
    /// [`Self::right_neighbors_into`] but over the earlier ones too: those the window was ranked
    /// against when it arrived, though they may since have left.
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

    /// Which arithmetic the stream is currently running: `Narrow`'s `m·QT` offsets
    /// through the fast kernel, or `Wide`, the exact i64 fallback until the buffer fits an anchor again.
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
            Some(right) => checkpoint::wrap_right(checkpoint::KIND_BOUNDED, &plain, |buf| {
                checkpoint::put_columns(
                    buf,
                    right.held_windows().map(|w| right.column(w)).collect(),
                );
            }),
        }
    }

    fn checkpoint_plain(&self) -> Vec<u8> {
        let base = self.ticks_seen() - self.ticks.len() as u64;
        let first_row = self.rows - self.windows.len() as u64;
        let spill = checkpoint::Spill {
            constant: (first_row..self.rows)
                .map(|row| self.cold.constant_too_wide(row))
                .collect(),
        };
        let withheld = self.withheld.as_slice();
        let live = &withheld[withheld.partition_point(|&(position, _)| position + 1 < base)..];
        if live.is_empty() {
            let diagonals = self.absolute_diagonals();
            return checkpoint::write_bounded(checkpoint::BoundedSnapshot {
                m: self.m,
                k: self.k,
                max_offset: self.max_offset,
                scale: self.scale,
                seen_real_sample: self.seen_real_sample,
                rows: self.rows,
                ticks: self.ticks.as_slice(),
                windows: &self.exact_windows(live, base, None),
                diagonals: &diagonals,
                had_tick: self.had_tick.as_slice(),
                spill,
            });
        }
        // One tick before the buffer, so a gap opening it holds a tick withheld just before.
        let before = live.first().map_or(
            0,
            |&(position, tick)| {
                if position < base { tick } else { 0 }
            },
        );
        let mut ticks: Vec<i64> = core::iter::once(before)
            .chain(self.ticks.as_slice().iter().copied())
            .collect();
        let mut had: Vec<bool> = core::iter::once(true)
            .chain(self.had_tick.as_slice().iter().copied())
            .collect();
        let shifted: Vec<(usize, i64)> = live
            .iter()
            .map(|&(position, tick)| ((position + 1 - base) as usize, tick))
            .collect();
        unhold(&mut ticks, &mut had, &shifted);
        let (ticks, had) = (ticks.split_off(1), had.split_off(1));
        let replayed = self.exact_windows(live, base, Some((&ticks, &had)));
        let m = self.m.get();
        let mut diagonals = self.diagonals.to_vec();
        diagonals.fill(0);
        if let Some(live_hi) = self.live_offset_hi() {
            let row_start = ticks.len() - m;
            for d in (self.exclusion as u64 + 1)..=live_hi {
                let neighbor = row_start - d as usize;
                diagonals[(self.max_offset as u64 - d) as usize] = cross_term::wide(
                    &ticks[row_start..row_start + m],
                    &ticks[neighbor..neighbor + m],
                );
            }
        }
        checkpoint::write_bounded(checkpoint::BoundedSnapshot {
            m: self.m,
            k: self.k,
            max_offset: self.max_offset,
            scale: self.scale,
            seen_real_sample: self.seen_real_sample,
            rows: self.rows,
            ticks: &ticks,
            windows: &replayed,
            diagonals: &diagonals,
            had_tick: &had,
            spill,
        })
    }

    /// The live windows as the samples make them: the ones the hot arithmetic holds, each with
    /// a withheld leaving tick (`live`, from position `base − 1` on) made exact, and, given the
    /// exact `ticks` and presence `had` of the buffer, every spilled window replayed from them
    /// under its flag.
    fn exact_windows(
        &self,
        live: &[(u64, i64)],
        base: u64,
        replay: Option<(&[i64], &[bool])>,
    ) -> Vec<TickWindow> {
        let first_row = self.rows - self.windows.len() as u64;
        let mut windows = self.windows.as_slice().to_vec();
        if let Some(&(position, exact)) = live.first()
            && position + 1 == base
            && let Some(oldest) = windows.first_mut()
            && first_row > 0
            && self.max_offset > 0
        {
            oldest.leaving = exact;
        }
        let Some((ticks, had)) = replay else {
            return windows;
        };
        let m = self.m.get();
        let mut made = Vec::with_capacity(windows.len());
        checkpoint::replay_stats(self.m, ticks, had, |_, window| {
            made.push(window);
            true
        });
        for (at, window) in windows.iter_mut().enumerate() {
            let start = (first_row + at as u64 - base) as usize;
            window.entering = ticks[start + m - 1];
            if start > 0 {
                window.leaving = ticks[start - 1];
            }
            let row = first_row + at as u64;
            if self.cold.spilled(row) {
                window.stats = made[start].under(Some(self.cold.spilled_constant(row)));
            } else if made[start].kind == WindowKind::NonFinite {
                window.stats = made[start];
            }
        }
        windows
    }

    /// `diagonals` with every live one as its absolute `QT`, whichever arithmetic the stream is
    /// running.
    fn absolute_diagonals(&self) -> Cow<'_, [i64]> {
        if self.path == Path::Wide {
            return Cow::Borrowed(&self.diagonals);
        }
        Cow::Owned(self.live_diagonals_mapped(|qt_m, sum_a, sum_b| {
            narrow::absolute_qt(self.anchor, self.m, qt_m, sum_a, sum_b)
        }))
    }

    /// A copy of `diagonals` with every live one replaced by `convert(diagonal, newest window's
    /// sum, that diagonal's neighbor window's sum)`.
    fn live_diagonals_mapped(&self, convert: impl Fn(i64, i64, i64) -> i64) -> Vec<i64> {
        let mut mapped = self.diagonals.to_vec();
        let exclusion = self.exclusion as u64;
        let Some(live_hi) = self.live_offset_hi().filter(|&hi| hi > exclusion) else {
            return mapped;
        };
        let windows = self.windows.as_slice();
        let len = windows.len() as u64;
        let newest_sum = windows[(len - 1) as usize].stats.sum;
        for d in (exclusion + 1)..=live_hi {
            let slot = (self.max_offset as u64 - d) as usize;
            let neighbor_sum = windows[(len - 1 - d) as usize].stats.sum;
            mapped[slot] = convert(self.diagonals[slot], newest_sum, neighbor_sum);
        }
        mapped
    }

    /// Rebuilds a stream from [`Self::checkpoint`]'s own bytes: every ring is rebuilt fresh
    /// through its own public constructor (never by poking a private field), including the
    /// hidden state behind `stats`/`fill`. Any truncated or garbled input is rejected with a
    /// [`CheckpointError`] rather than panicking.
    pub fn restore(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let Some(mut wrapped) = checkpoint::open_right(bytes, checkpoint::KIND_BOUNDED)? else {
            return Self::restore_plain(bytes);
        };
        let mut stream = Self::restore_plain(wrapped.inner)?;
        let held = stream.windows.as_slice();
        let columns = checkpoint::get_columns(&mut wrapped.reader, stream.k.get(), held.len())?;
        wrapped.reader.finish()?;
        let windows = stream.max_offset + 1;
        let reach = stream.max_offset.saturating_sub(stream.exclusion);
        let mut right =
            Right::held(stream.m, stream.k, windows, reach).ok_or(CheckpointError::Inconsistent)?;
        let first = stream.rows - held.len() as u64;
        for (at, column) in columns.into_iter().enumerate() {
            let row = first + at as u64;
            right.begin_row(row, held[at].stats);
            if !right.load(row, column) {
                return Err(CheckpointError::Inconsistent);
            }
        }
        stream.right = Some(right);
        Ok(stream)
    }

    fn restore_plain(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let mut parts: checkpoint::BoundedParts = checkpoint::read_bounded(bytes)?;
        let samples = Self::withhold_restored(&mut parts)?;
        let exclusion = crate::exclusion::zone(parts.m.get());
        let w = parts
            .max_offset
            .checked_add(parts.m.get())
            .ok_or(CheckpointError::Inconsistent)?;
        let w = NonZeroUsize::new(w).ok_or(CheckpointError::Inconsistent)?;
        let capacity_windows = parts
            .max_offset
            .checked_add(1)
            .and_then(NonZeroUsize::new)
            .ok_or(CheckpointError::Inconsistent)?;

        let mut ticks = MirroredRing::new(w, 0i64);
        for &t in &parts.ticks {
            ticks.push(t);
        }
        let mut windows = MirroredRing::new(capacity_windows, UNREAD_WINDOW);
        for &win in &parts.windows {
            windows.push(win);
        }
        let mut had_tick = MirroredRing::new(w, false);
        for &had in &parts.had_tick {
            had_tick.push(had);
        }

        // `WindowStats` exposes no state but `new`/`push`: replaying every live tick with its
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

        let neighbors = parts.max_offset.saturating_sub(exclusion);
        let mut stream = Self {
            m: parts.m,
            exclusion,
            max_offset: parts.max_offset,
            scale: parts.scale,
            k: parts.k,
            fill,
            stats,
            rows: parts.rows,
            ticks,
            windows,
            diagonals: parts.diagonals.into_boxed_slice(),
            anchor: Anchor { tick: 0 },
            path: Path::Wide,
            tile: vec![0.0f64; DEFAULT_TILE_LANES].into_boxed_slice(),
            acc: RowAcc::new(),
            acc_reserved: false,
            nearest: Nearest::with_k(&UNREAD_WINDOW.stats, parts.m.get(), parts.k, 0),
            entries: Vec::with_capacity(reserve_for(parts.k.get(), neighbors)),
            wide: WideGather::new(capacity_windows.get().min(DEFAULT_TILE_LANES)),
            col_enter: MirroredRing::new(capacity_windows, 0i32),
            col_leave: MirroredRing::new(capacity_windows, 0i32),
            col_sum: MirroredRing::new(capacity_windows, 0i32),
            col_r: MirroredRing::new(capacity_windows, 0.0f64),
            col_bias: MirroredRing::new(capacity_windows, 0.0f64),
            had_tick,
            seen_real_sample: parts.seen_real_sample,
            routes: RouteCounts::default(),
            cold: Box::new(Cold::new(parts.m, parts.k, Some(capacity_windows.get()))),
            cold_end: 0,
            hot_k: parts.k.get(),
            spill_end: 0,
            withheld: MirroredRing::new(w.saturating_add(1), (0, 0)),
            outliers: OutlierScan::default(),
            exact: MirroredRing::new(w, 0),
            exact_row: u64::MAX,
            kept_result: Box::new(None),
            right: None,
        };
        stream.choose_route();
        stream.restore_cold(&samples)?;
        Ok(stream)
    }

    /// Turns `parts`, which hold the samples as they came, into what the hot arithmetic holds
    /// once the stream withholds what [`withheld_indices`] says of the ticks held (the oldest
    /// one's level is the first present tick held); returns the samples as they came.
    fn withhold_restored(
        parts: &mut checkpoint::BoundedParts,
    ) -> Result<Restored, CheckpointError> {
        let bad = CheckpointError::Inconsistent;
        let m = parts.m.get();
        let ticks = parts.ticks.clone();
        let had = parts.had_tick.clone();
        let mut made = Vec::with_capacity(parts.windows.len());
        checkpoint::replay_stats(parts.m, &ticks, &had, |_, window| {
            made.push(window);
            true
        })
        .ok_or(bad)?;
        let flags: Vec<Option<bool>> = parts
            .windows
            .iter()
            .zip(&made)
            .enumerate()
            .map(|(at, (stored, &made))| match stored.stats.kind {
                WindowKind::NonFinite => parts
                    .spill
                    .constant
                    .get(at)
                    .copied()
                    .unwrap_or(false)
                    .then_some(true),
                _ => flag_of(stored.stats, made),
            })
            .collect();
        let total = if parts.rows == 0 {
            ticks.len() as u64
        } else {
            parts.rows + m as u64 - 1
        };
        let base = total - ticks.len() as u64;
        let (withheld, outliers) = if spill::stream::available(m) {
            withheld_indices(m, &ticks, &had)
        } else {
            (Vec::new(), OutlierScan::default())
        };
        if !withheld.is_empty() {
            hold(&mut parts.ticks, &mut parts.had_tick, &withheld);
            let mut hot = Vec::with_capacity(parts.windows.len());
            checkpoint::replay_stats(parts.m, &parts.ticks, &parts.had_tick, |_, window| {
                hot.push(window);
                true
            })
            .ok_or(bad)?;
            for (at, window) in parts.windows.iter_mut().enumerate() {
                window.stats = hot[at].under(flags[at]);
                window.entering = parts.ticks[at + m - 1];
                if at > 0 {
                    window.leaving = parts.ticks[at - 1];
                }
            }
            let exclusion = crate::exclusion::zone(m) as u64;
            let live_hi = parts
                .rows
                .checked_sub(1)
                .map(|row| row.min(parts.max_offset as u64));
            let row_start = parts.ticks.len().saturating_sub(m);
            for (slot, diagonal) in parts.diagonals.iter_mut().enumerate() {
                let d = (parts.max_offset - slot) as u64;
                if live_hi.is_some_and(|hi| d > exclusion && d <= hi) {
                    let neighbor = row_start - d as usize;
                    *diagonal = cross_term::wide(
                        &parts.ticks[row_start..row_start + m],
                        &parts.ticks[neighbor..neighbor + m],
                    );
                }
            }
        }
        Ok(Restored {
            withheld: withheld
                .iter()
                .map(|&at| (base + at as u64, ticks[at]))
                .collect(),
            outliers,
            ticks,
            had,
            flags,
            base,
        })
    }

    /// Rebuilds the cold route's state from the samples as they came: every live window
    /// classified again, as when it was born.
    fn restore_cold(&mut self, samples: &Restored) -> Result<(), CheckpointError> {
        self.outliers = samples.outliers;
        for &withheld in &samples.withheld {
            self.withheld.push(withheld);
            self.spill_end = self.spill_end.max(withheld.0 + 1);
        }
        let m = self.m.get() as u64;
        let base = samples.base;
        let gap = |position: u64| !samples.had[(position - base) as usize];
        for (at, window) in self.windows.as_slice().iter().enumerate() {
            let row = base + at as u64;
            let view = RowView {
                row,
                exact: &samples.ticks,
                base,
                oldest: row,
                hot_window: window.stats,
                flag: samples.flags[at],
                holds_withheld: samples
                    .withheld
                    .iter()
                    .any(|&(position, _)| (row..row + m).contains(&position)),
                gap: &gap,
                hot: |_: u64| window.stats,
            };
            let spilled = self.cold.classify(&view);
            if samples.flags[at] == Some(true)
                && window.stats.kind == WindowKind::NonFinite
                && !spilled
            {
                return Err(CheckpointError::Inconsistent);
            }
            if spilled {
                self.spill_end = self.spill_end.max(row + self.max_offset as u64 + 1);
            }
        }
        self.cold_end = if self.path == Path::Wide {
            u64::MAX
        } else {
            self.spill_end
        };
        self.hot_k = self.hot_k_for_rows();
        Ok(())
    }

    /// Sets the route of a stream holding absolute diagonals the way a fresh stream chooses it:
    /// narrow, with a centered anchor and its columns and diagonals rebuilt, when one anchor
    /// covers every tick the narrow arithmetic reads; otherwise the exact i64 arithmetic.
    fn choose_route(&mut self) {
        let anchor = match self.tick_span(None) {
            Some((lo, hi)) => Anchor::spanning(lo, hi, self.m, self.rows),
            None => Some(Anchor { tick: 0 }),
        };
        if let Some(anchor) = anchor {
            let m = self.m;
            let diagonals = self.live_diagonals_mapped(|qt, sum_a, sum_b| {
                narrow::narrow_qt_m(anchor, m, qt, sum_a, sum_b)
            });
            self.diagonals = diagonals.into_boxed_slice();
            self.anchor = anchor;
            self.path = Path::Narrow;
        } else {
            self.path = Path::Wide;
        }
        let windows: Vec<TickWindow> = self.windows.as_slice().to_vec();
        let first_row = self.rows - windows.len() as u64;
        for (pos, window) in windows.into_iter().enumerate() {
            let has_leaving = first_row + pos as u64 > 0 && self.max_offset > 0;
            self.push_window_columns(window, has_leaving);
        }
    }

    /// The lowest and highest tick the narrow arithmetic reads: the ring's ticks, the oldest live
    /// window's `leaving` once window 0 is gone, and `incoming`, a tick about to join; and the
    /// outlier rule's level, which the ring may no longer hold, so every anchor covers it. `None`
    /// when there are none.
    fn tick_span(&self, incoming: Option<i64>) -> Option<(i64, i64)> {
        let mut span = incoming.map(|tick| (tick, tick));
        let mut widen = |tick: i64| {
            span = Some(span.map_or((tick, tick), |(lo, hi)| (lo.min(tick), hi.max(tick))));
        };
        for &t in self.ticks.as_slice() {
            widen(t);
        }
        // The oldest live window's own `leaving` tick sits one position before anything
        // `self.ticks` still retains, and `narrow::advance`'s leaving-side term reads it every
        // row that window is the offset-`max_offset` neighbor — *unless* that oldest window is
        // window 0 itself (ramp-up hasn't pushed it out of the ring yet), whose own `leaving` is
        // an unread placeholder (`0`, never a real tick: it has no predecessor), since window 0
        // is always the fresh diagonal (`cross_term::wide`, never `advance`) for every row it is
        // ever a neighbor on. Folding that placeholder in anyway, at a high price level, makes
        // the span look thousands of ticks wide and forces an immediate, permanent, needless
        // switch to Wide; excluding it is exact, not just a tighter approximation, because that
        // value is never read by anything live.
        // With no neighbor slots (`w == m`) every `leaving` is that placeholder.
        let window_0_evicted = self.rows > self.max_offset as u64 + 1 && self.max_offset > 0;
        if window_0_evicted && let Some(oldest) = self.windows.as_slice().first() {
            widen(oldest.leaving);
        }
        if let Some(level) = self.outliers.level() {
            widen(level);
        }
        span
    }

    /// Test-only: the anchor `diagonals` is currently offset from, so a test can count distinct
    /// re-anchors (by watching this change) without `maintain_narrow_range` needing its own
    /// counter field.
    #[cfg(test)]
    fn anchor(&self) -> Anchor {
        self.anchor
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
        let entry = match winner {
            Winner::One(winner) => winner.map_or(Entry::NONE, |w| winner_entry(self.m.get(), w)),
            Winner::Kept => self.kept_result.map_or(Entry::NONE, |kept| kept.entry()),
        };
        Some((row, entry))
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
        let exact = match winner {
            Winner::One(winner) => winner.map(|w| ExactDistance::of_winner(self.m.get(), w)),
            Winner::Kept => *self.kept_result,
        };
        Some((row, exact))
    }

    /// `None` until the first window completes; then the newest window's row and its `k` nearest
    /// earlier neighbors inside the last `w` samples, nearest first under the exact order with
    /// ties to the smaller index: fewer when fewer exist, none for a window without an eligible
    /// neighbor. The slice holds until the next update.
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
            let nearest = match winner {
                Winner::One(winner) => {
                    winner.map_or(Entry::NONE, |w| winner_entry(self.m.get(), w))
                }
                Winner::Kept => self.kept_result.map_or(Entry::NONE, |kept| kept.entry()),
            };
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
            Sanitized::Leading => (0, false),
        };
        self.had_tick.push(had_tick);
        // Checked (and, if needed, re-anchored or switched) before `tick` joins the ring: so
        // long as any conversion here needs every live diagonal's full raw-tick span, that
        // span is still exactly what it was after the last row, undisturbed by this tick's
        // own push (which could otherwise evict the oldest live window's first tick in the
        // same call that needs it).
        if let Some(level) = self.maintain_narrow_range(tick) {
            (tick, had_tick) = (level, false);
        }
        if had_tick {
            self.outliers.keep(tick);
        }
        self.seen_real_sample |= had_tick;
        self.routes.count_update(self.path);
        self.ticks.push(tick);
        let window = self.stats.push(tick, had_tick)?.under(constant);

        let row = self.rows;
        let m = self.m.get();
        let ticks = self.ticks.as_slice();
        // The tick this window drops as it slides from `row - 1`: still in `ticks` exactly
        // when some later row could still read this window as a neighbor (`max_offset >= 1`);
        // otherwise it is never read, and the placeholder is never observed. Also false on
        // every row, not only row 0, whenever `w == m` (`max_offset == 0`): `ticks` never
        // holds more than `m` entries then, so there is never a dropped tick to report.
        let has_leaving = ticks.len() > m;
        let leaving = if has_leaving {
            ticks[ticks.len() - 1 - m]
        } else {
            0
        };
        let row_window = TickWindow {
            entering: tick,
            leaving,
            stats: window,
        };
        self.windows.push(row_window);
        self.push_window_columns(row_window, has_leaving);
        if let Some(right) = &mut self.right {
            right.begin_row(row, row_window.stats);
        }

        let winner = match self.hot_k {
            1 => Winner::One(self.rank_hot_one(row, row_window)),
            0 => return Some(self.rank_watched(row, row_window, constant)),
            _ => {
                let nearest = self.rank_hot_k(row, row_window);
                *self.kept_result = nearest;
                Winner::Kept
            }
        };
        self.rows += 1;
        let row_index = SampleIndex::new(row).expect("row is below NONE's sentinel");
        Some((row_index, winner))
    }

    /// The hot arithmetic's nearest neighbor of `row` (its window `row_window`), not yet reported.
    #[inline(always)]
    fn rank_hot_one(&mut self, row: u64, row_window: TickWindow) -> Option<Candidate> {
        let (window, m) = (row_window.stats, self.m.get());
        let mut nearest = Nearest::new(&window, m);
        self.offer_row(row, row_window, m, &mut nearest);
        if let Some(right) = &mut self.right {
            right.offer_left(nearest.kept());
        }
        nearest.finish_winner()
    }

    /// The hot arithmetic's `k` nearest neighbors of `row` in `self.entries`, and the nearest.
    #[inline(always)]
    fn rank_hot_k(&mut self, row: u64, row_window: TickWindow) -> Option<ExactDistance> {
        let (window, m) = (row_window.stats, self.m.get());
        let neighbors = self.max_offset.saturating_sub(self.exclusion);
        let mut kept = core::mem::replace(&mut self.nearest, Nearest::new(&window, m));
        kept.restart(&window, m, self.k, neighbors);
        self.offer_row(row, row_window, m, &mut kept);
        if let Some(right) = &mut self.right {
            right.offer_left(kept.kept());
        }
        let nearest = kept.drain_k(&mut self.entries);
        self.nearest = kept;
        nearest
    }

    /// `k`, or `0` while rows may need the cold route ([`Self::hot_k`]).
    fn hot_k_for_rows(&self) -> usize {
        if self.rows < self.cold_end {
            0
        } else {
            self.k.get()
        }
    }

    /// [`Self::rank_hot`] followed by the cold route ([`Self::rank_cold`]), for a row that may
    /// need it.
    #[cold]
    #[inline(never)]
    fn rank_watched(
        &mut self,
        row: u64,
        row_window: TickWindow,
        constant: Option<bool>,
    ) -> (SampleIndex, Winner) {
        let m = self.m.get();
        let hot = if self.k == NonZeroUsize::MIN {
            self.rank_hot_one(row, row_window)
                .map(|w| ExactDistance::of_winner(m, w))
        } else {
            self.rank_hot_k(row, row_window)
        };
        *self.kept_result = self.rank_cold(row, row_window.stats, constant, hot).1;
        self.rows += 1;
        self.hot_k = self.hot_k_for_rows();
        let row_index = SampleIndex::new(row).expect("row is below NONE's sentinel");
        (row_index, Winner::Kept)
    }

    /// The position of the next tick.
    fn ticks_seen(&self) -> u64 {
        if self.rows == 0 {
            self.ticks.len() as u64
        } else {
            self.rows + self.m.get() as u64 - 1
        }
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
        self.cold.available()
            && self.seen_real_sample
            && self.had_tick.as_slice().last() == Some(&true)
    }

    /// Withholds `tick`, the newest sample, from the hot arithmetic, which holds the newest tick
    /// it has in its place (returned), as for a gap.
    #[cold]
    #[inline(never)]
    fn withhold(&mut self, tick: i64) -> i64 {
        let level = *self.ticks.as_slice().last().expect("a tick precedes");
        let position = self.ticks_seen();
        self.had_tick.set_newest(false);
        self.withheld.push((position, tick));
        self.outliers.withhold();
        self.routes.withheld_ticks += 1;
        self.watch(position + 1);
        level
    }

    /// Makes the rows below `end` run the cold route; clears the spilled bitmap first when the
    /// next row is not watched yet (no spilled window is live then).
    fn watch(&mut self, end: u64) {
        if self.cold_end <= self.rows {
            self.cold.clear();
        }
        self.cold_end = self.cold_end.max(end);
        if end != u64::MAX {
            self.spill_end = self.spill_end.max(end);
        }
        self.hot_k = self.hot_k_for_rows();
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
        let m = self.m.get() as u64;
        let newest = row + m - 1;
        let base = newest + 1 - self.ticks.len() as u64;
        let withheld = self.withheld.as_slice();
        let live = &withheld[withheld.partition_point(|&(position, _)| position < base)..];
        let exact: &[i64] = if live.is_empty() {
            self.ticks.as_slice()
        } else {
            if self.exact_row.checked_add(1) == Some(row) {
                let newest_tick = match live.last() {
                    Some(&(position, tick)) if position == newest => tick,
                    _ => *self.ticks.as_slice().last().expect("the row's ticks"),
                };
                self.exact.push(newest_tick);
            } else {
                let mut patches = live.iter().peekable();
                for (at, &tick) in self.ticks.as_slice().iter().enumerate() {
                    let exact = match patches.peek() {
                        Some(&&(position, withheld)) if position == base + at as u64 => {
                            patches.next();
                            withheld
                        }
                        _ => tick,
                    };
                    self.exact.push(exact);
                }
            }
            self.exact_row = row;
            let held = self.exact.as_slice();
            &held[held.len() - self.ticks.len()..]
        };
        let had = self.had_tick.as_slice();
        let gap = |position: u64| {
            !had[(position - base) as usize]
                && live
                    .binary_search_by_key(&position, |&(withheld, _)| withheld)
                    .is_err()
        };
        let windows = self.windows.as_slice();
        let len = windows.len() as u64;
        let view = RowView {
            row,
            exact,
            base,
            oldest: row.saturating_sub(self.max_offset as u64),
            hot_window: window,
            flag,
            holds_withheld: live.last().is_some_and(|&(position, _)| position >= row),
            gap: &gap,
            hot: |neighbor: u64| windows[(len - 1 - (row - neighbor)) as usize].stats,
        };
        let Some(result) = self
            .cold
            .rank(&view, &mut self.entries, hot, self.right.as_mut())
        else {
            return (hot.map_or(Entry::NONE, |hot| hot.entry()), hot);
        };
        if result.spilled {
            self.spill_end = self.spill_end.max(row + self.max_offset as u64 + 1);
            self.cold_end = self.cold_end.max(self.spill_end);
        }
        self.routes.cold_rows += u64::from(result.ranked);
        (result.entry, result.exact)
    }

    /// Pushes `row_window`'s own columns, once: `(r, bias)` always, and, while `Narrow`, its
    /// narrowed ones against `self.anchor` as it stands right now, already known to fit it
    /// (`maintain_narrow_range` checked this tick before it joined `self.ticks`). Once `Wide`,
    /// `self.anchor` is frozen for good, so nothing is narrowed against it again.
    /// `row_window.leaving` is a placeholder (`0`, never a real tick, never read) whenever
    /// `has_leaving` is false: narrowing it through [`narrow::tick_offset`] could fail the
    /// anchor's bound at a high price level, so a fixed dummy goes in instead.
    fn push_window_columns(&mut self, row_window: TickWindow, has_leaving: bool) {
        let (r, bias) = columns::r_bias(&row_window.stats);
        self.col_r.push(r);
        self.col_bias.push(bias);
        if self.path == Path::Wide {
            self.col_sum.push(0);
            self.col_enter.push(0);
            self.col_leave.push(0);
            return;
        }
        let anchor = self.anchor;
        let m = self.m;
        self.col_sum
            .push(narrow::sum_offset(anchor, m, row_window.stats.sum));
        self.col_enter
            .push(narrow::tick_offset(anchor, m, row_window.entering));
        let leave = if has_leaving {
            narrow::tick_offset(anchor, m, row_window.leaving)
        } else {
            0i32
        };
        self.col_leave.push(leave);
    }

    /// Advances every tracked offset `exclusion + 1 ..= min(row, max_offset)` by one step and
    /// offers its neighbor to `nearest`. The offset born this row (`d == row`) comes from a
    /// direct dot product against the oldest window still held; every other one continues from
    /// its own slot's value at the previous row, along the same diagonal. No slot ever moves
    /// and none is ever dropped: an offset beyond `max_offset` never gets a slot to begin with,
    /// since its window has already left the last `w` samples.
    ///
    /// The row is cut into tile-sized chunks of ring slots (oldest neighbor first, so indices
    /// ascend). While `Narrow` each chunk runs through [`pass_chunk`] on `m·QT` offsets from
    /// `self.anchor`, reading each neighbor's columns straight off `self.col_*`; once `Wide` it
    /// runs through [`pass_chunk_wide`] on absolute `QT`, with the chunk's exact enter, leave and
    /// sum gathered from the window ring.
    fn offer_row(&mut self, row: u64, row_window: TickWindow, m: usize, nearest: &mut Nearest) {
        let exclusion = self.exclusion as u64;
        let limit = row.min(self.max_offset as u64);
        if limit <= exclusion {
            return;
        }
        let max_offset = self.max_offset as u64;
        // Ring position (oldest neighbor first): slot `d` maps to is `max_offset - d`, so as
        // `d` ascends over `(exclusion + 1)..=limit`, slot descends over `lo..hi`, reversed.
        let lo = (max_offset - limit) as usize;
        let hi = (max_offset - exclusion) as usize;
        let anchor = self.anchor;
        let m_nz = self.m;
        // `col_*` is pushed in lockstep with `windows` (same capacity), so its ring position
        // matches a slot by this same constant shift every row (zero once the ring is full).
        let skew = self.windows.capacity() - self.windows.len();
        let arithmetic = match self.path {
            Path::Narrow => RowArithmetic::Narrow(RowScalars {
                me: narrow::row_scaled_offset(anchor, m_nz, row_window.entering),
                ml: narrow::row_scaled_offset(anchor, m_nz, row_window.leaving),
                si: narrow::sum_offset(anchor, m_nz, row_window.stats.sum),
                ri: match row_window.stats.kind {
                    WindowKind::Valid { inv_sqrt, .. } => inv_sqrt,
                    _ => 0.0,
                },
                conversion: Conversion::FULL_RANGE,
            }),
            Path::Wide => RowArithmetic::Wide(WideRow {
                enter: row_window.entering,
                leave: row_window.leaving,
                sum: row_window.stats.sum,
            }),
        };

        let windows = self.windows.as_slice();
        let len = windows.len() as u64;
        // The neighbor window at ring slot `slot`: offset `d = max_offset - slot` back.
        let window_at = |slot: usize| &windows[(len - 1 - (max_offset - slot as u64)) as usize];
        let (r, bias) = (self.col_r.as_slice(), self.col_bias.as_slice());

        self.acc.begin(
            &row_window.stats,
            self.k,
            self.max_offset.saturating_sub(self.exclusion),
        );
        if !self.acc_reserved {
            // The bound is fixed, so the first row with neighbors reserves for every later row.
            self.acc.reserve_worst_case(self.tile.len());
            self.acc_reserved = true;
        }
        let mut continuing_start = lo;
        if limit == row {
            // `row - d == 0`: the oldest neighbor of all.
            let q = lo - skew;
            let ticks = self.ticks.as_slice();
            let oldest = window_at(lo);
            let neighbor = |_: usize| oldest.stats;
            match arithmetic {
                RowArithmetic::Narrow(scalars) => {
                    let side = Offsets {
                        ticks,
                        anchor,
                        m: m_nz,
                    };
                    let qt0 = CrossTerms::of(side, m).at(ticks.len() - m, 0);
                    self.diagonals[lo] = qt0;
                    let started = self.acc.start_lane(
                        qt0,
                        &ChunkCols {
                            enter: &[],
                            leave: &[],
                            sum: &self.col_sum.as_slice()[q..q + 1],
                            r: &r[q..q + 1],
                            bias: &bias[q..q + 1],
                            first: 0,
                            neighbor: &neighbor,
                        },
                        scalars,
                    );
                    if let (Some(sink), Some(lane)) = (self.right.as_mut(), started) {
                        sink.offer_lane::<RankRow>(0, self.acc.id(row as usize), lane, || {
                            crate::columns::variance(&neighbor(0))
                        });
                    }
                }
                RowArithmetic::Wide(scalars) => {
                    let qt0 = CrossTerms::of(ticks, m).at(ticks.len() - m, 0);
                    self.diagonals[lo] = qt0;
                    let started = self.acc.start_lane_wide(
                        qt0,
                        &WideChunkCols {
                            enter: &[],
                            leave: &[],
                            sum: &[oldest.stats.sum],
                            r: &r[q..q + 1],
                            bias: &bias[q..q + 1],
                            first: 0,
                            neighbor: &neighbor,
                        },
                        scalars,
                        m,
                    );
                    if let (Some(sink), Some(lane)) = (self.right.as_mut(), started) {
                        sink.offer_lane::<WideRank>(0, self.acc.id(row as usize), lane, || {
                            crate::columns::variance(&neighbor(0))
                        });
                    }
                }
            }
            continuing_start = lo + 1;
        }
        let mut start = continuing_start;
        while start < hi {
            let end = (start + self.tile.len()).min(hi);
            let (p, q) = (start - skew, end - skew);
            let neighbor = |lane: usize| window_at(start + lane).stats;
            let first = (row - (max_offset - start as u64)) as usize;
            if crate::route::resume_due(row as usize) {
                let ticks = self.ticks.as_slice();
                let row_start = ticks.len() - m;
                let first_start = row_start.checked_sub((max_offset - start as u64) as usize);
                let states = &mut self.diagonals[start..end];
                match (arithmetic, first_start) {
                    (_, None) => {}
                    (RowArithmetic::Narrow(_), Some(first_start)) => {
                        let side = Offsets {
                            ticks,
                            anchor,
                            m: m_nz,
                        };
                        CrossTerms::of(side, m).force_resume(
                            row as usize,
                            (row_start, first_start),
                            states,
                        );
                    }
                    (RowArithmetic::Wide(_), Some(first_start)) => {
                        CrossTerms::of(ticks, m).force_resume(
                            row as usize,
                            (row_start, first_start),
                            states,
                        );
                    }
                }
            }
            let visit =
                crate::batch::survivor::stream_visits(row as usize, first..first + (end - start));
            debug_assert!(visit, "a stream visits every block");
            match arithmetic {
                RowArithmetic::Narrow(scalars) => {
                    let cols = ChunkCols {
                        enter: &self.col_enter.as_slice()[p..q],
                        leave: &self.col_leave.as_slice()[p..q],
                        sum: &self.col_sum.as_slice()[p..q],
                        r: &r[p..q],
                        bias: &bias[p..q],
                        first,
                        neighbor: &neighbor,
                    };
                    match self.right.as_mut() {
                        None => pass_chunk(
                            &mut self.diagonals[start..end],
                            cols,
                            scalars,
                            &mut self.acc,
                            &mut self.tile,
                        ),
                        Some(sink) => pass_chunk_offering(
                            &mut self.diagonals[start..end],
                            cols,
                            scalars,
                            &mut self.acc,
                            &mut self.tile,
                            Some((sink, row as usize)),
                        ),
                    }
                }
                RowArithmetic::Wide(scalars) => {
                    let count = end - start;
                    let gather = &mut self.wide;
                    for lane in 0..count {
                        let window = window_at(start + lane);
                        gather.enter[lane] = window.entering;
                        gather.leave[lane] = window.leaving;
                        gather.sum[lane] = window.stats.sum;
                    }
                    pass_chunk_wide(
                        &mut self.diagonals[start..end],
                        WideChunkCols {
                            enter: &gather.enter[..count],
                            leave: &gather.leave[..count],
                            sum: &gather.sum[..count],
                            r: &r[p..q],
                            bias: &bias[p..q],
                            first,
                            neighbor: &neighbor,
                        },
                        scalars,
                        m,
                        &mut self.acc,
                        &mut self.tile,
                    );
                    if let Some(sink) = self.right.as_mut().filter(|_| self.acc.ranks()) {
                        let ranked = &self.diagonals[start..end];
                        let tile = &self.tile[..count];
                        sink.offer_chunk::<WideRank>(
                            first,
                            self.acc.id(row as usize),
                            tile,
                            |lane| {
                                centered_cross_fast(
                                    ranked[lane],
                                    m as i64,
                                    scalars.sum,
                                    gather.sum[lane],
                                )
                            },
                            |lane| crate::columns::variance(&neighbor(lane)),
                        );
                    }
                }
            }
            start = end;
        }
        self.acc.finish(nearest);
    }

    /// Checked every tick. A no-op once `self.path` is `Wide`, or while `tick` still
    /// fits `self.anchor`. Otherwise tries to re-anchor every live diagonal exactly
    /// ([`narrow::reanchor`]) to a new anchor covering the buffer's whole current span
    /// (including `tick`); when even the best anchor could not cover that span, switches to
    /// `Wide` instead, converting every live diagonal's state exactly. This runs
    /// before `self.rows` is incremented for whatever row `tick` completes (if any), so "live"
    /// below means the diagonals `offer_row` last wrote: offsets
    /// `exclusion + 1 ..= min(self.rows - 1, max_offset)`.
    ///
    /// Returns the tick held in place of `tick` when `tick` is withheld instead
    /// ([`Self::withholds`]). Asked where the tick does not fit the anchor, the stream is on the
    /// i64 arithmetic: every anchor covers the level ([`Self::tick_span`]), so a tick that fits
    /// lies within twice the anchor's bound of it, which is inside the rule's limit.
    #[inline(always)]
    fn maintain_narrow_range(&mut self, tick: i64) -> Option<i64> {
        if self.path == Path::Wide {
            if self.withholds(tick) {
                return Some(self.withhold(tick));
            }
            self.return_to_narrow_if_fits(tick);
            return None;
        }
        if crate::route::wide(false) {
            self.switch_to_wide();
            return self.withholds(tick).then(|| self.withhold(tick));
        }
        let needed = self.anchor.offset(tick, self.m).is_none();
        if !crate::route::reanchor_due(self.rows, needed) {
            return self.forced_withhold().then(|| self.withhold(tick));
        }
        if if needed {
            self.withholds(tick)
        } else {
            self.forced_withhold()
        } {
            return Some(self.withhold(tick));
        }

        // `tick` has not joined `self.ticks` yet (see the call site), so it is folded in
        // explicitly here alongside the ring's current content.
        let (lo, hi) = self
            .tick_span(Some(tick))
            .expect("the incoming tick is always part of the span");

        // Centered, not pinned to `lo`'s own edge: an anchor at `lo + bound` leaves no margin
        // below `lo` at all, so on a walk that keeps drifting (new lows as often as new highs)
        // almost every downward tick would re-anchor again immediately. Centering splits
        // whatever margin `2 * bound - (hi - lo)` is left over evenly between both tails.
        let Some(new_anchor) = Anchor::spanning(lo, hi, self.m, self.rows) else {
            self.switch_to_wide();
            return None;
        };
        self.reanchor_live_diagonals(new_anchor);
        self.anchor = new_anchor;
        self.routes.reanchors += 1;
        None
    }

    /// While `Wide`, every `max(m, 64)` rows (the check scans the whole buffer, so it is not
    /// made more often) and only when the buffer plus `tick` spans no more than half what one
    /// anchor covers ([`Anchor::roomy_spanning`]), so a span hovering at the limit cannot flap
    /// between the paths. The switch is O(w) work on the row it happens, like a re-anchor, and
    /// writes only storage `new` allocated. A forced wide route never returns.
    fn return_to_narrow_if_fits(&mut self, tick: i64) {
        let every = self.m.get().max(64) as u64;
        if !self.rows.is_multiple_of(every) || crate::route::forced_wide() {
            return;
        }
        let (lo, hi) = self
            .tick_span(Some(tick))
            .expect("the incoming tick is always part of the span");
        let Some(anchor) = Anchor::roomy_spanning(lo, hi, self.m, self.rows) else {
            return;
        };
        self.anchor = anchor;
        self.rebuild_narrow_state();
        self.path = Path::Narrow;
        self.routes.switches_to_narrow += 1;
        self.cold_end = self.spill_end;
        self.hot_k = self.hot_k_for_rows();
    }

    /// Rewrites the live columns and diagonals, which hold absolute values on the wide path, as
    /// offsets from `self.anchor`, in place.
    fn rebuild_narrow_state(&mut self) {
        let anchor = self.anchor;
        let m = self.m;
        let windows = self.windows.as_slice();
        let first_row = self.rows - windows.len() as u64;
        let has_leaving = |pos: usize| first_row + pos as u64 > 0 && self.max_offset > 0;
        let mut pos = 0;
        self.col_sum.update_live(|v| {
            *v = narrow::sum_offset(anchor, m, windows[pos].stats.sum);
            pos += 1;
        });
        pos = 0;
        self.col_enter.update_live(|v| {
            *v = narrow::tick_offset(anchor, m, windows[pos].entering);
            pos += 1;
        });
        pos = 0;
        self.col_leave.update_live(|v| {
            *v = if has_leaving(pos) {
                narrow::tick_offset(anchor, m, windows[pos].leaving)
            } else {
                0
            };
            pos += 1;
        });

        let Some(live_hi) = self.live_offset_hi() else {
            return;
        };
        let exclusion = self.exclusion as u64;
        let len = windows.len() as u64;
        let newest_sum = windows[(len - 1) as usize].stats.sum;
        for d in (exclusion + 1)..=live_hi {
            let slot = (self.max_offset as u64 - d) as usize;
            let neighbor_sum = windows[(len - 1 - d) as usize].stats.sum;
            self.diagonals[slot] =
                narrow::narrow_qt_m(anchor, m, self.diagonals[slot], newest_sum, neighbor_sum);
        }
    }

    /// The highest offset `offer_row` has already written a diagonal for, or `None` before the
    /// first row.
    fn live_offset_hi(&self) -> Option<u64> {
        (self.rows > 0).then(|| (self.rows - 1).min(self.max_offset as u64))
    }

    /// Re-anchors every live diagonal from `self.anchor` to `new_anchor`, exactly
    /// ([`narrow::reanchor`]): `sa`/`sb` are each pair's two windows' own sum offsets under the
    /// *old* anchor, read fresh from their permanently stored absolute sums rather than kept
    /// incrementally, so this never accumulates its own rounding beyond `reanchor`'s one step.
    /// Shifts every `col_*` entry the same exact way (`enter`/`leave` by `-Δ`, `sum` by
    /// `-m·Δ`), unconditionally: a window not yet live as a diagonal (inside the exclusion
    /// zone) still owes its own columns a correct anchor once it does become live later.
    fn reanchor_live_diagonals(&mut self, new_anchor: Anchor) {
        let delta = new_anchor.tick - self.anchor.tick;
        let m = self.m;
        let m_delta = (m.get() as i64).wrapping_mul(delta);
        self.col_enter
            .update_live(|v| *v = (i64::from(*v) - delta) as i32);
        self.col_leave
            .update_live(|v| *v = (i64::from(*v) - delta) as i32);
        self.col_sum
            .update_live(|v| *v = i64::from(*v).wrapping_sub(m_delta) as i32);

        let Some(live_hi) = self.live_offset_hi() else {
            return;
        };
        let exclusion = self.exclusion as u64;
        if live_hi <= exclusion {
            return;
        }
        let max_offset = self.max_offset as u64;
        let windows = self.windows.as_slice();
        let len = windows.len() as u64;
        let newest = windows[(len - 1) as usize];
        let sa = newest
            .stats
            .sum
            .wrapping_sub((m.get() as i64).wrapping_mul(self.anchor.tick));
        for d in (exclusion + 1)..=live_hi {
            let neighbor = windows[(len - 1 - d) as usize];
            let slot = (max_offset - d) as usize;
            let sb = neighbor
                .stats
                .sum
                .wrapping_sub((m.get() as i64).wrapping_mul(self.anchor.tick));
            let (qt_m, _, _) = narrow::reanchor(delta, m, self.diagonals[slot], sa, sb);
            self.diagonals[slot] = qt_m;
        }
    }

    /// Switches to the exact i64 fallback, converting every live diagonal's
    /// state exactly: freshly computed absolute `QT` ([`cross_term::wide`]) for the pair
    /// `(self.rows - 1, self.rows - 1 - d)`, the same pair `self.diagonals[d - 1]` already
    /// represents, so `offer_row` can advance it normally on every row from here on. Runs
    /// before `tick` joins `self.ticks` (see the call site), so `self.ticks` here still ends
    /// exactly at row `self.rows - 1`'s own last tick, the same buffer `row_start` below always
    /// assumed: were this run after the push, the oldest live window's first tick could already
    /// have been evicted to make room for `tick`.
    fn switch_to_wide(&mut self) {
        self.routes.switches_to_wide += 1;
        self.watch(u64::MAX);
        let Some(live_hi) = self.live_offset_hi() else {
            self.path = Path::Wide;
            return;
        };
        let exclusion = self.exclusion as u64;
        if live_hi > exclusion {
            let ticks = self.ticks.as_slice();
            let row_start = ticks.len() - self.m.get();
            let pairs = CrossTerms::of(ticks, self.m.get());
            let max_offset = self.max_offset as u64;
            for d in (exclusion + 1)..=live_hi {
                self.diagonals[(max_offset - d) as usize] =
                    pairs.at(row_start, row_start - d as usize);
            }
        }
        self.path = Path::Wide;
    }
}

/// A row's nearest as [`BoundedStream::advance`] leaves it: the one winner of the hot arithmetic,
/// not yet reported, or a result kept in `kept_result` (with `k` above one, or from the cold
/// route).
enum Winner {
    One(Option<Candidate>),
    Kept,
}

/// The samples a checkpoint holds as they came, and what restoring withholds of them.
struct Restored {
    /// Positions and exact ticks withheld, oldest first.
    withheld: Vec<(u64, i64)>,
    /// The outlier rule's level and run after the ticks held.
    outliers: OutlierScan,
    /// The ticks held, from position `base`, with their presence.
    ticks: Vec<i64>,
    had: Vec<bool>,
    /// The caller's flag each live window was stored under.
    flags: Vec<Option<bool>>,
    base: u64,
}

impl crate::stream_api::Stream for BoundedStream {
    fn update_flagged(&mut self, x: f64, constant: Option<bool>) -> Option<(SampleIndex, Entry)> {
        self.update_flagged(x, constant)
    }
}

#[cfg(test)]
mod tests;
