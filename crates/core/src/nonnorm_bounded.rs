//! The bounded non-normalized stream: the causal left nearest-neighbor profile of the last `w`
//! samples, under squared Euclidean distance on raw ticks, in memory that never grows past
//! `O(w)`. Every offset `row − neighbor` that a window inside the last `w` samples can have owns
//! one fixed diagonal, advanced in place each row, so nothing is ever evicted or shifted.
//!
//! A row runs in `u64` while `m·R² < 2⁶⁴` over the range `R` of the last `w` ticks, in `u128`
//! while `m·R² < 2¹²⁸`, and has no neighbor beyond that. The range is the horizon's, so a shock
//! that leaves the horizon returns the stream to `u64`: a diagonal is kept modulo `2¹²⁸`, which
//! is exact whenever the row it serves fits the width it runs in, so either switch is free.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::batch::nonnorm::{KeptNearest, write_neighbors};
use crate::checkpoint::{self, CheckpointError};
use crate::index::SampleIndex;
use crate::kernel::Reducer;
use crate::narrow::RouteCounts;
use crate::nonnorm::{
    NearestSquared, NonNormNeighbor, RouteLog, RowView, RowWidth, advance_row, complete_windows,
    routed_row_width, sanitized_tick, start_diagonal, trailing_run,
};
use crate::nonnorm_lanes::Fit;
use crate::quantize::Scale;
use crate::right_nonnorm::{RightSquared, Tee};
use crate::ring::MirroredRing;
use crate::sanitize::ForwardFill;
use crate::stats::MonotonicDeque;

/// Causal, exact left nearest-neighbor profile on raw ticks over the last `w` samples: a window
/// older than that is never a candidate, however long the stream runs.
pub struct BoundedNonNormStream {
    m: NonZeroUsize,
    /// `ceil(m/4)`: a neighbor this close or closer is never eligible, for any row.
    exclusion: usize,
    horizon: NonZeroUsize,
    scale: Scale,
    /// How many nearest earlier neighbors each row keeps.
    k: NonZeroUsize,
    fill: ForwardFill<i64>,
    /// Samples seen so far; the next completed window is row `total − m + 1`.
    total: u64,
    /// The last `w + 1` sanitized ticks: the oldest is the tick a diagonal at the widest lag
    /// drops when the newest arrives.
    ticks: MirroredRing<i64>,
    /// Whether each of the last `w` samples was present.
    present: MirroredRing<bool>,
    /// Whether each of the last `w − m + 1` windows has every tick present.
    complete: MirroredRing<bool>,
    /// Consecutive present ticks ending at the newest, up to `m`.
    run: usize,
    lowest: MonotonicDeque<false>,
    highest: MonotonicDeque<true>,
    /// `diagonals[k]` is the newest row's squared distance to the window `exclusion + 1 + k`
    /// earlier, modulo `2¹²⁸`, for every lag that has had a row; later ones are zero.
    diagonals: Box<[u128]>,
    /// The newest row's kept pairs and their neighbors, reserved for the most any row can keep by
    /// the first row with `k` above one, then reused; room for one neighbor is there from the
    /// start.
    kept: Vec<(u128, u64)>,
    neighbors: Vec<NonNormNeighbor>,
    routes: RouteLog,
    /// Each held window's nearest later neighbors, when the stream looks right.
    right: Option<RightSquared>,
}

impl BoundedNonNormStream {
    /// `w` is the number of trailing samples searched; `None` when `w` is less than `m`, since
    /// no window that long could ever stay inside it.
    pub fn new(m: NonZeroUsize, w: NonZeroUsize, scale: Scale) -> Option<Self> {
        Self::with_k(m, w, scale, NonZeroUsize::MIN)
    }

    /// [`Self::new`] keeping the `k` nearest earlier neighbors of each window
    /// ([`Self::update_topk`]); memory stays `O(w + k)`, and a `k` beyond the windows that fit in
    /// `w` reserves only those.
    pub fn with_k(m: NonZeroUsize, w: NonZeroUsize, scale: Scale, k: NonZeroUsize) -> Option<Self> {
        let max_lag = w.get().checked_sub(m.get())?;
        let exclusion = crate::exclusion::zone(m.get());
        Some(Self {
            m,
            exclusion,
            horizon: w,
            scale,
            k,
            fill: ForwardFill::default(),
            total: 0,
            ticks: MirroredRing::new(w.checked_add(1)?, 0),
            present: MirroredRing::new(w, false),
            complete: MirroredRing::new(NonZeroUsize::new(max_lag + 1)?, false),
            run: 0,
            lowest: MonotonicDeque::new(w),
            highest: MonotonicDeque::new(w),
            diagonals: vec![0; max_lag.saturating_sub(exclusion)].into_boxed_slice(),
            kept: Vec::new(),
            neighbors: Vec::with_capacity(1),
            routes: RouteLog::default(),
            right: None,
        })
    }

    /// [`Self::with_k`] that also keeps, for each window it holds (the last `w − m + 1`), its `k`
    /// nearest *later* neighbors and its `k` nearest over both sides, the earlier ones as the
    /// stream found them ([`Self::right_neighbors_into`], [`Self::two_sided_neighbors_into`]).
    /// This looks ahead of a window: an entry for a window can change when later samples arrive,
    /// so nothing causal may read it. The stream's own neighbors are those of [`Self::with_k`].
    /// `None` when `w` is less than `m`, or when the lists cannot be allocated.
    pub fn with_right_side(
        m: NonZeroUsize,
        w: NonZeroUsize,
        scale: Scale,
        k: NonZeroUsize,
    ) -> Option<Self> {
        let mut stream = Self::with_k(m, w, scale, k)?;
        let windows = w.get() - m.get() + 1;
        let reach = windows.saturating_sub(stream.exclusion + 1);
        stream.right = Some(RightSquared::held(k, windows, reach)?);
        Some(stream)
    }

    /// Whether the stream keeps the right side ([`Self::with_right_side`]).
    pub fn looks_right(&self) -> bool {
        self.right.is_some()
    }

    /// The windows whose right and two-sided neighbors the stream holds (the last `w − m + 1`), or none when it
    /// does not look right.
    pub fn right_windows(&self) -> core::ops::Range<u64> {
        self.right.as_ref().map_or(0..0, RightSquared::held_windows)
    }

    /// Window `window`'s `k` nearest later neighbors so far, nearest first by exact squared
    /// distance with ties to the smaller index, in `out` (cleared first); none for a window not
    /// held or a stream that does not look right.
    pub fn right_neighbors_into(&self, window: u64, out: &mut Vec<NonNormNeighbor>) {
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
    /// [`Self::right_neighbors_into`] but over the earlier ones too.
    pub fn two_sided_neighbors_into(&self, window: u64, out: &mut Vec<NonNormNeighbor>) {
        out.clear();
        if let Some(right) = self
            .right
            .as_ref()
            .filter(|_| self.right_windows().contains(&window))
        {
            right.both_into(window, out);
        }
    }

    /// How many nearest earlier neighbors each window keeps.
    pub fn k(&self) -> NonZeroUsize {
        self.k
    }

    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// Updates by arithmetic and route changes since this stream was created or restored;
    /// informational only, never part of a checkpoint.
    #[doc(hidden)]
    pub fn route_counts(&self) -> RouteCounts {
        self.routes.counts()
    }

    /// `None` until the first window completes; then `(row, neighbor)` for the newest window,
    /// searched among the windows inside the last `w` samples: its nearest earlier neighbor, the
    /// first of what [`Self::update_topk`] keeps. A window with a missing sample has no neighbor,
    /// and is never one.
    pub fn update(&mut self, x: f64) -> Option<(SampleIndex, NonNormNeighbor)> {
        self.advance(x)
    }

    /// `None` until the first window completes; then the newest window's row and its `k` nearest
    /// earlier neighbors inside the last `w` samples, nearest first by exact squared distance
    /// with ties to the smaller index: fewer when fewer exist, none for a window without an
    /// eligible neighbor. The slice holds until the next update.
    pub fn update_topk(&mut self, x: f64) -> Option<(SampleIndex, &[NonNormNeighbor])> {
        let (row, nearest) = self.advance(x)?;
        if self.k == NonZeroUsize::MIN {
            self.neighbors.clear();
            if nearest.index.get().is_some() {
                self.neighbors.push(nearest);
            }
        }
        Some((row, &self.neighbors))
    }

    /// Folds in one sample and returns a completed window's row and nearest neighbor; with `k`
    /// above one its kept neighbors are left in `self.neighbors` too.
    fn advance(&mut self, x: f64) -> Option<(SampleIndex, NonNormNeighbor)> {
        let m = self.m.get();
        let w = self.horizon.get();
        let (tick, present) = sanitized_tick(&mut self.fill, x, self.scale);
        self.ticks.push(tick);
        self.present.push(present);
        self.lowest.push(self.total, tick, w);
        self.highest.push(self.total, tick, w);
        self.total += 1;
        self.run = if present { (self.run + 1).min(m) } else { 0 };
        if self.total < m as u64 {
            self.routes.record(RowWidth::Narrow);
            return None;
        }

        let row = self.total - m as u64;
        self.complete.push(self.run == m);
        if let Some(right) = &mut self.right {
            right.begin_row(row);
        }
        let width = routed_row_width(self.lowest.extreme(), self.highest.extreme(), m);
        self.routes.record(width);

        let first_lag = self.exclusion + 1;
        let lags = row.saturating_sub(self.exclusion as u64);
        let live = lags.min(self.diagonals.len() as u64) as usize;
        let fresh = row >= first_lag as u64 && lags == live as u64;
        let ticks = self.ticks.as_slice();
        // The tick a diagonal at the widest lag drops is one older than the extremes' horizon.
        let oldest = ticks.first().copied().unwrap_or(tick);
        let view = RowView {
            ticks,
            complete: self.complete.as_slice(),
            m,
            row,
            first_lag,
            fit: Fit::of(
                self.lowest.extreme().min(oldest),
                self.highest.extreme().max(oldest),
            ),
        };
        let nearest = if self.k == NonZeroUsize::MIN {
            let mut nearest = NearestSquared::default();
            match self.right.as_mut() {
                None => advance_row(
                    &mut self.diagonals[..live],
                    fresh,
                    &view,
                    width,
                    &mut nearest,
                ),
                Some(right) => {
                    let mut tee = Tee::new(&mut nearest, right, row);
                    advance_row(&mut self.diagonals[..live], fresh, &view, width, &mut tee);
                }
            }
            nearest.finish()
        } else {
            // The bound is fixed, so the first row reserves for every later one.
            let bound = self.diagonals.len();
            let mut kept = KeptNearest::new(&mut self.kept, self.k.get(), bound);
            match self.right.as_mut() {
                None => advance_row(&mut self.diagonals[..live], fresh, &view, width, &mut kept),
                Some(right) => {
                    let mut tee = Tee::new(&mut kept, right, row);
                    advance_row(&mut self.diagonals[..live], fresh, &view, width, &mut tee);
                }
            }
            write_neighbors(&self.kept, &mut self.neighbors, self.k.get(), bound);
            self.neighbors
                .first()
                .copied()
                .unwrap_or(NonNormNeighbor::NONE)
        };
        let row = SampleIndex::new(row).expect("a row is below the sentinel");
        Some((row, nearest))
    }

    /// Folds in every sample of `xs`, in order, appending each completed window's `(row,
    /// neighbor)` to `out` (never cleared): exactly what the same samples fed one at a time to
    /// [`Self::update`] would append.
    pub fn update_many(&mut self, xs: &[f64], out: &mut Vec<(SampleIndex, NonNormNeighbor)>) {
        for &x in xs {
            if let Some(row) = self.update(x) {
                out.push(row);
            }
        }
    }

    /// The last `w` sanitized ticks and whether each sample was present, versioned and
    /// little-endian (`checkpoint.rs` owns the format). [`Self::restore`] on these bytes
    /// continues bitwise identically to a stream that never stopped.
    pub fn checkpoint(&self) -> Vec<u8> {
        let plain = self.checkpoint_plain();
        match &self.right {
            None => plain,
            Some(right) => {
                checkpoint::wrap_right(checkpoint::KIND_NONNORM_BOUNDED, &plain, |buf| {
                    checkpoint::put_squared_columns(
                        buf,
                        right.held_windows().map(|w| right.column(w)).collect(),
                    );
                })
            }
        }
    }

    fn checkpoint_plain(&self) -> Vec<u8> {
        let ticks = self.ticks.as_slice();
        let reach = self.present.len();
        checkpoint::write_nonnorm_bounded(
            self.m,
            self.k,
            self.horizon,
            self.scale,
            self.total,
            &ticks[ticks.len() - reach..],
            self.present.as_slice(),
        )
    }

    /// The stream [`Self::checkpoint`] wrote, or the reason the bytes are not one.
    pub fn restore(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let Some(mut wrapped) = checkpoint::open_right(bytes, checkpoint::KIND_NONNORM_BOUNDED)?
        else {
            return Self::restore_plain(bytes);
        };
        let mut stream = Self::restore_plain(wrapped.inner)?;
        let held = stream.complete.len();
        let columns = checkpoint::get_squared_columns(&mut wrapped.reader, stream.k.get(), held)?;
        wrapped.reader.finish()?;
        let windows = stream.horizon.get() - stream.m.get() + 1;
        let reach = windows.saturating_sub(stream.exclusion + 1);
        let mut right =
            RightSquared::held(stream.k, windows, reach).ok_or(CheckpointError::Inconsistent)?;
        let first = (stream.total + 1).saturating_sub(stream.m.get() as u64) - held as u64;
        for (at, (right_list, both_list)) in columns.into_iter().enumerate() {
            let row = first + at as u64;
            right.begin_row(row);
            if !right.load(row, &right_list, &both_list) {
                return Err(CheckpointError::Inconsistent);
            }
        }
        stream.right = Some(right);
        Ok(stream)
    }

    fn restore_plain(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let parts = checkpoint::read_nonnorm_bounded(bytes)?;
        let (m, w) = (parts.m.get(), parts.horizon);
        let horizon = NonZeroUsize::new(w).ok_or(CheckpointError::Inconsistent)?;
        let mut stream = Self::with_k(parts.m, horizon, parts.scale, parts.k)
            .ok_or(CheckpointError::Inconsistent)?;

        let base = parts.total - parts.ticks.len() as u64;
        for (k, (&tick, &present)) in parts.ticks.iter().zip(&parts.present).enumerate() {
            stream.ticks.push(tick);
            stream.present.push(present);
            stream.lowest.push(base + k as u64, tick, w);
            stream.highest.push(base + k as u64, tick, w);
        }
        for complete in complete_windows(&parts.present, m) {
            stream.complete.push(complete);
        }
        if let Some(&newest) = parts.ticks.last() {
            let _ = stream.fill.push(Some(newest));
        }
        stream.total = parts.total;
        stream.run = trailing_run(&parts.present, m);

        let newest_row = parts.total.saturating_sub(m as u64).min(usize::MAX as u64) as usize;
        let live = newest_row
            .saturating_sub(stream.exclusion)
            .min(stream.diagonals.len());
        for (k, diagonal) in stream.diagonals[..live].iter_mut().enumerate() {
            let lag = stream.exclusion + 1 + k;
            *diagonal = start_diagonal(&parts.ticks, lag, m, RowWidth::Wide);
        }
        Ok(stream)
    }
}
