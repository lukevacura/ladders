//! The growing non-normalized stream: the causal left nearest-neighbor profile of a stream that
//! never forgets a sample, under squared Euclidean distance on raw ticks, as STUMPY's `aampi`
//! computes it. Each [`GrowingNonNormStream::update`] quantizes and forward-fills one sample and,
//! once a window completes, carries every diagonal one step with the O(1) residue recurrence and
//! returns the row's exact neighbor, never to be revisited.
//!
//! A diagonal is kept modulo `2¹²⁸` over the forward-filled ticks, so it is defined for every
//! pair, gap or not; only a pair of complete windows is ever offered. The row runs in `u64` while
//! `m·R² < 2⁶⁴` over the range `R` of every tick so far and in `u128` after that; the range only
//! grows, so the switch is one way. Once `m·R² ≥ 2¹²⁸` (`m` beyond 2²⁰ at the widest ticks), no
//! width holds a distance and every later row reports no neighbor.

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
use crate::sanitize::ForwardFill;

/// Causal, exact left nearest-neighbor profile on raw ticks over a growing stream: every array
/// extends as samples arrive, so a row once returned never changes.
pub struct GrowingNonNormStream {
    m: NonZeroUsize,
    /// `ceil(m/4)`: a neighbor this close or closer is never eligible, for any row.
    exclusion: usize,
    scale: Scale,
    /// How many nearest earlier neighbors each row keeps.
    k: NonZeroUsize,
    fill: ForwardFill<i64>,
    /// Every sanitized tick so far, and whether the sample was present.
    ticks: Vec<i64>,
    present: Vec<bool>,
    /// Whether each completed window has every tick present.
    complete: Vec<bool>,
    /// Consecutive present ticks ending at the newest, up to `m`.
    run: usize,
    lowest: i64,
    highest: i64,
    /// `diagonals[k]` is the newest row's squared distance to the window `exclusion + 1 + k`
    /// earlier, modulo `2¹²⁸`.
    diagonals: Vec<u128>,
    /// The newest row's kept pairs and their neighbors, reused by every row; room for one
    /// neighbor is there from the start.
    kept: Vec<(u128, u64)>,
    neighbors: Vec<NonNormNeighbor>,
    routes: RouteLog,
    /// Each window's nearest later neighbors, when the stream looks right.
    right: Option<RightSquared>,
}

impl GrowingNonNormStream {
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
            ticks: Vec::new(),
            present: Vec::new(),
            complete: Vec::new(),
            run: 0,
            lowest: i64::MAX,
            highest: i64::MIN,
            diagonals: Vec::new(),
            kept: Vec::new(),
            neighbors: Vec::with_capacity(1),
            routes: RouteLog::default(),
            right: None,
        }
    }

    /// [`Self::with_k`] that also keeps, for each window, its `k` nearest *later* neighbors and
    /// its `k` nearest over both sides ([`Self::right_neighbors_into`],
    /// [`Self::two_sided_neighbors_into`]). This looks ahead of a window: an entry for a window
    /// can change when later samples arrive, so nothing causal may read it. The stream's own
    /// neighbors are those of [`Self::with_k`].
    pub fn with_right_side(m: NonZeroUsize, scale: Scale, k: NonZeroUsize) -> Self {
        let mut stream = Self::with_k(m, scale, k);
        stream.right = Some(RightSquared::growing(k));
        stream
    }

    /// Whether the stream keeps the right side ([`Self::with_right_side`]).
    pub fn looks_right(&self) -> bool {
        self.right.is_some()
    }

    /// The windows whose right and two-sided neighbors the stream holds: every window so far, or none when it
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

    /// `None` until the first window completes; then `(row, neighbor)` for the newest window: its
    /// nearest earlier neighbor, the first of what [`Self::update_topk`] keeps. A window with a
    /// missing sample has no neighbor, and is never one.
    pub fn update(&mut self, x: f64) -> Option<(SampleIndex, NonNormNeighbor)> {
        self.advance(x)
    }

    /// `None` until the first window completes; then the newest window's row and its `k` nearest
    /// earlier neighbors, nearest first by exact squared distance with ties to the smaller index:
    /// fewer when fewer exist, none for a window without an eligible neighbor. The slice holds
    /// until the next update.
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
        let (tick, present) = sanitized_tick(&mut self.fill, x, self.scale);
        self.ticks.push(tick);
        self.present.push(present);
        self.lowest = self.lowest.min(tick);
        self.highest = self.highest.max(tick);
        self.run = if present { (self.run + 1).min(m) } else { 0 };
        if self.ticks.len() < m {
            self.routes.record(RowWidth::Narrow);
            return None;
        }

        let row = self.complete.len();
        self.complete.push(self.run == m);
        if let Some(right) = &mut self.right {
            right.begin_row(row as u64);
        }
        let width = routed_row_width(self.lowest, self.highest, m);
        self.routes.record(width);

        let first_lag = self.exclusion + 1;
        let fresh = row >= first_lag;
        if fresh {
            self.diagonals.push(0);
        }
        let view = RowView {
            ticks: &self.ticks,
            complete: &self.complete,
            m,
            row: row as u64,
            first_lag,
            fit: Fit::of(self.lowest, self.highest),
        };
        let nearest = if self.k == NonZeroUsize::MIN {
            let mut nearest = NearestSquared::default();
            match self.right.as_mut() {
                None => advance_row(&mut self.diagonals, fresh, &view, width, &mut nearest),
                Some(right) => {
                    let mut tee = Tee::new(&mut nearest, right, row as u64);
                    advance_row(&mut self.diagonals, fresh, &view, width, &mut tee);
                }
            }
            nearest.finish()
        } else {
            let bound = self.diagonals.len();
            let mut kept = KeptNearest::new(&mut self.kept, self.k.get(), bound);
            match self.right.as_mut() {
                None => advance_row(&mut self.diagonals, fresh, &view, width, &mut kept),
                Some(right) => {
                    let mut tee = Tee::new(&mut kept, right, row as u64);
                    advance_row(&mut self.diagonals, fresh, &view, width, &mut tee);
                }
            }
            write_neighbors(&self.kept, &mut self.neighbors, self.k.get(), bound);
            self.neighbors
                .first()
                .copied()
                .unwrap_or(NonNormNeighbor::NONE)
        };
        let row = SampleIndex::new(row as u64).expect("a row is below the sentinel");
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

    /// Every sanitized tick and whether its sample was present, versioned and little-endian
    /// (`checkpoint.rs` owns the format). [`Self::restore`] on these bytes continues bitwise
    /// identically to a stream that never stopped.
    pub fn checkpoint(&self) -> Vec<u8> {
        let plain = checkpoint::write_nonnorm_growing(
            self.m,
            self.k,
            self.scale,
            &self.ticks,
            &self.present,
        );
        match &self.right {
            None => plain,
            Some(right) => {
                checkpoint::wrap_right(checkpoint::KIND_NONNORM_GROWING, &plain, |buf| {
                    checkpoint::put_squared_columns(
                        buf,
                        right.held_windows().map(|w| right.column(w)).collect(),
                    );
                })
            }
        }
    }

    /// The stream [`Self::checkpoint`] wrote, or the reason the bytes are not one.
    pub fn restore(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let Some(mut wrapped) = checkpoint::open_right(bytes, checkpoint::KIND_NONNORM_GROWING)?
        else {
            return Self::restore_plain(bytes);
        };
        let mut stream = Self::restore_plain(wrapped.inner)?;
        let columns = checkpoint::get_squared_columns(
            &mut wrapped.reader,
            stream.k.get(),
            stream.complete.len(),
        )?;
        wrapped.reader.finish()?;
        let mut right = RightSquared::growing(stream.k);
        for (row, (right_list, both_list)) in columns.into_iter().enumerate() {
            right.begin_row(row as u64);
            if !right.load(row as u64, &right_list, &both_list) {
                return Err(CheckpointError::Inconsistent);
            }
        }
        stream.right = Some(right);
        Ok(stream)
    }

    fn restore_plain(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let parts = checkpoint::read_nonnorm_growing(bytes)?;
        let m = parts.m.get();
        let mut fill = ForwardFill::default();
        if let Some(&newest) = parts.ticks.last() {
            let _ = fill.push(Some(newest));
        }
        let complete = complete_windows(&parts.present, m);
        let rows = complete.len();
        let exclusion = crate::exclusion::zone(m);
        let diagonals = (exclusion + 1..=rows.saturating_sub(1))
            .map(|lag| start_diagonal(&parts.ticks, lag, m, RowWidth::Wide))
            .collect();
        Ok(Self {
            m: parts.m,
            exclusion,
            scale: parts.scale,
            k: parts.k,
            fill,
            lowest: parts.ticks.iter().copied().min().unwrap_or(i64::MAX),
            highest: parts.ticks.iter().copied().max().unwrap_or(i64::MIN),
            run: trailing_run(&parts.present, m),
            ticks: parts.ticks,
            present: parts.present,
            complete,
            diagonals,
            kept: Vec::new(),
            neighbors: Vec::with_capacity(1),
            routes: RouteLog::default(),
            right: None,
        })
    }
}
