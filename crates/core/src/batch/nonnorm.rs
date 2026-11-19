//! The non-normalized profiles (squared Euclidean distance on raw ticks) on the batch tile
//! scheduler: the left profile, the two-sided profile and the AB-join, each the exact nearest
//! neighbor, ties to the smaller index, equal on bits to the exact reference for any thread count
//! and tile shape. The two-sided profile and the AB-join also keep the `k` nearest neighbors
//! ([`two_sided_topk_profile`], [`ab_join_topk`]).
//!
//! A tile starts each diagonal's residue from the exact sum of its first pair and carries it with
//! the O(1) recurrence. The residue is exact modulo `2^64` or `2^128`, so a tile runs in `u64`
//! when `m·R² < 2⁶⁴` over the range `R` of the ticks it can see, and in `u128` otherwise; both
//! give the same distance, so the width is only a speed choice of the tile. A `u64` tile runs row
//! by row through the lane kernel, many diagonals at once; a `u128` tile walks each diagonal down
//! its rows. A row has no neighbor when `m·R² ≥ 2¹²⁸` over every present tick from the series' start
//! through its own window (and, in a join, all of `b`): no width holds its distances, and the
//! rule belongs to the row, never to a tile.

mod topk;

pub(crate) use self::topk::{KeptNearest, write_neighbors};
pub use self::topk::{
    ab_join_topk, ab_join_topk_tiled, two_sided_topk_profile, two_sided_topk_profile_tiled,
};

use super::pass::{Shape, push_none_rows};
use super::plan::{
    SlotRuns, Tile, TileCut, TilePlan, default_chunk_lanes, threads_for_join, threads_for_size,
};
use super::schedule::{BatchReport, ColumnGroups, RIGHT_GROUP, TileWork, run_tile_work};
use super::tile::RanWide;
use crate::capacity::{CapacityError, abort_on_capacity};
use crate::index::SampleIndex;
use crate::kernel::{Metric, Reducer};
use crate::nonnorm::{
    NearestSquared, NonNormNeighbor, NonNormTwoSided, NonNormalized, Residue, RowWidth, TickStep,
    complete_windows, routed_row_width, row_width, sanitized_tick,
};
use crate::nonnorm_lanes::{self, ColumnMins, Fit, Lanes, Pass};
use crate::quantize::Scale;
use crate::sanitize::ForwardFill;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// A series as the arithmetic sees it: its forward-filled ticks, whether each tick was present
/// (a filled one is not data), and whether each window has every tick present.
pub(crate) struct TickSeries {
    pub(super) ticks: Vec<i64>,
    pub(super) present: Vec<bool>,
    pub(super) complete: Vec<bool>,
}

impl TickSeries {
    pub(crate) fn build(series: &[f64], m: NonZeroUsize, scale: Scale) -> Self {
        let mut fill = ForwardFill::<i64>::default();
        let mut ticks = Vec::with_capacity(series.len());
        let mut present = Vec::with_capacity(series.len());
        for &x in series {
            let (tick, was_present) = sanitized_tick(&mut fill, x, scale);
            ticks.push(tick);
            present.push(was_present);
        }
        Self {
            ticks,
            complete: complete_windows(&present, m.get()),
            present,
        }
    }

    /// The lowest and highest present tick, `None` for no tick.
    pub(crate) fn present_range(&self) -> Option<(i64, i64)> {
        let mut present = self
            .ticks
            .iter()
            .zip(&self.present)
            .filter_map(|(&tick, &was_present)| was_present.then_some(tick));
        let first = present.next()?;
        Some(present.fold((first, first), |(lowest, highest), tick| {
            (lowest.min(tick), highest.max(tick))
        }))
    }

    pub(super) fn windows(&self) -> usize {
        self.complete.len()
    }

    /// The series of `samples` (`None` for a missing one) without a scale in between, for tests
    /// that need ticks no scale can reach.
    #[cfg(test)]
    pub(super) fn from_samples(samples: &[Option<i64>], m: usize) -> Self {
        let mut last = 0;
        let ticks = samples
            .iter()
            .map(|sample| {
                last = sample.unwrap_or(last);
                last
            })
            .collect();
        let present: Vec<bool> = samples.iter().map(Option::is_some).collect();
        Self {
            ticks,
            complete: complete_windows(&present, m),
            present,
        }
    }
}

/// The lowest and highest of `ticks`, `None` for no ticks.
pub(super) fn range_of(ticks: &[i64]) -> Option<(i64, i64)> {
    let lowest = ticks.iter().copied().min()?;
    let highest = ticks.iter().copied().max()?;
    Some((lowest, highest))
}

pub(super) fn joined((lo_a, hi_a): (i64, i64), (lo_b, hi_b): (i64, i64)) -> (i64, i64) {
    (lo_a.min(lo_b), hi_a.max(hi_b))
}

/// The first row whose distances no width holds: `m·R² ≥ 2¹²⁸` over the present ticks from the start
/// through the row's own window and over `also`. `windows` if there is none. The range only
/// grows, so every row from it on has no neighbor.
pub(super) fn first_unrepresentable_row(
    series: &TickSeries,
    also: Option<(i64, i64)>,
    m: usize,
) -> usize {
    let (mut lo, mut hi) = also.unwrap_or((i64::MAX, i64::MIN));
    for (end, (&tick, &was_present)) in series.ticks.iter().zip(&series.present).enumerate() {
        if was_present {
            lo = lo.min(tick);
            hi = hi.max(tick);
        }
        if end + 1 >= m && row_width(lo, hi, m) == RowWidth::Unrepresentable {
            return end + 1 - m;
        }
    }
    (series.ticks.len() + 1).saturating_sub(m)
}

/// The nearest later neighbor of every window, from the pairs every tile offers: one candidate per
/// column under a lock per [`RIGHT_GROUP`] columns, the smallest `(squared distance, index)`.
struct RightMins {
    groups: ColumnGroups<Vec<Candidate>>,
}

/// A candidate neighbor: its exact squared distance and window start, [`NO_CANDIDATE`] for none.
pub(super) type Candidate = (u128, u64);

const NO_CANDIDATE: Candidate = (u128::MAX, u64::MAX);

impl RightMins {
    fn new(windows: usize) -> Self {
        Self::try_new(windows).unwrap_or_else(|e| abort_on_capacity(e))
    }

    /// [`Self::new`], or a capacity error when the candidates cannot be allocated.
    fn try_new(windows: usize) -> Result<Self, CapacityError> {
        let groups = ColumnGroups::try_new(windows, |columns| {
            Ok::<_, CapacityError>(vec![NO_CANDIDATE; columns.len()])
        })?;
        Ok(Self { groups })
    }

    /// Folds in a tile's candidates for columns `first..first + part.len()`.
    #[inline(never)]
    fn merge(&self, first: usize, part: &[Candidate]) {
        self.groups
            .merge(first..first + part.len(), |mins, shared| {
                let at = shared.start % RIGHT_GROUP;
                for (min, &candidate) in mins[at..]
                    .iter_mut()
                    .zip(&part[shared.start - first..shared.end - first])
                {
                    *min = (*min).min(candidate);
                }
            });
    }

    fn finish(self) -> Vec<NonNormNeighbor> {
        let mut out = Vec::new();
        self.groups.flatten(&mut out, |mins, out| {
            out.extend(
                mins.into_iter()
                    .map(|(squared_ticks, index)| neighbor(squared_ticks, index)),
            );
        });
        out
    }
}

pub(super) fn neighbor(squared_ticks: u128, index: u64) -> NonNormNeighbor {
    match SampleIndex::new(index) {
        Some(index) => NonNormNeighbor {
            squared_ticks,
            index,
        },
        None => NonNormNeighbor::NONE,
    }
}

/// What the tile engine needs of a keeper to run its tiles on several threads: `Sync` when the
/// build has threads, nothing when it has none.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) trait Shared: Sync {}

#[cfg(all(feature = "parallel", not(loom)))]
impl<T: Sync> Shared for T {}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) trait Shared {}

#[cfg(not(all(feature = "parallel", not(loom))))]
impl<T> Shared for T {}

/// What a tile keeps of the pairs it walks, and how the tiles' keeps merge into a row's finished
/// entries: the one nearest neighbor ([`Single`]) or the `k` nearest ([`topk::TopK`]). The
/// walk itself is the same for both; a keeper owns only the per-row partial, the per-column
/// candidates a tile offers to the right side, and how a finished row lays out its entries.
pub(super) trait Keep: Shared {
    /// A row's partial: what one tile, or the tiles merged so far, found for it.
    type Row: Send;
    /// A tile's candidates for the right side of its columns.
    type Columns: Send;
    /// One finished entry of a row.
    type Out: Send;

    fn new_row(&self) -> Self::Row;
    /// Empties `row`'s partial for its next tile or band.
    fn clear(&self, row: &mut Self::Row);
    fn offer(&self, row: &mut Self::Row, candidate: Candidate);
    /// Folds `from` into `into`, leaving `from` empty; order-independent.
    fn merge(&self, into: &mut Self::Row, from: &mut Self::Row);
    /// Appends row `index`'s finished entries, emptying its partial.
    fn finish(&self, index: usize, row: &mut Self::Row, out: &mut Vec<(u64, Self::Out)>);
    /// The entry of a row with no neighbor.
    fn none() -> Self::Out;

    fn new_columns(&self) -> Self::Columns;
    /// Whether the tiles offer to the right side at all.
    fn wants_columns(&self) -> bool;
    /// Empties the candidates for a tile of `count` columns and `rows` rows.
    fn reset_columns(&self, columns: &mut Self::Columns, count: usize, rows: usize);
    fn offer_column(&self, columns: &mut Self::Columns, slot: usize, candidate: Candidate);
    /// Folds a tile's candidates for columns `first..` into the shared right side.
    fn merge_columns(&self, first: usize, columns: &mut Self::Columns);

    /// Steps and offers one row of a `u64` tile's lanes: by default the lane kernel only steps
    /// them, and every complete pair goes through [`Self::offer`] and [`Self::offer_column`].
    fn lane_row(&self, partial: &mut Self::Row, columns: &mut Self::Columns, lanes: LaneRow) {
        if lanes.advance {
            nonnorm_lanes::ascending(
                lanes.residues,
                lanes.step,
                lanes.lanes,
                lanes.fit,
                Pass::Advance,
                None,
            );
        }
        if !lanes.offer {
            return;
        }
        let offers_columns = self.wants_columns();
        let pairs = lanes.residues.iter().zip(lanes.lanes.complete);
        for (i, (&squared, &complete)) in pairs.enumerate() {
            if complete {
                let column = lanes.column + i;
                self.offer(partial, (u128::from(squared), column as u64));
                if offers_columns {
                    let candidate = (u128::from(squared), lanes.row as u64);
                    self.offer_column(columns, lanes.slot + i, candidate);
                }
            }
        }
    }
}

/// One row of a `u64` tile's lanes: lane `i` pairs window `row` with window `column + i`, the
/// tile's column slot `slot + i`.
pub(super) struct LaneRow<'a> {
    residues: &'a mut [u64],
    step: TickStep,
    lanes: Lanes<'a>,
    fit: Fit,
    /// Whether the residues step from the previous row first; otherwise they already hold this
    /// row's pairs.
    advance: bool,
    /// Whether the pairs are offered: the row's own window is complete.
    offer: bool,
    row: usize,
    column: usize,
    slot: usize,
}

/// The one nearest neighbor of each row, and with `right` the one nearest later neighbor of each
/// column.
pub(super) struct Single<'a> {
    right: Option<&'a RightMins>,
}

/// A tile's nearest row per column: `candidates` from a `u128` tile, `squared` and `rows` (no row
/// where `squared` is `u64::MAX`) from a `u64` one.
#[derive(Default)]
pub(super) struct SingleColumns {
    candidates: Vec<Candidate>,
    squared: Vec<u64>,
    rows: Vec<u64>,
}

impl Keep for Single<'_> {
    type Row = NearestSquared;
    type Columns = SingleColumns;
    type Out = NonNormNeighbor;

    fn new_row(&self) -> NearestSquared {
        NearestSquared::default()
    }

    fn clear(&self, row: &mut NearestSquared) {
        *row = NearestSquared::default();
    }

    #[inline(always)]
    fn offer(&self, row: &mut NearestSquared, candidate: Candidate) {
        row.offer(candidate);
    }

    fn merge(&self, into: &mut NearestSquared, from: &mut NearestSquared) {
        into.merge(from);
    }

    fn finish(
        &self,
        index: usize,
        row: &mut NearestSquared,
        out: &mut Vec<(u64, NonNormNeighbor)>,
    ) {
        out.push((index as u64, core::mem::take(row).finish()));
    }

    fn none() -> NonNormNeighbor {
        NonNormNeighbor::NONE
    }

    fn new_columns(&self) -> SingleColumns {
        SingleColumns::default()
    }

    fn wants_columns(&self) -> bool {
        self.right.is_some()
    }

    fn reset_columns(&self, columns: &mut SingleColumns, count: usize, _rows: usize) {
        for part in [&mut columns.squared, &mut columns.rows] {
            part.clear();
            if self.right.is_some() {
                part.resize(count, u64::MAX);
            }
        }
        columns.candidates.clear();
        if self.right.is_some() {
            columns.candidates.resize(count, NO_CANDIDATE);
        }
    }

    #[inline(always)]
    fn offer_column(&self, columns: &mut SingleColumns, slot: usize, candidate: Candidate) {
        let part = &mut columns.candidates[slot];
        *part = (*part).min(candidate);
    }

    fn merge_columns(&self, first: usize, columns: &mut SingleColumns) {
        if let Some(right) = self.right {
            let lanes = columns.squared.iter().zip(&columns.rows);
            for (part, (&squared, &row)) in columns.candidates.iter_mut().zip(lanes) {
                if squared != u64::MAX {
                    *part = (*part).min((u128::from(squared), row));
                }
            }
            right.merge(first, &columns.candidates);
        }
    }

    #[inline(always)]
    fn lane_row(&self, partial: &mut NearestSquared, columns: &mut SingleColumns, lanes: LaneRow) {
        let pass = match (lanes.advance, lanes.offer) {
            (true, true) => Pass::AdvanceAndRank,
            (true, false) => Pass::Advance,
            (false, true) => Pass::Rank,
            (false, false) => return,
        };
        let slots = lanes.slot..lanes.slot + lanes.residues.len();
        let mut mins = (self.right.is_some() && lanes.offer).then(|| ColumnMins {
            squared: &mut columns.squared[slots.clone()],
            rows: &mut columns.rows[slots],
            row: lanes.row as u64,
        });
        let nearest = nonnorm_lanes::ascending(
            lanes.residues,
            lanes.step,
            lanes.lanes,
            lanes.fit,
            pass,
            mins.as_mut(),
        );
        if let Some((squared, i)) = nearest {
            partial.offer((u128::from(squared), (lanes.column + i) as u64));
        }
    }
}

/// What a tile works in: a partial per band row, and the candidates it found for the right side.
pub(super) struct Scratch<K: Keep> {
    rows: Vec<K::Row>,
    first_column: usize,
    columns: K::Columns,
    /// A `u64` tile's residues, one per lane of a chunk.
    residues: Vec<u64>,
}

/// The non-normalized metric's tile work over the windows of `rows` against those of `cols`.
pub(super) struct Tiles<'a, K> {
    rows: &'a TickSeries,
    cols: &'a TickSeries,
    m: usize,
    shape: Shape,
    /// Rows from here on have no neighbor.
    unrepresentable_from: usize,
    keep: K,
    /// The diagonals to walk, `None` for all.
    runs: Option<&'a SlotRuns>,
}

impl<K: Keep> Tiles<'_, K> {
    /// The width the tile's pairs are exact in, by the range of every tick its two sides hold,
    /// and how far apart those ticks are.
    fn width(
        &self,
        rows: core::ops::Range<usize>,
        columns: &core::ops::Range<usize>,
    ) -> (RowWidth, Fit) {
        let m = self.m;
        let row_ticks = &self.rows.ticks[rows.start.saturating_sub(1)..rows.end - 1 + m];
        let column_ticks = &self.cols.ticks[columns.start..columns.end - 1 + m];
        match (range_of(row_ticks), range_of(column_ticks)) {
            (Some(a), Some(b)) => {
                let (lo, hi) = joined(a, b);
                (routed_row_width(lo, hi, m), Fit::of(lo, hi))
            }
            _ => (RowWidth::Narrow, Fit::I32),
        }
    }

    /// The tile's pairs in `u64` through the lane kernel: each run of its slots in chunks of
    /// `chunk_lanes`, every row of a chunk in turn. Lane `i` of a chunk ending at slot `end` is
    /// slot `end − 1 − i`, so its column at row `r` is `r + offset(end − 1) + i` and a row's lanes
    /// are consecutive columns. A lane starts from the exact sum at the tile's first row and where
    /// its column is window 0, and steps with the recurrence everywhere else.
    fn walk_lanes(
        &self,
        tile: &Tile,
        rows: core::ops::Range<usize>,
        scratch: &mut Scratch<K>,
        chunk_lanes: usize,
        fit: Fit,
    ) {
        let m = self.m;
        let (a, b) = (&self.rows.ticks, &self.cols.ticks);
        let n_cols = self.cols.windows() as i64;
        let chunk_lanes = chunk_lanes.max(1);
        let whole = self.runs.is_none().then_some(tile.qa..tile.qb);
        let chosen = self
            .runs
            .into_iter()
            .flat_map(|runs| runs.within(tile.qa, tile.qb));
        for run in whole.into_iter().chain(chosen) {
            for start in run.clone().step_by(chunk_lanes) {
                let end = (start + chunk_lanes).min(run.end);
                let width = end - start;
                if scratch.residues.len() < width {
                    scratch.residues.resize(width, 0);
                }
                let base = self.shape.offset(end - 1);
                for row in rows.clone() {
                    let at = row as i64 + base;
                    let lo = (-at).clamp(0, width as i64) as usize;
                    let hi = (n_cols - at).clamp(0, width as i64) as usize;
                    if lo >= hi {
                        continue;
                    }
                    let column = (at + lo as i64) as usize;
                    let first = row == rows.start;
                    if first {
                        for (i, residue) in scratch.residues[lo..hi].iter_mut().enumerate() {
                            *residue = start_residue(a, b, row, column + i, m);
                        }
                    } else if column == 0 {
                        // Lane `lo` meets window 0 here: it starts, and the others step.
                        scratch.residues[lo] = start_residue(a, b, row, 0, m);
                        let stepping = (lo + 1..hi, 1);
                        self.lane_row(tile, scratch, row, stepping, true, false, fit);
                    }
                    let advance = !first && column != 0;
                    let offer = self.rows.complete[row];
                    self.lane_row(tile, scratch, row, (lo..hi, column), advance, offer, fit);
                }
            }
        }
    }

    /// Row `row`'s lanes `lanes`, the first at column `column`, through the keeper.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn lane_row(
        &self,
        tile: &Tile,
        scratch: &mut Scratch<K>,
        row: usize,
        (lanes, column): (core::ops::Range<usize>, usize),
        advance: bool,
        offer: bool,
        fit: Fit,
    ) {
        let (a, b, m) = (&self.rows.ticks, &self.cols.ticks, self.m);
        let count = lanes.len();
        if count == 0 || !(advance || offer) {
            return;
        }
        let enter = &b[column + m - 1..column + m - 1 + count];
        let (step, leave) = if advance {
            let step = TickStep {
                entering: a[row + m - 1],
                leaving: a[row - 1],
            };
            (step, &b[column - 1..column - 1 + count])
        } else {
            // Nothing steps, so the ticks only fill the shape.
            let unused = TickStep {
                entering: 0,
                leaving: 0,
            };
            (unused, enter)
        };
        let lane_row = LaneRow {
            residues: &mut scratch.residues[lanes],
            step,
            lanes: Lanes {
                enter,
                leave,
                complete: &self.cols.complete[column..column + count],
            },
            fit,
            advance,
            offer,
            row,
            column,
            slot: column - scratch.first_column,
        };
        self.keep.lane_row(
            &mut scratch.rows[row - tile.rows.start],
            &mut scratch.columns,
            lane_row,
        );
    }

    /// Every diagonal of the tile down its rows, into the band's partials (and the right side):
    /// the `u128` tiles' walk, and in `u64` the oracle the lane walk is checked against.
    fn walk<R: Residue>(
        &self,
        tile: &Tile,
        rows: core::ops::Range<usize>,
        scratch: &mut Scratch<K>,
    ) {
        let m = self.m;
        let offers_columns = self.keep.wants_columns();
        let (a, b): (&[i64], &[i64]) = (&self.rows.ticks, &self.cols.ticks);
        let (a_complete, b_complete): (&[bool], &[bool]) =
            (&self.rows.complete, &self.cols.complete);
        let unused = TickStep {
            entering: 0,
            leaving: 0,
        };
        for slot in tile.slots(self.runs) {
            let offset = self.shape.offset(slot);
            let first = (rows.start as i64).max(-offset);
            let end = (rows.end as i64).min(self.cols.windows() as i64 - offset);
            if first >= end {
                continue;
            }
            let (first, end) = (first as usize, end as usize);
            let column = (first as i64 + offset) as usize;
            let mut residue = NonNormalized::<R>::start(
                &a[first..first + m],
                &unused,
                &b[column..column + m],
                &unused,
            );
            for row in first..end {
                let column = (row as i64 + offset) as usize;
                if row > first {
                    residue = NonNormalized::<R>::advance(
                        residue,
                        &TickStep {
                            entering: a[row + m - 1],
                            leaving: a[row - 1],
                        },
                        &TickStep {
                            entering: b[column + m - 1],
                            leaving: b[column - 1],
                        },
                    );
                }
                if a_complete[row] && b_complete[column] {
                    let squared = residue.as_u128();
                    self.keep.offer(
                        &mut scratch.rows[row - tile.rows.start],
                        (squared, column as u64),
                    );
                    if offers_columns {
                        self.keep.offer_column(
                            &mut scratch.columns,
                            column - scratch.first_column,
                            (squared, row as u64),
                        );
                    }
                }
            }
        }
    }
}

impl<K: Keep> TileWork for Tiles<'_, K> {
    type Scratch = Scratch<K>;
    type Partial = K::Row;
    type Row = K::Out;

    fn scratch(&self, plan: TilePlan) -> Scratch<K> {
        Scratch {
            rows: (0..plan.band_rows).map(|_| self.keep.new_row()).collect(),
            first_column: 0,
            columns: self.keep.new_columns(),
            residues: Vec::new(),
        }
    }

    fn partials<'a>(&self, scratch: &'a mut Scratch<K>) -> &'a mut [K::Row] {
        &mut scratch.rows
    }

    fn run(&self, tile: &Tile, scratch: &mut Scratch<K>, chunk_lanes: usize) -> RanWide {
        let columns = self.shape.columns(tile);
        let rows = tile.rows.start..tile.rows.end.min(self.unrepresentable_from);
        for partial in &mut scratch.rows {
            self.keep.clear(partial);
        }
        if rows.is_empty() || columns.is_empty() {
            return false;
        }
        scratch.first_column = columns.start;
        self.keep
            .reset_columns(&mut scratch.columns, columns.len(), tile.rows.len());
        let (width, fit) = self.width(rows.clone(), &columns);
        match width {
            RowWidth::Narrow => self.walk_lanes(tile, rows, scratch, chunk_lanes, fit),
            _ => self.walk::<u128>(tile, rows, scratch),
        }
        if self.keep.wants_columns() {
            self.keep
                .merge_columns(scratch.first_column, &mut scratch.columns);
        }
        width != RowWidth::Narrow
    }

    fn new_partial(&self) -> K::Row {
        self.keep.new_row()
    }

    fn begin(&self, partial: &mut K::Row, _row: usize) {
        self.keep.clear(partial);
    }

    fn merge(&self, into: &mut K::Row, from: &mut K::Row) {
        self.keep.merge(into, from);
    }

    fn finish(&self, row: usize, partial: &mut K::Row, out: &mut Vec<(u64, K::Out)>) {
        self.keep.finish(row, partial, out);
    }

    fn push_none_rows(&self, count: usize, out: &mut Vec<(u64, K::Out)>) {
        push_none_rows(count, out, K::none);
    }

    #[cfg(test)]
    fn visits(&self, _partial: &K::Row) -> Vec<usize> {
        Vec::new()
    }
}

/// The exact residue of window `row` of `a` against window `column` of `b`, in `u64`.
fn start_residue(a: &[i64], b: &[i64], row: usize, column: usize, m: usize) -> u64 {
    let unused = TickStep {
        entering: 0,
        leaving: 0,
    };
    NonNormalized::<u64>::start(&a[row..row + m], &unused, &b[column..column + m], &unused)
}

/// The tile shape `plan` fixes, or the one [`TilePlan::for_threads`] cuts for `threads`.
pub(super) fn left_plan(
    windows: usize,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
) -> TilePlan {
    plan.unwrap_or_else(|| {
        TilePlan::for_threads(windows, m.get(), threads.get(), default_chunk_lanes())
    })
}

/// The profile of `ticks` against itself, kept by `keep` into `results`; `keep` also receives
/// every pair for the earlier window's right side when it asks for one.
pub(super) fn left_rows<K: Keep>(
    ticks: &TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
    runs: Option<&SlotRuns>,
    keep: K,
    mut results: Vec<(u64, K::Out)>,
) -> (Vec<(u64, K::Out)>, BatchReport) {
    let windows = ticks.windows();
    results.reserve_exact(windows.saturating_sub(results.len()));
    let tiles = Tiles {
        rows: ticks,
        cols: ticks,
        m: m.get(),
        shape: Shape::Triangle {
            exclusion: crate::exclusion::zone(m.get()),
        },
        unrepresentable_from: first_unrepresentable_row(ticks, None, m.get()),
        keep,
        runs,
    };
    let cut = TileCut::triangle(windows, m.get(), left_plan(windows, m, threads, plan), runs);
    let report = run_tile_work(&tiles, cut, threads, &mut results);
    (results, report)
}

/// The exact non-normalized left profile of `series`: per window start `i`, in order, the nearest
/// of the earlier windows `j ≤ i − ⌈m/4⌉ − 1` by squared Euclidean distance on the raw ticks
/// (`series` over `scale`), ties to the smaller `j`, on up to `threads` threads. A window with a
/// gap, or with no eligible neighbor, has none; a gap window is never a neighbor. A window whose
/// distances no `u128` holds (`m·R² ≥ 2¹²⁸` over every tick through its own window, which needs
/// `m` beyond 2²⁰ at the widest ticks; a missing sample is not a tick) has none either. Equal on bits to
/// the exact reference, whatever `threads` is.
pub fn left_profile(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Vec<(u64, NonNormNeighbor)> {
    left_profile_report(series, m, scale, threads).0
}

/// [`left_profile`], and how many independent tiles ran and how many of them ran in `u128`
/// because their own range was too wide for `u64`.
#[doc(hidden)]
pub fn left_profile_report(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, NonNormNeighbor)>, BatchReport) {
    let threads = threads_for_size(series.len(), m, threads);
    left_rows(
        &TickSeries::build(series, m, scale),
        m,
        threads,
        None,
        None,
        Single { right: None },
        Vec::new(),
    )
}

/// [`left_profile_report`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes (rounded up to a whole number of `chunk_lanes`).
/// Neither they nor `threads` change a bit.
#[doc(hidden)]
pub fn left_profile_report_tiled(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> (Vec<(u64, NonNormNeighbor)>, BatchReport) {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    left_rows(
        &TickSeries::build(series, m, scale),
        m,
        threads,
        Some(plan),
        None,
        Single { right: None },
        Vec::new(),
    )
}

/// The exact non-normalized two-sided profile of `series`, on up to `threads` threads: per window
/// start, in order, the left neighbor (exactly [`left_profile`]'s), the right neighbor, and the
/// nearer of the two, ties to the smaller index. Looking right trades on future information. One
/// tiled pass offers each pair to both windows. Equal on bits to the exact reference, whatever
/// `threads` is.
pub fn two_sided_profile(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Vec<(u64, NonNormTwoSided)> {
    let threads = threads_for_size(series.len(), m, threads);
    two_sided_rows(&TickSeries::build(series, m, scale), m, threads, None).0
}

/// [`two_sided_profile`] with the tile shape as parameters, for tests and tuning: as
/// [`left_profile_report_tiled`].
#[doc(hidden)]
pub fn two_sided_profile_tiled(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> (Vec<(u64, NonNormTwoSided)>, BatchReport) {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    two_sided_rows(&TickSeries::build(series, m, scale), m, threads, Some(plan))
}

/// The left and the right nearest neighbor of every window over only `runs`' diagonals, each the
/// exact nearest of the pairs on them (ties to the smaller index), on up to `threads` threads.
pub(crate) fn two_sided_over(
    ticks: &TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    runs: &SlotRuns,
) -> (Vec<NonNormNeighbor>, Vec<NonNormNeighbor>) {
    let threads = threads_for_size(ticks.ticks.len(), m, threads);
    let book = RightMins::new(ticks.windows());
    let (left, _) = left_rows(
        ticks,
        m,
        threads,
        None,
        Some(runs),
        Single { right: Some(&book) },
        Vec::new(),
    );
    (
        left.into_iter().map(|(_, neighbor)| neighbor).collect(),
        book.finish(),
    )
}

/// [`two_sided_profile_tiled`] over only the self-join pairs on the diagonals `diagonals` (a
/// diagonal is `j − i` of a pair, past the exclusion zone; any other number is ignored), for tests
/// and tuning: each side's nearest over those pairs, and the nearer of the two.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn two_sided_profile_over_tiled(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    diagonals: &[i64],
) -> (Vec<(u64, NonNormTwoSided)>, BatchReport) {
    let ticks = TickSeries::build(series, m, scale);
    let past = crate::exclusion::zone(m.get()) + 1;
    let slots = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(d).ok()?.checked_sub(past))
        .collect();
    let runs = SlotRuns::from_slots(slots);
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let book = RightMins::new(ticks.windows());
    let (left, report) = left_rows(
        &ticks,
        m,
        threads,
        Some(plan),
        Some(&runs),
        Single { right: Some(&book) },
        Vec::new(),
    );
    let profile = left
        .into_iter()
        .zip(book.finish())
        .map(|((row, left), right)| {
            let sides = NonNormTwoSided {
                left,
                right,
                nearest: left.nearer(right),
            };
            (row, sides)
        })
        .collect();
    (profile, report)
}

fn two_sided_rows(
    ticks: &TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
) -> (Vec<(u64, NonNormTwoSided)>, BatchReport) {
    let book = RightMins::new(ticks.windows());
    let (left, report) = left_rows(
        ticks,
        m,
        threads,
        plan,
        None,
        Single { right: Some(&book) },
        Vec::new(),
    );
    let profile = left
        .into_iter()
        .zip(book.finish())
        .map(|((row, left), right)| {
            let nearest = match (left.index.get(), right.index.get()) {
                (None, _) => right,
                (_, None) => left,
                // The left window starts before the right one, so equal distances go left.
                _ if right.squared_ticks < left.squared_ticks => right,
                _ => left,
            };
            (
                row,
                NonNormTwoSided {
                    left,
                    right,
                    nearest,
                },
            )
        })
        .collect();
    (profile, report)
}

/// Every window of `a` against every window of `b` under squared Euclidean distance on raw
/// ticks, both over the same `scale` (their distances only compare in one unit), no exclusion
/// zone: per `a` window, in order, its exact nearest `b` window (ties to the smaller index; none
/// for a window with a gap, or when `b` has no complete window), on up to `threads` threads. A
/// window of `a` has none too when `m·R² ≥ 2¹²⁸` over the ticks of `a` through its own window
/// and all of `b`. Equal on bits to the exact reference, whatever `threads` is.
pub fn ab_join(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Vec<(u64, NonNormNeighbor)> {
    ab_join_report(a, b, m, scale, threads).0
}

/// [`ab_join`], and how many independent tiles ran and how many of them ran in `u128` because
/// their own range was too wide for `u64`.
#[doc(hidden)]
pub fn ab_join_report(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, NonNormNeighbor)>, BatchReport) {
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    let threads = threads_for_join(windows(a.len()), windows(b.len()), threads);
    join_rows(
        &TickSeries::build(a, m, scale),
        &TickSeries::build(b, m, scale),
        m,
        threads,
        None,
        None,
    )
}

/// [`ab_join_report`] with the tile shape as parameters, for tests and tuning: as
/// [`left_profile_report_tiled`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_report_tiled(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> (Vec<(u64, NonNormNeighbor)>, BatchReport) {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    join_rows(
        &TickSeries::build(a, m, scale),
        &TickSeries::build(b, m, scale),
        m,
        threads,
        Some(plan),
        None,
    )
}

/// Every window of `a`'s nearest window of `b` over only `runs`' diagonals (a diagonal is a `b`
/// window minus an `a` window), each the exact nearest of the pairs on them (ties to the smaller
/// index), on up to `threads` threads.
pub(crate) fn ab_join_over(
    a: &TickSeries,
    b: &TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    runs: &SlotRuns,
) -> Vec<NonNormNeighbor> {
    let threads = threads_for_join(a.windows(), b.windows(), threads);
    let (rows, _) = join_rows(a, b, m, threads, None, Some(runs));
    rows.into_iter().map(|(_, neighbor)| neighbor).collect()
}

/// [`ab_join_report_tiled`] over only the pairs on the diagonals `diagonals` (a diagonal is a `b`
/// window minus an `a` window; any other number is ignored), for tests and tuning.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_over_tiled(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    diagonals: &[i64],
) -> (Vec<(u64, NonNormNeighbor)>, BatchReport) {
    let (a, b) = (
        TickSeries::build(a, m, scale),
        TickSeries::build(b, m, scale),
    );
    let last = b.windows() as i64 - 1;
    let slots = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(last.checked_sub(d)?).ok())
        .collect();
    let runs = SlotRuns::from_slots(slots);
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    join_rows(&a, &b, m, threads, Some(plan), Some(&runs))
}

fn join_rows(
    a: &TickSeries,
    b: &TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
    runs: Option<&SlotRuns>,
) -> (Vec<(u64, NonNormNeighbor)>, BatchReport) {
    join_kept(
        a,
        b,
        m,
        threads,
        plan,
        runs,
        Single { right: None },
        Vec::new(),
    )
}

/// The join of `a` against `b`, kept by `keep` into `results`.
#[allow(clippy::too_many_arguments)]
pub(super) fn join_kept<K: Keep>(
    a: &TickSeries,
    b: &TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
    runs: Option<&SlotRuns>,
    keep: K,
    mut results: Vec<(u64, K::Out)>,
) -> (Vec<(u64, K::Out)>, BatchReport) {
    let (n_a, n_b) = (a.windows(), b.windows());
    results.reserve_exact(n_a.saturating_sub(results.len()));
    if n_a == 0 || n_b == 0 {
        push_none_rows(n_a, &mut results, K::none);
        let report = BatchReport::of(0, 0);
        return (results, report);
    }
    let tiles = Tiles {
        rows: a,
        cols: b,
        m: m.get(),
        shape: Shape::Join { n_b },
        unrepresentable_from: first_unrepresentable_row(a, b.present_range(), m.get()),
        keep,
        runs,
    };
    let plan = plan.unwrap_or_else(|| {
        TilePlan::for_join(n_a, n_b, m.get(), threads.get(), default_chunk_lanes())
    });
    let cut = TileCut::join(n_a, n_b, plan, runs);
    let report = run_tile_work(&tiles, cut, threads, &mut results);
    (results, report)
}

#[cfg(test)]
mod tests;
