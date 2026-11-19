//! The non-normalized two-sided profile and AB-join with the `k` nearest neighbors per window: the
//! same tile walk as the one-neighbor profiles, with a keeper ([`TopK`]) that holds each row's `k`
//! smallest `(squared distance, index)` pairs in order instead of one. The order is total over
//! exact integers, so no tolerance or near-tie pass exists: the kept lists are the `k` smallest
//! pairs, ties to the smaller index, whatever the thread count and tile shape.

use super::super::kept::{finite_from, group_by_row, join_list_len, side_counts};
use super::super::plan::threads_for_size;
use super::super::plan::{PARTIALS_BYTES, TilePlan, default_chunk_lanes, threads_for_join};
use super::super::schedule::{BatchReport, ColumnGroups, RIGHT_GROUP};
use super::{
    Candidate, Keep, RightMins, Single, TickSeries, join_kept, left_plan, left_rows, neighbor,
};
use crate::capacity::{CapacityError, clamp_count, reserve_for, reserved_vec};
use crate::kbest;
use crate::kernel::Reducer;
use crate::nonnorm::{NonNormNeighbor, NonNormTwoSidedTopk};
use crate::quantize::Scale;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// The grouped top-k join and the run's report, or the capacity refusal.
type TiledJoin = Result<(Vec<(u64, Vec<NonNormNeighbor>)>, BatchReport), CapacityError>;

/// Inserts `candidate` into the ascending `list`, which keeps at most `limit`: the largest falls
/// off a full list, and a candidate no smaller than a full list's largest is dropped.
#[inline(always)]
fn keep_nearest(list: &mut Vec<Candidate>, limit: usize, candidate: Candidate) {
    kbest::keep(list, limit, candidate, nearer);
}

/// Whether `a` is nearer than `b`: the smaller squared distance, then the smaller index.
#[inline(always)]
fn nearer(a: &Candidate, b: &Candidate) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// One row's `k` nearest as a reducer over a list the caller owns and reuses, so a stream's row
/// allocates only while the neighbors it can have (`bound`) are still fewer than `k`. The list ends
/// ascending.
pub(crate) struct KeptNearest<'a> {
    list: &'a mut Vec<Candidate>,
    limit: usize,
}

impl<'a> KeptNearest<'a> {
    /// Starts a row on `list` (cleared) that keeps at most `limit` of at most `bound` pairs.
    pub(crate) fn new(list: &'a mut Vec<Candidate>, limit: usize, bound: usize) -> Self {
        list.clear();
        list.reserve(reserve_for(limit, bound));
        Self { list, limit }
    }
}

impl Reducer for KeptNearest<'_> {
    type Candidate = Candidate;
    type Output = ();

    #[inline]
    fn offer(&mut self, candidate: Candidate) {
        kbest::keep_rarely(self.list, self.limit, candidate, nearer);
    }

    fn finish(self) {}
}

/// `kept` as neighbors, replacing what `out` held; `limit` and `bound` are as for
/// [`KeptNearest::new`].
pub(crate) fn write_neighbors(
    kept: &[Candidate],
    out: &mut Vec<NonNormNeighbor>,
    limit: usize,
    bound: usize,
) {
    out.clear();
    out.reserve(reserve_for(limit, bound));
    out.extend(
        kept.iter()
            .map(|&(squared, index)| neighbor(squared, index)),
    );
}

/// Folds the ascending `from` into the ascending `into`, which keeps at most `limit`.
fn merge_nearest(into: &mut Vec<Candidate>, limit: usize, from: &[Candidate]) {
    for &candidate in from {
        if into.len() >= limit && into.last().is_some_and(|&worst| candidate >= worst) {
            break;
        }
        keep_nearest(into, limit, candidate);
    }
}

/// A tile's candidates for the right side, one ascending list per column of the tile.
pub(super) struct ColumnLists {
    lists: Vec<Vec<Candidate>>,
    count: usize,
}

/// Every window's `k` nearest later neighbors, from the pairs every tile offers: an ascending list
/// per column under a lock per [`RIGHT_GROUP`] columns. A list is reserved for the most it can
/// ever hold, the finite windows that can reach its column, so nothing grows after the start.
pub(super) struct RightTops {
    k: usize,
    groups: ColumnGroups<Vec<Vec<Candidate>>>,
}

impl RightTops {
    pub(super) fn new(
        series: &TickSeries,
        k: NonZeroUsize,
        exclusion: usize,
    ) -> Result<Self, CapacityError> {
        let windows = series.windows();
        let finite_from = finite_from(windows, |i| series.complete[i]);
        let groups = ColumnGroups::try_new(windows, |columns| {
            let mut lists: Vec<Vec<Candidate>> = Vec::new();
            lists
                .try_reserve_exact(columns.len())
                .map_err(|_| CapacityError)?;
            for column in columns {
                let mut list = Vec::new();
                if series.complete[column] {
                    let reach = finite_from[(column + exclusion + 1).min(windows)];
                    list.try_reserve_exact(k.get().min(reach))
                        .map_err(|_| CapacityError)?;
                }
                lists.push(list);
            }
            Ok::<_, CapacityError>(lists)
        })?;
        Ok(Self { k: k.get(), groups })
    }

    /// Folds in a tile's lists for columns `first..first + part.len()`.
    #[inline(never)]
    fn merge(&self, first: usize, part: &[Vec<Candidate>]) {
        self.groups
            .merge(first..first + part.len(), |lists, shared| {
                let group_start = shared.start / RIGHT_GROUP * RIGHT_GROUP;
                for column in shared {
                    merge_nearest(
                        &mut lists[column - group_start],
                        self.k,
                        &part[column - first],
                    );
                }
            });
    }

    /// Each window's kept later neighbors, in order, appended to `out` (one empty entry for a
    /// window with none), in the layout [`TopK`] gives a row's entries.
    pub(super) fn finish(
        self,
        mut out: Vec<(u64, NonNormNeighbor)>,
    ) -> Vec<(u64, NonNormNeighbor)> {
        let mut row = 0u64;
        self.groups.flatten(&mut out, |lists, out| {
            for list in lists {
                push_row(row, &list, out);
                row += 1;
            }
        });
        out
    }
}

/// `row`'s kept neighbors as entries, or one empty entry when there are none.
fn push_row(row: u64, kept: &[Candidate], out: &mut Vec<(u64, NonNormNeighbor)>) {
    if kept.is_empty() {
        out.push((row, NonNormNeighbor::NONE));
    } else {
        out.extend(
            kept.iter()
                .map(|&(squared, index)| (row, neighbor(squared, index))),
        );
    }
}

/// The `k` nearest neighbors of each row and, with `right`, the `k` nearest later neighbors of
/// each column.
pub(super) struct TopK<'a> {
    k: usize,
    right: Option<&'a RightTops>,
}

impl Keep for TopK<'_> {
    type Row = Vec<Candidate>;
    type Columns = ColumnLists;
    type Out = NonNormNeighbor;

    fn new_row(&self) -> Vec<Candidate> {
        Vec::new()
    }

    fn clear(&self, row: &mut Vec<Candidate>) {
        row.clear();
    }

    #[inline(always)]
    fn offer(&self, row: &mut Vec<Candidate>, candidate: Candidate) {
        keep_nearest(row, self.k, candidate);
    }

    fn merge(&self, into: &mut Vec<Candidate>, from: &mut Vec<Candidate>) {
        merge_nearest(into, self.k, from);
        from.clear();
    }

    fn finish(
        &self,
        index: usize,
        row: &mut Vec<Candidate>,
        out: &mut Vec<(u64, NonNormNeighbor)>,
    ) {
        push_row(index as u64, &core::mem::take(row), out);
    }

    fn none() -> NonNormNeighbor {
        NonNormNeighbor::NONE
    }

    fn new_columns(&self) -> ColumnLists {
        ColumnLists {
            lists: Vec::new(),
            count: 0,
        }
    }

    fn wants_columns(&self) -> bool {
        self.right.is_some()
    }

    fn reset_columns(&self, columns: &mut ColumnLists, count: usize, _rows: usize) {
        for list in columns.lists.iter_mut().take(count) {
            list.clear();
        }
        if columns.lists.len() < count {
            columns.lists.resize_with(count, Vec::new);
        }
        columns.count = count;
    }

    #[inline(always)]
    fn offer_column(&self, columns: &mut ColumnLists, slot: usize, candidate: Candidate) {
        keep_nearest(&mut columns.lists[slot], self.k, candidate);
    }

    fn merge_columns(&self, first: usize, columns: &mut ColumnLists) {
        if let Some(right) = self.right {
            right.merge(first, &columns.lists[..columns.count]);
        }
    }
}

/// `plan` with its band and tile cut down until the lists a tile can hold fit
/// `PARTIALS_BYTES`: each of a band's rows keeps up to `min(k, tile lanes)` pairs, and each of a
/// tile's `lanes + band` columns up to `min(k, band)`. Only the schedule changes, never a bit, and
/// a thread's scratch stays flat in the series length for any `k`.
fn fitted_plan(plan: TilePlan, k: NonZeroUsize, windows: usize) -> TilePlan {
    let budget = PARTIALS_BYTES / core::mem::size_of::<Candidate>();
    let held = |band: usize, lanes: usize| {
        band.saturating_mul(k.get().min(lanes))
            .saturating_add(lanes.saturating_add(band).saturating_mul(k.get().min(band)))
    };
    let (mut band, mut lanes) = (plan.band_rows.min(windows.max(1)), plan.tile_lanes);
    while held(band, lanes) > budget && (band > 1 || lanes > 1) {
        if lanes >= band {
            lanes = (lanes / 2).max(1);
        } else {
            band = (band / 2).max(1);
        }
    }
    TilePlan {
        band_rows: band,
        chunk_lanes: plan.chunk_lanes.min(lanes),
        tile_lanes: lanes,
        k,
        ..plan
    }
}

/// `k` limited to the windows of `series` that can be a neighbor (at least one).
fn clamp_k(series: &TickSeries, k: NonZeroUsize) -> NonZeroUsize {
    let complete = series.complete.iter().filter(|&&complete| complete).count();
    NonZeroUsize::new(clamp_count(k.get(), complete)).unwrap_or(NonZeroUsize::MIN)
}

/// The exact non-normalized two-sided profile of `series` with the `k` nearest neighbors per
/// window over both sides (`k` limited to the windows that can be a neighbor), on up to `threads`
/// threads: per window start, in order, the left and right nearest ([`super::two_sided_profile`]'s
/// own) and up to `k` neighbors nearest first over both sides by `(squared distance, index)`, ties
/// to the smaller index; none for a window with a gap or with no eligible neighbor. Equal on bits
/// to the exact reference, whatever `threads` and the tile shape are; with `k` of one, `nearest`
/// holds [`super::two_sided_profile`]'s nearer side.
///
/// `Err(`[`CapacityError`]`)` when an allocation it needs is too large to make: the neighbors to
/// keep (the sum over windows of `min(k, windows that can reach it)`, per side) overflow the
/// address space or the allocator refuses them. That is decided before any work, and no count is
/// refused for its size alone.
pub fn two_sided_topk_profile(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> Result<Vec<(u64, NonNormTwoSidedTopk)>, CapacityError> {
    let threads = threads_for_size(series.len(), m, threads);
    Ok(two_sided_topk_with(series, m, scale, threads, None, k)?.0)
}

/// [`two_sided_topk_profile`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes. Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn two_sided_topk_profile_tiled(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
) -> Result<(Vec<(u64, NonNormTwoSidedTopk)>, BatchReport), CapacityError> {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    two_sided_topk_with(series, m, scale, threads, Some(plan), k)
}

fn two_sided_topk_with(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> Result<(Vec<(u64, NonNormTwoSidedTopk)>, BatchReport), CapacityError> {
    let ticks = TickSeries::build(series, m, scale);
    let windows = ticks.windows();
    let k = clamp_k(&ticks, k);
    let exclusion = crate::exclusion::zone(m.get());
    let (left_count, right_count) = side_counts(windows, |i| ticks.complete[i], k, exclusion);
    let left_out = reserved_vec(left_count.list_len())?;
    let right_out = reserved_vec(right_count.list_len())?;
    let mut profile = reserved_vec(windows as u128)?;
    let plan = fitted_plan(left_plan(windows, m, threads, plan), k, windows);
    if k == NonZeroUsize::MIN {
        // One neighbor per side is the single keeper's walk, whose sides are the same two lists.
        let book = RightMins::try_new(windows)?;
        let keep = Single { right: Some(&book) };
        let (left, report) = left_rows(&ticks, m, threads, Some(plan), None, keep, left_out);
        for ((row, left), right) in left.into_iter().zip(book.finish()) {
            let nearest = nearest_of_both(&[(row, left)], &[(row, right)], k);
            profile.push((
                row,
                NonNormTwoSidedTopk {
                    left,
                    right,
                    nearest,
                },
            ));
        }
        return Ok((profile, report));
    }
    let book = RightTops::new(&ticks, k, exclusion)?;
    let keep = TopK {
        k: k.get(),
        right: Some(&book),
    };
    let (left, report) = left_rows(&ticks, m, threads, Some(plan), None, keep, left_out);
    let right = book.finish(right_out);
    let (mut left, mut right) = (left.as_slice(), right.as_slice());
    while let (Some(&(row, _)), Some(_)) = (left.first(), right.first()) {
        let (left_run, left_rest) = left.split_at(left.iter().take_while(|e| e.0 == row).count());
        let (right_run, right_rest) =
            right.split_at(right.iter().take_while(|e| e.0 == row).count());
        (left, right) = (left_rest, right_rest);
        profile.push((
            row,
            NonNormTwoSidedTopk {
                left: left_run[0].1,
                right: right_run[0].1,
                nearest: nearest_of_both(left_run, right_run, k),
            },
        ));
    }
    Ok((profile, report))
}

/// The up to `k` nearest neighbors in the two ascending runs of one row's entries (a run holding
/// no neighbor is one empty entry): their merge by `(squared distance, index)`. Left windows start
/// before right ones, so an equal pair of distances takes the left, the smaller index.
fn nearest_of_both(
    left: &[(u64, NonNormNeighbor)],
    right: &[(u64, NonNormNeighbor)],
    k: NonZeroUsize,
) -> Vec<NonNormNeighbor> {
    let present = |run: &[(u64, NonNormNeighbor)]| {
        run.iter()
            .map(|&(_, neighbor)| neighbor)
            .filter(|neighbor| neighbor.index.get().is_some())
            .collect::<Vec<_>>()
    };
    let (left, right) = (present(left), present(right));
    let mut merged = Vec::with_capacity(k.get().min(left.len() + right.len()));
    let (mut l, mut r) = (left.iter().peekable(), right.iter().peekable());
    while merged.len() < k.get() {
        let key = |n: &&NonNormNeighbor| (n.squared_ticks, n.index.get());
        let take_left = match (l.peek(), r.peek()) {
            (Some(a), Some(b)) => key(a) <= key(b),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let next = if take_left { l.next() } else { r.next() };
        merged.extend(next.copied());
    }
    merged
}

/// Every window of `a` against every window of `b` under squared Euclidean distance on raw ticks,
/// both over the same `scale`, no exclusion zone, with the `k` nearest `b` windows per `a` window
/// (`k` limited to the windows of `b` that can be a neighbor): per `a` window, in order, up to `k`
/// neighbors nearest first (ties to the smaller index; none for a window with a gap, or when `b`
/// has no complete window). Equal on bits to the exact reference, whatever `threads` and the tile
/// shape are; with `k` of one, each list holds [`super::ab_join`]'s neighbor.
///
/// `Err(`[`CapacityError`]`)` when the neighbors to keep (the complete `a` windows times
/// `min(k, complete `b` windows)`) overflow the address space or the allocator refuses them,
/// decided before any work; no count is refused for its size alone.
pub fn ab_join_topk(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> Result<Vec<(u64, Vec<NonNormNeighbor>)>, CapacityError> {
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    let threads = threads_for_join(windows(a.len()), windows(b.len()), threads);
    Ok(ab_join_topk_with(a, b, m, scale, threads, None, k)?.0)
}

/// [`ab_join_topk`] with the tile shape as parameters, for tests and tuning: bands of `band_rows`
/// rows, tiles of `tile_lanes` slots. Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_topk_tiled(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
) -> TiledJoin {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    ab_join_topk_with(a, b, m, scale, threads, Some(plan), k)
}

fn ab_join_topk_with(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> TiledJoin {
    let (ta, tb) = (
        TickSeries::build(a, m, scale),
        TickSeries::build(b, m, scale),
    );
    let (n_a, n_b) = (ta.windows(), tb.windows());
    let k = clamp_k(&tb, k);
    let complete_a = ta.complete.iter().filter(|&&complete| complete).count();
    let results = reserved_vec(join_list_len(n_a, complete_a, k))?;
    let plan = (n_a > 0 && n_b > 0).then(|| {
        let plan = plan.unwrap_or_else(|| {
            TilePlan::for_join(n_a, n_b, m.get(), threads.get(), default_chunk_lanes())
        });
        fitted_plan(plan, k, n_a)
    });
    let (flat, report) = if k == NonZeroUsize::MIN {
        let keep = Single { right: None };
        join_kept(&ta, &tb, m, threads, plan, None, keep, results)
    } else {
        let keep = TopK {
            k: k.get(),
            right: None,
        };
        join_kept(&ta, &tb, m, threads, plan, None, keep, results)
    };
    let grouped = group_by_row(flat, k, n_a, |neighbor| neighbor.index.get().is_some());
    Ok((grouped, report))
}

#[cfg(test)]
mod tests;
