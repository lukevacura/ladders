use super::cold::{Cold, JoinCut, JoinSide, SpillReport, wide_cells_join, with_cold_join};
use super::kept::{group_by_row, join_list_len};
use super::left::has_neighbor;
use super::one_loop::join as one_loop_join;
use super::pass::{Shape, finish_rows, push_none_rows};
use super::plan::{
    Engine, SlotRuns, Tile, TileCut, TilePlan, default_chunk_lanes, threads_for_join,
};
use super::quantized::Quantized;
use super::reanchor::{TileSides, drifting_tile_partials};
use super::schedule::{BatchReport, run_tiles};
use super::sparse::walk;
use super::tile::{
    NarrowTile, RanWide, Scratch, WideTile, anchors_of, join_start_column, narrow_tile_partials,
    wide_tile_partials,
};
use super::two_sided::{KeptList, Pairs, merge_kept};
use crate::capacity::{CapacityError, abort_on_capacity, clamp_count, reserved_vec};
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::narrow::{self, Anchor};
use crate::quantize::Scale;
use crate::row::{
    ChunkCols, NarrowColumns, RowAcc, WideChunkCols, WideRow, narrow_row_scalars, pass_chunk,
    pass_chunk_wide, window_at_narrow,
};
use crate::spill::Reach;
use crate::stats::{JoinClass, Series};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// One tile of the AB-join: [`tile_partials`](super::tile::tile_partials) with the rows from `a` and the columns from `b`,
/// each narrowed from its own anchor. Row 0's slots hold the pair one row before it: its exact
/// start minus the entering term (its leaving tick is a zero in front of the row ticks), so the
/// same pass advances it.
fn join_tile_partials(
    (a, b): (&Quantized, &Quantized),
    m: NonZeroUsize,
    chunk_lanes: usize,
    tile: &Tile,
    scratch: &mut Scratch,
    k: NonZeroUsize,
    (runs, joined): (Option<&SlotRuns>, Option<&JoinClass>),
) -> RanWide {
    let mu = m.get();
    let n_b = b.windows.len();
    let shape = Shape::Join { n_b };
    let (r0, r1) = (tile.rows.start, tile.rows.end);
    let columns = shape.columns(tile);
    let (cb, col_end) = (columns.start, columns.end);
    let row_ticks = &a.ticks[r0.saturating_sub(1)..r1 - 1 + mu];
    let col_ticks = &b.ticks[cb..col_end - 1 + mu];
    let anchors = anchors_of(row_ticks, col_ticks, m).filter(|_| crate::route::forced_cell() == 0);
    let Some((row_anchor, col_anchor)) = anchors else {
        let (a_cols, b_cols) = (a.columns(), b.columns());
        let sides = TileSides {
            rows: &a_cols,
            row_windows: &a.windows,
            cols: &b_cols,
            conversion: a.range.joint(b.range).conversion(),
            class: None,
        };
        if drifting_tile_partials(&sides, m, shape, chunk_lanes, tile, scratch, k, None, runs) {
            return false;
        }
        let pairs = CrossTerms::new(&a.ticks[..], 0, &b.ticks[..], 0, mu);
        let row_scalars = |row: usize| WideRow {
            enter: a.ticks[row + mu - 1],
            leave: row.checked_sub(1).map_or(0, |before| a.ticks[before]),
            sum: a.sum[row],
        };
        let start_qt = |slot: usize| {
            join_start_column(n_b, r0, slot).map(|j| match r0.checked_sub(1) {
                Some(before) => pairs.at(before, j - 1),
                None => pairs.before_first_row(j),
            })
        };
        wide_tile_partials(
            &WideTile {
                cols: &b_cols,
                windows: &a.windows,
                row_scalars: &row_scalars,
                pairs,
                start_qt: &start_qt,
                m: mu,
                shape,
                chunk_lanes,
                k,
                runs,
                paa: None,
                class: None,
                joined,
            },
            tile,
            scratch,
            None,
        );
        return true;
    };
    let Scratch {
        qt,
        tile: rho,
        accs,
        scalars,
        row_ticks: rt,
        row_sums: rs,
        col_ticks: ct,
        col_sums: cs,
        ..
    } = scratch;
    narrow::narrow_ticks_into(row_anchor, m, row_ticks, rt);
    if r0 == 0 {
        rt.insert(0, 0);
    }
    narrow::narrow_sums_into(row_anchor, m, &a.sum[r0..r1], rs);
    narrow::narrow_ticks_into(col_anchor, m, col_ticks, ct);
    narrow::narrow_sums_into(col_anchor, m, &b.sum[cb..col_end], cs);
    let conversion = a.range.joint(b.range).conversion();
    let row_cols = NarrowColumns {
        ticks: rt,
        sum: rs,
        r: &a.r,
        bias: &a.bias,
        variance: &a.variance,
        conversion,
    };
    scalars.clear();
    scalars.extend(
        tile.rows.clone().map(|row| {
            narrow_row_scalars(&row_cols, &a.windows[row], row + 1 - r0, mu, rs[row - r0])
        }),
    );
    let col_cols = NarrowColumns {
        ticks: ct,
        sum: cs,
        r: &b.r[cb..col_end],
        bias: &b.bias[cb..col_end],
        variance: &b.variance[cb..col_end],
        conversion,
    };
    let row_frame = if r0 == 0 { &rt[1..] } else { &rt[..] };
    let pairs = CrossTerms::new(row_frame, r0.saturating_sub(1), &ct[..], cb, mu);
    let start_qt = |slot: usize| {
        join_start_column(n_b, r0, slot).map(|j| match r0.checked_sub(1) {
            Some(before) => pairs.at(before, j - 1),
            None => pairs.before_first_row(j),
        })
    };
    narrow_tile_partials(
        &NarrowTile {
            cols: &col_cols,
            base: cb,
            scalars,
            pairs,
            start_qt: &start_qt,
            m: mu,
            shape,
            chunk_lanes,
            k,
            runs,
            paa: None,
            class: None,
            joined,
        },
        tile,
        qt,
        rho,
        accs,
        &a.windows,
        None,
    );
    false
}

/// Every window of `a` against every window of `b`, no exclusion zone: per `a` window, in order,
/// its exact nearest `b` window (ties to the smaller index; none for a missing `a` window, or when
/// `b` has no window to meet). Each series is quantized at its own scale and, on the narrow path,
/// anchored on its own; either one too wide for an anchor sends both to the exact i64 pass. A
/// window that has a sample missing has no neighbor; one the hot passes cannot rank gets its exact
/// neighbor from the cold route, as in [`ab_join`]. Equal on bits to the exact reference.
#[doc(hidden)]
pub fn ab_join_single_threaded<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
) -> Vec<(u64, Entry)> {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let scales = (scale_a, scale_b);
    let ((entries, _), _, _) = ab_join_with(
        a,
        b,
        m,
        scales,
        NonZeroUsize::MIN,
        default_chunk_lanes(),
        None,
        NonZeroUsize::MIN,
    );
    entries
}

/// [`ab_join_single_threaded`] on up to `threads` threads (only with the `parallel` feature;
/// without it tiles run in order on this thread). Equal on bits to it and to
/// `crate::naive::naive_ab_join`, for any `threads`.
pub fn ab_join<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
) -> Vec<(u64, Entry)> {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    ab_join_report(a, b, m, scale_a, scale_b, threads).0
}

/// [`ab_join`], and how many independent tiles ran and how many of them needed the exact i64
/// pass. One thread, or too few pairs to repay spawning, is the single-anchor pass over the whole
/// matrix (one tile); otherwise independent tiles each take an anchor for their own rows of `a`
/// and another for their own columns of `b`, and only a tile whose own side is still too wide runs
/// the exact pass. The entries are the same bit for bit for any `threads`.
#[doc(hidden)]
pub fn ab_join_report<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, Entry)>, BatchReport) {
    let (entries, report, _) = ab_join_spill_report(a, b, m, scale_a, scale_b, threads);
    (entries, report)
}

/// [`ab_join`] without the cold route: a window the hot passes cannot rank has no neighbor and is
/// nobody's. For the procedures built on a join whose own exact comparison cannot rank such a
/// window either, so they see the windows they always saw.
pub(crate) fn ab_join_hot<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    (scale_a, scale_b): (Scale, Scale),
    threads: NonZeroUsize,
) -> Vec<(u64, Entry)> {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    let threads = threads_for_join(windows(a.len()), windows(b.len()), threads);
    let one = NonZeroUsize::MIN;
    let (qa, qb) = join_sides((a, b), m, (scale_a, scale_b), threads, one);
    let results = Vec::with_capacity(qa.windows.len());
    let ((entries, _), _) = ab_join_on(
        &qa,
        &qb,
        m,
        threads,
        default_chunk_lanes(),
        None,
        one,
        results,
        None,
    );
    entries
}

/// [`ab_join_report`] and what the cold route did: a window of either series that the hot passes
/// cannot rank (a window holding a tick far from its series' level, or whose variance numerator
/// passes `2⁶³`) is ranked, or ranked against, from the exact ticks. A window with no distance
/// without the cold route always gets one; whether the cold route also takes the windows the exact
/// i64 pass ranks is the cost model's decision. Equal on bits for any `threads`.
#[doc(hidden)]
pub fn ab_join_spill_report<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, Entry)>, BatchReport, SpillReport) {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    let threads = threads_for_join(windows(a.len()), windows(b.len()), threads);
    let ((entries, _), report, spill) = ab_join_with(
        a,
        b,
        m,
        (scale_a, scale_b),
        threads,
        default_chunk_lanes(),
        None,
        NonZeroUsize::MIN,
    );
    (entries, report, spill)
}

/// Every window of `a` against every window of `b`, no exclusion zone, with the `k` nearest `b`
/// windows per `a` window (`k` clamped to the number of `b` windows): per `a` window, in order, up
/// to `k` entries nearest first (ties to the smaller index; none for a missing `a` window, or when
/// `b` has no window to meet). Equal on bits to the exact reference, whatever `threads` is; with
/// `k` of one, each entry is [`ab_join`]'s.
///
/// `Err(`[`CapacityError`]`)` when an allocation it needs is too large to make: the neighbors to
/// keep (the finite `a` windows times `min(k, finite `b` windows)`) overflow the address space
/// or the allocator refuses them. That is decided before any work for the windows the hot passes
/// rank, and band by band for the windows only the cold route ranks. No count is refused for its
/// size alone.
pub fn ab_join_topk<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> Result<Vec<(u64, Vec<Entry>)>, CapacityError> {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    let threads = threads_for_join(windows(a.len()), windows(b.len()), threads);
    let ((flat, k), _, _) = ab_join_spilling(
        (a, b),
        m,
        (scale_a, scale_b),
        threads,
        default_chunk_lanes(),
        None,
        k,
    )?;
    Ok(group_by_row(flat, k, windows(a.len()), has_neighbor))
}

/// The join of `a` and `b` as flat lists, with the windows of either series that the hot passes
/// cannot rank ranked cold and merged in; the lists' `k` (at most `b`'s windows), the hot passes'
/// report and what the cold route did.
#[allow(clippy::too_many_arguments)]
fn ab_join_spilling(
    (a, b): (Series<'_>, Series<'_>),
    m: NonZeroUsize,
    (scale_a, scale_b): (Scale, Scale),
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> Result<(FlatJoin, BatchReport, SpillReport), CapacityError> {
    let (mut qa, mut qb) = join_sides((a, b), m, (scale_a, scale_b), threads, k);
    let (n_a, n_b) = (qa.windows.len(), qb.windows.len());
    let wide_cells = |ticks_a: &[i64], ticks_b: &[i64]| {
        let coverable = Anchor::can_cover(ticks_a, m)
            && Anchor::can_cover(ticks_b, m)
            && crate::route::forced_cell() == 0;
        let layout = join_layout(
            (n_a, n_b),
            m,
            threads,
            chunk_lanes,
            plan,
            k,
            (false, coverable),
        );
        wide_cells_join(ticks_a, ticks_b, m, layout)
    };
    let run = |hot_a: &Quantized, hot_b: &Quantized, cold: Option<&Cold>| {
        let hot_k = hot_b.clamp_k(k);
        let results = reserved_vec(join_list_len(
            hot_a.windows.len(),
            hot_a.finite_windows(),
            hot_k,
        ))?;
        let ((flat, hot_k), report) = ab_join_on(
            hot_a,
            hot_b,
            m,
            threads,
            chunk_lanes,
            plan,
            hot_k,
            results,
            None,
        );
        let Some(cold) = cold else {
            return Ok(((flat, hot_k), report, SpillReport::default()));
        };
        let mut spill = cold.report();
        let kept = NonZeroUsize::new(clamp_count(k.get(), n_b)).unwrap_or(NonZeroUsize::MIN);
        let flat = cold.merge(Reach::Join, flat, (m, kept), threads, &mut spill)?;
        Ok(((flat, kept), report, spill))
    };
    with_cold_join(
        JoinSide {
            series: a,
            scale: scale_a,
            quantized: &mut qa,
        },
        JoinSide {
            series: b,
            scale: scale_b,
            quantized: &mut qb,
        },
        m,
        wide_cells,
        run,
    )
}

/// The two series of a join over every diagonal, each built under the gate's join rule for `k`
/// neighbors per window (the two-sided engine's thresholds for one, the top-k's for more): where
/// it turns PAA on, both sides carry their coefficients and the one-loop route can run the join; where it leaves the filter off, the join runs plain.
fn join_sides(
    (a, b): (Series<'_>, Series<'_>),
    m: NonZeroUsize,
    (scale_a, scale_b): (Scale, Scale),
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> (Quantized, Quantized) {
    let engine = Engine::join(k.get());
    (
        Quantized::build_on(a, m, scale_a, threads, engine),
        Quantized::build_on(b, m, scale_b, threads, engine),
    )
}

/// The up to `k` nearest `b` windows of every `a` window over only `runs`' diagonals (a diagonal is
/// a `b` window minus an `a` window), as [`Nearest::push_kept`](crate::nearest::Nearest::push_kept)
/// lays them out: each is the exact top `k` of the pairs on those diagonals.
pub(crate) fn ab_join_over(
    qa: &Quantized,
    qb: &Quantized,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    runs: &SlotRuns,
) -> Result<KeptList, CapacityError> {
    let threads = threads_for_join(qa.windows.len(), qb.windows.len(), threads);
    let k = qb.clamp_k(k);
    let (long, short) = runs.split()?;
    if short.is_empty() || !long.is_empty() {
        let results = reserved_vec(join_list_len(qa.windows.len(), qa.finite_windows(), k))?;
        let ((tiled, _), _) = ab_join_on(
            qa,
            qb,
            m,
            threads,
            default_chunk_lanes(),
            None,
            k,
            results,
            Some(&long),
        );
        if short.is_empty() {
            return Ok(tiled);
        }
        let walked = walk_join(qa, qb, m, threads, k, &short)?;
        return merge_kept((Pairs { rows: qa, cols: qb }, None), m, k, &tiled, &walked);
    }
    walk_join(qa, qb, m, threads, k, &short)
}

/// The up to `k` nearest `b` windows of every `a` window over the diagonals at `slots`
/// (ascending), by the diagonal-major walk.
fn walk_join(
    qa: &Quantized,
    qb: &Quantized,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    slots: &[usize],
) -> Result<KeptList, CapacityError> {
    let last = qb.windows.len() as i64 - 1;
    let mut offsets: Vec<i64> = Vec::new();
    offsets
        .try_reserve_exact(slots.len())
        .map_err(|_| CapacityError)?;
    offsets.extend(slots.iter().map(|&slot| last - slot as i64));
    walk((qa, qb), m, threads, k, &offsets)
}

/// [`ab_join_topk_over_tiled`] by the diagonal-major walk alone, for tests and tuning. It does
/// not depend on `threads`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_topk_over_walked(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    diagonals: &[i64],
) -> Result<Vec<(u64, Vec<Entry>)>, CapacityError> {
    let (qa, qb) = (
        Quantized::build(a, m, scale_a),
        Quantized::build(b, m, scale_b),
    );
    let k = qb.clamp_k(k);
    let last = qb.windows.len() as i64 - 1;
    let mut slots: Vec<usize> = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(last.checked_sub(d)?).ok())
        .collect();
    slots.sort_unstable();
    slots.dedup();
    let flat = walk_join(&qa, &qb, m, threads, k, &slots)?;
    Ok(group_by_row(flat, k, qa.windows.len(), has_neighbor))
}

/// [`ab_join_topk_tiled`] over only the pairs on the diagonals `diagonals` (a diagonal is a `b`
/// window minus an `a` window; any other number is ignored), for tests and tuning: the profile the
/// anytime join holds after visiting them. Neither the tile shape nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_topk_over_tiled(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
    diagonals: &[i64],
) -> TiledTopk {
    let (qa, qb) = (
        Quantized::build(a, m, scale_a),
        Quantized::build(b, m, scale_b),
    );
    let k = qb.clamp_k(k);
    let last = qb.windows.len() as i64 - 1;
    let slots = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(last.checked_sub(d)?).ok())
        .collect();
    let runs = SlotRuns::from_slots(slots);
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let results = reserved_vec(join_list_len(qa.windows.len(), qa.finite_windows(), k))?;
    let ((flat, k), report) = ab_join_on(
        &qa,
        &qb,
        m,
        threads,
        chunk_lanes.get(),
        Some(plan),
        k,
        results,
        Some(&runs),
    );
    Ok((
        group_by_row(flat, k, qa.windows.len(), has_neighbor),
        report,
    ))
}

/// [`ab_join_topk`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` slots (rounded up to a whole number of `chunk_lanes`),
/// always cut into tiles. Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_topk_tiled<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
) -> TiledTopk {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let ((flat, k), report, _) = ab_join_spilling(
        (a, b),
        m,
        (scale_a, scale_b),
        threads,
        chunk_lanes.get(),
        Some(plan),
        k,
    )?;
    let windows = (a.len() + 1).saturating_sub(m.get());
    Ok((group_by_row(flat, k, windows, has_neighbor), report))
}

/// [`ab_join_report`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` slots (rounded up to a whole number of `chunk_lanes`),
/// always cut into tiles. Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn ab_join_report_tiled<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> (Vec<(u64, Entry)>, BatchReport) {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let ((entries, _), report, _) = ab_join_with(
        a,
        b,
        m,
        (scale_a, scale_b),
        threads,
        chunk_lanes.get(),
        Some(plan),
        NonZeroUsize::MIN,
    );
    (entries, report)
}

/// The grouped top-k join and the run's report, or the capacity refusal.
type TiledTopk = Result<(Vec<(u64, Vec<Entry>)>, BatchReport), CapacityError>;

/// The flat entries (as [`Nearest::push_kept`](crate::nearest::Nearest::push_kept) lays them out)
/// and the `k` they were kept at, clamped to `b`'s finite windows.
type FlatJoin = (Vec<(u64, Entry)>, NonZeroUsize);

#[allow(clippy::too_many_arguments)]
fn ab_join_with(
    a: Series<'_>,
    b: Series<'_>,
    m: NonZeroUsize,
    scales: (Scale, Scale),
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> (FlatJoin, BatchReport, SpillReport) {
    ab_join_spilling((a, b), m, scales, threads, chunk_lanes, plan, k)
        .unwrap_or_else(|e| abort_on_capacity(e))
}

/// [`ab_join_with`] on already quantized series.
#[allow(clippy::too_many_arguments)]
pub(super) fn ab_join_on(
    qa: &Quantized,
    qb: &Quantized,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
    mut results: Vec<(u64, Entry)>,
    runs: Option<&SlotRuns>,
) -> (FlatJoin, BatchReport) {
    let (n_a, n_b) = (qa.windows.len(), qb.windows.len());
    let k = qb.clamp_k(k);
    if join_without_pairs(n_a, n_b, &mut results) {
        return ((results, k), BatchReport::of(0, 0));
    }
    let coverable = Anchor::can_cover(&qa.ticks, m)
        && Anchor::can_cover(&qb.ticks, m)
        && crate::route::forced_cell() == 0;
    // What the constant class settles of the join's pairs, over every diagonal only: a run over
    // chosen diagonals need not meet the windows that settle the others.
    let class = if runs.is_none() {
        JoinClass::of(&qa.windows, &qb.windows, k)
    } else {
        None
    };
    // The one-loop route runs where it is taken and gives the run back untouched where it cannot;
    // the same bits as the plain passes below either way.
    if one_loop_join::taken(runs.is_some(), qa.paa.is_some() && qb.paa.is_some()) {
        let plan = TilePlan {
            k,
            survivor: qa.filter,
            ..plan.unwrap_or_else(|| {
                TilePlan::for_join(n_a, n_b, m.get(), threads.get(), chunk_lanes)
            })
        };
        if let Some(report) =
            one_loop_join::profile((qa, qb), m, plan, threads, class.as_ref(), &mut results)
        {
            return ((results, k), report);
        }
    }
    let layout = join_layout(
        (n_a, n_b),
        m,
        threads,
        chunk_lanes,
        plan,
        k,
        // The one-pass engine walks a row at a time and asks nothing of the class: a run with one
        // is cut into tiles, whose blocks of rows ask.
        (runs.is_some(), coverable && class.is_none()),
    );
    let JoinCut::Tiles(plan) = layout else {
        let wide = ab_join_quantized(qa, qb, m, chunk_lanes, k, &mut results);
        return ((results, k), BatchReport::of(1, usize::from(wide)));
    };
    let cut = TileCut::join(n_a, n_b, plan, runs);
    let body = |tile: &Tile, scratch: &mut Scratch, chunk_lanes: usize| {
        join_tile_partials(
            (qa, qb),
            m,
            chunk_lanes,
            tile,
            scratch,
            k,
            (runs, class.as_ref()),
        )
    };
    let report = run_tiles(&qa.windows, m.get(), cut, threads, &mut results, &body);
    ((results, k), report)
}

/// How a join of `n_a` by `n_b` windows is cut: one pass over the whole matrix when nothing asks
/// for tiles (no tile shape, no restricted diagonals, one thread or no threading) and each series
/// fits one anchor, else the tiles of `plan` or of the thread count's own cut, at `k` neighbors
/// per row. `(restricted, coverable)` say whether
/// diagonals are restricted and whether each series fits one anchor.
fn join_layout(
    (n_a, n_b): (usize, usize),
    m: NonZeroUsize,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
    (restricted, coverable): (bool, bool),
) -> JoinCut {
    let in_one_pass = plan.is_none()
        && !restricted
        && coverable
        && (threads.get() == 1 || !cfg!(all(feature = "parallel", not(loom))));
    if in_one_pass {
        return JoinCut::OnePass;
    }
    JoinCut::Tiles(TilePlan {
        k,
        ..plan.unwrap_or_else(|| TilePlan::for_join(n_a, n_b, m.get(), threads.get(), chunk_lanes))
    })
}

/// Fills `results` and returns true when the join has no pair to rank: no rows, or rows with no
/// window of `b` to meet (each [`Entry::NONE`]).
fn join_without_pairs(n_a: usize, n_b: usize, results: &mut Vec<(u64, Entry)>) -> bool {
    if n_b == 0 {
        push_none_rows(n_a, results, || Entry::NONE);
    }
    n_a == 0 || n_b == 0
}

/// The one-pass join engine alone, with `b`'s windows walked `chunk_lanes` at a time.
#[cfg(test)]
pub(super) fn ab_join_chunked<'a>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    chunk_lanes: usize,
) -> Vec<(u64, Entry)> {
    let a: Series<'a> = a.into();
    let b: Series<'a> = b.into();
    let (qa, qb) = (
        Quantized::build(a, m, scale_a),
        Quantized::build(b, m, scale_b),
    );
    let mut results = Vec::with_capacity(qa.windows.len());
    if !join_without_pairs(qa.windows.len(), qb.windows.len(), &mut results) {
        ab_join_quantized(&qa, &qb, m, chunk_lanes, NonZeroUsize::MIN, &mut results);
    }
    results
}

/// Every row of `qa` against every window of `qb`, both non-empty, in one pass over the whole
/// matrix; whether it ran the exact i64 pass.
fn ab_join_quantized(
    qa: &Quantized,
    qb: &Quantized,
    m: NonZeroUsize,
    chunk_lanes: usize,
    k: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
) -> RanWide {
    match (
        Anchor::covering(&qa.ticks, m),
        Anchor::covering(&qb.ticks, m),
    ) {
        (Some(anchor_a), Some(anchor_b)) => {
            ab_rows_narrow(qa, qb, (anchor_a, anchor_b), m, (chunk_lanes, k), results);
            false
        }
        _ => {
            ab_rows_wide(qa, qb, m, chunk_lanes, k, results);
            true
        }
    }
}

/// The diagonal of `a`'s window `row` and `b`'s window `col` lives in slot `a_windows − 1 − row +
/// col`, so a row's neighbors `1..` are consecutive slots and the pair one step back is the same
/// slot. Row 0's slots hold the pair one row before it: its exact start minus the entering term
/// (its leaving tick is the zero in front of the row ticks), so the same pass advances it.
fn ab_rows_narrow(
    qa: &Quantized,
    qb: &Quantized,
    (anchor_a, anchor_b): (Anchor, Anchor),
    m: NonZeroUsize,
    (chunk_lanes, k): (usize, NonZeroUsize),
    results: &mut Vec<(u64, Entry)>,
) {
    let mu = m.get();
    let (n_a, n_b) = (qa.windows.len(), qb.windows.len());
    let (mut a_ticks, mut a_sums, mut b_ticks, mut b_sums) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    narrow::narrow_ticks_into(anchor_a, m, &qa.ticks, &mut a_ticks);
    a_ticks.insert(0, 0);
    narrow::narrow_sums_into(anchor_a, m, &qa.sum, &mut a_sums);
    narrow::narrow_ticks_into(anchor_b, m, &qb.ticks, &mut b_ticks);
    narrow::narrow_sums_into(anchor_b, m, &qb.sum, &mut b_sums);
    let conversion = qa.range.joint(qb.range).conversion();
    let a_cols = NarrowColumns {
        ticks: &a_ticks,
        sum: &a_sums,
        r: &qa.r,
        bias: &qa.bias,
        variance: &qa.variance,
        conversion,
    };
    let b_cols = NarrowColumns {
        ticks: &b_ticks,
        sum: &b_sums,
        r: &qb.r,
        bias: &qb.bias,
        variance: &qb.variance,
        conversion,
    };
    let scalars =
        |row: usize| narrow_row_scalars(&a_cols, &qa.windows[row], row + 1, mu, a_sums[row]);
    let pairs = CrossTerms::new(&a_ticks[1..], 0, &b_ticks[..], 0, mu);

    let mut qt = vec![0i64; n_a + n_b - 1];
    for col in 1..n_b {
        qt[n_a - 1 + col] = pairs.before_first_row(col);
    }
    let mut tile = vec![0.0; chunk_lanes.min(n_b)];
    let mut acc = RowAcc::new();
    for row in 0..n_a {
        acc.begin(&qa.windows[row], k, n_b);
        let scalars = scalars(row);
        let slot = n_a - 1 - row;
        let qt0 = pairs.at(row, 0);
        qt[slot] = qt0;
        let neighbor0 = |_: usize| window_at_narrow(&b_cols, 0);
        acc.start_lane(
            qt0,
            &ChunkCols {
                enter: &[],
                leave: &[],
                sum: &b_sums[..1],
                r: &qb.r[..1],
                bias: &qb.bias[..1],
                first: 0,
                neighbor: &neighbor0,
            },
            scalars,
        );
        for (lo, hi) in (1..n_b)
            .step_by(chunk_lanes)
            .map(|lo| (lo, (lo + chunk_lanes).min(n_b)))
        {
            let neighbor = |lane: usize| window_at_narrow(&b_cols, lo + lane);
            // A forced resume replaces the states the row visits.
            pairs.force_resume(row, (row, lo), &mut qt[slot + lo..slot + hi]);
            pass_chunk(
                &mut qt[slot + lo..slot + hi],
                ChunkCols {
                    enter: &b_ticks[lo + mu - 1..hi + mu - 1],
                    leave: &b_ticks[lo - 1..hi - 1],
                    sum: &b_sums[lo..hi],
                    r: &qb.r[lo..hi],
                    bias: &qb.bias[lo..hi],
                    first: lo,
                    neighbor: &neighbor,
                },
                scalars,
                &mut acc,
                &mut tile,
            );
        }
        finish_rows(
            &qa.windows,
            mu,
            row,
            k,
            core::slice::from_mut(&mut acc),
            results,
        );
    }
}

/// [`ab_rows_narrow`] on the exact i64 ticks, with unscaled `qt` slots.
fn ab_rows_wide(
    qa: &Quantized,
    qb: &Quantized,
    m: NonZeroUsize,
    chunk_lanes: usize,
    k: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
) {
    let mu = m.get();
    let (n_a, n_b) = (qa.windows.len(), qb.windows.len());
    let b_cols = qb.columns();
    let window_of = |j: usize| {
        crate::columns::window_of(
            b_cols.sum[j],
            b_cols.r[j],
            b_cols.bias[j],
            b_cols.variance[j],
        )
    };
    let scalars = |row: usize| WideRow {
        enter: qa.ticks[row + mu - 1],
        leave: row.checked_sub(1).map_or(0, |before| qa.ticks[before]),
        sum: qa.sum[row],
    };
    let pairs = CrossTerms::new(&qa.ticks[..], 0, &qb.ticks[..], 0, mu);

    let mut qt = vec![0i64; n_a + n_b - 1];
    for col in 1..n_b {
        qt[n_a - 1 + col] = pairs.before_first_row(col);
    }
    let mut tile = vec![0.0; chunk_lanes.min(n_b)];
    let mut acc = RowAcc::new();
    for row in 0..n_a {
        acc.begin(&qa.windows[row], k, n_b);
        let scalars = scalars(row);
        let slot = n_a - 1 - row;
        let qt0 = pairs.at(row, 0);
        qt[slot] = qt0;
        let neighbor0 = |_: usize| window_of(0);
        acc.start_lane_wide(
            qt0,
            &WideChunkCols {
                enter: &[],
                leave: &[],
                sum: &b_cols.sum[..1],
                r: &b_cols.r[..1],
                bias: &b_cols.bias[..1],
                first: 0,
                neighbor: &neighbor0,
            },
            scalars,
            mu,
        );
        for (lo, hi) in (1..n_b)
            .step_by(chunk_lanes)
            .map(|lo| (lo, (lo + chunk_lanes).min(n_b)))
        {
            let neighbor = |lane: usize| window_of(lo + lane);
            pairs.force_resume(row, (row, lo), &mut qt[slot + lo..slot + hi]);
            pass_chunk_wide(
                &mut qt[slot + lo..slot + hi],
                WideChunkCols {
                    enter: &qb.ticks[lo + mu - 1..hi + mu - 1],
                    leave: &qb.ticks[lo - 1..hi - 1],
                    sum: &b_cols.sum[lo..hi],
                    r: &b_cols.r[lo..hi],
                    bias: &b_cols.bias[lo..hi],
                    first: lo,
                    neighbor: &neighbor,
                },
                scalars,
                mu,
                &mut acc,
                &mut tile,
            );
        }
        finish_rows(
            &qa.windows,
            mu,
            row,
            k,
            core::slice::from_mut(&mut acc),
            results,
        );
    }
}
