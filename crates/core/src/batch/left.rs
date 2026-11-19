use super::cold::{Cold, SpillReport, wide_cells_triangle, with_cold};
use super::kept::group_by_row;
use super::one_loop;
use super::pass::{BLOCK_ROWS, Shape, tiled_narrow_profile};
use super::plan::{
    Engine, SlotRuns, Tile, TileCut, TilePlan, default_chunk_lanes, threads_for_size,
};
use super::quantized::Quantized;
use super::schedule::{BatchReport, run_tiles};
use super::tile::{NarrowTile, Scratch, narrow_tile_partials, tile_partials, triangle_start_pair};
use super::two_sided::RightBook;
use crate::bound_paa::PaaSides;
use crate::capacity::{abort_on_capacity, clamp_count};
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::narrow::Anchor;
use crate::quantize::Scale;
use crate::row::{NarrowColumns, narrow_row_scalars};
use crate::spill::Reach;
use crate::stats::{ConstantClass, Series, Window};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// The plain narrow engine over one anchor for the whole series ([`narrow_tile_partials`] with the
/// series' own columns, every pair computed), on up to `threads` threads.
pub(super) fn tiled_narrow_profile_par(
    cols: &NarrowColumns,
    (windows, m): (&[Window], usize),
    plan: TilePlan,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    right: Option<&RightBook>,
    (runs, class): (Option<&SlotRuns>, Option<&ConstantClass>),
) -> BatchReport {
    let (exclusion, k) = (crate::exclusion::zone(m), plan.k);
    let shape = Shape::Triangle { exclusion };
    let pairs = CrossTerms::of(cols.ticks, m);
    let body = |tile: &Tile, scratch: &mut Scratch, chunk_lanes: usize| {
        let Scratch {
            qt,
            tile: rho,
            accs,
            scalars,
            right: part,
            ..
        } = scratch;
        scalars.clear();
        scalars.extend(
            tile.rows
                .clone()
                .map(|row| narrow_row_scalars(cols, &windows[row], row, m, cols.sum[row])),
        );
        narrow_tile_partials(
            &NarrowTile {
                cols,
                base: 0,
                scalars,
                pairs,
                start_qt: &|slot| {
                    triangle_start_pair(exclusion, tile.rows.start, slot)
                        .map(|(i, j)| pairs.at(i, j))
                },
                m,
                shape,
                chunk_lanes,
                k,
                runs,
                paa: None,
                class,
                joined: None,
            },
            tile,
            qt,
            rho,
            accs,
            windows,
            right.map(|book| (book, part)),
        );
        false
    };
    let cut = TileCut::triangle(windows.len(), m, plan, runs);
    run_tiles(windows, m, cut, threads, results, &body)
}

/// The exact left nearest-neighbor matrix profile over `series`: [`left_profile_with_chunk`]
/// at [`default_chunk_lanes`]'s own chunk length, NEON on `aarch64`.
pub fn left_profile<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
) -> Vec<(u64, Entry)> {
    let series: Series<'a> = series.into();
    left_profile_with_chunk(series, m, scale, default_chunk_lanes())
}

/// [`left_profile`], parameterized for tuning and benchmarking: `chunk_lanes` is the narrow
/// path's own lag-chunk width (clamped to at least one lane). Quantized and forward-filled first (the same
/// [`crate::quantize`]/[`crate::sanitize`] pieces every engine uses), then every row advanced
/// and ranked through the narrow path's own tiled engine, in order, or, when the series is too
/// wide for one anchor, through independent tiles that each take their own. One entry per
/// window start, in order; equal to `crate::naive::naive_left_profile`'s row on the same
/// quantized samples, bit for bit, whatever `chunk_lanes` is.
#[doc(hidden)]
pub fn left_profile_with_chunk<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    chunk_lanes: usize,
) -> Vec<(u64, Entry)> {
    let series: Series<'a> = series.into();
    left_profile_with_blocks(series, m, scale, chunk_lanes, BLOCK_ROWS)
}

/// [`left_profile_with_chunk`] with the row block size too (clamped to at least one row): the
/// block size, like the chunk width, changes only locality, never a bit.
pub(super) fn left_profile_with_blocks<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    chunk_lanes: usize,
    block_rows: usize,
) -> Vec<(u64, Entry)> {
    let series: Series<'a> = series.into();
    profile_with(
        series,
        m,
        scale,
        NonZeroUsize::MIN,
        chunk_lanes.max(1),
        block_rows.max(1),
        None,
    )
    .0
}

/// [`left_profile`] on up to `threads` threads (feature `parallel`), cut into tiles by
/// `tile_plan`. Equal to [`left_profile`] on bits, for any `threads`.
#[cfg(all(feature = "parallel", not(loom)))]
pub fn left_profile_threads<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Vec<(u64, Entry)> {
    let series: Series<'a> = series.into();
    left_profile_report(series, m, scale, threads).0
}

/// [`left_profile`] on up to `threads` threads (only with the `parallel` feature; without it
/// tiles run in order on this thread), and how many independent tiles ran and how many of them
/// needed the exact i64 pass.
///
/// When the whole series fits one narrow anchor, the single-anchor engines run (one thread, or
/// a series too short for the threads asked, is [`left_profile`]'s own engine)
/// and no tile is wide. Otherwise independent tiles each take an anchor for their own rows and
/// another for their own columns, and only a tile whose own side is still too wide runs the
/// exact pass. The entries are the same bit for bit for any `threads`.
#[doc(hidden)]
pub fn left_profile_report<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, Entry)>, BatchReport) {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    profile_with(
        series,
        m,
        scale,
        threads,
        default_chunk_lanes(),
        BLOCK_ROWS,
        None,
    )
}

/// The exact left profile with the `k` nearest earlier neighbors per window: per window start, in
/// order, up to `k` entries nearest first (ties to the smaller index; none for a window with no
/// eligible neighbor), from the same tiled pass as [`left_profile_report`]. Equal on bits to the
/// exact oracle's top-k on the same quantized samples, whatever `threads` is; with `k` of one,
/// each entry is [`left_profile`]'s.
pub fn left_topk_profile<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> Vec<(u64, Vec<Entry>)> {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    topk_with(series, m, scale, threads, default_chunk_lanes(), None, k).0
}

/// [`left_topk_profile`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes (rounded up to a whole number of `chunk_lanes`).
/// Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn left_topk_profile_tiled<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
) -> (Vec<(u64, Vec<Entry>)>, BatchReport) {
    let series: Series<'a> = series.into();
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    topk_with(series, m, scale, threads, chunk_lanes.get(), Some(plan), k)
}

fn topk_with(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> (Vec<(u64, Vec<Entry>)>, BatchReport) {
    let engine = Engine::self_join(false, k.get());
    let quantized = Quantized::build_on(series, m, scale, threads, engine);
    let windows = quantized.windows.len();
    let k = NonZeroUsize::new(clamp_count(k.get(), windows)).unwrap_or(NonZeroUsize::MIN);
    let (flat, report, _) = profile_cold(
        series,
        scale,
        quantized,
        (m, k),
        threads,
        (chunk_lanes, BLOCK_ROWS),
        plan,
    );
    (group_by_row(flat, k, windows, has_neighbor), report)
}

/// Whether `entry` holds a neighbor.
pub(crate) fn has_neighbor(entry: &Entry) -> bool {
    entry.index().get().is_some()
}

/// [`left_profile_threads`] with the block height (`band_rows`) and lag-chunk width
/// (`chunk_lanes`, as in [`left_profile_with_chunk`]) as parameters, for tests and tuning, and
/// tiles one chunk wide. Neither they nor `threads` change a bit.
#[cfg(all(feature = "parallel", not(loom)))]
#[doc(hidden)]
pub fn left_profile_threads_with<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
) -> Vec<(u64, Entry)> {
    let series: Series<'a> = series.into();
    left_profile_threads_tiled(
        series,
        m,
        scale,
        threads,
        band_rows,
        chunk_lanes,
        chunk_lanes,
    )
}

/// [`left_profile_threads_with`] with the tile width too: `tile_lanes` is rounded up to a whole
/// number of `chunk_lanes` (a tile is never narrower than the L1 chunk it walks), and changes
/// only scheduling, never a bit.
#[cfg(all(feature = "parallel", not(loom)))]
#[doc(hidden)]
pub fn left_profile_threads_tiled<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> Vec<(u64, Entry)> {
    let series: Series<'a> = series.into();
    left_profile_report_tiled(
        series,
        m,
        scale,
        threads,
        band_rows,
        chunk_lanes,
        tile_lanes,
    )
    .0
}

/// [`left_profile_report`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes (rounded up to a whole number of `chunk_lanes`).
/// Neither they nor `threads` change a bit.
#[doc(hidden)]
pub fn left_profile_report_tiled<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> (Vec<(u64, Entry)>, BatchReport) {
    let series: Series<'a> = series.into();
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    profile_with(
        series,
        m,
        scale,
        threads,
        chunk_lanes.get(),
        BLOCK_ROWS,
        Some(plan),
    )
}

/// Quantizes `series`, then runs the single-anchor engines when the whole series fits one anchor
/// (the single-threaded one with `threads` of one or the `parallel` feature off, else the
/// threaded one) and the independent per-tile-anchor tiles otherwise. `plan` fixes the tiles of
/// either threaded or per-tile run; `None` takes `tile_plan`'s.
fn profile_with(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    block_rows: usize,
    plan: Option<TilePlan>,
) -> (Vec<(u64, Entry)>, BatchReport) {
    let quantized = Quantized::build_on(series, m, scale, threads, Engine::Left);
    let (profile, report, _) = profile_cold(
        series,
        scale,
        quantized,
        (m, NonZeroUsize::MIN),
        threads,
        (chunk_lanes, block_rows),
        plan,
    );
    (profile, report)
}

/// [`left_profile_report_tiled`] and what the cold route did: windows the hot passes cannot rank
/// (a window holding an outlier tick, or whose variance numerator passes
/// `2⁶³`) are ranked from the exact ticks and merged in. A window that has no distance without
/// the cold route always gets one; whether the cold route also takes the windows the exact i64
/// pass ranks is the cost model's decision. Equal on bits for any `threads`.
#[doc(hidden)]
pub fn left_profile_spill_report<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, Entry)>, BatchReport, SpillReport) {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    let quantized = Quantized::build_on(series, m, scale, threads, Engine::Left);
    profile_cold(
        series,
        scale,
        quantized,
        (m, NonZeroUsize::MIN),
        threads,
        (default_chunk_lanes(), BLOCK_ROWS),
        None,
    )
}

/// [`profile_on`] over `quantized`, with the windows the hot passes cannot rank ranked cold and
/// merged into the lists the hot passes return.
fn profile_cold(
    series: Series<'_>,
    scale: Scale,
    mut quantized: Quantized,
    (m, k): (NonZeroUsize, NonZeroUsize),
    threads: NonZeroUsize,
    (chunk_lanes, block_rows): (usize, usize),
    plan: Option<TilePlan>,
) -> (Vec<(u64, Entry)>, BatchReport, SpillReport) {
    let windows = quantized.windows.len();
    let wide_cells = |ticks: &[i64]| {
        let cut = resolved_plan(windows, m, k, threads, chunk_lanes, plan);
        wide_cells_triangle(ticks, m, cut)
    };
    let run = |hot: &Quantized, whole: Option<Anchor>, cold: Option<&Cold>| {
        let (results, report) = profile_on(
            hot,
            whole,
            m,
            k,
            threads,
            chunk_lanes,
            block_rows,
            plan,
            Vec::with_capacity(windows),
            None,
            None,
        );
        match cold {
            None => Ok((results, report, SpillReport::default())),
            Some(cold) => {
                let mut spill = cold.report();
                let merged = cold.merge(Reach::Left, results, (m, k), threads, &mut spill)?;
                Ok((merged, report, spill))
            }
        }
    };
    with_cold(
        series,
        scale,
        &mut quantized,
        m,
        &[Reach::Left],
        wide_cells,
        run,
    )
    .unwrap_or_else(|e| abort_on_capacity(e))
}

/// The tiles `plan` asks for, or the thread count's own cut, for lists of `k` neighbors.
pub(super) fn resolved_plan(
    n_windows: usize,
    m: NonZeroUsize,
    k: NonZeroUsize,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
) -> TilePlan {
    TilePlan {
        k,
        ..plan.unwrap_or_else(|| {
            TilePlan::for_threads(n_windows, m.get(), threads.get(), chunk_lanes)
        })
    }
}

/// [`profile_with`] on already quantized `series`, whose `whole` is the one anchor that covers all
/// its ticks, if there is one (`quantized.narrow_anchor(m)`, decided once per run). With `right`, every tile also offers each pair
/// to the earlier window's right-neighbor accumulator, and the single-threaded engine is the
/// tile engine on one thread. With `runs`, only those diagonals are walked, always by tiles.
#[allow(clippy::too_many_arguments)]
pub(super) fn profile_on(
    quantized: &Quantized,
    whole: Option<Anchor>,
    m: NonZeroUsize,
    k: NonZeroUsize,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    block_rows: usize,
    plan: Option<TilePlan>,
    mut results: Vec<(u64, Entry)>,
    right: Option<&RightBook>,
    runs: Option<&SlotRuns>,
) -> (Vec<(u64, Entry)>, BatchReport) {
    let exclusion = crate::exclusion::zone(m.get());
    let n_windows = quantized.windows.len();
    let plan = TilePlan {
        survivor: quantized.filter,
        ..resolved_plan(n_windows, m, k, threads, chunk_lanes, plan)
    };
    let paa = quantized.paa.as_deref().map(PaaSides::of);
    let global = quantized.global_floor(right.is_some() && runs.is_none() && k.get() == 1);

    let report = match whole {
        Some(anchor) => {
            let (ticks_off, sum_off) = quantized.narrow_columns(anchor, m);
            let cols = NarrowColumns {
                ticks: &ticks_off,
                sum: &sum_off,
                r: &quantized.r,
                bias: &quantized.bias,
                variance: &quantized.variance,
                conversion: quantized.range.conversion(),
            };
            let tiled = right.is_some()
                || runs.is_some()
                || one_loop::taken(runs.is_some(), paa.is_some(), global.is_some());
            #[cfg(all(feature = "parallel", not(loom)))]
            let tiled = tiled || threads.get() > 1;
            if tiled {
                // The one-loop entry runs the route where it is taken and falls back to today's
                // path where it is not.
                let report = one_loop::profile(
                    &cols,
                    &quantized.windows,
                    m.get(),
                    plan,
                    threads,
                    &mut results,
                    right,
                    (runs, paa, quantized.constant_class(runs)),
                    global,
                );
                return (results, report);
            }
            tiled_narrow_profile(
                &cols,
                &quantized.windows,
                m.get(),
                exclusion,
                chunk_lanes,
                block_rows,
                k,
                quantized.constant_class(runs),
                &mut results,
            );
            BatchReport::of(1, 0)
        }
        None => {
            // No one anchor covers the series: where the one-loop route is forced on, each band
            // takes anchors of its own.
            if runs.is_none()
                && let Some(report) = one_loop::profile_per_band(
                    quantized,
                    m,
                    plan,
                    threads,
                    &mut results,
                    right,
                    global,
                )
            {
                return (results, report);
            }
            let body = |tile: &Tile, scratch: &mut Scratch, chunk_lanes: usize| {
                tile_partials(quantized, m, chunk_lanes, tile, scratch, k, right, runs)
            };
            let cut = TileCut::triangle(n_windows, m.get(), plan, runs);
            run_tiles(
                &quantized.windows,
                m.get(),
                cut,
                threads,
                &mut results,
                &body,
            )
        }
    };
    (results, report)
}
