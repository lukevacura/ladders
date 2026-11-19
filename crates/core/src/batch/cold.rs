//! The batch wiring of the cold exact route ([`crate::spill`]): windows the hot passes cannot
//! rank (a window holding an outlier tick, or whose variance numerator passes
//! `2⁶³`) get their exact nearest neighbors, and the hot passes run unchanged beside them.
//!
//! A series that fits one anchor (for a join, both do) has no such window and never reaches the
//! survey, and one that holds no outlier and no window past the exact domain is screened out in a
//! few passes over its ticks. Otherwise a
//! window that has no distance today (every sample present, variance past the exact domain) is
//! always ranked cold: the data alone decides it. The cost model only decides, for the windows
//! the exact i64 pass ranks today, whether the cold route takes them too and lets the outlier
//! ticks be replaced by stand-ins so the hot passes run on one anchor.

use super::plan::{Tile, TileCut, TilePlan};
use super::quantized::Quantized;
use super::tile::{join_tile_is_wide, triangle_tile_is_wide};
use crate::capacity::{CapacityError, clamp_count, reserved_vec};
use crate::entry::Entry;
use crate::narrow::Anchor;
use crate::quantize::{Scale, to_ticks};
use crate::spill::{Counts, HotRuns, Plan, Reach, Survey, nothing_to_rank};
use crate::stats::{Series, WindowKind};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Bands asked for per thread: enough that the bands, which differ in work, balance.
#[cfg(all(feature = "parallel", not(loom)))]
const BANDS_PER_THREAD: usize = 8;

/// A band is at least this many windows per sample of the window length, so each diagonal's exact
/// start (`m` multiplies) is a small part of its walk.
#[cfg(all(feature = "parallel", not(loom)))]
const BAND_MIN_WINDOWS_PER_M: usize = 16;

/// What a run did with the cold route.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillReport {
    /// The cost model's verdict, `None` when the series was not surveyed.
    pub plan: Option<Plan>,
    /// Whether the cost model (or a forced route) had the cold route take every window it could,
    /// with the outlier ticks replaced for the hot passes.
    pub engaged: bool,
    /// Ticks that are outliers by the data-only rule, when the series was surveyed (a join's
    /// two series together).
    pub outlier_ticks: usize,
    /// Windows ranked by the cold route (a join's two series together).
    pub spilled_windows: usize,
    /// What the cold pass walked.
    pub walked: Counts,
}

/// The windows the cold route ranks in one run.
pub(crate) struct Cold {
    survey: Survey<'static>,
    /// The cost model's verdict, `None` when only the windows with no distance were taken.
    plan: Option<Plan>,
    engaged: bool,
    outlier_ticks: usize,
    /// The windows of the rows' series.
    rows: usize,
}

/// The pairs the exact i64 pass ranks in `ticks` at `plan`'s cut: those of the triangle's tiles
/// that do not fit one anchor.
pub(super) fn wide_cells_triangle(ticks: &[i64], m: NonZeroUsize, plan: TilePlan) -> u64 {
    let windows = (ticks.len() + 1).saturating_sub(m.get());
    let exclusion = crate::exclusion::zone(m.get());
    let mut cells = 0u64;
    TileCut::visit_triangle(windows, m.get(), plan, |tile| {
        if triangle_tile_is_wide(ticks, m, tile, plan.chunk_lanes) {
            cells += tile
                .rows
                .clone()
                .map(|row| (row - exclusion).clamp(tile.qa, tile.qb) - tile.qa)
                .sum::<usize>() as u64;
        }
    });
    cells
}

/// How a join cuts its matrix.
#[derive(Clone, Copy)]
pub(super) enum JoinCut {
    /// One pass over the whole matrix.
    OnePass,
    /// Independent tiles.
    Tiles(TilePlan),
}

/// The pairs the exact i64 pass ranks in a join of `a`'s windows with `b`'s cut as `cut`.
pub(super) fn wide_cells_join(a: &[i64], b: &[i64], m: NonZeroUsize, cut: JoinCut) -> u64 {
    let windows = |ticks: &[i64]| (ticks.len() + 1).saturating_sub(m.get());
    let (n_a, n_b) = (windows(a), windows(b));
    if n_a == 0 || n_b == 0 {
        return 0;
    }
    match cut {
        JoinCut::OnePass if Anchor::can_cover(a, m) && Anchor::can_cover(b, m) => 0,
        JoinCut::OnePass => n_a as u64 * n_b as u64,
        JoinCut::Tiles(plan) => {
            let mut cells = 0u64;
            TileCut::visit_join(n_a, n_b, plan, |tile: &Tile| {
                if join_tile_is_wide(a, b, m, tile, plan.chunk_lanes) {
                    // Row `r` meets slots `r..r + n_b`.
                    cells += tile
                        .rows
                        .clone()
                        .map(|row| {
                            let (low, high) = (tile.qa.max(row), tile.qb.min(row + n_b));
                            high.saturating_sub(low)
                        })
                        .sum::<usize>() as u64;
                }
            });
            cells
        }
    }
}

/// Which ticks of `values` were measured. When no window is non-finite, every sample was.
fn measured(
    values: &[f64],
    scale: Scale,
    any_non_finite: bool,
) -> Result<Vec<bool>, CapacityError> {
    let mut had = reserved_vec(values.len() as u128)?;
    if any_non_finite {
        had.extend(values.iter().map(|&x| to_ticks(x, scale).is_some()));
    } else {
        had.resize(values.len(), true);
    }
    Ok(had)
}

/// One series as the survey needs it: which ticks were measured, and whether the screen can rule
/// out any window for the cold route.
struct Screened {
    had: Vec<bool>,
    nothing_to_rank: bool,
}

fn screen(
    series: Series<'_>,
    scale: Scale,
    quantized: &Quantized,
    m: NonZeroUsize,
) -> Result<Screened, CapacityError> {
    let windows = &quantized.windows;
    let non_finite = |row: usize| windows[row].kind == WindowKind::NonFinite;
    let any_non_finite = (0..windows.len()).any(non_finite);
    let had = measured(series.values(), scale, any_non_finite)?;
    let nothing_to_rank =
        nothing_to_rank(&quantized.ticks, &had, m.get(), any_non_finite, non_finite)?;
    Ok(Screened {
        had,
        nothing_to_rank,
    })
}

/// The cost model's verdict for `survey`'s passes of `reaches`, with `wide_cells` pairs on the
/// exact i64 pass today, when the cold route walks the pairs of the windows `rows` and `cols`
/// accept (the ones that have no distance without it) whatever the verdict.
fn price(
    survey: &Survey<'_>,
    reaches: &[Reach],
    wide_cells: u64,
    (rows, cols): (impl Fn(usize) -> bool, impl Fn(usize) -> bool),
) -> Result<Plan, CapacityError> {
    let required = survey.walk_where(reaches, rows, cols)?;
    Ok(super::plan::cold_route(
        &survey.shape_over(reaches, wide_cells, required)?,
    ))
}

/// Runs `body` on the windows the hot passes rank. When the series has windows they cannot rank
/// (and is inside the cold route's exact domain), `body` also gets the [`Cold`] route that ranks
/// them, to merge into what the hot passes return for each of `reaches`. `wide_cells` counts,
/// from the series' ticks, the pairs the exact i64 pass would rank, for the cost model.
pub(super) fn with_cold<T>(
    series: Series<'_>,
    scale: Scale,
    quantized: &mut Quantized,
    m: NonZeroUsize,
    reaches: &[Reach],
    wide_cells: impl FnOnce(&[i64]) -> u64,
    body: impl FnOnce(&Quantized, Option<Anchor>, Option<&Cold>) -> Result<T, CapacityError>,
) -> Result<T, CapacityError> {
    let (whole, cold) = cold_for(series, scale, quantized, m, Some((reaches, wide_cells)))?;
    body(quantized, whole, cold.as_ref())
}

/// The cold route of `series` for the windows only it can rank, decided by the data alone: those
/// with every sample present whose variance numerator passes `2⁶³`, which have no distance
/// without it. Neither the cost model nor a forced route enters, and `quantized` is left as it
/// is, so the hot passes rank every other pair, outlier ticks included. `None` when there is no
/// such window, or the series is outside the cold route's exact domain.
pub(crate) fn no_distance_cold(
    series: Series<'_>,
    scale: Scale,
    quantized: &mut Quantized,
    m: NonZeroUsize,
) -> Result<Option<Cold>, CapacityError> {
    let unpriced = None::<(&[Reach], fn(&[i64]) -> u64)>;
    Ok(cold_for(series, scale, quantized, m, unpriced)?.1)
}

/// The anchor of the whole series for the hot passes (when one covers it) and its cold route,
/// priced over the passes of `reaches` with `wide_cells` as in [`with_cold`], or without a
/// price taking only the windows with no distance. When the cost model engages the cold route,
/// the outlier ticks in `quantized` are replaced by stand-ins.
fn cold_for(
    series: Series<'_>,
    scale: Scale,
    quantized: &mut Quantized,
    m: NonZeroUsize,
    priced: Option<(&[Reach], impl FnOnce(&[i64]) -> u64)>,
) -> Result<(Option<Anchor>, Option<Cold>), CapacityError> {
    // A series inside one anchor has no window with a variance numerator past the exact domain
    // (it is at most 2⁶² there) and no tile on the exact i64 pass.
    let whole = quantized.narrow_anchor(m);
    if whole.is_some() {
        return Ok((whole, None));
    }
    let screened = screen(series, scale, quantized, m)?;
    if screened.nothing_to_rank {
        return Ok((whole, None));
    }
    let Some(survey) = Survey::owned(quantized.ticks.clone(), &screened.had, m)? else {
        return Ok((whole, None));
    };
    let mut survey = survey.flagged(series.constant_windows(), None);
    if !survey.any_spilled() {
        return Ok((whole, None));
    }
    let windows = &quantized.windows;
    let no_distance = |row: usize| windows[row].kind == WindowKind::NonFinite;
    let plan = match priced {
        Some((reaches, wide_cells)) => Some(price(
            &survey,
            reaches,
            wide_cells(&quantized.ticks),
            (no_distance, no_distance),
        )?),
        None => None,
    };
    let outlier_ticks = survey.outliers().iter().filter(|&&flag| flag).count();
    let engaged = plan.is_some_and(|plan| plan.engage);
    let whole = if engaged {
        let had = &screened.had;
        quantized.standing_in(had, survey.outliers(), survey.spilled(), m)?;
        quantized.narrow_anchor(m)
    } else {
        survey.keep_spilled_where(no_distance, no_distance)?;
        whole
    };
    let cold = Cold {
        survey,
        plan,
        engaged,
        outlier_ticks,
        rows: quantized.windows.len(),
    };
    Ok((whole, Some(cold)))
}

/// A join's two series, each quantized, with what the join needs of its source.
pub(crate) struct JoinSide<'a, 'q> {
    pub(crate) series: Series<'a>,
    pub(crate) scale: Scale,
    pub(crate) quantized: &'q mut Quantized,
}

/// [`with_cold`] for a join of the windows of `a` against those of `b`: `body` gets both
/// quantized series as the hot passes should see them. `wide_cells` counts the pairs the exact i64
/// pass would rank from the two series' ticks.
pub(super) fn with_cold_join<T>(
    mut a: JoinSide<'_, '_>,
    mut b: JoinSide<'_, '_>,
    m: NonZeroUsize,
    wide_cells: impl FnOnce(&[i64], &[i64]) -> u64,
    body: impl FnOnce(&Quantized, &Quantized, Option<&Cold>) -> Result<T, CapacityError>,
) -> Result<T, CapacityError> {
    let cold = cold_join_for(&mut a, &mut b, m, Some(wide_cells))?;
    body(a.quantized, b.quantized, cold.as_ref())
}

/// [`no_distance_cold`] for a join of the windows of `a` against those of `b`.
pub(crate) fn no_distance_cold_join(
    mut a: JoinSide<'_, '_>,
    mut b: JoinSide<'_, '_>,
    m: NonZeroUsize,
) -> Result<Option<Cold>, CapacityError> {
    cold_join_for(&mut a, &mut b, m, None::<fn(&[i64], &[i64]) -> u64>)
}

/// [`cold_for`] for a join of the windows of `a` against those of `b`.
fn cold_join_for(
    a: &mut JoinSide<'_, '_>,
    b: &mut JoinSide<'_, '_>,
    m: NonZeroUsize,
    priced: Option<impl FnOnce(&[i64], &[i64]) -> u64>,
) -> Result<Option<Cold>, CapacityError> {
    let (qa, qb) = (&mut *a.quantized, &mut *b.quantized);
    // Both series inside one anchor each: nothing to rank cold and no tile on the exact pass.
    if Anchor::can_cover(&qa.ticks, m) && Anchor::can_cover(&qb.ticks, m) {
        return Ok(None);
    }
    let (screened_a, screened_b) = (
        screen(a.series, a.scale, qa, m)?,
        screen(b.series, b.scale, qb, m)?,
    );
    if screened_a.nothing_to_rank && screened_b.nothing_to_rank {
        return Ok(None);
    }
    let Some(survey) = Survey::join_owned(
        (qa.ticks.clone(), &screened_a.had),
        (qb.ticks.clone(), &screened_b.had),
        m,
    )?
    else {
        return Ok(None);
    };
    let mut survey = survey.flagged(a.series.constant_windows(), b.series.constant_windows());
    if !survey.any_spilled() {
        return Ok(None);
    }
    let (rows, cols) = (&qa.windows, &qb.windows);
    let no_distance = (
        |row: usize| rows[row].kind == WindowKind::NonFinite,
        |col: usize| cols[col].kind == WindowKind::NonFinite,
    );
    let plan = match priced {
        Some(wide_cells) => Some(price(
            &survey,
            &[Reach::Join],
            wide_cells(&qa.ticks, &qb.ticks),
            no_distance,
        )?),
        None => None,
    };
    let outlier_ticks = survey
        .outliers()
        .iter()
        .chain(survey.col_outliers())
        .filter(|&&flag| flag)
        .count();
    let engaged = plan.is_some_and(|plan| plan.engage);
    if engaged {
        qa.standing_in(&screened_a.had, survey.outliers(), survey.spilled(), m)?;
        qb.standing_in(
            &screened_b.had,
            survey.col_outliers(),
            survey.col_spilled(),
            m,
        )?;
    } else {
        survey.keep_spilled_where(no_distance.0, no_distance.1)?;
    }
    Ok(Some(Cold {
        survey,
        plan,
        engaged,
        outlier_ticks,
        rows: qa.windows.len(),
    }))
}

impl Cold {
    /// `hot`, a flat kept list as the hot passes left it for `reach`, with every row's `k`
    /// nearest over the hot and the cold route's neighbors; what was walked is added to `report`.
    pub(super) fn merge(
        &self,
        reach: Reach,
        hot: Vec<(u64, Entry)>,
        (m, k): (NonZeroUsize, NonZeroUsize),
        threads: NonZeroUsize,
        report: &mut SpillReport,
    ) -> Result<Vec<(u64, Entry)>, CapacityError> {
        let affected = self.survey.affected_rows(reach);
        if affected.is_empty() {
            return Ok(hot);
        }
        let runs = HotRuns::new(&hot, self.rows)?;
        let rows = affected.len();
        #[cfg(all(feature = "parallel", not(loom)))]
        let wanted = if threads.get() == 1 {
            rows
        } else {
            rows.div_ceil(threads.get().saturating_mul(BANDS_PER_THREAD))
                .max(BAND_MIN_WINDOWS_PER_M.saturating_mul(m.get()))
        };
        #[cfg(not(all(feature = "parallel", not(loom))))]
        let wanted = {
            let _ = (m, threads);
            rows
        };
        let band_rows = wanted.min(self.survey.band_limit(k.get())).max(1);
        let bands = rows.div_ceil(band_rows);
        let task = |band: usize| {
            let start = affected.start + band * band_rows;
            let mut out = Vec::new();
            let mut counts = Counts::default();
            self.survey.merge_rows(
                reach,
                start..(start + band_rows).min(affected.end),
                k,
                &runs,
                &mut out,
                &mut counts,
            )?;
            Ok::<_, CapacityError>((out, counts))
        };
        #[cfg(all(feature = "parallel", not(loom)))]
        let parts = crate::par::for_each_chunk(threads, bands, task);
        #[cfg(not(all(feature = "parallel", not(loom))))]
        let parts: Vec<_> = (0..bands).map(task).collect();
        let (before, after) = (runs.before(affected.start), runs.tail(affected.end));
        let mut finished = Vec::new();
        finished
            .try_reserve_exact(parts.len())
            .map_err(|_| CapacityError)?;
        let mut total = (before.len() + after.len()) as u128;
        for part in parts {
            let (out, counts) = part?;
            total += out.len() as u128;
            report.walked += counts;
            finished.push(out);
        }
        let mut merged = Vec::new();
        merged
            .try_reserve_exact(usize::try_from(total).map_err(|_| CapacityError)?)
            .map_err(|_| CapacityError)?;
        merged.extend_from_slice(before);
        for out in finished {
            merged.extend(out);
        }
        merged.extend_from_slice(after);
        Ok(merged)
    }

    /// Each row's up to `k` nearest over only the pairs `reach` offers that touch a window the
    /// cold route ranks, as a flat list, and that `k` clamped to the neighbors' windows: an
    /// anytime profile's lists before it visits a hot pair.
    pub(crate) fn ranked_alone(
        &self,
        reach: Reach,
        (m, k): (NonZeroUsize, NonZeroUsize),
        threads: NonZeroUsize,
    ) -> Result<(Vec<(u64, Entry)>, NonZeroUsize), CapacityError> {
        let neighbors = self.survey.col_spilled().len();
        let kept = NonZeroUsize::new(clamp_count(k.get(), neighbors)).unwrap_or(NonZeroUsize::MIN);
        let mut empty = reserved_vec(self.rows as u128)?;
        empty.extend((0..self.rows as u64).map(|row| (row, Entry::NONE)));
        let mut report = SpillReport::default();
        let ranked = self.merge(reach, empty, (m, kept), threads, &mut report)?;
        Ok((ranked, kept))
    }

    /// Whether the cold route ranks any of row `row`'s window and neighbors `x` and `y`.
    pub(super) fn touches(&self, row: usize, x: usize, y: usize) -> bool {
        self.survey.touches(row, x, y)
    }

    /// Whether neighbor `x` is exactly nearer to row `row` than `y` is, ties to the smaller index.
    pub(super) fn nearer_neighbor(&self, row: usize, x: usize, y: usize) -> bool {
        self.survey.nearer_neighbor(row, x, y)
    }

    /// What the cold route did, before anything was walked.
    pub(super) fn report(&self) -> SpillReport {
        let count = |flags: &[bool]| flags.iter().filter(|&&flag| flag).count();
        SpillReport {
            plan: self.plan,
            engaged: self.engaged,
            outlier_ticks: self.outlier_ticks,
            spilled_windows: count(self.survey.spilled())
                + if self.survey.is_join() {
                    count(self.survey.col_spilled())
                } else {
                    0
                },
            walked: Counts::default(),
        }
    }
}
