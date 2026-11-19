use super::cold::{Cold, SpillReport, wide_cells_triangle, with_cold};
use super::kept::{finite_from, side_counts};
use super::left::{profile_on, resolved_plan};
use super::pass::BLOCK_ROWS;
use super::plan::{Engine, SlotRuns, TilePlan, default_chunk_lanes, threads_for_size};
use super::quantized::Quantized;
use super::schedule::{BatchReport, ColumnGroups};
use super::sparse::walk;
use crate::capacity::{CapacityError, abort_on_capacity, clamp_count, reserved_vec};
use crate::cross_term;
use crate::entry::{Entry, TwoSided, TwoSidedTopk};
use crate::estimate::Producer;
use crate::index::SampleIndex;
use crate::narrow::Anchor;
use crate::nearest::{ExactDistance, Nearest, clearly_nearer};
use crate::quantize::Scale;
use crate::row::{ColumnAcc, RankRow, WideRank, centered_cross_fast};
use crate::spill::Reach;
use crate::stats::{ConstantClass, Series, Window, WindowKind};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Whether `window` can be a neighbor.
pub(super) fn is_finite(window: &Window) -> bool {
    !matches!(window.kind, WindowKind::NonFinite)
}

/// Every window's right-neighbor accumulator: one [`ColumnAcc`] per [`RIGHT_GROUP`](super::schedule::RIGHT_GROUP) columns, each
/// tile merging its own partial in as it finishes. One entry per window however many tiles or
/// threads, and the merge is order-independent.
pub(super) struct RightBook {
    pub(super) groups: ColumnGroups<ColumnAcc>,
    /// The list [`Self::finish`] fills, reserved for exactly its entries.
    out: Vec<(u64, Entry)>,
    /// Test-only: the most candidates any accumulator held at once.
    #[cfg(test)]
    pub(super) peak_cands: crate::sync::AtomicUsize,
}

impl RightBook {
    /// A column's right neighbors are the finite windows past its exclusion zone, which is all
    /// that can ever offer to it. With the constant `class`, each column's floor starts from the
    /// bound the class guarantees its nearest, under the bound of both row passes' estimates.
    pub(super) fn new(
        windows: &[Window],
        k: NonZeroUsize,
        exclusion: usize,
        out: Vec<(u64, Entry)>,
        class: Option<&ConstantClass>,
    ) -> Result<Self, CapacityError> {
        let finite_from = finite_from(windows.len(), |i| is_finite(&windows[i]));
        let reach = |column: usize| finite_from[(column + exclusion + 1).min(windows.len())];
        let delta = RankRow::DELTA.max(WideRank::DELTA);
        let groups = ColumnGroups::try_new(windows.len(), |columns| {
            let mut group = ColumnAcc::new();
            group.try_begin(columns, windows, k, reach)?;
            if let Some(class) = class {
                group.seed_floors(class, delta);
            }
            Ok(group)
        })?;
        Ok(Self {
            groups,
            out,
            #[cfg(test)]
            peak_cands: crate::sync::AtomicUsize::new(0),
        })
    }

    /// Starts `part` (just begun) from the floors its columns have reached so far.
    pub(super) fn seed(&self, part: &mut ColumnAcc) {
        self.groups.merge(part.columns(), |group, shared| {
            part.raise_floors(group, shared);
        });
    }

    pub(super) fn merge(&self, part: &mut ColumnAcc, windows: &[Window]) {
        #[cfg(test)]
        self.peak_cands
            .fetch_max(part.peak_cands, crate::sync::Ordering::SeqCst);
        part.sort();
        self.groups.merge(part.columns(), |group, shared| {
            group.merge(part, shared, windows);
            #[cfg(test)]
            self.peak_cands
                .fetch_max(group.peak_cands, crate::sync::Ordering::SeqCst);
        });
    }

    /// Each window's up to `k` nearest later neighbors, in order, as [`Nearest::push_kept`] lays
    /// them out.
    pub(super) fn finish(
        self,
        windows: &[Window],
        m: usize,
        threads: NonZeroUsize,
    ) -> Vec<(u64, Entry)> {
        let mut entries = self.out;
        self.groups.flatten_on(threads, &mut entries, |group, out| {
            group.finish(windows, m, out)
        });
        entries
    }
}

/// The exact two-sided nearest-neighbor matrix profile over `series`, on up to `threads`
/// threads: per window start, in order, the left neighbor (exactly [`left_profile`](super::left::left_profile)'s entry), the
/// right neighbor, and the nearer of the two, ties to the smaller index. Looking right trades on
/// future information: the name says so. One tiled pass offers each pair to both windows, so the
/// right side costs one accumulator per window, not a second sweep. Equal on bits to the exact
/// oracle's two-sided profile, whatever `threads` is.
pub fn two_sided_profile<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Vec<(u64, TwoSided)> {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    two_sided_with(series, m, scale, threads, default_chunk_lanes(), None).0
}

/// [`two_sided_profile`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes (rounded up to a whole number of `chunk_lanes`).
/// Neither they nor `threads` change a bit.
#[doc(hidden)]
pub fn two_sided_profile_tiled<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> (Vec<(u64, TwoSided)>, BatchReport) {
    let series: Series<'a> = series.into();
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let (profile, report, _) =
        two_sided_with(series, m, scale, threads, chunk_lanes.get(), Some(plan));
    (profile, report)
}

/// [`two_sided_profile`] and what the cold route did: a window the hot passes cannot rank (a
/// window holding an outlier tick, or whose variance numerator passes `2⁶³`)
/// gets its exact left, right and nearest neighbors from the exact ticks. A window with no
/// distance without the cold route always gets one; whether the cold route also takes the windows
/// the exact i64 pass ranks is the cost model's decision. Equal on bits for any `threads`.
#[doc(hidden)]
pub fn two_sided_spill_report<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> (Vec<(u64, TwoSided)>, BatchReport, SpillReport) {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    two_sided_with(series, m, scale, threads, default_chunk_lanes(), None)
}

fn two_sided_with(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
) -> (Vec<(u64, TwoSided)>, BatchReport, SpillReport) {
    two_sided_of(
        Engine::TwoSided,
        series,
        m,
        scale,
        (threads, chunk_lanes),
        plan,
    )
    .unwrap_or_else(|e| abort_on_capacity(e))
}

/// The two-sided profile of a run that reports only its best pair, on up to `threads` threads:
/// the run keeps a global floor, so the entry of a window is exact only when its best pair is at
/// least as near as the series' best pair; the series' best pair, and every window whose nearest
/// ties it, are the plain profile's on bits. A capacity error when the run cannot allocate.
pub(crate) fn best_pair_profile(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Result<Vec<(u64, TwoSided)>, CapacityError> {
    let threads = threads_for_size(series.len(), m, threads);
    let (profile, _, _) = two_sided_of(
        Engine::BestPair,
        series,
        m,
        scale,
        (threads, default_chunk_lanes()),
        None,
    )?;
    Ok(profile)
}

/// A two-sided profile, the run's report and what the cold route did.
type Profiled = (Vec<(u64, TwoSided)>, BatchReport, SpillReport);

fn two_sided_of(
    engine: Engine,
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    (threads, chunk_lanes): (NonZeroUsize, usize),
    plan: Option<TilePlan>,
) -> Result<Profiled, CapacityError> {
    let quantized = Quantized::build_on(series, m, scale, threads, engine);
    two_sided_spilling(
        series,
        scale,
        quantized,
        m,
        (threads, chunk_lanes),
        plan,
        NonZeroUsize::MIN,
        |hot, left, right, report, spill, cold| {
            let profile = left
                .into_iter()
                .zip(right)
                .map(|((row, left), (_, right))| {
                    let nearest = nearest_of(hot, cold, m.get(), row as usize, left, right);
                    (
                        row,
                        TwoSided {
                            left,
                            right,
                            nearest,
                        },
                    )
                })
                .collect();
            Ok((profile, report, spill))
        },
    )
}

/// A flat list of kept neighbors, as [`Nearest::push_kept`] lays them out: per window, in order, a
/// run of entries nearest first, or one empty entry.
pub(crate) type KeptList = Vec<(u64, Entry)>;

/// One window's run of a [`KeptList`].
type Run<'a> = &'a [(u64, Entry)];

/// The quantized series, the left and the right side's kept neighbors per window as
/// [`Nearest::push_kept`] lays them out, and the run's report.
pub(super) type Sides = (Quantized, Vec<(u64, Entry)>, Vec<(u64, Entry)>, BatchReport);

/// Both sides' up to `k` nearest per window (`k` clamped to the number of windows).
pub(super) fn two_sided_sides(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> Sides {
    let engine = Engine::self_join(true, k.get());
    let quantized = Quantized::build_on(series, m, scale, threads, engine);
    two_sided_sides_of(quantized, m, threads, chunk_lanes, plan, k)
        .unwrap_or_else(|e| abort_on_capacity(e))
}

/// [`two_sided_sides`] on already quantized `series`.
pub(super) fn two_sided_sides_of(
    quantized: Quantized,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> Result<Sides, CapacityError> {
    let whole = quantized.narrow_anchor(m);
    let (left, right, report) =
        two_sided_sides_on(&quantized, whole, m, threads, chunk_lanes, plan, k, None)?;
    Ok((quantized, left, right, report))
}

/// The two sides' hot lists with the windows the hot passes cannot rank ranked cold and merged in,
/// handed to `finish` with the hot windows, the run's reports and the cold route: both sides'
/// up to `k` nearest per window.
#[allow(clippy::too_many_arguments)]
fn two_sided_spilling<T>(
    series: Series<'_>,
    scale: Scale,
    mut quantized: Quantized,
    m: NonZeroUsize,
    (threads, chunk_lanes): (NonZeroUsize, usize),
    plan: Option<TilePlan>,
    k: NonZeroUsize,
    finish: impl FnOnce(
        &Quantized,
        KeptList,
        KeptList,
        BatchReport,
        SpillReport,
        Option<&Cold>,
    ) -> Result<T, CapacityError>,
) -> Result<T, CapacityError> {
    let windows = quantized.windows.len();
    let wide_cells = |ticks: &[i64]| {
        let cut = resolved_plan(windows, m, k, threads, chunk_lanes, plan);
        wide_cells_triangle(ticks, m, cut)
    };
    let run = |hot: &Quantized, whole: Option<Anchor>, cold: Option<&Cold>| {
        let (left, right, report) =
            two_sided_sides_on(hot, whole, m, threads, chunk_lanes, plan, k, None)?;
        let Some(cold) = cold else {
            return finish(hot, left, right, report, SpillReport::default(), None);
        };
        let mut spill = cold.report();
        let kept = NonZeroUsize::new(clamp_count(k.get(), windows)).unwrap_or(NonZeroUsize::MIN);
        let left = cold.merge(Reach::Left, left, (m, kept), threads, &mut spill)?;
        let right = cold.merge(Reach::Right, right, (m, kept), threads, &mut spill)?;
        finish(hot, left, right, report, spill, Some(cold))
    };
    with_cold(
        series,
        scale,
        &mut quantized,
        m,
        &[Reach::Left, Reach::Right],
        wide_cells,
        run,
    )
}

/// The left and right sides of [`two_sided_sides_of`] on `quantized` in place, and the run's
/// report; with `runs`, only the pairs on those diagonals are offered.
#[allow(clippy::too_many_arguments)]
pub(super) fn two_sided_sides_on(
    quantized: &Quantized,
    whole: Option<Anchor>,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
    runs: Option<&SlotRuns>,
) -> Result<(KeptList, KeptList, BatchReport), CapacityError> {
    let k = quantized.clamp_k(k);
    let exclusion = crate::exclusion::zone(m.get());
    let (left_count, right_count) = side_counts(
        quantized.windows.len(),
        |i| is_finite(&quantized.windows[i]),
        k,
        exclusion,
    );
    let left_out = reserved_vec(left_count.list_len())?;
    let right_out = reserved_vec(right_count.list_len())?;
    let class = quantized.constant_class(runs);
    let book = RightBook::new(&quantized.windows, k, exclusion, right_out, class)?;
    let (left, report) = profile_on(
        quantized,
        whole,
        m,
        k,
        threads,
        chunk_lanes,
        BLOCK_ROWS,
        plan,
        left_out,
        Some(&book),
        runs,
    );
    let right = book.finish(&quantized.windows, m.get(), threads);
    Ok((left, right, report))
}

/// Both sides' up to `k` nearest per window over only the pairs on `runs`' diagonals, as
/// [`Nearest::push_kept`] lays them out: each is the exact top `k` of those pairs.
pub(crate) fn two_sided_sides_over(
    quantized: &Quantized,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    runs: &SlotRuns,
) -> Result<(KeptList, KeptList), CapacityError> {
    let threads = threads_for_size(quantized.ticks.len(), m, threads);
    let (long, short) = runs.split()?;
    let tiled = if short.is_empty() || !long.is_empty() {
        let (left, right, _) = two_sided_sides_on(
            quantized,
            quantized.narrow_anchor(m),
            m,
            threads,
            default_chunk_lanes(),
            None,
            k,
            Some(&long),
        )?;
        Some((left, right))
    } else {
        None
    };
    if short.is_empty() {
        return Ok(tiled.expect("a run with no walked slot is tiled"));
    }
    let (left, right) = walk_sides(quantized, m, threads, k, &short)?;
    let Some((tiled_left, tiled_right)) = tiled else {
        return Ok((left, right));
    };
    let pairs = Pairs {
        rows: quantized,
        cols: quantized,
    };
    Ok((
        merge_kept((pairs, None), m, k, &tiled_left, &left)?,
        merge_kept((pairs, None), m, k, &tiled_right, &right)?,
    ))
}

/// Both sides' up to `k` nearest per window over the self-join diagonals at `slots` (ascending),
/// by the diagonal-major walk.
fn walk_sides(
    quantized: &Quantized,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    slots: &[usize],
) -> Result<(KeptList, KeptList), CapacityError> {
    let k = quantized.clamp_k(k);
    let first = crate::exclusion::zone(m.get()) as i64 + 1;
    let mut lags: Vec<i64> = Vec::new();
    lags.try_reserve_exact(slots.len())
        .map_err(|_| CapacityError)?;
    lags.extend(slots.iter().map(|&slot| first + slot as i64));
    let pair = (quantized, quantized);
    let left_offsets: Vec<i64> = lags.iter().map(|&lag| -lag).collect();
    let left = walk(pair, m, threads, k, &left_offsets)?;
    let right = walk(pair, m, threads, k, &lags)?;
    Ok((left, right))
}

/// [`two_sided_topk_profile_over_tiled`] by the diagonal-major walk alone, for tests and tuning:
/// the profile after visiting the diagonals `diagonals` (any number that is no diagonal is
/// ignored). It does not depend on `threads`.
#[doc(hidden)]
pub fn two_sided_topk_profile_over_walked(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    diagonals: &[i64],
) -> Result<Vec<(u64, TwoSidedTopk)>, CapacityError> {
    let quantized = Quantized::build_on(series, m, scale, threads, Engine::Ungated);
    let k = quantized.clamp_k(k);
    let past = crate::exclusion::zone(m.get()) + 1;
    let mut slots: Vec<usize> = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(d).ok()?.checked_sub(past))
        .collect();
    slots.sort_unstable();
    slots.dedup();
    let (left, right) = walk_sides(&quantized, m, threads, k, &slots)?;
    merged_two_sided(&quantized, None, m, k, &left, &right)
}

/// The nearer of `row`'s left and right entries under the exact comparator, as
/// [`nearest_of`] picks it, with the exact correlation it ranks by.
pub(super) fn nearer_side(
    q: &Quantized,
    m: usize,
    row: usize,
    left: Entry,
    right: Entry,
) -> Option<ExactDistance> {
    let winner = nearest_of(q, None, m, row, left, right);
    let mut nearest = Nearest::new(&q.windows[row], m);
    offer_entry(q, m, row, &mut nearest, winner);
    nearest.finish_exact()
}

/// `row`'s nearer entry of `left` and `right` by [`entry_nearer`], ties to the left one, the
/// smaller index; an empty side yields the other.
fn nearest_of(
    q: &Quantized,
    cold: Option<&Cold>,
    m: usize,
    row: usize,
    left: Entry,
    right: Entry,
) -> Entry {
    let pairs = Pairs { rows: q, cols: q };
    match (left.index().get(), right.index().get()) {
        (Some(_), Some(_)) if entry_nearer(pairs, cold, m, row, right, left) => right,
        (Some(_), _) => left,
        (None, Some(_)) => right,
        (None, None) => Entry::NONE,
    }
}

/// Offers `entry`'s neighbor to `nearest`, a reduction of `row`'s window, with the pair's exact
/// centered cross term; an entry with no neighbor offers nothing.
fn offer_entry(q: &Quantized, m: usize, row: usize, nearest: &mut Nearest, entry: Entry) {
    Pairs { rows: q, cols: q }.offer(m, row, nearest, entry);
}

/// The series whose windows are the rows and the series whose windows are their neighbors (the
/// same one for a self-join): what an entry's neighbor index is resolved against.
#[derive(Clone, Copy)]
pub(crate) struct Pairs<'a> {
    pub(crate) rows: &'a Quantized,
    pub(crate) cols: &'a Quantized,
}

impl Pairs<'_> {
    /// [`offer_entry`] across the two series.
    fn offer(self, m: usize, row: usize, nearest: &mut Nearest, entry: Entry) {
        if let Some((n, neighbor, index)) = self.pair(m, row, entry) {
            nearest.offer(n, &neighbor, index);
        }
    }

    /// The exact pair `entry` names for `row`, as [`Nearest::offer`] takes it; `None` for an empty
    /// entry.
    fn pair(self, m: usize, row: usize, entry: Entry) -> Option<(i64, Window, SampleIndex)> {
        let neighbor = entry.index().get()? as usize;
        let qt = cross_term::wide(
            &self.rows.ticks[row..][..m],
            &self.cols.ticks[neighbor..][..m],
        );
        let (a, b) = (&self.rows.windows[row], &self.cols.windows[neighbor]);
        let n = centered_cross_fast(qt, m as i64, a.sum, b.sum);
        let index = SampleIndex::new(neighbor as u64).expect("a winner's index is not NONE");
        Some((n, *b, index))
    }
}

/// [`two_sided_topk_profile_tiled`] over only the self-join pairs on the diagonals `diagonals` (a
/// diagonal is `j − i` of a pair, past the exclusion zone; any other number is ignored), for tests
/// and tuning: the profile the anytime profiles hold after visiting them. Neither the tile shape
/// nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn two_sided_topk_profile_over_tiled(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
    diagonals: &[i64],
) -> Result<(Vec<(u64, TwoSidedTopk)>, BatchReport), CapacityError> {
    let quantized = Quantized::build_on(series, m, scale, threads, Engine::Ungated);
    let k = quantized.clamp_k(k);
    let past = crate::exclusion::zone(m.get()) + 1;
    let slots = diagonals
        .iter()
        .filter_map(|&d| usize::try_from(d).ok()?.checked_sub(past))
        .collect();
    let runs = SlotRuns::from_slots(slots);
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let (left, right, report) = two_sided_sides_on(
        &quantized,
        quantized.narrow_anchor(m),
        m,
        threads,
        chunk_lanes.get(),
        Some(plan),
        k,
        Some(&runs),
    )?;
    Ok((
        merged_two_sided(&quantized, None, m, k, &left, &right)?,
        report,
    ))
}

/// Whether `x` is exactly nearer to `row` than `y` is, ties to the smaller index: decided by the
/// entries' estimates when they are clearly apart, else by the exact cross terms.
fn entry_nearer(
    pairs: Pairs,
    cold: Option<&Cold>,
    m: usize,
    row: usize,
    x: Entry,
    y: Entry,
) -> bool {
    if clearly_nearer(x.rho(), y.rho()) {
        return true;
    }
    if clearly_nearer(y.rho(), x.rho()) {
        return false;
    }
    if let (Some(cold), Some(a), Some(b)) = (cold, x.index().get(), y.index().get()) {
        let (a, b) = (a as usize, b as usize);
        if cold.touches(row, a, b) {
            return cold.nearer_neighbor(row, a, b);
        }
    }
    let mut nearest = Nearest::selecting(&pairs.rows.windows[row], NonZeroUsize::MIN, 1);
    for entry in [x, y] {
        pairs.offer(m, row, &mut nearest, entry);
    }
    nearest.kept_indices().next() == x.index().get()
}

/// The entries of a row's run in a flat list, without the empty entry a row with none holds.
fn run_entries(run: &[(u64, Entry)]) -> &[(u64, Entry)] {
    match run.first() {
        Some((_, first)) if first.index().get().is_some() => run,
        _ => &[],
    }
}

/// Appends `row`'s up to `k` nearest of the two sorted runs `a` and `b`, which hold different
/// neighbors, nearest first; one empty entry when neither holds any.
fn merge_row(
    (pairs, cold): (Pairs, Option<&Cold>),
    m: usize,
    (row, k): (usize, usize),
    (a, b): (Run, Run),
    out: &mut Vec<(u64, Entry)>,
) -> Result<(), CapacityError> {
    let (a, b) = (run_entries(a), run_entries(b));
    out.try_reserve(k.min(a.len() + b.len()).max(1))
        .map_err(|_| CapacityError)?;
    let (mut i, mut j, before) = (0, 0, out.len());
    while out.len() - before < k {
        let entry = match (a.get(i), b.get(j)) {
            (None, None) => break,
            (Some(x), Some(y)) if entry_nearer(pairs, cold, m, row, y.1, x.1) => {
                j += 1;
                y
            }
            (Some(x), _) => {
                i += 1;
                x
            }
            (None, Some(y)) => {
                j += 1;
                y
            }
        };
        out.push(*entry);
    }
    if out.len() == before {
        out.push((row as u64, Entry::NONE));
    }
    Ok(())
}

/// The rows of `a` and `b`, flat lists over the same rows, each row's run taken in turn.
fn run_pairs<'a>(
    mut a: &'a [(u64, Entry)],
    mut b: &'a [(u64, Entry)],
) -> impl Iterator<Item = (usize, Run<'a>, Run<'a>)> {
    core::iter::from_fn(move || {
        let &(row, _) = a.first()?;
        let (run_a, rest_a) = a.split_at(a.iter().take_while(|e| e.0 == row).count());
        let (run_b, rest_b) = b.split_at(b.iter().take_while(|e| e.0 == row).count());
        (a, b) = (rest_a, rest_b);
        Some((row as usize, run_a, run_b))
    })
}

/// `old` and `new`, flat lists over the same rows holding different neighbors, merged into each
/// row's up to `k` nearest under the exact order (ties to the smaller index). Each side's top `k`
/// of two disjoint pair sets contains the top `k` of both, so this is the list a pass over both
/// sets would give, with an exact comparison only where the estimates cannot tell two entries
/// apart; that comparison is the cold route's when it ranks the row or either neighbor.
pub(crate) fn merge_kept(
    (pairs, cold): (Pairs, Option<&Cold>),
    m: NonZeroUsize,
    k: NonZeroUsize,
    old: &[(u64, Entry)],
    new: &[(u64, Entry)],
) -> Result<Vec<(u64, Entry)>, CapacityError> {
    let mut merged = reserved_vec(old.len() as u128)?;
    for (row, run_old, run_new) in run_pairs(old, new) {
        merge_row(
            (pairs, cold),
            m.get(),
            (row, k.get()),
            (run_old, run_new),
            &mut merged,
        )?;
    }
    Ok(merged)
}

/// Each row's left, right and up to `k` nearest over both sides, from the sides' flat lists as
/// [`two_sided_sides_over`] and [`merge_kept`] leave them: [`two_sided_topk_profile`]'s rows over
/// the pairs the lists were built from.
pub(crate) fn merged_two_sided(
    quantized: &Quantized,
    cold: Option<&Cold>,
    m: NonZeroUsize,
    k: NonZeroUsize,
    left: &[(u64, Entry)],
    right: &[(u64, Entry)],
) -> Result<Vec<(u64, TwoSidedTopk)>, CapacityError> {
    let mut rows = reserved_vec(quantized.windows.len() as u128)?;
    let mut both = Vec::new();
    for (row, run_left, run_right) in run_pairs(left, right) {
        both.clear();
        merge_row(
            (
                Pairs {
                    rows: quantized,
                    cols: quantized,
                },
                cold,
            ),
            m.get(),
            (row, k.get()),
            (run_left, run_right),
            &mut both,
        )?;
        let nearest = run_entries(&both).iter().map(|e| e.1).collect();
        rows.push((
            row as u64,
            TwoSidedTopk {
                left: run_left[0].1,
                right: run_right[0].1,
                nearest,
            },
        ));
    }
    Ok(rows)
}

/// The exact two-sided profile with the `k` nearest neighbors per window over both sides
/// (`k` clamped to the number of windows): per window start, in order, the left and right
/// nearest ([`two_sided_profile`]'s own) and up to `k` entries nearest first over both sides
/// under the exact order (ties to the smaller index; none for a window with no eligible
/// neighbor). Equal on bits to the exact oracle's two-sided top-k, whatever `threads` is; with
/// `k` of one, `nearest` holds [`two_sided_profile`]'s nearer side.
///
/// `Err(`[`CapacityError`]`)` when an allocation it needs is too large to make: the neighbors to
/// keep (the sum over windows of `min(k, finite windows that can reach it)`, per side) overflow
/// the address space or the allocator refuses them. That is decided before any work for the
/// windows the hot passes rank, and band by band for the windows only the cold route ranks. No
/// count is refused for its size alone.
pub fn two_sided_topk_profile<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> Result<Vec<(u64, TwoSidedTopk)>, CapacityError> {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    Ok(two_sided_topk_with(series, m, scale, threads, default_chunk_lanes(), None, k)?.0)
}

/// [`two_sided_topk_profile`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes (rounded up to a whole number of `chunk_lanes`).
/// Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn two_sided_topk_profile_tiled<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
    k: NonZeroUsize,
) -> Result<(Vec<(u64, TwoSidedTopk)>, BatchReport), CapacityError> {
    let series: Series<'a> = series.into();
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let (profile, report, _) =
        two_sided_topk_with(series, m, scale, threads, chunk_lanes.get(), Some(plan), k)?;
    Ok((profile, report))
}

/// A top-k two-sided profile, the run's report and what the cold route did.
type ToppedProfile = (Vec<(u64, TwoSidedTopk)>, BatchReport, SpillReport);

fn two_sided_topk_with(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    chunk_lanes: usize,
    plan: Option<TilePlan>,
    k: NonZeroUsize,
) -> Result<ToppedProfile, CapacityError> {
    let engine = Engine::self_join(true, k.get());
    let quantized = Quantized::build_on(series, m, scale, threads, engine);
    two_sided_spilling(
        series,
        scale,
        quantized,
        m,
        (threads, chunk_lanes),
        plan,
        k,
        |q, left, right, report, spill, cold| {
            let kept = NonZeroUsize::new(clamp_count(k.get(), q.windows.len()))
                .unwrap_or(NonZeroUsize::MIN);
            if cold.is_some() {
                let profile = merged_two_sided(q, cold, m, kept, &left, &right)?;
                return Ok((profile, report, spill));
            }
            let k = q.clamp_k(k);
            let mut profile = reserved_vec(q.windows.len() as u128)?;
            let (mut left, mut right) = (left.as_slice(), right.as_slice());
            while let (Some(&(row, _)), Some(_)) = (left.first(), right.first()) {
                let (left_run, left_rest) =
                    left.split_at(left.iter().take_while(|e| e.0 == row).count());
                let (right_run, right_rest) =
                    right.split_at(right.iter().take_while(|e| e.0 == row).count());
                (left, right) = (left_rest, right_rest);
                let at = row as usize;
                let mut nearest =
                    Nearest::with_k(&q.windows[at], m.get(), k, left_run.len() + right_run.len());
                let pairs = Pairs { rows: q, cols: q };
                nearest.select(
                    left_run
                        .iter()
                        .chain(right_run)
                        .filter_map(|&(_, entry)| pairs.pair(m.get(), at, entry)),
                );
                profile.push((
                    row,
                    TwoSidedTopk {
                        left: left_run[0].1,
                        right: right_run[0].1,
                        nearest: nearest.finish_k(),
                    },
                ));
            }
            Ok((profile, report, spill))
        },
    )
}

#[cfg(test)]
mod tests;
