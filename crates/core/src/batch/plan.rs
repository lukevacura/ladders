use super::pass::BLOCK_ROWS;
use super::survivor::Filter;
use crate::capacity::CapacityError;
use crate::row::RowAcc;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

mod costs;
mod gate;

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
pub(crate) use costs::break_even;
pub use costs::cold_route;
pub(crate) use costs::steps_through;
use costs::{MERGE, START, walks};

pub(crate) use gate::{Choice, Engine, RunShape, survivor_filter};

/// The most windows [`left_profile_threads`](super::left::left_profile_threads) runs single-threaded on any thread count:
/// spawning and merging cost more than they save (measured).
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) const SINGLE_THREADED_BELOW_ROWS: usize = 2_500;

/// The fewest windows a threaded run can repay its spawn at, on any thread count (measured).
#[cfg(all(feature = "parallel", not(loom)))]
const THREADED_MIN_ROWS: usize = 512;

/// Windows the cutoff grows by per thread asked: each thread spawned costs about this many
/// windows' work (measured).
#[cfg(all(feature = "parallel", not(loom)))]
const CUTOFF_ROWS_PER_THREAD: usize = 128;

/// Below this many windows a run on `threads` threads goes single-threaded: the cutoff grows
/// with the threads asked, from [`THREADED_MIN_ROWS`] up to [`SINGLE_THREADED_BELOW_ROWS`].
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) fn single_threaded_below_rows(threads: usize) -> usize {
    threads
        .saturating_mul(CUTOFF_ROWS_PER_THREAD)
        .clamp(THREADED_MIN_ROWS, SINGLE_THREADED_BELOW_ROWS)
}

/// Resident bytes per lane while a chunk is hot: `qt` (i64) plus the narrowed `enter`/`leave`/
/// `sum` columns (i32 each) plus `r` and `bias` (f64 each) and the `ρ̂` tile slot (f64).
const BYTES_PER_LANE: usize = 8 + 4 + 4 + 4 + 8 + 8 + 8;

/// A chunk's own working set: 32 KiB, measured as the fastest of 32/64/128 KiB on every target
/// and `n` tried, including the Apple aarch64 machine a larger budget was first guessed for.
const DEFAULT_CHUNK_BYTES: usize = 32 * 1024;

/// `bytes` of tiled hot-loop lanes, never zero (so a degenerate budget still makes progress one
/// lane at a time) — the same conversion [`default_chunk_lanes`] uses for its own budget, for
/// tuning or benchmarking a different one.
#[doc(hidden)]
pub fn chunk_lanes_for_bytes(bytes: usize) -> usize {
    (bytes / BYTES_PER_LANE).max(1)
}

/// The chunk length [`left_profile`](super::left::left_profile) picks by default: `DEFAULT_CHUNK_BYTES` of lanes.
#[doc(hidden)]
pub fn default_chunk_lanes() -> usize {
    chunk_lanes_for_bytes(DEFAULT_CHUNK_BYTES)
}

/// Tiles asked for per thread (2, 4, 8, 16 and 32 were timed at `n` of 10⁴ and 10⁵, `m` of 50
/// and 1000, 8 threads: all within 2% for 10⁵, and 8 or fewer were best at 10⁴ with `m = 1000`,
/// where more tiles only add starts).
pub(super) const TILES_PER_THREAD: usize = 8;

/// Bytes a thread's scratch may hold in each of its two parts (about an L2, portable): the band's
/// partial [`RowAcc`]s, and the tile's `qt` slots. So a thread's scratch never grows with `n`.
pub(super) const PARTIALS_BYTES: usize = 1 << 20;

/// What a cut covers: the left profile's triangle, or the AB-join's matrix of `n_a` rows by `n_b`
/// columns.
#[derive(Clone, Copy)]
enum Extent {
    Triangle { rows: usize },
    Join { n_a: usize, n_b: usize },
}

impl Extent {
    pub(super) fn rows(self) -> usize {
        match self {
            Self::Triangle { rows } => rows,
            Self::Join { n_a, .. } => n_a,
        }
    }

    /// The most slots any band has.
    fn lanes(self) -> usize {
        match self {
            Self::Triangle { rows } => rows,
            Self::Join { n_a, n_b } => n_a + n_b - 1,
        }
    }

    fn work(self) -> f64 {
        match self {
            Self::Triangle { rows } => rows as f64 * rows as f64 / 2.0,
            Self::Join { n_a, n_b } => n_a as f64 * n_b as f64,
        }
    }

    /// The tiles of `band` rows by `tile` lanes: from below for the triangle (band `i` of
    /// `bands` has at least `i · band` lanes, so `i · band / tile` tiles, and the last has
    /// `columns`; never above [`cut_tiles`]), exactly [`cut_join`]'s for the join.
    pub(super) fn tiles(self, band: usize, tile: usize) -> usize {
        match self {
            Self::Triangle { rows } => {
                let (bands, columns) = (rows.div_ceil(band) as f64, rows.div_ceil(tile) as f64);
                (band as f64 / tile as f64 * bands * (bands - 1.0) / 2.0 + columns) as usize
            }
            Self::Join { n_a, n_b } => {
                let bands = n_a.div_ceil(band);
                let last = n_a - (bands - 1) * band;
                (bands - 1) * (band + n_b - 1).div_ceil(tile) + (last + n_b - 1).div_ceil(tile)
            }
        }
    }
}

/// The band height and tile width [`left_profile_threads`](super::left::left_profile_threads) cuts a series of `n_windows` windows
/// into, for `threads` threads: `(band_rows, tile_lanes)`.
///
/// **Model.** With `W` the pair work, `B` the band height, `Wt` the tile width and `c` one lane's
/// cost in a row's pass, `T` threads take about
/// `(W / T) · (1 + START · m / B + 2 · MERGE / Wt) + B · Wt` (in units of `c`): the
/// start of every lane costs `START · m` per `B` rows, every row's merge `MERGE` per
/// `Wt` lanes, and the last tile to finish adds its own `B · Wt`. The start cost falls with `B`,
/// the merge cost with `Wt`, the tail rises with both.
///
/// **Rules.** The pair `(B, Wt)` minimizing the model among those that
/// 1. have `B` a multiple of [`BLOCK_ROWS`] (or the whole series) with `B · size_of::<RowAcc>()`
///    within `PARTIALS_BYTES` (the memory cap: per-thread scratch is flat in `n`),
/// 2. have `Wt` a whole number `k` of L1 chunks ([`default_chunk_lanes`]), never narrower, with
///    `Wt` slots of `qt` (8 bytes each) within `PARTIALS_BYTES` too, and
/// 3. cut at least `TILES_PER_THREAD × threads` tiles (or as many as the series allows, when it
///    is too small to reach that).
///
/// The start cost is weighed against the rest, not floored: a small series with a large `m` takes
/// short bands. More tiles than rule 3 asks for are taken only when the model prices them lower.
///
/// **Constants.** `MERGE` (one row merge `μ`, a lock and [`RowAcc::merge`]) and `START` (a
/// lane's start per window sample) are this target's measured ratios to `c`, owned by
/// [`costs`]; `TILES_PER_THREAD` is the timed best.
pub(super) fn tile_plan(n_windows: usize, m: usize, threads: usize) -> (usize, usize) {
    let rows = n_windows.saturating_sub(crate::exclusion::zone(m) + 1);
    plan_extent(Extent::Triangle { rows }, m, threads)
}

/// [`tile_plan`]'s model over any [`Extent`].
fn plan_extent(extent: Extent, m: usize, threads: usize) -> (usize, usize) {
    let chunk = default_chunk_lanes();
    let rows = extent.rows();
    if rows == 0 {
        return (BLOCK_ROWS, chunk);
    }
    let threads = threads.max(1);
    let target = TILES_PER_THREAD.saturating_mul(threads);
    let cap =
        (PARTIALS_BYTES / core::mem::size_of::<RowAcc>()).max(BLOCK_ROWS) / BLOCK_ROWS * BLOCK_ROWS;
    let tile_cap = (PARTIALS_BYTES / core::mem::size_of::<i64>() / chunk).max(1) * chunk;
    let work = extent.work();
    let tiles = |band: usize, tile: usize| extent.tiles(band, tile);
    let cost = |band: usize, tile: usize| {
        let (b, w) = (band as f64, tile as f64);
        work / threads as f64 * (1.0 + START * m as f64 / b + 2.0 * MERGE / w) + b * w
    };

    let mut best: Option<((usize, usize), usize, f64)> = None;
    let mut band = BLOCK_ROWS;
    loop {
        let band_rows = band.min(cap).min(rows);
        let mut k = 1;
        loop {
            let tile = (k * chunk).min(tile_cap);
            let (count, cost) = (tiles(band_rows, tile), cost(band_rows, tile));
            // Enough tiles beats few; among enough (or among equally few), the cheaper wins.
            let (count, better) = match best {
                None => (count, true),
                Some((_, bc, bcost)) => {
                    let (have, had) = (count.min(target), bc.min(target));
                    (count, have > had || (have == had && cost < bcost))
                }
            };
            if better {
                best = Some(((band_rows, tile), count, cost));
            }
            if tile >= extent.lanes() || tile >= tile_cap {
                break;
            }
            k += 1 + k / 4;
        }
        if band >= cap || band >= rows {
            break;
        }
        band *= 2;
    }
    best.map_or((BLOCK_ROWS, chunk), |(plan, _, _)| plan)
}

/// A set of diagonal slots as sorted runs, none empty and none touching another: the diagonals a
/// restricted run visits. Tiles keep their usual slot ranges; a restricted tile walks only the
/// slots of its range that are in the set, and a range with none is not cut at all.
pub(crate) struct SlotRuns {
    runs: Vec<core::ops::Range<usize>>,
}

impl SlotRuns {
    /// The runs of `slots`, which may come in any order; a repeated slot counts once.
    pub(crate) fn from_slots(mut slots: Vec<usize>) -> Self {
        slots.sort_unstable();
        slots.dedup();
        let mut runs: Vec<core::ops::Range<usize>> = Vec::new();
        for slot in slots {
            match runs.last_mut() {
                Some(run) if run.end == slot => run.end += 1,
                _ => runs.push(slot..slot + 1),
            }
        }
        Self { runs }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// The walk-or-tile split, the one place it is decided for every engine: the runs with at
    /// least this target's [`costs::WALK_BELOW_RUN`] slots as a set, and every slot of the shorter
    /// ones, in ascending order: the second part is what the diagonal-major walk takes. A run the
    /// route force sends the other way follows the force.
    pub(super) fn split(&self) -> Result<(Self, Vec<usize>), CapacityError> {
        let mut tiled = Vec::new();
        let mut walked: Vec<usize> = Vec::new();
        for run in &self.runs {
            if crate::route::sparse_walk(walks(run.len())) {
                walked.try_reserve(run.len()).map_err(|_| CapacityError)?;
                walked.extend(run.clone());
            } else {
                tiled.try_reserve(1).map_err(|_| CapacityError)?;
                tiled.push(run.clone());
            }
        }
        Ok((Self { runs: tiled }, walked))
    }

    /// The runs' parts inside `lo..hi`, in order.
    pub(super) fn within(
        &self,
        lo: usize,
        hi: usize,
    ) -> impl Iterator<Item = core::ops::Range<usize>> + '_ {
        let first = self.runs.partition_point(|run| run.end <= lo);
        self.runs[first..]
            .iter()
            .take_while(move |run| run.start < hi)
            .map(move |run| run.start.max(lo)..run.end.min(hi))
    }
}

/// `band`'s tiles over the slots `lanes`, each at most `tile_lanes` wide, pushed in ascending
/// slots; with `runs`, only the tiles whose range holds one of its slots. A band that keeps none
/// still gets one tile with no slot, so its rows are finished like any other's.
fn push_band_tiles(
    tiles: &mut impl FnMut(Tile),
    (band, rows): (usize, core::ops::Range<usize>),
    lanes: core::ops::Range<usize>,
    tile_lanes: usize,
    runs: Option<&SlotRuns>,
) {
    let mut column = 0;
    for qa in lanes.clone().step_by(tile_lanes) {
        let qb = (qa + tile_lanes).min(lanes.end);
        if runs.is_some_and(|runs| runs.within(qa, qb).next().is_none()) {
            continue;
        }
        tiles(Tile {
            band,
            rows: rows.clone(),
            column,
            qa,
            qb,
        });
        column += 1;
    }
    if column == 0 {
        tiles(Tile {
            band,
            rows,
            column: 0,
            qa: lanes.start,
            qb: lanes.start,
        });
    }
}

/// The triangle of `n_windows` windows cut into bands of `band_rows` rows (from the first row with
/// a lag, `exclusion + 1`) and tiles of `tile_lanes` lanes; the last of a band may be narrower. A
/// band's lanes are those of its last row. With `runs`, only the tiles holding one of its slots.
pub(super) fn cut_tiles(
    n_windows: usize,
    exclusion: usize,
    band_rows: usize,
    tile_lanes: usize,
    runs: Option<&SlotRuns>,
) -> Vec<Tile> {
    let mut tiles = Vec::new();
    each_tile(
        n_windows,
        exclusion,
        band_rows,
        tile_lanes,
        runs,
        &mut |tile| tiles.push(tile),
    );
    tiles
}

/// [`cut_tiles`]' tiles, handed to `visit` one by one in the same order.
fn each_tile(
    n_windows: usize,
    exclusion: usize,
    band_rows: usize,
    tile_lanes: usize,
    runs: Option<&SlotRuns>,
    visit: &mut impl FnMut(Tile),
) {
    let mut start = exclusion + 1;
    while start < n_windows {
        let band = (start - exclusion - 1) / band_rows;
        let rows = start..(start + band_rows).min(n_windows);
        // Lanes the band's last row has: slots `0..rows.end − 1 − exclusion`.
        let lanes = rows.end - 1 - exclusion;
        start = rows.end;
        push_band_tiles(visit, (band, rows), 0..lanes, tile_lanes, runs);
    }
}

/// The matrix of `n_a` rows by `n_b` columns cut into bands of `band_rows` rows and tiles of
/// `tile_lanes` slots; the last of a band may be narrower. A band's slots run from its first row
/// to its last row plus `n_b - 1`. With `runs`, only the tiles holding one of its slots.
fn cut_join(
    n_a: usize,
    n_b: usize,
    band_rows: usize,
    tile_lanes: usize,
    runs: Option<&SlotRuns>,
) -> Vec<Tile> {
    let mut tiles = Vec::new();
    each_join_tile(n_a, n_b, band_rows, tile_lanes, runs, &mut |tile| {
        tiles.push(tile);
    });
    tiles
}

/// [`cut_join`]'s tiles, handed to `visit` one by one in the same order.
fn each_join_tile(
    n_a: usize,
    n_b: usize,
    band_rows: usize,
    tile_lanes: usize,
    runs: Option<&SlotRuns>,
    visit: &mut impl FnMut(Tile),
) {
    let mut start = 0;
    while start < n_a {
        let rows = start..(start + band_rows).min(n_a);
        let lanes = start..rows.end + n_b - 1;
        let band = start / band_rows;
        start = rows.end;
        push_band_tiles(visit, (band, rows), lanes, tile_lanes, runs);
    }
}

/// The tiles an engine runs, cut under a clamped plan, and how many leading rows have no pair at
/// all (they are [`Entry::NONE`](crate::entry::Entry::NONE)).
pub(super) struct TileCut {
    pub(super) plan: TilePlan,
    pub(super) none_rows: usize,
    pub(super) tiles: Vec<Tile>,
}

impl TileCut {
    /// The left profile's triangle over `n_windows` windows, or only `runs`' diagonals of it.
    pub(super) fn triangle(
        n_windows: usize,
        m: usize,
        plan: TilePlan,
        runs: Option<&SlotRuns>,
    ) -> Self {
        let plan = plan.clamped(n_windows, n_windows);
        let exclusion = crate::exclusion::zone(m);
        Self {
            plan,
            none_rows: n_windows.min(exclusion + 1),
            tiles: cut_tiles(n_windows, exclusion, plan.band_rows, plan.tile_lanes, runs),
        }
    }

    /// The tiles [`Self::triangle`] cuts for all diagonals, handed to `visit` one by one without
    /// being collected.
    pub(super) fn visit_triangle(
        n_windows: usize,
        m: usize,
        plan: TilePlan,
        mut visit: impl FnMut(&Tile),
    ) {
        let plan = plan.clamped(n_windows, n_windows);
        each_tile(
            n_windows,
            crate::exclusion::zone(m),
            plan.band_rows,
            plan.tile_lanes,
            None,
            &mut |tile| visit(&tile),
        );
    }

    /// The tiles [`Self::join`] cuts for all diagonals, handed to `visit` one by one without being
    /// collected.
    pub(super) fn visit_join(n_a: usize, n_b: usize, plan: TilePlan, mut visit: impl FnMut(&Tile)) {
        let plan = plan.clamped(n_a, n_a + n_b - 1);
        each_join_tile(
            n_a,
            n_b,
            plan.band_rows,
            plan.tile_lanes,
            None,
            &mut |tile| {
                visit(&tile);
            },
        );
    }

    /// The AB-join's matrix, both sides non-empty, or only `runs`' diagonals of it.
    pub(super) fn join(n_a: usize, n_b: usize, plan: TilePlan, runs: Option<&SlotRuns>) -> Self {
        let plan = plan.clamped(n_a, n_a + n_b - 1);
        Self {
            plan,
            none_rows: 0,
            tiles: cut_join(n_a, n_b, plan.band_rows, plan.tile_lanes, runs),
        }
    }
}

/// What the tile engines cut the triangle into: bands of `band_rows` rows, tiles of `tile_lanes`
/// lanes (a whole number of `chunk_lanes`), walked in L1 sub-chunks of `chunk_lanes`.
#[derive(Clone, Copy)]
pub(super) struct TilePlan {
    pub(super) band_rows: usize,
    pub(super) chunk_lanes: usize,
    pub(super) tile_lanes: usize,
    /// Neighbors kept per row.
    pub(super) k: NonZeroUsize,
    /// The survivor filter the run installs: off until the run sets its own decision
    /// ([`survivor_filter`], carried by its quantized series).
    pub(super) survivor: Filter,
}

impl TilePlan {
    /// The given shape, `tile_lanes` rounded up to a whole number of `chunk_lanes`.
    pub(super) fn fixed(
        band_rows: NonZeroUsize,
        chunk_lanes: NonZeroUsize,
        tile_lanes: NonZeroUsize,
    ) -> Self {
        Self {
            band_rows: band_rows.get(),
            chunk_lanes: chunk_lanes.get(),
            tile_lanes: tile_lanes
                .get()
                .div_ceil(chunk_lanes.get())
                .saturating_mul(chunk_lanes.get()),
            k: NonZeroUsize::MIN,
            survivor: Filter::Off,
        }
    }

    /// No row exists beyond `max_rows` and no lane beyond `max_lanes`, so taller bands and wider
    /// chunks and tiles would only waste scratch.
    fn clamped(self, max_rows: usize, max_lanes: usize) -> Self {
        let chunk_lanes = self.chunk_lanes.clamp(1, max_lanes.max(1));
        Self {
            band_rows: self.band_rows.clamp(1, max_rows.max(1)),
            k: self.k,
            survivor: self.survivor,
            chunk_lanes,
            tile_lanes: self
                .tile_lanes
                .clamp(chunk_lanes, max_lanes.max(chunk_lanes)),
        }
    }

    /// [`tile_plan`]'s cut for `threads`, with tiles a whole number of `chunk_lanes`.
    pub(super) fn for_threads(
        n_windows: usize,
        m: usize,
        threads: usize,
        chunk_lanes: usize,
    ) -> Self {
        Self::rounded(tile_plan(n_windows, m, threads), chunk_lanes)
    }

    /// [`plan_extent`]'s cut of a join, with tiles a whole number of `chunk_lanes`.
    pub(super) fn for_join(
        n_a: usize,
        n_b: usize,
        m: usize,
        threads: usize,
        chunk_lanes: usize,
    ) -> Self {
        Self::rounded(
            plan_extent(Extent::Join { n_a, n_b }, m, threads),
            chunk_lanes,
        )
    }

    fn rounded((band_rows, tile_lanes): (usize, usize), chunk_lanes: usize) -> Self {
        Self {
            band_rows,
            chunk_lanes,
            tile_lanes: tile_lanes.div_ceil(chunk_lanes).saturating_mul(chunk_lanes),
            k: NonZeroUsize::MIN,
            survivor: Filter::Off,
        }
    }
}

impl Tile {
    /// The slots to walk, in ascending order: all of `qa..qb`, or with `runs` those in the set.
    pub(super) fn slots<'a>(&self, runs: Option<&'a SlotRuns>) -> impl Iterator<Item = usize> + 'a {
        let (qa, qb) = (self.qa, self.qb);
        let whole = runs.is_none().then_some(qa..qb).into_iter().flatten();
        let chosen = runs
            .into_iter()
            .flat_map(move |runs| runs.within(qa, qb))
            .flatten();
        whole.chain(chosen)
    }
}

/// One tile: rows `rows` (a band) against slots `qa..qb` (up to the tile width, walked in L1
/// sub-chunks), the `column`th of its band.
pub(super) struct Tile {
    pub(super) band: usize,
    pub(super) rows: core::ops::Range<usize>,
    #[cfg_attr(not(all(feature = "parallel", not(loom))), allow(dead_code))]
    pub(super) column: usize,
    pub(super) qa: usize,
    pub(super) qb: usize,
}

/// `threads`, or one when a series of `len` samples has too few windows to repay spawning.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) fn threads_for_size(len: usize, m: NonZeroUsize, threads: NonZeroUsize) -> NonZeroUsize {
    if (len + 1).saturating_sub(m.get()) < single_threaded_below_rows(threads.get()) {
        NonZeroUsize::MIN
    } else {
        threads
    }
}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) fn threads_for_size(
    _len: usize,
    _m: NonZeroUsize,
    threads: NonZeroUsize,
) -> NonZeroUsize {
    threads
}

/// `threads`, or one when `n_a` by `n_b` windows are too few pairs to repay spawning: the
/// self-join's own pair count at its single-threaded size.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) fn threads_for_join(n_a: usize, n_b: usize, threads: NonZeroUsize) -> NonZeroUsize {
    let rows = single_threaded_below_rows(threads.get());
    if n_a.saturating_mul(n_b) < rows * rows / 2 {
        NonZeroUsize::MIN
    } else {
        threads
    }
}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) fn threads_for_join(_n_a: usize, _n_b: usize, threads: NonZeroUsize) -> NonZeroUsize {
    threads
}

/// The worker threads [`left_profile_threads`](super::left::left_profile_threads) runs for a series of `n_windows` windows with
/// window length `m` and `threads` asked: one on the single-threaded path, otherwise
/// `min(threads, tiles)` of [`tile_plan`]'s cut. For tests of the layers above; assumes the
/// series fits one anchor (a wider one runs the same tiles).
#[cfg(all(feature = "parallel", not(loom)))]
#[doc(hidden)]
pub fn threaded_workers(n_windows: usize, m: usize, threads: usize) -> usize {
    if threads <= 1 || n_windows < single_threaded_below_rows(threads) {
        return 1;
    }
    let (band_rows, tile_lanes) = tile_plan(n_windows, m, threads);
    let tiles = cut_tiles(
        n_windows,
        crate::exclusion::zone(m),
        band_rows,
        tile_lanes,
        None,
    )
    .len();
    threads.min(tiles).max(1)
}

#[cfg(test)]
mod tests {
    use super::{TILES_PER_THREAD, tile_plan};

    /// Every engine's walk-or-tile split walks each run shorter than this target's threshold and
    /// tiles each other run, whole.
    #[cfg(not(any(loom, shuttle)))]
    #[test]
    fn the_split_walks_short_runs_and_tiles_the_rest() {
        use super::SlotRuns;
        use super::costs::WALK_BELOW_RUN;
        let _guard = crate::route::forced_route(crate::route::RouteForce::default());
        let short = 100..100 + WALK_BELOW_RUN - 1;
        let long = 200..200 + WALK_BELOW_RUN;
        let single = 300..301;
        let slots = short
            .clone()
            .chain(long.clone())
            .chain(single.clone())
            .collect();
        let (tiled, walked) = SlotRuns::from_slots(slots).split().unwrap();
        assert_eq!(tiled.runs, std::vec![long]);
        let expected: std::vec::Vec<usize> = short.chain(single).collect();
        assert_eq!(walked, expected);
    }

    #[test]
    fn a_thread_count_past_overflow_plans_like_any_larger_one() {
        let (n_windows, m) = (6_000, 32);
        let plan = tile_plan(n_windows, m, usize::MAX);
        assert!(plan.0 > 0 && plan.1 > 0);
        assert_eq!(
            plan,
            tile_plan(n_windows, m, usize::MAX / TILES_PER_THREAD + 1)
        );
        assert_eq!(plan, tile_plan(n_windows, m, usize::MAX / 2));
    }
}
