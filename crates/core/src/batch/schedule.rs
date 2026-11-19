#[cfg(test)]
use super::VISITS;
use super::pass::push_none_rows;
use super::plan::{Tile, TileCut, TilePlan};
use super::tile::{RanWide, Scratch};
#[cfg(all(test, feature = "parallel", not(loom)))]
use super::{FRESH_GROUPS, PAR_RUNS, PEAK_LIVE_BANDS, TILE_RENDEZVOUS};
use crate::capacity::CapacityError;
use crate::entry::Entry;
use crate::nearest::Nearest;
use crate::row::RowAcc;
use crate::stats::Window;
#[cfg(all(feature = "parallel", not(loom)))]
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Rows whose accumulators share one lock, and one cache-line-padded cell.
#[cfg(all(feature = "parallel", not(loom)))]
const ROW_GROUP: usize = 32;

/// 128 bytes (Apple's line size), so two threads' locks never share a line.
#[cfg(all(feature = "parallel", not(loom)))]
#[repr(align(128))]
struct Padded<T>(T);

/// The guard of `cell`, ignoring poison (a panic anywhere already propagates out of the pool).
#[cfg(all(feature = "parallel", not(loom)))]
fn lock<T>(cell: &crate::sync::Mutex<T>) -> impl core::ops::DerefMut<Target = T> + '_ {
    match cell.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Bands whose accumulators may be alive per thread: tiles are ordered so a band is complete
/// about this many bands (per thread) after its first tile starts, and no later. A smaller
/// window frees accumulators sooner; it must stay at least one band per thread for the tiles
/// that run side by side to belong to different bands (so merge into different locks).
#[cfg(all(feature = "parallel", not(loom)))]
const WINDOW_BANDS_PER_THREAD: usize = 1;

/// How many independent tiles an engine ran, and how many of them needed the exact i64 pass. Both
/// are fixed by the input and the tile shape.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchReport {
    pub tiles: usize,
    pub wide_tiles: usize,
}

impl BatchReport {
    /// A run of `tiles` tiles, `wide_tiles` of them on the exact i64 pass.
    pub fn of(tiles: usize, wide_tiles: usize) -> Self {
        Self { tiles, wide_tiles }
    }
}

/// What the tile scheduler asks of a metric: the scratch a tile works in, the per-row partials it
/// fills, how partials merge and rank, and the tile's own body. The scheduler knows nothing of
/// the arithmetic, so every metric runs on this one set of bands, tiles, locks and merges.
///
/// **Tile contract.** A tile's input is pure (its tick slices and the window stats) and its output
/// is a candidate set per row ([`Self::run`] fills the scratch's partials); partials meet only in
/// [`Self::merge`], which is order-independent. So the result does not depend on the executor, the
/// thread count or the tile shape.
pub(super) trait TileWork {
    type Scratch;
    type Partial;
    /// One finished row of the profile.
    type Row;

    fn scratch(&self, plan: TilePlan) -> Self::Scratch;
    /// The scratch's partials, one per band row.
    fn partials<'a>(&self, scratch: &'a mut Self::Scratch) -> &'a mut [Self::Partial];
    /// Fills the scratch's partials for `tile`'s rows from the tile's own pairs alone.
    fn run(&self, tile: &Tile, scratch: &mut Self::Scratch, chunk_lanes: usize) -> RanWide;
    /// Whether the threaded engine starts the top bands first (a walk that hands each band's lanes
    /// to the band below needs the band above done first), instead of the longest bands first. The
    /// tiles run one after another in the cut's order either way, which is top to bottom.
    #[cfg_attr(not(all(feature = "parallel", not(loom))), allow(dead_code))] // only the threaded engine orders by it
    fn top_down(&self) -> bool {
        false
    }
    fn new_partial(&self) -> Self::Partial;
    /// Empties `partial` for `row`.
    fn begin(&self, partial: &mut Self::Partial, row: usize);
    /// Folds `from` into `into`; order-independent.
    fn merge(&self, into: &mut Self::Partial, from: &mut Self::Partial);
    /// Appends `row`'s finished entries, from a partial every tile has merged into.
    fn finish(&self, row: usize, partial: &mut Self::Partial, out: &mut Vec<(u64, Self::Row)>);
    /// Appends `count` leading rows that have no pair at all.
    fn push_none_rows(&self, count: usize, out: &mut Vec<(u64, Self::Row)>);
    /// Test-only: the neighbors the partial was offered.
    #[cfg(test)]
    fn visits(&self, partial: &Self::Partial) -> Vec<usize>;
}

/// The z-normalized metric's tile work: its tile `body`, over the series' `windows`.
struct ZNormTiles<'a, F> {
    windows: &'a [Window],
    m: usize,
    k: NonZeroUsize,
    body: &'a F,
    #[cfg_attr(not(all(feature = "parallel", not(loom))), allow(dead_code))]
    // only the threaded engine orders by it
    top_down: bool,
}

impl<F: Fn(&Tile, &mut Scratch, usize) -> RanWide> TileWork for ZNormTiles<'_, F> {
    type Scratch = Scratch;
    type Partial = RowAcc;
    type Row = Entry;

    fn scratch(&self, plan: TilePlan) -> Scratch {
        Scratch::new(plan)
    }

    fn partials<'a>(&self, scratch: &'a mut Scratch) -> &'a mut [RowAcc] {
        &mut scratch.accs
    }

    fn run(&self, tile: &Tile, scratch: &mut Scratch, chunk_lanes: usize) -> RanWide {
        (self.body)(tile, scratch, chunk_lanes)
    }

    fn top_down(&self) -> bool {
        self.top_down
    }

    fn new_partial(&self) -> RowAcc {
        RowAcc::new()
    }

    fn begin(&self, partial: &mut RowAcc, row: usize) {
        partial.begin(&self.windows[row], self.k, self.windows.len());
    }

    fn merge(&self, into: &mut RowAcc, from: &mut RowAcc) {
        into.merge(from);
    }

    fn finish(&self, row: usize, partial: &mut RowAcc, out: &mut Vec<(u64, Entry)>) {
        let mut nearest = Nearest::with_k(&self.windows[row], self.m, self.k, self.windows.len());
        partial.finish(&mut nearest);
        nearest.push_kept(row as u64, self.k, out);
    }

    fn push_none_rows(&self, count: usize, out: &mut Vec<(u64, Entry)>) {
        push_none_rows(count, out, || Entry::NONE);
    }

    #[cfg(test)]
    fn visits(&self, partial: &RowAcc) -> Vec<usize> {
        partial.visits.clone()
    }
}

/// The tiles of `work`, run one after another on this thread, band by band: each tile's
/// partials merge into its band's accumulators, which rank into entries when the band's last tile
/// has merged. Needs no thread or lock, so it is the engine with the `parallel` feature off; the
/// threaded engine's result, bit for bit.
#[cfg_attr(all(feature = "parallel", not(loom)), allow(dead_code))]
fn tiles_in_order<W: TileWork>(
    work: &W,
    cut: TileCut,
    results: &mut Vec<(u64, W::Row)>,
) -> BatchReport {
    let TileCut {
        plan,
        none_rows,
        tiles,
    } = cut;
    work.push_none_rows(none_rows, results);
    if tiles.is_empty() {
        return BatchReport::of(0, 0);
    }
    let mut scratch = work.scratch(plan);
    let mut band: Vec<W::Partial> = (0..plan.band_rows).map(|_| work.new_partial()).collect();
    let mut wide_tiles = 0;
    for (i, tile) in tiles.iter().enumerate() {
        if i == 0 || tiles[i - 1].band != tile.band {
            for (acc, row) in band.iter_mut().zip(tile.rows.clone()) {
                work.begin(acc, row);
            }
        }
        wide_tiles += usize::from(work.run(tile, &mut scratch, plan.chunk_lanes));
        for (acc, mine) in band
            .iter_mut()
            .zip(&mut work.partials(&mut scratch)[..tile.rows.len()])
        {
            work.merge(acc, mine);
        }
        if tiles.get(i + 1).is_none_or(|next| next.band != tile.band) {
            for (row, acc) in tile.rows.clone().zip(band.iter_mut()) {
                #[cfg(test)]
                VISITS.with(|v| v.borrow_mut().push((row, work.visits(acc))));
                work.finish(row, acc, results);
            }
        }
    }
    BatchReport {
        tiles: tiles.len(),
        wide_tiles,
    }
}

/// [`tiles_in_order`] on an [`Executor`](crate::par::Executor), all through one `run` call (no
/// thread, atomic or `unsafe` here).
///
/// **Tiles.** A band of `band_rows` rows times `tile_lanes` lag slots (a whole number of
/// `chunk_lanes`; the last tile of a band may be narrower); empty ones are skipped. A tile does
/// not wait for any other: `body` fills its partial [`RowAcc`] per row from the tile's own
/// inputs alone, and the tile width is a scheduling knob (see [`tile_plan`](super::plan::tile_plan)); the L1 chunk width
/// stays the cache budget.
///
/// **Schedule.** Bands go from the last (longest) to the first, and each band's tiles are spread
/// evenly over the next `WINDOW_BANDS_PER_THREAD × threads` bands' worth of the order. So the
/// long tiles start first and the thin ones along the diagonal fill the tail (the longest-first
/// rule for independent tasks, within one smallest task of balanced), tiles that start together
/// belong to different bands and so merge into different row groups (band by band would have the
/// threads queue on one band's locks), and a band is complete soon after the window passes it.
/// No tile waits, so the order is only about balance, lock contention and memory.
///
/// **Merge.** A finished tile merges each row's partial into the band's accumulator for that row,
/// under a [`Mutex`] per [`ROW_GROUP`] rows, padded to 128 bytes. A band's accumulators are made
/// by its first tile to merge. Its last tile to merge ranks every row ([`RowAcc::finish`]) into
/// the band's entries and drops the accumulators. [`RowAcc::merge`] is order-independent, so
/// which tile lands first is irrelevant.
///
/// **Determinism.** A tile's rows see the same lane states and inputs the single-threaded
/// pass gives them, in row order; partials combine with the one `merge`/`finish` pair. The
/// result is the single-threaded one on bits, for any thread count, band height, tile width or
/// chunk width, and for any mix of narrow and wide tiles.
///
/// **Memory.** Live accumulators belong to the bands the window spans (at most about twice
/// the window, counting bands whose last tile is still merging), not to every row: `O(window ×
/// band_rows)`. Each running tile has scratch of `tile_lanes` slots, a tile of `chunk_lanes`, one
/// band's scalars and `band_rows` partials, plus its two narrowed sides (`O(band_rows +
/// tile_lanes + m)` ticks), taken from a shared stack and returned, so nothing is allocated per
/// tile after the first `threads` of them. [`tile_plan`](super::plan::tile_plan) caps the slots and the partials at
/// `PARTIALS_BYTES`, so a thread's scratch is flat in `n`. What remains `O(n)` is the series, its
/// columns and the finished entries, as in the single-threaded pass.
#[cfg(all(feature = "parallel", not(loom)))]
fn tiles_threaded<W>(
    work: &W,
    cut: TileCut,
    executor: &impl crate::par::Executor,
    results: &mut Vec<(u64, W::Row)>,
) -> BatchReport
where
    W: TileWork + Sync,
    W::Scratch: Send,
    W::Partial: Send,
    W::Row: Send,
{
    let TileCut {
        plan,
        none_rows,
        mut tiles,
    } = cut;
    let chunk_lanes = plan.chunk_lanes;
    use crate::sync::{Arc, Mutex};

    #[cfg(test)]
    PAR_RUNS.with(|runs| runs.set(runs.get() + 1));
    work.push_none_rows(none_rows, results);
    let Some(last) = tiles.last() else {
        return BatchReport::of(0, 0);
    };

    let bands = last.band + 1;
    let mut tiles_in_band = vec![0usize; bands];
    for t in &tiles {
        tiles_in_band[t.band] += 1;
    }
    // Longest bands first, each band's tiles spread evenly over the next `window` bands' worth of
    // the order: tiles that run side by side belong to different bands (so merge into different
    // row groups, where band by band would queue on one band's locks), and a band is complete
    // soon after the window passes it, so only about `window` bands' accumulators are alive.
    let window = (WINDOW_BANDS_PER_THREAD * executor.width().get()) as f64;
    let top_down = work.top_down();
    let priority = |t: &Tile| {
        let columns = tiles_in_band[t.band] as f64;
        let column = t.column as f64;
        let band = if top_down { t.band } else { bands - 1 - t.band };
        band as f64 + window * (column + 0.5) / columns
    };
    tiles.sort_by(|x, y| {
        priority(x)
            .total_cmp(&priority(y))
            .then(x.band.cmp(&y.band))
            .then(x.qa.cmp(&y.qa))
    });

    // A band's accumulators exist from its first tile's merge to its last tile's: then its rows
    // are finished into `done` and the accumulators dropped.
    let live: Vec<Mutex<Option<LiveBand<W::Partial>>>> =
        (0..bands).map(|_| Mutex::new(None)).collect();
    let done: Vec<Mutex<Option<BandDone<W::Row>>>> = (0..bands).map(|_| Mutex::new(None)).collect();
    let spare: Mutex<Vec<W::Scratch>> = Mutex::new(Vec::new());
    let spare_bands: Mutex<Vec<Vec<Vec<W::Partial>>>> = Mutex::new(Vec::new());
    #[cfg(test)]
    let tracker = Arc::new(LiveBands::default());
    #[cfg(test)]
    let rendezvous = Rendezvous::new(TILE_RENDEZVOUS.with(core::cell::Cell::get));

    let ran_wide = executor.run(tiles.len(), |i| {
        let tile = &tiles[i];
        #[cfg(test)]
        rendezvous.arrive();
        let (rows, band) = (&tile.rows, tile.band);
        let mut scratch = lock(&spare).pop().unwrap_or_else(|| work.scratch(plan));
        let wide = work.run(tile, &mut scratch, chunk_lanes);
        let accs = work.partials(&mut scratch);

        let target = lock(&live[band])
            .get_or_insert_with(|| {
                Arc::new(Band::open(
                    work,
                    &spare_bands,
                    rows.clone(),
                    tiles_in_band[band],
                    #[cfg(test)]
                    Arc::clone(&tracker),
                ))
            })
            .clone();
        for (g, start) in (0..rows.len()).step_by(ROW_GROUP).enumerate() {
            let end = (start + ROW_GROUP).min(rows.len());
            let mut cell = lock(&target.groups[g].0);
            for (acc, mine) in cell.iter_mut().zip(&mut accs[start..end]) {
                work.merge(acc, mine);
            }
        }
        lock(&spare).push(scratch);

        let last = {
            let mut left = lock(&target.left);
            *left -= 1;
            *left == 0
        };
        if last {
            *lock(&done[band]) = Some(target.finish(work, rows.start, &spare_bands));
            lock(&live[band]).take();
        }
        wide
    });

    #[cfg(test)]
    PEAK_LIVE_BANDS.with(|p| p.set(tracker.peak.load(crate::sync::Ordering::SeqCst)));
    #[cfg(test)]
    FRESH_GROUPS.with(|p| p.set(tracker.fresh_groups.load(crate::sync::Ordering::SeqCst)));
    for slot in done {
        let band = lock(&slot)
            .take()
            .expect("every band's last tile finishes it");
        results.extend(band.entries);
        #[cfg(test)]
        VISITS.with(|v| v.borrow_mut().extend(band.visits));
    }
    BatchReport {
        tiles: tiles.len(),
        wide_tiles: ran_wide.into_iter().filter(|&wide| wide).count(),
    }
}

/// The tile engine for the build: threaded with the `parallel` feature, else in order.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) fn run_tile_work<W>(
    work: &W,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, W::Row)>,
) -> BatchReport
where
    W: TileWork + Sync,
    W::Scratch: Send,
    W::Partial: Send,
    W::Row: Send,
{
    let executor = match crate::route::device() {
        crate::route::Device::Cpu => crate::par::Cpu { threads },
    };
    tiles_threaded(work, cut, &executor, results)
}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) fn run_tile_work<W: TileWork>(
    work: &W,
    cut: TileCut,
    _threads: NonZeroUsize,
    results: &mut Vec<(u64, W::Row)>,
) -> BatchReport {
    tiles_in_order(work, cut, results)
}

/// [`run_tile_work`] for the z-normalized metric: `body` fills a tile's partials, and every
/// tile.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) fn run_tiles(
    windows: &[Window],
    m: usize,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    body: &(impl Fn(&Tile, &mut Scratch, usize) -> RanWide + Sync),
) -> BatchReport {
    run_tiles_in(windows, m, cut, threads, results, body, false)
}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) fn run_tiles(
    windows: &[Window],
    m: usize,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    body: &impl Fn(&Tile, &mut Scratch, usize) -> RanWide,
) -> BatchReport {
    run_tiles_in(windows, m, cut, threads, results, body, false)
}

/// [`run_tiles`] with the threaded engine starting the top bands first, for a walk whose bands
/// hand lanes down: the result is the same on bits for any order, only how often a band finds the
/// lanes it wants differs.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) fn run_tiles_top_down(
    windows: &[Window],
    m: usize,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    body: &(impl Fn(&Tile, &mut Scratch, usize) -> RanWide + Sync),
) -> BatchReport {
    run_tiles_in(windows, m, cut, threads, results, body, true)
}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) fn run_tiles_top_down(
    windows: &[Window],
    m: usize,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    body: &impl Fn(&Tile, &mut Scratch, usize) -> RanWide,
) -> BatchReport {
    run_tiles_in(windows, m, cut, threads, results, body, true)
}

#[cfg(all(feature = "parallel", not(loom)))]
fn run_tiles_in(
    windows: &[Window],
    m: usize,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    body: &(impl Fn(&Tile, &mut Scratch, usize) -> RanWide + Sync),
    top_down: bool,
) -> BatchReport {
    let work = ZNormTiles {
        windows,
        m,
        k: cut.plan.k,
        body,
        top_down,
    };
    run_tile_work(&work, cut, threads, results)
}

#[cfg(not(all(feature = "parallel", not(loom))))]
fn run_tiles_in(
    windows: &[Window],
    m: usize,
    cut: TileCut,
    threads: NonZeroUsize,
    results: &mut Vec<(u64, Entry)>,
    body: &impl Fn(&Tile, &mut Scratch, usize) -> RanWide,
    top_down: bool,
) -> BatchReport {
    let work = ZNormTiles {
        windows,
        m,
        k: cut.plan.k,
        body,
        top_down,
    };
    run_tile_work(&work, cut, threads, results)
}

/// A value shared by the tiles of one run: behind a lock when tiles run on several threads,
/// else plain.
#[cfg(all(feature = "parallel", not(loom)))]
pub(super) struct Locked<T>(crate::sync::Mutex<T>);

#[cfg(all(feature = "parallel", not(loom)))]
impl<T> Locked<T> {
    pub(super) fn new(value: T) -> Self {
        Self(crate::sync::Mutex::new(value))
    }

    pub(super) fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut *lock(&self.0))
    }

    pub(super) fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(not(all(feature = "parallel", not(loom))))]
pub(super) struct Locked<T>(core::cell::RefCell<T>);

#[cfg(not(all(feature = "parallel", not(loom))))]
impl<T> Locked<T> {
    pub(super) fn new(value: T) -> Self {
        Self(core::cell::RefCell::new(value))
    }

    pub(super) fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }

    pub(super) fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

/// Columns whose right-neighbor accumulators share one lock.
pub(super) const RIGHT_GROUP: usize = 1024;

/// A per-column structure cut into locked groups of [`RIGHT_GROUP`] columns, which tiles merge
/// into as they finish and which flattens to one list in column order.
pub(super) struct ColumnGroups<T> {
    groups: Vec<Locked<T>>,
}

impl<T> ColumnGroups<T> {
    /// One group per [`RIGHT_GROUP`] of `columns`, each made by `make` from its column range.
    pub(super) fn try_new<E: From<CapacityError>>(
        columns: usize,
        mut make: impl FnMut(core::ops::Range<usize>) -> Result<T, E>,
    ) -> Result<Self, E> {
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(columns.div_ceil(RIGHT_GROUP))
            .map_err(|_| CapacityError)?;
        for start in (0..columns).step_by(RIGHT_GROUP) {
            groups.push(Locked::new(make(
                start..(start + RIGHT_GROUP).min(columns),
            )?));
        }
        Ok(Self { groups })
    }

    /// Runs `merge` under each group's lock with the part of `columns` that group holds.
    pub(super) fn merge(
        &self,
        columns: core::ops::Range<usize>,
        mut merge: impl FnMut(&mut T, core::ops::Range<usize>),
    ) {
        for g in columns.start / RIGHT_GROUP..columns.end.div_ceil(RIGHT_GROUP) {
            let shared =
                (g * RIGHT_GROUP).max(columns.start)..((g + 1) * RIGHT_GROUP).min(columns.end);
            self.groups[g].with(|group| merge(group, shared));
        }
    }

    /// Test-only: `f` on the group holding `column`.
    #[cfg(test)]
    pub(super) fn with_column<R>(&self, column: usize, f: impl FnOnce(&mut T) -> R) -> R {
        self.groups[column / RIGHT_GROUP].with(f)
    }

    /// Every group in column order, each appending its entries to `out`.
    pub(super) fn flatten<U>(self, out: &mut Vec<U>, mut finish: impl FnMut(T, &mut Vec<U>)) {
        for group in self.groups {
            finish(group.into_inner(), out);
        }
    }

    /// [`Self::flatten`] with `finish` on the group in place, and on up to `threads` threads when
    /// there is more than one group: each group finishes into a list of its own (one entry per
    /// column reserved), joined in column order. The list is the same for any `threads`.
    #[cfg(all(feature = "parallel", not(loom)))]
    pub(super) fn flatten_on<U: Send>(
        self,
        threads: NonZeroUsize,
        out: &mut Vec<U>,
        finish: impl Fn(&mut T, &mut Vec<U>) + Sync,
    ) where
        T: Send,
    {
        if threads.get() == 1 || self.groups.len() < 2 {
            return self.flatten(out, |mut group, out| finish(&mut group, out));
        }
        let parts = crate::par::for_each_chunk(threads, self.groups.len(), |g| {
            let mut part = Vec::with_capacity(RIGHT_GROUP);
            self.groups[g].with(|group| finish(group, &mut part));
            part
        });
        for part in parts {
            out.extend(part);
        }
    }

    #[cfg(not(all(feature = "parallel", not(loom))))]
    pub(super) fn flatten_on<U>(
        self,
        _threads: NonZeroUsize,
        out: &mut Vec<U>,
        finish: impl Fn(&mut T, &mut Vec<U>) + Sync,
    ) {
        self.flatten(out, |mut group, out| finish(&mut group, out));
    }
}

/// Test-only: tiles wait here until `need` of them are inside at once, which only `need` threads
/// can be. The clock only turns a missing thread into a failure instead of a hang.
#[cfg(all(test, feature = "parallel", not(loom)))]
struct Rendezvous {
    need: usize,
    inside: crate::sync::AtomicUsize,
}

#[cfg(all(test, feature = "parallel", not(loom)))]
impl Rendezvous {
    const HANG_GUARD: std::time::Duration = std::time::Duration::from_secs(60);

    fn new(need: usize) -> Self {
        Self {
            need,
            inside: crate::sync::AtomicUsize::new(0),
        }
    }

    fn arrive(&self) {
        if self.need == 0 {
            return;
        }
        self.inside.fetch_add(1, crate::sync::Ordering::SeqCst);
        // timing-ok: a hang-guard bound only; correct code never waits on the clock
        let start = std::time::Instant::now();
        while self.inside.load(crate::sync::Ordering::SeqCst) < self.need {
            assert!(
                // timing-ok: a hang-guard bound only; correct code never waits on the clock
                start.elapsed() < Self::HANG_GUARD,
                "fewer than {} tiles ran at once",
                self.need
            );
            // timing-ok: the loop is bounded by the hang guard; correct code never waits on the clock
            core::hint::spin_loop();
        }
    }
}

/// Test-only: how many bands' accumulators were alive at once in one engine run.
#[cfg(all(test, feature = "parallel", not(loom)))]
#[derive(Default)]
struct LiveBands {
    now: crate::sync::AtomicUsize,
    peak: crate::sync::AtomicUsize,
    fresh_groups: crate::sync::AtomicUsize,
}

/// One band's accumulators while its tiles run: a [`ROW_GROUP`] of rows per lock, and the number
/// of its tiles still to merge.
#[cfg(all(feature = "parallel", not(loom)))]
struct Band<P> {
    groups: Vec<Padded<crate::sync::Mutex<Vec<P>>>>,
    left: crate::sync::Mutex<usize>,
    #[cfg(test)]
    live: crate::sync::Arc<LiveBands>,
}

#[cfg(all(test, feature = "parallel", not(loom)))]
impl<P> Drop for Band<P> {
    fn drop(&mut self) {
        self.live.now.fetch_sub(1, crate::sync::Ordering::SeqCst);
    }
}

#[cfg(all(feature = "parallel", not(loom)))]
type LiveBand<P> = crate::sync::Arc<Band<P>>;

/// A finished band: its rows' entries in order (and, in tests, the neighbors each was offered).
#[cfg(all(feature = "parallel", not(loom)))]
struct BandDone<R> {
    entries: Vec<(u64, R)>,
    #[cfg(test)]
    visits: Vec<(usize, Vec<usize>)>,
}

#[cfg(all(feature = "parallel", not(loom)))]
impl<P> Band<P> {
    fn open<W: TileWork<Partial = P>>(
        work: &W,
        spare: &crate::sync::Mutex<Vec<Vec<Vec<P>>>>,
        rows: core::ops::Range<usize>,
        tiles: usize,
        #[cfg(test)] live: crate::sync::Arc<LiveBands>,
    ) -> Self {
        #[cfg(test)]
        {
            let now = live.now.fetch_add(1, crate::sync::Ordering::SeqCst) + 1;
            live.peak.fetch_max(now, crate::sync::Ordering::SeqCst);
        }
        let mut warm = lock(spare).pop().unwrap_or_default().into_iter();
        let groups = rows
            .clone()
            .step_by(ROW_GROUP)
            .map(|g| {
                let group = g..(g + ROW_GROUP).min(rows.end);
                let mut accs = warm.next().unwrap_or_else(|| {
                    #[cfg(test)]
                    live.fresh_groups
                        .fetch_add(1, crate::sync::Ordering::SeqCst);
                    Vec::new()
                });
                accs.resize_with(group.len(), || work.new_partial());
                for (acc, row) in accs.iter_mut().zip(group) {
                    work.begin(acc, row);
                }
                Padded(crate::sync::Mutex::new(accs))
            })
            .collect();
        Self {
            groups,
            left: crate::sync::Mutex::new(tiles),
            #[cfg(test)]
            live,
        }
    }

    /// Ranks every row of the band (all its tiles have merged), emptying the accumulators, which
    /// go to `spare` together for the next band to open with their buffers warm.
    pub(super) fn finish<W: TileWork<Partial = P>>(
        &self,
        work: &W,
        first: usize,
        spare: &crate::sync::Mutex<Vec<Vec<Vec<P>>>>,
    ) -> BandDone<W::Row> {
        let mut done = BandDone {
            entries: Vec::new(),
            #[cfg(test)]
            visits: Vec::new(),
        };
        let mut warm = Vec::with_capacity(self.groups.len());
        for (g, cell) in self.groups.iter().enumerate() {
            let mut accs = lock(&cell.0);
            for (r, acc) in accs.iter_mut().enumerate() {
                let row = first + g * ROW_GROUP + r;
                #[cfg(test)]
                done.visits.push((row, work.visits(acc)));
                work.finish(row, acc, &mut done.entries);
            }
            warm.push(core::mem::take(&mut *accs));
        }
        lock(spare).push(warm);
        done
    }
}
