//! The one test-only override of the route decisions: how a stream or batch tile picks narrow or
//! i64 arithmetic and where its anchor sits. The setting is process-wide, so `forced_route` holds
//! a lock until its guard drops. Present only for tests and the `oracle` feature (and not under a
//! model checker); a release build keeps the stubs, which force nothing.

/// Where a batch run's tiles execute. The CPU's threads are the only device so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)] // no executor reads it in a build without the threaded engine
pub enum Device {
    #[default]
    Cpu,
}

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
impl Device {
    /// Every device, for tests that run each force on each of them.
    pub const ALL: [Self; 1] = [Self::Cpu];
}

#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
mod on {
    use super::Device;
    use crate::batch::survivor::Filter;
    use crate::narrow::Anchor;
    use crate::sync::{AtomicBool, AtomicU64, AtomicUsize, Mutex, MutexGuard, Ordering::Relaxed};
    use core::num::NonZeroUsize;

    static LOCK: Mutex<()> = Mutex::new(());
    static WIDE: AtomicBool = AtomicBool::new(false);
    static PER_TILE: AtomicBool = AtomicBool::new(false);
    static REANCHOR_EVERY: AtomicU64 = AtomicU64::new(0);
    static ANCHOR_SEED: AtomicU64 = AtomicU64::new(0);
    static REANCHOR_CELL: AtomicUsize = AtomicUsize::new(0);

    static WIDE_SWITCHES: AtomicUsize = AtomicUsize::new(0);
    static EXTRA_REANCHORS: AtomicUsize = AtomicUsize::new(0);
    static CELL_REANCHORS: AtomicUsize = AtomicUsize::new(0);
    static OFF_CENTER_ANCHORS: AtomicUsize = AtomicUsize::new(0);
    static PER_TILE_SERIES: AtomicUsize = AtomicUsize::new(0);
    static SPILL: AtomicUsize = AtomicUsize::new(0);
    static SPILL_FORCED_ON: AtomicUsize = AtomicUsize::new(0);
    static SPILL_FORCED_OFF: AtomicUsize = AtomicUsize::new(0);
    static SPARSE: AtomicUsize = AtomicUsize::new(0);
    static DEVICE: AtomicUsize = AtomicUsize::new(Device::Cpu as usize);
    static SPARSE_FORCED_ON: AtomicUsize = AtomicUsize::new(0);
    static SPARSE_FORCED_OFF: AtomicUsize = AtomicUsize::new(0);
    static CONSTANT_ROWS_BY_LANE: AtomicBool = AtomicBool::new(false);
    static CONSTANT_ROW_SPANS_BY_LANE: AtomicUsize = AtomicUsize::new(0);
    static CONVERSION: AtomicUsize = AtomicUsize::new(0);
    static FAST_CONVERSION_REFUSED: AtomicUsize = AtomicUsize::new(0);
    static CONVERSION_FORCED_FULL_RANGE: AtomicUsize = AtomicUsize::new(0);
    static RESUME_EVERY: AtomicUsize = AtomicUsize::new(0);
    static RESUMES: AtomicUsize = AtomicUsize::new(0);
    static SURVIVOR: AtomicUsize = AtomicUsize::new(0);
    static CATCH_UP: AtomicUsize = AtomicUsize::new(0);
    static FLOORS: AtomicUsize = AtomicUsize::new(0);
    static FLOOR_RAISES: AtomicUsize = AtomicUsize::new(0);
    static GATE_ON: AtomicUsize = AtomicUsize::new(0);
    static PAA_BUILD_FAILS: AtomicBool = AtomicBool::new(false);
    static GATE_FALLBACKS: AtomicUsize = AtomicUsize::new(0);
    static PAA_TABLES: AtomicUsize = AtomicUsize::new(0);
    static FLOOR_TABLES: AtomicUsize = AtomicUsize::new(0);
    static NO_CONSTANT_CLASS: AtomicBool = AtomicBool::new(false);
    static NO_GLOBAL_FLOOR: AtomicBool = AtomicBool::new(false);
    static ONE_LOOP: AtomicUsize = AtomicUsize::new(0);
    static ONE_LOOP_NO_HANDOFF: AtomicBool = AtomicBool::new(false);
    static LOG_PAIRS: AtomicBool = AtomicBool::new(false);
    /// The pairs the exact pass evaluated, as `(row, first column, end column)` spans, in the order
    /// the threads recorded them.
    static PAIR_LOG: Mutex<alloc::vec::Vec<(usize, usize, usize)>> =
        Mutex::new(alloc::vec::Vec::new());
    static LADDER_ENTRIES_COARSE: AtomicUsize = AtomicUsize::new(0);
    static LADDER_ENTRIES_MID: AtomicUsize = AtomicUsize::new(0);
    static LADDER_ENTRIES_FINE: AtomicUsize = AtomicUsize::new(0);
    static SURVIVOR_RUNS: AtomicUsize = AtomicUsize::new(0);
    static LANES_RESTARTED: AtomicUsize = AtomicUsize::new(0);
    static LANES_HANDED_DOWN: AtomicUsize = AtomicUsize::new(0);
    static LANES_MOVED: AtomicUsize = AtomicUsize::new(0);
    static WIDE_BANDS: AtomicUsize = AtomicUsize::new(0);
    static BAND_NANOS: AtomicUsize = AtomicUsize::new(0);
    static SKIPPED_AREA: AtomicUsize = AtomicUsize::new(0);
    static COMPUTED_AREA: AtomicUsize = AtomicUsize::new(0);
    static LANE_STARTS: AtomicUsize = AtomicUsize::new(0);
    static PAIRS_COMPUTED: AtomicUsize = AtomicUsize::new(0);
    static LANES_RESUMED: AtomicUsize = AtomicUsize::new(0);
    static LANES_STEPPED: AtomicUsize = AtomicUsize::new(0);
    static LANE_STEPS: AtomicUsize = AtomicUsize::new(0);
    static SEEDS: AtomicUsize = AtomicUsize::new(0);
    static SEED_PAIRS: AtomicUsize = AtomicUsize::new(0);
    static SEED_RAISES: AtomicUsize = AtomicUsize::new(0);
    static SEED_NANOS: AtomicUsize = AtomicUsize::new(0);
    static GLOBAL_RISES: AtomicUsize = AtomicUsize::new(0);
    static GLOBAL_SEEDS: AtomicUsize = AtomicUsize::new(0);
    static CONSTANT_CLASSES_DROPPED: AtomicUsize = AtomicUsize::new(0);
    static CLASS_ROW_LANES: AtomicUsize = AtomicUsize::new(0);
    static CLASS_OFFER_LANES: AtomicUsize = AtomicUsize::new(0);
    static CLASS_COLUMN_PAIRS: AtomicUsize = AtomicUsize::new(0);
    static CLASS_ROW_SEEDS: AtomicUsize = AtomicUsize::new(0);
    static CLASS_COLUMN_SEEDS: AtomicUsize = AtomicUsize::new(0);
    static NO_CONSTANT_BLOCKS: AtomicBool = AtomicBool::new(false);
    static CLASS_BLOCKS: AtomicUsize = AtomicUsize::new(0);
    static CLASS_BLOCK_PAIRS: AtomicUsize = AtomicUsize::new(0);
    static SCALAR_START_SUM: AtomicBool = AtomicBool::new(false);
    static AVX2_START_SUMS: AtomicUsize = AtomicUsize::new(0);

    /// The scope a force applies in: the forcing thread's, inherited by the threads the
    /// engines spawn for it ([`Scope`]), `0` when nothing is forced. A thread outside it
    /// reads no force and counts nothing, so a test that forces nothing runs the natural
    /// routes beside one that forces, in the same process.
    static ACTIVE: AtomicU64 = AtomicU64::new(0);
    static NEXT_SCOPE: AtomicU64 = AtomicU64::new(1);
    crate::sync::thread_local! {
        static SCOPE: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
    }

    /// Whether this thread is inside the active force's scope.
    fn in_scope() -> bool {
        let active = ACTIVE.load(Relaxed);
        active != 0 && SCOPE.with(core::cell::Cell::get) == active
    }

    /// `flag`'s forced value inside the scope, `false` outside it.
    fn load_bool(flag: &AtomicBool) -> bool {
        in_scope() && flag.load(Relaxed)
    }

    /// `value`'s forced value inside the scope, `0` (every force's default) outside it.
    fn load_usize(value: &AtomicUsize) -> usize {
        if in_scope() { value.load(Relaxed) } else { 0 }
    }

    /// [`load_usize`] for a `u64` force.
    fn load_u64(value: &AtomicU64) -> u64 {
        if in_scope() { value.load(Relaxed) } else { 0 }
    }

    /// Counts one event of `counter` inside the scope; outside it, nothing.
    fn count(counter: &AtomicUsize, events: usize) {
        if in_scope() {
            counter.fetch_add(events, Relaxed);
        }
    }

    /// The per-lane class counters, counted on the thread that meets them and added to the shared
    /// counters in one step ([`flush_class`]): a constant lane is met once per row and lane, so a
    /// shared `fetch_add` per event is contended across the threads of a run and distorts its
    /// timing. Indices: row lanes, offer lanes, column pairs, row seeds, column seeds.
    const CLASS_SLOTS: usize = 5;

    crate::sync::thread_local! {
        static PENDING_CLASS: core::cell::Cell<[usize; CLASS_SLOTS]> =
            const { core::cell::Cell::new([0; CLASS_SLOTS]) };
    }

    /// Counts `events` of class counter `slot` on this thread, inside the scope; outside it,
    /// nothing.
    fn count_class(slot: usize, events: usize) {
        if in_scope() {
            PENDING_CLASS.with(|pending| {
                let mut counts = pending.get();
                counts[slot] += events;
                pending.set(counts);
            });
        }
    }

    /// Takes this thread's pending class counts and, when the thread is still inside the force's
    /// scope, adds them to the shared counters; a thread past its scope drops them.
    fn flush_class() {
        let counts = PENDING_CLASS.with(|pending| pending.replace([0; CLASS_SLOTS]));
        if counts != [0; CLASS_SLOTS] && in_scope() {
            for (counter, events) in [
                &CLASS_ROW_LANES,
                &CLASS_OFFER_LANES,
                &CLASS_COLUMN_PAIRS,
                &CLASS_ROW_SEEDS,
                &CLASS_COLUMN_SEEDS,
            ]
            .into_iter()
            .zip(counts)
            {
                counter.fetch_add(events, Relaxed);
            }
        }
    }

    /// The force scope of the thread that takes it, for a thread it spawns to enter.
    #[derive(Clone, Copy)]
    pub(crate) struct Scope(u64);

    /// Restores the thread's own scope when dropped.
    pub(crate) struct Entered(u64);

    impl Scope {
        /// This thread's scope.
        #[cfg(feature = "parallel")]
        pub(crate) fn current() -> Self {
            Self(SCOPE.with(core::cell::Cell::get))
        }

        /// Puts this thread in the scope until the returned value drops.
        pub(crate) fn enter(self) -> Entered {
            Entered(SCOPE.with(|scope| scope.replace(self.0)))
        }
    }

    impl Drop for Entered {
        fn drop(&mut self) {
            flush_class();
            SCOPE.with(|scope| scope.set(self.0));
        }
    }

    /// How the exact spill route is decided.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum SpillForce {
        /// The cost model decides.
        #[default]
        Auto,
        /// Never take the windows the exact i64 pass ranks; a window with no distance
        /// without the cold route still gets one.
        Off,
        /// Take every window that holds an outlier, whatever the cost model says.
        On,
    }

    /// Whether the one-loop route, the only pruning path, runs.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum OneLoop {
        /// What production does: the route runs wherever its conditions hold (every self-join
        /// family and the AB-join), and the plain pass elsewhere, with the survivor filter
        /// chosen by the gate (the PAA bound from a window length of several hundred).
        #[default]
        Auto,
        /// The plain pass everywhere: no pruning, every pair computed. The reference of every
        /// output bit.
        Off,
        /// As [`Self::Auto`], and the route's PAA bound is installed at every window length (the
        /// gate would leave a short window the plain pass), so a small series runs the route.
        On,
    }

    /// How a short run of visited diagonals is walked.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum SparseForce {
        /// The run length decides.
        #[default]
        Auto,
        /// Every run goes through the tiled pass.
        Off,
        /// Every run is walked diagonal by diagonal.
        On,
    }

    /// How a row kernel converts its cross terms to `f64`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum ConversionForce {
        /// Fast exactly where the cross-term bound fits, the full-range conversion elsewhere.
        #[default]
        Auto,
        /// The fast conversion wherever the bound fits; refused, and counted, where it does
        /// not.
        Fast,
        /// Always the full-range conversion, even where the bound fits.
        FullRange,
    }

    /// Which survivor filter batch runs install.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum SurvivorForce {
        /// The production gate decides from the run's shape, as in a build without the
        /// harness.
        #[default]
        Gate,
        /// No filter: the plain pass.
        Off,
        /// The PAA filter, the one-loop route's bound, at every window length.
        Paa,
    }

    /// How a skipped run's diagonal states catch up when the pass next visits them.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum CatchUpForce {
        /// The step-or-resume rule: a run shorter than the break-even steps, a longer one
        /// resumes.
        #[default]
        Rule,
        /// Every run steps through, wherever the sides hold its ticks (resumed elsewhere).
        Step,
        /// Every run resumes.
        Resume,
    }

    /// Where the survivor filter's row and column floors come from in a tiled self-join.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum FloorForce {
        /// The run's one table of floors, shared by every tile and thread: raised by an
        /// atomic max from each range's partials and read before each range's ladder (the
        /// library's route whenever the filter is on).
        #[default]
        Shared,
        /// No table: the route's ladder reads no floor but the ones its seeds and partials gave
        /// the band, so it proves nothing a tile's own partials do not.
        PerThread,
    }

    /// The seed pre-pass of a tiled self-join with the filter on: exact pairs computed before
    /// the pass, only to raise the shared floors.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum SeedForce {
        /// What a normal run takes: each group of rows computes the pairs of its best 8 column
        /// parts' best sub-blocks.
        #[default]
        Auto,
        /// No pre-pass.
        Off,
    }

    /// Which routes to force; the default forces none.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct RouteForce {
        /// Every stream and batch tile runs the exact i64 arithmetic.
        pub wide: bool,
        /// Batch skips the whole-series anchor, so every tile takes its own.
        pub per_tile: bool,
        /// A stream re-anchors whenever its tick count is a multiple of this; `0` never
        /// forces.
        pub reanchor_every: u64,
        /// Anchors sit pseudo-randomly from this seed anywhere that still covers their
        /// ticks; `0` keeps them centered.
        pub anchor_seed: u64,
        /// Batch tiles take the re-anchoring path and move their anchors every this many rows,
        /// each lane group at most this many lanes wide; `0` leaves the natural cut.
        pub reanchor_cell: usize,
        /// Whether windows outside the narrow anchor take the exact spill route.
        pub spill: SpillForce,
        /// Whether runs of visited diagonals are walked one diagonal at a time.
        pub sparse: SparseForce,
        /// Where batch tiles execute.
        pub device: Device,
        /// A constant row admits its lanes one by one, as any other row does, instead of only
        /// the few its fixed correlations let into its nearest.
        pub constant_rows_by_lane: bool,
        /// How row kernels convert their cross terms.
        pub conversion: ConversionForce,
        /// At every row that is a multiple of this, each continuing diagonal's state is
        /// replaced by the exact resume after a skipped run, as a pruned diagonal resumes
        /// (the same state whatever run it skipped); `0` never forces.
        pub resume_every: usize,
        /// Which survivor filter batch runs install.
        pub survivor: SurvivorForce,
        /// How a skipped run's diagonal states catch up.
        pub catch_up: CatchUpForce,
        /// Where the survivor filter's floors come from.
        pub floors: FloorForce,
        /// Every build of a table of PAA coefficients fails, as when there is no room for it
        /// or the series has more windows than a `u32` counts.
        pub paa_build_fails: bool,
        /// Batch self-joins run without the constant class: no pair is dropped for the class's
        /// earlier members, and no floor starts from its bound. (An AB-join has no row rule or
        /// floor of the class, only its blocks: see `no_constant_blocks`; this turns them off too.)
        pub no_constant_class: bool,
        /// Batch self-joins and AB-joins visit every block the constant class settles whole, as a
        /// run without the class's block query does.
        pub no_constant_blocks: bool,
        /// The narrow start sums and resume products of `cross_term` take the scalar fold, not
        /// the AVX2 one.
        pub scalar_start_sum: bool,
        /// A run that reports only its best pair keeps no global floor: it prunes on its rows' and
        /// columns' floors alone, and every window's entry is exact.
        pub no_global_floor: bool,
        /// Which seed pre-pass runs before a tiled self-join's tiles.
        pub seeds: SeedForce,
        /// Whether the one-loop route runs.
        pub one_loop: OneLoop,
        /// The one-loop route's bands never take the lanes the band above left: every handoff
        /// misses and every band starts its lanes lazily.
        pub one_loop_no_handoff: bool,
        /// The exact pass records every span of pairs it evaluates, for the guard's `pair_log`: a test
        /// that every pair is evaluated at most once and none is missed.
        pub log_pairs: bool,
    }

    /// How often each forced route changed a decision since its `forced` call.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Fired {
        /// Switches (or tiles) to i64 arithmetic that the natural logic would not have made.
        pub wide_switches: usize,
        /// Stream re-anchors, and batch cell boundaries, that the natural logic would not
        /// have made.
        pub extra_reanchors: usize,
        /// Batch cell boundaries where a tile moved its anchors because its ticks drifted.
        pub cell_reanchors: usize,
        /// Anchors placed somewhere other than the center.
        pub off_center_anchors: usize,
        /// Batch runs whose whole-series anchor was dropped although one existed.
        pub per_tile_series: usize,
        /// Decisions that spilled although the cost model declined.
        pub spill_forced_on: usize,
        /// Decisions that did not spill although the cost model engaged.
        pub spill_forced_off: usize,
        /// Runs walked diagonal by diagonal that the run length would have tiled.
        pub sparse_forced_on: usize,
        /// Runs tiled that the run length would have walked.
        pub sparse_forced_off: usize,
        /// Spans of a constant row admitted lane by lane.
        pub constant_row_spans_by_lane: usize,
        /// Forced fast conversions refused because the bound did not fit.
        pub fast_conversions_refused: usize,
        /// Conversion decisions forced to the full-range conversion although the bound fit.
        pub conversions_forced_full_range: usize,
        /// Diagonal states replaced by a forced resume.
        pub resumes: usize,
        /// Shared floors raised: atomic maxes that found a lower floor stored.
        pub floor_raises: usize,
        /// Runs the production gate chose a filter for, whatever was forced (a run it leaves
        /// off counts nothing, as no route changed).
        pub gate_on: usize,
        /// Runs the production gate chose PAA for whose coefficients could not be built, so
        /// they installed no filter instead.
        pub gate_fallbacks: usize,
        /// Tables of PAA coefficients built.
        pub paa_tables: usize,
        /// Tables of shared floors allocated.
        pub floor_tables: usize,
        /// Constant classes a run would have built but was forced to go without.
        pub constant_classes_dropped: usize,
        /// Constant lanes a row did not admit, beaten by the class's first members.
        pub class_row_lanes: usize,
        /// Lanes a constant row did not offer its columns, beaten by earlier constant windows.
        pub class_offer_lanes: usize,
        /// Pairs a constant column did not admit, beaten by earlier finite windows.
        pub class_column_pairs: usize,
        /// Rows whose floor started from the class's bound.
        pub class_row_seeds: usize,
        /// Columns whose floor started from the class's bound.
        pub class_column_seeds: usize,
        /// Blocks (rows against a part of slots) left out because the class settles every pair
        /// in them.
        pub class_blocks: usize,
        /// Pairs in those blocks.
        pub class_block_pairs: usize,
        /// Times a run's global floor rose.
        pub global_rises: usize,
        /// Runs whose global floor started from a pair the constant class holds.
        pub global_seeds: usize,
        /// Exact pairs the seed pre-pass computed.
        pub seed_pairs: usize,
        /// Shared floors the seed pre-pass raised.
        pub seed_raises: usize,
        /// Nanoseconds the seed pre-pass took, summed over its threads.
        pub seed_nanos: usize,
    }

    /// What the passes computed and how their lanes started and caught up, counted in every run.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Work {
        /// Lane states a tile started from an exact dot product when it opened.
        pub lane_starts: usize,
        /// Pairs the kernel computed exactly: the self-join passes only, an AB-join's are not
        /// counted.
        pub pairs_computed: usize,
        /// Lanes resumed from an exact dot product, after a skip or as the first start of a lane
        /// a tile left unstarted.
        pub lanes_resumed: usize,
        /// Lanes stepped through a skipped run by the exact recurrence.
        pub lanes_stepped: usize,
        /// The steps those lanes took in all.
        pub lane_steps: usize,
        /// Entries the one-loop route's ladder evaluated, per layer (blocks of 256, 64 and 16
        /// slots): a block under a skipped parent is not evaluated.
        pub ladder_entries: [usize; 3],
        /// Runs of surviving blocks the one-loop route computed, one call of the leaf compute
        /// each.
        pub survivor_runs: usize,
        /// Lanes the one-loop route restarted from an exact dot product (the part of
        /// `lanes_resumed` that is not a handoff); `lanes_stepped` counts its steps too.
        pub lanes_restarted: usize,
        /// Lanes a band took from the band above, with no restart.
        pub lanes_handed_down: usize,
        /// Lanes a band took from the band above and moved to its own anchors, exactly: the
        /// part of `lanes_handed_down` whose anchors differ (a run with per-band anchors).
        pub lanes_moved: usize,
        /// Tiles the one-loop route's band driver ran on the exact i64 ticks (wide tiles).
        pub wide_bands: usize,
        /// Nanoseconds the one-loop route's bands took, summed over their threads.
        pub band_nanos: usize,
        /// Pairs of the triangle the one-loop route skipped on a proof.
        pub skipped_area: usize,
        /// Pairs of the triangle the one-loop route computed.
        pub computed_area: usize,
    }

    /// Holds the route setting; dropping it clears every force. The force applies on the
    /// thread that took it and the threads the engines spawn for it, nowhere else.
    pub struct Guard {
        _entered: Entered,
        _held: MutexGuard<'static, ()>,
    }

    impl Guard {
        /// Narrow start sums and resume products sent to the AVX2 body since the `forced` call:
        /// always 0 off x86_64. Not a [`Fired`] field, as a natural run counts it too.
        pub fn avx2_start_sums(&self) -> usize {
            AVX2_START_SUMS.load(Relaxed)
        }

        pub fn fired(&self) -> Fired {
            flush_class();
            Fired {
                wide_switches: WIDE_SWITCHES.load(Relaxed),
                extra_reanchors: EXTRA_REANCHORS.load(Relaxed),
                cell_reanchors: CELL_REANCHORS.load(Relaxed),
                off_center_anchors: OFF_CENTER_ANCHORS.load(Relaxed),
                per_tile_series: PER_TILE_SERIES.load(Relaxed),
                spill_forced_on: SPILL_FORCED_ON.load(Relaxed),
                spill_forced_off: SPILL_FORCED_OFF.load(Relaxed),
                sparse_forced_on: SPARSE_FORCED_ON.load(Relaxed),
                sparse_forced_off: SPARSE_FORCED_OFF.load(Relaxed),
                constant_row_spans_by_lane: CONSTANT_ROW_SPANS_BY_LANE.load(Relaxed),
                fast_conversions_refused: FAST_CONVERSION_REFUSED.load(Relaxed),
                conversions_forced_full_range: CONVERSION_FORCED_FULL_RANGE.load(Relaxed),
                resumes: RESUMES.load(Relaxed),
                floor_raises: FLOOR_RAISES.load(Relaxed),
                gate_on: GATE_ON.load(Relaxed),
                gate_fallbacks: GATE_FALLBACKS.load(Relaxed),
                paa_tables: PAA_TABLES.load(Relaxed),
                floor_tables: FLOOR_TABLES.load(Relaxed),
                constant_classes_dropped: CONSTANT_CLASSES_DROPPED.load(Relaxed),
                class_row_lanes: CLASS_ROW_LANES.load(Relaxed),
                class_offer_lanes: CLASS_OFFER_LANES.load(Relaxed),
                class_column_pairs: CLASS_COLUMN_PAIRS.load(Relaxed),
                class_row_seeds: CLASS_ROW_SEEDS.load(Relaxed),
                class_column_seeds: CLASS_COLUMN_SEEDS.load(Relaxed),
                class_blocks: CLASS_BLOCKS.load(Relaxed),
                class_block_pairs: CLASS_BLOCK_PAIRS.load(Relaxed),
                global_rises: GLOBAL_RISES.load(Relaxed),
                global_seeds: GLOBAL_SEEDS.load(Relaxed),
                seed_pairs: SEED_PAIRS.load(Relaxed),
                seed_raises: SEED_RAISES.load(Relaxed),
                seed_nanos: SEED_NANOS.load(Relaxed),
            }
        }

        /// The work the passes did since the `forced` call, counted whether or not a route is
        /// forced, so not fields of [`Fired`], which reads zero when nothing is forced.
        pub fn work(&self) -> Work {
            Work {
                lane_starts: LANE_STARTS.load(Relaxed),
                pairs_computed: PAIRS_COMPUTED.load(Relaxed),
                lanes_resumed: LANES_RESUMED.load(Relaxed),
                lanes_stepped: LANES_STEPPED.load(Relaxed),
                lane_steps: LANE_STEPS.load(Relaxed),
                ladder_entries: [
                    LADDER_ENTRIES_COARSE.load(Relaxed),
                    LADDER_ENTRIES_MID.load(Relaxed),
                    LADDER_ENTRIES_FINE.load(Relaxed),
                ],
                survivor_runs: SURVIVOR_RUNS.load(Relaxed),
                lanes_restarted: LANES_RESTARTED.load(Relaxed),
                lanes_handed_down: LANES_HANDED_DOWN.load(Relaxed),
                lanes_moved: LANES_MOVED.load(Relaxed),
                wide_bands: WIDE_BANDS.load(Relaxed),
                band_nanos: BAND_NANOS.load(Relaxed),
                skipped_area: SKIPPED_AREA.load(Relaxed),
                computed_area: COMPUTED_AREA.load(Relaxed),
            }
        }
    }

    impl Guard {
        /// The spans of pairs the exact pass evaluated since the `forced` call, as
        /// `(row, first column, end column)`, when `log_pairs` was forced; nothing otherwise. Only
        /// pairs of a self-join's tile passes are recorded, and the seed pre-pass's are not.
        pub fn pair_log(&self) -> alloc::vec::Vec<(usize, usize, usize)> {
            PAIR_LOG.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.store(0, Relaxed);
            apply(RouteForce::default());
        }
    }

    fn apply(force: RouteForce) {
        WIDE.store(force.wide, Relaxed);
        PER_TILE.store(force.per_tile, Relaxed);
        REANCHOR_EVERY.store(force.reanchor_every, Relaxed);
        ANCHOR_SEED.store(force.anchor_seed, Relaxed);
        REANCHOR_CELL.store(force.reanchor_cell, Relaxed);
        SPILL.store(force.spill as usize, Relaxed);
        SPARSE.store(force.sparse as usize, Relaxed);
        DEVICE.store(force.device as usize, Relaxed);
        CONSTANT_ROWS_BY_LANE.store(force.constant_rows_by_lane, Relaxed);
        CONVERSION.store(force.conversion as usize, Relaxed);
        RESUME_EVERY.store(force.resume_every, Relaxed);
        SURVIVOR.store(force.survivor as usize, Relaxed);
        CATCH_UP.store(force.catch_up as usize, Relaxed);
        FLOORS.store(force.floors as usize, Relaxed);
        PAA_BUILD_FAILS.store(force.paa_build_fails, Relaxed);
        NO_CONSTANT_CLASS.store(force.no_constant_class, Relaxed);
        NO_CONSTANT_BLOCKS.store(force.no_constant_blocks, Relaxed);
        SCALAR_START_SUM.store(force.scalar_start_sum, Relaxed);
        NO_GLOBAL_FLOOR.store(force.no_global_floor, Relaxed);
        SEEDS.store(force.seeds as usize, Relaxed);
        ONE_LOOP.store(force.one_loop as usize, Relaxed);
        ONE_LOOP_NO_HANDOFF.store(force.one_loop_no_handoff, Relaxed);
        LOG_PAIRS.store(force.log_pairs, Relaxed);
        PAIR_LOG.lock().unwrap_or_else(|e| e.into_inner()).clear();
        for counter in [
            &FLOOR_RAISES,
            &WIDE_SWITCHES,
            &EXTRA_REANCHORS,
            &CELL_REANCHORS,
            &OFF_CENTER_ANCHORS,
            &PER_TILE_SERIES,
            &SPILL_FORCED_ON,
            &SPILL_FORCED_OFF,
            &SPARSE_FORCED_ON,
            &SPARSE_FORCED_OFF,
            &CONSTANT_ROW_SPANS_BY_LANE,
            &FAST_CONVERSION_REFUSED,
            &CONVERSION_FORCED_FULL_RANGE,
            &RESUMES,
            &GATE_ON,
            &GATE_FALLBACKS,
            &PAA_TABLES,
            &FLOOR_TABLES,
            &CONSTANT_CLASSES_DROPPED,
            &CLASS_ROW_LANES,
            &CLASS_OFFER_LANES,
            &CLASS_COLUMN_PAIRS,
            &CLASS_ROW_SEEDS,
            &CLASS_COLUMN_SEEDS,
            &CLASS_BLOCKS,
            &CLASS_BLOCK_PAIRS,
            &GLOBAL_RISES,
            &GLOBAL_SEEDS,
            &SEED_PAIRS,
            &SEED_RAISES,
            &SEED_NANOS,
            &LANE_STARTS,
            &PAIRS_COMPUTED,
            &LANES_RESUMED,
            &LANES_STEPPED,
            &LANE_STEPS,
            &LADDER_ENTRIES_COARSE,
            &LADDER_ENTRIES_MID,
            &LADDER_ENTRIES_FINE,
            &SURVIVOR_RUNS,
            &LANES_RESTARTED,
            &LANES_HANDED_DOWN,
            &LANES_MOVED,
            &WIDE_BANDS,
            &BAND_NANOS,
            &SKIPPED_AREA,
            &COMPUTED_AREA,
            &AVX2_START_SUMS,
        ] {
            counter.store(0, Relaxed);
        }
    }

    /// Makes `force` the setting of this thread, and of the threads the engines spawn for
    /// it, until the guard drops; waits for any other guard first. Other threads read no
    /// force and count nothing meanwhile.
    pub fn forced(force: RouteForce) -> Guard {
        let held = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        apply(force);
        let scope = NEXT_SCOPE.fetch_add(1, Relaxed);
        let entered = Scope(scope).enter();
        ACTIVE.store(scope, Relaxed);
        Guard {
            _entered: entered,
            _held: held,
        }
    }

    /// `natural`, or true when wide arithmetic is forced.
    pub(crate) fn wide(natural: bool) -> bool {
        let forced = load_bool(&WIDE);
        if forced && !natural {
            count(&WIDE_SWITCHES, 1);
        }
        natural || forced
    }

    /// `natural`, unless the spill route is forced on or off.
    pub(crate) fn spill_engaged(natural: bool) -> bool {
        match load_usize(&SPILL) {
            force if force == SpillForce::Off as usize => {
                if natural {
                    count(&SPILL_FORCED_OFF, 1);
                }
                false
            }
            force if force == SpillForce::On as usize => {
                if !natural {
                    count(&SPILL_FORCED_ON, 1);
                }
                true
            }
            _ => natural,
        }
    }

    /// `natural`, unless walking every run diagonal by diagonal is forced on or off.
    pub(crate) fn sparse_walk(natural: bool) -> bool {
        match load_usize(&SPARSE) {
            force if force == SparseForce::Off as usize => {
                if natural {
                    count(&SPARSE_FORCED_OFF, 1);
                }
                false
            }
            force if force == SparseForce::On as usize => {
                if !natural {
                    count(&SPARSE_FORCED_ON, 1);
                }
                true
            }
            _ => natural,
        }
    }

    /// Whether a row kernel takes the fast conversion, given whether the cross-term bound
    /// `fits`: never where it does not, whatever is forced.
    pub(crate) fn fast_conversion(fits: bool) -> bool {
        match load_usize(&CONVERSION) {
            force if force == ConversionForce::Fast as usize => {
                if !fits {
                    count(&FAST_CONVERSION_REFUSED, 1);
                }
                fits
            }
            force if force == ConversionForce::FullRange as usize => {
                if fits {
                    count(&CONVERSION_FORCED_FULL_RANGE, 1);
                }
                false
            }
            _ => fits,
        }
    }

    /// Whether a resume is forced at row `row`.
    pub(crate) fn resume_due(row: usize) -> bool {
        let every = load_usize(&RESUME_EVERY);
        every != 0 && row.is_multiple_of(every)
    }

    /// Counts `states` diagonal states a forced resume replaced.
    pub(crate) fn count_resumes(states: usize) {
        count(&RESUMES, states);
    }

    /// Whether a skipped run steps through: `natural` (the step-or-resume rule's answer),
    /// unless a branch is forced.
    pub(crate) fn catch_up_steps(natural: bool) -> bool {
        match load_usize(&CATCH_UP) {
            f if f == CatchUpForce::Step as usize => true,
            f if f == CatchUpForce::Resume as usize => false,
            _ => natural,
        }
    }

    /// The forced survivor filter, `None` when the gate decides.
    ///
    /// A forced one-loop route reads the PAA bound, so it installs that filter wherever no other
    /// filter is forced (the gate's own choice, which turns PAA on only from a window length of
    /// several hundred, would leave the route nothing to ask).
    pub(crate) fn forced_survivor() -> Option<Filter> {
        match load_usize(&SURVIVOR) {
            f if f == SurvivorForce::Off as usize => Some(Filter::Off),
            f if f == SurvivorForce::Paa as usize => Some(Filter::Paa),
            _ if one_loop() == OneLoop::On => Some(Filter::Paa),
            _ => None,
        }
    }

    /// Whether a table of PAA coefficients is forced to fail its build.
    pub(crate) fn paa_build_fails() -> bool {
        load_bool(&PAA_BUILD_FAILS)
    }

    /// Counts one run whose gated PAA fell back to no filter.
    pub(crate) fn count_gate_fallback() {
        count(&GATE_FALLBACKS, 1);
    }

    /// Counts one run the production gate chose a filter for.
    pub(crate) fn count_gate_on() {
        count(&GATE_ON, 1);
    }

    /// Counts one table of PAA coefficients built.
    pub(crate) fn count_paa_table() {
        count(&PAA_TABLES, 1);
    }

    /// Counts one table of shared floors allocated.
    pub(crate) fn count_floor_table() {
        count(&FLOOR_TABLES, 1);
    }

    /// `natural`, or false when per-thread floors are forced.
    pub(crate) fn floors_shared(natural: bool) -> bool {
        natural && load_usize(&FLOORS) != FloorForce::PerThread as usize
    }

    /// Counts `raises` shared floors raised.
    pub(crate) fn count_floor_raises(raises: usize) {
        if raises > 0 {
            count(&FLOOR_RAISES, raises);
        }
    }

    /// Whether a constant row's span takes its own admission: false when lane-by-lane
    /// admission is forced.
    pub(crate) fn constant_row_admission() -> bool {
        let by_lane = load_bool(&CONSTANT_ROWS_BY_LANE);
        if by_lane {
            count(&CONSTANT_ROW_SPANS_BY_LANE, 1);
        }
        !by_lane
    }

    /// `natural`, or false when running without the constant class is forced.
    pub(crate) fn constant_class(natural: bool) -> bool {
        let off = load_bool(&NO_CONSTANT_CLASS);
        if off && natural {
            count(&CONSTANT_CLASSES_DROPPED, 1);
        }
        natural && !off
    }

    /// Whether a pass asks the constant class about whole blocks: false when visiting every
    /// block the class settles is forced.
    pub(crate) fn constant_blocks() -> bool {
        !load_bool(&NO_CONSTANT_BLOCKS)
    }

    /// Counts `blocks` blocks of `pairs` pairs in all left out because the class settles them all.
    pub(crate) fn count_class_blocks(blocks: usize, pairs: usize) {
        count(&CLASS_BLOCKS, blocks);
        count(&CLASS_BLOCK_PAIRS, pairs);
    }

    /// Whether a best-pair run keeps no global floor: true when that is forced.
    pub(crate) fn no_global_floor() -> bool {
        load_bool(&NO_GLOBAL_FLOOR)
    }

    /// The one-loop route a run takes: the forced one, production's where none is.
    pub(crate) fn one_loop() -> OneLoop {
        match load_usize(&ONE_LOOP) {
            s if s == OneLoop::On as usize => OneLoop::On,
            s if s == OneLoop::Off as usize => OneLoop::Off,
            _ => OneLoop::Auto,
        }
    }

    /// Whether every lane handoff between one-loop bands misses: true when forced.
    pub(crate) fn one_loop_no_handoff() -> bool {
        load_bool(&ONE_LOOP_NO_HANDOFF)
    }

    /// Counts what a one-loop run evaluated, computed and handed down since its counters were last
    /// taken.
    pub(crate) fn count_one_loop(counts: &crate::batch::one_loop::Counts) {
        count(&LADDER_ENTRIES_COARSE, counts.entries[0]);
        count(&LADDER_ENTRIES_MID, counts.entries[1]);
        count(&LADDER_ENTRIES_FINE, counts.entries[2]);
        count(&SURVIVOR_RUNS, counts.survivor_runs);
        count(&LANES_STEPPED, counts.lanes_stepped);
        count(&LANES_RESTARTED, counts.lanes_restarted);
        count(&LANES_HANDED_DOWN, counts.lanes_handed_down);
        count(&LANES_MOVED, counts.lanes_moved);
        count(&WIDE_BANDS, counts.wide_bands);
        count(&BAND_NANOS, counts.band_nanos);
        count(&SKIPPED_AREA, counts.skipped_area);
        count(&COMPUTED_AREA, counts.computed_area);
        count(&PAIRS_COMPUTED, counts.computed_area);
        count(&LANES_RESUMED, counts.lanes_restarted);
        count(&LANE_STEPS, counts.lane_steps);
    }

    /// Records that the exact pass evaluated the pairs of row `row` with the columns `columns`,
    /// when the pair log is forced on.
    pub(crate) fn record_pairs(row: usize, columns: core::ops::Range<usize>) {
        if load_bool(&LOG_PAIRS) {
            PAIR_LOG.lock().unwrap_or_else(|e| e.into_inner()).push((
                row,
                columns.start,
                columns.end,
            ));
        }
    }

    /// The period of forced resumes in rows, `0` when none is forced.
    pub(crate) fn resume_period() -> usize {
        load_usize(&RESUME_EVERY)
    }

    /// The seed pre-pass a run takes: the forced one, the best 8 parts where none is.
    pub(crate) fn seeds() -> SeedForce {
        match load_usize(&SEEDS) {
            s if s == SeedForce::Off as usize => SeedForce::Off,
            _ => SeedForce::Auto,
        }
    }

    /// Counts the lanes a skipped run's catch-up resumed or stepped, and the steps it took.
    pub(crate) fn count_catch_up(caught: &crate::cross_term::CatchUp) {
        count(&LANES_STEPPED, caught.stepped);
        count(&LANES_RESUMED, caught.resumed);
        count(&LANE_STEPS, caught.steps);
    }

    /// Counts the `pairs` the plain pass computed exactly.
    pub(crate) fn count_pairs(pairs: usize) {
        count(&PAIRS_COMPUTED, pairs);
    }

    /// Counts the `lanes` a tile started from an exact dot product when it opened.
    pub(crate) fn count_lane_starts(lanes: usize) {
        count(&LANE_STARTS, lanes);
    }

    /// Counts a seed pre-pass's exact `pairs`, the floors it raised and the `nanos` it took.
    pub(crate) fn count_seeds(pairs: usize, raises: usize, nanos: usize) {
        count(&SEED_PAIRS, pairs);
        count(&SEED_RAISES, raises);
        count(&SEED_NANOS, nanos);
    }

    /// Counts one rise of a global floor.
    pub(crate) fn count_global_rise() {
        count(&GLOBAL_RISES, 1);
    }

    /// Counts one global floor that started from a constant pair.
    pub(crate) fn count_global_seed() {
        count(&GLOBAL_SEEDS, 1);
    }

    /// Whether the narrow start sums take their AVX2 body: `detected`, unless the scalar sum is
    /// forced. Counts each sum sent to the AVX2 body.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn avx2_start_sum(detected: bool) -> bool {
        let taken = detected && !load_bool(&SCALAR_START_SUM);
        if taken {
            count(&AVX2_START_SUMS, 1);
        }
        taken
    }

    /// Counts one constant lane a row did not admit.
    pub(crate) fn count_class_row_lane() {
        count_class(0, 1);
    }

    /// Counts `lanes` lanes a constant row did not offer.
    pub(crate) fn count_class_offer_lanes(lanes: usize) {
        if lanes > 0 {
            count_class(1, lanes);
        }
    }

    /// Counts one pair a constant column did not admit.
    pub(crate) fn count_class_column_pair() {
        count_class(2, 1);
    }

    /// Counts one row whose floor started from the class's bound.
    pub(crate) fn count_class_row_seed() {
        count_class(3, 1);
    }

    /// Counts one column whose floor started from the class's bound.
    pub(crate) fn count_class_column_seed() {
        count_class(4, 1);
    }

    /// The device batch tiles execute on.
    #[cfg(all(feature = "parallel", not(loom)))]
    pub(crate) fn device() -> Device {
        match load_usize(&DEVICE) {
            d if d == Device::Cpu as usize => Device::Cpu,
            d => unreachable!("device {d} is not a device"),
        }
    }

    /// Whether wide arithmetic is forced, without counting it as a decision.
    pub(crate) fn forced_wide() -> bool {
        load_bool(&WIDE)
    }

    /// The forced cell size of batch tiles, `0` when none is forced.
    pub(crate) fn forced_cell() -> usize {
        load_usize(&REANCHOR_CELL)
    }

    /// Counts one cell boundary a batch tile re-anchored at: forced when the forced cell size
    /// ended the segment, natural when the ticks' drift did.
    pub(crate) fn count_cell_boundary(forced: bool) {
        let counter = if forced {
            &EXTRA_REANCHORS
        } else {
            &CELL_REANCHORS
        };
        count(counter, 1);
    }

    /// `needed`, or true when a stream that has seen `ticks` ticks must re-anchor now.
    pub(crate) fn reanchor_due(ticks: u64, needed: bool) -> bool {
        let every = load_u64(&REANCHOR_EVERY);
        let forced = every != 0 && ticks.is_multiple_of(every);
        if forced && !needed {
            count(&EXTRA_REANCHORS, 1);
        }
        needed || forced
    }

    /// Whether a batch run keeps the whole-series anchor that exists; false when per-tile
    /// anchors or a cell size are forced.
    pub(crate) fn keep_whole_series_anchor() -> bool {
        let keep = !load_bool(&PER_TILE);
        if !keep {
            count(&PER_TILE_SERIES, 1);
        }
        keep && load_usize(&REANCHOR_CELL) == 0
    }

    fn mix(seed: u64, parts: [u64; 3]) -> u64 {
        let mut x = seed;
        for part in parts {
            x = (x ^ part).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 32;
        }
        x
    }

    /// `centered`, or an anchor anywhere (edges included) that still covers `lo..=hi` at `m`.
    pub(crate) fn place(centered: Anchor, lo: i64, hi: i64, m: NonZeroUsize, salt: u64) -> Anchor {
        let seed = load_u64(&ANCHOR_SEED);
        if seed == 0 {
            return centered;
        }
        let bound = i64::from(i32::MAX) / m.get() as i64;
        let (first, last) = (hi.saturating_sub(bound), lo.saturating_add(bound));
        let width = (last - first) as u64 + 1;
        let r = mix(seed, [lo as u64, hi as u64, salt]);
        let placed = Anchor {
            tick: match r & 3 {
                0 => first,
                1 => last,
                _ => first + ((r >> 2) % width) as i64,
            },
        };
        if placed != centered {
            count(&OFF_CENTER_ANCHORS, 1);
        }
        placed
    }
}

#[cfg(not(all(any(test, feature = "oracle"), not(any(loom, shuttle)))))]
mod on {
    #[cfg(all(feature = "parallel", not(loom)))]
    use super::Device;
    use crate::batch::survivor::Filter;
    use crate::narrow::Anchor;
    use core::num::NonZeroUsize;

    /// Which runs take the one-loop route: outside tests and the oracle, every run it supports.
    #[allow(dead_code)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum OneLoop {
        Auto,
        Off,
        On,
    }

    /// The seed pre-pass a run takes: always the best 8 parts outside tests and the oracle.
    #[allow(dead_code)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum SeedForce {
        Auto,
        Off,
    }

    #[cfg(all(feature = "parallel", not(loom)))]
    pub(crate) fn device() -> Device {
        Device::Cpu
    }
    pub(crate) fn wide(natural: bool) -> bool {
        natural
    }
    pub(crate) fn forced_wide() -> bool {
        false
    }
    pub(crate) fn forced_cell() -> usize {
        0
    }
    pub(crate) fn count_cell_boundary(_forced: bool) {}
    pub(crate) fn spill_engaged(natural: bool) -> bool {
        natural
    }
    pub(crate) fn sparse_walk(natural: bool) -> bool {
        natural
    }
    pub(crate) fn fast_conversion(fits: bool) -> bool {
        fits
    }
    #[inline(always)]
    pub(crate) fn resume_due(_row: usize) -> bool {
        false
    }
    pub(crate) fn count_resumes(_states: usize) {}
    #[inline(always)]
    pub(crate) fn catch_up_steps(natural: bool) -> bool {
        natural
    }
    /// No force scope outside tests and the oracle: a spawned thread enters nothing.
    #[cfg(feature = "parallel")]
    #[derive(Clone, Copy)]
    pub(crate) struct Scope;
    #[cfg(feature = "parallel")]
    impl Scope {
        #[inline(always)]
        pub(crate) fn current() -> Self {
            Self
        }
        #[inline(always)]
        pub(crate) fn enter(self) -> Entered {
            Entered
        }
    }
    /// Nothing to restore.
    #[cfg(feature = "parallel")]
    pub(crate) struct Entered;
    #[inline(always)]
    pub(crate) fn forced_survivor() -> Option<Filter> {
        None
    }
    #[inline(always)]
    pub(crate) fn paa_build_fails() -> bool {
        false
    }
    #[inline(always)]
    pub(crate) fn count_gate_fallback() {}
    #[inline(always)]
    pub(crate) fn count_gate_on() {}
    #[inline(always)]
    pub(crate) fn count_paa_table() {}
    #[inline(always)]
    pub(crate) fn count_floor_table() {}
    #[inline(always)]
    pub(crate) fn floors_shared(natural: bool) -> bool {
        natural
    }
    #[inline(always)]
    pub(crate) fn count_floor_raises(_raises: usize) {}
    #[inline(always)]
    pub(crate) fn constant_row_admission() -> bool {
        true
    }
    #[inline(always)]
    pub(crate) fn no_global_floor() -> bool {
        false
    }
    #[inline(always)]
    pub(crate) fn one_loop() -> OneLoop {
        OneLoop::Auto
    }
    #[inline(always)]
    pub(crate) fn one_loop_no_handoff() -> bool {
        false
    }
    #[inline(always)]
    pub(crate) fn count_one_loop(_counts: &crate::batch::one_loop::Counts) {}
    #[inline(always)]
    pub(crate) fn record_pairs(_row: usize, _columns: core::ops::Range<usize>) {}
    #[inline(always)]
    pub(crate) fn resume_period() -> usize {
        0
    }
    #[inline(always)]
    pub(crate) fn seeds() -> SeedForce {
        SeedForce::Auto
    }
    #[inline(always)]
    pub(crate) fn count_pairs(_pairs: usize) {}
    #[inline(always)]
    pub(crate) fn count_catch_up(_caught: &crate::cross_term::CatchUp) {}
    #[inline(always)]
    pub(crate) fn count_lane_starts(_lanes: usize) {}
    #[inline(always)]
    pub(crate) fn count_seeds(_pairs: usize, _raises: usize, _nanos: usize) {}
    #[inline(always)]
    pub(crate) fn count_global_rise() {}
    #[inline(always)]
    pub(crate) fn count_global_seed() {}
    #[inline(always)]
    pub(crate) fn constant_class(natural: bool) -> bool {
        natural
    }
    #[inline(always)]
    pub(crate) fn constant_blocks() -> bool {
        true
    }
    #[inline(always)]
    pub(crate) fn count_class_blocks(_blocks: usize, _pairs: usize) {}
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(crate) fn avx2_start_sum(detected: bool) -> bool {
        detected
    }
    #[inline(always)]
    pub(crate) fn count_class_row_lane() {}
    #[inline(always)]
    pub(crate) fn count_class_offer_lanes(_lanes: usize) {}
    #[inline(always)]
    pub(crate) fn count_class_column_pair() {}
    #[inline(always)]
    pub(crate) fn count_class_row_seed() {}
    #[inline(always)]
    pub(crate) fn count_class_column_seed() {}
    pub(crate) fn reanchor_due(_ticks: u64, needed: bool) -> bool {
        needed
    }
    pub(crate) fn keep_whole_series_anchor() -> bool {
        true
    }
    pub(crate) fn place(
        centered: Anchor,
        _lo: i64,
        _hi: i64,
        _m: NonZeroUsize,
        _salt: u64,
    ) -> Anchor {
        centered
    }
}

#[cfg(feature = "parallel")]
pub(crate) use on::Scope;
#[cfg(target_arch = "x86_64")]
pub(crate) use on::avx2_start_sum;
#[cfg(all(feature = "parallel", not(loom)))]
pub(crate) use on::device;
#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
pub use on::{
    CatchUpForce, ConversionForce, Fired, FloorForce, OneLoop, RouteForce, SeedForce, SparseForce,
    SpillForce, SurvivorForce, Work, forced as forced_route,
};
#[cfg(not(all(any(test, feature = "oracle"), not(any(loom, shuttle)))))]
pub(crate) use on::{OneLoop, SeedForce};
pub(crate) use on::{
    catch_up_steps, constant_blocks, constant_class, constant_row_admission, count_catch_up,
    count_cell_boundary, count_class_blocks, count_class_column_pair, count_class_column_seed,
    count_class_offer_lanes, count_class_row_lane, count_class_row_seed, count_floor_raises,
    count_floor_table, count_gate_fallback, count_gate_on, count_global_rise, count_global_seed,
    count_lane_starts, count_paa_table, count_pairs, count_resumes, count_seeds, fast_conversion,
    floors_shared, forced_cell, forced_survivor, forced_wide, keep_whole_series_anchor,
    no_global_floor, one_loop, paa_build_fails, place, reanchor_due, record_pairs, resume_due,
    resume_period, seeds, sparse_walk, spill_engaged, wide,
};
pub(crate) use on::{count_one_loop, one_loop_no_handoff};
