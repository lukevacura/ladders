//! The row accumulator: one row's partial result over the lanes its passes rank.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::{
    ChunkCols, RankRow, RowFast, WideChunkCols, WideRank, WideRow, centered_cross_fast, row_fast,
};
use crate::batch::survivor::RowSurvey;
use crate::columns;
use crate::estimate::{Producer, band_floor};
use crate::index::SampleIndex;
use crate::kbest::{self, Blocks, Champion};
use crate::kernel_neon::RowScalars;
use crate::narrow;
use crate::nearest::Nearest;
use crate::stats::{RowClass, Window, WindowKind};

/// A neighbor that cleared the running floor: everything `Nearest::offer` needs.
pub(super) struct Cand {
    pub(super) index: usize,
    n: i64,
    neighbor: Window,
    pub(super) rho_hat: f64,
}

/// Candidates beyond this many are first pruned against the running floor; after a prune the
/// limit is twice what is left, up to twice `k`, so with a large `k` the prunes still cost a
/// fixed amount per candidate.
pub(super) const PRUNE_AT: usize = 64;

/// Candidates still beyond this many after the floor prune (exact ties), or beyond `k`, are
/// reduced to the exact nearest `k`.
pub(super) const RESOLVE_AT: usize = 2 * PRUNE_AT;

/// One row's partial result (the one narrow row pass): the `k` largest `ρ̂` seen so far and
/// every candidate at or above `band_floor` of the `k`-th of them. Reused across rows
/// (`begin` resets it; its buffers are kept), so no row allocates after warm-up.
pub struct RowAcc {
    window: Window,
    k: usize,
    /// The most neighbors the row's input can offer, which bounds what `k` reserves, held up to
    /// the reservation limit, past which it bounds nothing.
    bound: u32,
    /// The `k` largest finite `ρ̂`, as a heap whose root is the `k`-th largest once there are `k`.
    pub(super) top: Vec<f64>,
    /// The largest error bound of the producers that ranked this row's lanes.
    delta: f64,
    /// What the constant class settles for this row; its seed rises with the partials merged in.
    class: RowClass,
    pub(super) cands: Vec<Cand>,
    /// The run of exact ties the last candidates admitted form; for a constant row, the run of
    /// its constant neighbors.
    champion: Champion,
    /// For a constant row, the run of its valid neighbors.
    valid_run: Champion,
    /// The candidate count past which the next chunk prunes, saturating.
    prune_at: u32,
    kept: Vec<u64>,
    /// The reduction `resolve_ties` selects with, kept between rows so a repeated resolution
    /// allocates nothing.
    selector: Option<Box<Nearest>>,
    /// Test-only record: every neighbor index this row's pass was fed, in order.
    #[cfg(test)]
    pub(crate) visits: Vec<usize>,
}

impl RowAcc {
    pub fn new() -> Self {
        Self {
            window: Window {
                sum: 0,
                kind: WindowKind::NonFinite,
            },
            k: 1,
            bound: 1,
            top: Vec::new(),
            delta: 0.0,
            class: RowClass::NONE,
            cands: Vec::new(),
            champion: Champion::NONE,
            valid_run: Champion::NONE,
            prune_at: PRUNE_AT as u32,
            kept: Vec::new(),
            selector: None,
            #[cfg(test)]
            visits: Vec::new(),
        }
    }

    /// Starts a row whose own window is `row`, keeping the `k` nearest of at most `bound`
    /// neighbors, whose reservation is `min(k, bound)` up to a fixed limit.
    pub fn begin(&mut self, row: &Window, k: NonZeroUsize, bound: usize) {
        self.bound = bound.min(crate::capacity::RESERVE_LIMIT) as u32;
        self.window = *row;
        self.k = k.get();
        self.top.clear();
        self.delta = 0.0;
        self.class = RowClass::NONE;
        self.cands.clear();
        self.champion = Champion::NONE;
        self.valid_run = Champion::NONE;
        self.prune_at = PRUNE_AT as u32;
        #[cfg(test)]
        self.visits.clear();
        let wanted = crate::capacity::reserve_for(self.k, bound);
        if self.top.capacity() < wanted {
            self.top.reserve(wanted);
        }
    }

    /// After [`Self::begin`], reserves everything the row can grow when it is fed chunks of at
    /// most `lanes` lanes, so that a stream calling this on every row allocates nothing after its
    /// first, however late a row first ties many neighbors. Between chunks the candidate list
    /// holds at most twice `k` or the resolution limit, whichever is larger (past the limit it is
    /// resolved down to the `k` nearest, and a prune leaves the next limit at twice what is left,
    /// up to twice `k`) and a chunk adds at most
    /// `lanes`, all within the row's `bound` neighbors; the kept indices number at most `k`; the
    /// selecting reduction may rank the whole list at once. Batch rows (many and short-lived) and a
    /// growing stream (whose `bound` grows with every row, so that its buffers grow with it) do
    /// not call it.
    pub(crate) fn reserve_worst_case(&mut self, lanes: usize) {
        let between = self.k.saturating_mul(2).max(RESOLVE_AT);
        let held = between.saturating_add(lanes);
        let worst = crate::capacity::reserve_for(held, self.bound());
        self.cands.reserve(worst.saturating_sub(self.cands.len()));
        self.kept.reserve(
            crate::capacity::reserve_for(self.k, self.bound()).saturating_sub(self.kept.len()),
        );
        let k = NonZeroUsize::new(self.k).expect("k is nonzero");
        let (window, bound) = (self.window, self.bound());
        let selector = self
            .selector
            .get_or_insert_with(|| Box::new(Nearest::selecting(&window, k, bound)));
        selector.restart(&window, 0, k, bound);
        selector.reserve_selection(worst);
    }

    /// After [`Self::begin`], takes what the constant class settles for this row: the constant
    /// lanes it need not admit, the columns it need not offer, and the floor it starts from.
    pub(crate) fn take_class(&mut self, class: RowClass) {
        self.class = class;
        if class.seed() != f64::NEG_INFINITY {
            crate::route::count_class_row_seed();
        }
    }

    /// The candidate count past which the list is resolved to the exact nearest `k`.
    fn resolve_at(&self) -> usize {
        self.k.max(RESOLVE_AT)
    }

    fn note(&mut self, rho_hat: f64) {
        if rho_hat != f64::NEG_INFINITY {
            kbest::heap_keep(&mut self.top, self.k, rho_hat, |a, b| a > b);
        }
    }

    /// Whether a lane at `rho_hat` would enter the `k` largest: below `k` known, any finite one.
    fn enters(&self, rho_hat: f64) -> bool {
        if self.top.len() == self.k {
            rho_hat > self.top[0]
        } else {
            rho_hat != f64::NEG_INFINITY
        }
    }

    /// The near-tie floor, under the producers' bound, of the `k`-th largest `ρ̂` so far or the
    /// seed if that is higher; open until either is known. The seed is an exact correlation the
    /// row's `k`-th nearest reaches, so a lane that could tie it is at or above this floor.
    pub(super) fn floor(&self) -> f64 {
        let kth = if self.top.len() == self.k {
            self.top[0]
        } else {
            f64::NEG_INFINITY
        };
        band_floor(kth.max(self.class.seed()), self.delta)
    }

    /// Folds one producer's ranked span in: updates the top `k`, then keeps the lanes at or above
    /// the running floor. A block whose maximum cannot enter the top `k` is not read for it, and
    /// one whose maximum is below the floor is not rescanned. `first` is the absolute index of the
    /// span's first neighbor, `neighbor` reads a lane's whole window back, `exact_n` is a lane's
    /// exact centered cross term and `constant` says whether a lane's neighbor is constant. A
    /// constant lane past the constant class's first `k` is not admitted. A constant row takes its
    /// own admission instead (`absorb_constant_row`).
    pub(super) fn absorb_with<P: Producer>(
        &mut self,
        blocks: Blocks<P>,
        first: usize,
        neighbor: impl Fn(usize) -> Window,
        exact_n: impl Fn(usize) -> i64,
        constant: impl Fn(usize) -> bool,
    ) {
        self.delta = self.delta.max(P::DELTA);
        if matches!(self.row(), Some(RowFast::Constant)) && crate::route::constant_row_admission() {
            self.absorb_constant_row(&blocks, first, &neighbor, &exact_n);
            self.prune();
            return;
        }
        if self.k == 1 {
            self.note(blocks.maximum());
        } else {
            for (b, &max) in blocks.maxima().iter().enumerate() {
                if self.enters(max) {
                    for &rho_hat in blocks.block(b) {
                        self.note(rho_hat);
                    }
                }
            }
        }
        let floor = self.floor();
        let through = self.class.constants_through();
        blocks.at_or_above(floor, |lane, rho_hat| {
            let index = first + lane;
            if self.champion.outranks_index(index as u64, self.k) {
                return false;
            }
            if index > through && constant(lane) {
                crate::route::count_class_row_lane();
                return true;
            }
            self.admit_lane::<P>(index, rho_hat, || exact_n(lane), || neighbor(lane));
            true
        });
        self.prune();
    }

    /// Past the prune limit, drops the candidates below the running floor and resolves the
    /// exact ties left.
    fn prune(&mut self) {
        if self.cands.len() > self.prune_at as usize {
            let floor = self.floor();
            self.cands.retain(|c| c.rho_hat >= floor);
            self.resolve_ties();
            let held = self.cands.len().min(self.k);
            let prune_at = held.saturating_mul(2).max(PRUNE_AT);
            self.prune_at = u32::try_from(prune_at).unwrap_or(u32::MAX);
        }
    }

    /// A constant row's span: its correlation with a neighbor is fixed by the neighbor's kind
    /// alone, one for a constant neighbor and one half for a valid one, so its lanes tie in two
    /// runs and only the first `k` lanes of each can enter its nearest. Each run admits its
    /// lanes in ascending order until `k` of them from smaller indices beat the next, and the
    /// valid one admits none once `k` constant neighbors are held. Every lane it skips is beaten
    /// by `k` candidates it holds, so the row's `k` largest estimates are unchanged.
    fn absorb_constant_row<P: Producer>(
        &mut self,
        blocks: &Blocks<P>,
        first: usize,
        neighbor: &impl Fn(usize) -> Window,
        exact_n: &impl Fn(usize) -> i64,
    ) {
        self.champion = self.admit_run(blocks, 1.0, self.champion, first, neighbor, exact_n);
        if !self.champion.is_settled(self.k) {
            self.valid_run = self.admit_run(blocks, 0.5, self.valid_run, first, neighbor, exact_n);
        }
    }

    /// Admits the span's lanes ranked exactly `level`, ascending, extending `run` by each, until
    /// `run` holds `k` lanes from smaller indices that beat the next; returns the run.
    fn admit_run<P: Producer>(
        &mut self,
        blocks: &Blocks<P>,
        level: f64,
        mut run: Champion,
        first: usize,
        neighbor: &impl Fn(usize) -> Window,
        exact_n: &impl Fn(usize) -> i64,
    ) -> Champion {
        let k = self.k;
        let at = |index: usize| Champion::of(index as u64, 0, 0, level);
        if run.beats(&at(first), k) {
            return run;
        }
        blocks.at_or_above(level, |lane, rho_hat| {
            if rho_hat != level {
                return true;
            }
            let index = first + lane;
            let pair = at(index);
            if run.beats(&pair, k) {
                return false;
            }
            self.note(level);
            self.cands.push(Cand {
                index,
                n: exact_n(lane),
                neighbor: neighbor(lane),
                rho_hat: level,
            });
            run = run.then::<P>(pair, || 0, 0);
            true
        });
        run
    }

    /// Admits the lane at `index` as a candidate. A lane no better than the `k`-th best estimate is
    /// one a tie may have decided: unless `k` lanes of a run just before it tie it exactly, or
    /// have a correlation of exactly one, and have smaller indices (they win the tie), it is
    /// admitted and joins the run.
    #[inline(never)]
    fn admit_lane<P: Producer>(
        &mut self,
        index: usize,
        rho_hat: f64,
        n: impl FnOnce() -> i64,
        neighbor: impl FnOnce() -> Window,
    ) {
        if rho_hat == f64::NEG_INFINITY {
            return;
        }
        let (n, neighbor) = (n(), neighbor());
        if self.top.len() == self.k && rho_hat <= self.top[0] {
            let other = columns::variance(&neighbor);
            let pair = Champion::of(index as u64, n, other, rho_hat);
            if self.champion.beats(&pair, self.k) {
                return;
            }
            self.champion =
                self.champion
                    .then::<P>(pair, || columns::variance(&self.window), other);
        }
        self.cands.push(Cand {
            index,
            n,
            neighbor,
            rho_hat,
        });
    }

    /// Past [`Self::resolve_at`] candidates, keeps only the exact nearest `k`, all ranked at once
    /// and matched back by a sorted search, so a resolution costs a log factor per candidate
    /// whatever `k` is. Dropping the rest is safe: the exact order is total, so a candidate beaten
    /// by `k` kept ones is never among the row's `k` nearest, whatever arrives later.
    pub(super) fn resolve_ties(&mut self) {
        if self.cands.len() > self.resolve_at() {
            self.keep_exact_nearest();
        }
    }

    /// Leaves the candidates their exact nearest `k`.
    fn keep_exact_nearest(&mut self) {
        let k = NonZeroUsize::new(self.k).expect("k is nonzero");
        let mut nearest = self
            .selector
            .take()
            .unwrap_or_else(|| Box::new(Nearest::selecting(&self.window, k, self.bound())));
        nearest.restart(&self.window, 0, k, self.bound());
        nearest.select(self.cands.iter().map(|c| {
            let index = SampleIndex::new(c.index as u64).expect("index is below NONE's sentinel");
            (c.n, c.neighbor, index)
        }));
        self.kept.clear();
        self.kept.extend(nearest.kept_indices());
        self.kept.sort_unstable();
        self.selector = Some(nearest);
        let kept = &self.kept;
        self.cands
            .retain(|c| kept.binary_search(&(c.index as u64)).is_ok());
    }

    /// Combines another accumulator of the same row (same `k`): for tiling and threads.
    pub fn merge(&mut self, other: &mut RowAcc) {
        // Copied, not drained: merging the same partial twice offers its pairs twice, and the
        // once-per-pair record sees it.
        #[cfg(test)]
        self.visits.extend_from_slice(&other.visits);
        self.delta = self.delta.max(other.delta);
        self.class.merge_seed(&other.class);
        for &rho_hat in &other.top {
            self.note(rho_hat);
        }
        self.cands.append(&mut other.cands);
        // The merged floor only rises as more partials arrive, and `finish` keeps only
        // candidates at or above the final floor, so dropping below this one loses nothing;
        // pruning at every merge keeps a row's candidates few however many partials meet.
        let floor = self.floor();
        self.cands.retain(|c| c.rho_hat >= floor);
        self.resolve_ties();
    }

    /// Offers every candidate at or above the final floor to `reducer`, then empties `self`.
    pub fn finish(&mut self, reducer: &mut Nearest) {
        let floor = self.floor();
        self.cands.retain(|c| c.rho_hat >= floor);
        if self.k > crate::nearest::SELECT_ALL_ABOVE && self.cands.len() > self.k {
            self.keep_exact_nearest();
        }
        reducer.select(self.cands.drain(..).map(|c| {
            let index = SampleIndex::new(c.index as u64).expect("index is below NONE's sentinel");
            (c.n, c.neighbor, index)
        }));
        self.top.clear();
    }

    /// The row's one brand new diagonal (`qt0` already final, not advanced) as a one-lane
    /// chunk: `cols` describes just that neighbor.
    pub(crate) fn start_lane(
        &mut self,
        qt0: i64,
        cols: &ChunkCols,
        row: RowScalars,
    ) -> Option<RankedLane> {
        let n = narrow::cross(qt0, row.si, cols.sum[0]);
        self.start_lane_with::<RankRow>(n, cols.first, cols.r[0], cols.bias[0], &|| {
            (cols.neighbor)(0)
        })
    }

    /// [`Self::start_lane`] once the lane's exact centered cross term `n` is known, however the
    /// caller arrived at it: the neighbor is window `first` with columns `r` and `bias`.
    fn start_lane_with<P: Producer>(
        &mut self,
        n: i64,
        first: usize,
        r: f64,
        bias: f64,
        neighbor: &dyn Fn() -> Window,
    ) -> Option<RankedLane> {
        #[cfg(test)]
        self.visits.push(first);
        let row_kind = self.row()?;
        self.delta = self.delta.max(P::DELTA);
        let rho_hat = match row_kind {
            RowFast::Constant => columns::constant_row_rank(r, bias),
            RowFast::Valid { inv_sqrt } => columns::valid_row_rank(n, inv_sqrt, r, bias),
        };
        // The row's first lane: nothing is known yet, so the floor is open and it is a candidate.
        self.note(rho_hat);
        if rho_hat != f64::NEG_INFINITY {
            self.cands.push(Cand {
                index: first,
                n,
                neighbor: neighbor(),
                rho_hat,
            });
        }
        Some(RankedLane { rho_hat, n })
    }

    /// [`Self::start_lane`] on the exact i64 arithmetic: `qt0` is the unscaled dot product.
    pub(crate) fn start_lane_wide(
        &mut self,
        qt0: i64,
        cols: &WideChunkCols,
        row: WideRow,
        m: usize,
    ) -> Option<RankedLane> {
        let n = centered_cross_fast(qt0, m as i64, row.sum, cols.sum[0]);
        self.start_lane_with::<WideRank>(n, cols.first, cols.r[0], cols.bias[0], &|| {
            (cols.neighbor)(0)
        })
    }

    /// The row as the survivor filter judges a block of it: its kind, and for a valid row its
    /// current floor, read here and never copied.
    pub(crate) fn survey(&self) -> RowSurvey {
        match self.row() {
            None => RowSurvey::Silent,
            Some(RowFast::Constant) => RowSurvey::Constant,
            Some(RowFast::Valid { .. }) => RowSurvey::Valid {
                floor: self.floor(),
            },
        }
    }

    /// The row's own kind as the pass reads it, `None` for a non-finite row.
    pub(super) fn row(&self) -> Option<RowFast> {
        row_fast(&self.window)
    }

    /// This row as the columns it offers pairs to know it.
    pub(crate) fn id(&self, index: usize) -> RowId {
        let constant = matches!(self.window.kind, WindowKind::Constant);
        let settled = self.class.settled();
        RowId {
            index,
            variance: columns::variance(&self.window),
            offers_from: if constant { settled } else { 0 },
            finite_before: if constant { 0 } else { settled },
        }
    }

    /// The bound the row reserves under: past the reservation limit, the limit itself reserves
    /// the same.
    fn bound(&self) -> usize {
        self.bound as usize
    }

    /// Whether this row ranks its lanes at all: a non-finite row only advances.
    pub(crate) fn ranks(&self) -> bool {
        self.row().is_some()
    }
}

/// A started lane's ranking `ρ̂` and exact centered cross term, for a caller that reduces the
/// other side of the pair too.
#[derive(Clone, Copy)]
pub struct RankedLane {
    pub rho_hat: f64,
    pub n: i64,
}

/// A row offering pairs to the columns: its window index, and its variance numerator (`0` for a
/// constant window), which with the pair's cross term fixes the pair's exact correlation; and what
/// the constant class settles for its offers ([`RowClass::offers_from`] and
/// [`RowClass::finite_before`]), `0` for a row it settles nothing for.
#[derive(Clone, Copy)]
pub(crate) struct RowId {
    pub(crate) index: usize,
    pub(crate) variance: u64,
    pub(crate) offers_from: usize,
    pub(crate) finite_before: usize,
}

impl Default for RowAcc {
    fn default() -> Self {
        Self::new()
    }
}
