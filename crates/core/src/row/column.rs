//! The column accumulator: each column's partial result over the rows that offer it pairs.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::{RankedLane, RowId};
use crate::entry::Entry;
use crate::estimate::{Producer, band_floor};
use crate::index::SampleIndex;
use crate::kbest::{self, Champion};
use crate::kernel_neon::{GROUP_LANES, SPAN_LANES};
use crate::nearest::Nearest;
use crate::stats::{ConstantClass, Window, WindowKind};

/// The run of exact ties among the pairs a column admitted without them bettering its best, in the
/// tile `epoch`, and the floor the column had when the run settled it, if it did.
#[derive(Clone, Copy)]
struct Held {
    epoch: u64,
    champion: Champion,
    /// `NaN` until the column settles.
    settled_floor: f64,
}

impl Held {
    const EMPTY: Self = Self {
        epoch: 0,
        champion: Champion::NONE,
        settled_floor: f64::NAN,
    };

    /// A new tile's first pair for the column: nothing held yet.
    fn fresh(epoch: u64) -> Self {
        Self {
            epoch,
            ..Self::EMPTY
        }
    }
}

/// The floor candidates of column `slot` are judged by: the one a settled column had when it
/// settled (its `floor` entry reads `∞` meanwhile), else its `floor` entry.
fn floor_at(floor: &[f64], champions: &[Held], epoch: u64, slot: usize) -> f64 {
    match champions.get(slot) {
        Some(held) if held.epoch == epoch && !held.settled_floor.is_nan() => held.settled_floor,
        _ => floor[slot],
    }
}

/// A pair that cleared its column's running floor: the row it came from and the exact `n`.
#[derive(Clone, Copy)]
pub(super) struct ColumnCand {
    pub(super) column: usize,
    pub(super) row: usize,
    pub(super) n: i64,
    pub(super) rho_hat: f64,
}

/// Candidates beyond this many are pruned against the columns' floors as they arrive.
pub(super) const COLUMN_PRUNE_AT: usize = 64;

/// Candidates per column beyond which (after the floor prune, so only under exact ties) the list
/// is reduced to each column's exact nearest.
const COLUMN_RESOLVE_PER_COLUMN: usize = 4;

/// The columns' side of the pass: for each column of a range, the `k` largest `ρ̂` any rows have
/// offered and every pair at or above `band_floor` of the `k`-th. Where [`RowAcc`](super::RowAcc) reduces one row
/// over its lanes, this reduces each lane over the rows that pass through it, so a pair seen once
/// from its later window is offered to the earlier window's right neighbor too. Reused across
/// tiles (`begin` resets it; its buffers are kept).
pub struct ColumnAcc {
    base: usize,
    k: NonZeroUsize,
    /// Each column's largest `ρ̂` as a heap whose root is its `k`-th largest, `-∞` where fewer have
    /// arrived: column `c` owns `offset[c]..offset[c + 1]`, `min(k, rows that can reach it)` slots,
    /// so a column nothing can reach owns none.
    pub(super) top: Vec<f64>,
    pub(super) offset: Vec<usize>,
    /// `band_floor` of the column's `k`-th largest, open (`-∞`) until `k` have arrived, or
    /// `∞` for a non-finite column that takes no candidate.
    pub(super) floor: Vec<f64>,
    /// The largest error bound of the producers that ranked the pairs offered so far.
    delta: f64,
    /// Per column, the [`RowId::finite_before`] from which a row is beaten by earlier finite
    /// windows ([`ConstantClass::column_cuts`]), `usize::MAX` for a column the class settles no
    /// row for; empty without the class.
    cuts: Vec<usize>,
    pub(super) cands: Vec<ColumnCand>,
    /// Scratch for [`Self::sort`]: where each column's candidates start, and how far they are placed.
    starts: Vec<usize>,
    placed: Vec<usize>,
    /// Each column's run of exact ties among the pairs last admitted; a slot whose `epoch` is not
    /// this tile's holds nothing.
    champions: Vec<Held>,
    epoch: u64,
    kept: Vec<u64>,
    prune_at: usize,
    /// Test-only record: the most candidates held at once since `begin`.
    #[cfg(test)]
    pub(crate) peak_cands: usize,
}

impl ColumnAcc {
    pub fn new() -> Self {
        Self {
            base: 0,
            k: NonZeroUsize::MIN,
            top: Vec::new(),
            offset: Vec::new(),
            floor: Vec::new(),
            delta: 0.0,
            cuts: Vec::new(),
            cands: Vec::new(),
            starts: Vec::new(),
            placed: Vec::new(),
            champions: Vec::new(),
            epoch: 0,
            kept: Vec::new(),
            prune_at: COLUMN_PRUNE_AT,
            #[cfg(test)]
            peak_cands: 0,
        }
    }

    /// Starts the columns `columns` of `windows`, keeping the `k` nearest rows of each, with
    /// nothing offered yet. `reach(column)` is the most rows that can ever offer to that column
    /// (a non-finite column takes none): its slots are sized by `min(k, reach)`, in full, so
    /// nothing grows afterwards and a huge `k` reserves no more than the input can fill.
    pub(crate) fn begin(
        &mut self,
        columns: core::ops::Range<usize>,
        windows: &[Window],
        k: NonZeroUsize,
        reach: impl Fn(usize) -> usize,
    ) {
        self.try_begin(columns, windows, k, reach)
            .unwrap_or_else(|e| crate::capacity::abort_on_capacity(e));
    }

    /// [`Self::begin`], reserving the slots fallibly: a capacity error when they cannot be
    /// allocated.
    pub(crate) fn try_begin(
        &mut self,
        columns: core::ops::Range<usize>,
        windows: &[Window],
        k: NonZeroUsize,
        reach: impl Fn(usize) -> usize,
    ) -> Result<(), crate::capacity::CapacityError> {
        self.base = columns.start;
        self.k = k;
        self.offset.clear();
        self.offset.push(0);
        let mut slots = 0;
        for (column, window) in columns.clone().zip(&windows[columns.clone()]) {
            if !matches!(window.kind, WindowKind::NonFinite) {
                slots += k.get().min(reach(column));
            }
            self.offset.push(slots);
        }
        self.top.clear();
        self.top
            .try_reserve_exact(slots)
            .map_err(|_| crate::capacity::CapacityError)?;
        self.top.resize(slots, f64::NEG_INFINITY);
        self.floor.clear();
        self.floor
            .extend(windows[columns].iter().map(|w| match w.kind {
                WindowKind::NonFinite => f64::INFINITY,
                _ => f64::NEG_INFINITY,
            }));
        self.epoch += 1;
        self.delta = 0.0;
        self.cuts.clear();
        self.cands.clear();
        self.prune_at = COLUMN_PRUNE_AT;
        #[cfg(test)]
        {
            self.peak_cands = 0;
        }
        Ok(())
    }

    /// Offers row `row`'s ranked chunk (`tile[lane]` = `ρ̂` against column `first + lane`) to the
    /// columns; `n_of(lane)` is the lane's exact centered cross term and `variance_of(lane)` its
    /// column's variance, each read only for a candidate.
    /// The lanes before the row's [`RowId::offers_from`] are not offered.
    #[inline]
    pub(crate) fn offer_chunk<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        n_of: impl Fn(usize) -> i64,
        variance_of: impl Fn(usize) -> u64,
    ) {
        let skip = row.offers_from.saturating_sub(first).min(tile.len());
        crate::route::count_class_offer_lanes(skip);
        for (span, lanes) in tile[skip..].chunks(SPAN_LANES).enumerate() {
            let lo = skip + span * SPAN_LANES;
            let floors = self.floors(first + lo, lanes.len());
            let hits = lanes
                .chunks(GROUP_LANES)
                .zip(floors.chunks(GROUP_LANES))
                .enumerate()
                .fold(0u64, |hits, (g, (group, floors))| {
                    let hit = group
                        .iter()
                        .zip(floors)
                        .fold(false, |hit, (&rho_hat, &floor)| hit | (rho_hat >= floor));
                    hits | u64::from(hit) << g
                });
            self.offer_groups::<P>(
                first + lo,
                row,
                lanes,
                hits,
                |lane| n_of(lo + lane),
                |lane| variance_of(lo + lane),
            );
        }
    }

    /// The floors of columns `first..first + len`, one per lane of a chunk starting there: a
    /// column whose nearest row is settled reads `∞`, so its later pairs are not read at all.
    pub(crate) fn floors(&self, first: usize, len: usize) -> &[f64] {
        &self.floor[first - self.base..][..len]
    }

    /// [`Self::offer_chunk`] on a span of at most [`SPAN_LANES`] lanes whose groups at or above a
    /// floor are already known: bit `g` of `hits` is set when some lane of group `g` is (as
    /// [`kernel_neon::rank_row_groups`](crate::kernel_neon::rank_row_groups) reports it), and only those groups are read.
    pub(crate) fn offer_groups<P: Producer>(
        &mut self,
        first: usize,
        row: RowId,
        tile: &[f64],
        mut hits: u64,
        n_of: impl Fn(usize) -> i64,
        variance_of: impl Fn(usize) -> u64,
    ) {
        self.delta = self.delta.max(P::DELTA);
        self.hold_champions();
        let start = first - self.base;
        while hits != 0 {
            let lo = hits.trailing_zeros() as usize * GROUP_LANES;
            hits &= hits - 1;
            let group = tile.iter().enumerate().skip(lo).take(GROUP_LANES);
            for (lane, &rho_hat) in group {
                if rho_hat >= self.floor[start + lane] {
                    self.admit::<P>(start + lane, row, rho_hat, n_of(lane), || variance_of(lane));
                }
            }
        }
    }

    /// Offers one ranked pair `(row, column)`; `variance` reads the column's variance.
    pub(crate) fn offer_lane<P: Producer>(
        &mut self,
        column: usize,
        row: RowId,
        lane: RankedLane,
        variance: impl FnOnce() -> u64,
    ) {
        if column < row.offers_from {
            crate::route::count_class_offer_lanes(1);
            return;
        }
        self.delta = self.delta.max(P::DELTA);
        self.hold_champions();
        let slot = column - self.base;
        if lane.rho_hat >= self.floor[slot] {
            self.admit::<P>(slot, row, lane.rho_hat, lane.n, variance);
        }
    }

    /// Room for every column's champion, made when pairs are first offered: an accumulator that
    /// only merges others never holds any.
    fn hold_champions(&mut self) {
        if self.champions.len() < self.floor.len() {
            self.champions.resize(self.floor.len(), Held::EMPTY);
        }
    }

    /// Admits the pair as a candidate of column `slot`. A pair that does not beat the column's best
    /// estimate is one a tie may have decided: unless `k` pairs of a run just before it tie it
    /// exactly, or have a correlation of exactly one, and come from smaller rows (they win the
    /// tie), it is admitted and joins the run. Rows reach a column in ascending order (a column's
    /// pairs come with its diagonals, in ascending slots), so once `k` pairs of correlation one
    /// are admitted nothing later can beat them, and the column's floor rises to `∞`; a checked
    /// build leaves the floor and drops those pairs here instead, asserting the order. A row past
    /// the column's cut ([`ConstantClass::column_cuts`]) is not admitted.
    fn admit<P: Producer>(
        &mut self,
        slot: usize,
        row: RowId,
        rho_hat: f64,
        n: i64,
        variance: impl FnOnce() -> u64,
    ) {
        if rho_hat == f64::NEG_INFINITY {
            return;
        }
        if self
            .cuts
            .get(slot)
            .is_some_and(|&cut| row.finite_before >= cut)
        {
            crate::route::count_class_column_pair();
            return;
        }
        let (epoch, k) = (self.epoch, self.k.get());
        if cfg!(debug_assertions)
            && let Some(held) = self.champions.get(slot)
            && held.epoch == epoch
            && held.champion.is_settled(k)
        {
            debug_assert!(
                held.champion.outranks_index(row.index as u64, k),
                "rows reach a column in ascending order"
            );
            return;
        }
        let mut settles = false;
        if !self.note(slot, rho_hat)
            && let Some(held) = self.champions.get_mut(slot)
        {
            if held.epoch != epoch {
                *held = Held::fresh(epoch);
            }
            let pair = Champion::of(row.index as u64, n, row.variance, rho_hat);
            if held.champion.beats(&pair, k) {
                return;
            }
            held.champion = held.champion.then::<P>(pair, variance, row.variance);
            settles = held.champion.is_settled(k) && !cfg!(debug_assertions);
        }
        self.cands.push(ColumnCand {
            column: self.base + slot,
            row: row.index,
            n,
            rho_hat,
        });
        if settles {
            let held = &mut self.champions[slot];
            held.settled_floor = self.floor[slot];
            self.floor[slot] = f64::INFINITY;
        }
        #[cfg(test)]
        {
            self.peak_cands = self.peak_cands.max(self.cands.len());
        }
    }

    /// The floor the column's candidates are judged by: the one it had when it settled, if it has.
    fn floor_of(&self, slot: usize) -> f64 {
        floor_at(&self.floor, &self.champions, self.epoch, slot)
    }

    /// Folds `rho_hat` into the column's `k` largest and moves its floor with the `k`-th. Returns
    /// whether it is among them now: better than the `k`-th (a tie is not).
    fn note(&mut self, slot: usize, rho_hat: f64) -> bool {
        let top = &mut self.top[self.offset[slot]..self.offset[slot + 1]];
        let among = kbest::heap_keep_in(top, rho_hat, |a, b| a > b);
        if among {
            let floor = band_floor(top[0], self.delta);
            self.floor[slot] = self.floor[slot].max(floor);
        }
        among
    }

    /// After [`Self::try_begin`], takes the cuts `class` sets for these columns at this `k`.
    pub(crate) fn cut_constant_columns(&mut self, class: &ConstantClass) {
        let (base, columns) = (self.base, self.columns());
        self.cuts.clear();
        self.cuts.resize(columns.len(), usize::MAX);
        let cuts = &mut self.cuts;
        class.column_cuts(columns, self.k, |column, cut| cuts[column - base] = cut);
    }

    /// Raises each column's floor to what `class` guarantees its nearest reach, under the bound
    /// `delta` of every producer whose estimates its floor will be compared with.
    pub(crate) fn seed_floors(&mut self, class: &ConstantClass, delta: f64) {
        self.delta = self.delta.max(delta);
        let (base, columns, delta) = (self.base, self.columns(), self.delta);
        let (offset, floor) = (&self.offset, &mut self.floor);
        let slots = |column: usize| offset[column - base + 1] - offset[column - base];
        class.column_seeds(columns, slots, |column, rho| {
            let floor = &mut floor[column - base];
            *floor = floor.max(band_floor(rho, delta));
            crate::route::count_class_column_seed();
        });
    }

    /// Raises the floors of `columns` (in both) to `other`'s where those are higher: a pair below
    /// a column's floor anywhere is beaten by `k` pairs that column already has, so it can never
    /// be among its nearest, and a part that starts from the shared floors admits only pairs
    /// that still can.
    pub(crate) fn raise_floors(&mut self, other: &ColumnAcc, columns: core::ops::Range<usize>) {
        let mine = &mut self.floor[columns.start - self.base..columns.end - self.base];
        let theirs = &other.floor[columns.start - other.base..columns.end - other.base];
        for (floor, &shared) in mine.iter_mut().zip(theirs) {
            *floor = floor.max(shared);
        }
        self.delta = self.delta.max(other.delta);
    }

    /// Bounds the candidate list once it passes its limit: drops what the floors rule out, then
    /// reduces each column to its exact nearest row. Called between blocks of offers, so the list
    /// stays within the columns' count plus one block's pairs however many ties there are.
    pub(crate) fn compact(&mut self, windows: &[Window]) {
        if self.cands.len() <= self.prune_at {
            return;
        }
        self.prune();
        if self.cands.len() > COLUMN_RESOLVE_PER_COLUMN * self.top.len().max(COLUMN_PRUNE_AT) {
            self.keep_exact_nearest(windows);
        }
        self.prune_at = (2 * self.cands.len()).max(COLUMN_PRUNE_AT);
    }

    fn prune(&mut self) {
        let (base, epoch) = (self.base, self.epoch);
        let (floor, champions) = (&self.floor, &self.champions);
        self.cands
            .retain(|c| c.rho_hat >= floor_at(floor, champions, epoch, c.column - base));
    }

    /// Leaves each column its exact nearest `k` candidates (ties to the smaller row). Dropping the
    /// others is safe: the exact order is total, so a candidate beaten by `k` kept ones is never
    /// among the column's `k` nearest, whichever pairs arrive later.
    pub(super) fn keep_exact_nearest(&mut self, windows: &[Window]) {
        self.sort();
        let mut kept = 0;
        let mut start = 0;
        while start < self.cands.len() {
            let column = self.cands[start].column;
            let end = start
                + self.cands[start..]
                    .iter()
                    .take_while(|c| c.column == column)
                    .count();
            if end - start > self.k.get() {
                let mut nearest = Nearest::selecting(&windows[column], self.k, end - start);
                nearest.select(self.cands[start..end].iter().map(|c| {
                    let row =
                        SampleIndex::new(c.row as u64).expect("index is below NONE's sentinel");
                    (c.n, windows[c.row], row)
                }));
                self.kept.clear();
                self.kept.extend(nearest.kept_indices());
                self.kept.sort_unstable();
                for i in start..end {
                    if self.kept.binary_search(&(self.cands[i].row as u64)).is_ok() {
                        self.cands[kept] = self.cands[i];
                        kept += 1;
                    }
                }
            } else {
                self.cands.copy_within(start..end, kept);
                kept += end - start;
            }
            start = end;
        }
        self.cands.truncate(kept);
    }

    /// Orders the candidates by column, for [`Self::merge`] and [`Self::finish`].
    pub(crate) fn sort(&mut self) {
        let (len, columns) = (self.cands.len(), self.floor.len());
        if len < 2 {
            return;
        }
        if len.saturating_mul(4) < columns {
            self.cands.sort_unstable_by_key(|c| c.column);
            return;
        }
        // Many candidates over few columns: count them, then swap each into its column's run, in
        // place, linear in both.
        let base = self.base;
        self.starts.clear();
        self.starts.resize(columns + 1, 0);
        for c in &self.cands {
            self.starts[c.column - base + 1] += 1;
        }
        for column in 0..columns {
            self.starts[column + 1] += self.starts[column];
        }
        self.placed.clear();
        self.placed.extend_from_slice(&self.starts[..columns]);
        for column in 0..columns {
            while self.placed[column] < self.starts[column + 1] {
                let at = self.placed[column];
                let owner = self.cands[at].column - base;
                if owner == column {
                    self.placed[column] += 1;
                } else {
                    self.cands.swap(at, self.placed[owner]);
                    self.placed[owner] += 1;
                }
            }
        }
    }

    /// Takes `other`'s share of `columns` (which must lie in both; `other` sorted): for tiling and
    /// threads. The merged floor only rises as more partials arrive and `finish` keeps only
    /// candidates at or above the final one, so dropping below the current one loses nothing.
    pub fn merge(
        &mut self,
        other: &ColumnAcc,
        columns: core::ops::Range<usize>,
        windows: &[Window],
    ) {
        self.delta = self.delta.max(other.delta);
        for column in columns.clone() {
            let (mine, theirs) = (column - self.base, column - other.base);
            for &rho_hat in &other.top[other.offset[theirs]..other.offset[theirs + 1]] {
                if rho_hat != f64::NEG_INFINITY {
                    self.note(mine, rho_hat);
                }
            }
        }
        let from = other.cands.partition_point(|c| c.column < columns.start);
        let to = other.cands.partition_point(|c| c.column < columns.end);
        let (base, floor) = (self.base, &self.floor);
        self.cands.extend(
            other.cands[from..to]
                .iter()
                .filter(|c| c.rho_hat >= floor[c.column - base]),
        );
        #[cfg(test)]
        {
            self.peak_cands = self.peak_cands.max(self.cands.len());
        }
        self.compact(windows);
    }

    /// Test-only: the largest `ρ̂` the column has seen, descending, `-∞` where fewer arrived.
    #[cfg(test)]
    pub(crate) fn top_of(&self, column: usize) -> Vec<f64> {
        let slot = column - self.base;
        let mut top = self.top[self.offset[slot]..self.offset[slot + 1]].to_vec();
        top.sort_by(|a, b| b.total_cmp(a));
        top
    }

    /// Whether column `column` holds room for all `k` of its nearest here, so that its floor is a
    /// floor of the whole run's, not only of the rows this part can offer it.
    #[inline(always)]
    pub(crate) fn keeps_k(&self, column: usize) -> bool {
        let slot = column - self.base;
        self.offset[slot + 1] - self.offset[slot] == self.k.get()
    }

    /// Test-only: sets column `column`'s floor, as offers would raise it; `∞` settles it as a
    /// release build does once `k` pairs of correlation one from its smallest rows are admitted,
    /// which a checked build never does in the pass.
    #[cfg(test)]
    pub(crate) fn set_floor_for_test(&mut self, column: usize, floor: f64) {
        self.floor[column - self.base] = floor;
    }

    /// The columns' range.
    pub(crate) fn columns(&self) -> core::ops::Range<usize> {
        self.base..self.base + self.floor.len()
    }

    /// Every column's nearest rows, in column order, appended to `out` the way a batch row is,
    /// does: each column's candidates at or above its final floor offered to a [`Nearest`] of that
    /// column's own window.
    pub fn finish(&mut self, windows: &[Window], m: usize, out: &mut Vec<(u64, Entry)>) {
        self.sort();
        let mut next = 0;
        for column in self.columns() {
            let run = self.cands[next..]
                .iter()
                .take_while(|c| c.column == column)
                .count();
            let mut nearest = Nearest::with_k(&windows[column], m, self.k, run);
            let floor = self.floor_of(column - self.base);
            nearest.select(
                self.cands[next..next + run]
                    .iter()
                    .filter(|c| c.rho_hat >= floor)
                    .map(|c| {
                        let index =
                            SampleIndex::new(c.row as u64).expect("index is below NONE's sentinel");
                        (c.n, windows[c.row], index)
                    }),
            );
            next += run;
            nearest.push_kept(column as u64, self.k, out);
        }
        self.cands.clear();
    }
}

impl Default for ColumnAcc {
    fn default() -> Self {
        Self::new()
    }
}
