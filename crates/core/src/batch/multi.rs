//! The multi-dimensional profile on the batch tile scheduler: for every window and every `k`, the
//! eligible window whose mean over the `k + 1` chosen dimensions is smallest, equal on bits to the
//! exact reference at any thread count and tile shape.
//!
//! The matrix of windows by windows is cut into tiles as an AB-join of the series with itself,
//! and a tile walks each of its diagonals, each dimension down the diagonal's cells with the
//! metric's own recurrence ([`Metric`]), then the cells across the dimensions. A cell is first
//! offered to the estimate screen (`Screen`): from a bound at or below each dimension's exact
//! distance, a cell whose every mean is provably farther than its row's best so far is settled
//! with no exact distance. Every other cell is read from its row's side only: the dimensions'
//! distances of row `i` against candidate `j`, ordered once per cell by `order_dimensions_into`,
//! summed into the means of the first `k + 1` ([`MeanDistance`]) and offered to the row's `k`th
//! slot, which keeps the smallest, ties to the smaller index. The screen settles only cells the
//! exact path would take nothing from, so it decides how fast a profile is found, never its bits;
//! the forced wide route turns it off. Memory is `O(d·n)`: the dimensions, and a slot per
//! dimension and row.
//!
//! Two means whose values are within the error bound of each other are ordered exactly (see
//! `MeanOrder`), which needs the distances of the dimensions each is the mean of. A candidate
//! has them in hand; a slot's best does not keep them (a slot taking a nearer mean costs one
//! store). They are found again from the pair when the first exact decision against that slot
//! needs them, and kept until the slot takes another window: `d(d + 1)/2` distances per band row,
//! flat in `n`.
//!
//! The arithmetic of a cell is one function, `walk_tile`, generic over the [`Metric`]; what a
//! metric adds (the per-window values it reads, and a pair's distance from the state it carries)
//! reaches it as two closures, the only places the metrics differ. The non-normalized metric
//! differs in nothing else but the width its tiles run in (`u64` or `u128`, by the range of the
//! ticks a tile holds, as in [`super::nonnorm`]) and its rule for pairs no width holds.

use super::nonnorm::{TickSeries, first_unrepresentable_row, joined, range_of};
use super::pass::{Shape, push_none_rows};
use super::plan::{Tile, TileCut, TilePlan, default_chunk_lanes, threads_for_join};
use super::quantized::Quantized;
use super::schedule::{BatchReport, TileWork, run_tile_work};
use super::tile::RanWide;
use crate::index::SampleIndex;
use crate::kernel::Metric;
use crate::multidim::{
    DimensionDistance, IncludeOrder, MeanDistance, MeanOrder, MeanWork, MultiError, MultiProfile,
    checked_include, exact_order, order_dimensions_into,
};
use crate::nearest::ExactDistance;
use crate::nonnorm::{NonNormalized, Residue, RowWidth, SquaredTicks, TickStep, routed_row_width};
use crate::pattern::exact_from_cross;
use crate::quantize::Scale;
use crate::route;
use crate::row::centered_cross_fast;
use crate::stats::Series;
use crate::znorm::{TickWindow, ZNorm};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZeroUsize;
use screen::{Screen, root_lower, znorm_bounds};

/// A row's best candidate for one `k`: the nearest mean so far and its window start,
/// [`NO_NEIGHBOR`] for none.
type Slot<D> = (MeanDistance<D>, u64);

const NO_NEIGHBOR: u64 = u64::MAX;

fn empty_slots<D: DimensionDistance>(count: usize) -> Vec<Slot<D>> {
    vec![(MeanDistance::MISSING, NO_NEIGHBOR); count]
}

/// Where the distances of slot `k`'s dimensions start in a row's flat list of them: slot `k`
/// keeps its `k + 1`.
const fn terms_start(k: usize) -> usize {
    k * (k + 1) / 2
}

/// A slot while a row is built: the value of its best mean and the window it is the mean for, and
/// the value past which a mean is decidedly farther (its value times the error ratio), so that the
/// mean a slot has no use for costs one comparison.
///
/// `fixed` is the exact mean when every distance in it is one whose exact value no window pair
/// can change: the number of them that are `√m` (the rest are zero), since the mean is then that
/// number times `√m` and two such means compare by it, with no distance to find again;
/// [`NOT_FIXED`] otherwise.
#[derive(Clone, Copy)]
struct Entry {
    value: f64,
    farther: f64,
    column: u64,
    fixed: u32,
}

const NOT_FIXED: u32 = u32::MAX;

const NO_ENTRY: Entry = Entry {
    value: f64::INFINITY,
    farther: f64::INFINITY,
    column: NO_NEIGHBOR,
    fixed: NOT_FIXED,
};

/// A row's best candidates, one entry per `k`.
struct Best<D> {
    entries: Vec<Entry>,
    /// The slots whose mean the values of the cell in hand could not place, with the value.
    near: Vec<(usize, f64)>,
    dimension: core::marker::PhantomData<fn() -> D>,
}

impl<D: DimensionDistance> Best<D> {
    fn new(count: usize) -> Self {
        Self {
            entries: vec![NO_ENTRY; count],
            near: Vec::with_capacity(count),
            dimension: core::marker::PhantomData,
        }
    }

    /// No candidate for any slot.
    fn clear(&mut self) {
        self.entries.fill(NO_ENTRY);
        self.near.clear();
    }

    fn take(&mut self, ranking: &MeanOrder, k: usize, value: f64, column: u64, fixed: u32) {
        self.entries[k] = Entry {
            value,
            farther: value * ranking.ratio(k),
            column,
            fixed,
        };
    }

    /// The finished slots.
    fn slots(&self) -> Vec<Slot<D>> {
        self.entries
            .iter()
            .enumerate()
            .map(|(k, entry)| {
                if entry.column == NO_NEIGHBOR {
                    (MeanDistance::MISSING, NO_NEIGHBOR)
                } else {
                    (MeanDistance::new(entry.value, k + 1), entry.column)
                }
            })
            .collect()
    }
}

/// The buffers that find a pair's distances again: every dimension's distance, the order they
/// go in, and the distances in that order.
struct Found<D> {
    distances: Vec<Option<D>>,
    order: Vec<usize>,
    ordered: Vec<Option<D>>,
}

/// What an exact decision against a row's slots needs, made when a row first needs one (nearly no
/// row does), so that a band's partials stay small whatever the number of dimensions:
/// `terms` holds, for each slot, the distances of the `k + 1` dimensions its best is the mean of,
/// if `terms_of[k]` is the window they were found for.
struct Extras<D> {
    work: MeanWork<D>,
    found: Found<D>,
    terms: Vec<Option<D>>,
    terms_of: Vec<u64>,
}

impl<D: DimensionDistance> Extras<D> {
    fn new(count: usize) -> Box<Self> {
        Box::new(Self {
            work: MeanWork::new(),
            found: Found {
                distances: Vec::new(),
                order: Vec::new(),
                ordered: Vec::new(),
            },
            terms: vec![None; terms_start(count)],
            terms_of: vec![NO_NEIGHBOR; count],
        })
    }
}

/// One row's best candidates, one per `k`, and the extras an exact decision against them needs.
/// The row's finished slots leave without them.
struct Partial<D> {
    row: usize,
    best: Best<D>,
    extras: Option<Box<Extras<D>>>,
}

impl<D: DimensionDistance> Partial<D> {
    fn new(count: usize) -> Self {
        Self {
            row: 0,
            best: Best::new(count),
            extras: None,
        }
    }

    /// Empties the entries for `row`.
    fn begin(&mut self, row: usize) {
        self.row = row;
        self.best.clear();
        if let Some(extras) = &mut self.extras {
            extras.terms_of.fill(NO_NEIGHBOR);
        }
    }
}

/// What finds a pair's distances again: `find(row, column, out)` puts every dimension's distance
/// for the pair in `out`, which `include` and `discords` then order.
struct Pairs<'a, F> {
    find: &'a F,
    include: &'a [usize],
    discords: bool,
}

/// Takes a mean of value `value` for window `column` into slot `k` if its value decides that it is
/// nearer than the slot's, and lists `k` in `near` if the values could not decide. A mean farther
/// than the slot's by more than the error bound, nearly every one, costs one comparison; a tie in
/// value goes on to an exact decision.
#[inline(always)]
fn offer_by_value<D: DimensionDistance>(
    ranking: &MeanOrder,
    best: &mut Best<D>,
    k: usize,
    value: f64,
    column: u64,
) {
    let entry = &mut best.entries[k];
    if value > entry.farther {
        return;
    }
    let farther = value * ranking.ratio(k);
    if farther < entry.value {
        *entry = Entry {
            value,
            farther,
            column,
            fixed: NOT_FIXED,
        };
    } else if value != f64::INFINITY {
        best.near.push((k, value));
    }
}

/// [`offer_by_value`] for a cell whose first distances are fixed, as a flat or constant window
/// makes them: `fixed` counts the `√m` among the first `k + 1` while all of them are fixed, and
/// two fixed means order by it exactly, whatever their values round to, with no distance found
/// again.
#[inline(never)]
fn offer_fixed_cell<D: DimensionDistance>(
    ranking: &MeanOrder,
    best: &mut Best<D>,
    column: u64,
    ordered: &[Option<D>],
) {
    let mut roots = 0u32;
    let mut all_fixed = true;
    let mut sum = 0.0;
    for (k, distance) in ordered.iter().enumerate() {
        let Some(distance) = distance else {
            // This and every later mean is missing, and a missing mean is never taken.
            return;
        };
        let fixed = if all_fixed { distance.fixed() } else { None };
        // A distance that is exactly zero adds exactly zero, with no root to take.
        sum += match fixed {
            Some(false) => 0.0,
            _ => distance.root(),
        };
        let fixed = match fixed {
            Some(root) => {
                roots += u32::from(root);
                roots
            }
            None => {
                all_fixed = false;
                NOT_FIXED
            }
        };
        let value = if k == 0 { sum } else { sum / (k + 1) as f64 };
        let entry = &best.entries[k];
        let takes = if entry.column == NO_NEIGHBOR {
            true
        } else if fixed != NOT_FIXED && entry.fixed != NOT_FIXED {
            fixed < entry.fixed || (fixed == entry.fixed && column < entry.column)
        } else if value > entry.farther {
            false
        } else if value * ranking.ratio(k) < entry.value {
            true
        } else {
            best.near.push((k, value));
            false
        };
        if takes {
            best.take(ranking, k, value, column, fixed);
        }
    }
}

/// Makes `terms` hold, for slot `k`, the first `k + 1` distances of the pair of `row` and window
/// `column`, in their order, unless it does already; and the same for every slot whose best is
/// that window, which the one search gives.
#[allow(clippy::too_many_arguments)]
fn ensure_held<D: DimensionDistance, F: Fn(usize, usize, &mut Vec<Option<D>>)>(
    pairs: &Pairs<F>,
    found: &mut Found<D>,
    entries: &[Entry],
    terms: &mut [Option<D>],
    terms_of: &mut [u64],
    row: usize,
    k: usize,
    column: u64,
) {
    if terms_of[k] == column {
        return;
    }
    let Found {
        distances,
        order,
        ordered,
    } = found;
    (pairs.find)(row, column as usize, distances);
    order_dimensions_into(
        order,
        distances,
        pairs.include,
        pairs.discords,
        IncludeOrder::AsGiven,
    );
    ordered.clear();
    ordered.extend(order.iter().map(|&a| distances[a]));
    for (slot, entry) in entries.iter().enumerate() {
        if entry.column == column && terms_of[slot] != column {
            terms[terms_start(slot)..][..=slot].copy_from_slice(&ordered[..=slot]);
            terms_of[slot] = column;
        }
    }
    debug_assert_eq!(terms_of[k], column);
}

/// [`offer_by_value`] for the slots `near` lists, whose means the values could not place: takes a
/// mean for window `column` if it is exactly nearer than the slot's best, or equal with a smaller
/// window start. The first `k + 1` of `candidate` are the distances slot `k`'s mean is of; the
/// slot's are found again if not kept.
#[cold]
#[inline(never)]
fn offer_exactly<D: DimensionDistance, F: Fn(usize, usize, &mut Vec<Option<D>>)>(
    pairs: &Pairs<F>,
    ranking: &MeanOrder,
    partial: &mut Partial<D>,
    column: u64,
    candidate: &[Option<D>],
) {
    let Partial { row, best, extras } = partial;
    let Extras {
        work,
        found,
        terms,
        terms_of,
    } = &mut **extras.get_or_insert_with(|| Extras::new(best.entries.len()));
    while let Some((k, value)) = best.near.pop() {
        let held = best.entries[k].column;
        ensure_held(pairs, found, &best.entries, terms, terms_of, *row, k, held);
        let held_terms = &terms[terms_start(k)..][..=k];
        // Distances equal one by one make the means equal, and the kept ones then stand for the
        // candidate's too: a window that ties with the best again and again costs no search.
        let same = candidate[..=k].iter().zip(held_terms).all(|(a, b)| a == b);
        let order = if same {
            crate::multidim::exact_decision();
            Ordering::Equal
        } else {
            exact_order(work, &candidate[..=k], held_terms)
        };
        if order == Ordering::Less || (order == Ordering::Equal && column < held) {
            best.take(ranking, k, value, column, NOT_FIXED);
            if same {
                terms_of[k] = column;
            }
        }
    }
}

/// What a tile works in: the partial slots of each band row, and the buffers one cell uses.
struct Scratch<D> {
    rows: Vec<Partial<D>>,
    ranking: MeanOrder,
    screen: Screen,
    /// A diagonal's bounds, dimension by dimension.
    bounds: Vec<f64>,
    distances: Vec<Option<D>>,
    ordered: Vec<Option<D>>,
    order: Vec<usize>,
}

/// What every tile of a run shares.
struct Cut<'a> {
    windows: usize,
    exclusion: usize,
    include: &'a [usize],
    discords: bool,
}

/// Every dimension's distance for the pair of window `row` and window `column`, started from
/// their ticks as a tile starts a diagonal.
fn pair_distances<M: Metric, D>(
    ticks: &[&[i64]],
    m: usize,
    window: &impl Fn(usize, usize) -> M::Step,
    distance: &impl Fn(usize, M::Diagonal, usize, usize) -> Option<D>,
    row: usize,
    column: usize,
    out: &mut Vec<Option<D>>,
) {
    out.clear();
    out.extend(ticks.iter().enumerate().map(|(a, series)| {
        let state = M::start(
            &series[row..row + m],
            &window(a, row),
            &series[column..column + m],
            &window(a, column),
        );
        distance(a, state, row, column)
    }));
}

/// What a metric gives the walk: `window(a, i)` is dimension `a`'s per-window values at window
/// `i`; `distance(a, state, i, j)` is its distance for the pair `(i, j)` whose state is `state`,
/// from row `i`'s side; `states(a, i, j, cells, out)` appends the states of the `cells` cells
/// `(i + c, j + c)`, and `bounds(a, i, j, states, out)` writes to `out` a bound at or below the
/// exact distance of each, infinite when there is none.
struct Dimensions<'a, W, F, S, L> {
    ticks: &'a [&'a [i64]],
    window: &'a W,
    distance: &'a F,
    states: &'a S,
    bounds: &'a L,
}

/// Every diagonal of `tile` outside the exclusion zone down its rows, all `ticks.len()`
/// dimensions at once, into the band's partial slots. With `screened`, a cell the bounds settle
/// (see [`Screen`]) costs no exact distance; without it, every cell takes the exact path.
fn walk_tile<M: Metric, D: DimensionDistance>(
    cut: &Cut,
    m: usize,
    dimensions: &Dimensions<
        impl Fn(usize, usize) -> M::Step,
        impl Fn(usize, M::Diagonal, usize, usize) -> Option<D>,
        impl Fn(usize, usize, usize, usize, &mut Vec<M::Diagonal>),
        impl Fn(usize, usize, usize, &[M::Diagonal], Strided),
    >,
    tile: &Tile,
    scratch: &mut Scratch<D>,
    screened: bool,
) {
    let Dimensions {
        ticks,
        window,
        distance,
        states: states_of,
        bounds: bounds_of,
    } = *dimensions;
    let Scratch {
        rows,
        ranking,
        screen,
        bounds,
        distances,
        ordered,
        order,
    } = scratch;
    let find = |row: usize, column: usize, out: &mut Vec<Option<D>>| {
        pair_distances::<M, D>(ticks, m, window, distance, row, column, out);
    };
    let pairs = Pairs {
        find: &find,
        include: cut.include,
        discords: cut.discords,
    };
    let windows = cut.windows as i64;
    let count = ticks.len();
    // A diagonal's states, dimension by dimension, each down the diagonal's cells; its bounds,
    // cell by cell, each across the dimensions.
    let mut states: Vec<M::Diagonal> = Vec::with_capacity(tile.rows.len() * count);
    bounds.clear();
    bounds.reserve(tile.rows.len() * count);
    let mut settled = 0;
    for slot in tile.qa..tile.qb {
        let offset = windows - 1 - slot as i64;
        if offset.unsigned_abs() as usize <= cut.exclusion {
            continue;
        }
        let first = (tile.rows.start as i64).max(-offset);
        let end = (tile.rows.end as i64).min(windows - offset);
        if first >= end {
            continue;
        }
        let (first, end) = (first as usize, end as usize);
        let start_column = (first as i64 + offset) as usize;
        let cells = end - first;
        states.clear();
        if screened {
            bounds.resize(cells * count, 0.0);
        }
        // Each dimension down the whole diagonal, then the cells across the dimensions.
        for a in 0..count {
            states_of(a, first, start_column, cells, &mut states);
            if screened {
                let out = bounds[a..].iter_mut().step_by(count);
                bounds_of(a, first, start_column, &states[a * cells..], out);
            }
        }
        for cell in 0..cells {
            let row = first + cell;
            let column = start_column + cell;
            let partial = &mut rows[row - tile.rows.start];
            if screened && screen.settles(row, &bounds[cell * count..][..count]) {
                settled += 1;
                continue;
            }
            distances.clear();
            distances
                .extend((0..count).map(|a| distance(a, states[a * cells + cell], row, column)));
            order_dimensions_into(
                order,
                distances,
                cut.include,
                cut.discords,
                IncludeOrder::AsGiven,
            );
            ordered.clear();
            ordered.extend(order.iter().map(|&a| distances[a]));
            if matches!(ordered.first(), Some(Some(first)) if first.fixed().is_some()) {
                offer_fixed_cell(ranking, &mut partial.best, column as u64, ordered);
            } else {
                MeanDistance::each_prefix(ordered, |k, mean| {
                    offer_by_value(ranking, &mut partial.best, k, mean.to_f64(), column as u64);
                });
            }
            if !partial.best.near.is_empty() {
                offer_exactly(&pairs, ranking, partial, column as u64, ordered);
            }
            if screened {
                let entries = &partial.best.entries;
                screen.lower(row, |k| entries[k].farther);
            }
        }
    }
    crate::multidim::cells_screened(settled);
}

/// Every `count`th bound of a diagonal's, from one dimension's first: where that dimension's
/// bounds go, cell by cell.
type Strided<'a> = core::iter::StepBy<core::slice::IterMut<'a, f64>>;

/// The window nearest `row` past the exclusion zone: after it, or before it at the end.
fn seed_column(row: usize, windows: usize, exclusion: usize) -> Option<usize> {
    let after = row + exclusion + 1;
    if after < windows {
        Some(after)
    } else {
        row.checked_sub(exclusion + 1)
    }
}

/// Appends the states of the `cells` cells of a diagonal from `(row, column)` in one dimension
/// of `ticks`: `first` is the two windows' steps at the first cell, and `rest` the steps from the
/// second cell on.
fn diagonal_states<M: Metric>(
    ticks: &[i64],
    m: usize,
    (row, column): (usize, usize),
    cells: usize,
    first: (M::Step, M::Step),
    rest: impl Iterator<Item = (M::Step, M::Step)>,
    out: &mut Vec<M::Diagonal>,
) {
    let mut state = M::start(
        &ticks[row..row + m],
        &first.0,
        &ticks[column..column + m],
        &first.1,
    );
    out.push(state);
    out.extend(rest.take(cells - 1).map(|(a, b)| {
        state = M::advance(state, &a, &b);
        state
    }));
}

/// The tile work of a multi-dimensional profile: `body` walks a tile into the scratch.
struct Tiles<'a, D, F, P> {
    distance: core::marker::PhantomData<fn() -> D>,
    count: usize,
    ranking: MeanOrder,
    cut: &'a Cut<'a>,
    find: &'a P,
    body: &'a F,
}

impl<D: DimensionDistance, F, P: Fn(usize, usize, &mut Vec<Option<D>>)> Tiles<'_, D, F, P> {
    /// Starts the screen on a tile: a band it has not walked just before starts each row's
    /// ceilings from one window the row is surely offered, the nearest past the exclusion zone
    /// (which a smooth series makes near), its distances found exactly whatever the tile's width.
    fn begin_screen(&self, tile: &Tile, scratch: &mut Scratch<D>) {
        let Scratch {
            screen,
            ranking,
            distances,
            ordered,
            order,
            ..
        } = scratch;
        if !screen.begin(&tile.rows) {
            return;
        }
        for row in tile.rows.clone() {
            let Some(column) = seed_column(row, self.cut.windows, self.cut.exclusion) else {
                continue;
            };
            (self.find)(row, column, distances);
            order_dimensions_into(
                order,
                distances,
                self.cut.include,
                self.cut.discords,
                IncludeOrder::AsGiven,
            );
            ordered.clear();
            ordered.extend(order.iter().map(|&a| distances[a]));
            MeanDistance::each_prefix(ordered, |k, mean| {
                if !mean.is_missing() {
                    screen.lower_one(row, k, mean.to_f64() * ranking.ratio(k));
                }
            });
        }
    }
}

impl<
    D: DimensionDistance + Send,
    F: Fn(&Tile, &mut Scratch<D>) -> RanWide + Sync,
    P: Fn(usize, usize, &mut Vec<Option<D>>) + Sync,
> TileWork for Tiles<'_, D, F, P>
{
    type Scratch = Scratch<D>;
    type Partial = Partial<D>;
    type Row = Vec<Slot<D>>;

    fn scratch(&self, plan: TilePlan) -> Scratch<D> {
        Scratch {
            rows: (0..plan.band_rows)
                .map(|_| Partial::new(self.count))
                .collect(),
            ranking: self.ranking.clone(),
            screen: Screen::new(
                self.count,
                self.cut.include,
                self.cut.discords,
                plan.band_rows,
            ),
            bounds: Vec::new(),
            distances: Vec::with_capacity(self.count),
            ordered: Vec::with_capacity(self.count),
            order: Vec::with_capacity(self.count),
        }
    }

    fn partials<'a>(&self, scratch: &'a mut Scratch<D>) -> &'a mut [Partial<D>] {
        &mut scratch.rows
    }

    fn run(&self, tile: &Tile, scratch: &mut Scratch<D>, _chunk_lanes: usize) -> RanWide {
        for (row, partial) in tile.rows.clone().zip(&mut scratch.rows) {
            partial.row = row;
            partial.best.clear();
            if let Some(extras) = &mut partial.extras {
                extras.terms_of.fill(NO_NEIGHBOR);
            }
        }
        self.begin_screen(tile, scratch);
        (self.body)(tile, scratch)
    }

    fn new_partial(&self) -> Partial<D> {
        Partial::new(self.count)
    }

    fn begin(&self, partial: &mut Partial<D>, row: usize) {
        partial.begin(row);
    }

    fn merge(&self, into: &mut Partial<D>, from: &mut Partial<D>) {
        let pairs = Pairs {
            find: self.find,
            include: self.cut.include,
            discords: self.cut.discords,
        };
        let Partial { row, best, extras } = from;
        let Partial {
            best: into_best,
            extras: into_extras,
            ..
        } = into;
        for k in 0..best.entries.len() {
            let entry = best.entries[k];
            let column = entry.column;
            if column == NO_NEIGHBOR {
                continue;
            }
            let held_entry = into_best.entries[k];
            let held = held_entry.column;
            let order = if held == NO_NEIGHBOR {
                Ordering::Less
            } else if entry.fixed != NOT_FIXED && held_entry.fixed != NOT_FIXED {
                entry.fixed.cmp(&held_entry.fixed)
            } else if entry.value > held_entry.farther {
                Ordering::Greater
            } else if entry.value * self.ranking.ratio(k) < held_entry.value {
                Ordering::Less
            } else {
                let count = best.entries.len();
                let Extras {
                    work,
                    found,
                    terms,
                    terms_of,
                } = &mut **extras.get_or_insert_with(|| Extras::new(count));
                let Extras {
                    terms: into_terms,
                    terms_of: into_terms_of,
                    ..
                } = &mut **into_extras.get_or_insert_with(|| Extras::new(count));
                let held_entries = &into_best.entries;
                ensure_held(
                    &pairs,
                    found,
                    held_entries,
                    into_terms,
                    into_terms_of,
                    *row,
                    k,
                    held,
                );
                ensure_held(
                    &pairs,
                    found,
                    &best.entries,
                    terms,
                    terms_of,
                    *row,
                    k,
                    column,
                );
                let candidate = &terms[terms_start(k)..][..=k];
                exact_order(work, candidate, &into_terms[terms_start(k)..][..=k])
            };
            if order == Ordering::Less || (order == Ordering::Equal && column < held) {
                into_best.take(&self.ranking, k, entry.value, column, entry.fixed);
            }
        }
    }

    fn finish(&self, row: usize, partial: &mut Partial<D>, out: &mut Vec<(u64, Vec<Slot<D>>)>) {
        out.push((row as u64, partial.best.slots()));
    }

    fn push_none_rows(&self, count: usize, out: &mut Vec<(u64, Vec<Slot<D>>)>) {
        push_none_rows(count, out, || empty_slots(self.count));
    }

    #[cfg(test)]
    fn visits(&self, _partial: &Partial<D>) -> Vec<usize> {
        Vec::new()
    }
}

/// The tile shape `plan` fixes, or the one [`TilePlan::for_join`] cuts for `threads`, with the
/// band's partial slots (one per dimension and row) kept to the budget of one.
fn multi_plan(
    windows: usize,
    m: usize,
    count: usize,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
) -> TilePlan {
    plan.unwrap_or_else(|| {
        let mut plan =
            TilePlan::for_join(windows, windows, m, threads.get(), default_chunk_lanes());
        plan.band_rows = (plan.band_rows / count).max(1);
        plan
    })
}

/// Runs `body` over the tiles of `windows` windows against themselves, and gathers the rows into
/// a profile of `count` dimensions.
#[allow(clippy::too_many_arguments)]
fn profile_with<D: DimensionDistance + Send>(
    windows: usize,
    m: usize,
    count: usize,
    cut: &Cut,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
    find: &(impl Fn(usize, usize, &mut Vec<Option<D>>) + Sync),
    body: &(impl Fn(&Tile, &mut Scratch<D>) -> RanWide + Sync),
) -> (MultiProfile<D>, BatchReport) {
    let mut distance = vec![Vec::with_capacity(windows); count];
    let mut index = vec![Vec::with_capacity(windows); count];
    if windows == 0 {
        let report = BatchReport::of(0, 0);
        return (MultiProfile { distance, index }, report);
    }
    let work = Tiles {
        distance: core::marker::PhantomData,
        count,
        ranking: MeanOrder::new(count, D::ROOT_RELATIVE_ERROR),
        cut,
        find,
        body,
    };
    let plan = multi_plan(windows, m, count, threads, plan);
    let mut rows = Vec::with_capacity(windows);
    let report = run_tile_work(
        &work,
        TileCut::join(windows, windows, plan, None),
        threads,
        &mut rows,
    );
    for (_, slots) in rows {
        for (k, (mean, column)) in slots.into_iter().enumerate() {
            distance[k].push(mean);
            index[k].push(SampleIndex::new(column).unwrap_or(SampleIndex::NONE));
        }
    }
    (MultiProfile { distance, index }, report)
}

/// The shared checks of an input: at least one dimension, all of one length, a window count, and
/// the `include` list.
fn checked(
    lengths: impl ExactSizeIterator<Item = usize> + Clone,
    m: NonZeroUsize,
    include: &[usize],
) -> Result<(usize, Vec<usize>), MultiError> {
    let count = lengths.len();
    let mut lengths = lengths;
    let Some(n) = lengths.next() else {
        return Err(MultiError::NoDimensions);
    };
    if lengths.any(|length| length != n) {
        return Err(MultiError::UnequalLengths);
    }
    Ok((
        (n + 1).saturating_sub(m.get()),
        checked_include(include, count)?,
    ))
}

fn znorm_profile(
    dimensions: &[Series<'_>],
    m: NonZeroUsize,
    scales: &[Scale],
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
) -> Result<(MultiProfile, BatchReport), MultiError> {
    let (windows, include) = checked(dimensions.iter().map(|series| series.len()), m, include)?;
    if scales.len() != dimensions.len() {
        return Err(MultiError::ScaleCount);
    }
    let mu = m.get();
    let quantized: Vec<Quantized> = dimensions
        .iter()
        .zip(scales)
        .map(|(&series, &scale)| Quantized::build(series, m, scale))
        .collect();
    let ticks: Vec<&[i64]> = quantized.iter().map(|q| q.ticks.as_slice()).collect();
    let cut = Cut {
        windows,
        exclusion: crate::exclusion::zone(mu),
        include: &include,
        discords,
    };
    // The two closures are where the z-normalized metric differs from any other.
    let window = |a: usize, i: usize| TickWindow {
        entering: ticks[a][i + mu - 1],
        leaving: if i > 0 { ticks[a][i - 1] } else { 0 },
        stats: quantized[a].windows[i],
    };
    let distance = |a: usize, qt: i64, i: usize, j: usize| -> Option<ExactDistance> {
        let stats = &quantized[a].windows;
        let n = centered_cross_fast(qt, mu as i64, stats[i].sum, stats[j].sum);
        exact_from_cross(n, &stats[i], &stats[j], mu, j as u64)
    };
    let states = |a: usize, i: usize, j: usize, cells: usize, out: &mut Vec<i64>| {
        let (ticks, windows) = (&quantized[a].ticks, &quantized[a].windows);
        // From a diagonal's second cell on, every window has a tick it just dropped.
        let steps = |i: usize| {
            ticks[i + mu..]
                .iter()
                .zip(&ticks[i..])
                .zip(&windows[i + 1..])
                .map(|((&entering, &leaving), &stats)| TickWindow {
                    entering,
                    leaving,
                    stats,
                })
        };
        let first = (window(a, i), window(a, j));
        diagonal_states::<ZNorm>(ticks, mu, (i, j), cells, first, steps(i).zip(steps(j)), out);
    };
    let bounds = |a: usize, i: usize, j: usize, states: &[i64], out: Strided| {
        znorm_bounds(&quantized[a], mu, i, j, states, out);
    };
    let find = |row: usize, column: usize, out: &mut Vec<Option<ExactDistance>>| {
        pair_distances::<ZNorm, _>(&ticks, mu, &window, &distance, row, column, out);
    };
    let metric = Dimensions {
        ticks: &ticks,
        window: &window,
        distance: &distance,
        states: &states,
        bounds: &bounds,
    };
    let body = |tile: &Tile, scratch: &mut Scratch<ExactDistance>| {
        // Every tile runs the exact `i64` arithmetic; the forced wide route only turns the
        // screen off.
        let screened = !route::wide(false);
        walk_tile::<ZNorm, _>(&cut, mu, &metric, tile, scratch, screened);
        true
    };
    Ok(profile_with(
        windows,
        mu,
        dimensions.len(),
        &cut,
        threads,
        plan,
        &find,
        &body,
    ))
}

/// The exact z-normalized multi-dimensional profile of `dimensions` (equal-length series, each
/// quantized to its own entry of `scales`), on up to `threads` threads: per window start and per
/// `k`, the nearest window outside the exclusion zone by the mean of the `k + 1` chosen
/// dimensions' distances, ties to the smaller index (STUMPY's `mstump`). `include` lists
/// dimensions that always come first, in the order given; `discords` orders the others farthest
/// first. A dimension whose window has a gap has no distance for the pair and sorts last. Equal
/// on bits to `naive_multi_profile`, whatever `threads` is.
pub fn mstump<'a, S: Into<Series<'a>> + Copy>(
    dimensions: &[S],
    m: NonZeroUsize,
    scales: &[Scale],
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
) -> Result<MultiProfile, MultiError> {
    mstump_report(dimensions, m, scales, include, discords, threads).map(|(profile, _)| profile)
}

/// [`mstump`], and how many independent tiles ran (every one on the exact `i64` arithmetic).
#[doc(hidden)]
pub fn mstump_report<'a, S: Into<Series<'a>> + Copy>(
    dimensions: &[S],
    m: NonZeroUsize,
    scales: &[Scale],
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
) -> Result<(MultiProfile, BatchReport), MultiError> {
    let dimensions: Vec<Series<'a>> = dimensions.iter().map(|&series| series.into()).collect();
    let windows = dimensions
        .first()
        .map_or(0, |series| (series.len() + 1).saturating_sub(m.get()));
    let threads = threads_for_join(windows, windows, threads);
    znorm_profile(&dimensions, m, scales, include, discords, threads, None)
}

/// [`mstump_report`] with the tile shape as parameters, for tests and tuning: bands of
/// `band_rows` rows, tiles of `tile_lanes` lanes (rounded up to a whole number of `chunk_lanes`).
/// Neither they nor `threads` change a bit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn mstump_report_tiled<'a, S: Into<Series<'a>> + Copy>(
    dimensions: &[S],
    m: NonZeroUsize,
    scales: &[Scale],
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> Result<(MultiProfile, BatchReport), MultiError> {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    let dimensions: Vec<Series<'a>> = dimensions.iter().map(|&series| series.into()).collect();
    znorm_profile(
        &dimensions,
        m,
        scales,
        include,
        discords,
        threads,
        Some(plan),
    )
}

/// Whether any dimension's pairs in `tile` need `u128` (or an unrepresentable range, which no
/// pair of the tile then reads): by the range of every tick its rows and columns hold.
fn tile_is_wide(series: &[TickSeries], m: usize, windows: usize, tile: &Tile) -> bool {
    let columns = Shape::Join { n_b: windows }.columns(tile);
    series.iter().any(|dimension| {
        let row_ticks = &dimension.ticks[tile.rows.start..tile.rows.end - 1 + m];
        let column_ticks = &dimension.ticks[columns.start..columns.end - 1 + m];
        match (range_of(row_ticks), range_of(column_ticks)) {
            (Some(a), Some(b)) => {
                let (lowest, highest) = joined(a, b);
                routed_row_width(lowest, highest, m) != RowWidth::Narrow
            }
            _ => false,
        }
    })
}

/// A non-normalized window: the ticks sliding it.
fn tick_window<'a>(ticks: &'a [&'a [i64]], m: usize) -> impl Fn(usize, usize) -> TickStep + 'a {
    move |a: usize, i: usize| TickStep {
        entering: ticks[a][i + m - 1],
        leaving: if i > 0 { ticks[a][i - 1] } else { 0 },
    }
}

/// A non-normalized pair's distance is the residue itself, missing for a window with a gap and for
/// the pairs of a dimension past its first unrepresentable row.
fn residue_distance<'a, R: Residue>(
    series: &'a [TickSeries],
    unrepresentable: &'a [usize],
) -> impl Fn(usize, R, usize, usize) -> Option<SquaredTicks> + 'a {
    move |a: usize, residue: R, i: usize, j: usize| {
        (series[a].complete[i] && series[a].complete[j] && i.max(j) < unrepresentable[a])
            .then(|| SquaredTicks(residue.as_u128()))
    }
}

/// `walk_tile` for the non-normalized metric in residue width `R`. The two closures are where
/// it differs from the z-normalized one.
fn nonnorm_walk<R: Residue>(
    cut: &Cut,
    m: usize,
    series: &[TickSeries],
    unrepresentable: &[usize],
    tile: &Tile,
    scratch: &mut Scratch<SquaredTicks>,
) {
    let ticks: Vec<&[i64]> = series.iter().map(|s| s.ticks.as_slice()).collect();
    let window = tick_window(&ticks, m);
    let distance = residue_distance::<R>(series, unrepresentable);
    let bounds = |a: usize, i: usize, j: usize, states: &[R], out: Strided| {
        let dimension = &series[a];
        let (rows, columns) = (&dimension.complete[i..], &dimension.complete[j..]);
        let unrepresentable = unrepresentable[a].saturating_sub(i.max(j));
        let cells = states.iter().zip(rows).zip(columns).enumerate();
        for (bound, (cell, ((residue, &row), &column))) in out.zip(cells) {
            *bound = if row && column && cell < unrepresentable {
                root_lower(residue.as_u128())
            } else {
                f64::INFINITY
            };
        }
    };
    let states = |a: usize, i: usize, j: usize, cells: usize, out: &mut Vec<R>| {
        let ticks = ticks[a];
        // From a diagonal's second cell on, every window has a tick it just dropped.
        let steps = |i: usize| {
            ticks[i + m..]
                .iter()
                .zip(&ticks[i..])
                .map(|(&entering, &leaving)| TickStep { entering, leaving })
        };
        let first = (window(a, i), window(a, j));
        let rest = steps(i).zip(steps(j));
        diagonal_states::<NonNormalized<R>>(ticks, m, (i, j), cells, first, rest, out);
    };
    let dimensions = Dimensions {
        ticks: &ticks,
        window: &window,
        distance: &distance,
        states: &states,
        bounds: &bounds,
    };
    let screened = !route::forced_wide();
    walk_tile::<NonNormalized<R>, SquaredTicks>(cut, m, &dimensions, tile, scratch, screened);
}

fn nonnorm_on(
    series: &[TickSeries],
    m: NonZeroUsize,
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
) -> (MultiProfile<SquaredTicks>, BatchReport) {
    let mu = m.get();
    let windows = series[0].windows();
    let cut = Cut {
        windows,
        exclusion: crate::exclusion::zone(mu),
        include,
        discords,
    };
    let unrepresentable: Vec<usize> = series
        .iter()
        .map(|dimension| first_unrepresentable_row(dimension, None, mu))
        .collect();
    let ticks: Vec<&[i64]> = series.iter().map(|s| s.ticks.as_slice()).collect();
    let window = tick_window(&ticks, mu);
    let distance = residue_distance::<u128>(series, &unrepresentable);
    let find = |row: usize, column: usize, out: &mut Vec<Option<SquaredTicks>>| {
        pair_distances::<NonNormalized<u128>, SquaredTicks>(
            &ticks, mu, &window, &distance, row, column, out,
        );
    };
    let body = |tile: &Tile, scratch: &mut Scratch<SquaredTicks>| {
        let wide = tile_is_wide(series, mu, windows, tile);
        if wide {
            nonnorm_walk::<u128>(&cut, mu, series, &unrepresentable, tile, scratch);
        } else {
            nonnorm_walk::<u64>(&cut, mu, series, &unrepresentable, tile, scratch);
        }
        wide
    };
    profile_with(windows, mu, series.len(), &cut, threads, plan, &find, &body)
}

fn nonnorm_profile(
    dimensions: &[&[f64]],
    m: NonZeroUsize,
    scale: Scale,
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
    plan: Option<TilePlan>,
) -> Result<(MultiProfile<SquaredTicks>, BatchReport), MultiError> {
    let (_, include) = checked(dimensions.iter().map(|series| series.len()), m, include)?;
    let series: Vec<TickSeries> = dimensions
        .iter()
        .map(|dimension| TickSeries::build(dimension, m, scale))
        .collect();
    Ok(nonnorm_on(&series, m, &include, discords, threads, plan))
}

/// The exact non-normalized multi-dimensional profile of `dimensions` (equal-length series, all
/// quantized to the one `scale`, their distances comparing in one unit), on up to `threads`
/// threads: per window start and per `k`, the nearest window outside the exclusion zone by the
/// mean of the `k + 1` chosen dimensions' Euclidean distances on the raw ticks (STUMPY's
/// `maamp`), in ticks, ties to the smaller index. `include`, `discords` and gaps are as in
/// [`mstump`]. A dimension's pair has no distance either when `m·R² ≥ 2¹²⁸` over the dimension's
/// ticks from its start through the later window. Equal on bits to
/// `naive_multi_profile_nonnorm`, whatever
/// `threads` is.
pub fn maamp(
    dimensions: &[&[f64]],
    m: NonZeroUsize,
    scale: Scale,
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
) -> Result<MultiProfile<SquaredTicks>, MultiError> {
    maamp_report(dimensions, m, scale, include, discords, threads).map(|(profile, _)| profile)
}

/// [`maamp`], and how many independent tiles ran and how many of them ran in `u128` because
/// their own range was too wide for `u64` (or because the wide route is forced, which also takes
/// every cell through the exact path, without the estimate screen).
#[doc(hidden)]
pub fn maamp_report(
    dimensions: &[&[f64]],
    m: NonZeroUsize,
    scale: Scale,
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
) -> Result<(MultiProfile<SquaredTicks>, BatchReport), MultiError> {
    let windows = dimensions
        .first()
        .map_or(0, |series| (series.len() + 1).saturating_sub(m.get()));
    let threads = threads_for_join(windows, windows, threads);
    nonnorm_profile(dimensions, m, scale, include, discords, threads, None)
}

/// [`maamp_report`] with the tile shape as parameters, as [`mstump_report_tiled`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn maamp_report_tiled(
    dimensions: &[&[f64]],
    m: NonZeroUsize,
    scale: Scale,
    include: &[usize],
    discords: bool,
    threads: NonZeroUsize,
    band_rows: NonZeroUsize,
    chunk_lanes: NonZeroUsize,
    tile_lanes: NonZeroUsize,
) -> Result<(MultiProfile<SquaredTicks>, BatchReport), MultiError> {
    let plan = TilePlan::fixed(band_rows, chunk_lanes, tile_lanes);
    nonnorm_profile(dimensions, m, scale, include, discords, threads, Some(plan))
}

mod screen;
#[cfg(test)]
mod tests;
