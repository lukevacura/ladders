//! The constant windows of a series as one class. A window's exact correlation with a constant
//! window is fixed by its own kind alone: one for a constant window, one half for any other finite
//! one. So every member of the class offers a window the same correlation, ties go to the smaller
//! index, and the earliest members a window can see settle the ties every later member would
//! offer it. Derived from the windows' kinds as the constancy rule (and the caller's flags) left
//! them, rebuilt whenever they change, and never stored with a run's state.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Range;

use super::{Window, WindowKind};

/// The constant windows of one series and where its finite windows fall, at one exclusion zone.
pub(crate) struct ConstantClass {
    exclusion: usize,
    /// The constant windows, ascending.
    constants: Vec<u32>,
    /// `finite_before[i]`: how many windows before window `i` are finite, for `i` up to the
    /// window count.
    finite_before: Vec<u32>,
    /// `ordinary_before[i]`: how many windows before window `i` are ordinary (finite and not
    /// constant), for `i` up to the window count.
    ordinary_before: Vec<u32>,
    /// The finite windows, ascending.
    finites: Vec<u32>,
}

/// The [`RowsClass::gap`] of a block with a row nothing settles the offers of.
const NO_GAP: u32 = u32::MAX;

/// The class as a left pass keeping `k` neighbors asks it about whole blocks of pairs: consecutive
/// rows against consecutive columns ([`ConstantClass::blocks`]).
#[derive(Clone, Copy)]
pub(crate) struct ClassBlocks<'a> {
    class: &'a ConstantClass,
    k: usize,
}

/// What the class settles for a block of consecutive rows, whatever columns the block meets
/// ([`ClassBlocks::settles`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowsClass {
    /// The largest of the rows' distances back to the window that settles their offers
    /// ([`ClassBlocks::gap`]): at a lag more than this past the exclusion zone, every row's offer
    /// to a column is beaten there by `k` earlier rows'.
    gap: u32,
    /// Whether some row is ordinary.
    ordinary: bool,
}

impl RowsClass {
    /// The class of the rows of both.
    pub(crate) fn join(self, other: Self) -> Self {
        Self {
            gap: self.gap.max(other.gap),
            ordinary: self.ordinary || other.ordinary,
        }
    }
}

/// What the class settles for one row of a left pass that keeps `k` neighbors, whose lanes are its
/// left neighbors and whose offers go to their right sides, in two words: a row accumulator holds
/// one per band row. Window indices fit in 31 bits, as the class exists only then.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowClass {
    /// A constant lane past this window is beaten by the `k` constant windows up to it, which
    /// the row also sees, by index at the same correlation: it is never among the row's nearest.
    /// `u32::MAX` for none ([`Self::constants_through`]).
    constants_through: u32,
    /// [`Self::settled`] in the low bits, and [`SEEDED`] when the row has a seed.
    settled: u32,
}

/// The bit of [`RowClass::settled`] that marks a seeded row.
const SEEDED: u32 = 1 << 31;

impl RowClass {
    /// A row the class settles nothing for.
    pub(crate) const NONE: Self = Self {
        constants_through: u32::MAX,
        settled: 0,
    };

    /// The last constant lane the row admits, `usize::MAX` for every one.
    pub(crate) fn constants_through(&self) -> usize {
        match self.constants_through {
            u32::MAX => usize::MAX,
            through => through as usize,
        }
    }

    /// For a constant row, the first column it can be among the nearest of: any earlier column
    /// has `k` constant windows between its zone and the row, offering it the same correlation
    /// from smaller indices. For any other row, how many windows before it are finite: a constant
    /// column with `k` of them past its zone has them all at a correlation at least the row's one
    /// half. `0` for a row the class settles nothing for.
    pub(crate) fn settled(&self) -> usize {
        (self.settled & !SEEDED) as usize
    }

    /// A correlation the row's `k`-th nearest is known to reach, `-∞` when none is known: one half
    /// for a finite row that is not constant with `k` constant windows to its left.
    pub(crate) fn seed(&self) -> f64 {
        if self.settled & SEEDED != 0 {
            0.5
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Takes `other`'s seed too: both are known of the same row.
    pub(crate) fn merge_seed(&mut self, other: &Self) {
        self.settled |= other.settled & SEEDED;
    }
}

fn is_constant(window: &Window) -> bool {
    matches!(window.kind, WindowKind::Constant)
}

fn is_finite(window: &Window) -> bool {
    !matches!(window.kind, WindowKind::NonFinite)
}

impl ConstantClass {
    /// The class of `windows` under the exclusion zone `exclusion`. `None` when no window is
    /// constant (the class settles nothing), when there are more windows than 31 bits count,
    /// when there is no room for it, or when the route harness forces it off: the passes then run
    /// without it, which only costs time.
    pub(crate) fn of(windows: &[Window], exclusion: usize) -> Option<Self> {
        let natural = windows.len() < SEEDED as usize && windows.iter().any(is_constant);
        if !crate::route::constant_class(natural) {
            return None;
        }
        let mut constants = Vec::new();
        constants
            .try_reserve_exact(windows.iter().filter(|w| is_constant(w)).count())
            .ok()?;
        let mut finite_before = Vec::new();
        finite_before.try_reserve_exact(windows.len() + 1).ok()?;
        let mut ordinary_before = Vec::new();
        ordinary_before.try_reserve_exact(windows.len() + 1).ok()?;
        let mut finites = Vec::new();
        finites
            .try_reserve_exact(windows.iter().filter(|w| is_finite(w)).count())
            .ok()?;
        let mut ordinary = 0u32;
        for (i, window) in windows.iter().enumerate() {
            finite_before.push(finites.len() as u32);
            ordinary_before.push(ordinary);
            match window.kind {
                WindowKind::Constant => constants.push(i as u32),
                WindowKind::Valid { .. } => ordinary += 1,
                WindowKind::NonFinite => continue,
            }
            finites.push(i as u32);
        }
        finite_before.push(finites.len() as u32);
        ordinary_before.push(ordinary);
        Some(Self {
            exclusion,
            constants,
            finite_before,
            ordinary_before,
            finites,
        })
    }

    /// The class as a left pass keeping `k` neighbors asks it about whole blocks of pairs; `None`
    /// when the route harness forces the pass to visit every block.
    pub(crate) fn blocks(&self, k: NonZeroUsize) -> Option<ClassBlocks<'_>> {
        crate::route::constant_blocks().then_some(ClassBlocks {
            class: self,
            k: k.get(),
        })
    }

    /// How many of `windows` are ordinary: finite and not constant.
    fn ordinary(&self, windows: &Range<usize>) -> usize {
        (self.ordinary_before[windows.end] - self.ordinary_before[windows.start]) as usize
    }

    /// The first constant window at or after `window`, as its rank among them.
    fn rank_from(&self, window: usize) -> usize {
        self.constants.partition_point(|&c| (c as usize) < window)
    }

    /// The first window past `column`'s exclusion zone on its right, or the window count.
    fn past_zone(&self, column: usize) -> usize {
        column
            .saturating_add(self.exclusion + 1)
            .min(self.finite_before.len() - 1)
    }

    /// The two earliest constant windows that lie outside each other's exclusion zone, `None` when
    /// there are none: a pair of correlation one, the best a pair can have. If any two constant
    /// windows lie outside each other's zone, the first constant window and the first one past
    /// its zone do.
    pub(crate) fn top_pair(&self) -> Option<(usize, usize)> {
        let first = *self.constants.first()? as usize;
        let past = first.saturating_add(self.exclusion + 1);
        let at = self.constants.partition_point(|&c| (c as usize) < past);
        Some((first, *self.constants.get(at)? as usize))
    }

    /// What row `row` of a pass keeping `k` neighbors takes from the class.
    pub(crate) fn row(&self, row: usize, k: NonZeroUsize) -> RowClass {
        let k = k.get();
        let constants_through = self.constants.get(k - 1).copied().unwrap_or(u32::MAX);
        let rank = self.rank_from(row);
        if self.constants.get(rank).is_some_and(|&c| c as usize == row) {
            let offers_from = rank.checked_sub(k).map_or(0, |earlier| {
                self.constants[earlier]
                    .saturating_sub(u32::try_from(self.exclusion).unwrap_or(u32::MAX))
            });
            return RowClass {
                constants_through,
                settled: offers_from,
            };
        }
        let finite = self.finite_before[row + 1] > self.finite_before[row];
        let sees_k = (constants_through as usize) < row.saturating_sub(self.exclusion);
        RowClass {
            constants_through,
            settled: self.finite_before[row] | if finite && sees_k { SEEDED } else { 0 },
        }
    }

    /// Calls `cut(column, from)` for each constant column of `columns` with the
    /// [`RowClass::settled`] from which a row that is not constant has `k` finite windows
    /// between the column's zone and itself, all at a correlation at least its own with the column
    /// and from smaller indices: such a row is never among the column's `k` nearest.
    pub(crate) fn column_cuts(
        &self,
        columns: Range<usize>,
        k: NonZeroUsize,
        mut cut: impl FnMut(usize, usize),
    ) {
        for &column in &self.constants[self.rank_from(columns.start)..] {
            let column = column as usize;
            if column >= columns.end {
                break;
            }
            let past = self.finite_before[self.past_zone(column)] as usize;
            cut(column, past.saturating_add(k.get()));
        }
    }

    /// Calls `seed(column, rho)` for each column of `columns` whose nearest `slots(column)` right
    /// neighbors are known to reach correlation `rho`: one for a constant column with that many
    /// constant windows past its zone, else one half for a constant column, or for a finite one
    /// with that many constant windows past its zone. A column with no slot is skipped.
    pub(crate) fn column_seeds(
        &self,
        columns: Range<usize>,
        slots: impl Fn(usize) -> usize,
        mut seed: impl FnMut(usize, f64),
    ) {
        let mut own = self.rank_from(columns.start);
        let mut zone = self.rank_from(self.past_zone(columns.start));
        for column in columns {
            let slots = slots(column);
            while own < self.constants.len() && (self.constants[own] as usize) < column {
                own += 1;
            }
            let past = self.past_zone(column);
            while zone < self.constants.len() && (self.constants[zone] as usize) < past {
                zone += 1;
            }
            if slots == 0 {
                continue;
            }
            let constant = self
                .constants
                .get(own)
                .is_some_and(|&c| c as usize == column);
            let reached = self.constants.len() - zone >= slots;
            match (constant, reached) {
                (true, true) => seed(column, 1.0),
                (true, false) | (false, true) => seed(column, 0.5),
                (false, false) => {}
            }
        }
    }
}

impl ClassBlocks<'_> {
    /// How far before `row` the window that settles its offers lies: the `k`-th constant window
    /// before a constant row, the `k`-th finite window before an ordinary one. Every column more
    /// than that past its exclusion zone before the row has `k` rows between its zone and the row
    /// that offer it at least the row's correlation from smaller indices, so the row is not among
    /// its `k` nearest. [`NO_GAP`] when fewer than `k` such windows precede the row, and `0` for a
    /// non-finite row, which offers nothing.
    fn gap(&self, row: usize) -> u32 {
        let class = self.class;
        let (finite, ordinary) = (class.finite_before[row], class.ordinary_before[row]);
        if class.finite_before[row + 1] == finite {
            return 0;
        }
        let settling = if class.ordinary_before[row + 1] > ordinary {
            (finite as usize)
                .checked_sub(self.k)
                .map(|kth| class.finites[kth])
        } else {
            ((finite - ordinary) as usize)
                .checked_sub(self.k)
                .map(|kth| class.constants[kth])
        };
        settling.map_or(NO_GAP, |window| row as u32 - window)
    }

    /// What the class settles for the block of `rows`, read once for every part of `columns` it
    /// is then asked about; `None` when it settles no pair of the block with them: the block
    /// pairs only ordinary windows, or some row's offers are settled at no lag.
    #[inline]
    #[cfg(test)]
    pub(crate) fn rows(&self, rows: Range<usize>, columns: &Range<usize>) -> Option<RowsClass> {
        let ordinary = self.class.ordinary(&rows) > 0;
        if ordinary && self.class.ordinary(columns) == columns.len() {
            return None;
        }
        let gap = rows.map(|row| self.gap(row)).max()?;
        (gap != NO_GAP).then_some(RowsClass { gap, ordinary })
    }

    /// What the class settles for the one row `row`: [`RowsClass::join`] of consecutive rows'
    /// is the class of the block, so a pass that asks about many blocks reads each row once.
    pub(crate) fn row_class(&self, row: usize) -> RowsClass {
        let class = self.class;
        RowsClass {
            gap: self.gap(row),
            ordinary: class.ordinary_before[row + 1] > class.ordinary_before[row],
        }
    }

    /// Calls `run(slots)` for each maximal run of `slots` that the class does not settle for the
    /// block of `rows` (whose [`RowsClass`] is `class`, the join of its rows'
    /// [`Self::row_class`]es), ascending: a self-join pass over `rows` computes exactly those.
    /// Slot `s` pairs row `r` with column `r - exclusion - 1 - s`, so a slot is a lag, the same for
    /// every row. The slots go in parts of `part` slots, cut at its multiples (and at the range's
    /// ends); a part is left out whole or not at all. Counts each part left out.
    ///
    /// This is the one question both passes ask of the class: the plain pass of each (rows, chunk)
    /// and the one-loop route of each (row block, survivor run) call it and compute what it names.
    #[inline]
    pub(crate) fn unsettled_in(
        &self,
        class: RowsClass,
        rows: &Range<usize>,
        slots: &Range<usize>,
        part: usize,
        mut run: impl FnMut(Range<usize>),
    ) {
        let windows = self.class.finite_before.len() - 1;
        let lag = self.class.exclusion + 1;
        // Some row's offers are settled at no lag, or every lag of the range is within the rows'
        // gap: no part is settled.
        if slots.is_empty()
            || class.gap == NO_GAP
            || lag + slots.end <= (class.gap as usize).saturating_add(self.class.exclusion) + 1
        {
            if !slots.is_empty() {
                run(slots.clone());
            }
            return;
        }
        let columns_of = |slots: &Range<usize>| {
            let hi = rows.end.saturating_sub(lag + slots.start).min(windows);
            let lo = (rows.start + 1).saturating_sub(lag + slots.end).min(hi);
            lo..hi
        };
        let mut kept: Option<usize> = None;
        let (mut left_out, mut pairs) = (0, 0);
        let mut at = slots.start;
        while at < slots.end {
            let end = ((at / part + 1) * part).min(slots.end);
            let here = at..end;
            if self.settles(&class, &columns_of(&here), lag + at) {
                left_out += 1;
                pairs += rows.len() * here.len();
                if let Some(start) = kept.take() {
                    run(start..at);
                }
            } else if kept.is_none() {
                kept = Some(at);
            }
            at = end;
        }
        if let Some(start) = kept {
            run(start..slots.end);
        }
        if left_out > 0 {
            crate::route::count_class_blocks(left_out, pairs);
        }
    }

    /// Whether the class settles every pair of the block of `rows` with a column of `columns`,
    /// each at a lag of `lag` or more: no such pair is among a row's `k` nearest left neighbors
    /// or a column's `k` nearest right ones, so the pass may leave them all out. That is so when
    /// all three hold. The block pairs no two ordinary windows, so each of its correlations is
    /// fixed by the kinds of its two windows. `lag` is more than the rows' gap plus the exclusion
    /// zone, so each row's offer is beaten at each column ([`Self::gap`]). And `columns` hold
    /// none of the first `k` constant windows and none of the first `k` finite ones: any other
    /// column has `k` windows before it that a row seeing it sees too, at a correlation at least
    /// as large and from smaller indices, the constant windows for a constant column and the
    /// finite ones for an ordinary column, which only a constant row of the block pairs with.
    #[inline]
    pub(crate) fn settles(&self, rows: &RowsClass, columns: &Range<usize>, lag: usize) -> bool {
        let class = self.class;
        // The cheapest of the three conditions first: most parts of a long series lie at lags the
        // rows' offers are not yet beaten at.
        if lag <= (rows.gap as usize).saturating_add(class.exclusion) {
            return false;
        }
        let before = |window: usize| {
            let finite = class.finite_before[window] as usize;
            (finite, finite - class.ordinary_before[window] as usize)
        };
        let (first, end) = (before(columns.start), before(columns.end));
        // One of the first `k` of a kind is in the columns: fewer than `k` lie before them, and
        // one lies in them.
        let nearest =
            (first.0 < self.k && end.0 > first.0) || (first.1 < self.k && end.1 > first.1);
        let ordinary = rows.ordinary && class.ordinary(columns) > 0;
        !nearest && !ordinary
    }
}

/// A window of `a` that holds a sample missing: it offers and takes nothing.
const ROW_NON_FINITE: u8 = 0;
/// A window of `a` that is constant.
const ROW_CONSTANT: u8 = 1;
/// A window of `a` that is finite and not constant.
const ROW_ORDINARY: u8 = 2;

/// The class as an AB-join asks it about whole blocks of pairs: consecutive rows of `a` against
/// the columns of `b` a part of slots meets ([`JoinClass::of`]). The join keeps the `k` nearest
/// `b` windows of each `a` window and nothing for a `b` window, so only a row's own offers
/// matter, and there is no exclusion zone and no left side: a row's partners are all of `b`.
pub(crate) struct JoinClass {
    /// The windows of `b`.
    n_b: usize,
    /// The kind of each window of `a`.
    rows: Vec<u8>,
    /// `ordinary_before[i]`: how many windows of `b` before window `i` are ordinary, for `i` up to
    /// the window count.
    ordinary_before: Vec<u32>,
    /// The first `k` constant windows of `b`, ascending.
    first_constants: Vec<u32>,
    /// The first `k` finite windows of `b`, ascending.
    first_finites: Vec<u32>,
}

/// What a block of consecutive rows of `a` needs of the columns it meets ([`JoinClass::row_class`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct JoinRows {
    /// Whether some row is ordinary.
    ordinary: bool,
    /// Whether some row is constant.
    constant: bool,
}

impl JoinRows {
    /// The class of the rows of both.
    pub(crate) fn join(self, other: Self) -> Self {
        Self {
            ordinary: self.ordinary || other.ordinary,
            constant: self.constant || other.constant,
        }
    }
}

impl JoinClass {
    /// The class of a join of `a`'s windows against `b`'s that keeps `k` neighbors per row. A
    /// window's correlation with a constant window is fixed by its own kind (one for a constant
    /// window, one half for any other finite one), and ties go to the smaller index, so a row
    /// needs only these columns of `b`, and any others it meets are settled:
    ///
    /// - an ordinary row: every ordinary column (its correlation is its own), and the first `k`
    ///   constant ones. A later constant column offers one half, as those `k` already do, from
    ///   smaller indices.
    /// - a constant row: the first `k` constant columns and the first `k` finite ones. Every
    ///   finite column offers it one half or one, so a later constant column is beaten by the `k`
    ///   constant ones before it, and a later ordinary one by the `k` finite ones before it.
    /// - a non-finite row: none.
    ///
    /// `None` when no window of either side is constant (the class settles nothing), when there
    /// are more windows than 31 bits count, when there is no room for it, or when the route
    /// harness forces the passes to visit every block: they then compute every pair, which only
    /// costs time.
    pub(crate) fn of(a: &[Window], b: &[Window], k: NonZeroUsize) -> Option<Self> {
        let natural = a.len().max(b.len()) < SEEDED as usize
            && (a.iter().any(is_constant) || b.iter().any(is_constant));
        if !crate::route::constant_class(natural) || !crate::route::constant_blocks() {
            return None;
        }
        let kept = k.get().min(b.len());
        let mut rows = Vec::new();
        rows.try_reserve_exact(a.len()).ok()?;
        rows.extend(a.iter().map(|window| match window.kind {
            WindowKind::NonFinite => ROW_NON_FINITE,
            WindowKind::Constant => ROW_CONSTANT,
            WindowKind::Valid { .. } => ROW_ORDINARY,
        }));
        let mut ordinary_before = Vec::new();
        ordinary_before.try_reserve_exact(b.len() + 1).ok()?;
        let (mut first_constants, mut first_finites) = (Vec::new(), Vec::new());
        first_constants.try_reserve_exact(kept).ok()?;
        first_finites.try_reserve_exact(kept).ok()?;
        let mut ordinary = 0u32;
        for (i, window) in b.iter().enumerate() {
            ordinary_before.push(ordinary);
            match window.kind {
                WindowKind::Constant if first_constants.len() < kept => {
                    first_constants.push(i as u32)
                }
                WindowKind::Valid { .. } => ordinary += 1,
                _ => {}
            }
            if is_finite(window) && first_finites.len() < kept {
                first_finites.push(i as u32);
            }
        }
        ordinary_before.push(ordinary);
        Some(Self {
            n_b: b.len(),
            rows,
            ordinary_before,
            first_constants,
            first_finites,
        })
    }

    /// What the class needs for the one row `row` of `a`: [`JoinRows::join`] of consecutive rows'
    /// is the class of the block, so a pass that asks about many blocks reads each row once.
    pub(crate) fn row_class(&self, row: usize) -> JoinRows {
        let kind = self.rows[row];
        JoinRows {
            ordinary: kind == ROW_ORDINARY,
            constant: kind == ROW_CONSTANT,
        }
    }

    /// Whether any of the ascending `windows` lies in `columns`.
    fn holds(windows: &[u32], columns: &Range<usize>) -> bool {
        let at = windows.partition_point(|&w| (w as usize) < columns.start);
        windows.get(at).is_some_and(|&w| (w as usize) < columns.end)
    }

    /// Whether the class settles every pair of a block of rows of class `rows` with a column of
    /// `columns`: none is among its row's `k` nearest, so a pass may leave them all out.
    pub(crate) fn settles(&self, rows: JoinRows, columns: &Range<usize>) -> bool {
        if rows.ordinary
            && (self.ordinary_before[columns.end] > self.ordinary_before[columns.start]
                || Self::holds(&self.first_constants, columns))
        {
            return false;
        }
        !(rows.constant
            && (Self::holds(&self.first_constants, columns)
                || Self::holds(&self.first_finites, columns)))
    }

    /// Calls `run(slots)` for each maximal run of `slots` that the class does not settle for the
    /// block of `rows` (whose class is `class`, the join of its rows' [`Self::row_class`]es),
    /// ascending: a join pass over `rows` computes exactly those. Slot `s` pairs row `r` with
    /// column `r + n_b - 1 - s`, which is a pair only inside `b`. The slots go in parts of `part`
    /// slots, cut at its multiples (and at the range's ends); a part is left out whole or not at
    /// all, and a part that meets no column of `b` is not asked about. Counts each part left out.
    ///
    /// The join's counterpart of [`ClassBlocks::unsettled_in`], the one question both passes ask.
    #[inline]
    pub(crate) fn unsettled_in(
        &self,
        class: JoinRows,
        rows: &Range<usize>,
        slots: &Range<usize>,
        part: usize,
        mut run: impl FnMut(Range<usize>),
    ) {
        let n_b = self.n_b;
        let columns_of = |slots: &Range<usize>| {
            let hi = (rows.end + n_b).saturating_sub(1 + slots.start).min(n_b);
            let lo = (rows.start + n_b).saturating_sub(slots.end).min(hi);
            lo..hi
        };
        let mut kept: Option<usize> = None;
        let (mut left_out, mut pairs) = (0, 0);
        let mut at = slots.start;
        while at < slots.end {
            let end = ((at / part + 1) * part).min(slots.end);
            let here = at..end;
            let columns = columns_of(&here);
            if !columns.is_empty() && self.settles(class, &columns) {
                left_out += 1;
                pairs += rows
                    .clone()
                    .map(|row| {
                        let (lo, hi) = (here.start.max(row), here.end.min(row + n_b));
                        hi.saturating_sub(lo)
                    })
                    .sum::<usize>();
                if let Some(start) = kept.take() {
                    run(start..at);
                }
            } else if kept.is_none() {
                kept = Some(at);
            }
            at = end;
        }
        if let Some(start) = kept {
            run(start..slots.end);
        }
        if left_out > 0 {
            crate::route::count_class_blocks(left_out, pairs);
        }
    }
}

#[cfg(test)]
mod tests;
