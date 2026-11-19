//! The exact cross term of two windows, `QT = Σₖ aₖ·bₖ`, from their ticks: the one place a
//! diagonal's state is started, restarted (a tile's first row, a re-anchored cell's first state, a
//! stream's switch to i64 arithmetic, a checkpoint restore) or resumed after a skipped run.
//!
//! Every form is one wrapping sum of products, so it is exact mod 2⁶⁴ in any order: the state it
//! returns is, bit for bit, the one the recurrence reaches along the diagonal. A side's element
//! type is its arithmetic: raw `i64` ticks give the unscaled `QT` of the exact path; `i32` offsets
//! from an anchor, or raw ticks read as `Offsets` from one, give the narrow path's `m·QT`.
//!
//! What computes a related quantity elsewhere, and why it stays there:
//! - `narrow::reanchor`, `narrow::reanchor_pair`, `narrow::absolute_qt` and `narrow::narrow_qt_m`
//!   move a state that already exists to another anchor, or between the narrow and the i64
//!   arithmetic, in a constant number of operations from the two windows' sums; they read no
//!   ticks, and they live with the anchors they move between.
//! - `spill::dot` sums exactly in a wider integer, without wrapping, for the cold route's pairs
//!   whose cross term does not fit `i64`.
//! - The non-normalized metric's `start` sums squared differences, not products.

use crate::narrow::{self, Anchor};
use core::num::NonZeroUsize;

/// `Σ aₖ·bₖ` over two windows of raw ticks, wrapping: the exact path's `QT`.
#[inline]
pub fn wide(a: &[i64], b: &[i64]) -> i64 {
    sum_of_products(a, b, |x, y| x.wrapping_mul(y))
}

/// Whether the narrow sums take their AVX2 body: the CPU has AVX2 and no test forced the scalar
/// sum. The wide (`i64`) sums stay scalar, as AVX2 has no 64-bit multiply.
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_route() -> bool {
    crate::route::avx2_start_sum(crate::kernel_avx2::detected())
}

/// `m·Σ aₖ·bₖ` over two windows of offsets from their anchors, wrapping: the narrow path's `m·QT`.
#[inline]
pub fn narrow(a: &[i32], b: &[i32], m: usize) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if avx2_route() {
        // SAFETY: `avx2_route` says the running CPU has AVX2.
        return unsafe { crate::kernel_avx2::narrow_dot(a, b) }.wrapping_mul(m as i64);
    }
    sum_of_products(a, b, |x, y| i64::from(x).wrapping_mul(i64::from(y))).wrapping_mul(m as i64)
}

#[inline(always)]
fn sum_of_products<T: Copy>(a: &[T], b: &[T], product: impl Fn(T, T) -> i64) -> i64 {
    a.iter()
        .zip(b)
        .fold(0i64, |sum, (&x, &y)| sum.wrapping_add(product(x, y)))
}

/// Raw ticks read as offsets from `anchor`, each already known to fit it at window length `m`.
#[derive(Clone, Copy)]
pub(crate) struct Offsets<'a> {
    pub ticks: &'a [i64],
    pub anchor: Anchor,
    pub m: NonZeroUsize,
}

/// Lanes [`cross_lanes_of`] accumulates together, in registers.
const LANE_BLOCK: usize = 8;

/// `states[l] = Σₖ a[k]·b[l + k]` over the `a.len()` ticks of `a`, wrapping, for every lane `l`
/// of `states` (`b` holds `states.len() + a.len() − 1` ticks): the cross terms of one window of
/// `a` with consecutive windows of `b`. Blocks of [`LANE_BLOCK`] lanes go together, the lanes
/// inner and their sums in registers, so each tick of `a` is read once per block and the
/// products vectorize; the lanes past the last block take [`sum_of_products`]. Every lane is the
/// same wrapping sum as [`sum_of_products`], in another order, so the same bits.
#[inline(always)]
fn cross_lanes_of<T: Copy>(
    a: &[T],
    b: &[T],
    states: &mut [i64],
    product: impl Fn(T, T) -> i64 + Copy,
) {
    let mut blocks = states.chunks_exact_mut(LANE_BLOCK);
    let mut first = 0;
    for block in &mut blocks {
        let mut sums = [0i64; LANE_BLOCK];
        let b = &b[first..][..a.len() + LANE_BLOCK - 1];
        for (k, &x) in a.iter().enumerate() {
            let ys = &b[k..k + LANE_BLOCK];
            for (sum, &y) in sums.iter_mut().zip(ys) {
                *sum = sum.wrapping_add(product(x, y));
            }
        }
        block.copy_from_slice(&sums);
        first += LANE_BLOCK;
    }
    for (l, state) in blocks.into_remainder().iter_mut().enumerate() {
        *state = sum_of_products(a, &b[first + l..][..a.len()], product);
    }
}

/// One side of a pair's windows, in the arithmetic its diagonal state is kept in.
pub trait Side: Copy {
    /// How many ticks the side holds.
    fn ticks(self) -> usize;
    /// The cross term of the `len` ticks from `i` of `self` and from `j` of `other`.
    fn cross(self, i: usize, other: Self, j: usize, len: usize, m: usize) -> i64;
    /// [`Self::cross`] of the `m` ticks from `i` of `self` with the `m` ticks from `j + l` of
    /// `other`, into `states[l]` for every lane `l`.
    fn cross_lanes(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]);
    /// Advances every lane `l` of `states` one step along its diagonal, by the kernel's own
    /// recurrence: the pair's windows leave ticks `i` of `self` and `j + l` of `other`, and enter
    /// the ticks `m` past them. Returns `false`, touching nothing, where the arithmetic's step
    /// does not hold the ticks (an offset whose `m`-scaled value leaves `i32`, which no narrow
    /// side holds).
    fn step(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) -> bool;
}

/// The row's entering and leaving offsets `(enter, leave)` scaled by `m`, as
/// [`crate::row::narrow_row_scalars`] scales them for `narrow::advance`; `None` where one leaves
/// `i32`, which no narrow side's offset does. Computed in `i32`, so the step's products stay
/// `i32 × i32` and vectorize.
#[inline(always)]
fn scaled_row((enter, leave): (i32, i32), m: usize) -> Option<(i32, i32)> {
    let m = i32::try_from(m).ok()?;
    Some((enter.checked_mul(m)?, leave.checked_mul(m)?))
}

/// One narrow step of every lane, `narrow::advance` with the row's offsets scaled by `m`
/// ([`scaled_row`]) and the columns' entering and leaving offsets `(te, tl)` unscaled.
#[inline(always)]
fn narrow_step(row: (i32, i32), m: usize, (te, tl): (&[i32], &[i32]), states: &mut [i64]) -> bool {
    let Some((me, ml)) = scaled_row(row, m) else {
        return false;
    };
    for ((q, &e), &l) in states.iter_mut().zip(te).zip(tl) {
        *q = narrow::advance(*q, me, ml, e, l);
    }
    true
}

impl Side for &[i64] {
    fn ticks(self) -> usize {
        self.len()
    }
    #[inline]
    fn cross(self, i: usize, other: Self, j: usize, len: usize, _m: usize) -> i64 {
        wide(&self[i..][..len], &other[j..][..len])
    }
    #[inline]
    fn cross_lanes(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) {
        let lanes = states.len();
        cross_lanes_of(
            &self[i..][..m],
            &other[j..][..lanes + m - 1],
            states,
            |x, y| x.wrapping_mul(y),
        );
    }
    #[inline]
    fn step(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) -> bool {
        let lanes = states.len();
        let (enter, leave) = (self[i + m], self[i]);
        let (te, tl) = (&other[j + m..][..lanes], &other[j..][..lanes]);
        for (q, (&e, &l)) in states.iter_mut().zip(te.iter().zip(tl)) {
            *q = crate::row::advance_lane(*q, enter, leave, e, l);
        }
        true
    }
}

impl Side for &[i32] {
    fn ticks(self) -> usize {
        self.len()
    }
    #[inline]
    fn cross(self, i: usize, other: Self, j: usize, len: usize, m: usize) -> i64 {
        narrow(&self[i..][..len], &other[j..][..len], m)
    }
    #[inline]
    fn cross_lanes(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) {
        let lanes = states.len();
        let (a, b) = (&self[i..][..m], &other[j..][..lanes + m - 1]);
        #[cfg(target_arch = "x86_64")]
        if avx2_route() {
            // SAFETY: `avx2_route` says the running CPU has AVX2.
            unsafe { crate::kernel_avx2::narrow_lanes(a, b, states) };
            for state in states.iter_mut() {
                *state = state.wrapping_mul(m as i64);
            }
            return;
        }
        cross_lanes_of(a, b, states, |x, y| i64::from(x).wrapping_mul(i64::from(y)));
        for state in states.iter_mut() {
            *state = state.wrapping_mul(m as i64);
        }
    }
    #[inline]
    fn step(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) -> bool {
        let lanes = states.len();
        narrow_step(
            (self[i + m], self[i]),
            m,
            (&other[j + m..][..lanes], &other[j..][..lanes]),
            states,
        )
    }
}

impl Offsets<'_> {
    fn offset(self, k: usize) -> i32 {
        narrow::tick_offset(self.anchor, self.m, self.ticks[k])
    }
}

impl Side for Offsets<'_> {
    fn ticks(self) -> usize {
        self.ticks.len()
    }
    #[inline]
    fn cross(self, i: usize, other: Self, j: usize, len: usize, m: usize) -> i64 {
        sum_of_products(&self.ticks[i..][..len], &other.ticks[j..][..len], |x, y| {
            i64::from(narrow::tick_offset(self.anchor, self.m, x))
                .wrapping_mul(i64::from(narrow::tick_offset(other.anchor, other.m, y)))
        })
        .wrapping_mul(m as i64)
    }
    fn cross_lanes(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) {
        for (l, state) in states.iter_mut().enumerate() {
            *state = self.cross(i, other, j + l, m, m);
        }
    }
    fn step(self, i: usize, other: Self, j: usize, m: usize, states: &mut [i64]) -> bool {
        let Some((me, ml)) = scaled_row((self.offset(i + m), self.offset(i)), m) else {
            return false;
        };
        for (l, q) in states.iter_mut().enumerate() {
            *q = narrow::advance(*q, me, ml, other.offset(j + m + l), other.offset(j + l));
        }
        true
    }
}

/// The pairs of a row side's windows with a column side's: `rows` starts at row window
/// `first_row`, `cols` at column window `first_col`, so a pair is named by its windows' indices
/// in the whole series, whatever part of it the sides hold.
#[derive(Clone, Copy)]
pub struct CrossTerms<S> {
    rows: S,
    first_row: usize,
    cols: S,
    first_col: usize,
    m: usize,
}

impl<S: Side> CrossTerms<S> {
    pub fn new(rows: S, first_row: usize, cols: S, first_col: usize, m: usize) -> Self {
        Self {
            rows,
            first_row,
            cols,
            first_col,
            m,
        }
    }

    /// The self-join of `ticks`, both sides from window 0.
    pub fn of(ticks: S, m: usize) -> Self {
        Self::new(ticks, 0, ticks, 0, m)
    }

    /// The cross term of row window `i` with column window `j`, both held by the sides.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> i64 {
        let m = self.m;
        S::cross(
            self.rows,
            i - self.first_row,
            self.cols,
            j - self.first_col,
            m,
            m,
        )
    }

    /// The state a diagonal through row window 0 and column window `j` carries before row 0, so
    /// that one step with a leaving row tick of zero reaches pair `(0, j)`: the cross term of all
    /// but the windows' last ticks. Row window 0 must be the sides' first.
    #[inline]
    pub(crate) fn before_first_row(&self, j: usize) -> i64 {
        debug_assert_eq!(self.first_row, 0, "row window 0 is the sides' first");
        let m = self.m;
        S::cross(self.rows, 0, self.cols, j - self.first_col, m - 1, m)
    }

    /// The resume entry: the state of the diagonal through pair `from = (i, j)` after `run`
    /// steps, the pair `(i + run, j + run)`, computed directly instead of advancing through the
    /// run. `None` (never a panic) when `from` is before the sides or the diagonal leaves them
    /// within the run.
    pub fn resume(&self, (i, j): (usize, usize), run: usize) -> Option<i64> {
        let m = self.m;
        let row = i.checked_sub(self.first_row)?.checked_add(run)?;
        let col = j.checked_sub(self.first_col)?.checked_add(run)?;
        let fits =
            row.checked_add(m)? <= self.rows.ticks() && col.checked_add(m)? <= self.cols.ticks();
        fits.then(|| S::cross(self.rows, row, self.cols, col, m, m))
    }

    /// Where a resume is forced at row `due` (a test-only route), replaces every state of
    /// `states`, lane `l` carrying pair `(row − 1, first + l − 1)` into row `row`, by its exact
    /// cross term ([`Self::resume_lanes`]), as a diagonal resumes after a skipped run of any
    /// length. Does nothing where none is forced, which is everywhere outside tests.
    #[inline(always)]
    pub(crate) fn force_resume(
        &self,
        due: usize,
        (row, first): (usize, usize),
        states: &mut [i64],
    ) {
        if crate::route::resume_due(due) {
            crate::route::count_resumes(self.resume_lanes((row, first), states));
        }
    }

    /// The lanes `l` of `lanes` lanes whose pair `(i, j + l)` the sides hold, as a range.
    fn held_lanes(&self, i: usize, j: usize, lanes: usize) -> core::ops::Range<usize> {
        let m = self.m;
        let row_held = i
            .checked_sub(self.first_row)
            .and_then(|row| row.checked_add(m))
            .is_some_and(|end| end <= self.rows.ticks());
        // Lane `l` is held when `first_col ≤ j + l` and `j + l − first_col + m ≤ ticks`.
        let lo = self.first_col.saturating_sub(j).min(lanes);
        let hi = (self.first_col + self.cols.ticks())
            .checked_sub(m)
            .and_then(|last| (last + 1).checked_sub(j))
            .map_or(0, |end| end.min(lanes));
        if row_held && lo < hi { lo..hi } else { 0..0 }
    }

    /// Replaces every state of `states`, lane `l` carrying pair `(row − 1, first + l − 1)` into
    /// row `row`, by that pair's exact cross term, [`Self::resume`]'s value whatever run it
    /// resumes after: how a diagonal resumes after a skipped run. The lanes go together, one row
    /// window against consecutive column windows with the lanes inner, so the products vectorize
    /// ([`Side::cross_lanes`]); the same wrapping sum in another order, so the same bits. Returns
    /// how many states it replaced; a lane whose pair the sides do not hold is left as it was.
    pub(crate) fn resume_lanes(&self, (row, first): (usize, usize), states: &mut [i64]) -> usize {
        let (Some(i), Some(j)) = (row.checked_sub(1), first.checked_sub(1)) else {
            return 0;
        };
        let held = self.held_lanes(i, j, states.len());
        if held.is_empty() {
            return 0;
        }
        S::cross_lanes(
            self.rows,
            i - self.first_row,
            self.cols,
            j + held.start - self.first_col,
            self.m,
            &mut states[held.clone()],
        );
        held.len()
    }

    /// Advances every state of `states` through the `run` steps a skip left it behind by, by the
    /// kernel's exact recurrence ([`Side::step`], no ranking): lane `l` carries pair
    /// `(row − 1 − run, first + l − 1 − run)` and is left carrying `(row − 1, first + l − 1)` into
    /// row `row`. Returns `false` when the sides do not hold every tick of every lane's run (a
    /// run that starts before the sides, as a re-anchored cell's rebuilt sides do, or ends past
    /// them), or the arithmetic refuses a step; the states are then to be resumed.
    pub(crate) fn step_lanes(
        &self,
        run: usize,
        (row, first): (usize, usize),
        states: &mut [i64],
    ) -> bool {
        let m = self.m;
        let lanes = states.len();
        let held = || {
            let i = row
                .checked_sub(1)?
                .checked_sub(run)?
                .checked_sub(self.first_row)?;
            let j = first
                .checked_sub(1)?
                .checked_sub(run)?
                .checked_sub(self.first_col)?;
            let row_end = i.checked_add(run)?.checked_add(m)?;
            let col_end = j.checked_add(run)?.checked_add(lanes)?.checked_add(m - 1)?;
            (row_end <= self.rows.ticks() && col_end <= self.cols.ticks()).then_some((i, j))
        };
        let Some((i, j)) = held() else {
            return false;
        };
        (0..run).all(|t| S::step(self.rows, i + t, self.cols, j + t, m, states))
    }

    /// The step-or-resume rule, the one place a skipped run's states catch up: a run the planner
    /// steps through at this `m` ([`crate::batch::steps_through`], shorter than this target's
    /// [`crate::batch::break_even`]) steps ([`Self::step_lanes`]), any other run, or one the sides
    /// cannot step, resumes ([`Self::resume_lanes`]). Both are exact, so the states are the
    /// same bits either way; only the work differs. The test-only route can force either branch.
    pub(crate) fn advance_or_resume(
        &self,
        run: usize,
        at: (usize, usize),
        states: &mut [i64],
    ) -> CatchUp {
        if run != UNSTARTED
            && crate::route::catch_up_steps(crate::batch::steps_through(run, self.m))
            && self.step_lanes(run, at, states)
        {
            return CatchUp {
                resumed: 0,
                stepped: states.len(),
                steps: states.len() * run,
            };
        }
        CatchUp {
            resumed: self.resume_lanes(at, states),
            stepped: 0,
            steps: 0,
        }
    }
}

/// The run a lane that a tile has not started yet is caught up through: no step count reaches its
/// state from nowhere, so [`CrossTerms::advance_or_resume`] always resumes it.
pub(crate) const UNSTARTED: usize = usize::MAX;

/// How a skipped run's states caught up: how many were resumed, how many stepped through, and the
/// steps those took in all (lanes stepped times the run).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CatchUp {
    pub(crate) resumed: usize,
    pub(crate) stepped: usize,
    pub(crate) steps: usize,
}

/// The catch-up's own entry points, for the bench drivers that time them against each other; not
/// part of the library's interface (the pass reaches them through
/// `CrossTerms::advance_or_resume`), and built only with the `oracle` feature.
#[cfg(all(feature = "oracle", not(any(loom, shuttle))))]
#[doc(hidden)]
pub mod bench {
    use super::{CrossTerms, Side};

    /// `CrossTerms::step_lanes`.
    pub fn step_lanes<S: Side>(
        pairs: &CrossTerms<S>,
        run: usize,
        at: (usize, usize),
        states: &mut [i64],
    ) -> bool {
        pairs.step_lanes(run, at, states)
    }

    /// `CrossTerms::resume_lanes`.
    pub fn resume_lanes<S: Side>(
        pairs: &CrossTerms<S>,
        at: (usize, usize),
        states: &mut [i64],
    ) -> usize {
        pairs.resume_lanes(at, states)
    }

    /// This target's catch-up crossover, owned by the planner.
    pub const fn break_even(m: usize) -> usize {
        crate::batch::break_even(m)
    }
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod tests;
