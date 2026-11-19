//! Every machine cost constant and route crossover, the only place they are kept: what each
//! counted operation costs on this target, and the crossovers decided from them. They are machine
//! constants, measured by a cost driver timing (or counting the instructions of) each operation,
//! never fitted to a dataset's outcome. Each target reads its own table, selected by
//! `cfg(target_arch)`; any other target takes the larger of the two measured costs, each.

use crate::spill::{Counts, Plan, Shape, Width};

/// One measured fit of the catch-up crossover: `m · num / den + plus` steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fit {
    pub(crate) num: usize,
    pub(crate) den: usize,
    pub(crate) plus: usize,
}

impl Fit {
    const fn at(self, m: usize) -> usize {
        let whole = (m / self.den).saturating_mul(self.num);
        let part = (m % self.den) * self.num / self.den;
        whole.saturating_add(part).saturating_add(self.plus)
    }
}

/// What the cold route costs and saves, in instructions retired per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColdCosts {
    /// One pair of a cold pass, per tier.
    pub(crate) cell_128: u64,
    pub(crate) cell_256: u64,
    /// One multiply-add of the exact dot product that starts a chain, per tier.
    pub(crate) dot_term_128: u64,
    pub(crate) dot_term_256: u64,
    /// What a pair costs on the exact i64 pass beyond a pair on the narrow pass.
    pub(crate) wide_extra_cell: u64,
}

/// One target's costs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Costs {
    /// The run length from which resuming a skipped run's lanes costs less than stepping them:
    /// the largest of these fits.
    pub(crate) catch_up: &'static [Fit],
    /// One row merge (a lock and `RowAcc::merge`) of a tile's partials, in plain-pass pairs.
    pub(crate) merge: f64,
    /// A tile's start per lane and per window sample, in plain-pass pairs.
    pub(crate) start: f64,
    /// A run of consecutive visited diagonals shorter than this is walked diagonal by diagonal;
    /// the tiled pass takes the longer ones.
    pub(crate) walk_below_run: usize,
    /// The cold route's costs.
    pub(crate) cold: ColdCosts,
}

/// The NEON fit of the catch-up crossover, `m/4 + 4`.
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
const NEON_CATCH_UP: Fit = Fit {
    num: 1,
    den: 4,
    plus: 4,
};
/// The AVX2 fit of the catch-up crossover, `3m/8`.
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
const X86_CATCH_UP: Fit = Fit {
    num: 3,
    den: 8,
    plus: 0,
};

/// Apple M-series, NEON.
///
/// The catch-up crossover, by a cost driver timing `CrossTerms::step_lanes` against
/// `CrossTerms::resume_lanes` (64 lanes, a run of 32, narrow offsets) on an Apple M1 Pro:
/// `resume_ns / step_ns` at m 16, 64, 256 and 1024 was 7.5, 19, 63 and 243 (best of three,
/// 2026-11-03), and 8.6, 22.8, 75.9, 288.6 and 8.8, 22.5, 73.1, 280.1 (the run-plan cost driver,
/// median of five rounds, two runs, 2026-11-04); `m/4 + 4` (8, 20, 68, 260) lies between the
/// sessions.
///
/// The tile cut's merge and start, on an Apple M1 Pro (release, a random walk at a market offset):
/// a pair 0.53 ns (single-threaded `left_profile`, n 5·10⁴, m 50 and 1000); a row merge 12 to 15
/// ns for tiles one chunk wide, rising to about 50 ns for tiles of ten chunks or more, so 25 to 30
/// pairs, 28 here; a start 0.134 to 0.150 ns per lane and sample at m 50 (0.25 pairs) and 0.066
/// ns at m 1000 (0.125 pairs), 0.15 here, weighted to the large m where it matters. The run-plan
/// cost driver's fit of the same model on the two-sided profile (tiled on one thread, n 32768,
/// median of five rounds, two runs, 2026-11-04, load average 5 to 20) agrees: a merge 30 and 35
/// pairs at m 50 and 19 and 23 at m 1000, a start 0.22 and 0.29 pairs at m 50 and 0.10 and 0.11
/// at m 1000.
///
/// The walk-or-tile split, by the run-plan cost driver on the same machine and date (one thread,
/// median of five rounds, two runs): over runs of r consecutive diagonals at a tenth of all
/// diagonals (n 32768, m 256), the walk costs 4.4 ns per pair at every r, the tiled pass at its
/// better of two shapes 5.5, 4.6 and 3.2 ns at r 8, 12 and 16, so the tiles win from 16.
///
/// The cold route, in instructions retired (`/usr/bin/time -l`), by the run-plan cost driver on
/// the same machine and date, each from two sizes' difference: a narrow pair 8.7 and an exact
/// i64 pair 22.8 (the left profile, the i64 pass forced), so 14 more; a cold pair 128 in the
/// 128-bit tier and 208 in the 256-bit tier, and a dot-product term 29 and 61.
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
pub(crate) const AARCH64: Costs = Costs {
    catch_up: &[NEON_CATCH_UP],
    merge: 28.0,
    start: 0.15,
    walk_below_run: 16,
    cold: ColdCosts {
        cell_128: 128,
        cell_256: 208,
        dot_term_128: 29,
        dot_term_256: 61,
        wide_extra_cell: 14,
    },
};

/// The EC2 x86 box (c6a.4xlarge, AMD EPYC 7R13), AVX2.
///
/// The catch-up crossover, by the same drivers on the box: `resume_ns / step_ns` at m 16, 64, 256
/// and 1024 was 5.8, 21.5, 85 and 337 (median, 2026-11-03) and 5.6, 20.5, 106.0 and 420.3 (the
/// run-plan cost driver, median of five rounds, 2026-11-04); the fit is through the two sessions'
/// mean (5.7, 21, 95.5 and 378.7), `3m/8` (6, 24, 96, 384).
///
/// The tile cut's merge and start, by the run-plan cost driver's fit on the box (one session,
/// 2026-11-04, a pair 0.708 ns): a merge 34.6 pairs at m 50 and 26.0 at m 1000, the mean of the
/// two, 30, here; a start 0.80 pairs at m 50 and 0.58 at m 1000, 0.6 here, weighted to the large
/// m where it matters.
///
/// The walk-or-tile split, by the run-plan cost driver on the box (one session, 2026-11-04): the
/// walk costs 6.77 ns per pair, the tiled pass 6.82 at runs of 12 and 5.30 at 16, so the tiles win
/// from 16.
///
/// The cold route is provisional: the box has no instruction counter, so it keeps the counts every
/// target used before they were per target (`benches/spill.rs` on an Apple M1 Pro: a cold pair 104
/// and 178, a dot-product term 25 and 165, an exact i64 pair 13 more than a narrow one).
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
pub(crate) const X86_64: Costs = Costs {
    catch_up: &[X86_CATCH_UP],
    merge: 30.0,
    start: 0.6,
    walk_below_run: 16,
    cold: ColdCosts {
        cell_128: 104,
        cell_256: 178,
        dot_term_128: 25,
        dot_term_256: 165,
        wide_extra_cell: 13,
    },
};

/// The larger of two costs.
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
const fn larger(a: u64, b: u64) -> u64 {
    if a > b { a } else { b }
}

/// The larger of two costs in pairs.
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
const fn larger_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Any other target: the larger of the two measured costs, each, and both catch-up fits.
#[cfg_attr(not(all(test, not(any(loom, shuttle)))), allow(dead_code))] // each target reads one table
pub(crate) const OTHER: Costs = Costs {
    catch_up: &[NEON_CATCH_UP, X86_CATCH_UP],
    merge: larger_f64(AARCH64.merge, X86_64.merge),
    start: larger_f64(AARCH64.start, X86_64.start),
    walk_below_run: larger(AARCH64.walk_below_run as u64, X86_64.walk_below_run as u64) as usize,
    cold: ColdCosts {
        cell_128: larger(AARCH64.cold.cell_128, X86_64.cold.cell_128),
        cell_256: larger(AARCH64.cold.cell_256, X86_64.cold.cell_256),
        dot_term_128: larger(AARCH64.cold.dot_term_128, X86_64.cold.dot_term_128),
        dot_term_256: larger(AARCH64.cold.dot_term_256, X86_64.cold.dot_term_256),
        wide_extra_cell: larger(AARCH64.cold.wide_extra_cell, X86_64.cold.wide_extra_cell),
    },
};

/// This target's costs.
#[cfg(target_arch = "aarch64")]
pub(crate) const TARGET: Costs = AARCH64;
/// This target's costs.
#[cfg(target_arch = "x86_64")]
pub(crate) const TARGET: Costs = X86_64;
/// This target's costs.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) const TARGET: Costs = OTHER;

/// One row merge on this target, in pairs.
pub(crate) const MERGE: f64 = TARGET.merge;
/// A tile's start per lane and sample on this target, in pairs.
pub(crate) const START: f64 = TARGET.start;
/// The walk-or-tile split on this target.
pub(crate) const WALK_BELOW_RUN: usize = TARGET.walk_below_run;

impl Costs {
    /// The run length, in steps, at window length `m`, from which resuming a skipped run's lanes
    /// is cheaper than stepping them through it: the largest of the table's fits.
    pub(crate) const fn break_even(&self, m: usize) -> usize {
        let mut best = 0;
        let mut i = 0;
        while i < self.catch_up.len() {
            let at = self.catch_up[i].at(m);
            if at > best {
                best = at;
            }
            i += 1;
        }
        best
    }
}

/// The run length, in steps, at window length `m`, from which resuming a skipped run's lanes is
/// cheaper than stepping them through it, on this target.
///
/// Catching up `L` lanes costs `L·run·s` by stepping (`s`, one lane-step of the recurrence: two
/// products and two sums per lane, vectorized over the lanes) and `L·m·p` by resuming them
/// together (`p`, one product and sum per lane and tick, the lanes accumulated in registers), so
/// they break even at `run = m·p/s`, a fixed share of `m` plus a small per-lane overhead. The share
/// differs by target, so each target takes its own fit (see its table); the exact i64 arithmetic
/// breaks even later, but it is the cold route, and the fits are the narrow arithmetic's.
pub(crate) const fn break_even(m: usize) -> usize {
    TARGET.break_even(m)
}

/// Whether a run of `run` consecutive visited diagonals is walked diagonal by diagonal rather than
/// tiled.
pub(crate) const fn walks(run: usize) -> bool {
    run < WALK_BELOW_RUN
}

/// Whether a skipped run of `run` steps at window length `m` steps through rather than resumes.
pub(crate) const fn steps_through(run: usize, m: usize) -> bool {
    run < break_even(m)
}

impl ColdCosts {
    /// What taking every spilled window adds to the walk the data requires anyway.
    pub(crate) fn cold_cost(&self, shape: &Shape) -> u64 {
        let (cell, term) = match shape.width {
            Width::Narrow128 => (self.cell_128, self.dot_term_128),
            Width::Wide256 => (self.cell_256, self.dot_term_256),
        };
        let extra = Counts {
            cells: shape.walk.cells.saturating_sub(shape.required.cells),
            chains: shape.walk.chains.saturating_sub(shape.required.chains),
        };
        extra.cells.saturating_mul(cell).saturating_add(
            extra
                .chains
                .saturating_mul(shape.m as u64)
                .saturating_mul(term),
        )
    }

    /// What the exact i64 pass the cold route replaces costs beyond the narrow pass.
    pub(crate) fn saved_cost(&self, shape: &Shape) -> u64 {
        shape.wide_cells_today.saturating_mul(self.wide_extra_cell)
    }

    /// The cold route takes every spilled window when it walks anything, there is an exact i64
    /// pass to save, and what taking them adds to the walk costs no more than that pass; the hot
    /// passes' own work is the same either way.
    pub(crate) fn plan(&self, shape: &Shape) -> Plan {
        let (cold_cost, saved_cost) = (self.cold_cost(shape), self.saved_cost(shape));
        let natural = shape.walk.cells > 0 && saved_cost > 0 && cold_cost <= saved_cost;
        Plan {
            engage: crate::route::spill_engaged(natural),
            natural,
            cold_cost,
            saved_cost,
        }
    }
}

/// The cold route's verdict for a run of `shape` on this target.
#[doc(hidden)]
pub fn cold_route(shape: &Shape) -> Plan {
    TARGET.cold.plan(shape)
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod tests;
