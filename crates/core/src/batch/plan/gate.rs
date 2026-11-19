//! The survivor filter's route gate: whether a run installs the PAA producer or no filter at all,
//! decided once from the run's shape before any column is built or any tile runs, and never
//! changed mid-run. It is a route decision: every result is the plain pass's whatever it decides,
//! and only the time differs.

use crate::batch::survivor::Filter;

/// The self-join engine a run's shape names, for the gate. Only the first three were measured;
/// every other run keeps the filter off, except the best-pair run, which always asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Engine {
    /// The left profile.
    Left,
    /// The two-sided profile: the left and right sides from one pass.
    TwoSided,
    /// The left or the two-sided profile keeping more than one neighbor per window.
    TopK,
    /// The two-sided profile of a run that reports only its best pair: it keeps a global floor,
    /// which the per-row thresholds were not tuned for, so it asks at every window length.
    BestPair,
    /// The AB-join over every diagonal keeping one neighbor per window: the two-sided profile's
    /// rule, since the route's floors are the rows' alone and the join shares no table of floors
    /// between its sides. It has no constant class (a join has no exclusion zone).
    Join,
    /// The AB-join over every diagonal keeping more than one neighbor per window: the top-k
    /// rule.
    JoinTopK,
    /// A run the gate leaves off: passes over chosen diagonals only and walks, none of them
    /// measured.
    Ungated,
}

impl Engine {
    /// The left (or, with `two_sided`, the two-sided) self-join engine keeping `k` neighbors.
    pub(crate) fn self_join(two_sided: bool, k: usize) -> Self {
        match (k > 1, two_sided) {
            (true, _) => Self::TopK,
            (false, false) => Self::Left,
            (false, true) => Self::TwoSided,
        }
    }

    /// The AB-join over every diagonal keeping `k` neighbors per window.
    pub(crate) fn join(k: usize) -> Self {
        if k > 1 { Self::JoinTopK } else { Self::Join }
    }
}

/// What the gate reads of a run: its window length, its engine and the threads it runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RunShape {
    pub(crate) m: usize,
    pub(crate) engine: Engine,
    pub(crate) threads: usize,
}

/// The least window length at which an engine turns PAA on, on one thread and on more.
///
/// The thresholds ([`LEFT_FROM_M`], [`TWO_SIDED_FROM_M`], [`TOP_K_FROM_M`]) come from the x86 box
/// sweep (EPYC 7R13, 2026-11-03) of PAA with shared floors against the plain pass at n 65536 on a
/// grid, a float, a messy and a real BTC series, at 1 and 8 threads and m 256, 384, 512, 768, 1024
/// and 2048, five runs a cell, comparing medians. Each threshold is the first swept m at which PAA
/// beats plain on every series with some margin, and it keeps beating it at every larger m: the
/// left profile on more threads skips m 384, where PAA won on the grid series by under 1%. The
/// left profile on one thread is the exception: m 256, the lowest swept, already wins on every
/// series, though by only 2% on two of them. Top-k was not measured and follows the two-sided
/// profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FromM {
    pub(crate) one_thread: usize,
    pub(crate) threads: usize,
}

impl FromM {
    /// The threshold for a run on `threads` threads.
    pub(crate) const fn at(self, threads: usize) -> usize {
        if threads > 1 {
            self.threads
        } else {
            self.one_thread
        }
    }
}

/// From which m the left profile turns PAA on.
///
/// One thread: at m 256 PAA took 0.75 to 0.98 of the plain time, the thinnest margins on the grid
/// and float series (0.98 each); m 256 is the lowest m swept, and below it (m 64) PAA lost on all
/// but one cell of a provisional run. More threads: at m 512 PAA took 0.65 to 0.94, at m 384 0.73
/// to 1.00 (the grid series, 0.997).
pub(crate) const LEFT_FROM_M: FromM = FromM {
    one_thread: 256,
    threads: 512,
};

/// From which m the two-sided profile turns PAA on.
///
/// One thread: at m 512 PAA took 0.79 to 0.97 of the plain time (the messy series, 0.97), at
/// m 384 0.86 to 1.02 (the grid series lost). More threads: at m 768 PAA took 0.76 to 0.95 (the
/// messy series, 0.95), at m 512 0.84 to 1.05 (the grid series lost).
pub(crate) const TWO_SIDED_FROM_M: FromM = FromM {
    one_thread: 512,
    threads: 768,
};

/// From which m the top-k profiles turn PAA on: not measured yet, so the two-sided profile's.
pub(crate) const TOP_K_FROM_M: FromM = TWO_SIDED_FROM_M;

/// The production rule: PAA at or past the engine's threshold, otherwise no filter; a best-pair
/// run always takes PAA.
pub(crate) fn gate(shape: RunShape) -> Filter {
    let from = match shape.engine {
        Engine::Left => LEFT_FROM_M,
        Engine::TwoSided | Engine::Join => TWO_SIDED_FROM_M,
        Engine::TopK | Engine::JoinTopK => TOP_K_FROM_M,
        Engine::BestPair => return Filter::Paa,
        Engine::Ungated => return Filter::Off,
    };
    if shape.m >= from.at(shape.threads) {
        Filter::Paa
    } else {
        Filter::Off
    }
}

/// The survivor filter a run of `shape` installs, decided once, before it builds its columns: the
/// production gate's choice, unless the route harness forces a filter. The only place a run's
/// filter is decided; the run keeps the [`Choice`] and installs it with its coefficients
/// ([`Choice::install`]).
pub(crate) fn survivor_filter(shape: RunShape) -> Choice {
    let gated = gate(shape);
    if gated != Filter::Off {
        crate::route::count_gate_on();
    }
    Choice {
        gated,
        forced: crate::route::forced_survivor(),
    }
}

/// A run's survivor filter as [`survivor_filter`] decided it: the gate's choice, and the filter the
/// route harness forces in its place, if any.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Choice {
    gated: Filter,
    forced: Option<Filter>,
}

impl Choice {
    /// The filter the run installs and the PAA coefficients it reads, which `build` makes, only
    /// asked when the filter reads them. Where the gate chose PAA and `build` gives nothing (no
    /// room for the table, or more windows than a `u32` counts), the run installs no filter
    /// instead, so it builds no frontier and no shared table: the plain pass, never a pass that
    /// asks every block and is refused. A forced filter stays installed whatever `build` gives (a
    /// forced PAA producer without its table is refused block by block, and counted).
    pub(crate) fn install<T>(self, build: impl FnOnce() -> Option<T>) -> (Filter, Option<T>) {
        let filter = self.forced.unwrap_or(self.gated);
        if !filter.reads_paa() {
            return (filter, None);
        }
        match build() {
            Some(paa) => (filter, Some(paa)),
            None if self.forced.is_none() => {
                crate::route::count_gate_fallback();
                (Filter::Off, None)
            }
            None => (filter, None),
        }
    }
}

#[cfg(all(test, not(any(loom, shuttle))))]
mod tests;
