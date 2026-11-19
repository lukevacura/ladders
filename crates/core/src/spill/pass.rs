//! The cold pass: every pair that touches a spilled window, walked along its diagonal with the exact
//! dot-product recurrence (one exact dot product starts each chain of consecutive pairs), each row
//! keeping its `k` nearest and merging them with what the hot passes found. A pass over rows
//! `band` allocates once, up front, and a band's rows do not depend on any other band's, so bands
//! run on separate threads and give the same lists for any cut.

use alloc::vec::Vec;
use core::ops::{Range, RangeInclusive};

use super::{Arith, Pick, Spilled, Truth, cold_nearer, merge_nearest};
use crate::capacity::{CapacityError, reserved_vec};
use crate::entry::Entry;
use crate::kbest;
use crate::nearest::{Candidate, clearly_nearer, nearer, winner_entry};

/// What a cold pass walked: pairs offered, and chains started (each costs one length-`m` dot).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub cells: u64,
    pub chains: u64,
}

impl core::ops::AddAssign for Counts {
    fn add_assign(&mut self, other: Self) {
        self.cells += other.cells;
        self.chains += other.chains;
    }
}

/// Which neighbors a pass offers each row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reach {
    /// The windows before a row, outside its exclusion zone.
    Left,
    /// The windows after a row, outside its exclusion zone.
    Right,
    /// Every window of the other series, in a join.
    Join,
}

/// One series' spilled windows as a pass reads them: the flags, their maximal runs, and how many
/// windows are spilled before each position.
#[derive(Clone, Copy)]
pub(super) struct Marks<'a> {
    pub(super) flags: &'a [bool],
    pub(super) runs: &'a [(usize, usize)],
    pub(super) before: &'a [usize],
}

impl Marks<'_> {
    /// Whether every window is spilled.
    fn all(&self) -> bool {
        self.runs.len() == 1 && self.runs[0] == (0, self.flags.len() - 1)
    }
}

/// The pairs a pass offers: row `i` of the rows' series with window `i + offset` of the
/// neighbors' series (the same series for a self-join), for the offsets in `offsets`, when either
/// window is spilled.
pub(super) struct Cover<'a> {
    pub(super) rows: Marks<'a>,
    pub(super) cols: Marks<'a>,
    pub(super) offsets: RangeInclusive<isize>,
}

impl Cover<'_> {
    /// The neighbors row `row` is offered, spilled or not.
    fn neighbors(&self, row: usize) -> Range<usize> {
        let columns = self.cols.flags.len() as isize;
        let first = (row as isize + *self.offsets.start()).max(0);
        let last = (row as isize + *self.offsets.end()).min(columns - 1);
        if first > last {
            0..0
        } else {
            first as usize..last as usize + 1
        }
    }

    /// Calls `visit(offset, first, last, unspilled_rows_only)` for each chain of consecutive pairs
    /// `(i, i + offset)`, `i` in `first..=last`, rows within `band`, that touches a spilled
    /// window. A pair whose row is spilled belongs to a chain of its row's run, and is offered;
    /// any other pair belongs to a chain of its neighbor's run, whose spilled rows are not offered.
    fn chains(&self, band: Range<usize>, mut visit: impl FnMut(isize, usize, usize, bool)) {
        let (rows, columns) = (
            self.rows.flags.len() as isize,
            self.cols.flags.len() as isize,
        );
        if band.is_empty() || rows == 0 || columns == 0 {
            return;
        }
        let (lowest, highest) = (*self.offsets.start(), *self.offsets.end());
        let (band_first, band_last) = (band.start as isize, band.end as isize - 1);
        for &(start, end) in self.rows.runs {
            let (first, last) = (
                (start as isize).max(band_first),
                (end as isize).min(band_last),
            );
            if first > last {
                continue;
            }
            for offset in lowest.max(-last)..=highest.min(columns - 1 - first) {
                let (from, to) = (first.max(-offset), last.min(columns - 1 - offset));
                visit(offset, from as usize, to as usize, false);
            }
        }
        if self.rows.all() {
            return;
        }
        for &(start, end) in self.cols.runs {
            let (start, end) = (start as isize, end as isize);
            for offset in lowest.max(start - band_last)..=highest.min(end - band_first) {
                let from = band_first.max(start - offset).max(0);
                let to = band_last.min(end - offset).min(rows - 1);
                if from <= to {
                    visit(offset, from as usize, to as usize, true);
                }
            }
        }
    }

    /// The pairs and chains a pass over rows `band` walks, counted without walking them.
    pub(super) fn count(&self, band: Range<usize>) -> Counts {
        let mut counts = Counts::default();
        self.chains(band, |_, first, last, unspilled_only| {
            counts.chains += 1;
            let rows = last - first + 1;
            let spilled = self.rows.before[last + 1] - self.rows.before[first];
            counts.cells += (if unspilled_only { rows - spilled } else { rows }) as u64;
        });
        counts
    }
}

/// The cold candidates of a band of rows: each row's `k` nearest so far, nearest first, in room
/// reserved for all rows at once (a row has room for as many candidates as it can be offered, at
/// most `k`), so offering allocates nothing.
pub(super) struct Lists<A> {
    /// Row `r` of the band owns `slots[starts[r]..starts[r + 1]]`, of which `lens[r]` are kept.
    starts: Vec<usize>,
    lens: Vec<usize>,
    slots: Vec<Candidate<Spilled<A>>>,
}

impl<A: Arith> Lists<A> {
    /// Room for `band`'s rows at `k` neighbors each.
    fn new(cover: &Cover, band: Range<usize>, k: usize) -> Result<Self, CapacityError> {
        let mut starts = reserved_vec::<usize>(band.len() as u128 + 1)?;
        let mut total = 0usize;
        for row in band.clone() {
            starts.push(total);
            let neighbors = cover.neighbors(row);
            let offered = if cover.rows.flags[row] {
                neighbors.len()
            } else {
                cover.cols.before[neighbors.end] - cover.cols.before[neighbors.start]
            };
            total = total.checked_add(k.min(offered)).ok_or(CapacityError)?;
        }
        starts.push(total);
        let mut lens = reserved_vec::<usize>(band.len() as u128)?;
        lens.resize(band.len(), 0);
        let mut slots = reserved_vec::<Candidate<Spilled<A>>>(total as u128)?;
        slots.resize(
            total,
            Candidate {
                index: 0,
                rho_hat: 0.0,
                correlation: Spilled::BothConstant,
            },
        );
        Ok(Self {
            starts,
            lens,
            slots,
        })
    }

    fn room(&self, row: usize) -> usize {
        self.starts[row + 1] - self.starts[row]
    }

    /// Whether `row`'s candidates are full and the worst already beats estimate `rho_hat`
    /// clearly: offering it changes nothing.
    fn clearly_beats(&self, row: usize, rho_hat: f64) -> bool {
        let len = self.lens[row];
        len > 0
            && len == self.room(row)
            && clearly_nearer(self.slots[self.starts[row] + len - 1].rho_hat, rho_hat)
    }

    fn offer(&mut self, row: usize, candidate: Candidate<Spilled<A>>) {
        let kept = &mut self.slots[self.starts[row]..self.starts[row + 1]];
        kbest::keep_within(kept, &mut self.lens[row], candidate, |a, b| nearer(*a, *b));
    }

    fn candidates(&self, row: usize) -> &[Candidate<Spilled<A>>] {
        &self.slots[self.starts[row]..self.starts[row] + self.lens[row]]
    }
}

/// A hot profile's flat kept list, indexed by row: the entries of a row's run, nearest first, or
/// one empty entry.
pub(crate) struct HotRuns<'a> {
    list: &'a [(u64, Entry)],
    starts: Vec<usize>,
}

impl<'a> HotRuns<'a> {
    /// `list` holds `rows` runs in row order, the `r`th run's entries tagged `r`.
    pub(crate) fn new(list: &'a [(u64, Entry)], rows: usize) -> Result<Self, CapacityError> {
        let mut starts = reserved_vec::<usize>(rows as u128 + 1)?;
        let mut at = 0;
        for row in 0..rows {
            starts.push(at);
            while list.get(at).is_some_and(|&(tag, _)| tag == row as u64) {
                at += 1;
            }
        }
        starts.push(at);
        Ok(Self { list, starts })
    }

    fn run(&self, row: usize) -> &'a [(u64, Entry)] {
        &self.list[self.starts[row]..self.starts[row + 1]]
    }

    /// The runs of the rows before `row`.
    pub(crate) fn before(&self, row: usize) -> &'a [(u64, Entry)] {
        &self.list[..self.starts[row]]
    }

    /// The runs of `row` and the rows after it.
    pub(crate) fn tail(&self, row: usize) -> &'a [(u64, Entry)] {
        &self.list[self.starts[row]..]
    }
}

/// A hot row's run without the empty entry a row with no neighbor holds.
fn real_entries(run: &[(u64, Entry)]) -> &[(u64, Entry)] {
    match run.first() {
        Some((_, first)) if first.index().get().is_some() => run,
        _ => &[],
    }
}

impl<A: Arith> Truth<'_, A> {
    /// Rows `band`'s `k` best candidates among the pairs `cover` offers that touch a spilled
    /// window.
    pub(super) fn cold_band(
        &self,
        cover: &Cover,
        band: Range<usize>,
        k: usize,
        counts: &mut Counts,
    ) -> Result<Lists<A>, CapacityError> {
        let mut best = Lists::new(cover, band.clone(), k)?;
        let m = self.m;
        let (rows, columns) = (&*self.rows.ticks, &*self.cols().ticks);
        let windows = (&self.rows.windows[..], &self.cols().windows[..]);
        let chain = |offset: isize, first: usize, last: usize, unspilled_only: bool| {
            counts.chains += 1;
            let neighbor = |i: usize| (i as isize + offset) as usize;
            let mut qt = self.dot(first, neighbor(first));
            for i in first..=last {
                let j = neighbor(i);
                if !(unspilled_only && cover.rows.flags[i]) {
                    counts.cells += 1;
                    let row = i - band.start;
                    let skip = |rho_hat| best.clearly_beats(row, rho_hat);
                    if let Some(candidate) = self.candidate(windows, (i, j), qt, skip) {
                        best.offer(row, candidate);
                    }
                }
                if i < last {
                    qt = qt
                        .add(A::product(rows[i + m], columns[j + m]))
                        .sub(A::product(rows[i], columns[j]));
                }
            }
        };
        cover.chains(band.clone(), chain);
        Ok(best)
    }

    /// Appends row `i`'s `k` nearest of its hot entries and its cold candidates, nearest first
    /// (one empty entry when it has none). Both lists are nearest first and share no neighbor.
    fn merge_row(
        &self,
        i: usize,
        hot: &[(u64, Entry)],
        cold: &[Candidate<Spilled<A>>],
        k: usize,
        out: &mut Vec<(u64, Entry)>,
    ) {
        let hot = real_entries(hot);
        let before = out.len();
        let cold_wins = |candidate, at: usize| self.cold_wins(i, candidate, hot[at].1);
        merge_nearest(hot.len(), cold, k, cold_wins, |pick| match pick {
            Pick::Cold(at) => out.push((i as u64, winner_entry(self.m, cold[at]))),
            Pick::Hot(at) => out.push(hot[at]),
        });
        if out.len() == before {
            out.push((i as u64, Entry::NONE));
        }
    }

    /// Whether `cold` is exactly nearer to row `i` than `hot`, the hot passes' entry. The hot
    /// neighbor's exact candidate is built only to settle a near-tie.
    fn cold_wins(&self, i: usize, cold: Candidate<Spilled<A>>, hot: Entry) -> bool {
        cold_nearer(cold, hot, || {
            hot.index()
                .get()
                .and_then(|neighbor| self.pair(i, neighbor as usize))
        })
    }

    /// Appends the final lists of rows `band` to `out`.
    pub(super) fn merge_band(
        &self,
        cover: &Cover,
        band: Range<usize>,
        k: usize,
        hot: &HotRuns,
        out: &mut Vec<(u64, Entry)>,
        counts: &mut Counts,
    ) -> Result<(), CapacityError> {
        let cold = self.cold_band(cover, band.clone(), k, counts)?;
        let mut entries = 0usize;
        for (at, row) in band.clone().enumerate() {
            let offered = real_entries(hot.run(row)).len() + cold.candidates(at).len();
            entries = entries.saturating_add(k.min(offered).max(1));
        }
        out.try_reserve_exact(entries).map_err(|_| CapacityError)?;
        for (at, row) in band.enumerate() {
            self.merge_row(row, hot.run(row), cold.candidates(at), k, out);
        }
        Ok(())
    }

    /// Whether neighbor `x` is exactly nearer to row `row` than `y` is, ties to the smaller index.
    pub(super) fn nearer_neighbor(&self, row: usize, x: usize, y: usize) -> bool {
        match (self.pair(row, x), self.pair(row, y)) {
            (Some(x), Some(y)) => nearer(x, y),
            (Some(_), None) => true,
            _ => false,
        }
    }
}
