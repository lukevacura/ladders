//! The cold route driven as a batch engine would: today's left profile over the series with its
//! outlier ticks masked as gaps, then the cold route folded in. Shared by the spill tests and the
//! route harness.
#![allow(dead_code)]

use ladde::batch::cold_route;
use ladde::batch::left_profile_report_tiled;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::quantize::{Scale, to_ticks};
use ladde::spill::{Counts, Plan, Survey, Width};
use std::num::NonZeroUsize;

pub use super::spill_ref::Bits;

pub fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

pub fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

pub fn bits(entries: &[(u64, Entry)]) -> Vec<Bits> {
    entries
        .iter()
        .map(|&(row, e)| {
            (
                row,
                e.rho().to_bits(),
                e.squared_distance().to_bits(),
                e.index().get().unwrap_or(u64::MAX),
            )
        })
        .collect()
}

/// A deterministic integer random walk.
pub fn walk(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut level = 1000i64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            level += (state % 9) as i64 - 4;
            level as f64
        })
        .collect()
}

/// `series` with `(position, shock)` added.
pub fn with_spikes(mut series: Vec<f64>, spikes: &[(usize, f64)]) -> Vec<f64> {
    for &(at, shock) in spikes {
        series[at] += shock;
    }
    series
}

/// Whole ticks of `series` at `scale`, a gap filled with the previous tick (the first one for a
/// leading gap), and whether each tick was measured.
pub fn ticks_of(series: &[f64], scale: Scale) -> (Vec<i64>, Vec<bool>) {
    let measured: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let first = measured.iter().flatten().next().copied().unwrap_or(0);
    let mut last = first;
    let ticks = measured
        .iter()
        .map(|tick| {
            last = tick.unwrap_or(last);
            last
        })
        .collect();
    (ticks, measured.iter().map(Option::is_some).collect())
}

/// What a run did.
#[derive(Debug)]
pub struct Outcome {
    pub entries: Vec<(u64, Entry)>,
    /// The cost model's verdict, `None` when the series is outside the cold route's domain.
    pub plan: Option<Plan>,
    /// What the cold pass walked; zero when it did not run.
    pub walked: Counts,
    pub width: Option<Width>,
    pub spilled_windows: usize,
    pub outlier_ticks: usize,
}

fn today(series: &[f64], m: usize, scale: Scale) -> (Vec<(u64, Entry)>, usize, usize) {
    let (entries, report) =
        left_profile_report_tiled(series, nz(m), scale, nz(2), nz(4), nz(2), nz(16));
    (entries, report.tiles, report.wide_tiles)
}

/// The left profile through today's routes alone.
pub fn left_today(series: &[f64], m: usize, scale: Scale) -> Vec<(u64, Entry)> {
    today(series, m, scale).0
}

/// The left profile with the cold route taken or declined by the cost model and the forced route
/// in effect (`forced_route` is process-wide: hold its guard around the call). The pairs today's
/// routes would run on the exact i64 pass are those of its wide tiles.
pub fn left_spill(series: &[f64], m: usize, scale: Scale) -> Outcome {
    left_spill_sharing(series, m, scale, None)
}

/// [`left_spill`] with `wide_per_mille` of all pairs taken as today's exact i64 pairs, when given:
/// the cost model's input as a caller with larger tiles would measure it.
pub fn left_spill_sharing(
    series: &[f64],
    m: usize,
    scale: Scale,
    wide_per_mille: Option<u64>,
) -> Outcome {
    let (ticks, had) = ticks_of(series, scale);
    let (entries, tiles, wide_tiles) = today(series, m, scale);
    let plain = |entries| Outcome {
        entries,
        plan: None,
        walked: Counts::default(),
        width: None,
        spilled_windows: 0,
        outlier_ticks: 0,
    };
    let Some(survey) = Survey::new(&ticks, &had, nz(m)).unwrap() else {
        return plain(entries);
    };
    let windows = (ticks.len() + 1).saturating_sub(m);
    let exclusion = ladde::exclusion::zone(m);
    let pairs: u64 = (0..windows)
        .map(|row| row.saturating_sub(exclusion) as u64)
        .sum();
    let wide_cells_today = match wide_per_mille {
        Some(share) => pairs * share / 1000,
        None if tiles == 0 => 0,
        None => pairs * wide_tiles as u64 / tiles as u64,
    };
    let plan = cold_route(&survey.shape(wide_cells_today).unwrap());
    let outlier_ticks = survey.outliers().iter().filter(|&&flag| flag).count();
    let spilled_windows = survey.spilled().iter().filter(|&&flag| flag).count();
    let mut outcome = Outcome {
        entries,
        plan: Some(plan),
        walked: Counts::default(),
        width: Some(survey.width()),
        spilled_windows,
        outlier_ticks,
    };
    if plan.engage {
        let masked: Vec<f64> = series
            .iter()
            .zip(survey.outliers())
            .map(|(&x, &flag)| if flag { f64::NAN } else { x })
            .collect();
        let mut entries = left_today(&masked, m, scale);
        entries.sort_by_key(|&(row, _)| row);
        outcome.walked = survey.merge_left(&mut entries).unwrap();
        outcome.entries = entries;
    }
    outcome
}

/// The exact reference profile of `series`.
pub fn reference(series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let (ticks, had) = ticks_of(series, scale);
    super::spill_ref::left_profile(&ticks, &had, m)
}

/// An entry from its bits.
pub fn entry_of(bits: Bits) -> Entry {
    let index = SampleIndex::new(bits.3).unwrap_or(SampleIndex::NONE);
    Entry::new(f64::from_bits(bits.1), f64::from_bits(bits.2), index).unwrap()
}

/// The exact two-sided profile of `series`: per row, left, right and nearest.
pub fn exact_two_sided(series: &[f64], scale: Scale, m: usize) -> Vec<[Bits; 3]> {
    let (ticks, had) = ticks_of(series, scale);
    super::spill_ref::two_sided(&ticks, &had, m)
}

/// The exact two-sided top-k profile of `series`: per row, left, right and the `k` nearest.
pub fn exact_two_sided_topk(
    series: &[f64],
    scale: Scale,
    m: usize,
    k: usize,
) -> Vec<(Bits, Bits, Vec<Bits>)> {
    let (ticks, had) = ticks_of(series, scale);
    super::spill_ref::two_sided_topk(&ticks, &had, m, k)
}

/// The exact AB-join of `a` against `b`: per row of `a`, its nearest window of `b`.
pub fn exact_join(a: &[f64], scale_a: Scale, b: &[f64], scale_b: Scale, m: usize) -> Vec<Bits> {
    let ((ticks_a, had_a), (ticks_b, had_b)) = (ticks_of(a, scale_a), ticks_of(b, scale_b));
    super::spill_ref::join((&ticks_a, &had_a), (&ticks_b, &had_b), m)
}

/// The exact top-k AB-join of `a` against `b`: per row of `a`, its `k` nearest windows of `b`.
pub fn exact_join_topk(
    a: &[f64],
    scale_a: Scale,
    b: &[f64],
    scale_b: Scale,
    m: usize,
    k: usize,
) -> Vec<Vec<Bits>> {
    let ((ticks_a, had_a), (ticks_b, had_b)) = (ticks_of(a, scale_a), ticks_of(b, scale_b));
    super::spill_ref::join_topk((&ticks_a, &had_a), (&ticks_b, &had_b), m, k)
}

/// Whether a pair is on one of `diagonals` (`column − row`) or touches a window with no distance
/// in the hot passes, of the rows' series (`rows`) or the neighbors' (`cols`).
fn anytime_pair<'a>(
    diagonals: &'a [i64],
    rows: &'a [bool],
    cols: &'a [bool],
) -> impl Fn(usize, usize) -> bool + 'a {
    move |i, j| diagonals.contains(&(j as i64 - i as i64)) || rows[i] || cols[j]
}

/// An anytime self-join of `series` after visiting `diagonals` (`|j − i|`): the exact two-sided
/// top-k profile over the pairs on those diagonals and every pair that touches a window with no
/// distance in the hot passes.
pub fn anytime_two_sided_topk(
    series: &[f64],
    scale: Scale,
    m: usize,
    k: usize,
    diagonals: &[i64],
) -> Vec<(Bits, Bits, Vec<Bits>)> {
    let (ticks, had) = ticks_of(series, scale);
    let none = super::spill_ref::no_distance(&ticks, &had, m);
    let both: Vec<i64> = diagonals.iter().flat_map(|&d| [d, -d]).collect();
    let keep = anytime_pair(&both, &none, &none);
    super::spill_ref::two_sided_topk_where(&ticks, &had, m, k, &keep)
}

/// An anytime join of `a` against `b` after visiting `diagonals` (a window of `b` minus a window
/// of `a`): the exact top-k join over the pairs on those diagonals and every pair that touches a
/// window of either series with no distance in the hot passes.
pub fn anytime_join_topk(
    (a, scale_a): (&[f64], Scale),
    (b, scale_b): (&[f64], Scale),
    m: usize,
    k: usize,
    diagonals: &[i64],
) -> Vec<Vec<Bits>> {
    let ((ticks_a, had_a), (ticks_b, had_b)) = (ticks_of(a, scale_a), ticks_of(b, scale_b));
    let none_a = super::spill_ref::no_distance(&ticks_a, &had_a, m);
    let none_b = super::spill_ref::no_distance(&ticks_b, &had_b, m);
    let keep = anytime_pair(diagonals, &none_a, &none_b);
    super::spill_ref::join_topk_where((&ticks_a, &had_a), (&ticks_b, &had_b), m, k, &keep)
}
