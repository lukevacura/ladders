#![cfg(not(any(loom, shuttle)))]
//! The streams rank the windows their hot arithmetic cannot (one holding a withheld outlier
//! tick, one whose variance numerator passes 2⁶³) by the exact cold route: every row equals the
//! exact reference over the same samples, bounded to the last `w` samples or growing, for any
//! `k`; checkpoints taken mid-shock continue exactly; the bounded stream allocates nothing once
//! its first row with neighbors has reserved its scratch; batch and both streams flag the same
//! ticks as outliers.

#[path = "common/spill_ref.rs"]
mod spill_ref;

use ladde::bounded::BoundedStream;
use ladde::checkpoint::with_recomputed_checksum;
use ladde::entry::Entry;
use ladde::quantize::{Scale, to_ticks};
use ladde::spill::{lone_deviation_limit, outlier_ticks};
use ladde::stream::GrowingStream;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use std::num::NonZeroUsize;
use testkit::alloc_count::{CountingAllocator, allocations_in};
use testkit::generators::seeded;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

type Bits = spill_ref::Bits;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn bits(row: u64, e: Entry) -> Bits {
    (
        row,
        e.rho().to_bits(),
        e.squared_distance().to_bits(),
        e.index().get().unwrap_or(u64::MAX),
    )
}

fn ticks_of(series: &[f64]) -> (Vec<i64>, Vec<bool>) {
    let mut last = 0;
    let measured: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, unit())).collect();
    let ticks = measured
        .iter()
        .map(|t| {
            last = t.unwrap_or(last);
            last
        })
        .collect();
    (ticks, measured.iter().map(Option::is_some).collect())
}

/// The exact left profile, on only the last `w` samples of each row when given.
fn reference(series: &[f64], m: usize, w: Option<usize>) -> Vec<Bits> {
    let (ticks, had) = ticks_of(series);
    match w {
        Some(w) => spill_ref::left_profile_where(&ticks, &had, m, &|i, j| i - j <= w - m),
        None => spill_ref::left_profile(&ticks, &had, m),
    }
}

/// The exact `k` nearest of every row, on only the last `w` samples of each row when given.
fn reference_topk(series: &[f64], m: usize, w: Option<usize>, k: usize) -> Vec<Vec<Bits>> {
    let (ticks, had) = ticks_of(series);
    let Some(w) = w else {
        return spill_ref::left_topk(&ticks, &had, m, k);
    };
    (0..(ticks.len() + 1).saturating_sub(m))
        .map(|i| {
            let start = (i + m).saturating_sub(w);
            let local = spill_ref::left_topk(&ticks[start..i + m], &had[start..i + m], m, k);
            local
                .last()
                .expect("the slice holds row i's window")
                .iter()
                .map(|&(_, rho, distance, index)| (i as u64, rho, distance, index + start as u64))
                .collect()
        })
        .collect()
}

fn bounded(series: &[f64], m: usize, w: usize) -> Vec<Bits> {
    let mut stream = BoundedStream::new(nz(m), nz(w), unit()).unwrap();
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, e)| bits(row.get().unwrap(), e))
        .collect()
}

fn growing(series: &[f64], m: usize) -> Vec<Bits> {
    let mut stream = GrowingStream::new(nz(m), unit());
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, e)| bits(row.get().unwrap(), e))
        .collect()
}

fn bounded_topk(series: &[f64], m: usize, w: usize, k: usize) -> Vec<Vec<Bits>> {
    let mut stream = BoundedStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
    series
        .iter()
        .filter_map(|&x| {
            let (row, kept) = stream.update_topk(x)?;
            let row = row.get().unwrap();
            Some(kept.iter().map(|&e| bits(row, e)).collect())
        })
        .collect()
}

fn growing_topk(series: &[f64], m: usize, k: usize) -> Vec<Vec<Bits>> {
    let mut stream = GrowingStream::with_k(nz(m), unit(), nz(k));
    series
        .iter()
        .filter_map(|&x| {
            let (row, kept) = stream.update_topk(x)?;
            let row = row.get().unwrap();
            Some(kept.iter().map(|&e| bits(row, e)).collect())
        })
        .collect()
}

/// `got` equals `expected`, or the first differing row is reported.
fn same<T: PartialEq + std::fmt::Debug>(got: &[T], expected: &[T], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: rows");
    if let Some((at, (g, e))) = got
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (g, e))| g != e)
    {
        panic!("{what}: row {at} got {g:?} expected {e:?}");
    }
}

fn walk(len: usize, seed: u64, level: f64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut value = level;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            value += (state % 9) as f64 - 4.0;
            value
        })
        .collect()
}

/// Shocked walks: one print, adjacent prints, opposite signs, a run longer than `m`, prints at
/// the first and last samples and beside gaps; at levels where the multiple is an outlier
/// (withheld) and where it is not (the exact i64 arithmetic, or windows past `2⁶³`).
fn shocked(m: usize, len: usize) -> Vec<(String, Vec<f64>)> {
    let mut out = Vec::new();
    for &level in &[1e5, 1e7, 1e9, 1e12] {
        for &times in &[1.5, 10.0, 1000.0] {
            let mut one = walk(len, m as u64 + 1, level);
            one[len / 3] *= times;
            out.push((format!("{level} x{times} one"), one));
            let mut many = walk(len, m as u64 + 2, level);
            for at in [1, 2, len / 4, len / 4 + 1, len / 2, len - 1] {
                many[at] *= times;
            }
            many[len / 2 + 1] = -many[len / 2 + 1] * times;
            many[len / 4 - 1] = f64::NAN;
            many[len / 2 + 2] = f64::NAN;
            out.push((format!("{level} x{times} many"), many));
            let mut run = walk(len, m as u64 + 3, level);
            for value in &mut run[len / 3..len / 3 + m + 3] {
                *value *= times;
            }
            out.push((format!("{level} x{times} run"), run));
        }
    }
    out
}

#[test]
fn shocks_equal_the_exact_reference() {
    for (m, w) in [(2, 9), (5, 23), (8, 40), (13, 60)] {
        for (name, series) in shocked(m, 140) {
            same(
                &bounded(&series, m, w),
                &reference(&series, m, Some(w)),
                &format!("bounded m {m} w {w} {name}"),
            );
            same(
                &growing(&series, m),
                &reference(&series, m, None),
                &format!("growing m {m} {name}"),
            );
        }
    }
}

#[test]
fn shocks_leave_the_horizon_and_the_bounded_stream_returns_to_narrow() {
    let (m, w) = (8, 30);
    let mut series = walk(400, 9, 1e6);
    series[40] += 3e9;
    series[41] -= 3e9;
    series[200] += 8e9;
    let mut stream = BoundedStream::new(nz(m), nz(w), unit()).unwrap();
    let got: Vec<Bits> = series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, e)| bits(row.get().unwrap(), e))
        .collect();
    same(&got, &reference(&series, m, Some(w)), "bounded");
    let counts = stream.route_counts();
    assert_eq!(counts.withheld_ticks, 3);
    assert!(counts.cold_rows > 0);
    assert_eq!(
        counts.wide_updates, 0,
        "a withheld outlier keeps the narrow arithmetic"
    );
}

/// A staircase climbing 1.6e9 every two samples: at `m = 4` no tick is far enough from the one
/// before to be withheld, but a window spanning a step has a variance numerator past 2⁶³.
fn staircase(len: usize) -> Vec<f64> {
    (0..len)
        .map(|k| (k / 2) as f64 * 1.6e9 + (k % 3) as f64)
        .collect()
}

#[test]
fn a_window_too_wide_with_no_outlier_gets_an_exact_distance() {
    let (series, m) = (staircase(60), 4);
    same(
        &growing(&series, m),
        &reference(&series, m, None),
        "growing",
    );
    same(
        &bounded(&series, m, 20),
        &reference(&series, m, Some(20)),
        "bounded",
    );
    let mut stream = GrowingStream::new(nz(m), unit());
    for &x in &series {
        let _ = stream.update(x);
    }
    assert_eq!(stream.route_counts().withheld_ticks, 0);
    assert!(stream.route_counts().cold_rows > 0);
}

#[test]
fn long_windows_run_the_256_bit_tier() {
    let m = 1030;
    let mut series = walk(1100, 5, 1e13);
    series[1040] *= 3.0;
    series[600] *= -2.0;
    same(
        &growing(&series, m),
        &reference(&series, m, None),
        "growing",
    );
    same(
        &bounded(&series, m, 1080),
        &reference(&series, m, Some(1080)),
        "bounded",
    );
}

#[test]
fn every_k_equals_the_exact_reference() {
    for (m, w) in [(4, 25), (8, 40)] {
        for (name, series) in shocked(m, 90).into_iter().step_by(3) {
            for k in [1, 2, 5, 40] {
                same(
                    &bounded_topk(&series, m, w, k),
                    &reference_topk(&series, m, Some(w), k),
                    &format!("bounded m {m} k {k} {name}"),
                );
                same(
                    &growing_topk(&series, m, k),
                    &reference_topk(&series, m, None, k),
                    &format!("growing m {m} k {k} {name}"),
                );
            }
        }
    }
}

#[test]
fn update_exact_reports_what_update_does_past_two_to_the_63_too() {
    let m = 6;
    let mut series = walk(120, 3, 1e6);
    series[50] += 4e9;
    // The fuzz case: a window past 2⁶³ with no withheld tick, at a tiny scale.
    let tiny: Vec<f64> = vec![0.0, 3e-303, 0.0, 3e-303, 0.0, 3e-303];
    for (series, m, scale) in [
        (&series, m, unit()),
        (&tiny, 2, Scale::new(1e-318).unwrap()),
    ] {
        let mut bounded = BoundedStream::new(nz(m), nz(40), scale).unwrap();
        let mut bounded_twin = BoundedStream::new(nz(m), nz(40), scale).unwrap();
        let mut growing = GrowingStream::new(nz(m), scale);
        let mut growing_twin = GrowingStream::new(nz(m), scale);
        let mut ranked = 0;
        for &x in series {
            for (exact, plain) in [
                (bounded.update_exact(x), bounded_twin.update(x)),
                (growing.update_exact(x), growing_twin.update(x)),
            ] {
                let (Some((row, exact)), Some((_, entry))) = (exact, plain) else {
                    continue;
                };
                let got = exact.map_or(Entry::NONE, |exact| exact.entry());
                assert_eq!(bits(0, got), bits(0, entry), "row {row:?}");
                ranked += usize::from(entry.index().get().is_some());
            }
        }
        assert!(ranked > 0);
    }
}

#[test]
fn discords_score_windows_past_two_to_the_63() {
    use ladde::discord::DiscordTracker;
    // A wave with one print: the windows holding it have variance numerators past 2⁶³, and the
    // top discord is the window farthest from its nearest by the exact reference.
    let m = 8;
    let mut series: Vec<f64> = (0..200)
        .map(|k| (1e6 + 1e3 * ((k % 20) as f64 * 0.314).sin()).round())
        .collect();
    series[120] += 6e9;
    let expected = reference(&series, m, None);
    let farthest = expected
        .iter()
        .filter(|&&(_, _, _, index)| index != u64::MAX)
        .max_by(|a, b| f64::from_bits(a.2).total_cmp(&f64::from_bits(b.2)))
        .unwrap()
        .0;
    let mut tracker = DiscordTracker::new(nz(m), 1);
    let mut stream = GrowingStream::new(nz(m), unit());
    let mut scored_past = 0;
    for &x in &series {
        if let Some((row, score)) = stream.update_exact(x) {
            let holds = (113..=120).contains(&row.get().unwrap());
            scored_past += usize::from(holds && score.is_some());
            tracker.update(row, score);
        }
    }
    assert_eq!(scored_past, 8);
    assert_eq!(tracker.top()[0].index().get(), Some(farthest));
}

#[test]
fn checkpoints_mid_shock_continue_exactly() {
    for (m, w) in [(8, 40), (5, 23)] {
        let mut series = walk(150, m as u64, 1e6);
        for &at in &[30usize, 31, 70] {
            series[at] += 3e9;
        }
        series[32] = f64::NAN;
        series[50] = f64::NAN;
        for k in [1, 3] {
            let expected = bounded_topk(&series, m, w, k);
            let expected_growing = growing_topk(&series, m, k);
            for split in [25, 31, 32, 33, 45, 72, 80, 120] {
                let mut stream = BoundedStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
                let mut growing = GrowingStream::with_k(nz(m), unit(), nz(k));
                let mut rows = Vec::new();
                let mut rows_growing = Vec::new();
                for &x in &series[..split] {
                    if let Some((row, kept)) = stream.update_topk(x) {
                        rows.push(kept.iter().map(|&e| bits(row.get().unwrap(), e)).collect());
                    }
                    if let Some((row, kept)) = growing.update_topk(x) {
                        rows_growing
                            .push(kept.iter().map(|&e| bits(row.get().unwrap(), e)).collect());
                    }
                }
                let bytes = stream.checkpoint();
                let mut restored = BoundedStream::restore(&bytes).unwrap();
                assert_eq!(restored.checkpoint(), bytes, "m {m} split {split}");
                let mut restored_growing = GrowingStream::restore(&growing.checkpoint()).unwrap();
                for &x in &series[split..] {
                    if let Some((row, kept)) = restored.update_topk(x) {
                        rows.push(kept.iter().map(|&e| bits(row.get().unwrap(), e)).collect());
                    }
                    if let Some((row, kept)) = restored_growing.update_topk(x) {
                        rows_growing
                            .push(kept.iter().map(|&e| bits(row.get().unwrap(), e)).collect());
                    }
                }
                same(
                    &rows,
                    &expected,
                    &format!("bounded m {m} k {k} split {split}"),
                );
                same(
                    &rows_growing,
                    &expected_growing,
                    &format!("growing m {m} k {k} split {split}"),
                );
            }
        }
    }
}

#[test]
fn a_clean_checkpoint_keeps_its_version_and_a_withheld_tick_does_not_change_one() {
    let m = 8;
    let clean = walk(80, 1, 1e6);
    let mut shocked = clean.clone();
    shocked[60] += 3e9;
    for series in [&clean, &shocked] {
        let mut stream = BoundedStream::new(nz(m), nz(40), unit()).unwrap();
        let mut growing = GrowingStream::new(nz(m), unit());
        for &x in series.iter() {
            let _ = stream.update(x);
            let _ = growing.update(x);
        }
        // The samples as they came, in the format a stream of one neighbor always wrote.
        assert_eq!(stream.checkpoint()[4], 3);
        assert_eq!(growing.checkpoint()[4], 3);
    }
}

#[test]
fn a_constant_flag_on_a_window_past_two_to_the_63_survives_a_checkpoint() {
    let (series, m) = (staircase(40), 4);
    let flag = |k: usize| Some(k.is_multiple_of(5));
    let run = |split: usize| {
        let mut stream = BoundedStream::new(nz(m), nz(20), unit()).unwrap();
        assert_eq!(stream.window_length().get(), m);
        let mut rows = Vec::new();
        for (k, &x) in series[..split].iter().enumerate() {
            rows.extend(
                stream
                    .update_flagged(x, flag(k))
                    .map(|(r, e)| bits(r.get().unwrap(), e)),
            );
        }
        let bytes = stream.checkpoint();
        let mut restored = BoundedStream::restore(&bytes).unwrap();
        for (k, &x) in series.iter().enumerate().skip(split) {
            rows.extend(
                restored
                    .update_flagged(x, flag(k))
                    .map(|(r, e)| bits(r.get().unwrap(), e)),
            );
        }
        (rows, bytes)
    };
    let run_growing = |split: usize| {
        let mut stream = GrowingStream::new(nz(m), unit());
        let mut rows = Vec::new();
        for (k, &x) in series[..split].iter().enumerate() {
            rows.extend(
                stream
                    .update_flagged(x, flag(k))
                    .map(|(r, e)| bits(r.get().unwrap(), e)),
            );
        }
        let mut restored = GrowingStream::restore(&stream.checkpoint()).unwrap();
        for (k, &x) in series.iter().enumerate().skip(split) {
            rows.extend(
                restored
                    .update_flagged(x, flag(k))
                    .map(|(r, e)| bits(r.get().unwrap(), e)),
            );
        }
        rows
    };
    let expected_growing = run_growing(series.len());
    for split in 4..series.len() {
        same(
            &run_growing(split),
            &expected_growing,
            &format!("growing split {split}"),
        );
    }
    let (expected, _) = run(series.len());
    let mut flagged_version = false;
    for split in 4..series.len() {
        let (rows, bytes) = run(split);
        same(&rows, &expected, &format!("split {split}"));
        flagged_version |= bytes[4] == 5;
        // A version-5 checkpoint whose flags are all cleared is not one a stream writes.
        if bytes[4] == 5 {
            let windows = (split + 1 - m).min(20 + 1 - m);
            let mut cleared = bytes.clone();
            let end = cleared.len() - 8;
            cleared[end - windows.div_ceil(8)..end].fill(0);
            assert!(BoundedStream::restore(&with_recomputed_checksum(&cleared)).is_err());
        }
    }
    assert!(
        flagged_version,
        "some checkpoint holds a flagged window past 2⁶³"
    );
}

#[test]
fn the_bounded_stream_allocates_nothing_after_creation_through_shocks() {
    for (m, w) in [(8, 40), (5, 300), (1030, 1100)] {
        for k in [1, 4] {
            let mut series = walk(2000.max(3 * w), 4, 1e6);
            for at in (w + 7..series.len()).step_by(97) {
                series[at] *= 2000.0;
            }
            for at in (w + 50..series.len()).step_by(211) {
                series[at] = f64::NAN;
            }
            let mut stream = BoundedStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
            // The hot arithmetic reserves its row scratch by the first row with neighbors.
            for &x in &series[..w] {
                let _ = stream.update_topk(x);
            }
            let (_, count) = allocations_in(|| {
                for &x in &series[w..] {
                    let _ = stream.update_topk(x);
                }
            });
            assert_eq!(count, 0, "m {m} w {w} k {k}");
            assert!(stream.route_counts().withheld_ticks > 0);
        }
    }
}

#[test]
fn exact_ties_between_hot_and_cold_neighbors_go_to_the_smaller_index() {
    // A constant row is equally near every valid window: those the hot arithmetic ranks come
    // first, those holding the withheld print after them.
    let m = 4;
    let mut series = walk(40, 11, 1e6);
    series[30] += 5e9;
    series.extend([1e6; 12]);
    for w in [30, 60] {
        same(
            &bounded(&series, m, w),
            &reference(&series, m, Some(w)),
            &format!("bounded w {w}"),
        );
        for k in [2, 6] {
            same(
                &bounded_topk(&series, m, w, k),
                &reference_topk(&series, m, Some(w), k),
                &format!("bounded w {w} k {k}"),
            );
        }
    }
    same(
        &growing(&series, m),
        &reference(&series, m, None),
        "growing",
    );
    for k in [2, 6] {
        same(
            &growing_topk(&series, m, k),
            &reference_topk(&series, m, None, k),
            &format!("growing k {k}"),
        );
    }
}

/// Series of extreme ticks at `scale`: a large print after a small one (a fuzz case), long
/// alternations, gaps and repeats.
fn extremes(scale: f64) -> Vec<Vec<f64>> {
    let big = 2f64.powi(52) * scale;
    vec![
        vec![7.29111875131153e-304, 0.0],
        vec![0.0, big, 0.0, -big, big, f64::NAN, big, 0.0, 0.0, -big],
        (0..40)
            .map(|k| {
                if k % 3 == 0 {
                    big
                } else {
                    (k % 5) as f64 * scale
                }
            })
            .collect(),
        vec![big; 12],
    ]
}

#[test]
fn the_shortest_windows_and_buffers_at_tiny_scales_never_panic_and_stay_exact() {
    use ladde::nonnorm_bounded::BoundedNonNormStream;
    use ladde::nonnorm_stream::GrowingNonNormStream;
    for raw in [5.7928e-319, 1e-300, 1.0] {
        let scale = Scale::new(raw).unwrap();
        for series in extremes(raw) {
            for (m, w) in [(1, 1), (1, 2), (2, 2), (3, 3), (2, 3)] {
                for k in [1, 3] {
                    let mut bounded = BoundedStream::with_k(nz(m), nz(w), scale, nz(k)).unwrap();
                    let mut growing = GrowingStream::with_k(nz(m), scale, nz(k));
                    for (at, &x) in series.iter().enumerate() {
                        let _ = bounded.update_topk(x);
                        let _ = growing.update_topk(x);
                        if at == series.len() / 2 {
                            bounded = BoundedStream::restore(&bounded.checkpoint()).unwrap();
                            growing = GrowingStream::restore(&growing.checkpoint()).unwrap();
                        }
                    }
                }
                let mut bounded = BoundedNonNormStream::new(nz(m), nz(w), scale).unwrap();
                let mut growing = GrowingNonNormStream::new(nz(m), scale);
                for &x in &series {
                    let _ = bounded.update(x);
                    let _ = growing.update(x);
                }
            }
        }
    }
    // At unit scale every sample is its own tick, so the rows meet the reference too.
    for series in extremes(1.0) {
        for (m, w) in [(1, 1), (1, 2), (2, 2), (2, 3)] {
            let what = format!("m {m} w {w}");
            same(
                &bounded(&series, m, w),
                &reference(&series, m, Some(w)),
                &what,
            );
            same(&growing(&series, m), &reference(&series, m, None), &what);
        }
    }
}

/// Per sample of `series`, whether a stream fed it withheld that sample, read from its route
/// counts.
fn withheld_by(series: &[f64], mut update: impl FnMut(f64) -> u64) -> Vec<bool> {
    let mut before = 0;
    series
        .iter()
        .map(|&x| {
            let after = update(x);
            let withheld = after > before;
            before = after;
            withheld
        })
        .collect()
}

/// Integer series of every shape the outlier rule decides on, with a window length and a bounded
/// stream's buffer: walks, few levels and short repeats at a level, with shocks (prints at a
/// multiple of the price, runs of them shorter and longer than `m`, steps at and just past the
/// limit) and gaps, single and in a run long enough for a bounded stream to forget the level.
fn rule_series() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    let levels = prop::sample::select(vec![0.0, 1e3, 1e7, 1e9, 1e12]);
    let shock = (any::<Index>(), 0usize..7, 1usize..30);
    (
        (1usize..14, 0usize..40, 0u8..3, levels),
        vec(0u8..=255, 60..300),
        vec(shock, 0..8),
        (vec(any::<Index>(), 0..60), any::<Index>(), 0usize..150),
    )
        .prop_map(
            |((m, extra, shape, level), steps, shocks, (gaps, run_at, run))| {
                let limit = lone_deviation_limit(m).min(1 << 40) as f64;
                let mut walk = 0.0;
                let mut series: Vec<f64> = steps
                    .iter()
                    .enumerate()
                    .map(|(i, &step)| {
                        walk += f64::from(step % 9) - 4.0;
                        level
                            + match shape {
                                0 => walk,
                                1 => f64::from(step % 3),
                                _ => (i % (2 + usize::from(step % 2) * 5)) as f64,
                            }
                    })
                    .collect();
                let len = series.len();
                for (at, kind, run) in shocks {
                    let at = at.index(len);
                    for value in &mut series[at..(at + run).min(len)] {
                        *value = match kind {
                            0 => *value * 1.5,
                            1 => *value * 10.0,
                            2 => *value * 1000.0,
                            3 => -*value * 1000.0,
                            4 => *value + limit,
                            5 => *value + limit + 1.0,
                            _ => *value - limit - 1.0,
                        }
                        .round();
                    }
                }
                for at in gaps {
                    series[at.index(len)] = f64::NAN;
                }
                let run_at = run_at.index(len);
                for value in &mut series[run_at..(run_at + run).min(len)] {
                    *value = f64::NAN;
                }
                (series, m, m + extra)
            },
        )
}

proptest! {
    #![proptest_config(seeded(96))]

    /// Batch and both streams flag the same ticks as outliers on the same samples: the growing
    /// stream over everything it was fed, the bounded one over everything too, and once restored
    /// from a checkpoint over the samples it holds and those fed after.
    #[test]
    fn batch_and_streams_classify_the_same_ticks(
        (series, m, w) in rule_series(),
        cut in any::<Index>(),
    ) {
        let (ticks, had) = ticks_of(&series);
        let batch = outlier_ticks(&ticks, &had, m).unwrap();
        let cut = cut.index(series.len());
        let (before, after) = series.split_at(cut);

        let mut growing = GrowingStream::new(nz(m), unit());
        let mut by_growing = withheld_by(before, |x| {
            growing.update(x);
            growing.route_counts().withheld_ticks
        });
        let mut growing = GrowingStream::restore(&growing.checkpoint()).unwrap();
        by_growing.extend(withheld_by(after, |x| {
            growing.update(x);
            growing.route_counts().withheld_ticks
        }));
        prop_assert_eq!(&by_growing, &batch, "growing, m {}, restored at {}", m, cut);

        let mut bounded = BoundedStream::new(nz(m), nz(w), unit()).unwrap();
        let by_bounded = withheld_by(&series, |x| {
            bounded.update(x);
            bounded.route_counts().withheld_ticks
        });
        prop_assert_eq!(&by_bounded, &batch, "bounded, m {} w {}", m, w);

        let mut bounded = BoundedStream::new(nz(m), nz(w), unit()).unwrap();
        for &x in before {
            bounded.update(x);
        }
        let mut bounded = BoundedStream::restore(&bounded.checkpoint()).unwrap();
        let held = cut.min(w);
        let (ticks, had) = ticks_of(&series[cut - held..]);
        let own = outlier_ticks(&ticks, &had, m).unwrap();
        let by_restored = withheld_by(after, |x| {
            bounded.update(x);
            bounded.route_counts().withheld_ticks
        });
        prop_assert_eq!(
            &by_restored[..],
            &own[held..],
            "bounded restored at {}, m {} w {}",
            cut,
            m,
            w
        );
    }
}

#[test]
fn a_bounded_stream_measures_from_a_level_its_buffer_no_longer_holds() {
    // An outlier, then gaps until the buffer holds none of the level: a tick beside the outlier
    // fits the buffer's anchor, yet it is an outlier against the level, as batch says.
    let (m, w) = (3, 3);
    let far = lone_deviation_limit(m) as f64 * 2.0;
    // Long enough a gap that the stream leaves the exact i64 arithmetic for a narrow anchor
    // around the outlier the gap is filled with.
    let mut series = vec![5.0, 6.0, 5.0, far];
    series.extend([f64::NAN; 200]);
    series.extend([far + 1.0, 4.0, 5.0]);
    let (ticks, had) = ticks_of(&series);
    let batch = outlier_ticks(&ticks, &had, m).unwrap();
    let flagged: Vec<usize> = (0..series.len()).filter(|&p| batch[p]).collect();
    assert_eq!(flagged, [3, 204]);
    let mut stream = BoundedStream::new(nz(m), nz(w), unit()).unwrap();
    let by_bounded = withheld_by(&series, |x| {
        stream.update(x);
        stream.route_counts().withheld_ticks
    });
    assert_eq!(by_bounded, batch);
    same(
        &bounded(&series, m, w),
        &reference(&series, m, Some(w)),
        "bounded",
    );
}
