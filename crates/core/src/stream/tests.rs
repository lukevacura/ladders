extern crate std;

use super::GrowingStream;
use crate::entry::Entry;
use crate::naive::naive_left_profile;
use crate::quantize::{Scale, to_ticks};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec::Vec;
use testkit::generators::{MARKET_TICK, market_series_and_m};

fn bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn run(series: &[f64], m: usize, scale: Scale) -> Vec<(u64, Entry)> {
    let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, entry)| (row.get().expect("row is never NONE"), entry))
        .collect()
}

fn series_m_and_split() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    market_series_and_m(0..48).prop_flat_map(|(series, m)| {
        let len = series.len();
        (Just(series), Just(m), 0..=len)
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Appending arbitrary future samples never changes a row this node has already
    /// returned: the prefix-only run's entries reappear bit for bit at the same row once
    /// the rest of the series is fed to a fresh run.
    #[test]
    fn causality((series, m, split) in series_m_and_split()) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let from_prefix = run(&series[..split], m, scale);
        let from_full = run(&series, m, scale);

        for &(row, entry) in &from_prefix {
            let (_, later) = from_full
                .iter()
                .find(|&&(r, _)| r == row)
                .expect("a row returned from the prefix is still returned from the full run");
            prop_assert_eq!(bits(entry), bits(*later));
        }
    }
}

// ---------- checkpoint/restore ----------

/// Checkpoints a fresh stream at exactly `split` samples into `series`, restores, feeds the
/// rest, and asserts the result matches an uninterrupted run over the whole series, bit for
/// bit — row for row, not just in aggregate.
fn assert_checkpoint_resume_matches(series: &[f64], m: usize, scale: Scale, split: usize) {
    assert_checkpoint_resume_after(series, m, scale, split, |_| {});
}

/// [`assert_checkpoint_resume_matches`], with `at_checkpoint` first asserting the stream is
/// in the state the test claims to cover (a re-anchor, the switch to Wide) before it is
/// checkpointed, so a series that never gets there cannot pass vacuously.
fn assert_checkpoint_resume_after(
    series: &[f64],
    m: usize,
    scale: Scale,
    split: usize,
    at_checkpoint: impl FnOnce(&GrowingStream),
) {
    let uninterrupted = run(series, m, scale);

    let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
    let mut resumed: Vec<(u64, Entry)> = Vec::new();
    for &x in &series[..split] {
        if let Some((row, entry)) = stream.update(x) {
            resumed.push((row.get().expect("row is never NONE"), entry));
        }
    }
    at_checkpoint(&stream);
    let bytes = stream.checkpoint();
    let mut restored = GrowingStream::restore(&bytes).expect("a genuine checkpoint restores");
    for &x in &series[split..] {
        if let Some((row, entry)) = restored.update(x) {
            resumed.push((row.get().expect("row is never NONE"), entry));
        }
    }

    assert_eq!(resumed.len(), uninterrupted.len(), "m {m}, split {split}");
    for (&(row, entry), &(expected_row, expected_entry)) in resumed.iter().zip(&uninterrupted) {
        assert_eq!(row, expected_row, "m {m}, split {split}");
        assert_eq!(
            bits(entry),
            bits(expected_entry),
            "m {m}, split {split}, row {row}"
        );
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Checkpointing at any point and restoring, then feeding the rest, reproduces exactly
    /// what an uninterrupted stream over the same series would have returned: the one
    /// invariant this node exists to close. `split` ranges over the whole series,
    /// so warm-up (the window has not completed once yet) is exactly as likely as any
    /// later position.
    #[test]
    fn checkpoint_resume((series, m, split) in series_m_and_split()) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        assert_checkpoint_resume_matches(&series, m, scale, split);
    }
}

#[test]
fn checkpoint_resume_during_warm_up() {
    let m = 6;
    let scale = Scale::new(MARKET_TICK).unwrap();
    let series: Vec<f64> = (0..4).map(|k| 100.0 + k as f64 * 0.01).collect();
    assert_checkpoint_resume_matches(&series, m, scale, 2);
}

#[test]
fn checkpoint_resume_right_after_a_re_anchor() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;
    let warm_up = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0];
    let jump = [b + 1000.0, b + 1001.0, b + 999.0, b + 1002.0, b + 998.0];
    let tail = [1.0, -2.0, 3.0, 5.0, -4.0, 2.0, -1.0, 0.5];

    let mut series = warm_up.to_vec();
    series.extend_from_slice(&jump);
    let split = series.len(); // right after the jump, which re-anchors.
    series.extend_from_slice(&tail);

    assert_checkpoint_resume_after(&series, m, scale, split, |stream| {
        // The jump moved the anchor off its initial 0 without leaving the narrow path.
        assert_ne!(stream.anchor, Anchor { tick: 0 });
        assert_eq!(stream.path(), Path::Narrow);
    });
}

#[test]
fn checkpoint_resume_after_the_switch_to_wide() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;
    let warm_up = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0];
    let jump = [
        3.0 * b,
        3.0 * b + 1.0,
        3.0 * b - 1.0,
        3.0 * b + 2.0,
        3.0 * b - 2.0,
    ];
    let tail = [
        3.0 * b + 3.0,
        3.0 * b - 3.0,
        3.0 * b + 4.0,
        3.0 * b,
        3.0 * b + 1.0,
    ];

    let mut series = warm_up.to_vec();
    series.extend_from_slice(&jump);
    let split = series.len(); // right after the jump, which switches to Wide permanently.
    series.extend_from_slice(&tail);

    assert_checkpoint_resume_after(&series, m, scale, split, |stream| {
        assert_eq!(stream.path(), Path::Wide);
    });
}

// ---------- the narrow path: re-anchor, the switch to Wide, and real market data ----------

use super::{Anchor, Path};

/// `bound(m)`, independently of [`crate::narrow::Narrow::fits`]'s own arithmetic (mirrors
/// `narrow.rs`'s and `bounded.rs`'s own test helper of the same name).
fn bound(m: usize) -> i64 {
    i64::from(i32::MAX) / m as i64
}

use crate::spill::tests::spill_ref;

/// The exact rows a stream reports over `series` (the last `w` samples searched, when given):
/// every window with its samples gets its exact nearest, whatever its variance numerator.
fn exact_rows(series: &[f64], m: usize, w: Option<usize>, scale: Scale) -> Vec<spill_ref::Bits> {
    let mut last = 0;
    let measured: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let ticks: Vec<i64> = measured
        .iter()
        .map(|tick| {
            last = tick.unwrap_or(last);
            last
        })
        .collect();
    let had: Vec<bool> = measured.iter().map(Option::is_some).collect();
    match w {
        Some(w) => spill_ref::left_profile_where(&ticks, &had, m, &|i, j| i - j <= w - m),
        None => spill_ref::left_profile(&ticks, &had, m),
    }
}

/// Drives `series` through a fresh `GrowingStream`, checking every row against the exact
/// reference bit for bit as it goes, and returns the stream's `path()` after the very last
/// sample.
fn run_checked(series: &[f64], m: usize, scale: Scale) -> Path {
    let expected = exact_rows(series, m, None, scale);
    let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
    for &x in series {
        if let Some((row, entry)) = stream.update(x) {
            let row = row.get().expect("row is never NONE");
            let (_, rho, distance, index) = expected[row as usize];
            assert_eq!(bits(entry), (rho, distance, index), "m {m}, row {row}");
        }
    }
    stream.path()
}

#[test]
fn a_tick_past_the_anchor_s_bound_re_anchors_and_stays_narrow() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;

    // A varied (never-constant) warm-up, as in `bounded.rs`'s own version of this test:
    // every diagonal's `m·QT` carries a genuinely non-zero state into the re-anchor.
    let mut series = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0].to_vec();
    // The jump: the current anchor (still `0`) cannot hold this tick, but the whole
    // history's span (0..=b + 1002) still fits one re-anchored anchor (far short of
    // `2 * b`), so this must re-anchor, never switch to Wide.
    series.extend([b + 1000.0, b + 1001.0, b + 999.0, b + 1002.0, b + 998.0]);

    assert_eq!(run_checked(&series, m, scale), Path::Narrow);
}

#[test]
fn a_span_past_twice_the_bound_switches_to_wide_permanently() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;

    let mut series = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0].to_vec();
    // The jump: `3 * b` is more than `2 * b` away from the warm-up ticks (all within a
    // few units of `0`), so no anchor could cover both ends.
    series.extend([
        3.0 * b,
        3.0 * b + 1.0,
        3.0 * b - 1.0,
        3.0 * b + 2.0,
        3.0 * b - 2.0,
    ]);

    assert_eq!(run_checked(&series, m, scale), Path::Wide);
}

/// `m·tick` alone passes i64 here (4096 · 2.4e15; naive stays exact in i128 up to about this level), while every offset from the anchor stays
/// tiny: plain updates must neither overflow in a debug build nor leave the exact answer.
#[test]
fn a_large_window_at_a_large_tick_level_matches_naive() {
    let m = 4096;
    let scale = Scale::new(1.0).unwrap();
    let mut value = 2.4e15;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let series: Vec<f64> = (0..m + 24)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            value += ((state >> 33) % 5) as f64 - 2.0;
            value
        })
        .collect();
    assert_eq!(run_checked(&series, m, scale), Path::Narrow);
}

#[test]
fn a_span_of_exactly_twice_the_bound_stays_narrow() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m);
    // The whole history's own min (`0`) and max (`2 * b`) are exactly `2 * b` apart: the
    // tightest span a single anchor can still cover.
    let series = [
        0.0,
        1.0,
        2.0,
        (2 * b) as f64,
        (2 * b - 1) as f64,
        (2 * b - 2) as f64,
        1.0,
        2.0,
    ];
    assert_eq!(run_checked(&series, m, scale), Path::Narrow);
}

/// A leading NaN gap (so the first several ticks are the forward-fill placeholder `0`,
/// never a real value), then real data at a price level far beyond any anchor that could
/// still cover `0`: unlike `BoundedStream`'s own "window 0 leaving" placeholder, which
/// only one ring slot ever reads and which ages out once that slot is reused, growing mode
/// never evicts, so this placeholder is read by every window whose own `leave`/`enter`
/// lands on it, for as long as the stream runs. The span check must see it like any other
/// stored tick, so this switches to Wide, correctly, rather than trying (and failing) to
/// narrow a tick that stays live forever.
#[test]
fn a_leading_gap_before_real_distant_data_switches_to_wide() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let mut series = [f64::NAN; 10].to_vec();
    let level = 1e10;
    series.extend([level, level + 1.0, level - 1.0, level + 2.0, level - 2.0]);
    assert_eq!(run_checked(&series, m, scale), Path::Wide);
}

#[test]
fn a_span_one_past_twice_the_bound_switches_to_wide() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m);
    // One tick past the exact boundary above: no anchor can cover it any longer.
    let series = [
        0.0,
        1.0,
        2.0,
        (2 * b + 1) as f64,
        (2 * b) as f64,
        (2 * b - 1) as f64,
        1.0,
        2.0,
    ];
    assert_eq!(run_checked(&series, m, scale), Path::Wide);
}

/// A high-price-level market series (`1e10` ticks), NaN gaps scattered through it, and
/// `20_000+` samples: the real-data shape `BoundedStream`'s own placeholder bug used to
/// break on. `GrowingStream` keeps no ring and no placeholder tick at all (its span check
/// reads only real, already-sanitized ticks), so this is this node's own evidence that the
/// growing path never needs that exclusion in the first place, not a regression guard for
/// a bug it could have inherited. Checked against `naive_left_profile` on a short prefix
/// (`naive` is quadratic; the full 20,000+ samples would never finish) and `path()`
/// asserted `Narrow` at the end, over the whole series.
#[test]
fn market_level_series_at_a_high_price_stays_narrow() {
    let m = 50;
    let scale = Scale::new(1.0).unwrap();
    let level = 10_000_000_000i64;
    let len = 20_500usize;

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut level_offset = 0i64;
    let series: Vec<f64> = (0..len)
        .map(|k| {
            if k % 733 == 0 && k > 0 {
                return f64::NAN;
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let step = ((state >> 33) % 5) as i64 - 2; // -2..=2
            level_offset += step;
            (level + level_offset) as f64
        })
        .collect();

    let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
    let mut entries: Vec<(u64, Entry)> = Vec::new();
    for &x in &series {
        if let Some((row, entry)) = stream.update(x) {
            entries.push((row.get().expect("row is never NONE"), entry));
        }
    }
    assert_eq!(stream.path(), Path::Narrow);

    // A prefix short enough for `naive` to finish on: a few hundred samples, still deep
    // enough to exercise many live diagonals.
    let prefix_len = 400;
    let samples: Vec<Option<i64>> = series[..prefix_len]
        .iter()
        .map(|&x| to_ticks(x, scale))
        .collect();
    let expected = naive_left_profile(&samples, m);
    for &(row, expected_entry) in &expected {
        let (_, actual_entry) = entries
            .iter()
            .find(|&&(r, _)| r == row)
            .expect("every naive row is also returned by the stream");
        assert_eq!(bits(*actual_entry), bits(expected_entry), "row {row}");
    }
}

/// Mirrors `row::tests::exact_repeat_beats_a_near_miss_even_when_rho_hat_ties`, driven
/// through `GrowingStream` end to end instead of calling `advance_and_rank_narrow`
/// directly: only the exact `u128` fallback inside `Nearest`, reached through the near-tie
/// band, can tell the true exact match from the one-tick near miss once both ρ̂ clamp to
/// `1.0`. A candidate set narrowed to just the row's own maximum (dropping the `2ε` band)
/// would offer `Nearest` only the near miss and never reach the exact match at all.
#[test]
fn narrow_path_exact_repeat_beats_a_near_miss_even_when_rho_hat_ties() {
    const A: [f64; 4] = [235_732_534.0, -190_400_685.0, 248_938_659.0, -501_918_519.0];
    const B: [f64; 4] = [235_732_534.0, -190_400_685.0, 248_938_660.0, -501_918_519.0];
    const FILLER: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    let m = 4;
    let scale = Scale::new(1.0).unwrap();

    let mut series: Vec<f64> = Vec::new();
    series.extend_from_slice(&A); // window 0: the near miss.
    series.extend_from_slice(&FILLER);
    series.extend_from_slice(&B); // window 8: the exact match.
    series.extend_from_slice(&FILLER);
    series.extend_from_slice(&B); // window 16: the row, an exact repeat of window 8.

    let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
    let mut last: Option<Entry> = None;
    for &x in &series {
        if let Some((_, entry)) = stream.update(x) {
            last = Some(entry);
        }
    }
    assert_eq!(stream.path(), Path::Narrow);
    let entry = last.expect("the series completes at least one window");
    assert_eq!(entry.index().get(), Some(8));
    assert_eq!(entry.rho(), 1.0);
    assert_eq!(entry.squared_distance(), 0.0);
}

/// Like `run_checked`, but also records this stream's own `anchor` and `path` after every
/// sample: for a test that needs to see re-anchoring happen, not just infer it from the
/// final `path()`.
fn run_checked_with_anchors(series: &[f64], m: usize, scale: Scale) -> Vec<(Anchor, Path)> {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let expected = naive_left_profile(&samples, m);
    let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
    let mut trace = Vec::new();
    for &x in series {
        if let Some((row, entry)) = stream.update(x) {
            let row = row.get().expect("row is never NONE");
            let (_, expected_entry) = expected
                .iter()
                .find(|&&(r, _)| r == row)
                .expect("naive returns one entry per row too");
            assert_eq!(bits(entry), bits(*expected_entry), "m {m}, row {row}");
        }
        trace.push((stream.anchor, stream.path()));
    }
    trace
}

/// A market-level series (level ≈ 1e10, `m = 4`) that re-anchors at least three times —
/// each jump lands at an increasing fraction of `bound(m)` past the current anchor,
/// forcing `maintain_narrow_range` to re-center while the whole history's span still fits
/// one anchor — then a final jump pushes the span past `2 * bound(m)`, switching
/// permanently to `Wide`. Checked against `naive_left_profile` bit for bit on every row
/// (`run_checked`'s own loop, via `run_checked_with_anchors`), with the stream's `anchor`
/// also recorded after every sample so the re-anchor count and the final switch are both
/// directly observed, not just inferred from the end state. Catches a re-anchor that skips
/// `S → S − m·Δ` (the sum shift): the live diagonals this series carries through
/// each re-anchor would then carry a corrupted `m·QT` into every row after it, mismatching
/// `naive` well before the end.
#[test]
fn market_level_series_re_anchors_several_times_then_switches_to_wide() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let level = 10_000_000_000i64;
    let b = bound(m);

    let mut series: Vec<f64> = Vec::new();
    // Varied warm-up at `level`, as in the hand-worked re-anchor tests above: every live
    // diagonal's `m·QT` carries a genuinely non-zero state before the first jump.
    for &d in &[0i64, 1, -1, 2, -2, 1, -1, 0, 2, -2] {
        series.push((level + d) as f64);
    }
    // Three jumps, each at an increasing fraction of `b` past `level`: each one re-anchors
    // (the jump alone no longer fits the current anchor) without switching (the whole
    // history's span still fits one re-centered anchor).
    for &tenths in &[12i64, 18, 19] {
        let jump = level + (b * tenths) / 10;
        for &d in &[0i64, 1, -1, 2, -2] {
            series.push((jump + d) as f64);
        }
    }
    // The final jump: past `2 * b` from `level` (the history's own low end), so no anchor
    // could cover both ends any longer.
    let wide_jump = level + 2 * b + 1000;
    for &d in &[0i64, 1, -1, 2, -2] {
        series.push((wide_jump + d) as f64);
    }

    let trace = run_checked_with_anchors(&series, m, scale);
    assert_eq!(trace.last().map(|&(_, path)| path), Some(Path::Wide));

    let mut anchors_while_narrow: Vec<Anchor> = Vec::new();
    let mut saw_wide = false;
    for &(anchor, path) in &trace {
        match path {
            Path::Narrow => {
                assert!(!saw_wide, "path reverted from Wide back to Narrow");
                if anchors_while_narrow.last() != Some(&anchor) {
                    anchors_while_narrow.push(anchor);
                }
            }
            Path::Wide => saw_wide = true,
        }
    }
    assert!(
        anchors_while_narrow.len() >= 4,
        "expected the initial anchor plus at least 3 re-anchors (4 distinct anchors) \
         while Narrow, saw {}",
        anchors_while_narrow.len()
    );
}

// ---------- the front-growing `qt` buffer: no shift, but still exact across reallocations ----------

/// A 300-sample run (`m = 4`) grows `qt`'s eligible count past 190, crossing several of
/// `FrontGrowBuffer`'s own front-exhaustion reallocations (roughly doubling capacity, at
/// eligible counts 1, 5, 11, 23, 47, 95 and 191): every row still matches
/// `naive_left_profile` bit for bit, so a reallocation carries every live diagonal's state
/// across exactly, the same way `reanchor_live_qt` and `switch_to_wide` already have to.
#[test]
fn a_long_growing_run_crosses_several_reallocations_and_still_matches_naive() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let mut state = 0x5EED_u64;
    let mut value = 0i64;
    let series: Vec<f64> = (0..300)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let step = ((state >> 33) % 5) as i64 - 2; // -2..=2
            value += step;
            value as f64
        })
        .collect();
    assert_eq!(run_checked(&series, m, scale), Path::Narrow);
}
