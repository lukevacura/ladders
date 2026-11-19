extern crate std;

use super::BoundedStream;
use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::naive::{naive_left_profile, windowed_entry};
use crate::quantize::{Scale, to_ticks};
use crate::stream::GrowingStream;
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec::Vec;
use testkit::generators::{
    MARKET_OFFSET, MARKET_TICK, market_series_and_m, quantized, random_walk, with_gaps,
};

fn bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn run(series: &[f64], m: usize, w: usize, scale: Scale) -> Vec<(u64, Entry)> {
    let mut stream = BoundedStream::new(
        NonZeroUsize::new(m).unwrap(),
        NonZeroUsize::new(w).unwrap(),
        scale,
    )
    .expect("w is always m or more here");
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, entry)| (row.get().expect("row is never NONE"), entry))
        .collect()
}

/// The same series through a stream whose tile is `tile_len` lanes: chunking is invisible.
fn run_tiled(
    series: &[f64],
    m: usize,
    w: usize,
    scale: Scale,
    tile_len: usize,
) -> Vec<(u64, Entry)> {
    let mut stream = BoundedStream::new(
        NonZeroUsize::new(m).unwrap(),
        NonZeroUsize::new(w).unwrap(),
        scale,
    )
    .expect("w is always m or more here");
    stream.tile = std::vec![0.0; tile_len].into_boxed_slice();
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, entry)| (row.get().expect("row is never NONE"), entry))
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Tile lengths 1, 7, 64 and the default give the same bits, through ramp-up and a
    /// full ring alike.
    #[test]
    fn tile_length_never_changes_a_bit(
        (series, m) in market_series_and_m(8..80),
        extra in 0usize..40,
    ) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let w = (m + extra).min(series.len().max(m));
        let base = run(&series, m, w, scale);
        for tile_len in [1usize, 7, 64] {
            let tiled = run_tiled(&series, m, w, scale, tile_len);
            prop_assert_eq!(base.len(), tiled.len());
            for (a, b) in base.iter().zip(&tiled) {
                prop_assert_eq!(a.0, b.0);
                prop_assert_eq!(bits(a.1), bits(b.1));
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Before `w` samples have arrived, `BoundedStream` returns exactly what `GrowingStream`
    /// would: the bound has nothing to do yet.
    #[test]
    fn bounded_eq_growing_before_full((series, m) in market_series_and_m(0..48), extra in 0usize..32) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let w = m + extra;

        let mut bounded = BoundedStream::new(
            NonZeroUsize::new(m).unwrap(),
            NonZeroUsize::new(w).unwrap(),
            scale,
        )
        .expect("w is always m or more here");
        let mut growing = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);

        for (k, &x) in series.iter().enumerate() {
            let from_bounded = bounded.update(x);
            let from_growing = growing.update(x);
            let total_ticks = k + 1;
            if total_ticks <= w {
                prop_assert_eq!(
                    from_bounded.map(|(row, e)| (row.get(), bits(e))),
                    from_growing.map(|(row, e)| (row.get(), bits(e))),
                    "k {}, m {}, w {}", k, m, w
                );
            }
        }
    }
}

#[test]
fn window_of_exactly_w_equal_to_m_never_has_a_neighbor() {
    let scale = Scale::new(MARKET_TICK).unwrap();
    let m = NonZeroUsize::new(4).unwrap();
    let mut stream = BoundedStream::new(m, m, scale).expect("w == m is a valid shape");
    for x in [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0,
    ] {
        if let Some((_, entry)) = stream.update(x) {
            assert_eq!(bits(entry), bits(Entry::NONE));
        }
    }
}

#[test]
fn window_just_above_the_exclusion_zone_finds_its_one_candidate() {
    // m = 4, exclusion = 1, w = m + exclusion + 1 = 6: exactly one window (two samples
    // behind the oldest end of the bound) can ever be a candidate, and only once it clears
    // the exclusion zone.
    let scale = Scale::new(1.0).unwrap();
    let m = NonZeroUsize::new(4).unwrap();
    let w = NonZeroUsize::new(6).unwrap();
    let mut stream = BoundedStream::new(m, w, scale).expect("w is above m here");
    let series = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0];
    let mut last = Entry::NONE;
    for &x in &series {
        if let Some((_, entry)) = stream.update(x) {
            last = entry;
        }
    }
    // offset 2 is the only one ever eligible (offset 1 sits inside the exclusion zone,
    // offset 3 would need a window already past the w = 6 bound), so row 4's only
    // candidate is row 2.
    assert_eq!(last.index(), SampleIndex::new(2).unwrap());
}

/// A deterministic walk of `len` samples with a run of non-finite values straddling every
/// multiple of `w`: exactly where the ring keeps evicting, so forward-fill-across-eviction
/// and the missing-count bookkeeping both fire on nearly every one of the many evictions
/// this series causes, not just once or twice.
fn long_series_with_gaps_near_evictions(len: usize, w: usize) -> Vec<f64> {
    let mut series: Vec<f64> = (0..len)
        .map(|k| {
            let k = k as f64;
            (k * 0.37).sin() * 10.0 + (k * 0.071).cos() * 5.0 + k * 0.01
        })
        .collect();
    let mut at = w;
    while at + 1 < len {
        for offset in [-1i64, 0, 1] {
            let idx = (at as i64 + offset) as usize;
            series[idx] = f64::NAN;
        }
        at += w;
    }
    series
}

/// The same bit-for-bit check as `bounded_eq_naive_window`, but over series long enough
/// (thousands of samples against `w` of a few dozen) that the ring wraps and evicts
/// hundreds of times, with non-finite runs placed right where each eviction lands — not
/// just the handful of evictions the proptest above reaches with its short series and
/// small `w`.
#[test]
fn bounded_eq_naive_window_over_many_evictions() {
    let scale = Scale::new(MARKET_TICK).unwrap();
    let len = 2_500;
    for m in [4usize, 9] {
        let exclusion = crate::exclusion::zone(m);
        for w in [m + exclusion + 1, 50] {
            let series = long_series_with_gaps_near_evictions(len, w);
            let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
            let actual = run(&series, m, w, scale);
            for &(row, entry) in &actual {
                let expected = windowed_entry(&samples, m, w, row as usize);
                assert_eq!(bits(entry), bits(expected), "m {m}, w {w}, row {row}");
            }
        }
    }
}

/// A market-level offset (`~1e10`) for the ramp-up property test below: far past
/// `MARKET_OFFSET`, so its quantized walk exercises a genuinely different tick magnitude
/// than the other `naive`-comparison tests in this file.
const RAMPUP_HIGH_LEVEL: f64 = 1e10;

/// Series length for the property test below, biased toward the short end (`2..30`,
/// weight 3) where ramp-up's own edge cases live — the exclusion zone leaving zero or one
/// candidate, a ring that is still mostly empty — while still reaching the full `2..200`
/// range (weight 1 past 30) often enough to vary the shapes actually drawn.
fn rampup_length() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => 2usize..30,
        1 => 30usize..200,
    ]
}

/// A series of about `l` samples (a gapped shape splices in a few more of its own) and a
/// `w` drawn past both its actual length and `m` (so `BoundedStream::new` always accepts
/// it, even when the series is shorter than `m`), so [`BoundedStream::windows`]'s ring
/// never fills while it runs: every row it produces is a ramp-up row.
fn rampup_series_and_w(m: usize) -> impl Strategy<Value = (Vec<f64>, usize)> {
    rampup_length().prop_flat_map(move |l| {
        let len = l..=l;
        let shape = prop_oneof![
            random_walk(len.clone(), MARKET_OFFSET),
            quantized(len.clone(), MARKET_TICK, RAMPUP_HIGH_LEVEL),
            with_gaps(random_walk(len.clone(), MARKET_OFFSET)),
            with_gaps(quantized(len, MARKET_TICK, RAMPUP_HIGH_LEVEL)),
        ];
        shape.prop_flat_map(move |series| {
            let n = series.len().max(m);
            let w = (n + 1)..(4 * n + 1);
            (Just(series), w)
        })
    })
}

/// `m`, and a series/`w` pair drawn against that same `m` (see
/// [`rampup_series_and_w`]) — threaded through one strategy so `w` can depend on `m`.
fn rampup_case() -> impl Strategy<Value = (usize, Vec<f64>, usize)> {
    (3usize..12)
        .prop_flat_map(|m| rampup_series_and_w(m).prop_map(move |(series, w)| (m, series, w)))
}

proptest! {
    // Biased toward short series above, so the default case count already lands densely
    // on the short end; kept modest regardless, since `windowed_entry`'s own check is
    // cubic in the series length here (`w` never caps it, unlike the other proptests
    // above).
    #![proptest_config(testkit::generators::seeded(48))]
    /// This node's own guarantee: a bounded stream whose bound `w` exceeds the series
    /// length equals `windowed_entry` on bits at every row, on every case drawn here
    /// (short or long) — pinning `ring_pos`'s `slot - skew` mapping, the only thing that
    /// tells a ramp-up row's ring position apart from its raw slot number; an off-by-one
    /// there only shows up on short series, where the ring is still mostly empty, so
    /// nothing here may skip the check based on length. `w` is always drawn past the
    /// series' own length (`L+1..4L`), so `windows`'s ring never fills and every row
    /// checked here is a ramp-up row, over plain random walks, market-level (~1e10)
    /// walks, and both with NaN/inf gaps spliced in. Whenever `naive` itself finds a real
    /// neighbor anywhere past the exclusion zone, `BoundedStream` must have found at
    /// least one too (checked on every case, so the test cannot pass on empty output);
    /// once the series is long enough that a few gap-invalidated windows can't flip it,
    /// most rows must carry a real neighbor.
    #[test]
    fn bounded_eq_naive_window_during_rampup((m, series, w) in rampup_case()) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
        let actual = run(&series, m, w, scale);

        let mut total = 0usize;
        let mut with_neighbor = 0usize;
        for &(row, entry) in &actual {
            let expected = windowed_entry(&samples, m, w, row as usize);
            prop_assert_eq!(bits(entry), bits(expected), "m {}, w {}, row {}", m, w, row);
            total += 1;
            if !entry.index().is_none() {
                with_neighbor += 1;
            }
        }

        // Weak, unconditional guard: `naive_left_profile` over the whole series is
        // exactly `windowed_entry`'s own answer for every row here (its window is
        // never capped by `w`, since `w` always exceeds the series), so this is just a
        // single pass over the same reference rather than a new one. If it finds a real
        // neighbor anywhere past the exclusion zone, `actual` must have found at least
        // one too, however short `series` is — catching a `BoundedStream` that quietly
        // returns too few or no rows, which the bit-for-bit loop above cannot: it only
        // ever compares rows `actual` itself produced.
        let exclusion = crate::exclusion::zone(m);
        let naive_has_a_neighbor = naive_left_profile(&samples, m)
            .into_iter()
            .any(|(row, entry)| row as usize > exclusion && !entry.index().is_none());
        if naive_has_a_neighbor {
            prop_assert!(
                with_neighbor > 0,
                "m {}, w {}: naive found a neighbor somewhere but BoundedStream found none",
                m, w
            );
        }

        // Stronger majority check: only meaningful once the series is long enough that
        // even the worst case `with_gaps` can throw (at most three runs of at most three
        // bad ticks each, invalidating at most `3 * (3 + m - 1)` windows) still leaves
        // most rows with a genuine neighbor — a short series can legitimately fail this
        // on gaps alone, which is why it is gated and the check above is not.
        if series.len() > m + 100 {
            prop_assert!(
                with_neighbor * 2 > total,
                "m {}, w {}: only {}/{} rows had a neighbor",
                m, w, with_neighbor, total
            );
        }
    }
}

// ---------- checkpoint/restore ----------

/// Checkpoints a fresh stream at exactly `split` samples into `series`, restores, feeds
/// the rest, and asserts the result matches an uninterrupted run over the whole series,
/// bit for bit — row for row, not just in aggregate.
fn assert_checkpoint_resume_matches(
    series: &[f64],
    m: usize,
    w: usize,
    scale: Scale,
    split: usize,
) {
    assert_checkpoint_resume_after(series, m, w, scale, split, |_| {});
}

/// [`assert_checkpoint_resume_matches`], with `at_checkpoint` first asserting the stream is
/// in the state the test claims to cover (a re-anchor, the switch to Wide, an eviction)
/// before it is checkpointed, so a series that never gets there cannot pass vacuously.
fn assert_checkpoint_resume_after(
    series: &[f64],
    m: usize,
    w: usize,
    scale: Scale,
    split: usize,
    at_checkpoint: impl FnOnce(&BoundedStream),
) {
    let uninterrupted = run(series, m, w, scale);

    let mut stream = BoundedStream::new(
        NonZeroUsize::new(m).unwrap(),
        NonZeroUsize::new(w).unwrap(),
        scale,
    )
    .expect("w is always m or more here");
    let mut resumed: Vec<(u64, Entry)> = Vec::new();
    for &x in &series[..split] {
        if let Some((row, entry)) = stream.update(x) {
            resumed.push((row.get().expect("row is never NONE"), entry));
        }
    }
    at_checkpoint(&stream);
    let bytes = stream.checkpoint();
    let mut restored = BoundedStream::restore(&bytes).expect("a genuine checkpoint restores");
    for &x in &series[split..] {
        if let Some((row, entry)) = restored.update(x) {
            resumed.push((row.get().expect("row is never NONE"), entry));
        }
    }

    assert_eq!(
        resumed.len(),
        uninterrupted.len(),
        "m {m}, w {w}, split {split}"
    );
    for (&(row, entry), &(expected_row, expected_entry)) in resumed.iter().zip(&uninterrupted) {
        assert_eq!(row, expected_row, "m {m}, w {w}, split {split}");
        assert_eq!(
            bits(entry),
            bits(expected_entry),
            "m {m}, w {w}, split {split}, row {row}"
        );
    }
}

/// A market series, an `m`/`w` pair drawn the same way [`bounded_eq_naive_window`] does, and
/// a checkpoint position anywhere in the series (ramp-up included, since `split` ranges
/// over the whole length).
fn series_m_w_and_split() -> impl Strategy<Value = (Vec<f64>, usize, usize, usize)> {
    market_series_and_m(0..48).prop_flat_map(|(series, m)| {
        let exclusion = crate::exclusion::zone(m);
        let extras = prop_oneof![
            Just(0usize),
            Just(1),
            Just(2),
            Just(exclusion),
            Just(exclusion + 1),
            Just(5),
            Just(32),
        ];
        let len = series.len();
        (Just(series), Just(m), extras, 0..=len)
            .prop_map(|(series, m, extra, split)| (series, m, m + extra, split))
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Checkpointing at any point and restoring, then feeding the rest, reproduces exactly
    /// what an uninterrupted stream over the same series would have returned: the one
    /// invariant this node exists to close.
    #[test]
    fn checkpoint_resume((series, m, w, split) in series_m_w_and_split()) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        assert_checkpoint_resume_matches(&series, m, w, scale, split);
    }
}

#[test]
fn checkpoint_resume_mid_ramp_up() {
    let m = 5;
    let w = 50; // far exceeds the series below, so every row here is a ramp-up row.
    let scale = Scale::new(MARKET_TICK).unwrap();
    let series: Vec<f64> = (0..20).map(|k| 100.0 + (k as f64) * 0.01).collect();
    assert_checkpoint_resume_after(&series, m, w, scale, 8, |stream| {
        assert!(stream.rows > 0 && stream.windows.len() < stream.windows.capacity());
    });
}

#[test]
fn checkpoint_resume_right_after_an_eviction() {
    let m = 4;
    let w = 7; // max_offset = 3, capacity_windows = 4: row 4 is the first eviction.
    let scale = Scale::new(1.0).unwrap();
    let series: Vec<f64> = (0..20)
        .map(|k| (k as f64 * 0.37).sin() * 5.0 + k as f64 * 0.1)
        .collect();
    // Row 4 completes at the 8th sample (`m + capacity_windows`); split right after it.
    assert_checkpoint_resume_after(&series, m, w, scale, 8, |stream| {
        // Row 4 is the first to push a window out of the full ring.
        assert_eq!(stream.rows, 5);
        assert_eq!(stream.windows.len(), stream.windows.capacity());
    });
}

#[test]
fn checkpoint_resume_right_after_a_re_anchor() {
    let m = 4;
    let w = 14;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;
    let warm_up = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0];
    let jump = [b + 1000.0, b + 1001.0, b + 999.0, b + 1002.0, b + 998.0];
    let tail = [1.0, -2.0, 3.0, 5.0, -4.0, 2.0, -1.0, 0.5];

    let mut series = warm_up.to_vec();
    series.extend_from_slice(&jump);
    let split = series.len(); // right after the jump, which re-anchors.
    series.extend_from_slice(&tail);

    assert_checkpoint_resume_after(&series, m, w, scale, split, |stream| {
        assert_ne!(stream.anchor, crate::narrow::Anchor { tick: 0 });
        assert_eq!(stream.path(), Path::Narrow);
    });
}

#[test]
fn checkpoint_resume_after_the_switch_to_wide() {
    let m = 4;
    let w = 14;
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
    let split = series.len(); // right after the jump, which switches to Wide.
    series.extend_from_slice(&tail);

    assert_checkpoint_resume_after(&series, m, w, scale, split, |stream| {
        assert_eq!(stream.path(), Path::Wide);
    });
}

// ---------- re-anchor and the switch to Wide, forced mid-stream ----------

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

use super::Path;

/// Drives `series` through a fresh `BoundedStream`, checking every row against the exact
/// reference bit for bit as it goes (so a wrong re-anchor or a wide-switch that
/// drops state fails at the row it first goes wrong, not only in a final summary), and
/// returns the stream's `path()` after the very last sample.
fn run_checked(series: &[f64], m: usize, w: usize, scale: Scale) -> Path {
    run_checked_stream(series, m, w, scale).path()
}

/// [`run_checked`], returning the stream itself.
fn run_checked_stream(series: &[f64], m: usize, w: usize, scale: Scale) -> BoundedStream {
    let expected = exact_rows(series, m, Some(w), scale);
    let mut stream = BoundedStream::new(
        NonZeroUsize::new(m).unwrap(),
        NonZeroUsize::new(w).unwrap(),
        scale,
    )
    .expect("w is always m or more here");
    for &x in series {
        if let Some((row, entry)) = stream.update(x) {
            let row = row.get().expect("row is never NONE");
            let (_, rho, distance, index) = expected[row as usize];
            assert_eq!(
                bits(entry),
                (rho, distance, index),
                "m {m}, w {w}, row {row}"
            );
        }
    }
    stream
}

/// `bound(m)`, independently of `Narrow::fits`'s own arithmetic (mirrors `narrow.rs`'s own
/// test helper of the same name).
fn bound(m: usize) -> i64 {
    i64::from(i32::MAX) / m as i64
}

/// Like [`run_checked`], but for series long enough that checking every row against
/// `windowed_entry` (quadratic in `w`) would never finish: checks only every
/// `check_every`-th row, plus the very last one, driven over every sample regardless of
/// which rows get checked. Returns `path()` after the last sample, and how many times
/// `anchor()` actually changed value along the way (a re-anchor, or the one-time switch to
/// Wide, each count once; a tick that already fit the current anchor counts zero).
fn run_checked_sparse(
    series: &[f64],
    m: usize,
    w: usize,
    scale: Scale,
    check_every: u64,
) -> (Path, u64) {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let mut stream = BoundedStream::new(
        NonZeroUsize::new(m).unwrap(),
        NonZeroUsize::new(w).unwrap(),
        scale,
    )
    .expect("w is always m or more here");
    let mut last: Option<(u64, Entry)> = None;
    let mut anchor_changes = 0u64;
    let mut last_anchor = stream.anchor();
    for &x in series {
        if let Some((row, entry)) = stream.update(x) {
            let row = row.get().expect("row is never NONE");
            if row % check_every == 0 {
                let expected = windowed_entry(&samples, m, w, row as usize);
                assert_eq!(bits(entry), bits(expected), "m {m}, w {w}, row {row}");
            }
            last = Some((row, entry));
        }
        let anchor = stream.anchor();
        if anchor != last_anchor {
            anchor_changes += 1;
            last_anchor = anchor;
        }
    }
    if let Some((row, entry)) = last {
        let expected = windowed_entry(&samples, m, w, row as usize);
        assert_eq!(
            bits(entry),
            bits(expected),
            "m {m}, w {w}, row {row} (last)"
        );
    }
    (stream.path(), anchor_changes)
}

#[test]
fn a_tick_past_the_anchor_s_bound_re_anchors_and_stays_narrow() {
    let m = 4;
    let w = 14; // max_offset = 10, exclusion = 1: several diagonals are live before the jump.
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;

    // A varied (never-constant) warm-up: every diagonal's `m·QT` carries a genuinely
    // non-zero state into the re-anchor, so a `reanchor` that silently no-ops cannot pass
    // by the all-zero coincidence a constant warm-up would let through.
    let mut series = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0].to_vec();
    // The jump: anchor 0 cannot hold this tick (it exceeds `bound`), but the whole buffer's
    // span (0..=bound + 1000) still fits a single re-anchored anchor (it is far short of
    // `2 * bound`), so this must re-anchor, never switch to Wide.
    series.extend([b + 1000.0, b + 1001.0, b + 999.0, b + 1002.0, b + 998.0]);

    let path = run_checked(&series, m, w, scale);
    assert_eq!(path, Path::Narrow);
}

/// `warm` varied small ticks, `shock` ticks at `level`, then `tail` varied small ticks.
fn shocked_series(warm: usize, shock: usize, level: f64, tail: usize) -> Vec<f64> {
    let small = |k: usize| ((k * 5) % 11) as f64 - 5.0;
    (0..warm)
        .map(small)
        .chain((0..shock).map(|_| level))
        .chain((0..tail).map(small))
        .collect()
}

#[test]
fn a_shock_that_leaves_the_buffer_returns_to_narrow_exactly() {
    let (m, w) = (4, 14);
    let scale = Scale::new(1.0).unwrap();
    let series = shocked_series(30, 3, 3.0 * bound(m) as f64, 150);
    let stream = run_checked_stream(&series, m, w, scale);
    assert_eq!(stream.path(), Path::Narrow);
    let counts = stream.route_counts();
    assert_eq!((counts.switches_to_wide, counts.switches_to_narrow), (1, 1));
    assert!(counts.wide_updates > 0);
}

#[test]
fn a_span_hovering_inside_the_limit_never_returns_to_narrow() {
    let (m, w) = (4, 14);
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;
    // Wide first, then a spike of 1.5 bounds every 10 samples: always inside one anchor's
    // reach, never inside the margin a return needs.
    let mut series = shocked_series(30, 3, 3.0 * b, 20);
    series.extend((0..400).map(|k| if k % 10 == 0 { 1.5 * b } else { (k % 7) as f64 }));
    let stream = run_checked_stream(&series, m, w, scale);
    assert_eq!(stream.path(), Path::Wide);
    assert_eq!(stream.route_counts().switches_to_narrow, 0);
}

#[test]
fn repeated_shocks_switch_back_at_most_once_per_check() {
    let (m, w) = (4, 14);
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;
    let mut series = Vec::new();
    for round in 0..12 {
        series.extend(shocked_series(10 + round, 2, 3.0 * b, 12 + 7 * round));
    }
    let stream = run_checked_stream(&series, m, w, scale);
    let counts = stream.route_counts();
    assert!(counts.switches_to_narrow > 0);
    assert!(counts.switches_to_narrow <= counts.switches_to_wide);
    assert!(counts.switches_to_narrow <= series.len() as u64 / 64);
}

#[test]
fn checkpoint_resume_around_the_return_to_narrow() {
    let (m, w) = (4, 14);
    let scale = Scale::new(1.0).unwrap();
    let series = shocked_series(30, 3, 3.0 * bound(m) as f64, 150);
    let mut stream = BoundedStream::new(
        NonZeroUsize::new(m).unwrap(),
        NonZeroUsize::new(w).unwrap(),
        scale,
    )
    .unwrap();
    let returned_at = series
        .iter()
        .position(|&x| {
            stream.update(x);
            stream.route_counts().switches_to_narrow == 1
        })
        .expect("the shock leaves the buffer and the stream returns")
        + 1;
    for split in [
        returned_at - 20,
        returned_at - 1,
        returned_at,
        returned_at + 20,
    ] {
        assert_checkpoint_resume_after(&series, m, w, scale, split, |_| {});
    }
}

#[test]
fn a_span_past_twice_the_bound_switches_to_wide() {
    let m = 4;
    let w = 14;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m) as f64;

    // Varied, as above: a `switch_to_wide` that sets `path` without recomputing absolute
    // `QT` would otherwise go undetected by an all-zero (and so already-zero-QT) warm-up.
    let mut series = [1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0].to_vec();
    // The jump: 3 * bound is more than `2 * bound` away from the warm-up ticks (all within
    // a few units of `0`), so no anchor could cover both ends — this must switch to Wide,
    // and stay there until the next return check (every `max(m, 64)` rows), far past
    // this short tail.
    series.extend([
        3.0 * b,
        3.0 * b + 1.0,
        3.0 * b - 1.0,
        3.0 * b + 2.0,
        3.0 * b - 2.0,
    ]);

    let path = run_checked(&series, m, w, scale);
    assert_eq!(path, Path::Wide);
}

/// Five re-anchors in a row, each landing a plateau whose every live tick sits exactly at
/// the new anchor's `+bound` (the tightest span a re-anchor, rather than a switch to Wide,
/// can still cover), then one final jump of `2 * bound + 1` past the fifth plateau, which
/// no anchor could cover: this must switch to Wide. `w + 2` ticks flush each plateau fully
/// through the ring before the next jump, so every live diagonal is re-anchored (or, after
/// the last jump, converted) while sitting exactly on that boundary, in both `cargo test`
/// and `cargo test --release`: an anchor one off either way would push some live tick's
/// offset one past `bound`, caught here, bit for bit, by `run_checked`'s own check against
/// `windowed_entry` on every row.
#[test]
fn repeated_boundary_re_anchors_then_exact_switch_stay_bit_exact_in_release() {
    let m = 4;
    let w = 14;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m);
    let flush = w + 2;

    let mut series: Vec<f64> = Vec::new();
    let mut level = 0i64;
    series.extend(std::iter::repeat_n(level as f64, flush));

    for _ in 0..5 {
        level += 2 * b;
        series.push(level as f64);
        series.extend(std::iter::repeat_n(level as f64, flush));
    }

    // One jump of `2 * bound + 1` past the fifth plateau: no anchor could cover a span that
    // wide, so this must switch to Wide rather than re-anchor again.
    level += 2 * b + 1;
    series.push(level as f64);
    series.extend(std::iter::repeat_n(level as f64, flush));

    let path = run_checked(&series, m, w, scale);
    assert_eq!(path, Path::Wide);
}

/// A high-price-level market series: `1e10` ticks (so window 0's `leaving` placeholder, at
/// `0`, would poison the span check by ten billion ticks if it were ever folded in), a span
/// of only a few dozen ticks (so centering the anchor should settle it after at most a
/// handful of re-anchors, never forcing a switch to Wide), NaN gaps scattered through it,
/// and `20_000+` samples (long enough to run the ring around many times over, at `w =
/// 1_000`). Checked against `windowed_entry` sparsely (quadratic in `w`, so not on
/// every row) and `path()` asserted `Narrow` at the end — the real-data bug this node
/// fixes (bounded.rs's `maintain_narrow_range`, before: every configuration switched to
/// Wide permanently within a few hundred samples of a series shaped just like this one).
#[test]
fn market_level_series_at_a_high_price_stays_narrow() {
    let m = 50;
    let w = 1000;
    let scale = Scale::new(1.0).unwrap();
    let level = 10_000_000_000i64;
    let len = 20_500usize;

    // A deterministic pseudo-random walk, not a bounded oscillation: a walk that keeps
    // drifting (setting new running highs and lows throughout, as a real market series
    // does) is what actually exercises an edge-pinned anchor's own weakness (no margin on
    // one side), where a bounded, periodic wiggle would not — once such a wiggle's full
    // range has been seen once, no anchor choice needs touching again.
    let b = bound(m);
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut level_offset = 0i64;
    let series: Vec<f64> = (0..len)
        .map(|k| {
            if k % 733 == 0 && k > 0 {
                return f64::NAN;
            }
            // One extra re-anchor forced early (row ~150, well before window 0 ages out of
            // the ring at row `max_offset = w - m = 950`): exercises the placeholder-leaving
            // exclusion specifically while window 0 is still the ring's oldest, not only
            // the very first re-anchor (row 0), where `windows` is still empty and the bug
            // can never fire regardless.
            if k == 200 {
                level_offset += b + 1_000;
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let step = ((state >> 33) % 5) as i64 - 2; // -2..=2
            level_offset += step;
            (level + level_offset) as f64
        })
        .collect();

    let (path, anchor_changes) = run_checked_sparse(&series, m, w, scale, 2_000);
    assert_eq!(path, Path::Narrow);
    // A centered anchor's margin comfortably covers this walk's whole eventual range (a few
    // hundred ticks at most, against a bound in the tens of millions) after a small handful
    // of re-anchors; an edge-pinned one would re-anchor on the order of the walk's own
    // number of new running extrema (order `sqrt(len)`, so in the hundreds here).
    assert!(
        anchor_changes <= 20,
        "expected only a handful of re-anchors, got {anchor_changes}"
    );
}

/// `m·tick` alone passes i64 here (4096 · 2.4e15; naive stays exact in i128 up to about this level), while every offset from the anchor stays
/// tiny: plain updates must neither overflow in a debug build nor leave the exact answer.
#[test]
fn a_large_window_at_a_large_tick_level_matches_naive() {
    let m = 4096;
    let w = m + 10;
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
    assert_eq!(run_checked(&series, m, w, scale), Path::Narrow);
}

/// A span that truly exceeds `2 * bound` (not an artifact of the placeholder-leaving bug)
/// still switches to Wide, even at the same high price level and scale as the test above:
/// excluding window 0's placeholder didn't make the genuine-wide-switch case unreachable.
#[test]
fn a_genuinely_wide_span_at_a_high_price_level_still_switches() {
    // Small m/w (naive's own check is quadratic in w, so this stays a fast unit test), but
    // still at the same ten-billion-tick price level as the test above: the placeholder
    // exclusion didn't make a genuine wide switch unreachable at that level either.
    let m = 4;
    let w = 14;
    let scale = Scale::new(1.0).unwrap();
    let level = 10_000_000_000i64;
    let b = bound(m);

    let mut series: Vec<f64> = (0..10)
        .map(|k| level as f64 + (k as f64 * 0.7).sin() * 3.0)
        .collect();
    // A jump of more than `2 * bound` past the current plateau: no anchor could cover both.
    series.push(level as f64 + (3 * b) as f64);
    series.extend((0..5).map(|k| level as f64 + (3 * b) as f64 + (k as f64) - 2.0));

    let path = run_checked(&series, m, w, scale);
    assert_eq!(path, Path::Wide);
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// A market-shaped warm-up (plenty of live diagonals by the time it ends), then one
    /// injected jump whose size is drawn to force either outcome, then a small tail near
    /// the jumped-to level: every row throughout matches `windowed_entry` bit for bit,
    /// on both sides of the jump, whichever path it forces. The warm-up and tail are
    /// bounded-amplitude walks rather than the full market generator: this test's own job
    /// is the anchor mechanics (re-anchor vs. the switch to Wide), already exercised at
    /// market scale by the other proptests above, not re-covering market shapes here — and
    /// a bounded walk keeps the jump's size relative to the actual running span
    /// predictable, which the market generators' own occasional internal re-leveling
    /// (up to `MARKET_OFFSET`) would not.
    #[test]
    fn forced_transition_mid_stream_matches_naive_throughout(
        m in 2usize..20,
        warm_up in prop::collection::vec(-5.0..=5.0, 8..24),
        tail in prop::collection::vec(-5.0..=5.0, 8..24),
        force_wide in prop::bool::ANY,
        jitter in -100i64..=100,
    ) {
        let scale = Scale::new(1.0).unwrap();
        let exclusion = crate::exclusion::zone(m);
        let w = m + exclusion + 6;
        let b = bound(m);
        let jump = if force_wide { 3 * b } else { b + 1000 } + jitter;

        let mut series = warm_up;
        series.push(jump as f64);
        series.extend(tail.iter().map(|&t| jump as f64 + t));

        let path = run_checked(&series, m, w, scale);
        if force_wide {
            prop_assert_eq!(path, Path::Wide);
        } else {
            prop_assert_eq!(path, Path::Narrow);
        }
    }
}
