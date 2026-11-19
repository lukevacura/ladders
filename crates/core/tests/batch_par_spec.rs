#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! Pins `batch::left_profile_threads`/`left_profile_threads_with`
//! bit for bit against `batch::left_profile`: the whole point of the banded/chunked thread
//! scheduler is that no thread count, band size or chunk size may ever change a bit.
//! `threads`, `band_rows` and `chunk_lanes` are swept explicitly, never left to proptest's own
//! strategy, so every value the spec names is exercised on every generated series rather than
//! diluted across cases (mirrors `batch.rs`'s own `tiled_matches_naive_across_chunk_lengths`).
//!
//! Every generated series is extended with a guaranteed exact repeat ([`with_guaranteed_match`])
//! before it is checked, so "non-trivial output" is asserted outright, never behind a
//! `prop_assume!` that could silently narrow the input space down to nothing.

use core::num::NonZeroUsize;

use ladde::batch::{left_profile, left_profile_threads, left_profile_threads_with};
use ladde::entry::Entry;
use ladde::quantize::Scale;
use proptest::collection::SizeRange;
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, quantized, random_walk, with_flat_runs, with_gaps};

/// `(ρ, d², index)` bits: the exact triple the bitwise-determinism guarantee covers, matching
/// `batch.rs`'s own private `bits` helper (not reusable from here: different compilation unit).
fn bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

/// A whole profile's bits, row included: two profiles are identical, bit for bit, exactly when
/// this is equal.
fn profile_bits(profile: &[(u64, Entry)]) -> Vec<(u64, u64, u64, u64)> {
    profile
        .iter()
        .map(|&(row, entry)| {
            let (rho, dist, index) = bits(entry);
            (row, rho, dist, index)
        })
        .collect()
}

/// At least one row has a real neighbor: rules out a determinism check that passes only
/// because both sides trivially agree on [`Entry::NONE`] everywhere.
fn has_a_real_neighbor(profile: &[(u64, Entry)]) -> bool {
    profile
        .iter()
        .any(|&(_, entry)| entry.index().get().is_some())
}

/// Profile rows for a series of `n` samples at window length `m`: `n − m + 1`, or `0` when
/// there is no room for a single window.
fn rows_for(n: usize, m: usize) -> usize {
    if n >= m { n - m + 1 } else { 0 }
}

const THREADS: [usize; 5] = [1, 2, 3, 7, 16];

/// `band_rows` values the spec names, resolved against this series' own row count: `1`, `7`,
/// `64`, and one past the last row (so a single band covers everything).
fn band_rows_values(rows: usize) -> [usize; 4] {
    [1, 7, 64, rows + 1]
}

/// `chunk_lanes` values the spec names, resolved the same way.
fn chunk_lanes_values(rows: usize) -> [usize; 4] {
    [1, 37, 1170, rows + 1]
}

/// Appends a fresh, non-constant, finite block of `m` values three times, each copy separated
/// by an `m`-long NaN gap: guarantees, regardless of what the category strategy drew, that the
/// last row has not merely one eligible exact match but an exact *tie* between the first and
/// second copies (both equally ρ = 1, d² = 0 against it), so the determinism checks exercise
/// `Nearest`'s own smaller-index tie-break under threading, not only a lone dominant match. Also
/// guarantees non-trivial output outright, so every check below can assert it unconditionally
/// rather than hoping a random draw produced one.
fn with_guaranteed_match(mut series: Vec<f64>, m: usize) -> Vec<f64> {
    let block: Vec<f64> = (0..m).map(|k| 1_000.0 + k as f64).collect();
    for _ in 0..2 {
        series.extend_from_slice(&block);
        series.extend(std::iter::repeat_n(f64::NAN, m));
    }
    series.extend_from_slice(&block);
    series
}

/// A fixed, reproducible walk around the spec's own market level: no `rand` dependency and no
/// proptest randomness, so the repeatability check drives the exact same input on all twenty of
/// its calls.
fn deterministic_series(len: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut level = LEVEL;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let step = ((state >> 56) as i64 % 5) - 2; // -2..=2
            level += step as f64;
            level
        })
        .collect()
}

const LEVEL: f64 = 1e10;

fn at_least(len: impl Into<SizeRange>, min: usize) -> SizeRange {
    let (start, end) = len.into().start_end_incl();
    SizeRange::new(start.max(min)..=end.max(min))
}

fn m_values() -> impl Strategy<Value = usize> {
    2usize..=6
}

/// Plain random walks around zero — the "random walk" category.
fn random_walk_and_m(len: impl Into<SizeRange>) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let len = len.into();
    m_values().prop_flat_map(move |m| {
        random_walk(at_least(len.clone(), m), 0.0)
            .prop_map(move |series| (with_guaranteed_match(series, m), m))
    })
}

/// Market-level series at `level ≈ 1e10` (testkit's own generators, with the offset raised to
/// the spec's own level rather than testkit's smaller default) — the "market-level" category.
fn market_level_and_m(len: impl Into<SizeRange>) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let len = len.into();
    m_values().prop_flat_map(move |m| {
        prop_oneof![
            random_walk(at_least(len.clone(), m), LEVEL),
            quantized(at_least(len.clone(), m), MARKET_TICK, LEVEL),
        ]
        .prop_map(move |series| (with_guaranteed_match(series, m), m))
    })
}

/// A market-level series with one maximal flat run spliced in — the "flat runs" category.
fn flat_run_and_m(len: impl Into<SizeRange>) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let len = len.into();
    m_values().prop_flat_map(move |m| {
        with_flat_runs(quantized(at_least(len.clone(), m), MARKET_TICK, LEVEL), m)
            .prop_map(move |series| (with_guaranteed_match(series, m), m))
    })
}

/// A market-level series with NaN/∞ gaps spliced in — the "NaN gaps" category.
fn nan_gap_and_m(len: impl Into<SizeRange>) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let len = len.into();
    m_values().prop_flat_map(move |m| {
        with_gaps(quantized(at_least(len.clone(), m), MARKET_TICK, LEVEL))
            .prop_map(move |series| (with_guaranteed_match(series, m), m))
    })
}

/// `left_profile_threads_with` at every `(threads, band_rows, chunk_lanes)` the spec names
/// equals `left_profile` on bits, for this one series — the full cross product, since a
/// scheduling bug can depend on how a band boundary and a chunk boundary line up together, not
/// only on either alone.
fn assert_all_combinations_match(series: &[f64], m: usize, scale: Scale) {
    let m_nz = NonZeroUsize::new(m).unwrap();
    let expected = left_profile(series, m_nz, scale);
    assert!(
        has_a_real_neighbor(&expected),
        "guaranteed-match tail produced no real neighbor"
    );
    let expected_bits = profile_bits(&expected);
    let rows = rows_for(series.len(), m);

    for &threads in &THREADS {
        for &band_rows in &band_rows_values(rows) {
            for &chunk_lanes in &chunk_lanes_values(rows) {
                let actual = left_profile_threads_with(
                    series,
                    m_nz,
                    scale,
                    NonZeroUsize::new(threads).unwrap(),
                    NonZeroUsize::new(band_rows).unwrap(),
                    NonZeroUsize::new(chunk_lanes).unwrap(),
                );
                assert_eq!(
                    profile_bits(&actual),
                    expected_bits,
                    "threads {threads}, band_rows {band_rows}, chunk_lanes {chunk_lanes}"
                );
            }
        }
    }
}

// ---------- determinism: left_profile_threads_with vs. left_profile, on bits ----------

proptest! {
    // 5 threads x 4 band_rows x 4 chunk_lanes = 80 combinations checked per case; kept low so
    // the full cross product (needed because a bug can depend on how a band and a chunk line
    // up together) stays affordable, the same trade `tiled_matches_naive_across_chunk_lengths`
    // makes in `batch.rs` with its own `cases: 24` for a smaller combination count.
    #![proptest_config(testkit::generators::seeded(8))]

    #[test]
    fn threads_with_matches_left_profile_on_random_walks((series, m) in random_walk_and_m(0..40)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        assert_all_combinations_match(&series, m, scale);
    }

    #[test]
    fn threads_with_matches_left_profile_on_market_level_series((series, m) in market_level_and_m(0..40)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        assert_all_combinations_match(&series, m, scale);
    }

    #[test]
    fn threads_with_matches_left_profile_with_flat_runs((series, m) in flat_run_and_m(0..40)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        assert_all_combinations_match(&series, m, scale);
    }

    #[test]
    fn threads_with_matches_left_profile_with_nan_gaps((series, m) in nan_gap_and_m(0..40)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        assert_all_combinations_match(&series, m, scale);
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(10))]

    /// The public, default-band/chunk entry point equals `left_profile` on bits too, at every
    /// thread count: the binding interface is both functions, not only the `_with` one tests
    /// otherwise only exercise.
    #[test]
    fn threads_matches_left_profile_at_default_band_and_chunk((series, m) in market_level_and_m(0..40)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let m_nz = NonZeroUsize::new(m).unwrap();
        let expected = left_profile(&series, m_nz, scale);
        assert!(has_a_real_neighbor(&expected));
        let expected_bits = profile_bits(&expected);
        for &threads in &THREADS {
            let actual = left_profile_threads(&series, m_nz, scale, NonZeroUsize::new(threads).unwrap());
            assert_eq!(profile_bits(&actual), expected_bits, "threads {threads}");
        }
    }
}

// ---------- edge cases ----------

#[test]
fn n_less_than_m_is_empty_on_every_thread_count() {
    let series = [1.0_f64, 2.0, 3.0]; // n = 3
    let m = NonZeroUsize::new(5).unwrap(); // m = 5 > n
    let scale = Scale::new(1.0).unwrap();
    assert!(left_profile(&series, m, scale).is_empty());
    for &threads in &THREADS {
        let actual = left_profile_threads(&series, m, scale, NonZeroUsize::new(threads).unwrap());
        assert!(actual.is_empty(), "threads {threads}");
    }
}

#[test]
fn n_just_over_m_matches_on_every_thread_count() {
    let m = 5usize;
    let series: Vec<f64> = (0..m + 1).map(|k| k as f64).collect(); // n = m + 1: 2 rows
    let scale = Scale::new(1.0).unwrap();
    let m_nz = NonZeroUsize::new(m).unwrap();
    let expected = left_profile(&series, m_nz, scale);
    let expected_bits = profile_bits(&expected);
    for &threads in &THREADS {
        let actual =
            left_profile_threads(&series, m_nz, scale, NonZeroUsize::new(threads).unwrap());
        assert_eq!(profile_bits(&actual), expected_bits, "threads {threads}");
    }
}

#[test]
fn threads_exceed_rows() {
    // n = m + 1: exactly 2 rows, far fewer than the largest thread count below.
    let m = 4usize;
    let series: Vec<f64> = (0..m + 1).map(|k| k as f64).collect();
    let scale = Scale::new(1.0).unwrap();
    let m_nz = NonZeroUsize::new(m).unwrap();
    let expected = left_profile(&series, m_nz, scale);
    let actual = left_profile_threads(&series, m_nz, scale, NonZeroUsize::new(16).unwrap());
    assert_eq!(profile_bits(&actual), profile_bits(&expected));
}

#[test]
fn every_row_is_inside_the_exclusion_zone() {
    let m = 7usize; // exclusion = ceil(7/4) = 2
    let n = m + 2; // rows = 3 (indices 0, 1, 2), every one's own eligible count is zero
    let series: Vec<f64> = (0..n).map(|k| k as f64).collect();
    let scale = Scale::new(1.0).unwrap();
    let m_nz = NonZeroUsize::new(m).unwrap();
    let expected = left_profile(&series, m_nz, scale);
    assert!(expected.iter().all(|&(_, entry)| entry.index().is_none()));
    let expected_bits = profile_bits(&expected);
    for &threads in &THREADS {
        let actual =
            left_profile_threads(&series, m_nz, scale, NonZeroUsize::new(threads).unwrap());
        assert_eq!(profile_bits(&actual), expected_bits, "threads {threads}");
    }
}

#[test]
fn a_single_band_covers_every_row() {
    let m = NonZeroUsize::new(5).unwrap();
    let scale = Scale::new(MARKET_TICK).unwrap();
    let series = with_guaranteed_match(deterministic_series(120), m.get());
    let expected = left_profile(&series, m, scale);
    assert!(has_a_real_neighbor(&expected));
    let expected_bits = profile_bits(&expected);
    let rows = rows_for(series.len(), m.get());
    let band_rows = NonZeroUsize::new(rows + 10).unwrap(); // one band covers everything

    for &threads in &THREADS {
        for &chunk_lanes in &[1usize, 37, rows + 10] {
            let actual = left_profile_threads_with(
                &series,
                m,
                scale,
                NonZeroUsize::new(threads).unwrap(),
                band_rows,
                NonZeroUsize::new(chunk_lanes).unwrap(),
            );
            assert_eq!(
                profile_bits(&actual),
                expected_bits,
                "threads {threads}, chunk_lanes {chunk_lanes}"
            );
        }
    }
}

#[test]
fn an_all_constant_series_matches_on_every_combination() {
    let series = [42.0_f64; 80];
    let m = NonZeroUsize::new(5).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let expected = left_profile(&series, m, scale);
    let expected_bits = profile_bits(&expected);
    let rows = rows_for(series.len(), m.get());

    for &threads in &THREADS {
        for &band_rows in &band_rows_values(rows) {
            for &chunk_lanes in &chunk_lanes_values(rows) {
                let actual = left_profile_threads_with(
                    &series,
                    m,
                    scale,
                    NonZeroUsize::new(threads).unwrap(),
                    NonZeroUsize::new(band_rows).unwrap(),
                    NonZeroUsize::new(chunk_lanes).unwrap(),
                );
                assert_eq!(
                    profile_bits(&actual),
                    expected_bits,
                    "threads {threads}, band_rows {band_rows}, chunk_lanes {chunk_lanes}"
                );
            }
        }
    }
}

/// A tick span far beyond any per-`m` narrow-anchor bound (`⌊(2³¹ − 1)/m⌋`, `narrow_anchor`'s
/// own gate): forces `left_profile`'s wide (i64) fallback path, not the narrow one every
/// other case in this file exercises. The "whatever thread count, or tiling" guarantee names the
/// arithmetic path too, not only the band/chunk shape, so the threaded engine must match
/// `left_profile` here as well.
#[test]
fn wide_fallback_path_matches_on_every_combination() {
    let m = NonZeroUsize::new(5).unwrap();
    let scale = Scale::new(1.0).unwrap();
    const SPAN: f64 = 2_000_000_000.0; // » 2 * floor((2^31 - 1) / 5) = 2 * 429_496_729
    let alternating: Vec<f64> = (0..30)
        .map(|k| if k % 2 == 0 { 0.0 } else { SPAN })
        .collect();
    let series = with_guaranteed_match(alternating, m.get());
    let expected = left_profile(&series, m, scale);
    assert!(has_a_real_neighbor(&expected));
    let expected_bits = profile_bits(&expected);
    let rows = rows_for(series.len(), m.get());

    for &threads in &THREADS {
        for &band_rows in &band_rows_values(rows) {
            for &chunk_lanes in &chunk_lanes_values(rows) {
                let actual = left_profile_threads_with(
                    &series,
                    m,
                    scale,
                    NonZeroUsize::new(threads).unwrap(),
                    NonZeroUsize::new(band_rows).unwrap(),
                    NonZeroUsize::new(chunk_lanes).unwrap(),
                );
                assert_eq!(
                    profile_bits(&actual),
                    expected_bits,
                    "threads {threads}, band_rows {band_rows}, chunk_lanes {chunk_lanes}"
                );
            }
        }
    }
}

// ---------- repeatability: the same call, 20 times, bit for bit ----------

/// Twenty runs of the same call at `threads = 8`, on a series long enough to span several bands
/// and chunks at the fixed, non-trivial `band_rows`/`chunk_lanes` below: a race that only
/// sometimes flips a bit (a chunk occasionally mis-claimed, a merge occasionally reading a
/// partial update) would otherwise pass on a lucky first run.
#[test]
fn eight_threads_is_repeatable_across_twenty_runs() {
    let m = NonZeroUsize::new(16).unwrap();
    let scale = Scale::new(MARKET_TICK).unwrap();
    let series = with_guaranteed_match(deterministic_series(400), m.get());
    let threads = NonZeroUsize::new(8).unwrap();
    let band_rows = NonZeroUsize::new(7).unwrap();
    let chunk_lanes = NonZeroUsize::new(37).unwrap();

    let first = left_profile_threads_with(&series, m, scale, threads, band_rows, chunk_lanes);
    assert!(has_a_real_neighbor(&first));
    let first_bits = profile_bits(&first);
    for run in 1..20 {
        let actual = left_profile_threads_with(&series, m, scale, threads, band_rows, chunk_lanes);
        assert_eq!(profile_bits(&actual), first_bits, "run {run}");
    }
}
