#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! `GrowingNonNormStream` equals the exact non-normalized reference on bits, however its samples
//! are chunked, and a checkpoint resumes it bit for bit.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::checkpoint::{CheckpointError, with_recomputed_checksum};
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use nonnorm::*;
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series_and_m};

fn run(series: &[f64], scale: Scale, m: usize) -> Vec<Row> {
    let mut stream = GrowingNonNormStream::new(nz(m), scale);
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(stream_row)
        .collect()
}

fn unforced() -> impl Drop {
    forced_route(RouteForce::default())
}

fn forced_wide() -> RouteForce {
    RouteForce {
        wide: true,
        ..RouteForce::default()
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn random_series_equal_the_reference(series in series(), m in 1usize..=6) {
        let _unforced = unforced();
        prop_assert_eq!(run(&series, unit(), m), naive_rows(&series, unit(), m));
    }

    #[test]
    fn tie_heavy_series_equal_the_reference(series in tie_series(), m in 1usize..=7) {
        let _unforced = unforced();
        prop_assert_eq!(run(&series, unit(), m), naive_rows(&series, unit(), m));
    }

    #[test]
    fn a_window_nearly_the_series_equals_the_reference(
        series in series(),
        back in 0usize..3,
    ) {
        let _unforced = unforced();
        let m = series.len().saturating_sub(back).max(1);
        prop_assert_eq!(run(&series, unit(), m), naive_rows(&series, unit(), m));
    }

    #[test]
    fn market_series_equal_the_reference_at_their_tick(
        (series, m) in market_series_and_m(0..48),
    ) {
        let _unforced = unforced();
        let scale = Scale::new(MARKET_TICK).unwrap();
        prop_assert_eq!(run(&series, scale, m), naive_rows(&series, scale, m));
    }

    #[test]
    fn update_many_equals_update_for_any_chunking(
        series in series(),
        m in 1usize..=6,
        chunk in 1usize..12,
    ) {
        let _unforced = unforced();
        let mut stream = GrowingNonNormStream::new(nz(m), unit());
        let mut out = Vec::new();
        for piece in series.chunks(chunk) {
            stream.update_many(piece, &mut out);
        }
        let rows: Vec<_> = out.into_iter().map(stream_row).collect();
        prop_assert_eq!(rows, run(&series, unit(), m));
    }

    #[test]
    fn a_checkpoint_resumes_bit_for_bit_on_either_route(
        series in series(),
        m in 1usize..=6,
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        for force in [RouteForce::default(), forced_wide()] {
            let _forced = forced_route(force);
            let mut whole = GrowingNonNormStream::new(nz(m), unit());
            let expected: Vec<_> =
                series.iter().filter_map(|&x| whole.update(x)).map(stream_row).collect();
            let mut stream = GrowingNonNormStream::new(nz(m), unit());
            let mut rows: Vec<_> =
                series[..split].iter().filter_map(|&x| stream.update(x)).map(stream_row).collect();
            let bytes = stream.checkpoint();
            let mut resumed = GrowingNonNormStream::restore(&bytes).expect("a checkpoint restores");
            prop_assert_eq!(resumed.checkpoint(), bytes);
            rows.extend(series[split..].iter().filter_map(|&x| resumed.update(x)).map(stream_row));
            prop_assert_eq!(&rows, &expected, "{:?} split {}", force, split);
            prop_assert_eq!(resumed.checkpoint(), whole.checkpoint());
        }
    }

    #[test]
    fn raw_damage_is_always_refused(
        series in series(),
        m in 1usize..=6,
        at in any::<usize>(),
        xor in 1u8..=255,
        cut in any::<usize>(),
    ) {
        let _unforced = unforced();
        let mut stream = GrowingNonNormStream::new(nz(m), unit());
        for &x in &series {
            let _ = stream.update(x);
        }
        let bytes = stream.checkpoint();
        let mut flipped = bytes.clone();
        let at = at % flipped.len();
        flipped[at] ^= xor;
        prop_assert!(GrowingNonNormStream::restore(&flipped).is_err(), "flip at {}", at);
        let cut = cut % bytes.len();
        prop_assert!(GrowingNonNormStream::restore(&bytes[..cut]).is_err(), "cut at {}", cut);
    }

    #[test]
    fn damaged_checkpoints_never_panic(
        series in series(),
        m in 1usize..=6,
        cut in any::<usize>(),
        flips in prop::collection::vec((any::<usize>(), any::<u8>()), 0..4),
    ) {
        let _unforced = unforced();
        let mut stream = GrowingNonNormStream::new(nz(m), unit());
        for &x in &series {
            let _ = stream.update(x);
        }
        let mut bytes = stream.checkpoint();
        for (at, xor) in flips {
            let at = at % bytes.len();
            bytes[at] ^= xor;
        }
        bytes.truncate(cut % (bytes.len() + 1));
        let bytes = with_recomputed_checksum(&bytes);
        if let Ok(mut restored) = GrowingNonNormStream::restore(&bytes) {
            let _ = restored.update(0.0);
        }
    }
}

#[test]
fn flat_series_and_flat_runs_tie_to_the_smaller_index() {
    let _unforced = unforced();
    let flat = [7.0; 20];
    assert_eq!(run(&flat, unit(), 4), naive_rows(&flat, unit(), 4));
    assert_eq!(run(&flat, unit(), 4)[10], (10, 0, 0));
    let runs: Vec<f64> = (0..40)
        .map(|k| {
            if (k / 7) % 2 == 0 {
                0.0
            } else {
                (1u64 << 33) as f64
            }
        })
        .collect();
    for m in [3, 5, 8] {
        assert_eq!(run(&runs, unit(), m), naive_rows(&runs, unit(), m));
    }
}

#[test]
fn gaps_at_the_start_the_end_and_between_windows() {
    let _unforced = unforced();
    let mut series: Vec<f64> = (0..30).map(|k| (k * 7 % 11) as f64).collect();
    for gap in [0, 1, 13, 14, 28, 29] {
        series[gap] = f64::NAN;
    }
    for m in [3, 4] {
        assert_eq!(run(&series, unit(), m), naive_rows(&series, unit(), m));
    }
    assert_eq!(
        run(&series[..14], unit(), 3),
        naive_rows(&series[..14], unit(), 3)
    );
}

#[test]
fn a_leading_gap_before_far_away_ticks_is_exact() {
    let _unforced = unforced();
    let mut series = vec![f64::NAN; 6];
    series.extend((0..30).map(|k| (1u64 << 40) as f64 + (k % 4) as f64));
    assert_eq!(run(&series, unit(), 4), naive_rows(&series, unit(), 4));
}

#[test]
fn rows_either_side_of_the_bound_agree_with_the_reference() {
    let _unforced = unforced();
    let mut series: Vec<f64> = (0..16).map(|k| (k % 5) as f64).collect();
    series.extend((0..16).map(|k| (1u64 << 33) as f64 + (k % 5) as f64));
    series.extend((0..8).map(|k| (k % 5) as f64));
    let rows = run(&series, unit(), 4);
    assert_eq!(rows, naive_rows(&series, unit(), 4));
    assert!(rows[16].1 > u128::from(u64::MAX));
    assert!(rows[10].1 < u128::from(u64::MAX));
}

#[test]
fn distances_are_euclidean_in_the_callers_units() {
    let _unforced = unforced();
    let scale = Scale::new(0.25).unwrap();
    let mut stream = GrowingNonNormStream::new(nz(2), scale);
    let rows: Vec<_> = [1.0, 2.0, 3.0, 1.0, 2.0, 9.0]
        .iter()
        .filter_map(|&x| stream.update(x))
        .collect();
    let [.., (_, last)] = rows[..] else {
        panic!("six samples complete five windows")
    };
    assert_eq!(last.index().get(), Some(1));
    assert_eq!(last.squared_ticks(), 24 * 24);
    assert_eq!(last.euclidean(scale), 6.0);
}

#[test]
fn a_checkpoint_of_another_kind_or_format_is_rejected() {
    let _unforced = unforced();
    assert_eq!(
        GrowingNonNormStream::restore(&[]).err(),
        Some(CheckpointError::Truncated)
    );
    let zeros = vec![0u8; 64];
    assert!(GrowingNonNormStream::restore(&zeros).is_err());
}

/// `m·R²` reaches `2¹²⁸` at `m = 2²⁰` and the widest ticks: rows report no neighbor, nothing
/// panics, and the stream still checkpoints and restores.
#[test]
fn a_range_no_width_holds_gives_rows_without_a_neighbor() {
    let _unforced = unforced();
    let m = 1 << 20;
    let extreme = (1u64 << 53) as f64;
    let mut stream = GrowingNonNormStream::new(nz(m), unit());
    let rows: Vec<_> = (0..m + 4)
        .filter_map(|k| stream.update(if k % 2 == 0 { extreme } else { -extreme }))
        .map(stream_row)
        .collect();
    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|&(_, _, neighbor)| neighbor == u64::MAX));
    let restored = GrowingNonNormStream::restore(&stream.checkpoint()).expect("restores");
    assert_eq!(restored.checkpoint(), stream.checkpoint());
}

/// `m·R²` exactly `2⁶⁴`: the one pair that far apart would wrap to zero in `u64`.
#[test]
fn a_pair_exactly_two_to_the_64_apart_is_not_wrapped() {
    let _unforced = unforced();
    let mut series = vec![0.0; 4];
    series.push(f64::NAN);
    series.extend([(1u64 << 31) as f64; 4]);
    let rows = run(&series, unit(), 4);
    assert_eq!(rows, naive_rows(&series, unit(), 4));
    assert_eq!(rows[5], (5, 1u128 << 64, 0));
}

#[test]
fn just_under_the_bound_stays_exact() {
    let _unforced = unforced();
    let mut series = vec![0.0; 4];
    series.push(f64::NAN);
    series.extend([((1u64 << 31) - 1) as f64; 4]);
    let rows = run(&series, unit(), 4);
    assert_eq!(rows, naive_rows(&series, unit(), 4));
    assert_eq!(rows[5].1, 4 * ((1u128 << 31) - 1).pow(2));
}

/// Header (magic, version, kind), `m`, the scale, then the tick count and the first tick.
const TICKS_AT: usize = 4 + 2 + 1 + 8 + 8;

fn checkpoint_of_24() -> Vec<u8> {
    let mut stream = GrowingNonNormStream::new(nz(4), unit());
    for k in 0..24 {
        let _ = stream.update(5.0 + f64::from(k % 3));
    }
    stream.checkpoint()
}

#[test]
fn a_tick_outside_the_quantizers_range_is_rejected() {
    let _unforced = unforced();
    let mut bytes = checkpoint_of_24();
    bytes[TICKS_AT + 8..TICKS_AT + 16].copy_from_slice(&i64::MAX.to_le_bytes());
    let bytes = with_recomputed_checksum(&bytes);
    assert_eq!(
        GrowingNonNormStream::restore(&bytes).err(),
        Some(CheckpointError::Inconsistent)
    );
}

#[test]
fn flags_that_contradict_the_ticks_are_rejected() {
    let _unforced = unforced();
    let mut bytes = checkpoint_of_24();
    // 24 flags pack into the last three bytes; clear the first tick's: a missing first sample
    // is the placeholder `0`, not the `5` stored.
    let first = bytes.len() - 8 - 3;
    bytes[first] ^= 1;
    let bytes = with_recomputed_checksum(&bytes);
    assert_eq!(
        GrowingNonNormStream::restore(&bytes).err(),
        Some(CheckpointError::Inconsistent)
    );
}
