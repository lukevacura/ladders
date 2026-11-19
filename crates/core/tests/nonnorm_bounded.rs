#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! `BoundedNonNormStream` equals the exact windowed reference on bits, however its samples are
//! chunked, and a checkpoint resumes it bit for bit.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::checkpoint::{CheckpointError, with_recomputed_checksum};
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::route::{RouteForce, forced_route};
use nonnorm::*;
use proptest::prelude::*;

fn stream(m: usize, w: usize) -> BoundedNonNormStream {
    BoundedNonNormStream::new(nz(m), nz(w), unit()).expect("w is at least m")
}

fn run(series: &[f64], m: usize, w: usize) -> Vec<Row> {
    let mut stream = stream(m, w);
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
    fn random_series_equal_the_windowed_reference(
        series in series(),
        m in 1usize..=6,
        extra in 0usize..30,
    ) {
        let _unforced = unforced();
        let w = m + extra;
        prop_assert_eq!(run(&series, m, w), naive_window_rows(&series, unit(), m, w));
    }

    #[test]
    fn tie_heavy_series_equal_the_windowed_reference(
        series in tie_series(),
        m in 1usize..=7,
        extra in 0usize..30,
    ) {
        let _unforced = unforced();
        let w = m + extra;
        prop_assert_eq!(run(&series, m, w), naive_window_rows(&series, unit(), m, w));
    }

    #[test]
    fn shocks_that_leave_the_window_equal_the_windowed_reference(
        series in shocked(),
        m in 2usize..=6,
        extra in 0usize..30,
    ) {
        let _unforced = unforced();
        let w = m + extra;
        prop_assert_eq!(run(&series, m, w), naive_window_rows(&series, unit(), m, w));
    }

    #[test]
    fn a_horizon_past_the_series_equals_the_growing_stream(
        series in series(),
        m in 1usize..=6,
    ) {
        let _unforced = unforced();
        let mut growing = GrowingNonNormStream::new(nz(m), unit());
        let expected: Vec<_> =
            series.iter().filter_map(|&x| growing.update(x)).map(stream_row).collect();
        prop_assert_eq!(run(&series, m, series.len() + 1 + m), expected);
    }

    #[test]
    fn update_many_equals_update_for_any_chunking(
        series in series(),
        m in 1usize..=6,
        extra in 0usize..30,
        chunk in 1usize..12,
    ) {
        let _unforced = unforced();
        let mut chunked = stream(m, m + extra);
        let mut out = Vec::new();
        for piece in series.chunks(chunk) {
            chunked.update_many(piece, &mut out);
        }
        let rows: Vec<_> = out.into_iter().map(stream_row).collect();
        prop_assert_eq!(rows, run(&series, m, m + extra));
    }

    #[test]
    fn a_checkpoint_resumes_bit_for_bit_on_either_route(
        series in prop_oneof![series(), shocked()],
        m in 2usize..=6,
        extra in 0usize..30,
        pick in any::<usize>(),
    ) {
        let w = m + extra;
        let split = pick % (series.len() + 1);
        for force in [RouteForce::default(), forced_wide()] {
            let _forced = forced_route(force);
            let mut whole = stream(m, w);
            let expected: Vec<_> =
                series.iter().filter_map(|&x| whole.update(x)).map(stream_row).collect();
            let mut first = stream(m, w);
            let mut rows: Vec<_> =
                series[..split].iter().filter_map(|&x| first.update(x)).map(stream_row).collect();
            let bytes = first.checkpoint();
            let mut resumed = BoundedNonNormStream::restore(&bytes).expect("a checkpoint restores");
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
        extra in 0usize..30,
        at in any::<usize>(),
        xor in 1u8..=255,
        cut in any::<usize>(),
    ) {
        let _unforced = unforced();
        let mut stream = stream(m, m + extra);
        for &x in &series {
            let _ = stream.update(x);
        }
        let bytes = stream.checkpoint();
        let mut flipped = bytes.clone();
        let at = at % flipped.len();
        flipped[at] ^= xor;
        prop_assert!(BoundedNonNormStream::restore(&flipped).is_err(), "flip at {}", at);
        let cut = cut % bytes.len();
        prop_assert!(BoundedNonNormStream::restore(&bytes[..cut]).is_err(), "cut at {}", cut);
    }

    #[test]
    fn damaged_checkpoints_never_panic(
        series in series(),
        m in 1usize..=6,
        extra in 0usize..30,
        cut in any::<usize>(),
        flips in prop::collection::vec((any::<usize>(), any::<u8>()), 0..4),
    ) {
        let _unforced = unforced();
        let mut stream = stream(m, m + extra);
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
        if let Ok(mut restored) = BoundedNonNormStream::restore(&bytes) {
            let _ = restored.update(0.0);
        }
    }
}

#[test]
fn a_horizon_shorter_than_the_window_is_rejected() {
    assert!(BoundedNonNormStream::new(nz(5), nz(4), unit()).is_none());
}

#[test]
fn a_horizon_of_one_window_has_no_neighbors() {
    let _unforced = unforced();
    let series: Vec<f64> = (0..30).map(|k| (k % 3) as f64).collect();
    let rows = run(&series, 4, 4);
    assert_eq!(rows.len(), 27);
    assert!(rows.iter().all(|&(_, _, neighbor)| neighbor == u64::MAX));
}

#[test]
fn only_windows_inside_the_last_w_samples_are_neighbors() {
    let _unforced = unforced();
    let series: Vec<f64> = (0..60).map(|k| ((k * 7) % 11) as f64).collect();
    let (m, w) = (4, 12);
    let rows = run(&series, m, w);
    assert_eq!(rows, naive_window_rows(&series, unit(), m, w));
    for &(row, _, neighbor) in &rows {
        if neighbor != u64::MAX {
            assert!(
                neighbor + w as u64 >= row + m as u64,
                "row {row} chose {neighbor}"
            );
        }
    }
    assert!(rows.iter().filter(|r| r.2 != u64::MAX).count() > 20);
}

#[test]
fn flat_series_ties_to_the_smaller_index_in_the_horizon() {
    let _unforced = unforced();
    let flat = [7.0; 40];
    let rows = run(&flat, 4, 12);
    assert_eq!(rows, naive_window_rows(&flat, unit(), 4, 12));
    assert_eq!(rows[30], (30, 0, 30 + 4 - 12));
}

#[test]
fn gaps_at_the_start_the_end_and_between_windows() {
    let _unforced = unforced();
    let mut series: Vec<f64> = (0..40).map(|k| (k * 7 % 11) as f64).collect();
    for gap in [0, 1, 13, 14, 38, 39] {
        series[gap] = f64::NAN;
    }
    for (m, w) in [(3, 9), (4, 20), (4, 4)] {
        assert_eq!(
            run(&series, m, w),
            naive_window_rows(&series, unit(), m, w),
            "m {m} w {w}"
        );
    }
}

/// Far ticks enter and leave the horizon, so the range grows past the `u64` bound and shrinks
/// back under it, in both directions, several times.
#[test]
fn a_shock_crossing_the_bound_both_ways_stays_exact() {
    let _unforced = unforced();
    let far = (1u64 << 33) as f64;
    let series: Vec<f64> = (0..200)
        .map(|k| match k {
            40..=43 | 90..=91 | 150 => far + (k % 3) as f64,
            _ => ((k * 5) % 11) as f64,
        })
        .collect();
    for (m, w) in [(4, 20), (3, 7), (5, 40)] {
        assert_eq!(
            run(&series, m, w),
            naive_window_rows(&series, unit(), m, w),
            "m {m} w {w}"
        );
    }
}

#[test]
fn a_range_no_width_holds_gives_rows_without_a_neighbor() {
    let _unforced = unforced();
    let m = 1 << 20;
    let extreme = (1u64 << 53) as f64;
    let mut stream = stream(m, m + 8);
    let rows: Vec<_> = (0..m + 4)
        .filter_map(|k| stream.update(if k % 2 == 0 { extreme } else { -extreme }))
        .map(stream_row)
        .collect();
    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|&(_, _, neighbor)| neighbor == u64::MAX));
    let restored = BoundedNonNormStream::restore(&stream.checkpoint()).expect("restores");
    assert_eq!(restored.checkpoint(), stream.checkpoint());
}

#[test]
fn a_checkpoint_of_another_kind_or_format_is_rejected() {
    let _unforced = unforced();
    assert_eq!(
        BoundedNonNormStream::restore(&[]).err(),
        Some(CheckpointError::Truncated)
    );
    let mut growing = GrowingNonNormStream::new(nz(4), unit());
    for k in 0..20 {
        let _ = growing.update(f64::from(k));
    }
    assert_eq!(
        BoundedNonNormStream::restore(&growing.checkpoint()).err(),
        Some(CheckpointError::Inconsistent)
    );
    let mut bounded = stream(4, 12);
    for k in 0..20 {
        let _ = bounded.update(f64::from(k));
    }
    assert_eq!(
        GrowingNonNormStream::restore(&bounded.checkpoint()).err(),
        Some(CheckpointError::Inconsistent)
    );
}

#[test]
fn a_checkpoint_claiming_a_huge_horizon_is_refused_before_allocating() {
    let _unforced = unforced();
    let mut bytes = stream(4, 12).checkpoint();
    // Header (magic, version, kind) then m, then the horizon.
    let at = 4 + 2 + 1 + 8;
    bytes[at..at + 8].copy_from_slice(&(1u64 << 50).to_le_bytes());
    let bytes = with_recomputed_checksum(&bytes);
    assert_eq!(
        BoundedNonNormStream::restore(&bytes).err(),
        Some(CheckpointError::Truncated)
    );
}

/// Header (magic, version, kind), `m`, the horizon, the scale, the sample count, the tick count
/// and the first tick.
const TOTAL_AT: usize = 4 + 2 + 1 + 8 + 8 + 8;
const TICKS_AT: usize = TOTAL_AT + 8;

fn checkpoint_after(samples: u32) -> Vec<u8> {
    let mut stream = stream(4, 16);
    for k in 0..samples {
        let _ = stream.update(5.0 + f64::from(k % 3));
    }
    stream.checkpoint()
}

/// The horizon is 16: below it the tick count is the sample count; past it, any larger count
/// keeps the same ticks.
#[test]
fn a_sample_count_that_disagrees_with_the_ticks_is_rejected() {
    let _unforced = unforced();
    let restore_with = |samples: u32, total: u64| {
        let mut bytes = checkpoint_after(samples);
        bytes[TOTAL_AT..TOTAL_AT + 8].copy_from_slice(&total.to_le_bytes());
        let bytes = with_recomputed_checksum(&bytes);
        BoundedNonNormStream::restore(&bytes).err()
    };
    for total in [0, 9, 11, 1 << 63] {
        assert_eq!(
            restore_with(10, total),
            Some(CheckpointError::Inconsistent),
            "{total}"
        );
    }
    for total in [0, 15, 1 << 63] {
        assert_eq!(
            restore_with(40, total),
            Some(CheckpointError::Inconsistent),
            "{total}"
        );
    }
    for total in [16, 39, 41, 1 << 62] {
        assert_eq!(restore_with(40, total), None, "{total}");
    }
}

#[test]
fn a_tick_outside_the_quantizers_range_is_rejected() {
    let _unforced = unforced();
    let mut bytes = checkpoint_after(40);
    bytes[TICKS_AT + 8..TICKS_AT + 16].copy_from_slice(&i64::MAX.to_le_bytes());
    let bytes = with_recomputed_checksum(&bytes);
    assert_eq!(
        BoundedNonNormStream::restore(&bytes).err(),
        Some(CheckpointError::Inconsistent)
    );
}

#[test]
fn flags_that_contradict_the_ticks_are_rejected() {
    let _unforced = unforced();
    let mut bytes = checkpoint_after(10);
    // 10 flags pack into the last two bytes; clear the first tick's: the stream's first sample
    // being missing would make it the placeholder `0`, not the `5` stored.
    let first = bytes.len() - 8 - 2;
    bytes[first] ^= 1;
    let bytes = with_recomputed_checksum(&bytes);
    assert_eq!(
        BoundedNonNormStream::restore(&bytes).err(),
        Some(CheckpointError::Inconsistent)
    );
}
