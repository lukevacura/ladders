#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized streams that keep `k` neighbors per row: the nearest of the `k` is the
//! one-neighbor stream's, a checkpoint carries `k` and resumes bit for bit, and a damaged `k` is
//! refused.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/nonnorm_topk.rs"]
mod nonnorm_topk;

use ladde::checkpoint::{CheckpointError, with_recomputed_checksum};
use ladde::index::SampleIndex;
use ladde::nonnorm::NonNormNeighbor;
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::route::{RouteForce, forced_route};
use nonnorm::*;
use nonnorm_topk::{Pair, kept_pairs};
use proptest::prelude::*;

type Rows = Vec<(u64, Vec<Pair>)>;

fn wide() -> RouteForce {
    RouteForce {
        wide: true,
        ..RouteForce::default()
    }
}

fn kept_row((row, kept): (SampleIndex, &[NonNormNeighbor])) -> (u64, Vec<Pair>) {
    (row.get().unwrap(), kept_pairs(kept))
}

fn grow(series: &[f64], m: usize, k: usize) -> (GrowingNonNormStream, Rows) {
    let mut stream = GrowingNonNormStream::with_k(nz(m), unit(), nz(k));
    let rows = series
        .iter()
        .filter_map(|&x| stream.update_topk(x).map(kept_row))
        .collect();
    (stream, rows)
}

fn bound(series: &[f64], m: usize, w: usize, k: usize) -> (BoundedNonNormStream, Rows) {
    let mut stream = BoundedNonNormStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
    let rows = series
        .iter()
        .filter_map(|&x| stream.update_topk(x).map(kept_row))
        .collect();
    (stream, rows)
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn the_nearest_of_k_is_the_one_neighbor_stream(
        series in prop_oneof![series(), tie_series()],
        m in 1usize..=6,
        k in 1usize..9,
        extra in 0usize..30,
    ) {
        let _serial = forced_route(RouteForce::default());
        let mut one = GrowingNonNormStream::new(nz(m), unit());
        let mut many = GrowingNonNormStream::with_k(nz(m), unit(), nz(k));
        let mut bounded_one = BoundedNonNormStream::new(nz(m), nz(m + extra), unit()).unwrap();
        let mut bounded_many =
            BoundedNonNormStream::with_k(nz(m), nz(m + extra), unit(), nz(k)).unwrap();
        for &x in &series {
            prop_assert_eq!(one.update(x), many.update(x));
            prop_assert_eq!(bounded_one.update(x), bounded_many.update(x));
        }
        prop_assert_eq!(many.k(), nz(k));
        prop_assert_eq!(bounded_many.k(), nz(k));
    }

    #[test]
    fn a_top_k_checkpoint_resumes_bit_for_bit_on_either_route(
        series in prop_oneof![series(), tie_series()],
        m in 1usize..=6,
        k in prop::sample::select(vec![1usize, 2, 3, 8]),
        extra in 0usize..30,
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        let w = m + extra;
        for force in [RouteForce::default(), wide()] {
            let _forced = forced_route(force);
            let (whole, expected) = grow(&series, m, k);
            let (stream, mut rows) = grow(&series[..split], m, k);
            let bytes = stream.checkpoint();
            let mut resumed = GrowingNonNormStream::restore(&bytes).expect("a checkpoint restores");
            prop_assert_eq!(resumed.k(), nz(k));
            prop_assert_eq!(resumed.checkpoint(), bytes);
            rows.extend(
                series[split..]
                    .iter()
                    .filter_map(|&x| resumed.update_topk(x).map(kept_row)),
            );
            prop_assert_eq!(&rows, &expected, "growing {:?} split {}", force, split);
            prop_assert_eq!(resumed.checkpoint(), whole.checkpoint());

            let (whole, expected) = bound(&series, m, w, k);
            let (stream, mut rows) = bound(&series[..split], m, w, k);
            let bytes = stream.checkpoint();
            let mut resumed = BoundedNonNormStream::restore(&bytes).expect("a checkpoint restores");
            prop_assert_eq!(resumed.k(), nz(k));
            prop_assert_eq!(resumed.checkpoint(), bytes);
            rows.extend(
                series[split..]
                    .iter()
                    .filter_map(|&x| resumed.update_topk(x).map(kept_row)),
            );
            prop_assert_eq!(&rows, &expected, "bounded {:?} split {} w {}", force, split, w);
            prop_assert_eq!(resumed.checkpoint(), whole.checkpoint());
        }
    }
}

/// The version after the magic: a stream of one writes what it always wrote.
fn version(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[4], bytes[5]])
}

#[test]
fn k_of_one_writes_the_old_version_and_more_writes_the_new_one() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = (0..40).map(|k| (k % 7) as f64).collect();
    for k in [1, 2, 5] {
        let expected = if k == 1 { 3 } else { 4 };
        assert_eq!(version(&grow(&series, 4, k).0.checkpoint()), expected);
        assert_eq!(version(&bound(&series, 4, 20, k).0.checkpoint()), expected);
    }
    let plain = {
        let mut stream = GrowingNonNormStream::new(nz(4), unit());
        for &x in &series {
            let _ = stream.update(x);
        }
        stream.checkpoint()
    };
    assert_eq!(grow(&series, 4, 1).0.checkpoint(), plain);
}

type Restore = fn(&[u8]) -> Result<(), CheckpointError>;

fn restore_growing(bytes: &[u8]) -> Result<(), CheckpointError> {
    GrowingNonNormStream::restore(bytes).map(|_| ())
}

fn restore_bounded(bytes: &[u8]) -> Result<(), CheckpointError> {
    BoundedNonNormStream::restore(bytes).map(|_| ())
}

#[test]
fn a_damaged_k_is_refused() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = (0..40).map(|k| (k % 7) as f64).collect();
    let cases: [(Vec<u8>, Restore); 2] = [
        (grow(&series, 4, 3).0.checkpoint(), restore_growing),
        (bound(&series, 4, 20, 3).0.checkpoint(), restore_bounded),
    ];
    for (bytes, restore) in cases {
        assert_eq!(restore(&bytes), Ok(()));
        // Seven bytes of header, eight of `m`, then `k`.
        let at = 7 + 8;
        for forged in [0u64, 1] {
            let mut damaged = bytes.clone();
            damaged[at..at + 8].copy_from_slice(&forged.to_le_bytes());
            assert_eq!(
                restore(&with_recomputed_checksum(&damaged)),
                Err(CheckpointError::Inconsistent),
                "k {forged}"
            );
        }
        let mut old_version = bytes.clone();
        old_version[4..6].copy_from_slice(&3u16.to_le_bytes());
        assert!(restore(&with_recomputed_checksum(&old_version)).is_err());
    }
}

#[test]
fn a_k_beyond_every_window_keeps_all_of_them() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = (0..60).map(|k| ((k * 5) % 11) as f64).collect();
    let k = nz(1 << 40);
    assert_eq!(
        grow(&series, 4, k.get()).1,
        nonnorm_topk::exact_left(&series, unit(), 4, k, None)
    );
    assert_eq!(
        bound(&series, 4, 30, k.get()).1,
        nonnorm_topk::exact_left(&series, unit(), 4, k, Some(30))
    );
}
