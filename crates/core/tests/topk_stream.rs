#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The streams that keep `k` neighbors per row: equal to the exact reference on tie-heavy and
//! gap series, a fixed case pinned to literal bits, `k` of one unchanged, and checkpoints that
//! carry `k`.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::checkpoint::{CheckpointError, with_recomputed_checksum};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use ladde::stream::GrowingStream;
use route::*;

const KS: [usize; 5] = [1, 2, 3, 8, 500];

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

/// Series of exact ties and gaps, each with the window length to run it at.
fn tie_heavy() -> Vec<(&'static str, Vec<f64>, usize)> {
    let level = 1000.0;
    let few_levels = |k: usize| ((k * 7 + k / 3) % 4) as f64;
    let mut gaps_everywhere: Vec<f64> = (0..130).map(|k| ((k * 5) % 9) as f64).collect();
    gaps_everywhere[..4].fill(f64::NAN);
    gaps_everywhere[40..43].fill(f64::NAN);
    gaps_everywhere[64] = f64::NAN;
    gaps_everywhere[126..].fill(f64::NAN);
    vec![
        ("constant", vec![5.0; 120], 4),
        (
            "constant runs between levels",
            {
                (0..150)
                    .map(|k| if (k / 17) % 2 == 0 { 0.0 } else { level })
                    .collect()
            },
            5,
        ),
        ("period 2", (0..120).map(|k| (k % 2) as f64).collect(), 4),
        ("period 4", (0..120).map(|k| (k % 4) as f64).collect(), 4),
        ("period 7", (0..140).map(|k| (k % 7) as f64).collect(), 6),
        ("few levels", (0..140).map(few_levels).collect(), 4),
        (
            "exact repeats with offsets",
            {
                (0..150)
                    .map(|k| ((k % 6) as f64) + 10.0 * ((k / 30) as f64))
                    .collect()
            },
            6,
        ),
        ("gaps at the start, middle and end", gaps_everywhere, 5),
        (
            "leading gap and a flat run",
            {
                let mut series: Vec<f64> = (0..110).map(|k| ((k * 3) % 8) as f64).collect();
                series[..9].fill(f64::NAN);
                series[30..60].fill(2.0);
                series
            },
            4,
        ),
    ]
}

fn assert_something_to_compare(rows: &[(u64, Vec<Bits>)], k: usize, name: &str) {
    if k >= 2 && name != "constant" {
        assert!(
            rows.iter().any(|(_, kept)| kept.len() >= 2),
            "{name}: no row kept two neighbors"
        );
    }
    assert!(rows.iter().any(|(_, kept)| !kept.is_empty()), "{name}");
}

#[test]
fn growing_equals_the_exact_reference_on_ties_and_gaps() {
    for force in [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ] {
        let _forced = forced_route(force);
        for (name, series, m) in tie_heavy() {
            let samples = samples(&series, unit());
            for k in KS {
                let expected = stream_topk_rows(&samples, m, k);
                assert_something_to_compare(&expected, k, name);
                assert_eq!(
                    growing_topk_rows(&series, m, unit(), k),
                    expected,
                    "{name} k {k} {force:?}"
                );
            }
        }
    }
}

#[test]
fn bounded_equals_the_windowed_exact_reference_on_ties_and_gaps() {
    for force in [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ] {
        let _forced = forced_route(force);
        for (name, series, m) in tie_heavy() {
            let samples = samples(&series, unit());
            for w in [m, m + 3, 2 * m + 5, 40, series.len() + 7] {
                for k in KS {
                    let got = bounded_topk_rows(&series, m, w, unit(), k);
                    assert_eq!(got.len(), series.len() + 1 - m);
                    for (row, kept) in &got {
                        assert_eq!(
                            kept,
                            &stream_window_topk_bits(&samples, m, w, *row as usize, k),
                            "{name} w {w} k {k} row {row} {force:?}"
                        );
                    }
                    if w > m + 8 {
                        assert_something_to_compare(&got, k, name);
                    }
                }
            }
        }
    }
}

/// A fixed series' last row keeps these three neighbors (checked against numpy's z-normalized
/// distances: 0.419, 0.575, 0.758): `rho` and `d²` as literal bits, then the index.
#[test]
fn a_fixed_case_is_pinned_to_literal_bits() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = [0, 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5].map(f64::from).to_vec();
    let rows = growing_topk_rows(&series, 3, unit(), 3);
    let last = &rows[rows.len() - 1];
    assert_eq!(last.0, 9);
    assert_eq!(
        last.1,
        vec![
            (9, 0x3fef_102e_9868_d705, 0x3fc6_7ba1_b62b_d76f, 3),
            (9, 0x3fee_3cb6_6051_f5e0, 0x3fd5_2773_7c28_797d, 1),
            (9, 0x3fec_ef29_285b_13b9, 0x3fe2_6509_0ddd_89a6, 6),
        ]
    );
}

#[test]
fn k_of_one_keeps_the_single_neighbor_stream() {
    let _serial = forced_route(RouteForce::default());
    let (_, series, m) = tie_heavy().remove(5);
    let mut single = GrowingStream::new(nz(m), unit());
    let mut kept = GrowingStream::with_k(nz(m), unit(), nz(1));
    for &x in &series {
        let single = single.update(x);
        let (row, entries) = match kept.update_topk(x) {
            Some((row, entries)) => (Some(row), entries.to_vec()),
            None => (None, Vec::new()),
        };
        assert_eq!(single.map(|(r, _)| r), row);
        if let Some((_, entry)) = single {
            let first = entries
                .first()
                .copied()
                .unwrap_or(ladde::entry::Entry::NONE);
            assert_eq!(bits(0, first), bits(0, entry));
        }
    }
    assert_eq!(single.checkpoint(), kept.checkpoint());
}

fn growing_after(series: &[f64], m: usize, k: usize) -> GrowingStream {
    let mut stream = GrowingStream::with_k(nz(m), unit(), nz(k));
    for &x in series {
        let _ = stream.update(x);
    }
    stream
}

fn bounded_after(series: &[f64], m: usize, w: usize, k: usize) -> BoundedStream {
    let mut stream = BoundedStream::with_k(nz(m), nz(w), unit(), nz(k)).unwrap();
    for &x in series {
        let _ = stream.update(x);
    }
    stream
}

fn rows_of_growing(stream: &mut GrowingStream, series: &[f64]) -> Vec<(u64, Vec<Bits>)> {
    series
        .iter()
        .filter_map(|&x| {
            stream
                .update_topk(x)
                .map(|(r, e)| (r.get().unwrap(), e.to_vec()))
        })
        .map(|(row, e)| (row, topk_bits(row, &e)))
        .collect()
}

fn rows_of_bounded(stream: &mut BoundedStream, series: &[f64]) -> Vec<(u64, Vec<Bits>)> {
    series
        .iter()
        .filter_map(|&x| {
            stream
                .update_topk(x)
                .map(|(r, e)| (r.get().unwrap(), e.to_vec()))
        })
        .map(|(row, e)| (row, topk_bits(row, &e)))
        .collect()
}

#[test]
fn a_restored_growing_stream_continues_with_its_k() {
    let _serial = forced_route(RouteForce::default());
    for (name, series, m) in tie_heavy() {
        for k in [2, 3, 8] {
            for split in [0, m - 1, m, m + 2, series.len() / 2, series.len() - 1] {
                let (head, tail) = series.split_at(split);
                let mut never_stopped = growing_after(head, m, k);
                let mut restored = GrowingStream::restore(&never_stopped.checkpoint()).unwrap();
                assert_eq!(restored.k().get(), k, "{name}");
                assert_eq!(
                    rows_of_growing(&mut restored, tail),
                    rows_of_growing(&mut never_stopped, tail),
                    "{name} k {k} split {split}"
                );
                assert_eq!(restored.checkpoint(), never_stopped.checkpoint());
            }
        }
    }
}

#[test]
fn a_restored_bounded_stream_continues_with_its_k() {
    let _serial = forced_route(RouteForce::default());
    for (name, series, m) in tie_heavy() {
        for w in [m, m + 6, 3 * m + 1] {
            for k in [2, 3, 8] {
                for split in [0, m - 1, m, w, w + 3, series.len() / 2, series.len() - 1] {
                    let (head, tail) = series.split_at(split.min(series.len()));
                    let mut never_stopped = bounded_after(head, m, w, k);
                    let mut restored = BoundedStream::restore(&never_stopped.checkpoint()).unwrap();
                    assert_eq!(restored.k().get(), k, "{name}");
                    assert_eq!(
                        rows_of_bounded(&mut restored, tail),
                        rows_of_bounded(&mut never_stopped, tail),
                        "{name} w {w} k {k} split {split}"
                    );
                    assert_eq!(restored.checkpoint(), never_stopped.checkpoint());
                }
            }
        }
    }
}

/// The checkpoint's header: magic, the version and the stream kind.
fn version(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[4], bytes[5]])
}

#[test]
fn k_of_one_writes_the_version_without_k_and_more_writes_the_next() {
    let _serial = forced_route(RouteForce::default());
    let (_, series, m) = tie_heavy().remove(7);
    assert_eq!(version(&growing_after(&series, m, 1).checkpoint()), 3);
    assert_eq!(version(&bounded_after(&series, m, 30, 1).checkpoint()), 3);
    assert_eq!(version(&growing_after(&series, m, 2).checkpoint()), 4);
    assert_eq!(version(&bounded_after(&series, m, 30, 2).checkpoint()), 4);
    assert_eq!(
        GrowingStream::restore(&growing_after(&series, m, 1).checkpoint())
            .unwrap()
            .k()
            .get(),
        1
    );
    assert_eq!(
        BoundedStream::restore(&bounded_after(&series, m, 30, 1).checkpoint())
            .unwrap()
            .k()
            .get(),
        1
    );
}

/// The byte offset of `k` in a checkpoint that has one: after the 7-byte header and `m`.
const K_AT: usize = 15;

fn with_k_bytes(mut bytes: Vec<u8>, k: u64) -> Vec<u8> {
    bytes[K_AT..K_AT + 8].copy_from_slice(&k.to_le_bytes());
    with_recomputed_checksum(&bytes)
}

#[test]
fn a_checkpoint_with_a_bad_k_is_refused() {
    let _serial = forced_route(RouteForce::default());
    let (_, series, m) = tie_heavy().remove(5);
    let growing = growing_after(&series, m, 3).checkpoint();
    let bounded = bounded_after(&series, m, 30, 3).checkpoint();
    for k in [0u64, 1] {
        assert_eq!(
            GrowingStream::restore(&with_k_bytes(growing.clone(), k)).err(),
            Some(CheckpointError::Inconsistent)
        );
        assert_eq!(
            BoundedStream::restore(&with_k_bytes(bounded.clone(), k)).err(),
            Some(CheckpointError::Inconsistent)
        );
    }
    // A k that nothing could hold is a larger stream, not a refusal or an allocation.
    let huge = 1u64 << 62;
    let growing = GrowingStream::restore(&with_k_bytes(growing.clone(), huge)).unwrap();
    assert_eq!(growing.k().get() as u64, huge);
    let bounded = BoundedStream::restore(&with_k_bytes(bounded.clone(), huge)).unwrap();
    assert_eq!(bounded.k().get() as u64, huge);
    // A version-3 checkpoint cannot carry a k: its bytes shifted into the next version are not
    // one either.
    let mut v4_as_v3 = growing_after(&series, m, 3).checkpoint();
    v4_as_v3[4] = 3;
    assert!(GrowingStream::restore(&with_recomputed_checksum(&v4_as_v3)).is_err());
    // Version 5 carries more after the same body, so a version-4 body is not one.
    let mut v4_as_v5 = growing_after(&series, m, 3).checkpoint();
    v4_as_v5[4] = 5;
    assert!(GrowingStream::restore(&with_recomputed_checksum(&v4_as_v5)).is_err());
    // Version 6 wraps a whole checkpoint, so a version-4 body is not one.
    let mut v4_as_v6 = growing_after(&series, m, 3).checkpoint();
    v4_as_v6[4] = 6;
    assert!(GrowingStream::restore(&with_recomputed_checksum(&v4_as_v6)).is_err());
    let mut v7 = growing_after(&series, m, 3).checkpoint();
    v7[4] = 7;
    assert_eq!(
        GrowingStream::restore(&with_recomputed_checksum(&v7)).err(),
        Some(CheckpointError::BadVersion)
    );
}

#[test]
fn a_k_beyond_the_windows_is_padding_not_an_error() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = (0..20).map(|k| ((k * 7) % 5) as f64).collect();
    let rows = growing_topk_rows(&series, 4, unit(), 1_000_000_000);
    let samples = samples(&series, unit());
    assert_eq!(rows, stream_topk_rows(&samples, 4, 500));
}
