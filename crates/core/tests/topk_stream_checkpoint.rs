#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Checkpoints of the streams that keep `k` neighbors per row: the version names the layout (a
//! `k` of one keeps the old one), a restored stream keeps its `k` and continues bit for bit, a
//! `k` no stream could hold is refused or harmless, and damaged bytes never panic or ask the
//! allocator for much.

use core::num::NonZeroUsize;

use ladde::bounded::BoundedStream;
use ladde::checkpoint::{CheckpointError, with_recomputed_checksum};
use ladde::quantize::Scale;
use ladde::stream::GrowingStream;
use testkit::alloc_count::{CountingAllocator, large_requests_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn data() -> Vec<f64> {
    (0..200)
        .map(|k| {
            if k % 23 < 2 {
                f64::NAN
            } else {
                ((k * 5) % 9) as f64 + (k / 60) as f64 * 1e3
            }
        })
        .collect()
}

type Rows = Vec<(u64, Vec<(u64, u64, u64)>)>;

fn bits(kept: &[ladde::entry::Entry]) -> Vec<(u64, u64, u64)> {
    kept.iter()
        .map(|e| {
            (
                e.rho().to_bits(),
                e.squared_distance().to_bits(),
                e.index().get().unwrap_or(u64::MAX),
            )
        })
        .collect()
}

fn growing_rows(stream: &mut GrowingStream, samples: &[f64]) -> Rows {
    samples
        .iter()
        .filter_map(|&x| {
            stream
                .update_topk(x)
                .map(|(row, kept)| (row.get().unwrap(), bits(kept)))
        })
        .collect()
}

fn bounded_rows(stream: &mut BoundedStream, samples: &[f64]) -> Rows {
    samples
        .iter()
        .filter_map(|&x| {
            stream
                .update_topk(x)
                .map(|(row, kept)| (row.get().unwrap(), bits(kept)))
        })
        .collect()
}

fn version(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[4], bytes[5]])
}

/// The byte offset of `k`: after the magic, the version, the kind and `m`.
const K_AT: usize = 15;

fn with_k(bytes: &[u8], k: u64) -> Vec<u8> {
    let mut changed = bytes.to_vec();
    changed[K_AT..K_AT + 8].copy_from_slice(&k.to_le_bytes());
    with_recomputed_checksum(&changed)
}

#[test]
fn a_restored_stream_keeps_its_k_and_continues_bit_for_bit() {
    let v = data();
    for k in [2usize, 3, 16] {
        let mut a = GrowingStream::with_k(nz(5), unit(), nz(k));
        let mut b = GrowingStream::with_k(nz(5), unit(), nz(k));
        growing_rows(&mut a, &v[..100]);
        growing_rows(&mut b, &v[..100]);
        let mut restored = GrowingStream::restore(&a.checkpoint()).unwrap();
        assert_eq!(restored.k().get(), k);
        assert_eq!(
            growing_rows(&mut restored, &v[100..]),
            growing_rows(&mut b, &v[100..])
        );
        assert_eq!(restored.checkpoint(), b.checkpoint());

        let mut a = BoundedStream::with_k(nz(5), nz(40), unit(), nz(k)).unwrap();
        let mut b = BoundedStream::with_k(nz(5), nz(40), unit(), nz(k)).unwrap();
        bounded_rows(&mut a, &v[..100]);
        bounded_rows(&mut b, &v[..100]);
        let mut restored = BoundedStream::restore(&a.checkpoint()).unwrap();
        assert_eq!(restored.k().get(), k);
        assert_eq!(
            bounded_rows(&mut restored, &v[100..]),
            bounded_rows(&mut b, &v[100..])
        );
        assert_eq!(restored.checkpoint(), b.checkpoint());
    }
}

#[test]
fn the_version_names_the_layout_and_a_bad_k_is_refused() {
    let v = data();
    let mut growing = GrowingStream::with_k(nz(5), unit(), nz(3));
    growing_rows(&mut growing, &v[..100]);
    let mut bounded = BoundedStream::with_k(nz(5), nz(40), unit(), nz(3)).unwrap();
    bounded_rows(&mut bounded, &v[..100]);
    let (growing_bytes, bounded_bytes) = (growing.checkpoint(), bounded.checkpoint());
    assert_eq!(version(&growing_bytes), 4);
    assert_eq!(version(&bounded_bytes), 4);

    let mut one = GrowingStream::new(nz(5), unit());
    growing_rows(&mut one, &v[..100]);
    let mut one_bounded = BoundedStream::new(nz(5), nz(40), unit()).unwrap();
    bounded_rows(&mut one_bounded, &v[..100]);
    assert_eq!(version(&one.checkpoint()), 3);
    assert_eq!(version(&one_bounded.checkpoint()), 3);

    for k in [0u64, 1] {
        assert_eq!(
            GrowingStream::restore(&with_k(&growing_bytes, k)).err(),
            Some(CheckpointError::Inconsistent),
            "growing k {k}"
        );
        assert_eq!(
            BoundedStream::restore(&with_k(&bounded_bytes, k)).err(),
            Some(CheckpointError::Inconsistent),
            "bounded k {k}"
        );
    }
    let mut flipped = growing_bytes.clone();
    flipped[K_AT] ^= 1;
    assert_eq!(
        GrowingStream::restore(&flipped).err(),
        Some(CheckpointError::BadChecksum)
    );

    // A version that disagrees with the body is never read as the other layout.
    for (body, label) in [
        (one.checkpoint(), 4u16),
        (growing_bytes.clone(), 3),
        (growing_bytes.clone(), 5),
        (growing_bytes.clone(), 2),
    ] {
        let mut relabelled = body.clone();
        relabelled[4..6].copy_from_slice(&label.to_le_bytes());
        assert!(
            GrowingStream::restore(&with_recomputed_checksum(&relabelled)).is_err(),
            "growing labelled {label}"
        );
        let mut relabelled = if body == growing_bytes {
            bounded_bytes.clone()
        } else {
            one_bounded.checkpoint()
        };
        relabelled[4..6].copy_from_slice(&label.to_le_bytes());
        assert!(
            BoundedStream::restore(&with_recomputed_checksum(&relabelled)).is_err(),
            "bounded labelled {label}"
        );
    }
}

#[test]
fn a_huge_k_costs_no_large_allocation() {
    let v = data();
    let mut growing = GrowingStream::with_k(nz(5), unit(), nz(3));
    growing_rows(&mut growing, &v[..100]);
    let mut bounded = BoundedStream::with_k(nz(5), nz(40), unit(), nz(3)).unwrap();
    bounded_rows(&mut bounded, &v[..100]);
    let (growing_bytes, bounded_bytes) = (growing.checkpoint(), bounded.checkpoint());
    for k in [u64::MAX, 1 << 62, 1 << 40, 1 << 33, 1 << 24] {
        let (_, large) = large_requests_in(1 << 22, || {
            if let Ok(mut stream) = GrowingStream::restore(&with_k(&growing_bytes, k)) {
                growing_rows(&mut stream, &v[100..]);
            }
            if let Ok(mut stream) = BoundedStream::restore(&with_k(&bounded_bytes, k)) {
                bounded_rows(&mut stream, &v[100..]);
            }
        });
        assert!(large.is_empty(), "k {k}: {large:?}");
    }
}

#[test]
fn damaged_bytes_never_panic_or_ask_for_much() {
    let v = data();
    let mut growing = GrowingStream::with_k(nz(5), unit(), nz(3));
    growing_rows(&mut growing, &v[..100]);
    let mut bounded = BoundedStream::with_k(nz(5), nz(40), unit(), nz(3)).unwrap();
    bounded_rows(&mut bounded, &v[..100]);
    for bytes in [growing.checkpoint(), bounded.checkpoint()] {
        let (_, large) = large_requests_in(1 << 22, || {
            for position in 0..bytes.len().saturating_sub(8) {
                for value in [0u8, 1, 0x7f, 0xff] {
                    let mut changed = bytes.clone();
                    changed[position] = value;
                    let changed = with_recomputed_checksum(&changed);
                    let _ = GrowingStream::restore(&changed);
                    let _ = BoundedStream::restore(&changed);
                }
            }
            for cut in 0..bytes.len() {
                let _ = GrowingStream::restore(&bytes[..cut]);
                let _ = BoundedStream::restore(&bytes[..cut]);
            }
        });
        assert!(large.is_empty(), "{large:?}");
    }
}
