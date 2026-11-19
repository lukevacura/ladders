#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A checkpoint is the same bytes whatever route wrote it and resumes under any other route.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::checkpoint::{CheckpointError, with_recomputed_checksum};
use ladde::narrow::Path;
use ladde::route::{RouteForce, forced_route};
use ladde::stream::GrowingStream;
use ladde::stream_api::Stream;
use proptest::prelude::*;
use route::*;

/// Checkpoint bytes of `stream` after each prefix length in `splits`, with `force` held for the
/// whole run.
fn checkpoints<S: Stream>(
    force: RouteForce,
    mut stream: S,
    series: &[f64],
    splits: &[usize],
    checkpoint: impl Fn(&S) -> Vec<u8>,
) -> Vec<Vec<u8>> {
    let _forced = forced_route(force);
    let mut out = Vec::new();
    for position in 0..=series.len() {
        if splits.contains(&position) {
            out.push(checkpoint(&stream));
        }
        if let Some(&x) = series.get(position) {
            let _ = stream.update(x);
        }
    }
    out
}

fn split_points(len: usize, picks: &[usize]) -> Vec<usize> {
    picks.iter().map(|p| p % (len + 1)).collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn growing_checkpoint_bytes_do_not_depend_on_the_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        picks in prop::collection::vec(any::<usize>(), 1..4),
    ) {
        let splits = split_points(series.len(), &picks);
        let run = |force| {
            checkpoints(force, GrowingStream::new(nz(m), scale), &series, &splits, |s| {
                s.checkpoint()
            })
        };
        let natural = run(RouteForce::default());
        for force in forces(seed) {
            prop_assert_eq!(&run(force), &natural, "{:?}", force);
        }
        let _natural = forced_route(RouteForce::default());
        for bytes in &natural {
            let restored = GrowingStream::restore(bytes).expect("a genuine checkpoint restores");
            prop_assert_eq!(&restored.checkpoint(), bytes);
        }
    }

    #[test]
    fn bounded_checkpoint_bytes_do_not_depend_on_the_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..40,
        picks in prop::collection::vec(any::<usize>(), 1..4),
    ) {
        let splits = split_points(series.len(), &picks);
        let run = |force| {
            let stream = BoundedStream::new(nz(m), nz(m + extra), scale).expect("w is at least m");
            checkpoints(force, stream, &series, &splits, |s| s.checkpoint())
        };
        let natural = run(RouteForce::default());
        for force in forces(seed) {
            prop_assert_eq!(&run(force), &natural, "{:?} w {}", force, m + extra);
        }
        let _natural = forced_route(RouteForce::default());
        for bytes in &natural {
            let restored = BoundedStream::restore(bytes).expect("a genuine checkpoint restores");
            prop_assert_eq!(&restored.checkpoint(), bytes);
        }
    }

    #[test]
    fn a_restored_stream_chooses_its_route_as_a_fresh_one_does(
        (series, m, scale) in cases(),
        extra in 0usize..40,
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        let _natural = forced_route(RouteForce::default());
        let mut growing = GrowingStream::new(nz(m), scale);
        let mut bounded = BoundedStream::new(nz(m), nz(m + extra), scale).expect("w is at least m");
        for &x in &series[..split] {
            let _ = growing.update(x);
            let _ = bounded.update(x);
        }
        let restored = GrowingStream::restore(&growing.checkpoint()).unwrap();
        prop_assert_eq!(restored.path(), growing.path());
        let restored = BoundedStream::restore(&bounded.checkpoint()).unwrap();
        if bounded.path() == Path::Narrow {
            prop_assert_eq!(restored.path(), Path::Narrow);
        }
    }

    #[test]
    fn growing_resumes_under_another_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        (before, after) in (0usize..8, 0usize..8),
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        let routes = forces(seed);
        let expected = stream_profile_bits(&samples(&series, scale), m);
        let (mut rows, bytes) = {
            let _forced = forced_route(routes[before]);
            let mut stream = GrowingStream::new(nz(m), scale);
            let rows: Vec<_> = series[..split].iter().filter_map(|&x| stream.update(x)).collect();
            (rows, stream.checkpoint())
        };
        {
            let _forced = forced_route(routes[after]);
            let mut restored = GrowingStream::restore(&bytes).expect("a genuine checkpoint restores");
            rows.extend(series[split..].iter().filter_map(|&x| restored.update(x)));
        }
        let never_stopped: Vec<_> = {
            let _natural = forced_route(RouteForce::default());
            let mut stream = GrowingStream::new(nz(m), scale);
            series.iter().filter_map(|&x| stream.update(x)).collect()
        };
        prop_assert_eq!(&stream_bits(&rows), &stream_bits(&never_stopped));
        prop_assert_eq!(
            &stream_bits(&rows), &expected,
            "before {:?} after {:?} split {}", routes[before], routes[after], split
        );
    }

    #[test]
    fn bounded_resumes_under_another_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..40,
        (before, after) in (0usize..8, 0usize..8),
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        let w = m + extra;
        let routes = forces(seed);
        let samples = samples(&series, scale);
        let new = || BoundedStream::new(nz(m), nz(w), scale).expect("w is at least m");
        let (mut rows, bytes) = {
            let _forced = forced_route(routes[before]);
            let mut stream = new();
            let rows: Vec<_> = series[..split].iter().filter_map(|&x| stream.update(x)).collect();
            (rows, stream.checkpoint())
        };
        {
            let _forced = forced_route(routes[after]);
            let mut restored = BoundedStream::restore(&bytes).expect("a genuine checkpoint restores");
            rows.extend(series[split..].iter().filter_map(|&x| restored.update(x)));
        }
        let never_stopped: Vec<_> = {
            let _natural = forced_route(RouteForce::default());
            let mut stream = new();
            series.iter().filter_map(|&x| stream.update(x)).collect()
        };
        let got = stream_bits(&rows);
        prop_assert_eq!(&got, &stream_bits(&never_stopped));
        for &row in &got {
            prop_assert_eq!(
                row,
                stream_window_bits(&samples, m, w, row.0 as usize),
                "before {:?} after {:?} split {} w {}", routes[before], routes[after], split, w
            );
        }
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn below(state: &mut u64, n: usize) -> usize {
    (xorshift(state) % n as u64) as usize
}

/// One damaged copy of `bytes`: a truncation, a bit flip, an overwritten byte, or three flips.
fn damaged(bytes: &[u8], state: &mut u64) -> Vec<u8> {
    let mut out = bytes.to_vec();
    match below(state, 4) {
        0 => out.truncate(below(state, bytes.len())),
        1 => out[below(state, bytes.len())] ^= 1 << below(state, 8),
        2 => out[below(state, bytes.len())] = xorshift(state) as u8,
        _ => {
            for _ in 0..3 {
                out[below(state, bytes.len())] ^= 1 << below(state, 8);
            }
        }
    }
    out
}

/// Every damaged copy of a genuine checkpoint must fail to restore. When the checksum is
/// recomputed over damage between the header and the presence flags of a checkpoint holding at
/// least `m` ticks, the restore either fails or continues on the very bits of the stream that
/// never stopped. Only the checksum covers the header fields (kind, sizes, scale, flags, row
/// count), the presence flags, and the ticks of a stream that has not yet completed a window:
/// altered, they can name a different valid stream. `flag_ticks` is how many ticks the
/// checkpoint holds.
fn assert_damage_never_diverges<S: Stream>(
    (force, seed): (RouteForce, u64),
    series: &[f64],
    split: usize,
    (header_len, flag_ticks, m): (usize, usize, usize),
    new: impl Fn() -> S,
    checkpoint: impl Fn(&S) -> Vec<u8>,
    restore: impl Fn(&[u8]) -> Result<S, CheckpointError>,
) -> Result<(), TestCaseError> {
    let _forced = forced_route(force);
    let mut stream = new();
    for &x in &series[..split] {
        let _ = stream.update(x);
    }
    let bytes = checkpoint(&stream);
    let expected: Vec<_> = series[split..]
        .iter()
        .filter_map(|&x| stream.update(x))
        .collect();
    let expected = stream_bits(&expected);
    let continues_equal = |candidate: &[u8]| match restore(candidate) {
        Err(_) => true,
        Ok(mut restored) => {
            let rows: Vec<_> = series[split..]
                .iter()
                .filter_map(|&x| restored.update(x))
                .collect();
            stream_bits(&rows) == expected
        }
    };
    let body_len = bytes.len() - 8;
    let flags_start = body_len - (8 + flag_ticks.div_ceil(8));

    let mut state = seed | 1;
    for _ in 0..60 {
        let candidate = damaged(&bytes, &mut state);
        if candidate != bytes {
            prop_assert!(restore(&candidate).is_err(), "damage restored");
        }
        let resealed = with_recomputed_checksum(&candidate);
        let in_fields = resealed.len() != bytes.len()
            || (0..resealed.len())
                .filter(|&at| resealed[at] != bytes[at])
                .all(|at| (header_len..flags_start).contains(&at));
        if resealed != bytes && in_fields && flag_ticks >= m {
            prop_assert!(continues_equal(&resealed), "resealed damage diverged");
        }
    }
    for at in flags_start..body_len {
        for bit in 0..8 {
            let mut candidate = bytes.clone();
            candidate[at] ^= 1 << bit;
            prop_assert!(restore(&candidate).is_err(), "flag flip restored");
        }
    }
    Ok(())
}

const GROWING_HEADER_LEN: usize = 4 + 2 + 1 + 8 + 8 + 1;
const BOUNDED_HEADER_LEN: usize = GROWING_HEADER_LEN + 8 + 8;

proptest! {
    #![proptest_config(testkit::generators::seeded(48))]

    #[test]
    fn damaged_growing_checkpoints_fail_or_continue_equal(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        which in 0usize..8,
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        assert_damage_never_diverges(
            (forces(seed)[which], seed), &series, split, (GROWING_HEADER_LEN, split, m),
            || GrowingStream::new(nz(m), scale),
            |s| s.checkpoint(),
            GrowingStream::restore,
        )?;
    }

    #[test]
    fn damaged_bounded_checkpoints_fail_or_continue_equal(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..12,
        which in 0usize..8,
        pick in any::<usize>(),
    ) {
        let split = pick % (series.len() + 1);
        assert_damage_never_diverges(
            (forces(seed)[which], seed), &series, split, (BOUNDED_HEADER_LEN, split.min(m + extra), m),
            || BoundedStream::new(nz(m), nz(m + extra), scale).expect("w is at least m"),
            |s| s.checkpoint(),
            BoundedStream::restore,
        )?;
    }
}

/// Before a window completes, nothing but the checksum ties the stored ticks to anything: an
/// altered tick restores as a different valid stream once the checksum is recomputed, and is
/// refused while it is not.
#[test]
fn ticks_stored_before_the_first_window_are_covered_by_the_checksum_alone() {
    let _natural = forced_route(RouteForce::default());
    let (m, scale) = (nz(4), ladde::quantize::Scale::new(1.0).unwrap());
    for stored in 1..4 {
        let mut growing = GrowingStream::new(m, scale);
        let mut bounded = BoundedStream::new(m, nz(9), scale).expect("w is at least m");
        for x in 1..=stored {
            let _ = growing.update(x as f64);
            let _ = bounded.update(x as f64);
        }
        for (bytes, header_len, restores) in [
            (
                growing.checkpoint(),
                GROWING_HEADER_LEN,
                restores_growing as fn(&[u8]) -> bool,
            ),
            (bounded.checkpoint(), BOUNDED_HEADER_LEN, restores_bounded),
        ] {
            for tick in 0..stored {
                let at = header_len + 8 + 8 * tick;
                let mut altered = bytes.clone();
                altered[at] ^= 1;
                assert!(!restores(&altered), "raw flip at {at} restored");
                assert!(
                    restores(&with_recomputed_checksum(&altered)),
                    "flip at {at}"
                );
            }
        }
    }
}

fn restores_growing(bytes: &[u8]) -> bool {
    GrowingStream::restore(bytes).is_ok()
}

fn restores_bounded(bytes: &[u8]) -> bool {
    BoundedStream::restore(bytes).is_ok()
}
