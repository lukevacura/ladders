#![cfg(not(any(loom, shuttle)))]

//! A round of the multi-dimensional motifs allocates a fixed number of times, however long the
//! series: a window's mean over the subspace is formed in a buffer the round owns, so the
//! allocations of a run grow with its rounds, never with its windows. `CountingAllocator` only
//! counts once it is the process's global allocator, so this is its own test binary (see
//! `multi_no_alloc.rs`); the work runs on the counting thread.

use core::num::NonZeroUsize;

use ladde::index::SampleIndex;
use ladde::motif::{Cutoffs, MultiMotifParams, mmotifs, mmotifs_nonnorm};
use ladde::pattern::MaxDistance;
use ladde::quantize::Scale;
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// A profile of arbitrary distances with many ties and neighbors scattered over the series: the
/// procedure takes it as given, so the engine that would make one is not what is counted.
fn profile_of(n: usize, dimensions: usize) -> (Vec<Vec<f64>>, Vec<Vec<SampleIndex>>) {
    let windows = n - 7;
    let distance = (0..dimensions)
        .map(|k| {
            (0..windows)
                .map(|i| ((i * 7 + k * 3) % 13) as f64 * 0.5)
                .collect()
        })
        .collect();
    let index = (0..dimensions)
        .map(|k| {
            (0..windows)
                .map(|i| SampleIndex::new(((i * 5 + k + 1) % windows) as u64).unwrap())
                .collect()
        })
        .collect();
    (distance, index)
}

/// Ties and a walk, so neither a constant column nor a lucky tie-free one hides a path.
fn dimension(n: usize, salt: usize) -> Vec<f64> {
    let mut level = 0.0;
    (0..n)
        .map(|t| {
            level += ((t * 7 + salt * 3) % 5) as f64 - 2.0;
            if t % 9 < 3 { 1.0 } else { level }
        })
        .collect()
}

/// The allocations of one run over `n` samples and `dimensions` dimensions, and its motifs.
fn run(
    n: usize,
    dimensions: usize,
    normalized: bool,
    params: &MultiMotifParams<'_>,
) -> (usize, usize) {
    let series: Vec<Vec<f64>> = (0..dimensions).map(|a| dimension(n, a)).collect();
    let refs: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let scale = Scale::new(1.0).unwrap();
    let (distance, index) = profile_of(n, dimensions);
    let set: Vec<(&[f64], Scale)> = refs.iter().map(|&s| (s, scale)).collect();
    let (found, allocations) = allocations_in(|| {
        if normalized {
            mmotifs(&set, nz(8), &distance, &index, params)
        } else {
            mmotifs_nonnorm(&refs, nz(8), scale, &distance, &index, params)
        }
        .unwrap()
    });
    (allocations, found.len())
}

#[test]
fn allocations_per_round_do_not_grow_with_the_series() {
    let params = MultiMotifParams {
        max_motifs: 1,
        min_neighbors: 0,
        max_matches: Some(10),
        max_distance: MaxDistance::Value(f64::INFINITY),
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..MultiMotifParams::default()
    };
    for (dimensions, normalized) in [(1, true), (3, true), (6, true), (1, false), (3, false)] {
        let (few, found_few) = run(2_000, dimensions, normalized, &params);
        let (many, found_many) = run(8_000, dimensions, normalized, &params);
        assert_eq!((found_few, found_many), (1, 1));
        assert!(
            many <= few,
            "d {dimensions}, normalized {normalized}: {few} allocations at n = 2000, {many} at n = 8000"
        );
    }
}
