#![cfg(not(any(loom, shuttle)))]

//! Peak heap use of the multi-dimensional motifs grows linearly with the series, on series of
//! exact ties as on a walk: a constant series puts every window within every threshold of every
//! other, and no candidate list may grow with the pairs. Its own test binary, with a process-wide
//! peak-tracking allocator, and a single test so nothing else allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::index::SampleIndex;
use ladde::motif::{Cutoffs, MultiMotifParams, mmotifs, mmotifs_nonnorm};
use ladde::pattern::MaxDistance;
use ladde::quantize::Scale;
use peak_alloc::peak_of;

fn walk(n: usize, salt: u64) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ salt;
    let mut level = 1e10;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 56) as i64 % 5 - 2) as f64;
            level
        })
        .collect()
}

type Series = dyn Fn(usize, u64) -> Vec<f64>;

/// A profile of arbitrary distances with many ties and neighbors scattered over the series: the
/// procedure takes it as given, so the engine that would make one is not what is measured.
fn profile_of(n: usize, dimensions: usize) -> (Vec<Vec<f64>>, Vec<Vec<SampleIndex>>) {
    let windows = n - 15;
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

/// Peak bytes of the unbounded run (every window matches every other) and of the many-motif run,
/// `normalized` or not.
fn peaks(make: &Series, n: usize, normalized: bool) -> [usize; 2] {
    let m = NonZeroUsize::new(16).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let series: Vec<Vec<f64>> = (0..3).map(|a| make(n, a)).collect();
    let dimensions: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let (distance, index) = profile_of(n, 3);
    let set: Vec<(&[f64], Scale)> = dimensions.iter().map(|&s| (s, scale)).collect();
    let everything = MultiMotifParams {
        max_motifs: 5,
        max_matches: None,
        max_distance: MaxDistance::Value(f64::INFINITY),
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..MultiMotifParams::default()
    };
    let many = MultiMotifParams {
        max_motifs: 4,
        min_neighbors: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..MultiMotifParams::default()
    };
    [everything, many].map(|params| {
        peak_of(|| {
            let found = if normalized {
                mmotifs(&set, m, &distance, &index, &params)
            } else {
                mmotifs_nonnorm(&dimensions, m, scale, &distance, &index, &params)
            }
            .unwrap();
            assert!(!found.is_empty());
        })
    })
}

#[test]
fn peak_heap_is_linear_in_the_series_on_ties_and_walks() {
    let (small, big) = (2_000, 8_000);
    let constant = |n: usize, _| vec![5.0; n];
    let period_two = |n: usize, a: u64| (0..n).map(|k| ((k + a as usize) % 2) as f64).collect();
    let walks = |n: usize, a: u64| walk(n, a);
    let kinds: [(&str, &Series); 3] = [
        ("constant", &constant),
        ("period two", &period_two),
        ("walk", &walks),
    ];
    for (name, make, normalized) in kinds
        .into_iter()
        .flat_map(|(name, make)| [(name, make, true), (name, make, false)])
    {
        let (few, many) = (peaks(make, small, normalized), peaks(make, big, normalized));
        let name = format!("{name}, normalized {normalized}");
        for (which, (few, many)) in ["everything", "many motifs"]
            .iter()
            .zip(few.iter().zip(&many))
        {
            // The ticks, the windows' values, the profile's copy and one round's distances of
            // three dimensions: a few hundred bytes per window.
            assert!(
                *many < big * 1024,
                "{name} {which}: {many} bytes at n = {big}"
            );
            assert!(
                many.saturating_sub(*few) < (big - small) * 1024,
                "{name} {which}: {few} bytes at n = {small}, {many} at n = {big}"
            );
        }
    }
}
