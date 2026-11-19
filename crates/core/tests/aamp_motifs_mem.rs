//! Peak heap use of non-normalized matching and motifs grows linearly with the series, on series
//! of exact ties as on a walk: a constant series puts every window within every threshold of every
//! other, and no candidate list may grow with the pairs. The peaks are measured single-threaded: a
//! threaded peak follows the schedule, and the threaded profile underneath is the two-sided one,
//! whose shared candidate lists grow in the order tiles merge, so not even the count of its large
//! allocations is the same from run to run. Its own test binary, with a process-wide
//! peak-tracking allocator, and a single test so nothing else allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::motif::{MotifParams, motifs_nonnorm};
use ladde::pattern::{MaxDistance, find_matches_nonnorm};
use ladde::quantize::Scale;
use peak_alloc::peak_of;

fn walk(n: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
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

type Series = dyn Fn(usize) -> Vec<f64>;

fn peaks(series: &[f64]) -> [usize; 2] {
    let m = NonZeroUsize::new(16).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let threads = NonZeroUsize::MIN;
    let params = MotifParams {
        max_motifs: 5,
        max_matches: None,
        max_distance: MaxDistance::Value(f64::INFINITY),
        cutoff: Some(f64::INFINITY),
        ..MotifParams::default()
    };
    [
        peak_of(|| {
            let found = motifs_nonnorm(series, m, scale, threads, &params);
            assert!(!found.is_empty());
        }),
        peak_of(|| {
            let found =
                find_matches_nonnorm(&series[..16], series, scale, MaxDistance::Default, None);
            assert!(!found.unwrap().is_empty());
        }),
    ]
}

#[test]
fn peak_heap_is_linear_in_the_series_on_ties_and_walks() {
    let (small, big) = (2_000, 8_000);
    let constant = |n: usize| vec![5.0; n];
    let period_two = |n: usize| (0..n).map(|k| (k % 2) as f64).collect::<Vec<_>>();
    let kinds: [(&str, &Series); 3] = [
        ("constant", &constant),
        ("period two", &period_two),
        ("walk", &walk),
    ];
    for (name, make) in kinds {
        let (few, many) = (peaks(&make(small)), peaks(&make(big)));
        for (which, (few, many)) in ["motifs", "match"].iter().zip(few.iter().zip(&many)) {
            // Ticks, the profile, the candidate list and the matches: a few hundred bytes per
            // window at most (five motifs of every window each).
            assert!(
                *many < big * 512,
                "{name} {which}: {many} bytes at n = {big}"
            );
            assert!(
                many.saturating_sub(*few) < (big - small) * 512,
                "{name} {which}: {few} bytes at n = {small}, {many} at n = {big}"
            );
        }
    }
}
