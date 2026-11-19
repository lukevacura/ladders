#![cfg(all(feature = "parallel", not(any(loom, shuttle))))]

//! Peak heap use of the non-normalized two-sided and AB-join top-k profiles stays linear in the
//! windows times the neighbors kept: on series where every pair ties exactly, quadrupling the
//! series at most about quadruples the peak, whatever `k`; a huge `k` sizes nothing by itself; and
//! a `k` past the windows costs what the neighbors it keeps cost. Its own test binary, with a
//! process-wide peak-tracking allocator and a single test so nothing else allocates meanwhile.
//! Every run uses one thread, so tiles run in a fixed order and the peak is deterministic.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::batch::nonnorm::{ab_join_topk, two_sided_topk_profile};
use ladde::quantize::Scale;
use peak_alloc::peak_of;

type Make = fn(usize) -> Vec<f64>;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

fn m() -> NonZeroUsize {
    nz(8)
}

fn two_sided_peak(series: &[f64], k: usize) -> usize {
    peak_of(|| {
        let profile = two_sided_topk_profile(series, m(), scale(), nz(1), nz(k)).unwrap();
        assert_eq!(profile.len(), series.len() - 7);
    })
}

fn join_peak(series: &[f64], k: usize) -> usize {
    peak_of(|| {
        let profile = ab_join_topk(series, series, m(), scale(), nz(1), nz(k)).unwrap();
        assert_eq!(profile.len(), series.len() - 7);
    })
}

#[test]
fn peak_heap_is_linear_in_the_windows_and_the_neighbors_kept() {
    let series: [(&str, Make); 3] = [
        ("constant", |n| vec![3.0; n]),
        ("period four", |n| {
            (0..n).map(|i| [0.0, 2.0, 1.0, 3.0][i % 4]).collect()
        }),
        ("walk", |n| {
            let mut state = 0x9E37_79B9_7F4A_7C15_u64;
            let mut level = 1e6;
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    level += ((state >> 56) as i64 % 5 - 2) as f64;
                    level
                })
                .collect()
        }),
    ];
    for (name, make) in series {
        for (k, small, large) in [
            (1usize, 2_000usize, 8_000usize),
            (4, 2_000, 8_000),
            (48, 1_000, 4_000),
        ] {
            for (which, run, fixed, per_neighbor) in [
                (
                    "two-sided",
                    &two_sided_peak as &dyn Fn(&[f64], usize) -> usize,
                    512,
                    256,
                ),
                ("join", &join_peak, 256, 128),
            ] {
                let (few, many) = (run(&make(small), k), run(&make(large), k));
                assert!(
                    many <= few * 11 / 2,
                    "{which} {name}, k {k}: peak {many} bytes at {large} vs {few} at {small}"
                );
                // The flat entries, the lists and the sides' right-neighbor lists: a fixed
                // number of bytes per window and per neighbor kept.
                assert!(
                    many < large * (fixed + per_neighbor * k),
                    "{which} {name}, k {k}: peak {many} bytes"
                );
            }
        }
    }
    huge_k_on_gap_series_reserves_no_more_than_the_input_can_fill();
    k_at_the_windows_and_beyond_costs_what_the_neighbors_kept_cost();
}

/// A huge `k` sizes nothing by itself: on a series with no or few complete windows the peak stays
/// a small multiple of the output, whatever the window count would square to.
fn huge_k_on_gap_series_reserves_no_more_than_the_input_can_fill() {
    let n = 20_000 + 7;
    let all_gaps = vec![f64::NAN; n];
    let mostly_gaps: Vec<f64> = (0..n)
        .map(|i| {
            if i % 4_000 < 40 {
                ((i * 7) % 11) as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    for (name, series) in [("all gaps", &all_gaps), ("mostly gaps", &mostly_gaps)] {
        for (which, peak) in [
            ("two-sided", two_sided_peak(series, usize::MAX)),
            ("join", join_peak(series, usize::MAX)),
        ] {
            assert!(
                peak < 20_000 * 1024,
                "{which} {name}: peak {peak} bytes for 20000 windows"
            );
        }
    }
}

/// On a dense series every window keeps up to `min(k, windows)` neighbors on each side, so the
/// kept neighbors number about `windows^2 / 2` per side; the peak is a fixed number of bytes per
/// kept neighbor, and a `k` past the windows changes nothing.
fn k_at_the_windows_and_beyond_costs_what_the_neighbors_kept_cost() {
    let n = 1_000 + 7;
    let windows = n - 7;
    let kept = windows * (windows - 1) / 2;
    let series: Vec<f64> = (0..n).map(|i| ((i * i / 3 + i) % 17) as f64).collect();
    for (name, run, limit) in [
        (
            "two-sided",
            &two_sided_peak as &dyn Fn(&[f64], usize) -> usize,
            256 * kept,
        ),
        ("join", &join_peak, 128 * windows * windows),
    ] {
        let at_windows = run(&series, windows);
        let beyond = run(&series, usize::MAX);
        assert!(
            at_windows <= limit,
            "{name}: peak {at_windows} bytes for {kept} kept neighbors per side"
        );
        assert_eq!(
            at_windows, beyond,
            "{name}: k past the windows changes the peak"
        );
    }
}
