#![cfg(feature = "parallel")]

//! Peak heap use of the two-sided and AB-join top-k batch profiles on series where every pair
//! ties exactly stays linear in the series length: doubling it at most about doubles the peak. Its own test binary, with a process-wide peak-tracking allocator and a single test so
//! nothing else allocates meanwhile. Every run uses one thread, so tiles run in a fixed order and
//! the peak is deterministic.

use core::num::NonZeroUsize;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ladde::batch::{ab_join_topk, two_sided_topk_profile, two_sided_topk_profile_tiled};
use ladde::quantize::Scale;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Peak;

fn grow(by: usize) {
    let now = CURRENT.fetch_add(by, Ordering::SeqCst) + by;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

// SAFETY: every call is forwarded unchanged to `System`; the counters are only bookkeeping.
unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: `ptr` and `layout` are the caller's, forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= layout.size() {
            grow(new_size - layout.size());
        } else {
            CURRENT.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
        }
        // SAFETY: `ptr`, `layout` and `new_size` are the caller's, forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Peak = Peak;

fn peak_of(f: impl FnOnce()) -> usize {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    f();
    PEAK.load(Ordering::SeqCst) - base
}

type Make = fn(usize) -> Vec<f64>;

#[test]
fn peak_heap_is_linear_in_the_series_under_exact_ties() {
    let m = NonZeroUsize::new(8).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let k = NonZeroUsize::new(4).unwrap();
    let series: [(&str, Make); 2] = [
        ("constant", |n| vec![3.0; n]),
        ("period four", |n| {
            (0..n).map(|i| [0.0, 2.0, 1.0, 3.0][i % 4]).collect()
        }),
    ];
    for (name, make) in series {
        for threads in [1usize] {
            let threads = NonZeroUsize::new(threads).unwrap();
            let two_sided = |n: usize| {
                let series = make(n);
                peak_of(|| {
                    let profile = two_sided_topk_profile(&series, m, scale, threads, k).unwrap();
                    assert_eq!(profile.len(), n - 7);
                })
            };
            let join = |n: usize| {
                let series = make(n);
                peak_of(|| {
                    let profile =
                        ab_join_topk(&series, &series, m, scale, scale, threads, k).unwrap();
                    assert_eq!(profile.len(), n - 7);
                })
            };
            for (which, run) in [
                ("two-sided", &two_sided as &dyn Fn(usize) -> usize),
                ("join", &join),
            ] {
                let (small, large) = (run(3_000), run(6_000));
                assert!(
                    large <= small * 11 / 4,
                    "{which} {name}, threads {threads}: peak {large} bytes at 6000 vs {small} at 3000"
                );
                assert!(
                    large < 6_000 * 1536 * (k.get() + 1),
                    "{which} {name}, threads {threads}: peak {large} bytes"
                );
            }
        }
    }
    huge_k_on_gap_series_reserves_no_more_than_the_input_can_fill();
    k_at_the_windows_and_beyond_costs_what_the_neighbors_kept_cost();
}

/// A huge `k` sizes nothing by itself: on a series with no or few finite windows the peak stays a
/// small multiple of the output, whatever the window count would square to.
fn huge_k_on_gap_series_reserves_no_more_than_the_input_can_fill() {
    let m = NonZeroUsize::new(8).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let huge = NonZeroUsize::new(usize::MAX).unwrap();
    let threads = NonZeroUsize::MIN;
    let windows = 5_000;
    let n = windows + 7;
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
        let two_sided = peak_of(|| {
            let profile = two_sided_topk_profile(series, m, scale, threads, huge).unwrap();
            assert_eq!(profile.len(), windows);
        });
        let join = peak_of(|| {
            let profile = ab_join_topk(series, series, m, scale, scale, threads, huge).unwrap();
            assert_eq!(profile.len(), windows);
        });
        for (which, peak) in [("two-sided", two_sided), ("join", join)] {
            assert!(
                peak < windows * 1024,
                "{which} {name}: peak {peak} bytes for {windows} windows"
            );
        }
    }
}

/// On a dense series every window keeps up to `min(k, windows)` neighbors on each side, so the
/// kept neighbors number about `windows^2 / 2` per side. The two-sided peak is a fixed number of
/// bytes per kept neighbor plus one tile's partials, a fixed number of bytes per pair of the
/// tile's band rows and lanes (no more lanes than windows; each row and each column keeps every
/// pair it is offered). The default run's peak is that of the tile shape its cut picks, and a `k`
/// past the windows changes nothing. The windows span more than one default chunk of lanes.
fn k_at_the_windows_and_beyond_costs_what_the_neighbors_kept_cost() {
    const PER_KEPT: usize = 200;
    const PER_TILE_PAIR: usize = 176;
    let m = NonZeroUsize::new(8).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let threads = NonZeroUsize::MIN;
    let chunk = ladde::batch::default_chunk_lanes();
    let windows = 750;
    assert!(windows > chunk);
    let n = windows + 7;
    let kept = windows * (windows - 1) / 2;
    let series: Vec<f64> = (0..n).map(|i| ((i * i / 3 + i) % 17) as f64).collect();
    let all = NonZeroUsize::new(windows).unwrap();
    let mut shapes = Vec::new();
    for band in (3..10).map(|i| 1usize << i) {
        for tile in [chunk, 2 * chunk] {
            let peak = peak_of(|| {
                let (profile, _) = two_sided_topk_profile_tiled(
                    &series,
                    m,
                    scale,
                    threads,
                    NonZeroUsize::new(band).unwrap(),
                    NonZeroUsize::new(chunk).unwrap(),
                    NonZeroUsize::new(tile).unwrap(),
                    all,
                )
                .unwrap();
                assert_eq!(profile.len(), windows);
            });
            assert!(
                peak <= PER_KEPT * kept + PER_TILE_PAIR * band * tile.min(windows),
                "two-sided, bands of {band} by {tile}: peak {peak} bytes for {kept} kept \
                 neighbors per side"
            );
            shapes.push(peak);
        }
    }
    for (name, run) in [
        (
            "two-sided",
            &(|k: usize| {
                peak_of(|| {
                    let k = NonZeroUsize::new(k).unwrap();
                    let profile = two_sided_topk_profile(&series, m, scale, threads, k).unwrap();
                    assert_eq!(profile.len(), windows);
                })
            }) as &dyn Fn(usize) -> usize,
        ),
        (
            "join",
            &(|k: usize| {
                peak_of(|| {
                    let k = NonZeroUsize::new(k).unwrap();
                    let profile =
                        ab_join_topk(&series, &series, m, scale, scale, threads, k).unwrap();
                    assert_eq!(profile.len(), windows);
                })
            }),
        ),
    ] {
        let at_windows = run(windows);
        let beyond = run(usize::MAX);
        if name == "join" {
            assert!(
                at_windows <= 96 * windows * windows,
                "{name}: peak {at_windows} bytes for {windows} windows"
            );
        } else {
            assert!(
                shapes.contains(&at_windows),
                "{name}: peak {at_windows} bytes is no tile shape's, of {shapes:?}"
            );
        }
        assert_eq!(
            at_windows, beyond,
            "{name}: k past the windows changes the peak"
        );
    }
}
