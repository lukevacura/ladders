//! Count-sized buffers reserve their true bound in full and nothing grows after warm-up (its
//! own test binary: the counting allocator is global, and counts this thread only).

use core::mem::size_of;
use core::num::NonZeroUsize;
use ladde::batch::{discords, left_topk_profile};
use ladde::discord::{Discord, DiscordTracker};
use ladde::entry::Entry;
use ladde::floss::Floss;
use ladde::index::SampleIndex;
use ladde::nearest::Nearest;
use ladde::quantize::Scale;
use ladde::row::RowAcc;
use ladde::stats::{Window, WindowStats};
use ladde::stream::GrowingStream;
use testkit::alloc_count::{CountingAllocator, LargeRequest, allocations_in, large_requests_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const BIG: usize = 100_000;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

fn fresh_blocks(requests: &[LargeRequest], bytes: usize) -> usize {
    requests
        .iter()
        .filter(|r| !r.realloc && r.bytes == bytes)
        .count()
}

fn any_of_size(requests: &[LargeRequest], bytes: usize) -> bool {
    requests.iter().any(|r| r.bytes == bytes)
}

fn floss_fill(width: usize, horizon: usize) -> Vec<LargeRequest> {
    let row = |r: usize| SampleIndex::new(r as u64).unwrap();
    let (_, requests) = large_requests_in(BIG, || {
        let mut floss = Floss::new(nz(4), nz(width), horizon, 3, 2);
        for r in 0..width + 10 {
            floss.update(row(r), Entry::NONE);
        }
    });
    requests
}

#[test]
fn floss_reserves_its_width_once_and_never_grows_it() {
    let width = 70_000;
    let bytes = width * size_of::<SampleIndex>();
    for horizon in [100, width - 1, 3 * width] {
        let requests = floss_fill(width, horizon);
        assert_eq!(fresh_blocks(&requests, bytes), 1, "horizon {horizon}");
        assert!(!any_of_size(&requests, 2 * bytes), "horizon {horizon}");
    }
}

#[test]
fn a_wide_bounded_floss_allocates_nothing_after_warm_up() {
    let width = 70_000;
    let mut floss = Floss::new(nz(4), nz(width), width - 1, 3, 2);
    let row = |r: usize| SampleIndex::new(r as u64).unwrap();
    for r in 0..width + 10 {
        floss.update(row(r), Entry::NONE);
    }
    let ((), count) = allocations_in(|| {
        for r in width + 10..2 * width {
            floss.update(row(r), Entry::NONE);
        }
    });
    assert_eq!(count, 0);
}

#[test]
fn an_unbounded_floss_allocates_nothing_after_warm_up_even_across_row_jumps() {
    let width = 1000;
    let mut floss = Floss::new(nz(4), nz(width), usize::MAX, 3, 2);
    let row = |r: u64| SampleIndex::new(r).unwrap();
    for r in 0..width as u64 + 10 {
        floss.update(row(r), Entry::NONE);
    }
    let ((), count) = allocations_in(|| {
        for start in [1u64 << 20, 1 << 50, (1 << 62) + 5] {
            for r in start..start + 3 * width as u64 {
                floss.update(row(r), Entry::NONE);
            }
        }
    });
    assert_eq!(count, 0);
    assert_eq!(floss.window_start(), Some((1 << 62) + 5 + 2 * width as u64));
    assert_eq!(floss.curve().len(), width);
}

#[test]
fn a_windowed_tracker_reserves_min_k_and_windows_once_and_never_grows_it() {
    let m = 6;
    let samples: Vec<f64> = (0..3000).map(|i| ((i * 37) % 101) as f64).collect();
    for (k, w) in [(200_000, m + 150_000), (150_000, m + 400_000)] {
        let windows = w - m + 1;
        let bound = k.min(windows);
        let mut stream = GrowingStream::new(nz(m), scale());
        let (_, requests) = large_requests_in(BIG, || {
            let mut tracker = DiscordTracker::windowed(nz(m), k, nz(w)).unwrap();
            for &x in &samples {
                if let Some((row, score)) = stream.update_exact(x) {
                    tracker.update(row, score);
                }
            }
        });
        let (list, picked) = (bound * size_of::<Discord>(), bound * size_of::<u64>());
        assert_eq!(fresh_blocks(&requests, list), 1, "k {k} w {w}");
        assert_eq!(fresh_blocks(&requests, picked), 1, "k {k} w {w}");
        assert!(
            requests.iter().all(|r| !r.realloc),
            "k {k} w {w} {requests:?}"
        );
    }
}

fn largest(requests: &[LargeRequest]) -> usize {
    requests.iter().map(|r| r.bytes).max().unwrap_or(0)
}

#[test]
fn a_huge_k_reserves_no_more_than_k_at_the_window_count() {
    let samples: Vec<f64> = (0..90).map(|i| ((i * 37) % 11) as f64).collect();
    let windows = samples.len() - 6 + 1;
    let topk = |k: usize| {
        large_requests_in(1024, || {
            left_topk_profile(&samples, nz(6), scale(), nz(1), nz(k))
        })
        .1
    };
    let found =
        |k: usize| large_requests_in(1024, || discords(&samples, nz(6), scale(), nz(1), k)).1;
    assert_eq!(largest(&topk(1_000_000_000)), largest(&topk(windows)));
    assert_eq!(largest(&found(1_000_000_000)), largest(&found(windows)));
}

fn some_window() -> Window {
    let mut stats = WindowStats::new(nz(3));
    [1, 2, 9]
        .into_iter()
        .filter_map(|tick| stats.push(tick, true))
        .last()
        .unwrap()
}

#[test]
fn a_batch_row_reserves_k_in_full_when_its_windows_allow() {
    let (k, window) = (70_000, some_window());
    let begun = |bound: usize| {
        let mut acc = RowAcc::new();
        large_requests_in(BIG, || acc.begin(&window, nz(k), bound)).1
    };
    let requests = begun(k);
    assert_eq!(fresh_blocks(&requests, k * size_of::<f64>()), 1);
    let beyond = begun(10 * k);
    assert_eq!(beyond, requests);
    let capped = begun(1000);
    assert!(capped.is_empty());
}

#[test]
fn a_batch_nearest_reserves_k_in_full_when_its_windows_allow() {
    let (k, window) = (70_000, some_window());
    let reserved =
        |bound: usize| large_requests_in(BIG, || Nearest::with_k(&window, 3, nz(k), bound)).1;
    let full = reserved(k);
    assert_eq!(full.len(), 1);
    assert_eq!(reserved(10 * k), full);
    let flat = reserved(65_536);
    assert_eq!(flat.len(), 1);
    assert_eq!(full[0].bytes * 65_536, flat[0].bytes * k);
}

/// Mostly gaps, so the output stays small while the series has more windows than the fixed
/// reservation limit; too slow unoptimized.
#[cfg(not(debug_assertions))]
#[test]
fn batch_top_k_reserves_k_in_full_when_the_series_has_that_many_windows() {
    let (m, k) = (8, 70_000);
    let mut series = vec![f64::NAN; k + 100];
    for (i, x) in series[k + 20..].iter_mut().enumerate() {
        *x = ((i * 37) % 11) as f64;
    }
    let (_, requests) = large_requests_in(BIG, || {
        left_topk_profile(&series, nz(m), scale(), nz(1), nz(k))
    });
    assert!(
        fresh_blocks(&requests, k * size_of::<f64>()) >= 1,
        "{requests:?}"
    );
}
