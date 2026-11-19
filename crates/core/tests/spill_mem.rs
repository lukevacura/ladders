#![cfg(not(any(loom, shuttle)))]

//! The cold route's memory is linear in the series and its allocation count does not grow with
//! it, also where every candidate ties exactly (constant, repeated and few-level series). Its own
//! test binary, with a process-wide allocation tracker, and a single test so nothing else
//! allocates meanwhile.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use ladde::entry::Entry;
use ladde::spill::Survey;
use peak_alloc::{large_allocations_of, peak_of};
use std::num::NonZeroUsize;

type Make = fn(usize) -> Vec<i64>;

const M: usize = 8;

fn with_spikes(mut ticks: Vec<i64>) -> Vec<i64> {
    let n = ticks.len();
    ticks[n / 3] += 700_000_000_000;
    ticks[2 * n / 3] -= 900_000_000_000;
    ticks
}

/// What `Survey::new` and `merge_left` need beyond the caller's own series and results.
fn run(ticks: &[i64], spilled: Option<Vec<bool>>) {
    let had = vec![true; ticks.len()];
    let m = NonZeroUsize::new(M).unwrap();
    let windows = ticks.len() - M + 1;
    let mut entries: Vec<(u64, Entry)> = (0..windows as u64).map(|r| (r, Entry::NONE)).collect();
    let survey = match spilled {
        Some(spilled) => Survey::with_spilled(ticks, &had, m, spilled),
        None => Survey::new(ticks, &had, m),
    }
    .unwrap()
    .unwrap();
    survey.merge_left(&mut entries).unwrap();
    assert_eq!(entries.len(), windows);
}

#[test]
fn memory_is_linear_and_allocations_do_not_grow_with_the_series() {
    let series: [(&str, Make); 3] = [
        ("constant", |n| with_spikes(vec![3; n])),
        ("period four", |n| {
            with_spikes((0..n).map(|i| [0, 2, 1, 3][i % 4]).collect())
        }),
        ("few levels", |n| {
            with_spikes((0..n).map(|i| [5, 5, 9, 9, 9, 1][(i / 7) % 6]).collect())
        }),
    ];
    for (name, make) in series {
        let measure = |n: usize| {
            let ticks = make(n);
            let mut allocations = 0;
            let peak = peak_of(|| {
                allocations = large_allocations_of(1, || run(&ticks, None));
            });
            (peak, allocations)
        };
        let ((small, small_count), (large, large_count)) = (measure(3_000), measure(12_000));
        assert!(
            large <= small * 9 / 2,
            "{name}: peak {large} bytes at 12000 vs {small} at 3000"
        );
        assert!(large < 12_000 * 512, "{name}: peak {large} bytes");
        assert_eq!(
            small_count, large_count,
            "{name}: allocation count grew with the series"
        );
    }
    // Every window cold, so every pair is walked and ties pile up at every row.
    for (name, make) in [
        ("constant", (|n| vec![3; n]) as Make),
        ("period four", |n| {
            (0..n).map(|i| [0, 2, 1, 3][i % 4]).collect()
        }),
    ] {
        let measure = |n: usize| {
            let ticks = make(n);
            let spilled = vec![true; n - M + 1];
            let mut allocations = 0;
            let peak = peak_of(|| {
                allocations = large_allocations_of(1, || run(&ticks, Some(spilled)));
            });
            (peak, allocations)
        };
        let ((small, small_count), (large, large_count)) = (measure(500), measure(2_000));
        assert!(
            large <= small * 9 / 2,
            "{name}, all cold: peak {large} bytes at 2000 vs {small} at 500"
        );
        assert_eq!(
            small_count, large_count,
            "{name}, all cold: allocations grew"
        );
    }
}
