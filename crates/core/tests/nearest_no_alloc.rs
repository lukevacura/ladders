//! `CountingAllocator` only counts once it is the process's global allocator, so this runs as
//! its own test binary rather than alongside `nearest`'s unit tests (see `stats_no_alloc.rs`).
//!
//! `Nearest::with_k` reserves its `k`-candidate storage once, up front; `offer` must never
//! allocate afterward, for `k = 1` (the `new`-equivalent path) and for `k > 1` (the general
//! top-k path) alike.

use core::num::NonZeroUsize;
use ladde::index::SampleIndex;
use ladde::nearest::Nearest;
use ladde::stats::{Window, WindowStats};
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// A window built only through the public `WindowStats` API.
fn window_of(ticks: &[i64]) -> Window {
    let m = NonZeroUsize::new(ticks.len()).expect("m > 0");
    let mut stats = WindowStats::new(m);
    let mut window = None;
    for &tick in ticks {
        window = stats.push(tick, true);
    }
    window.expect("pushing exactly m ticks completes the window on the last push")
}

/// Offers far more candidates than `k` to a single `with_k(.., k)` reducer, well past its
/// construction, and asserts none of that allocates.
fn offer_many_never_allocates(k: usize) {
    let row = window_of(&[1, 2, 9, 4, 7]);
    let neighbors: Vec<Window> = (0..16i64)
        .map(|i| window_of(&[i, i + 3, i + 1, i + 9, i + 2]))
        .collect();

    let mut nearest = Nearest::with_k(&row, 5, NonZeroUsize::new(k).expect("k > 0"), k);

    let (_, count) = allocations_in(|| {
        for round in 0..1_000u64 {
            for (j, neighbor) in neighbors.iter().enumerate() {
                let n = (round as i64) + j as i64;
                let sample = round * neighbors.len() as u64 + j as u64;
                let index = SampleIndex::new(sample).expect("sample is below NONE's sentinel");
                nearest.offer(n, neighbor, index);
            }
        }
    });

    assert_eq!(count, 0, "k = {k}");
}

#[test]
fn offer_never_allocates_after_with_k_one() {
    offer_many_never_allocates(1);
}

#[test]
fn offer_never_allocates_after_with_k_eight() {
    offer_many_never_allocates(8);
}

/// A reducer restarted with a different `k` keeps that `k`, from one and to one included.
#[test]
fn restart_takes_the_new_k() {
    let row = window_of(&[1, 2, 9, 4, 7]);
    let neighbors: Vec<Window> = (0..16i64)
        .map(|i| window_of(&[i, i + 3, i + 1, i + 9, i + 2]))
        .collect();
    let mut reused = Nearest::with_k(&row, 5, NonZeroUsize::new(8).unwrap(), neighbors.len());
    let mut out = Vec::new();
    for k in [3usize, 1, 8, 2, 8, 1] {
        reused.restart(&row, 5, NonZeroUsize::new(k).unwrap(), neighbors.len());
        for (j, neighbor) in neighbors.iter().enumerate() {
            let index = SampleIndex::new(j as u64).expect("below NONE");
            reused.offer((j as i64 * 5) % 13 - 4, neighbor, index);
        }
        reused.drain_k(&mut out);
        assert_eq!(out.len(), k, "k {k}");
    }
}

/// One reducer restarted for every row reserves nothing once its first row has, and each row
/// drains to what a fresh reducer over the same offers returns, on bits.
#[test]
fn restart_and_drain_reuse_the_reserved_storage_and_equal_a_fresh_reducer() {
    let rows: Vec<Window> = (0..6i64)
        .map(|r| window_of(&[r, 2, 9 - r, 4, 7 + r]))
        .collect();
    let neighbors: Vec<Window> = (0..16i64)
        .map(|i| window_of(&[i, i + 3, i + 1, i + 9, i + 2]))
        .collect();
    let bits = |e: &ladde::entry::Entry| {
        (
            e.rho().to_bits(),
            e.squared_distance().to_bits(),
            e.index().get(),
        )
    };
    for k in [1usize, 3, 8, 40] {
        let k_nz = NonZeroUsize::new(k).expect("k > 0");
        let mut reused = Nearest::with_k(&rows[0], 5, k_nz, neighbors.len());
        let mut out = Vec::with_capacity(k.min(neighbors.len()));
        for (r, row) in rows.iter().enumerate() {
            let offer = |nearest: &mut Nearest| {
                for (j, neighbor) in neighbors.iter().enumerate() {
                    let index = SampleIndex::new(j as u64).expect("below NONE");
                    nearest.offer((r as i64 * 3 + j as i64) % 11 - 3, neighbor, index);
                }
            };
            let (_, count) = allocations_in(|| {
                reused.restart(row, 5, k_nz, neighbors.len());
                offer(&mut reused);
                reused.drain_k(&mut out)
            });
            assert_eq!(count, 0, "k {k} row {r}");

            let mut fresh = Nearest::with_k(row, 5, k_nz, neighbors.len());
            offer(&mut fresh);
            let want: Vec<_> = fresh.finish_k().iter().map(bits).collect();
            let got: Vec<_> = out.iter().map(bits).collect();
            assert_eq!(got, want, "k {k} row {r}");
        }
    }
}
