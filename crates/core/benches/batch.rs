//! `batch` bench: ns per eligible pair of the batch pass, at `n ∈ {10 000, 100 000}` and
//! `m ∈ {50, 256}`, on a random walk at a market-level price offset (so the whole run stays on
//! the narrow NEON path). Sweeps the tiled narrow path's own chunk length — 32 KiB, 64 KiB
//! and 128 KiB of lanes — through [`left_profile_with_chunk`], alongside the plain
//! [`left_profile`] entry point (which picks its own chunk length by the same cfg-based default
//! [`ladde::batch::default_chunk_lanes`] uses).
//!
//! Hand-timed with `std::time::Instant`, matching `benches/row.rs`'s own reasoning for staying
//! off the `criterion` crate (see that file's module doc).

use ladde::batch::{chunk_lanes_for_bytes, left_profile, left_profile_with_chunk};
use ladde::quantize::Scale;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A random walk of `f64` prices, deterministic in `seed`, at a market-level offset (so the
/// quantized ticks sit far enough from `0` to exercise the narrow path's own anchor): steps
/// uniform in `-2..=2`, the same shape `benches/row.rs`'s own walk uses.
fn market_walk(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut value: i64 = 10_000_000_000;
    let mut walk = Vec::with_capacity(len);
    for _ in 0..len {
        let step = (splitmix64(&mut state) % 5) as i64 - 2;
        value = value.wrapping_add(step);
        walk.push(value as f64);
    }
    walk
}

/// The total number of eligible `(row, neighbor)` pairs the batch pass evaluates over
/// `num_windows` windows: `Σ max(0, row - exclusion)`, the same count `batch.rs`'s own
/// `eligible_count` sums to.
fn total_pairs(num_windows: usize, exclusion: usize) -> u64 {
    (0..num_windows)
        .map(|row| row.saturating_sub(exclusion) as u64)
        .sum()
}

fn bench_default(n: usize, m: usize) -> (u64, f64) {
    let exclusion = ladde::exclusion::zone(m);
    let series = market_walk(n, 0xC0FFEE_u64);
    let num_windows = n - m + 1;
    let pairs = total_pairs(num_windows, exclusion);
    let scale = Scale::new(1.0).unwrap();
    let m_nz = NonZeroUsize::new(m).unwrap();

    let start = Instant::now();
    let result = black_box(left_profile(black_box(&series), m_nz, scale));
    let elapsed = start.elapsed();
    black_box(&result);
    (pairs, elapsed.as_secs_f64() * 1e9 / pairs as f64)
}

fn bench_chunk(n: usize, m: usize, chunk_lanes: usize) -> (u64, f64) {
    let exclusion = ladde::exclusion::zone(m);
    let series = market_walk(n, 0xC0FFEE_u64);
    let num_windows = n - m + 1;
    let pairs = total_pairs(num_windows, exclusion);
    let scale = Scale::new(1.0).unwrap();
    let m_nz = NonZeroUsize::new(m).unwrap();

    let start = Instant::now();
    let result = black_box(left_profile_with_chunk(
        black_box(&series),
        m_nz,
        scale,
        chunk_lanes,
    ));
    let elapsed = start.elapsed();
    black_box(&result);
    (pairs, elapsed.as_secs_f64() * 1e9 / pairs as f64)
}

fn main() {
    for &m in &[50usize, 256] {
        println!("batch: ns per eligible pair, m = {m}");
        for &n in &[10_000usize, 100_000] {
            let (pairs, ns_per_pair) = bench_default(n, m);
            println!(
                "  left_profile (default chunk)        n = {n:>7}  pairs = {pairs:>12}  {ns_per_pair:.4} ns/pair"
            );
            for &kib in &[32usize, 64, 128] {
                let lanes = chunk_lanes_for_bytes(kib * 1024);
                let (_, ns) = bench_chunk(n, m, lanes);
                println!(
                    "  chunk {kib:>3} KiB (lanes = {lanes:>5})        n = {n:>7}  {ns:.4} ns/pair"
                );
            }
        }
    }
}
