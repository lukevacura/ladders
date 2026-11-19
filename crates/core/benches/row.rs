//! `row` bench: ns per lag for [`advance_and_rank`]'s fused pass, `m = 50`, at `n ∈ {2 000,
//! 19 000, 100 000}`, on a random walk of integer ticks.
//!
//! Hand-timed with `std::time::Instant` rather than the `criterion` crate: `criterion`
//! unconditionally pulls in `serde_json`, whose blanket `PartialEq<Value>` impls for the
//! integer types make every unannotated empty-slice comparison in this crate's own tests
//! (for example `crates/core/src/adapter.rs`'s `assert_eq!(reader.snapshot(), &[])`)
//! ambiguous at the type-checker — a conflict in a file outside this node's scope, so this
//! bench stays on `std` alone. The timed region is one call at the series' last row (its full
//! eligible count), repeated after a warm-up; setup (building the walk and its
//! struct-of-arrays columns) runs once, outside every timed call.

use ladde::kernel::Metric;
use ladde::row::{Columns, RowInputs, advance_and_rank, push_window, resize_and_shift};
use ladde::stats::{Window, WindowStats};
use ladde::znorm::{TickWindow, ZNorm};
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

const M: usize = 50;
const WARMUP_ITERS: u32 = 1_000;
const TIMED_ITERS: u32 = 20_000;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A random walk in integer ticks, deterministic in `seed`, with steps uniform in `-2..=2`.
fn random_walk(len: usize, seed: u64) -> Vec<i64> {
    let mut state = seed;
    let mut value: i64 = 1_000_000_000;
    let mut walk = Vec::with_capacity(len);
    for _ in 0..len {
        let step = (splitmix64(&mut state) % 5) as i64 - 2;
        value = value.wrapping_add(step);
        walk.push(value);
    }
    walk
}

fn windows_of(ticks: &[i64], m: usize) -> Vec<Window> {
    let mut stats = WindowStats::new(NonZeroUsize::new(m).unwrap());
    ticks
        .iter()
        .filter_map(|&tick| stats.push(tick, true))
        .collect()
}

struct Setup {
    ticks: Vec<i64>,
    sum: Vec<i64>,
    r: Vec<f64>,
    bias: Vec<f64>,
    variance: Vec<u64>,
    windows: Vec<Window>,
}

fn build(n: usize) -> Setup {
    let ticks = random_walk(n, 0xC0FFEE_u64);
    let windows = windows_of(&ticks, M);
    let mut sum = Vec::with_capacity(windows.len());
    let mut r = Vec::with_capacity(windows.len());
    let mut bias = Vec::with_capacity(windows.len());
    let mut variance = Vec::with_capacity(windows.len());
    for &window in &windows {
        push_window(window, &mut sum, &mut r, &mut bias, &mut variance);
    }
    Setup {
        ticks,
        sum,
        r,
        bias,
        variance,
        windows,
    }
}

fn tick_window_of(setup: &Setup, j: usize, m: usize) -> TickWindow {
    TickWindow {
        entering: setup.ticks[j + m - 1],
        leaving: if j == 0 { 0 } else { setup.ticks[j - 1] },
        stats: setup.windows[j],
    }
}

/// The `qt` a real caller must hold just before calling `advance_and_rank` for `row`: built
/// unshifted (`qt[j] = pair(row − 1, j)`, directly by a fresh `ZNorm::start` per neighbor
/// rather than by replaying every earlier row — exact regardless, since wrapping arithmetic is
/// path-independent, the same argument `row.rs`'s own tests rely on, so repeated timed calls
/// can each start from this same, correct state without an O(n²) replay), then grown and
/// shifted by [`resize_and_shift`] into the one layout `advance_and_rank` now expects: `qt[j]`
/// already holding pair `(row − 1, j − 1)` for every `j ≥ 1`. Sharing that one function with
/// `row.rs`'s own tests and `batch.rs`'s own call sites, rather than re-deriving the shift
/// here, means a future contract change breaks this bench's own build, not silently drifts
/// from it.
fn qt_before_row(setup: &Setup, m: usize, exclusion: usize, row: usize) -> Vec<i64> {
    let count = row - exclusion;
    let prev_row = row - 1;
    let prev_count = prev_row - exclusion;
    let prev_tw = tick_window_of(setup, prev_row, m);
    let mut qt = vec![0i64; prev_count];
    for (j, qt_slot) in qt.iter_mut().enumerate() {
        let neighbor_tw = tick_window_of(setup, j, m);
        *qt_slot = ZNorm::start(
            &setup.ticks[prev_row..prev_row + m],
            &prev_tw,
            &setup.ticks[j..j + m],
            &neighbor_tw,
        );
    }
    resize_and_shift(&mut qt, count);
    qt
}

/// Mean nanoseconds per lag of `TIMED_ITERS` calls at `n`'s last row, after `WARMUP_ITERS`
/// unmeasured calls. Each call starts from the same correct pre-row `qt` (reset from a
/// snapshot, untimed) so every repetition both avoids the double-advance that would corrupt
/// `qt` and measures the same, real, one-row cost every time.
fn bench_one(n: usize) -> (usize, f64) {
    let exclusion = ladde::exclusion::zone(M);
    let setup = build(n);
    let row = setup.windows.len() - 1;
    let count = row - exclusion;
    let cols = Columns {
        ticks: &setup.ticks,
        sum: &setup.sum,
        r: &setup.r,
        bias: &setup.bias,
        variance: &setup.variance,
    };
    let row_inputs = RowInputs {
        row,
        window: setup.windows[row],
    };
    let qt_snapshot = qt_before_row(&setup, M, exclusion, row);
    let mut qt = qt_snapshot.clone();
    let mut scratch = vec![0.0f64; count];

    for _ in 0..WARMUP_ITERS {
        qt.copy_from_slice(&qt_snapshot);
        black_box(advance_and_rank(
            black_box(&mut qt),
            black_box(&cols),
            black_box(&row_inputs),
            M,
            exclusion,
            black_box(&mut scratch),
        ));
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..TIMED_ITERS {
        qt.copy_from_slice(&qt_snapshot);
        let start = Instant::now();
        black_box(advance_and_rank(
            black_box(&mut qt),
            black_box(&cols),
            black_box(&row_inputs),
            M,
            exclusion,
            black_box(&mut scratch),
        ));
        total += start.elapsed();
    }

    let ns_per_call = total.as_secs_f64() * 1e9 / f64::from(TIMED_ITERS);
    (count, ns_per_call / count as f64)
}

fn main() {
    println!("row: ns per lag, m = {M}");
    for &n in &[2_000usize, 19_000, 100_000] {
        let (count, ns_per_lag) = bench_one(n);
        println!("  n = {n:>7}  lags = {count:>7}  {ns_per_lag:.3} ns/lag");
    }
}
