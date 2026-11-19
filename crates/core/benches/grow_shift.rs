//! `grow_shift` cost ladder (step 0 of the `grow-shift` node): ns per lag for
//! [`advance_and_rank_narrow`]'s fused pass (rung 3) and for `qt.copy_within` alone (rung 3b),
//! `m = 50`, at `n = 100 000`, on a market-level walk (the same shape `benches/batch.rs`'s own
//! `market_walk` uses, far enough from zero to stay on the narrow path). Rungs 1 and 2 of
//! the ladder are `benches/kernel_neon.rs`'s own `max_mode`/`scratch_mode`; rung 4 is an
//! end-to-end streaming benchmark on real market data, kept outside this repository.
//!
//! Step 1b adds rung 3c (a plain compare-and-collect scan over `scratch`, no per-candidate
//! work at all) and rung 3d (`advance_and_rank_narrow`'s own Pass-2 near-tie rescan, run
//! alone, over a scratch with one realistic winner): together they split what rung 3 minus
//! rung 2 still costs after Step 1 between the scan itself and the per-candidate work around
//! it (window reconstruction, `narrow::cross`, `Nearest::offer`).
//!
//! Hand-timed with `std::time::Instant`, matching `benches/row.rs`'s own reasoning for staying
//! off the `criterion` crate (see that file's module doc).

use ladde::columns;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::kernel::Metric;
use ladde::kernel_neon::Conversion;
use ladde::narrow::{self, Anchor};
use ladde::nearest::Nearest;
use ladde::row::{
    NarrowColumns, RowInputs, advance_and_rank_narrow, push_window, resize_and_shift,
};
use ladde::stats::{Window, WindowStats};
use ladde::znorm::{TickWindow, ZNorm};
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

const M: usize = 50;
/// [`ladde::nearest`]'s own near-tie band is `pub(crate)`, unreachable from a bench
/// crate; this is its exact definition (`row`'s own module doc derives the same band from it),
/// duplicated here the same way `narrow_columns` below re-expresses the narrow formula directly
/// instead of reaching for its `pub(crate)` helpers.
const EPSILON: f64 = 1.0 / (1u64 << 49) as f64;
const WARMUP_ITERS: u32 = 1_000;
const TIMED_ITERS: u32 = 20_000;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A market-level walk of ticks, deterministic in `seed`: the same shape `benches/batch.rs`'s
/// own `market_walk` uses (steps uniform in `-2..=2`, offset far from zero), duplicated here
/// directly in ticks rather than `f64` prices since this bench drives [`row`] functions, never
/// a stream's own quantizer.
fn market_walk(len: usize, seed: u64) -> Vec<i64> {
    let mut state = seed;
    let mut value: i64 = 10_000_000_000;
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
    let ticks = market_walk(n, 0xC0FFEE_u64);
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

/// Every tick and window sum, narrowed to one anchor centered on the whole walk's own tiny
/// span (mirrors `batch.rs`'s own whole-series anchor choice): a bench is outside the crate, so
/// it has no access to `row`'s own `pub(crate)` narrowing helpers and re-expresses the narrow
/// plain formula (`tick − anchor`, `sum − m·anchor`) directly instead.
fn narrow_columns(setup: &Setup, m: usize) -> (Vec<i32>, Vec<i32>) {
    let m_nz = NonZeroUsize::new(m).unwrap();
    let lo = *setup.ticks.iter().min().expect("walk is non-empty");
    let hi = *setup.ticks.iter().max().expect("walk is non-empty");
    let anchor = Anchor {
        tick: lo + (hi - lo) / 2,
    };
    let ticks_off: Vec<i32> = setup
        .ticks
        .iter()
        .map(|&tick| {
            anchor
                .offset(tick, m_nz)
                .expect("a market-level walk's own tiny span fits one anchor")
        })
        .collect();
    let m_i64 = m as i64;
    let sum_off: Vec<i32> = setup
        .sum
        .iter()
        .map(|&sum| {
            i32::try_from(sum - m_i64 * anchor.tick)
                .expect("a window's own sum offset fits i32 whenever each of its ticks does")
        })
        .collect();
    (ticks_off, sum_off)
}

/// The `m·QT` a real caller must hold just before calling `advance_and_rank_narrow` for
/// `row`: built unshifted (`qt[j] = m·QT(row − 1, j)`, directly by a fresh `ZNorm::start` per
/// neighbor — exact regardless of path, since wrapping arithmetic is path-independent, the
/// same argument `benches/row.rs`'s own `qt_before_row` relies on — scaled by `m` the same way
/// a fresh diagonal's own start is scaled inside `advance_and_rank_narrow`), then grown and
/// shifted by [`resize_and_shift`] into the one layout that function now expects: `qt[j]`
/// already holding pair `(row − 1, j − 1)` for every `j ≥ 1`. Sharing that one function with
/// `row`'s own tests and `batch.rs`'s own call sites, rather than re-deriving the shift
/// here, means a future contract change breaks this bench's own build, not silently drifts
/// from it.
fn qt_m_before_row(setup: &Setup, m: usize, exclusion: usize, row: usize) -> Vec<i64> {
    let count = row - exclusion;
    let prev_row = row - 1;
    let prev_count = prev_row - exclusion;
    let prev_tw = tick_window_of(setup, prev_row, m);
    let m_i64 = m as i64;
    let mut qt = vec![0i64; prev_count];
    for (j, qt_slot) in qt.iter_mut().enumerate() {
        let neighbor_tw = tick_window_of(setup, j, m);
        let qt_abs = ZNorm::start(
            &setup.ticks[prev_row..prev_row + m],
            &prev_tw,
            &setup.ticks[j..j + m],
            &neighbor_tw,
        );
        *qt_slot = qt_abs.wrapping_mul(m_i64);
    }
    resize_and_shift(&mut qt, count);
    qt
}

/// The `m·QT` Pass 2 itself reads: `qt[j] = m·QT(row, j)` directly, for every live neighbor `j`
/// (`0..count`) — the *post*-advance state Pass 1 would have already left behind, unlike
/// [`qt_m_before_row`]'s pre-row snapshot (built for feeding the full function, which still has
/// its own Pass 1 to run). Rung 3d skips Pass 1 entirely, so it needs this instead: feeding it
/// [`qt_m_before_row`]'s pre-row values would pair each `qt[j]` with the wrong row and the wrong
/// neighbor's own sum inside `narrow::cross`, risking exactly the kind of inconsistent state
/// that trips `Nearest`'s own Cauchy–Schwarz assertion. Built directly by a fresh `ZNorm::start`
/// per neighbor, the same way `qt_m_before_row` is.
fn qt_m_for_row(setup: &Setup, m: usize, exclusion: usize, row: usize) -> Vec<i64> {
    let count = row - exclusion;
    let mut qt = vec![0i64; count];
    let row_tw = tick_window_of(setup, row, m);
    let m_i64 = m as i64;
    for (j, qt_slot) in qt.iter_mut().enumerate().take(count) {
        let neighbor_tw = tick_window_of(setup, j, m);
        let qt_abs = ZNorm::start(
            &setup.ticks[row..row + m],
            &row_tw,
            &setup.ticks[j..j + m],
            &neighbor_tw,
        );
        *qt_slot = qt_abs.wrapping_mul(m_i64);
    }
    qt
}

/// Rung 3: mean nanoseconds per lag of `TIMED_ITERS` calls to `advance_and_rank_narrow` at
/// `n`'s last row, after `WARMUP_ITERS` unmeasured calls — `benches/row.rs`'s own `bench_one`,
/// on the narrow path instead of the wide one.
fn bench_advance_and_rank_narrow(n: usize) -> (usize, f64) {
    let exclusion = ladde::exclusion::zone(M);
    let setup = build(n);
    let row = setup.windows.len() - 1;
    let count = row - exclusion;
    let (ticks_off, sum_off) = narrow_columns(&setup, M);
    let cols = NarrowColumns {
        ticks: &ticks_off,
        sum: &sum_off,
        r: &setup.r,
        bias: &setup.bias,
        variance: &setup.variance,
        conversion: Conversion::FULL_RANGE,
    };
    let row_inputs = RowInputs {
        row,
        window: setup.windows[row],
    };
    let row_sum_offset = sum_off[row];
    let qt_snapshot = qt_m_before_row(&setup, M, exclusion, row);
    let mut qt = qt_snapshot.clone();
    let mut scratch = vec![0.0f64; count];

    for _ in 0..WARMUP_ITERS {
        qt.copy_from_slice(&qt_snapshot);
        black_box(advance_and_rank_narrow(
            black_box(&mut qt),
            black_box(&cols),
            black_box(&row_inputs),
            row_sum_offset,
            M,
            exclusion,
            black_box(&mut scratch),
        ));
    }

    let mut total = Duration::ZERO;
    for _ in 0..TIMED_ITERS {
        qt.copy_from_slice(&qt_snapshot);
        let start = Instant::now();
        black_box(advance_and_rank_narrow(
            black_box(&mut qt),
            black_box(&cols),
            black_box(&row_inputs),
            row_sum_offset,
            M,
            exclusion,
            black_box(&mut scratch),
        ));
        total += start.elapsed();
    }

    let ns_per_call = total.as_secs_f64() * 1e9 / f64::from(TIMED_ITERS);
    (count, ns_per_call / count as f64)
}

/// Rung 3b: `qt.copy_within` alone, at the same lag count rung 3 measures — isolating the
/// shift's own cost from the fresh diagonal and candidate rescan around it.
fn bench_copy_within(len: usize) -> f64 {
    let mut qt: Vec<i64> = (0..len as i64).collect();

    for _ in 0..WARMUP_ITERS {
        black_box(&mut qt).copy_within(0..len - 1, 1);
    }

    let mut total = Duration::ZERO;
    for _ in 0..TIMED_ITERS {
        let start = Instant::now();
        black_box(&mut qt).copy_within(0..len - 1, 1);
        total += start.elapsed();
    }
    total.as_secs_f64() * 1e9 / (f64::from(TIMED_ITERS) * len as f64)
}

/// A realistic Pass-2 scratch: one winner (`1.0`, a near-perfect correlation) at a random
/// position, every other lane spread across `[-1.0, 0.5)` — comfortably below the near-tie
/// threshold (`1.0 − 2·EPSILON`) by many orders of magnitude more than `2·EPSILON` itself, so
/// exactly one candidate ever clears it: the common case `advance_and_rank_narrow`'s own doc
/// comment calls out ("usually exactly one").
fn realistic_scratch(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut scratch: Vec<f64> = (0..len)
        .map(|_| {
            let bits = splitmix64(&mut state);
            -1.0 + 1.5 * ((bits >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect();
    let winner = (splitmix64(&mut state) as usize) % len;
    scratch[winner] = 1.0;
    scratch
}

/// The hot body [`bench_scan_alone`] times: a plain compare-and-collect scan. `#[inline(never)]`
/// and `black_box`ed arguments at every call site (never inside this body) so the loop itself
/// is free to vectorize however the backend sees fit — the same boundary every other bench in
/// this crate draws around the real function it calls, never inside that function's own loop.
#[inline(never)]
fn scan_pass(scratch: &[f64], threshold: f64, hits: &mut Vec<usize>) {
    hits.clear();
    for (j, &rho_hat) in scratch.iter().enumerate() {
        if rho_hat >= threshold {
            hits.push(j);
        }
    }
}

/// Rung 3c: a plain compare-and-collect scan over `scratch` alone — no window
/// reconstruction, no `narrow::cross`, no `Nearest` — the baseline the real Pass-2 rescan's
/// own scan cost is measured against.
fn bench_scan_alone(scratch: &[f64], threshold: f64) -> f64 {
    let len = scratch.len();
    let mut hits: Vec<usize> = Vec::with_capacity(4);

    for _ in 0..WARMUP_ITERS {
        scan_pass(black_box(scratch), black_box(threshold), &mut hits);
        black_box(&hits);
    }

    let mut total = Duration::ZERO;
    for _ in 0..TIMED_ITERS {
        let start = Instant::now();
        scan_pass(black_box(scratch), black_box(threshold), &mut hits);
        black_box(&hits);
        total += start.elapsed();
    }
    total.as_secs_f64() * 1e9 / (f64::from(TIMED_ITERS) * len as f64)
}

/// Lanes per chunk-skip group: `row`'s own `RESCAN_CHUNK` is private, so this is its exact
/// value, duplicated the same way this file already duplicates `EPSILON` and the narrowing
/// formula.
const RESCAN_CHUNK: usize = 64;

/// The hot body [`bench_rescan_alone`] times: exactly `row`'s own `rescan_near_tie`, plus
/// the one winner's real per-candidate work. `#[inline(never)]`, same reasoning as
/// [`scan_pass`]: every argument is `black_box`ed at the call site, never inside this loop.
#[inline(never)]
fn rescan_pass(
    row_window: &Window,
    m: usize,
    cols: &NarrowColumns,
    qt: &[i64],
    row_sum_offset: i32,
    scratch: &[f64],
    threshold: f64,
) -> Entry {
    let mut nearest = Nearest::new(row_window, m);
    for (chunk_index, chunk) in scratch.chunks(RESCAN_CHUNK).enumerate() {
        let any_hit = chunk
            .iter()
            .fold(false, |hit, &rho_hat| hit | (rho_hat >= threshold));
        if any_hit {
            let base = chunk_index * RESCAN_CHUNK;
            for (offset, &rho_hat) in chunk.iter().enumerate() {
                if rho_hat >= threshold {
                    let j = base + offset;
                    let neighbor = columns::window_of(0, cols.r[j], cols.bias[j], cols.variance[j]);
                    let n = narrow::cross(qt[j], row_sum_offset, cols.sum[j]);
                    let index = SampleIndex::new(j as u64).expect("j is below NONE's sentinel");
                    nearest.offer(n, &neighbor, index);
                }
            }
        }
    }
    nearest.finish()
}

/// Rung 3d: `advance_and_rank_narrow`'s own Pass 2 (the near-tie rescan), run alone — exactly
/// [`bench_scan_alone`]'s scan, plus the one winner's real per-candidate work (window
/// reconstruction, [`narrow::cross`], [`Nearest::offer`]) that function itself does.
fn bench_rescan_alone(
    row_window: Window,
    m: usize,
    cols: &NarrowColumns,
    qt: &[i64],
    row_sum_offset: i32,
    scratch: &[f64],
    threshold: f64,
) -> f64 {
    let len = scratch.len();

    for _ in 0..WARMUP_ITERS {
        black_box(rescan_pass(
            black_box(&row_window),
            m,
            black_box(cols),
            black_box(qt),
            row_sum_offset,
            black_box(scratch),
            black_box(threshold),
        ));
    }

    let mut total = Duration::ZERO;
    for _ in 0..TIMED_ITERS {
        let start = Instant::now();
        let entry = rescan_pass(
            black_box(&row_window),
            m,
            black_box(cols),
            black_box(qt),
            row_sum_offset,
            black_box(scratch),
            black_box(threshold),
        );
        total += start.elapsed();
        black_box(entry);
    }
    total.as_secs_f64() * 1e9 / (f64::from(TIMED_ITERS) * len as f64)
}

fn main() {
    println!("grow_shift ladder: ns per lag, m = {M}");
    let n = 100_000usize;
    let (count, narrow_ns_per_lag) = bench_advance_and_rank_narrow(n);
    println!(
        "  rung 3  advance_and_rank_narrow  n = {n:>7}  lags = {count:>7}  {narrow_ns_per_lag:.3} ns/lag"
    );
    let copy_ns_per_lag = bench_copy_within(count);
    println!(
        "  rung 3b copy_within alone        n = {n:>7}  lags = {count:>7}  {copy_ns_per_lag:.3} ns/lag"
    );

    // Step 1b: split what's left of rung 3 after the shift is gone, between the plain scan
    // and the real Pass-2 rescan around it (the same `count`, built fresh here since it needs
    // its own narrowed columns and qt, not just a length).
    let exclusion = ladde::exclusion::zone(M);
    let setup = build(n);
    let row = setup.windows.len() - 1;
    let (ticks_off, sum_off) = narrow_columns(&setup, M);
    let cols = NarrowColumns {
        ticks: &ticks_off,
        sum: &sum_off,
        r: &setup.r,
        bias: &setup.bias,
        variance: &setup.variance,
        conversion: Conversion::FULL_RANGE,
    };
    let qt = qt_m_for_row(&setup, M, exclusion, row);
    let row_sum_offset = sum_off[row];
    let scratch = realistic_scratch(count, 0x5CA1AB1E);
    let threshold = 1.0 - 2.0 * EPSILON;

    let scan_ns_per_lag = bench_scan_alone(&scratch, threshold);
    println!(
        "  rung 3c plain compare-and-collect n = {n:>7}  lags = {count:>7}  {scan_ns_per_lag:.3} ns/lag"
    );
    let rescan_ns_per_lag = bench_rescan_alone(
        setup.windows[row],
        M,
        &cols,
        &qt,
        row_sum_offset,
        &scratch,
        threshold,
    );
    println!(
        "  rung 3d Pass-2 rescan alone       n = {n:>7}  lags = {count:>7}  {rescan_ns_per_lag:.3} ns/lag"
    );
}
