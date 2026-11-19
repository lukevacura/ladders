//! One streaming tick is one new row over lags `e+1 ..= w` (`e = ceil(m/4)`), updating each
//! lag's state, converting to ρ, and folding into a running (max ρ, smallest index on ties).
//! Five representations of that row update are timed against each other: `f64` on centered
//! deviations; `i128` on raw ticks; `i64` on `i32` deviations from the series' first tick, with an
//! `i128` fallback when a step would not fit `i64`; `i64` wrapping on raw ticks directly, with no
//! fallback; and `i64` wrapping on the same `i32` deviations, also with no fallback. The two
//! wrapping forms are exact whenever each window's variance numerator `m·Q − S²` fits `i64`,
//! because the Cauchy–Schwarz bound then keeps every cross term within range too, and because
//! wrapping arithmetic is a ring homomorphism mod 2⁶⁴: only the final answer need fit, not any
//! intermediate. Two further variants split the wrapping-on-raw-ticks and centered forms into a
//! fill pass (ρ into a buffer, no branch) and a separate argmax pass, to see whether that
//! unblocks vectorization; a third does the same for wrapping-on-deviations. A correctness gate
//! runs before any timing, and panics rather than let a wrong number be timed.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use testkit::rng::splitmix64;

const PRICE_LEVEL_TICKS: i64 = 10_000_000_000; // 1e8 at a 0.01 tick
const STEP_BOUND: i64 = 8; // keeps the whole walk within a few million ticks of its start
const I64_FIT_BOUND: i128 = 1i128 << 63;

/// A random walk in integer ticks, deterministic in `seed`, with steps uniform in
/// `-step_bound..=step_bound`.
fn random_walk(len: usize, seed: u64, step_bound: i64) -> Vec<i64> {
    let mut state = seed;
    let mut value = PRICE_LEVEL_TICKS;
    let mut walk = Vec::with_capacity(len);
    walk.push(value);
    for _ in 1..len {
        let draw = (splitmix64(&mut state) % (2 * step_bound as u64 + 1)) as i64 - step_bound;
        value += draw;
        walk.push(value);
    }
    walk
}

fn lag_start(m: usize) -> usize {
    ladde::exclusion::zone(m) + 1
}

// ---------- per-window exact statistics, built once, outside every timed region ----------

struct CenteredWindows {
    df: Vec<f64>,
    dg: Vec<f64>,
    inv_norm: Vec<f64>,
}

/// `df`, `dg` and `inv_norm` for every window start `0..count`, from the centered covariance
/// recurrence's definitions, each computed directly (not accumulated) so no rounding carries
/// between windows.
fn centered_windows(x: &[f64], m: usize, count: usize) -> CenteredWindows {
    let m_f64 = m as f64;
    let mean: Vec<f64> = (0..count)
        .map(|k| x[k..k + m].iter().sum::<f64>() / m_f64)
        .collect();
    let mut inv_norm = vec![0.0; count];
    for k in 0..count {
        let variance = x[k..k + m]
            .iter()
            .map(|&value| (value - mean[k]) * (value - mean[k]))
            .sum::<f64>()
            / m_f64;
        inv_norm[k] = 1.0 / (m_f64.sqrt() * variance.sqrt());
    }
    let mut df = vec![0.0; count];
    let mut dg = vec![0.0; count];
    for k in 1..count {
        df[k] = (x[k + m - 1] - x[k - 1]) / 2.0;
        dg[k] = (x[k + m - 1] - mean[k]) + (x[k - 1] - mean[k - 1]);
    }
    CenteredWindows { df, dg, inv_norm }
}

struct TickWindows {
    sum: Vec<i64>,
    inv_den: Vec<f64>,
}

/// `S` and `1/√(m·Q − S²)` for every window start, from raw ticks, exact up to the one `as f64`.
fn tick_windows(x: &[i64], m: usize, count: usize) -> TickWindows {
    let m_i128 = m as i128;
    let mut sum = Vec::with_capacity(count);
    let mut inv_den = Vec::with_capacity(count);
    for k in 0..count {
        let window = &x[k..k + m];
        let window_sum: i64 = window.iter().sum();
        let sum_sq: i128 = window.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
        let denom = (m_i128 * sum_sq - i128::from(window_sum) * i128::from(window_sum)) as f64;
        sum.push(window_sum);
        inv_den.push(1.0 / denom.sqrt());
    }
    TickWindows { sum, inv_den }
}

struct DeviationWindows {
    sum: Vec<i64>,
    inv_den: Vec<f64>,
}

/// Same as `tick_windows`, on `i32` deviations from a fixed reference; `m·Q − S²` fits `i64` for
/// every grid point this bench exercises (the setup, unlike the timed row, need not fall back).
fn deviation_windows(y: &[i32], m: usize, count: usize) -> DeviationWindows {
    let m_i64 = m as i64;
    let mut sum = Vec::with_capacity(count);
    let mut inv_den = Vec::with_capacity(count);
    for k in 0..count {
        let window = &y[k..k + m];
        let window_sum: i64 = window.iter().map(|&v| i64::from(v)).sum();
        let sum_sq: i64 = window.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        let denom = (m_i64 * sum_sq - window_sum * window_sum) as f64;
        sum.push(window_sum);
        inv_den.push(1.0 / denom.sqrt());
    }
    DeviationWindows { sum, inv_den }
}

/// The exact `m·Q − S²` every wrapping variant relies on: computed once in `i128`, checked to
/// fit `i64` (the Cauchy–Schwarz bound this bench's wrapping row updates depend on), then the
/// one `as f64` conversion a wrapping row's own numerator will also need.
fn variance_numerator_inv_den(sum_sq: i128, window_sum: i128, m_i128: i128, window: usize) -> f64 {
    let numerator = m_i128 * sum_sq - window_sum * window_sum;
    assert!(
        (-I64_FIT_BOUND..I64_FIT_BOUND).contains(&numerator),
        "window {window}: variance numerator {numerator} does not fit i64, so no wrapping row \
         update can be trusted here"
    );
    1.0 / (numerator as f64).sqrt()
}

/// Same shape as `tick_windows`, with the per-window sum folded via `wrapping_add`: the state
/// `i64_wrapping`'s own row update is built from, on raw ticks.
fn wrapping_tick_windows(x: &[i64], m: usize, count: usize) -> TickWindows {
    let m_i128 = m as i128;
    let mut sum = Vec::with_capacity(count);
    let mut inv_den = Vec::with_capacity(count);
    for k in 0..count {
        let window = &x[k..k + m];
        let window_sum = window.iter().fold(0i64, |acc, &v| acc.wrapping_add(v));
        let sum_sq: i128 = window.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
        inv_den.push(variance_numerator_inv_den(
            sum_sq,
            i128::from(window_sum),
            m_i128,
            k,
        ));
        sum.push(window_sum);
    }
    TickWindows { sum, inv_den }
}

/// Same as `wrapping_tick_windows`, on `i32` deviations from a fixed reference: the state
/// `i32dev_wrapping`'s row update is built from. `m·Q − S²` is translation-invariant, so this
/// asserts the same bound as the raw-tick form.
fn wrapping_deviation_windows(y: &[i32], m: usize, count: usize) -> TickWindows {
    let m_i128 = m as i128;
    let mut sum = Vec::with_capacity(count);
    let mut inv_den = Vec::with_capacity(count);
    for k in 0..count {
        let window = &y[k..k + m];
        let window_sum = window
            .iter()
            .fold(0i64, |acc, &v| acc.wrapping_add(i64::from(v)));
        let sum_sq: i128 = window.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
        inv_den.push(variance_numerator_inv_den(
            sum_sq,
            i128::from(window_sum),
            m_i128,
            k,
        ));
        sum.push(window_sum);
    }
    TickWindows { sum, inv_den }
}

// ---------- anchors: the exact O(m) state for one specific pair, used only to seed the row
// ---------- before it, never inside a timed region ----------

fn anchor_centered(x: &[f64], a: usize, b: usize, m: usize) -> f64 {
    let mean_a = x[a..a + m].iter().sum::<f64>() / m as f64;
    let mean_b = x[b..b + m].iter().sum::<f64>() / m as f64;
    x[a..a + m]
        .iter()
        .zip(&x[b..b + m])
        .map(|(&xa, &xb)| (xa - mean_a) * (xb - mean_b))
        .sum()
}

fn anchor_dot_i128(x: &[i64], a: usize, b: usize, m: usize) -> i128 {
    x[a..a + m]
        .iter()
        .zip(&x[b..b + m])
        .map(|(&xa, &xb)| i128::from(xa) * i128::from(xb))
        .sum()
}

fn anchor_dot_i64(y: &[i32], a: usize, b: usize, m: usize) -> i64 {
    y[a..a + m]
        .iter()
        .zip(&y[b..b + m])
        .fold(0i64, |acc, (&ya, &yb)| {
            acc.wrapping_add(i64::from(ya) * i64::from(yb))
        })
}

fn anchor_dot_i64_wrapping_ticks(x: &[i64], a: usize, b: usize, m: usize) -> i64 {
    x[a..a + m]
        .iter()
        .zip(&x[b..b + m])
        .fold(0i64, |acc, (&xa, &xb)| {
            acc.wrapping_add(xa.wrapping_mul(xb))
        })
}

fn anchor_dot_i64_wrapping_dev(y: &[i32], a: usize, b: usize, m: usize) -> i64 {
    y[a..a + m]
        .iter()
        .zip(&y[b..b + m])
        .fold(0i64, |acc, (&ya, &yb)| {
            acc.wrapping_add(i64::from(ya).wrapping_mul(i64::from(yb)))
        })
}

// ---------- one row, each representation; the only code any bench_function times ----------

/// The row being updated (`i`), its window length, and the lag range `lo..=hi` it updates.
#[derive(Clone, Copy)]
struct RowShape {
    i: usize,
    m: usize,
    lo: usize,
    hi: usize,
}

fn row_f64_centered(
    shape: RowShape,
    df: &[f64],
    dg: &[f64],
    inv_norm: &[f64],
    cov: &mut [f64],
) -> (f64, usize) {
    let RowShape { i, lo, hi, .. } = shape;
    let mut best_rho = f64::NEG_INFINITY;
    let mut best_j = 0usize;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        cov[slot] += df[i] * dg[j] + df[j] * dg[i];
        let rho = cov[slot] * inv_norm[i] * inv_norm[j];
        if rho >= best_rho {
            best_rho = rho;
            best_j = j;
        }
    }
    (best_rho, best_j)
}

fn row_i128_ticks(
    shape: RowShape,
    x: &[i64],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i128],
) -> (f64, usize) {
    let RowShape { i, m, lo, hi } = shape;
    let mut best_rho = f64::NEG_INFINITY;
    let mut best_j = 0usize;
    let m_i128 = m as i128;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        qt[slot] += i128::from(x[i + m - 1]) * i128::from(x[j + m - 1])
            - i128::from(x[i - 1]) * i128::from(x[j - 1]);
        let numerator = (m_i128 * qt[slot] - i128::from(sum[i]) * i128::from(sum[j])) as f64;
        let rho = numerator * inv_den[i] * inv_den[j];
        if rho >= best_rho {
            best_rho = rho;
            best_j = j;
        }
    }
    (best_rho, best_j)
}

/// `m·qt − sum[i]·sum[j]`, in `i64` when every step fits, otherwise the same quantity in `i128`.
fn centered_numerator_i64_or_i128(m: i64, qt: i64, sum_i: i64, sum_j: i64) -> f64 {
    let via_i64 = m
        .checked_mul(qt)
        .zip(sum_i.checked_mul(sum_j))
        .and_then(|(mq, sij)| mq.checked_sub(sij));
    match via_i64 {
        Some(exact) => exact as f64,
        None => (i128::from(m) * i128::from(qt) - i128::from(sum_i) * i128::from(sum_j)) as f64,
    }
}

fn row_i64_deviations(
    shape: RowShape,
    y: &[i32],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
) -> (f64, usize) {
    let RowShape { i, m, lo, hi } = shape;
    let mut best_rho = f64::NEG_INFINITY;
    let mut best_j = 0usize;
    let m_i64 = m as i64;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        let delta = i64::from(y[i + m - 1]) * i64::from(y[j + m - 1])
            - i64::from(y[i - 1]) * i64::from(y[j - 1]);
        qt[slot] = qt[slot].wrapping_add(delta);
        let numerator = centered_numerator_i64_or_i128(m_i64, qt[slot], sum[i], sum[j]);
        let rho = numerator * inv_den[i] * inv_den[j];
        if rho >= best_rho {
            best_rho = rho;
            best_j = j;
        }
    }
    (best_rho, best_j)
}

fn row_i64_wrapping(
    shape: RowShape,
    x: &[i64],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
) -> (f64, usize) {
    let RowShape { i, m, lo, hi } = shape;
    let mut best_rho = f64::NEG_INFINITY;
    let mut best_j = 0usize;
    let m_i64 = m as i64;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        let delta = x[i + m - 1]
            .wrapping_mul(x[j + m - 1])
            .wrapping_sub(x[i - 1].wrapping_mul(x[j - 1]));
        qt[slot] = qt[slot].wrapping_add(delta);
        let numerator = m_i64
            .wrapping_mul(qt[slot])
            .wrapping_sub(sum[i].wrapping_mul(sum[j]));
        let rho = numerator as f64 * inv_den[i] * inv_den[j];
        if rho >= best_rho {
            best_rho = rho;
            best_j = j;
        }
    }
    (best_rho, best_j)
}

fn row_i32dev_wrapping(
    shape: RowShape,
    y: &[i32],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
) -> (f64, usize) {
    let RowShape { i, m, lo, hi } = shape;
    let mut best_rho = f64::NEG_INFINITY;
    let mut best_j = 0usize;
    let m_i64 = m as i64;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        let delta = i64::from(y[i + m - 1])
            .wrapping_mul(i64::from(y[j + m - 1]))
            .wrapping_sub(i64::from(y[i - 1]).wrapping_mul(i64::from(y[j - 1])));
        qt[slot] = qt[slot].wrapping_add(delta);
        let numerator = m_i64
            .wrapping_mul(qt[slot])
            .wrapping_sub(sum[i].wrapping_mul(sum[j]));
        let rho = numerator as f64 * inv_den[i] * inv_den[j];
        if rho >= best_rho {
            best_rho = rho;
            best_j = j;
        }
    }
    (best_rho, best_j)
}

// ---------- two-pass forms: fill ρ into a buffer with no branch, then argmax separately ----------

fn fill_f64_centered(
    shape: RowShape,
    df: &[f64],
    dg: &[f64],
    inv_norm: &[f64],
    cov: &mut [f64],
    rho: &mut [f64],
) {
    let RowShape { i, lo, hi, .. } = shape;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        cov[slot] += df[i] * dg[j] + df[j] * dg[i];
        rho[slot] = cov[slot] * inv_norm[i] * inv_norm[j];
    }
}

fn fill_i64_wrapping(
    shape: RowShape,
    x: &[i64],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
    rho: &mut [f64],
) {
    let RowShape { i, m, lo, hi } = shape;
    let m_i64 = m as i64;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        let delta = x[i + m - 1]
            .wrapping_mul(x[j + m - 1])
            .wrapping_sub(x[i - 1].wrapping_mul(x[j - 1]));
        qt[slot] = qt[slot].wrapping_add(delta);
        let numerator = m_i64
            .wrapping_mul(qt[slot])
            .wrapping_sub(sum[i].wrapping_mul(sum[j]));
        rho[slot] = numerator as f64 * inv_den[i] * inv_den[j];
    }
}

fn fill_i32dev_wrapping(
    shape: RowShape,
    y: &[i32],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
    rho: &mut [f64],
) {
    let RowShape { i, m, lo, hi } = shape;
    let m_i64 = m as i64;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        let delta = i64::from(y[i + m - 1])
            .wrapping_mul(i64::from(y[j + m - 1]))
            .wrapping_sub(i64::from(y[i - 1]).wrapping_mul(i64::from(y[j - 1])));
        qt[slot] = qt[slot].wrapping_add(delta);
        let numerator = m_i64
            .wrapping_mul(qt[slot])
            .wrapping_sub(sum[i].wrapping_mul(sum[j]));
        rho[slot] = numerator as f64 * inv_den[i] * inv_den[j];
    }
}

/// Max ρ with ties toward the smallest window-start index, scanning an already-filled buffer.
fn argmax_tie_smallest(shape: RowShape, rho: &[f64]) -> (f64, usize) {
    let RowShape { i, lo, hi, .. } = shape;
    let mut best_rho = f64::NEG_INFINITY;
    let mut best_j = 0usize;
    for d in lo..=hi {
        let j = i - d;
        let slot = d - lo;
        if rho[slot] >= best_rho {
            best_rho = rho[slot];
            best_j = j;
        }
    }
    (best_rho, best_j)
}

fn row_f64_centered_twopass(
    shape: RowShape,
    df: &[f64],
    dg: &[f64],
    inv_norm: &[f64],
    cov: &mut [f64],
    rho: &mut [f64],
) -> (f64, usize) {
    fill_f64_centered(shape, df, dg, inv_norm, cov, rho);
    argmax_tie_smallest(shape, rho)
}

fn row_i64_wrapping_twopass(
    shape: RowShape,
    x: &[i64],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
    rho: &mut [f64],
) -> (f64, usize) {
    fill_i64_wrapping(shape, x, sum, inv_den, qt, rho);
    argmax_tie_smallest(shape, rho)
}

fn row_i32dev_wrapping_twopass(
    shape: RowShape,
    y: &[i32],
    sum: &[i64],
    inv_den: &[f64],
    qt: &mut [i64],
    rho: &mut [f64],
) -> (f64, usize) {
    fill_i32dev_wrapping(shape, y, sum, inv_den, qt, rho);
    argmax_tie_smallest(shape, rho)
}

// ---------- correctness gate ----------

/// The small case: `i128_ticks`' ρ against an exact reference built from `testkit::hp`.
fn gate_against_exact_reference() {
    let m = 8;
    let walk = random_walk(64, 0xA5A5_1234_5678_9ABC, STEP_BOUND);
    let x_f64: Vec<f64> = walk.iter().map(|&t| t as f64).collect();
    let (a, b) = (10usize, 30usize);

    let window_a = &x_f64[a..a + m];
    let window_b = &x_f64[b..b + m];
    let exact_cross = testkit::hp::centered_cross(window_a, window_b).to_f64();
    let exact_sigma_a = testkit::hp::std(window_a);
    let exact_sigma_b = testkit::hp::std(window_b);
    let exact_rho = exact_cross / (m as f64 * exact_sigma_a * exact_sigma_b);

    let ticks_a = &walk[a..a + m];
    let ticks_b = &walk[b..b + m];
    let sum_a: i64 = ticks_a.iter().sum();
    let sum_b: i64 = ticks_b.iter().sum();
    let sq =
        |values: &[i64]| -> i128 { values.iter().map(|&v| i128::from(v) * i128::from(v)).sum() };
    let (q_a, q_b) = (sq(ticks_a), sq(ticks_b));
    let qt: i128 = ticks_a
        .iter()
        .zip(ticks_b)
        .map(|(&x, &y)| i128::from(x) * i128::from(y))
        .sum();
    let m_i128 = m as i128;
    let numerator = (m_i128 * qt - i128::from(sum_a) * i128::from(sum_b)) as f64;
    let denom_a = (m_i128 * q_a - i128::from(sum_a) * i128::from(sum_a)) as f64;
    let denom_b = (m_i128 * q_b - i128::from(sum_b) * i128::from(sum_b)) as f64;
    let integer_rho = numerator / (denom_a.sqrt() * denom_b.sqrt());

    let few_ulps = 8.0 * f64::EPSILON;
    let diff = (integer_rho - exact_rho).abs();
    assert!(
        diff <= few_ulps,
        "i128_ticks rho {integer_rho} vs exact reference {exact_rho}: diff {diff} exceeds {few_ulps}"
    );
}

/// One `(w, m)` case for `gate_variants_agree`. `check_deviations` is false for the one case
/// whose local range is large enough that `i64_deviations`'s own running state (not its
/// i64-or-i128 final step, which stays exact) can wrap: that variant's state is built from
/// deviations from a *fixed* reference, so it has strictly less headroom than the wrapping
/// forms, which only need the final answer, not the running state, to fit.
struct GateCase {
    w: usize,
    m: usize,
    step_bound: i64,
    check_deviations: bool,
}

const GATE_CASES: [GateCase; 5] = [
    GateCase {
        w: 2_000,
        m: 16,
        step_bound: STEP_BOUND,
        check_deviations: true,
    },
    GateCase {
        w: 2_000,
        m: 64,
        step_bound: STEP_BOUND,
        check_deviations: true,
    },
    GateCase {
        w: 2_000,
        m: 256,
        step_bound: STEP_BOUND,
        check_deviations: true,
    },
    GateCase {
        w: 20_000,
        m: 64,
        step_bound: STEP_BOUND,
        check_deviations: true,
    },
    GateCase {
        w: 2_000,
        m: 256,
        step_bound: 50_000,
        check_deviations: false,
    },
];

/// Across many consecutive rows, at several `(w, m)`: `i128_ticks`, `i64_wrapping` and
/// `i32dev_wrapping` must land on the same bits of ρ and the same argmax on every case (the
/// wrapping forms' only requirement, `m·Q − S² < 2⁶³` per window, is asserted inside the
/// `wrapping_*_windows` builders); `i64_deviations` is checked wherever its own headroom holds;
/// `f64_centered` must agree with the exact forms to 1e-9 everywhere.
fn gate_variants_agree() {
    const ROWS: usize = 300;
    for case in GATE_CASES {
        let GateCase {
            w,
            m,
            step_bound,
            check_deviations,
        } = case;
        let lo = lag_start(m);
        let hi = w;
        let count = w + ROWS + 1;
        let n = count + m;

        let ticks = random_walk(n, seed_for(w, m) ^ 0x1357_2468_ACE0_BDF1, step_bound);
        let x_f64: Vec<f64> = ticks.iter().map(|&t| t as f64).collect();
        let x0 = ticks[0];
        let y: Vec<i32> = ticks.iter().map(|&t| (t - x0) as i32).collect();

        let centered = centered_windows(&x_f64, m, count);
        let tick_stats = tick_windows(&ticks, m, count);
        let wrap_ticks = wrapping_tick_windows(&ticks, m, count);
        let wrap_dev = wrapping_deviation_windows(&y, m, count);
        let dev_stats = deviation_windows(&y, m, count);

        let i_start = w + 1;
        let mut cov: Vec<f64> = (lo..=hi)
            .map(|d| anchor_centered(&x_f64, i_start - 1, i_start - 1 - d, m))
            .collect();
        let mut qt128: Vec<i128> = (lo..=hi)
            .map(|d| anchor_dot_i128(&ticks, i_start - 1, i_start - 1 - d, m))
            .collect();
        let mut qt_wrap_ticks: Vec<i64> = (lo..=hi)
            .map(|d| anchor_dot_i64_wrapping_ticks(&ticks, i_start - 1, i_start - 1 - d, m))
            .collect();
        let mut qt_wrap_dev: Vec<i64> = (lo..=hi)
            .map(|d| anchor_dot_i64_wrapping_dev(&y, i_start - 1, i_start - 1 - d, m))
            .collect();
        let mut qt64: Vec<i64> = (lo..=hi)
            .map(|d| anchor_dot_i64(&y, i_start - 1, i_start - 1 - d, m))
            .collect();

        for row in 0..ROWS {
            let shape = RowShape {
                i: i_start + row,
                m,
                lo,
                hi,
            };
            let i = shape.i;

            let (rho_f64, _) = row_f64_centered(
                shape,
                &centered.df,
                &centered.dg,
                &centered.inv_norm,
                &mut cov,
            );
            let (rho_128, j_128) = row_i128_ticks(
                shape,
                &ticks,
                &tick_stats.sum,
                &tick_stats.inv_den,
                &mut qt128,
            );
            let (rho_wrap_ticks, j_wrap_ticks) = row_i64_wrapping(
                shape,
                &ticks,
                &wrap_ticks.sum,
                &wrap_ticks.inv_den,
                &mut qt_wrap_ticks,
            );
            let (rho_wrap_dev, j_wrap_dev) = row_i32dev_wrapping(
                shape,
                &y,
                &wrap_dev.sum,
                &wrap_dev.inv_den,
                &mut qt_wrap_dev,
            );

            assert_eq!(
                rho_128.to_bits(),
                rho_wrap_ticks.to_bits(),
                "row {i} (w={w}, m={m}): i128_ticks rho {rho_128} != i64_wrapping rho {rho_wrap_ticks}"
            );
            assert_eq!(
                j_128, j_wrap_ticks,
                "row {i} (w={w}, m={m}): i128_ticks argmax {j_128} != i64_wrapping argmax {j_wrap_ticks}"
            );
            assert_eq!(
                rho_128.to_bits(),
                rho_wrap_dev.to_bits(),
                "row {i} (w={w}, m={m}): i128_ticks rho {rho_128} != i32dev_wrapping rho {rho_wrap_dev}"
            );
            assert_eq!(
                j_128, j_wrap_dev,
                "row {i} (w={w}, m={m}): i128_ticks argmax {j_128} != i32dev_wrapping argmax {j_wrap_dev}"
            );

            let drift = (rho_128 - rho_f64).abs();
            assert!(
                drift <= 1e-9,
                "row {i} (w={w}, m={m}): f64_centered rho {rho_f64} vs integer rho {rho_128}, drift {drift}"
            );

            if check_deviations {
                let (rho_64, j_64) =
                    row_i64_deviations(shape, &y, &dev_stats.sum, &dev_stats.inv_den, &mut qt64);
                assert_eq!(
                    rho_128.to_bits(),
                    rho_64.to_bits(),
                    "row {i} (w={w}, m={m}): i128_ticks rho {rho_128} != i64_deviations rho {rho_64}"
                );
                assert_eq!(
                    j_128, j_64,
                    "row {i} (w={w}, m={m}): argmax {j_128} != {j_64}"
                );
            }
        }
    }
}

/// The two-pass forms (ρ into a buffer, then a separate argmax scan) must match their
/// single-pass counterparts bit for bit: same arithmetic, only reordered.
fn gate_twopass_matches_single_pass() {
    let (w, m) = (3_000usize, 64usize);
    let lo = lag_start(m);
    let hi = w;
    let count = w + 2;
    let n = count + m;

    let ticks = random_walk(n, 0x7777_8888_9999_AAAA, STEP_BOUND);
    let x_f64: Vec<f64> = ticks.iter().map(|&t| t as f64).collect();
    let x0 = ticks[0];
    let y: Vec<i32> = ticks.iter().map(|&t| (t - x0) as i32).collect();

    let centered = centered_windows(&x_f64, m, count);
    let wrap_ticks = wrapping_tick_windows(&ticks, m, count);
    let wrap_dev = wrapping_deviation_windows(&y, m, count);

    let i_row = w + 1;
    let shape = RowShape {
        i: i_row,
        m,
        lo,
        hi,
    };
    let base_cov: Vec<f64> = (lo..=hi)
        .map(|d| anchor_centered(&x_f64, i_row - 1, i_row - 1 - d, m))
        .collect();
    let base_qt_ticks: Vec<i64> = (lo..=hi)
        .map(|d| anchor_dot_i64_wrapping_ticks(&ticks, i_row - 1, i_row - 1 - d, m))
        .collect();
    let base_qt_dev: Vec<i64> = (lo..=hi)
        .map(|d| anchor_dot_i64_wrapping_dev(&y, i_row - 1, i_row - 1 - d, m))
        .collect();

    let mut rho_buf = vec![0.0; hi - lo + 1];

    let mut cov_a = base_cov.clone();
    let single_f64 = row_f64_centered(
        shape,
        &centered.df,
        &centered.dg,
        &centered.inv_norm,
        &mut cov_a,
    );
    let mut cov_b = base_cov.clone();
    let two_f64 = row_f64_centered_twopass(
        shape,
        &centered.df,
        &centered.dg,
        &centered.inv_norm,
        &mut cov_b,
        &mut rho_buf,
    );
    assert_eq!(single_f64.0.to_bits(), two_f64.0.to_bits());
    assert_eq!(single_f64.1, two_f64.1);

    let mut qt_a = base_qt_ticks.clone();
    let single_wrap = row_i64_wrapping(
        shape,
        &ticks,
        &wrap_ticks.sum,
        &wrap_ticks.inv_den,
        &mut qt_a,
    );
    let mut qt_b = base_qt_ticks.clone();
    let two_wrap = row_i64_wrapping_twopass(
        shape,
        &ticks,
        &wrap_ticks.sum,
        &wrap_ticks.inv_den,
        &mut qt_b,
        &mut rho_buf,
    );
    assert_eq!(single_wrap.0.to_bits(), two_wrap.0.to_bits());
    assert_eq!(single_wrap.1, two_wrap.1);

    let mut qt_c = base_qt_dev.clone();
    let single_dev = row_i32dev_wrapping(shape, &y, &wrap_dev.sum, &wrap_dev.inv_den, &mut qt_c);
    let mut qt_d = base_qt_dev.clone();
    let two_dev = row_i32dev_wrapping_twopass(
        shape,
        &y,
        &wrap_dev.sum,
        &wrap_dev.inv_den,
        &mut qt_d,
        &mut rho_buf,
    );
    assert_eq!(single_dev.0.to_bits(), two_dev.0.to_bits());
    assert_eq!(single_dev.1, two_dev.1);
}

fn correctness_gate() {
    gate_against_exact_reference();
    gate_variants_agree();
    gate_twopass_matches_single_pass();
}

// ---------- the grid ----------

fn seed_for(w: usize, m: usize) -> u64 {
    (w as u64)
        .wrapping_mul(1_000_003)
        .wrapping_add(m as u64)
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
}

const WIDTHS: [usize; 3] = [1_000, 10_000, 100_000];
const WINDOWS: [usize; 3] = [16, 64, 256];

fn bench_representation(c: &mut Criterion) {
    correctness_gate();

    for &w in &WIDTHS {
        for &m in &WINDOWS {
            let lo = lag_start(m);
            let hi = w;
            let i_row = w + 1;
            let count = i_row + 1;
            let n = count + m;

            let ticks = random_walk(n, seed_for(w, m), STEP_BOUND);
            let x_f64: Vec<f64> = ticks.iter().map(|&t| t as f64).collect();
            let x0 = ticks[0];
            let y: Vec<i32> = ticks.iter().map(|&t| (t - x0) as i32).collect();

            let centered = centered_windows(&x_f64, m, count);
            let tick_stats = tick_windows(&ticks, m, count);
            let dev_stats = deviation_windows(&y, m, count);
            let wrap_ticks = wrapping_tick_windows(&ticks, m, count);
            let wrap_dev = wrapping_deviation_windows(&y, m, count);

            let base_cov: Vec<f64> = (lo..=hi)
                .map(|d| anchor_centered(&x_f64, i_row - 1, i_row - 1 - d, m))
                .collect();
            let base_qt128: Vec<i128> = (lo..=hi)
                .map(|d| anchor_dot_i128(&ticks, i_row - 1, i_row - 1 - d, m))
                .collect();
            let base_qt64: Vec<i64> = (lo..=hi)
                .map(|d| anchor_dot_i64(&y, i_row - 1, i_row - 1 - d, m))
                .collect();
            let base_qt_wrap_ticks: Vec<i64> = (lo..=hi)
                .map(|d| anchor_dot_i64_wrapping_ticks(&ticks, i_row - 1, i_row - 1 - d, m))
                .collect();
            let base_qt_wrap_dev: Vec<i64> = (lo..=hi)
                .map(|d| anchor_dot_i64_wrapping_dev(&y, i_row - 1, i_row - 1 - d, m))
                .collect();

            let id = format!("w{w}_m{m}");
            let shape = RowShape {
                i: i_row,
                m,
                lo,
                hi,
            };
            let mut rho_buf = vec![0.0; hi - lo + 1];

            c.bench_function(&format!("f64_centered/{id}"), |b| {
                b.iter_batched(
                    || base_cov.clone(),
                    |mut cov| {
                        black_box(row_f64_centered(
                            shape,
                            &centered.df,
                            &centered.dg,
                            &centered.inv_norm,
                            &mut cov,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("i128_ticks/{id}"), |b| {
                b.iter_batched(
                    || base_qt128.clone(),
                    |mut qt| {
                        black_box(row_i128_ticks(
                            shape,
                            &ticks,
                            &tick_stats.sum,
                            &tick_stats.inv_den,
                            &mut qt,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("i64_deviations/{id}"), |b| {
                b.iter_batched(
                    || base_qt64.clone(),
                    |mut qt| {
                        black_box(row_i64_deviations(
                            shape,
                            &y,
                            &dev_stats.sum,
                            &dev_stats.inv_den,
                            &mut qt,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("i64_wrapping/{id}"), |b| {
                b.iter_batched(
                    || base_qt_wrap_ticks.clone(),
                    |mut qt| {
                        black_box(row_i64_wrapping(
                            shape,
                            &ticks,
                            &wrap_ticks.sum,
                            &wrap_ticks.inv_den,
                            &mut qt,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("i32dev_wrapping/{id}"), |b| {
                b.iter_batched(
                    || base_qt_wrap_dev.clone(),
                    |mut qt| {
                        black_box(row_i32dev_wrapping(
                            shape,
                            &y,
                            &wrap_dev.sum,
                            &wrap_dev.inv_den,
                            &mut qt,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("f64_centered_twopass/{id}"), |b| {
                b.iter_batched(
                    || base_cov.clone(),
                    |mut cov| {
                        black_box(row_f64_centered_twopass(
                            shape,
                            &centered.df,
                            &centered.dg,
                            &centered.inv_norm,
                            &mut cov,
                            &mut rho_buf,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("i64_wrapping_twopass/{id}"), |b| {
                b.iter_batched(
                    || base_qt_wrap_ticks.clone(),
                    |mut qt| {
                        black_box(row_i64_wrapping_twopass(
                            shape,
                            &ticks,
                            &wrap_ticks.sum,
                            &wrap_ticks.inv_den,
                            &mut qt,
                            &mut rho_buf,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });

            c.bench_function(&format!("i32dev_wrapping_twopass/{id}"), |b| {
                b.iter_batched(
                    || base_qt_wrap_dev.clone(),
                    |mut qt| {
                        black_box(row_i32dev_wrapping_twopass(
                            shape,
                            &y,
                            &wrap_dev.sum,
                            &wrap_dev.inv_den,
                            &mut qt,
                            &mut rho_buf,
                        ))
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30);
    targets = bench_representation
}
criterion_main!(benches);
