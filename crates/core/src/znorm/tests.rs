extern crate std;

use super::{TickWindow, ZNorm, centered_cross};
use crate::kernel::Metric;
use crate::quantize::{Scale, to_ticks};
use crate::sanitize::{ForwardFill, Sanitized};
use crate::stats::{Window, WindowKind, WindowStats};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec::Vec;
use testkit::generators::{MARKET_TICK, market_series_and_m};

fn window(sum: i64) -> Window {
    Window {
        sum,
        kind: WindowKind::Valid {
            variance_numerator: 1,
            inv_sqrt: 1.0,
        },
    }
}

fn tick_window(entering: i64, leaving: i64, sum: i64) -> TickWindow {
    TickWindow {
        entering,
        leaving,
        stats: window(sum),
    }
}

// ---------- hand-built cases ----------

/// `x = [1, 2, 3, 4]`, `m = 2`: `QT` and `N` at the diagonal's first cell, then one
/// `advance` step, both checked against values worked out by hand.
#[test]
fn small_series_matches_hand_worked_values() {
    let w0 = tick_window(2, 0, 3); // S0 = 1 + 2
    let w1 = tick_window(3, 1, 5); // sum 2 + 3, entering x[2], leaving x[0]
    let w2 = tick_window(4, 2, 7); // sum 3 + 4, entering x[3], leaving x[1]

    let qt01 = ZNorm::start(&[1, 2], &w0, &[2, 3], &w1);
    assert_eq!(qt01, 8); // 1·2 + 2·3
    assert_eq!(centered_cross(qt01, &w0, &w1, 2), 1); // 2·8 − 3·5

    let qt12 = ZNorm::advance(qt01, &w1, &w2);
    assert_eq!(qt12, 18); // 8 − 1·2 + 3·4, matching 2·3 + 3·4 directly
    assert_eq!(centered_cross(qt12, &w1, &w2, 2), 1); // 2·18 − 5·7
}

/// Two windows shifted up by `i64::MAX − 2` so every product and every sum overflows i64,
/// yet `N` is shift-invariant: it must come back exactly what the unshifted windows
/// `[0, 1]`, `[1, 2]` give (`1`), recovered from residues that individually wrapped many
/// times over on the way there.
#[test]
fn wrapping_products_and_sums_still_recover_the_exact_n() {
    let a = [i64::MAX - 2, i64::MAX - 1];
    let b = [i64::MAX - 1, i64::MAX];
    let wa = tick_window(0, 0, -5); // S = 2·(i64::MAX − 2) + 1, wrapped
    let wb = tick_window(0, 0, -3); // S = 2·(i64::MAX − 2) + 3, wrapped

    let qt = ZNorm::start(&a, &wa, &b, &wb);
    assert_eq!(qt, 8);
    assert_eq!(centered_cross(qt, &wa, &wb, 2), 1);
}

// ---------- property: centered_cross matches a direct i128 computation ----------

/// Non-finite or out-of-range samples are quantized then forward-filled; whether a tick
/// was itself present is out of scope here (`stats` owns that), so every quantized tick
/// counts as present.
fn to_tick_series(series: &[f64], scale: Scale) -> Vec<i64> {
    let mut fill = ForwardFill::default();
    series
        .iter()
        .map(|&x| match fill.push(to_ticks(x, scale)) {
            Sanitized::Present(t) | Sanitized::Filled(t) => t,
            Sanitized::Leading => 0,
        })
        .collect()
}

/// The `Window` at every valid position, via the same rolling statistics `cov`'s callers
/// read `S` from.
fn windows_of(ticks: &[i64], m: usize) -> Vec<Window> {
    let mut stats = WindowStats::new(NonZeroUsize::new(m).unwrap());
    ticks
        .iter()
        .filter_map(|&tick| stats.push(tick, true))
        .collect()
}

fn is_valid(window: Window) -> bool {
    matches!(window.kind, WindowKind::Valid { .. })
}

/// The `TickWindow` a caller would hand `cov` for the window starting at `i`: the tick
/// entering as it reaches that position, the tick leaving (`x[i−1]`, unused at `i = 0`,
/// where no `advance` ever reads it), and its own statistics.
fn tick_window_at(ticks: &[i64], windows: &[Window], i: usize, m: usize) -> TickWindow {
    TickWindow {
        entering: ticks[i + m - 1],
        leaving: if i == 0 { 0 } else { ticks[i - 1] },
        stats: windows[i],
    }
}

/// `N`, computed straight from the definitions in unbounded i128: no wrapping, no
/// recurrence, nothing shared with the code under test.
fn exact_centered_cross(ticks: &[i64], i: usize, j: usize, m: usize) -> i128 {
    let qt: i128 = (0..m)
        .map(|k| i128::from(ticks[i + k]) * i128::from(ticks[j + k]))
        .sum();
    let sum_i: i128 = ticks[i..i + m].iter().map(|&t| i128::from(t)).sum();
    let sum_j: i128 = ticks[j..j + m].iter().map(|&t| i128::from(t)).sum();
    (m as i128) * qt - sum_i * sum_j
}

/// Walks diagonal `d` (`j = i + d`) from its first cell, asserting `centered_cross` equals
/// the exact `N` at every pair of valid windows.
fn check_diagonal(ticks: &[i64], windows: &[Window], m: usize, d: usize) {
    if d >= windows.len() {
        return;
    }
    let count = windows.len() - d;
    let at = |i: usize| tick_window_at(ticks, windows, i, m);

    let mut qt = ZNorm::start(&ticks[0..m], &at(0), &ticks[d..d + m], &at(d));
    for i in 0..count {
        let j = i + d;
        if i > 0 {
            qt = ZNorm::advance(qt, &at(i), &at(j));
        }
        if is_valid(windows[i]) && is_valid(windows[j]) {
            let expected = exact_centered_cross(ticks, i, j, m);
            assert_eq!(
                i128::from(centered_cross(qt, &at(i), &at(j), m)),
                expected,
                "i {i}, j {j}, d {d}, m {m}"
            );
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// For a random market series and every `m` it holds at, every diagonal's `centered_cross`
    /// matches the exact `N`, at every pair of valid windows.
    #[test]
    fn centered_cross_matches_exact_n((series, m) in market_series_and_m(2..48)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let ticks = to_tick_series(&series, scale);
        let windows = windows_of(&ticks, m);
        for d in 0..windows.len() {
            check_diagonal(&ticks, &windows, m, d);
        }
    }
}

// ---------- a long diagonal at a high price level, where products wrap ----------

/// A deterministic walk around `10¹⁰` ticks: high enough that two ticks' product
/// (`~10²⁰`) overflows i64 many times over, varied enough (period-11 steps, net drift
/// near zero) that it is neither constant nor itself a straight line.
fn long_series_near_1e10(len: usize) -> Vec<i64> {
    const BASE: i64 = 10_000_000_000;
    const STEPS: [i64; 11] = [0, 2, -1, -2, 1, 0, -2, 2, -1, 1, -1];
    let mut level = BASE;
    (0..len)
        .map(|k| {
            if k > 0 {
                level += STEPS[k % STEPS.len()];
            }
            level
        })
        .collect()
}

#[test]
fn long_diagonal_at_a_high_price_level() {
    let m = 8;
    let ticks = long_series_near_1e10(10_000 + m); // d = 0 walks over 10⁴ steps
    let windows = windows_of(&ticks, m);
    assert!(windows.iter().any(|&w| is_valid(w)));
    for d in [0, 1, 7] {
        check_diagonal(&ticks, &windows, m, d);
    }
}
