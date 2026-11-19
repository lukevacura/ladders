extern crate std;

use super::{
    MAX_TICKS, SPAN_SLACK, Scale, ScaleChoice, clean_windows, joined, passes, pow2,
    round_ties_even, smallest_power_of_two_at_least, to_ticks,
};
use crate::sanitize::{ForwardFill, Sanitized};
use crate::stats::{WindowKind, WindowStats};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec::Vec;

#[test]
fn scale_rejects_non_positive_and_non_finite() {
    for value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(Scale::new(value), None, "{value:?}");
    }
}

#[test]
fn scale_accepts_positive_finite_values() {
    for value in [0.01, 1e-4, 1e-9, 1.0, f64::MAX] {
        assert_eq!(Scale::new(value).map(Scale::get), Some(value), "{value:?}");
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// `core`'s hand-rolled rounding against the real IEEE operation, which `std`
    /// still provides for test code even though the library itself is `no_std`.
    #[test]
    fn round_ties_even_matches_the_ieee_operation(bits in any::<u64>()) {
        let x = f64::from_bits(bits);
        prop_assume!(x.is_finite());
        prop_assert_eq!(round_ties_even(x).to_bits(), x.round_ties_even().to_bits());
    }
}

#[test]
fn ticks_round_to_nearest_even() {
    let scale = Scale::new(1.0).unwrap();
    for (value, expected) in [
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (3.5, 4),
        (-0.5, 0),
        (-1.5, -2),
        (-2.5, -2),
    ] {
        assert_eq!(to_ticks(value, scale), Some(expected), "{value}");
    }
}

#[test]
fn division_is_not_reciprocal_multiplication() {
    // Dividing by this scale lands value −4983 exactly on the tie −29671.5, resolved
    // to the even −29672. Multiplying by its reciprocal rounds the reciprocal first,
    // landing just short of the tie and missing it: −29671.499999999996, giving −29671.
    let scale = Scale::new(22.0 / 131.0).unwrap();
    assert_eq!(to_ticks(-4983.0, scale), Some(-29672));
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn on_grid_values_round_trip(k in -MAX_TICKS..=MAX_TICKS, scale_pick in 0..3usize) {
        let scale = Scale::new([0.01, 1e-4, 1e-9][scale_pick]).unwrap();
        let value = k as f64 * scale.get();
        // `k as f64 * scale` and then `/ scale` isn't always `k` again (double
        // rounding on the multiply): only assert the round trip where it provably is.
        prop_assume!(value.is_finite() && value / scale.get() == k as f64);
        prop_assert_eq!(to_ticks(value, scale), Some(k));
    }
}

#[test]
fn range_boundary() {
    let scale = Scale::new(1.0).unwrap();
    let max = MAX_TICKS as f64;
    let just_beyond = f64::from_bits(max.to_bits() + 1); // next representable f64 above max

    assert_eq!(to_ticks(max, scale), Some(MAX_TICKS));
    assert_eq!(to_ticks(-max, scale), Some(-MAX_TICKS));
    assert_eq!(to_ticks(just_beyond, scale), None);
    assert_eq!(to_ticks(-just_beyond, scale), None);
}

#[test]
fn quotient_overflowing_to_infinity_is_rejected() {
    let tiny = Scale::new(1e-300).unwrap();
    assert_eq!(to_ticks(f64::MAX, tiny), None);
    assert_eq!(to_ticks(f64::MIN, tiny), None);
}

#[test]
fn extreme_magnitude_is_rejected_without_overflow() {
    // A quotient this far negative casts to i64::MIN by saturation; a signed `.abs()`
    // taken on that cast, rather than on the float, would itself overflow and wrap
    // back to a negative number, hiding the very overflow it was meant to catch.
    let scale = Scale::new(1.0).unwrap();
    assert_eq!(to_ticks(-1e300, scale), None);
}

#[test]
fn nonfinite_yields_none() {
    let scale = Scale::new(1.0).unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(to_ticks(value, scale), None);
    }
}

// ---------- helpers behind Scale::finest_fitting ----------

#[test]
fn clean_windows_flags_each_window_holding_a_non_finite_sample() {
    let series = [
        1.0,
        f64::NAN,
        2.0,
        3.0,
        4.0,
        5.0,
        f64::INFINITY,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ];
    for m in 1..=4 {
        let expected: Vec<bool> = (0..=series.len() - m)
            .map(|row| series[row..row + m].iter().all(|x| x.is_finite()))
            .collect();
        assert_eq!(clean_windows(&series, m), expected, "m = {m}");
    }
}

#[test]
fn pow2_is_exact_in_both_directions() {
    for exponent in -40..=40 {
        assert_eq!(pow2(exponent), 2.0_f64.powi(exponent), "2^{exponent}");
    }
}

#[test]
fn smallest_power_of_two_at_least_is_the_ceiling_power() {
    for (min_scale, expected) in [
        (1.0, 1.0),
        (1.5, 2.0),
        (3.0, 4.0),
        (4.0, 4.0),
        (4.5, 8.0),
        (0.3, 0.5),
        (0.25, 0.25),
        (1000.0, 1024.0),
        (0.0, 1.0),
        (-2.0, 1.0),
    ] {
        assert_eq!(
            smallest_power_of_two_at_least(min_scale),
            expected,
            "{min_scale}"
        );
    }
}

#[test]
fn passes_checks_every_clean_window_not_only_the_first() {
    let m = NonZeroUsize::new(2).unwrap();
    // Only the last window's variance numerator (2 * 2 * 3e9^2 > 2^63) overflows at
    // scale 1; at scale 4 it fits.
    let series = [0.0, 0.0, 0.0, 3e9, -3e9];
    let clean = clean_windows(&series, 2);
    assert!(!passes(&series, m, Scale::new(1.0).unwrap(), &clean));
    assert!(passes(&series, m, Scale::new(4.0).unwrap(), &clean));

    // A dirty first window must not mask a later clean window that overflows.
    let series = [f64::NAN, 0.0, 0.0, 3e9, -3e9];
    let clean = clean_windows(&series, 2);
    assert!(!clean[0] && clean[3]);
    assert!(!passes(&series, m, Scale::new(1.0).unwrap(), &clean));
    assert!(passes(&series, m, Scale::new(4.0).unwrap(), &clean));
}

// ---------- Scale::finest_fitting ----------

/// A deterministic walk (the same small LCG other tests in this crate seed by hand):
/// `len` steps in `-2..=2`, starting at `level`.
fn deterministic_walk(level: f64, len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut value = level;
    let mut series = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let step = ((state >> 33) % 5) as i64 - 2; // -2..=2
        value += step as f64;
        series.push(value);
    }
    series
}

/// `(valid, constant, non_finite)` counts across every window `series` (`m` long)
/// completes at `scale`: the same quantization and window statistics [`passes`] runs,
/// but reporting what each window actually was instead of only whether every clean one
/// held.
fn classification_counts(series: &[f64], m: NonZeroUsize, scale: Scale) -> (usize, usize, usize) {
    let mut fill = ForwardFill::<i64>::default();
    let mut stats = WindowStats::new(m);
    let (mut valid, mut constant, mut non_finite) = (0usize, 0usize, 0usize);
    for &x in series {
        let (tick, had_tick) = match fill.push(to_ticks(x, scale)) {
            Sanitized::Present(t) => (t, true),
            Sanitized::Filled(t) => (t, false),
            Sanitized::Leading => (0, false),
        };
        if let Some(window) = stats.push(tick, had_tick) {
            match window.kind {
                WindowKind::Valid { .. } => valid += 1,
                WindowKind::Constant => constant += 1,
                WindowKind::NonFinite => non_finite += 1,
            }
        }
    }
    (valid, constant, non_finite)
}

#[test]
fn finest_fitting_fixes_the_market_level_regression() {
    // The literal report that found this bug: a scale rule fitting only single-tick
    // magnitude (`MAX_TICKS`) against a series like `100 + cumsum(randn(2000))`, `m =
    // 50`, picked a scale so fine every one of 1951 windows came out `NonFinite`: `V =
    // m·Q − S²` on ticks that large overflows `2⁶³` even though no individual tick
    // overflows `MAX_TICKS`.
    let m = NonZeroUsize::new(50).unwrap();
    let series = deterministic_walk(100.0, 2000, 0x9E3779B97F4A7C15);

    let scale = Scale::finest_fitting(&series, m).expect("every sample here is finite");
    let (valid, constant, non_finite) = classification_counts(&series, m, scale);

    assert_eq!(valid + constant + non_finite, series.len() - m.get() + 1);
    assert_eq!(
        non_finite,
        0,
        "every clean window must pass at scale {}",
        scale.get()
    );
    assert!(
        valid > 0,
        "a varying walk should produce Valid windows, not just Constant ones"
    );
}

#[test]
fn finest_fitting_holds_at_a_market_level_offset() {
    // The same check, moved to a genuinely market-scale level (~1e10) with small moves:
    // `max_abs` alone would pick a far coarser scale than the regression above needed,
    // so this is evidence the fix holds well away from that specific boundary too.
    let m = NonZeroUsize::new(50).unwrap();
    let series = deterministic_walk(1e10, 1000, 0xA5A5_A5A5_A5A5_A5A5);

    let scale = Scale::finest_fitting(&series, m).expect("every sample here is finite");
    let (valid, constant, non_finite) = classification_counts(&series, m, scale);

    assert_eq!(non_finite, 0);
    assert!(valid > 0);
    let _ = constant;
}

#[test]
fn finest_fitting_on_a_flat_series_is_the_magnitude_floor() {
    // Every window has range 0 (`V = 0` always, `Constant`): nothing here constrains
    // scale beyond the magnitude floor, which for an all-zero series is the arbitrary
    // baseline `1.0`.
    let m = NonZeroUsize::new(10).unwrap();
    let series: Vec<f64> = std::vec![0.0; 40];

    let scale = Scale::finest_fitting(&series, m).expect("every sample here is finite");
    assert_eq!(scale, Scale::new(1.0).unwrap());

    let (valid, constant, non_finite) = classification_counts(&series, m, scale);
    assert_eq!(valid, 0);
    assert_eq!(non_finite, 0);
    assert_eq!(constant, series.len() - m.get() + 1);
}

#[test]
fn finest_fitting_is_none_for_a_series_shorter_than_m() {
    let m = NonZeroUsize::new(5).unwrap();
    let series: Vec<f64> = std::vec![1.0, 2.0, 3.0];
    assert_eq!(Scale::finest_fitting(&series, m), None);
}

#[test]
fn finest_fitting_is_none_with_no_finite_sample_at_all() {
    let m = NonZeroUsize::new(3).unwrap();
    let series: Vec<f64> = std::vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    assert_eq!(Scale::finest_fitting(&series, m), None);
}

/// `m` and a fully finite series at least `m` long: a walk from a level in `0.0..=1e12`
/// (ordinary and market-scale starting points alike) with steps in `-1000.0..=1000.0` —
/// never `NaN` or infinite, so every window is clean and `finest_fitting` always returns
/// `Some`.
fn m_and_finite_series() -> impl Strategy<Value = (NonZeroUsize, Vec<f64>)> {
    (2usize..=6).prop_flat_map(|m| {
        (
            Just(m),
            0.0..=1e12_f64,
            proptest::collection::vec(-1000.0..=1000.0_f64, m..=60),
        )
            .prop_map(|(m, level, steps)| {
                let series: Vec<f64> = steps
                    .into_iter()
                    .scan(level, |acc, step| {
                        *acc += step;
                        Some(*acc)
                    })
                    .collect();
                (NonZeroUsize::new(m).unwrap(), series)
            })
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// The scale `finest_fitting` returns passes every clean window (every window here,
    /// `series` being fully finite by construction), and half of it fails at least one —
    /// so it really is the finest, not merely a passing one. Excludes the one series
    /// with no real constraint at all (every sample exactly `0.0`, so every window is
    /// `Constant` at any scale): `prop_assume!` skips it rather than asserting a
    /// "finest" that is not actually unique there.
    #[test]
    fn finest_fitting_scale_is_minimal((m, series) in m_and_finite_series()) {
        prop_assume!(series.iter().any(|&x| x != 0.0));

        let scale = Scale::finest_fitting(&series, m).expect("every sample here is finite");
        let clean = clean_windows(&series, m.get());
        prop_assert!(passes(&series, m, scale, &clean));

        let half = Scale::new(scale.get() / 2.0)
            .expect("halving a positive finite scale stays positive and finite");
        prop_assert!(!passes(&series, m, half, &clean));
    }
}

// ---------- Scale::infer ----------

fn candidates() -> std::vec::Vec<f64> {
    super::default_candidates().collect()
}

#[test]
fn candidate_family_is_decimal_then_binary_coarse_to_fine() {
    let c = candidates();
    assert_eq!(c.len(), 22 * 4 + 40);
    assert_eq!(&c[..5], &[5e9, 2.5e9, 2e9, 1e9, 5e8]);
    assert_eq!(c[22 * 4 - 1], 1e-12);
    assert_eq!(c[22 * 4 - 4], 5e-12);
    assert_eq!(c[22 * 4], 512.0);
    assert_eq!(*c.last().unwrap(), pow2(-30));
    // Spelled exactly as the literals are.
    for lit in [0.01, 0.025, 0.05, 0.02, 0.25, 1.0, 100.0, 2.5e-7, 1e-12] {
        assert!(c.contains(&lit), "{lit}");
    }
    assert!(c[..22 * 4].windows(2).all(|w| w[0] > w[1]));
    assert!(c[22 * 4..].windows(2).all(|w| w[0] > w[1]));
}

/// A walk on `grid` (steps of -3..=3 grid units from `start` units), with NaN gaps and a
/// flat run.
fn grid_walk(grid: f64, start: i64, steps: &[i64]) -> Vec<f64> {
    let mut level = start;
    let mut out: Vec<f64> = steps
        .iter()
        .map(|s| {
            level += s;
            level as f64 * grid
        })
        .collect();
    if out.len() > 12 {
        out[5] = f64::NAN;
        out[9] = f64::NAN;
        let flat = out[2];
        out[2..5].fill(flat);
    }
    out
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// The inferred scale is a candidate every finite sample lies on and the kernel
    /// accepts, and no earlier candidate is both; a walk on a known grid never lands finer
    /// than that grid's own candidate.
    #[test]
    fn infer_returns_the_first_candidate_that_holds_and_passes(
        grid in prop::sample::select(std::vec![0.01_f64, 0.25, 0.05, 1.0, 100.0]),
        start in -1000_i64..1000,
        steps in proptest::collection::vec(-3_i64..=3, 40..120),
        m in 3usize..8,
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let series = grid_walk(grid, start, &steps);
        prop_assume!(series.iter().any(|&x| x.is_finite() && x != 0.0));
        let scale = Scale::infer(&series, m).expect("finite samples present");
        let finite: Vec<f64> = series.iter().copied().filter(|x| x.is_finite()).collect();
        prop_assert!(scale.holds(&finite));
        let clean = clean_windows(&series, m.get());
        prop_assert!(passes(&series, m, scale, &clean));
        let position = candidates().iter().position(|&c| c == scale.get())
            .expect("a grid hit is a candidate");
        for &coarser in &candidates()[..position] {
            let s = Scale::new(coarser).unwrap();
            prop_assert!(!(s.holds(&finite) && passes(&series, m, s, &clean)), "{coarser}");
        }
        // The data's own grid is a candidate that holds, so the result is at least as coarse.
        prop_assert!(scale.get() >= grid, "{} vs {}", scale.get(), grid);
        prop_assert_eq!(Scale::infer(&series, m), Some(scale));
    }

    /// Float data on no candidate grid gets exactly `fast_fitting`'s scale, and
    /// `finest_fitting`'s on the opt-in.
    #[test]
    fn infer_on_off_grid_floats_is_fast_fitting(
        m in 2usize..6,
        series in proptest::collection::vec(0.0..=1000.0_f64, 6..60),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let off: Vec<f64> = series.iter().map(|x| x * core::f64::consts::PI + 1.0 / 3.0).collect();
        prop_assume!(candidates().iter().all(|&c| !Scale::new(c).unwrap().holds(&off)));
        prop_assert_eq!(Scale::infer(&off, m), Scale::fast_fitting(&off, m));
        prop_assert_eq!(Scale::infer_finest(&off, m), Scale::finest_fitting(&off, m));
        prop_assert!(Scale::infer(&off, m).is_some());
    }
}

#[test]
fn infer_finds_the_decimal_grid_and_not_a_binary_one() {
    let m = NonZeroUsize::new(4).unwrap();
    let series: std::vec::Vec<f64> = (0..40).map(|k| (k * k % 17) as f64 * 0.01).collect();
    assert_eq!(Scale::infer(&series, m), Scale::new(0.01));
    let quarter: std::vec::Vec<f64> = (0..40).map(|k| (k * k % 17) as f64 * 0.25).collect();
    assert_eq!(Scale::infer(&quarter, m), Scale::new(0.25));
}

#[test]
fn infer_skips_non_finite_samples_and_is_none_without_windows_or_finite_samples() {
    let m = NonZeroUsize::new(3).unwrap();
    let gappy = [0.05, f64::NAN, 0.10, f64::INFINITY, 0.15, 0.30, 0.20];
    assert_eq!(Scale::infer(&gappy, m), Scale::new(0.05));
    assert_eq!(Scale::infer(&[1.0, 2.0], m), None);
    assert_eq!(Scale::infer(&[f64::NAN; 6], m), None);
    assert_eq!(Scale::infer(&[], m), None);
}

#[test]
fn infer_rejects_a_fine_grid_the_kernel_would_not_accept() {
    // At this level the finest candidates "hold" (every float is a whole number of such
    // ticks) yet their tick counts overflow the kernel's range, so they must be skipped.
    let m = NonZeroUsize::new(3).unwrap();
    let series: std::vec::Vec<f64> = (0..20)
        .map(|k| 1e10 + (k as f64 * 0.7311).sin() * core::f64::consts::PI)
        .collect();
    let fine = Scale::new(pow2(-30)).unwrap();
    let clean = clean_windows(&series, 3);
    assert!(fine.holds(&series) && !passes(&series, m, fine, &clean));
    let scale = Scale::infer(&series, m).unwrap();
    assert_ne!(scale, fine);
    assert!(scale.holds(&series) && passes(&series, m, scale, &clean));
}

#[test]
fn infer_falls_back_when_the_holding_grid_overflows_the_kernel() {
    // Whole numbers up to 1e12 lie on the unit grid within the tick cap, but a window of
    // such ticks has a variance far past the kernel's range.
    let m = NonZeroUsize::new(4).unwrap();
    let series: std::vec::Vec<f64> = (0..30)
        .map(|k| ((k as f64 * 1.37).sin().abs() * 1e12).floor())
        .collect();
    let unit = Scale::new(1.0).unwrap();
    let clean = clean_windows(&series, 4);
    assert!(unit.holds(&series) && !passes(&series, m, unit, &clean));
    let scale = Scale::infer(&series, m).unwrap();
    assert_ne!(scale, unit);
    assert_eq!(Some(scale), Scale::fast_fitting(&series, m));
    assert_eq!(
        Scale::infer_finest(&series, m),
        Scale::finest_fitting(&series, m)
    );
}

/// Float data at a large level, `spread` wide: noise and a walk.
fn level_series(level: f64, spread: f64, walk: bool) -> std::vec::Vec<f64> {
    let mut x = 0.0;
    (0..60)
        .map(|k| {
            let r = (k as f64 * 2.399 + 0.3).sin() * 0.5 + (k as f64 * 0.71).cos() * 0.3;
            x = if walk { x + r } else { r };
            level + x * spread * core::f64::consts::PI
        })
        .collect()
}

#[test]
fn infer_on_floats_at_large_levels_is_fast_fitting() {
    let m = NonZeroUsize::new(8).unwrap();
    for level in [1e8, 1e10, 1e12, 1e14] {
        for spread in [1.0, 1e-3] {
            for walk in [false, true] {
                let series = level_series(level, spread, walk);
                // Below the level's float resolution the data is one repeated value, on
                // which any scale is as good as another.
                let mut sorted = series.clone();
                sorted.sort_by(f64::total_cmp);
                sorted.dedup();
                if sorted.len() < 3 {
                    continue;
                }
                assert_eq!(
                    Scale::infer(&series, m),
                    Scale::fast_fitting(&series, m),
                    "level {level} spread {spread} walk {walk}"
                );
                assert_eq!(
                    Scale::infer_finest(&series, m),
                    Scale::finest_fitting(&series, m),
                    "level {level} spread {spread} walk {walk}"
                );
            }
        }
    }
}

#[test]
fn infer_keeps_a_grid_whose_values_are_computed_two_ways() {
    let m = NonZeroUsize::new(4).unwrap();
    let mixed = [
        0.1 + 0.2,
        0.3,
        0.5,
        0.1,
        0.4,
        0.7 - 0.2,
        0.2,
        0.6,
        0.1 + 0.1 + 0.1,
    ];
    assert_ne!(mixed[0], mixed[1]);
    assert_eq!(Scale::infer(&mixed, m), Scale::new(0.1));
    let cents: std::vec::Vec<f64> = (0..60u32)
        .map(|k| {
            let c = (k * k * 7 % 23) as f64;
            if k % 2 == 0 { c * 0.01 } else { c / 100.0 }
        })
        .collect();
    assert!(cents.iter().any(|&x| x != 0.0));
    assert_eq!(Scale::infer(&cents, m), Scale::new(0.01));
}

#[test]
fn infer_keeps_a_real_grid_at_a_large_level() {
    let m = NonZeroUsize::new(8).unwrap();
    let cents: std::vec::Vec<f64> = (0..60)
        .map(|k| (10_000_000 + (k * k * 7 % 23) as i64) as f64 * 0.01 + 1e8)
        .collect();
    assert_eq!(Scale::infer(&cents, m), Scale::new(0.01));
    let ints: std::vec::Vec<f64> = (0..60).map(|k| 1e10 + (k * k * 7 % 23) as f64).collect();
    assert_eq!(Scale::infer(&ints, m), Scale::new(1.0));
}

#[test]
fn infer_cap_rejects_a_fine_grid_that_alone_holds_and_passes() {
    let m = NonZeroUsize::new(4).unwrap();
    // Level 1e10, spread 1: a fine grid holds, keeps values apart and passes, but is far
    // past the tick cap.
    let wide = level_series(1e10, 1.0, false);
    let fine = Scale::new(1e-5).unwrap();
    let clean = clean_windows(&wide, 4);
    assert!(fine.holds(&wide) && passes(&wide, m, fine, &clean));
    assert_ne!(Scale::infer(&wide, m), Some(fine));
}

// ---------- holds / for_query ----------

#[test]
fn grid_holds_at_every_level_for_decimal_and_dyadic_scales() {
    for scale in [0.01, 0.125] {
        let sc = Scale::new(scale).unwrap();
        for level in [1.0, 1e4, 1e8, 1.23e8] {
            let on: Vec<f64> = (0..50).map(|k| level + k as f64 * scale).collect();
            assert!(sc.holds(&on), "level {level} scale {scale}");
            let tenths: Vec<f64> = (0..50).map(|k| level + (k as f64 + 0.1) * scale).collect();
            assert!(!sc.holds(&tenths), "off by a tenth: {level} {scale}");
            let half: Vec<f64> = (0..50).map(|k| level + (k as f64 + 0.5) * scale).collect();
            assert!(!sc.holds(&half), "off by half: {level} {scale}");
            // Below a tick, and negative: the residual's sign must not matter.
            let shifts = [
                |k: f64, s: f64, l: f64| l + (k - 0.1) * s,
                |k: f64, s: f64, l: f64| l + (k - 0.5) * s,
                |k: f64, s: f64, l: f64| -(l + (k + 0.1) * s),
                |k: f64, s: f64, l: f64| -(l + (k - 0.1) * s),
            ];
            for (n, f) in shifts.iter().enumerate() {
                let off: Vec<f64> = (1..50).map(|k| f(k as f64, scale, level)).collect();
                assert!(!sc.holds(&off), "shape {n}: {level} {scale}");
            }
        }
    }
    let sc = Scale::new(0.01).unwrap();
    assert!(sc.holds(&[0.0, 0.5, -3.0]));
    assert!(!sc.holds(&[0.001]) && !sc.holds(&[f64::NAN]) && !sc.holds(&[f64::INFINITY]));
    assert!(sc.holds(&[]));
}

#[test]
fn for_query_picks_the_stream_scale_only_on_its_grid() {
    let stream = Scale::new(0.01).unwrap();
    let on = [1.23e8, 1.23e8 + 0.03, 1.23e8 + 0.01, 1.23e8 + 0.07];
    assert_eq!(stream.for_query(&on), Some(stream));
    let off = [0.0, 0.0005, 0.0012, 0.0003];
    let own = stream.for_query(&off).unwrap();
    assert_ne!(own, stream);
    assert_eq!(
        Some(own),
        Scale::finest_fitting(&off, NonZeroUsize::new(4).unwrap())
    );
    assert_eq!(stream.for_query(&[]), None);
}

// ---------- Scale::fast_fitting ----------

/// The default rule straight from its definition, one window and one span at a time: at `scale`
/// every finite sample has a tick, at most one clean window in 16 holds an outlier, and every
/// other clean window's ticks span at most `2·⌊(2³¹ − 1)/m⌋`, as do the ticks of every `m + 128`
/// finite samples holding no outlier (of a whole run of finite samples, at least `m` long, when
/// it is shorter).
fn fast_rule_holds(series: &[f64], m: NonZeroUsize, scale: Scale) -> bool {
    let m = m.get();
    let mut ticks = Vec::new();
    let mut had = Vec::new();
    for &x in series {
        match to_ticks(x, scale) {
            Some(tick) => (ticks.push(tick), had.push(true)),
            None if x.is_finite() => return false,
            None => (ticks.push(0), had.push(false)),
        };
    }
    let outliers = crate::spill::outlier_ticks(&ticks, &had, m).unwrap();
    let reach = 2 * (((1_i64 << 31) - 1) / m as i64);
    let fits = |range: core::ops::Range<usize>| {
        outliers[range.clone()].iter().any(|&o| o) || {
            let high = ticks[range.clone()].iter().max().unwrap();
            let low = ticks[range].iter().min().unwrap();
            high - low <= reach
        }
    };
    let (mut clean, mut excused) = (0usize, 0usize);
    for row in 0..=series.len() - m {
        let window = row..row + m;
        if series[window.clone()].iter().any(|x| !x.is_finite()) {
            continue;
        }
        clean += 1;
        if outliers[window.clone()].iter().any(|&o| o) {
            excused += 1;
            continue;
        }
        if !fits(window) {
            return false;
        }
    }
    let span = m + 128;
    let mut start = 0;
    while start < series.len() {
        let end = (start..series.len())
            .find(|&p| !series[p].is_finite())
            .unwrap_or(series.len());
        let run = end - start;
        if run >= m && run < span && !fits(start..end) {
            return false;
        }
        if run >= span && !(start..=end - span).all(|a| fits(a..a + span)) {
            return false;
        }
        start = end + 1;
    }
    excused <= clean / 16
}

/// Every power of two at or above the finest at which `series`' largest finite sample has a
/// tick, finest first, up to and including `scale`.
fn powers_up_to(series: &[f64], scale: Scale) -> Vec<Scale> {
    let max_abs = series
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0_f64, |a, x| a.max(x.abs()));
    let mut value = smallest_power_of_two_at_least(max_abs / MAX_TICKS as f64);
    let mut out = Vec::new();
    while value <= scale.get() {
        out.push(Scale::new(value).unwrap());
        value *= 2.0;
    }
    out
}

/// A float walk or noise around `level`, with `spikes` as `(position, factor)` multiplying a
/// sample and `gaps` as positions made NaN.
fn shaped_series(
    level: f64,
    steps: &[f64],
    walk: bool,
    spikes: &[(usize, f64)],
    gaps: &[usize],
) -> Vec<f64> {
    let mut x = 0.0;
    let mut series: Vec<f64> = steps
        .iter()
        .map(|&step| {
            x = if walk { x + step } else { step };
            level + x * core::f64::consts::PI
        })
        .collect();
    let n = series.len();
    for &(at, factor) in spikes {
        series[at % n] *= factor;
    }
    for &at in gaps {
        series[at % n] = f64::NAN;
    }
    series
}

fn shaped() -> impl Strategy<Value = (NonZeroUsize, Vec<f64>)> {
    (
        2usize..10,
        prop::sample::select(std::vec![0.0, 1.0, 100.0, 1e6]),
        proptest::collection::vec(-1.0..1.0_f64, 40..400),
        any::<bool>(),
        proptest::collection::vec(
            (
                any::<usize>(),
                prop::sample::select(std::vec![1e3, -1e3, 1.5, 1e6]),
            ),
            0..4,
        ),
        proptest::collection::vec(any::<usize>(), 0..3),
    )
        .prop_map(|(m, level, steps, walk, spikes, gaps)| {
            let series = shaped_series(level, &steps, walk, &spikes, &gaps);
            (NonZeroUsize::new(m).unwrap(), series)
        })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// `fast_fitting` is the finest power of two the rule holds at: it holds there and at no
    /// finer power of two down to the finest at which every finite sample has a tick.
    #[test]
    fn fast_fitting_is_the_finest_power_of_two_the_rule_holds_at((m, series) in shaped()) {
        let scale = Scale::fast_fitting(&series, m).expect("finite samples present");
        prop_assert_eq!(Scale::fast_fitting(&series, m), Some(scale));
        prop_assert!(fast_rule_holds(&series, m, scale));
        for finer in powers_up_to(&series, scale) {
            if finer != scale {
                prop_assert!(!fast_rule_holds(&series, m, finer), "{}", finer.get());
            }
        }
    }

    /// Data on a grid keeps it: the opt-in and the default agree wherever a grid is found.
    #[test]
    fn infer_and_infer_finest_agree_on_grid_data(
        grid in prop::sample::select(std::vec![0.01_f64, 0.25, 0.05, 1.0, 100.0]),
        start in -1000_i64..1000,
        steps in proptest::collection::vec(-3_i64..=3, 40..120),
        m in 3usize..8,
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let series = grid_walk(grid, start, &steps);
        prop_assert_eq!(Scale::infer(&series, m), Scale::infer_finest(&series, m));
    }
}

/// A float walk of `len` steps from 100 with steps under 0.05, the same every run.
fn float_walk(len: usize) -> Vec<f64> {
    let mut state = 11_u64;
    let mut x = 100.0;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x += ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.1;
            x
        })
        .collect()
}

#[test]
fn a_bad_print_leaves_the_default_scale_alone() {
    let m = NonZeroUsize::new(64).unwrap();
    let walk = float_walk(4000);
    let clean = Scale::infer(&walk, m).unwrap();
    let mut spiked = walk.clone();
    spiked[1500] *= 1000.0;
    spiked[3000] *= -1000.0;
    assert_eq!(Scale::infer(&spiked, m), Some(clean));
    // The finest exact scale coarsens until the windows holding the print are exact.
    let finest = Scale::infer_finest(&spiked, m).unwrap();
    assert!(finest.get() >= 512.0 * clean.get(), "{}", finest.get());
}

#[test]
fn too_many_bad_prints_coarsen_the_default_scale() {
    let m = NonZeroUsize::new(16).unwrap();
    let walk = float_walk(4000);
    let clean = Scale::infer(&walk, m).unwrap();
    let mut spiked = walk.clone();
    // A print every 160 samples: 16 windows in 160, one in 10, hold one.
    for at in (80..walk.len()).step_by(160) {
        spiked[at] *= 1000.0;
    }
    let scale = Scale::infer(&spiked, m).unwrap();
    assert!(scale.get() > clean.get());
    let sparse = {
        let mut sparse = walk.clone();
        // A print every 400 samples: one window in 25 holds one.
        for at in (80..walk.len()).step_by(400) {
            sparse[at] *= 1000.0;
        }
        sparse
    };
    assert_eq!(Scale::infer(&sparse, m), Some(clean));
}

#[test]
fn noise_at_m_64_gets_one_bit_less_than_the_finest_exact_scale() {
    let m = NonZeroUsize::new(64).unwrap();
    let mut state = 7_u64;
    let noise: Vec<f64> = (0..20_000)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect();
    assert_eq!(Scale::infer(&noise, m), Scale::new(pow2(-25)));
    assert_eq!(Scale::infer_finest(&noise, m), Scale::new(pow2(-26)));
}

#[test]
fn fast_fitting_does_not_excuse_data_whose_every_move_is_an_outlier() {
    // At a fine enough scale every move lies past the outlier limit, and every window holds an
    // outlier; the share keeps the rule from taking that scale.
    for m in [2usize, 3, 4, 8] {
        let m = NonZeroUsize::new(m).unwrap();
        let alternating: Vec<f64> = (0..200)
            .map(|k| {
                if k % 2 == 0 {
                    0.0
                } else {
                    core::f64::consts::PI
                }
            })
            .collect();
        let scale = Scale::fast_fitting(&alternating, m).unwrap();
        assert!(fast_rule_holds(&alternating, m, scale));
        let ticks = to_ticks(core::f64::consts::PI, scale).unwrap();
        assert!(
            ticks <= 2 * ((1 << 31) - 1) / m.get() as i64,
            "m {m}: {ticks}"
        );
        assert!(ticks > ((1 << 31) - 1) / m.get() as i64, "m {m}: {ticks}");
    }
}

#[test]
fn fast_fitting_is_none_as_finest_fitting_is() {
    let m = NonZeroUsize::new(3).unwrap();
    assert_eq!(Scale::fast_fitting(&[1.0, 2.0], m), None);
    assert_eq!(Scale::fast_fitting(&[f64::NAN; 6], m), None);
    assert_eq!(Scale::fast_fitting(&[], m), None);
    assert!(Scale::fast_fitting(&[0.0; 6], m).is_some());
    let extremes = [
        f64::MAX,
        -f64::MAX,
        f64::MAX,
        0.0,
        f64::MIN_POSITIVE,
        5e-324,
    ];
    let scale = Scale::fast_fitting(&extremes, m).unwrap();
    assert!(fast_rule_holds(&extremes, m, scale));
}

#[test]
fn scale_choice_resolves_given_and_inferred() {
    let m = NonZeroUsize::new(4).unwrap();
    let walk = float_walk(200);
    let given = Scale::new(0.5).unwrap();
    assert_eq!(ScaleChoice::Given(given).resolve(&walk, m), Some(given));
    assert_eq!(ScaleChoice::from(given), ScaleChoice::Given(given));
    assert_eq!(ScaleChoice::from(None), ScaleChoice::Fast);
    assert_eq!(ScaleChoice::default(), ScaleChoice::Fast);
    assert_eq!(
        ScaleChoice::Fast.resolve(&walk, m),
        Scale::fast_fitting(&walk, m)
    );
    assert_eq!(
        ScaleChoice::Finest.resolve(&walk, m),
        Scale::finest_fitting(&walk, m)
    );
    assert_eq!(ScaleChoice::Finest.resolve(&walk[..3], m), None);
}

#[test]
fn span_slack_is_the_rules_constant() {
    assert_eq!(SPAN_SLACK, 128);
}

/// A trend whose windows fit a finer scale than its spans do: the span rule takes the coarser.
#[test]
fn spans_coarsen_a_trend_its_windows_alone_would_not() {
    for m in [16usize, 64, 100] {
        let m = NonZeroUsize::new(m).unwrap();
        let walk: Vec<f64> = float_walk(4000)
            .iter()
            .enumerate()
            .map(|(k, x)| x * 1e-3 + k as f64 * core::f64::consts::PI)
            .collect();
        let scale = Scale::fast_fitting(&walk, m).unwrap();
        let finer = Scale::new(scale.get() / 2.0).unwrap();
        assert!(fast_rule_holds(&walk, m, scale));
        assert!(!fast_rule_holds(&walk, m, finer));
        let ticks: Vec<i64> = walk.iter().map(|&x| to_ticks(x, finer).unwrap()).collect();
        let reach = 2 * (((1_i64 << 31) - 1) / m.get() as i64);
        let spread = |w: &[i64]| w.iter().max().unwrap() - w.iter().min().unwrap();
        assert!(ticks.windows(m.get()).all(|w| spread(w) <= reach), "m {m}");
        assert!(
            ticks.windows(m.get() + 128).any(|w| spread(w) > reach),
            "m {m}"
        );
    }
}

/// A run of finite samples between gaps shorter than `m + SPAN_SLACK` is one span, and a gap's
/// samples take no part in any span.
#[test]
fn a_short_run_between_gaps_is_one_span_and_gaps_are_outside_every_span() {
    let m = NonZeroUsize::new(8).unwrap();
    let mut series = float_walk(3000);
    for p in (0..series.len()).step_by(100) {
        series[p] = f64::NAN;
    }
    let scale = Scale::fast_fitting(&series, m).unwrap();
    assert!(fast_rule_holds(&series, m, scale));
    let finer = Scale::new(scale.get() / 2.0).unwrap();
    assert!(!fast_rule_holds(&series, m, finer));
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Joined inference: the scale inferred for series at very different levels fits each series
    /// on its own and is the finest that does, and the seam the grid is read across is at least a
    /// span wide (no wider than the longest series).
    #[test]
    fn a_joined_scale_fits_each_series_on_its_own(
        m in 2usize..12,
        a in proptest::collection::vec(-1.0..1.0_f64, 20..300),
        b in proptest::collection::vec(-1.0..1.0_f64, 20..300),
        level in prop::sample::select(std::vec![1e-3, 1.0, 1e4, 1e8]),
        walk in any::<bool>(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let a = shaped_series(0.0, &a, walk, &[], &[]);
        let b = shaped_series(level, &b, !walk, &[], &[]);
        let parts: [&[f64]; 2] = [&a, &b];
        let series = joined(&parts, m);
        let seam = series.len() - a.len() - b.len();
        prop_assert_eq!(seam, (m.get() + SPAN_SLACK).min(a.len().max(b.len())));
        prop_assert!(series[a.len()..a.len() + seam].iter().all(|x| x.is_nan()));
        let scale = ScaleChoice::Fast.resolve_each(&parts, m).unwrap();
        for part in parts {
            prop_assert!(fast_rule_holds(part, m, scale), "{}", scale.get());
        }
        let finer = Scale::new(scale.get() / 2.0).unwrap();
        let is_grid = candidates().contains(&scale.get())
            && parts.iter().all(|part| scale.holds(part));
        prop_assert!(is_grid || parts.iter().any(|part| !fast_rule_holds(part, m, finer)));
        prop_assert_eq!(ScaleChoice::Fast.resolve_each(&[&a], m), Scale::infer(&a, m));
    }
}

/// A span is exactly `m + SPAN_SLACK` samples: on a ramp whose `m + 128` samples just fail to fit
/// and whose `m + 127` just fit, the rule fails, and one tick of step less, it holds.
#[test]
fn a_span_is_m_plus_slack_samples_exactly() {
    for m in [4usize, 16, 64] {
        let mz = NonZeroUsize::new(m).unwrap();
        let reach = 2 * (((1_i64 << 31) - 1) / m as i64);
        let unit = Scale::new(1.0).unwrap();
        let ramp = |step: i64| -> Vec<f64> { (0..1000).map(|k| (k * step) as f64).collect() };
        let too_steep = reach / (m as i64 + 126);
        assert!((m as i64 + 127) * too_steep > reach);
        let fitting = reach / (m as i64 + 127);
        for (step, holds) in [(too_steep, false), (fitting, true)] {
            let series = ramp(step);
            let clean = clean_windows(&series, m);
            let mut scratch = super::FitScratch::default();
            let fits = super::fits_fast(&series, mz, unit, &clean, &mut scratch);
            assert_eq!(fits, holds, "m {m}, step {step}");
            assert_eq!(
                fast_rule_holds(&series, mz, unit),
                holds,
                "m {m}, step {step}"
            );
        }
    }
}
