use super::{constant_row_rank, encode, r_bias, valid_row_rank, window_of};
use crate::stats::{Window, WindowKind};

fn window(kind: WindowKind) -> Window {
    Window { sum: 0, kind }
}

#[test]
fn r_bias_encodes_each_kind() {
    assert_eq!(
        r_bias(&window(WindowKind::Valid {
            variance_numerator: 4,
            inv_sqrt: 0.5,
        })),
        (0.5, 0.0)
    );
    assert_eq!(r_bias(&window(WindowKind::Constant)), (0.0, 0.5));
    assert_eq!(
        r_bias(&window(WindowKind::NonFinite)),
        (0.0, f64::NEG_INFINITY)
    );
}

#[test]
fn constant_row_rank_has_fixed_ratios() {
    // A constant neighbor: 1.0 exactly.
    assert_eq!(constant_row_rank(0.0, 0.5), 1.0);
    // A valid neighbor: 0.5 exactly, whatever `r` is.
    assert_eq!(constant_row_rank(0.25, 0.0), 0.5);
    // A non-finite neighbor is excluded.
    assert_eq!(constant_row_rank(0.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
}

#[test]
fn valid_row_rank_routes_through_rank_estimate() {
    // A Valid neighbor: `rank_estimate(n, ri, r)` directly, `bias` (`0.0`) adding nothing.
    let n: i64 = -68_661_172_291_171;
    let ri = 1.0 / libm::sqrt(834_911_684_991_698_f64);
    let r = 1.0 / libm::sqrt(275_720_901_775_960_f64);
    assert_eq!(
        valid_row_rank(n, ri, r, 0.0),
        crate::estimate::rank_estimate(n, ri, r)
    );
    // A Constant neighbor (`r = 0.0`, `bias = 0.5`): the product is exactly `0.0`
    // regardless of `n`'s sign, so the result is exactly `0.5`.
    assert_eq!(valid_row_rank(n, ri, 0.0, 0.5), 0.5);
    assert_eq!(valid_row_rank(-n, ri, 0.0, 0.5), 0.5);
    // A non-finite neighbor (`bias = −∞`): excluded regardless.
    assert_eq!(
        valid_row_rank(n, ri, 0.0, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
}

#[test]
fn encode_and_window_of_round_trip_every_kind() {
    for kind in [
        WindowKind::Valid {
            variance_numerator: 7,
            inv_sqrt: 0.25,
        },
        WindowKind::Constant,
        WindowKind::NonFinite,
    ] {
        let original = Window { sum: 42, kind };
        let (r, bias, variance) = encode(&original);
        assert_eq!(window_of(original.sum, r, bias, variance), original);
    }
}

#[cfg(not(any(loom, shuttle)))]
pub(crate) mod cross_range {
    extern crate std;

    use super::super::CrossRange;
    use crate::kernel_neon::Conversion;
    use crate::route::{ConversionForce, RouteForce, forced_route};
    use core::num::NonZeroUsize;
    use proptest::prelude::*;
    use std::vec::Vec;

    /// A window of `m = 8` ticks whose `V = m·Q − S²` is `2⁵¹ − 1`, and one of `m = 3` ticks whose
    /// `V` is `2⁵¹`. Every window that holds part of either, among zeros, has a `V` no larger.
    pub(crate) const JUST_INSIDE: [i64; 8] = [0, 0, 0, 0, 0, 0, 587_366, -17_842_265];
    pub(crate) const JUST_OUTSIDE: [i64; 3] = [0, 0, 1 << 25];

    fn nz(m: usize) -> NonZeroUsize {
        NonZeroUsize::new(m).unwrap()
    }

    /// `m·Q − S²` of one window, exactly, as the sum over pairs of their squared difference
    /// (Lagrange's identity); `None` past `u128`.
    fn variance(window: &[i64]) -> Option<u128> {
        let mut v = 0u128;
        for (i, &a) in window.iter().enumerate() {
            for &b in &window[i + 1..] {
                let d = u128::from(a.abs_diff(b));
                v = v.checked_add(d * d)?;
            }
        }
        Some(v)
    }

    /// The widest window directly, `u64::MAX` from `2⁶³` on, as the bound saturates.
    fn widest(ticks: &[i64], m: usize) -> u64 {
        ticks
            .windows(m)
            .map(|w| match variance(w) {
                Some(v) if v < 1 << 63 => v as u64,
                _ => u64::MAX,
            })
            .max()
            .unwrap_or(0)
    }

    /// `pattern` at both ends of a run of zeros longer than it: no window holds more than one
    /// copy, and the two windows that hold one whole repeat each other exactly.
    pub(crate) fn twice(pattern: &[i64]) -> Vec<i64> {
        let mut ticks = pattern.to_vec();
        ticks.extend(core::iter::repeat_n(0, 20));
        ticks.extend_from_slice(pattern);
        ticks
    }

    #[test]
    fn the_edge_windows_are_where_they_say() {
        assert_eq!(variance(&JUST_INSIDE), Some((1 << 51) - 1));
        assert_eq!(variance(&JUST_OUTSIDE), Some(1 << 51));
        assert_eq!(widest(&twice(&JUST_INSIDE), 8), (1 << 51) - 1);
        assert_eq!(widest(&twice(&JUST_OUTSIDE), 3), 1 << 51);
    }

    #[test]
    fn the_fast_conversion_fits_just_below_two_to_the_fifty_one() {
        let inside = CrossRange::over(&twice(&JUST_INSIDE), nz(8));
        let outside = CrossRange::over(&twice(&JUST_OUTSIDE), nz(3));
        assert!(inside.fits_fast_conversion());
        assert!(!outside.fits_fast_conversion());
        assert_eq!(inside.widest, (1 << 51) - 1);
        assert_eq!(outside.widest, 1 << 51);
    }

    #[test]
    fn constant_windows_and_extreme_window_lengths() {
        assert_eq!(CrossRange::over(&[7i64; 50], nz(5)).widest, 0);
        let spread: Vec<i64> = (0..50).map(|k| k * k * 1_000_003).collect();
        // One tick per window: every window is constant.
        assert_eq!(CrossRange::over(&spread, nz(1)).widest, 0);
        // One window, the whole series.
        assert_eq!(
            CrossRange::over(&spread, nz(50)).widest,
            widest(&spread, 50)
        );
        // No window at all.
        assert_eq!(CrossRange::over(&spread, nz(51)).widest, 0);
        assert!(CrossRange::over(&[], nz(3)).fits_fast_conversion());
    }

    #[test]
    fn a_window_past_two_to_the_sixty_three_saturates() {
        let range = CrossRange::over(&[i64::MIN / 2, i64::MAX / 2, 0, 1], nz(2));
        assert_eq!(range.widest, u64::MAX);
        assert!(!range.fits_fast_conversion());
        assert!(!CrossRange::UNKNOWN.fits_fast_conversion());
    }

    #[test]
    fn a_join_takes_the_wider_side() {
        let inside = CrossRange::over(&twice(&JUST_INSIDE), nz(8));
        let outside = CrossRange::over(&twice(&JUST_OUTSIDE), nz(3));
        let flat = CrossRange::over(&[3; 20], nz(8));
        assert_eq!(inside.joint(outside), outside);
        assert_eq!(outside.joint(inside), outside);
        assert_eq!(flat.joint(inside), inside);
        assert!(!inside.joint(outside).fits_fast_conversion());
    }

    #[test]
    fn the_conversion_is_fast_exactly_where_the_bound_fits() {
        let _natural = forced_route(RouteForce::default());
        let inside = CrossRange::over(&twice(&JUST_INSIDE), nz(8));
        let outside = CrossRange::over(&twice(&JUST_OUTSIDE), nz(3));
        assert_eq!(inside.conversion(), Conversion::FAST);
        assert_eq!(outside.conversion(), Conversion::FULL_RANGE);
        assert_eq!(CrossRange::UNKNOWN.conversion(), Conversion::FULL_RANGE);
    }

    #[test]
    fn a_forced_conversion_is_never_fast_where_the_bound_does_not_fit() {
        let inside = CrossRange::over(&twice(&JUST_INSIDE), nz(8));
        let outside = CrossRange::over(&twice(&JUST_OUTSIDE), nz(3));
        {
            let guard = forced_route(RouteForce {
                conversion: ConversionForce::FullRange,
                ..RouteForce::default()
            });
            assert_eq!(inside.conversion(), Conversion::FULL_RANGE);
            assert_eq!(outside.conversion(), Conversion::FULL_RANGE);
            assert_eq!(CrossRange::UNKNOWN.conversion(), Conversion::FULL_RANGE);
            let fired = guard.fired();
            assert_eq!(fired.conversions_forced_full_range, 1);
            assert_eq!(fired.fast_conversions_refused, 0);
        }
        let guard = forced_route(RouteForce {
            conversion: ConversionForce::Fast,
            ..RouteForce::default()
        });
        assert_eq!(inside.conversion(), Conversion::FAST);
        assert_eq!(outside.conversion(), Conversion::FULL_RANGE);
        let fired = guard.fired();
        assert_eq!(fired.conversions_forced_full_range, 0);
        assert_eq!(fired.fast_conversions_refused, 1);
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]

        /// The bound is the widest window's exact `V`, on ticks from a few levels apart to the
        /// whole `i64` range.
        #[test]
        fn the_bound_is_the_widest_window(
            ticks in prop::collection::vec(prop_oneof![
                -4i64..4,
                -(1i64 << 30)..(1i64 << 30),
                any::<i64>(),
            ], 0..60),
            m in 1usize..12,
        ) {
            prop_assert_eq!(CrossRange::over(&ticks, nz(m)).widest, widest(&ticks, m));
        }
    }
}
