extern crate std;

use super::{HIGH_MAGIC, HIGH_MAGIC_BITS, LOW_MAGIC_BITS, model};
use proptest::prelude::*;
use std::vec::Vec;

/// The values every full-range conversion is pinned on: zero and one, the 32-bit split's
/// boundaries, the fast conversion's and the exact-`f64` range's ends, both ends of `i64`, and,
/// above `2⁵³`, every halfway case (and its neighbors) at and around each power of two, where
/// rounding to nearest, ties to even, differs from truncation and from rounding ties away.
pub(crate) fn edges() -> Vec<i64> {
    let mut values = Vec::new();
    let mut both = |v: i64| {
        values.push(v);
        values.push(v.wrapping_neg());
    };
    for v in [0, 1, 2, 3] {
        both(v);
    }
    for p in [31, 32, 51, 52, 53] {
        for d in -1..=1 {
            both((1i64 << p) + d);
        }
    }
    for v in [i64::MAX, i64::MAX - 1, (1 << 62) + 1, u32::MAX as i64] {
        both(v);
    }
    // Around each power of two `2^p` above `2⁵³`: the spacing just above is `ulp = 2^(p−52)`,
    // just below half that. Multiples of a quarter of it on both sides reach every tie (odd and
    // even neighbors) and every value a quarter-spacing off one, each nudged by one.
    for p in 53..=62 {
        let ulp = 1i64 << (p - 52);
        let base = 1i64 << p;
        for k in -8..=8 {
            for d in -1..=1 {
                both(base + k * (ulp / 4).max(1) + d);
            }
        }
    }
    values.push(i64::MIN);
    values.push(i64::MIN + 1);
    values.push(i64::MIN + 2);
    values
}

/// A value of any magnitude: every exponent from `2⁰` to `2⁶³` equally often.
pub(crate) fn any_magnitude() -> impl Strategy<Value = i64> {
    prop_oneof![
        any::<i64>(),
        (any::<i64>(), 0u32..64).prop_map(|(v, s)| v >> s),
        Just(i64::MIN),
        Just(i64::MAX),
    ]
}

#[test]
fn the_constants_are_the_proved_values() {
    let two = |p: i32| 2f64.powi(p);
    assert_eq!(f64::from_bits(HIGH_MAGIC_BITS), two(84) + two(63));
    assert_eq!(f64::from_bits(LOW_MAGIC_BITS), two(52));
    assert_eq!(HIGH_MAGIC, two(84) + two(63) + two(52));
    // The sums above are exact: the three powers lie within 53 bits of one another.
    assert_eq!(HIGH_MAGIC - two(84) - two(63), two(52));
}

#[test]
fn the_model_is_as_f64_on_every_edge() {
    for x in edges() {
        assert_eq!(model(x).to_bits(), (x as f64).to_bits(), "{x}");
    }
}

#[test]
fn the_model_gives_positive_zero_for_zero() {
    assert_eq!(model(0).to_bits(), 0);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(20_000))]

    /// The model equals `x as f64` bit for bit on every `i64`.
    #[test]
    fn the_model_is_as_f64(x in any_magnitude()) {
        prop_assert_eq!(model(x).to_bits(), (x as f64).to_bits());
    }
}
