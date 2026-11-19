//! A high-precision reference for window statistics, so an f64 implementation's rounding can be
//! judged against it. Sums, means and centered sums of squares are accurate to about 2⁻¹⁰⁰
//! relative; centered cross products, whose exact value can be 0, to 2⁻¹⁰⁰·√(N_aa·N_bb).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// An unevaluated sum `hi + lo` with `|lo| ≤ ulp(hi) / 2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

impl DoubleDouble {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// Nearest f64 to the represented value.
    pub fn to_f64(self) -> f64 {
        self.hi
    }

    /// One Newton step from the f64 root.
    fn sqrt(self) -> Self {
        let root = self.hi.sqrt();
        if root == 0.0 || !root.is_finite() {
            return root.into();
        }
        let residual = self - DoubleDouble::from(root) * DoubleDouble::from(root);
        DoubleDouble::from(root) + residual / (2.0 * root)
    }
}

impl From<f64> for DoubleDouble {
    fn from(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }
}

/// Knuth's TwoSum: `sum + error == a + b` exactly.
fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let sum = a + b;
    let b_part = sum - a;
    let a_part = sum - b_part;
    let error = (a - a_part) + (b - b_part);
    DoubleDouble { hi: sum, lo: error }
}

/// TwoSum for `|a| ≥ |b|`.
fn fast_two_sum(a: f64, b: f64) -> DoubleDouble {
    let sum = a + b;
    let error = b - (sum - a);
    DoubleDouble { hi: sum, lo: error }
}

/// Dekker's split into two halves of at most 26 significant bits each, so their products are exact.
fn split(value: f64) -> (f64, f64) {
    const SPLITTER: f64 = (1u64 << 27) as f64 + 1.0;
    let scaled = SPLITTER * value;
    let high = scaled - (scaled - value);
    (high, value - high)
}

/// Dekker's TwoProduct: `product + error == a · b` exactly, without fused multiply-add.
fn two_product(a: f64, b: f64) -> DoubleDouble {
    let product = a * b;
    let (a_high, a_low) = split(a);
    let (b_high, b_low) = split(b);
    let error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low;
    DoubleDouble {
        hi: product,
        lo: error,
    }
}

impl Add for DoubleDouble {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let high = two_sum(self.hi, other.hi);
        let low = two_sum(self.lo, other.lo);
        let partial = fast_two_sum(high.hi, high.lo + low.hi);
        fast_two_sum(partial.hi, partial.lo + low.lo)
    }
}

impl Neg for DoubleDouble {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for DoubleDouble {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl Mul for DoubleDouble {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let product = two_product(self.hi, other.hi);
        let cross = self.hi * other.lo + self.lo * other.hi;
        fast_two_sum(product.hi, product.lo + cross)
    }
}

impl Div<f64> for DoubleDouble {
    type Output = Self;

    fn div(self, divisor: f64) -> Self {
        let first = self.hi / divisor;
        let remainder = self - two_product(first, divisor);
        let second = remainder.hi / divisor;
        fast_two_sum(first, second)
    }
}

pub fn sum(values: &[f64]) -> DoubleDouble {
    values
        .iter()
        .fold(DoubleDouble::ZERO, |total, &value| total + value.into())
}

pub fn mean(values: &[f64]) -> DoubleDouble {
    sum(values) / values.len() as f64
}

/// `Σ (a[k] − μ_a)(b[k] − μ_b)`, with μ in double-double.
pub fn centered_cross(a: &[f64], b: &[f64]) -> DoubleDouble {
    assert_eq!(a.len(), b.len(), "series lengths differ");
    let (a_mean, b_mean) = (mean(a), mean(b));
    a.iter()
        .zip(b)
        .fold(DoubleDouble::ZERO, |total, (&a_value, &b_value)| {
            total + (DoubleDouble::from(a_value) - a_mean) * (DoubleDouble::from(b_value) - b_mean)
        })
}

/// Population σ (ddof = 0, as numpy and stumpy), faithfully rounded to f64.
pub fn std(values: &[f64]) -> f64 {
    (centered_cross(values, values) / values.len() as f64)
        .sqrt()
        .to_f64()
}

#[cfg(test)]
mod tests {
    use super::{DoubleDouble, centered_cross, mean, std, sum, two_product};

    /// Deterministic integers in `[-bound, bound]`.
    fn integers(seed: u64, count: usize, bound: i64) -> Vec<i64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) % (2 * bound as u64 + 1)) as i64 - bound
            })
            .collect()
    }

    fn exact(value: DoubleDouble) -> i128 {
        assert_eq!(value.hi.fract(), 0.0);
        assert_eq!(value.lo.fract(), 0.0);
        value.hi as i128 + value.lo as i128
    }

    #[test]
    fn two_product_is_exact() {
        let near_one = 1.0 + 2f64.powi(-30);
        let square = two_product(near_one, near_one);
        assert_eq!(square.hi, 1.0 + 2f64.powi(-29));
        assert_eq!(square.lo, 2f64.powi(-60));

        let factors = integers(1, 200, (1 << 53) - 1);
        for pair in factors.chunks(2) {
            let (a, b) = (pair[0], pair[1]);
            let product = two_product(a as f64, b as f64);
            assert_eq!(exact(product), a as i128 * b as i128, "{a} × {b}");
        }
    }

    #[test]
    fn sum_cancels_exactly() {
        assert_eq!(sum(&[1e16, 1.0, -1e16]), DoubleDouble::from(1.0));
        assert_eq!(sum(&[1.0, 1e16, -1e16]), DoubleDouble::from(1.0));
        let tiny = 2f64.powi(-80);
        assert_eq!(sum(&[1.0, tiny, -1.0]), DoubleDouble::from(tiny));

        for seed in 0..20 {
            let large = integers(seed, 500, 1 << 62);
            let small = integers(seed + 100, 500, 1000);
            let mut values: Vec<f64> = Vec::new();
            for (&big, &little) in large.iter().zip(&small) {
                values.extend([big as f64, little as f64]);
            }
            values.extend(large.iter().rev().map(|&big| -(big as f64)));
            let expected: i128 = values.iter().map(|&value| value as i128).sum();
            assert_eq!(
                expected,
                small.iter().map(|&little| little as i128).sum::<i128>()
            );
            assert_eq!(exact(sum(&values)), expected, "seed {seed}");
        }
    }

    #[test]
    fn mean_divides_in_double_double() {
        let third = mean(&[1.0, 0.0, 0.0]);
        let residual = DoubleDouble::from(1.0) - third * DoubleDouble::from(3.0);
        assert!(residual.hi.abs() <= 2f64.powi(-100));
    }

    fn from_exact(value: i128) -> DoubleDouble {
        let hi = value as f64;
        let lo = (value - hi as i128) as f64;
        DoubleDouble { hi, lo }
    }

    /// `n·Σab − Σa·Σb`, which is `n` times the centered cross product.
    fn scaled_centered_cross(a: &[i64], b: &[i64]) -> i128 {
        let n = a.len() as i128;
        let a_sum: i128 = a.iter().map(|&v| v as i128).sum();
        let b_sum: i128 = b.iter().map(|&v| v as i128).sum();
        let product_sum: i128 = a.iter().zip(b).map(|(&x, &y)| x as i128 * y as i128).sum();
        n * product_sum - a_sum * b_sum
    }

    fn assert_centered_cross_matches_exact(a: &[i64], b: &[i64]) {
        let to_f64 = |values: &[i64]| -> Vec<f64> { values.iter().map(|&v| v as f64).collect() };
        let n = DoubleDouble::from(a.len() as f64);
        let error =
            centered_cross(&to_f64(a), &to_f64(b)) * n - from_exact(scaled_centered_cross(a, b));
        let cauchy_schwarz_bound =
            (scaled_centered_cross(a, a) as f64 * scaled_centered_cross(b, b) as f64).sqrt();
        assert!(error.hi.abs() <= 2f64.powi(-100) * cauchy_schwarz_bound);
    }

    #[test]
    fn centered_cross_offset() {
        for offset in [0, 100_000_000] {
            let shift =
                |values: Vec<i64>| -> Vec<i64> { values.into_iter().map(|v| v + offset).collect() };
            let a = shift(integers(7, 999, 1000));
            let b = shift(integers(8, 999, 1000));
            assert_centered_cross_matches_exact(&a, &b);
            assert_centered_cross_matches_exact(&a, &a);
        }
    }

    #[test]
    fn std_is_population_sigma() {
        assert_eq!(std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 2.0);
        assert_eq!(std(&[0.0, 2.0]), 1.0);
        assert_eq!(
            std(&[1e8 + 2.0, 1e8 + 4.0, 1e8 + 6.0]),
            (8.0f64 / 3.0).sqrt()
        );
    }

    #[test]
    fn sqrt_rounds_the_full_value() {
        let root_above_midpoint = DoubleDouble {
            hi: 1.0 + f64::EPSILON,
            lo: f64::EPSILON * f64::EPSILON,
        };
        assert_eq!(root_above_midpoint.sqrt().to_f64(), 1.0 + f64::EPSILON);
    }

    #[test]
    fn add_keeps_both_low_parts() {
        let a = DoubleDouble {
            hi: 1.0,
            lo: 2f64.powi(-60),
        };
        let b = DoubleDouble {
            hi: -1.0,
            lo: 2f64.powi(-120),
        };
        assert_eq!(
            a + b,
            DoubleDouble {
                hi: 2f64.powi(-60),
                lo: 2f64.powi(-120)
            }
        );
    }
}
