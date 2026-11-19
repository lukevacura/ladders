//! The exact sign of a signed sum of the distances a mean adds up.
//!
//! A non-normalized distance is `√R` for an integer `R`. A z-normalized one is
//! `√(2m(1 − ρ))` with `ρ = s·N/√D` for integers `N`, `D`: a plain root `√R / c` when `√D` is an
//! integer (or `N` is zero), a difference of two plain roots when `D(D − N²)` is a square, and
//! otherwise a nested root. [`Terms`] holds the signed sum as plain parts `±√R / c` and nested
//! roots, and [`Terms::sign`] decides its sign.
//!
//! **Method.** Every part and nested root is enclosed between two integers at a precision `p`
//! (scaled by `2^p`, by integer square roots and divisions that round outward), so the sum is
//! enclosed too. The sign is known when the enclosure excludes zero; otherwise `p` doubles.
//!
//! **Why it ends.** If the sum is not zero the enclosure's width, at most `2·(terms)/2^p`,
//! eventually falls below its distance from zero. If it is zero, two cases stop the loop:
//! - plain parts only: parts whose product of radicands is a square share a class, the square
//!   roots of different classes are independent over the rationals, so the sum is zero exactly
//!   when every class's rational coefficient sums to zero, which integer arithmetic decides;
//! - with nested roots: every term times the product `C` of the divisors is an algebraic integer
//!   (a nested root `√(2m − 2mN/√D)` times `D` satisfies `y⁴ − 4mD²y² + 4m²(D − N²)D³ = 0`),
//!   all conjugates of the sum are below `T·C·M` for `T` terms, and a nonzero algebraic integer
//!   of degree at most `Δ` has norm at least one, so the sum is at least
//!   `1 / (C·(T·C·M)^(Δ − 1))` in magnitude, and `p` past that bit count means zero. `Δ` is at
//!   most `2^(parts + 2·nested)`, so the bound is reached in practice only for few nested roots;
//!   a sum of many of them that is zero without cancelling term by term (which the caller removes
//!   first) would cost doubly exponentially much.
//!
//! **Caps.** Refinement is capped, so that no input costs unbounded time:
//! - a sum with nested roots stops at [`PRECISION_BUDGET`] = 2^14 bits, if its proved bound is
//!   past that;
//! - a sum of plain roots stops at `PLAIN_PRECISION_LIMIT` = 2^16 bits.
//!
//! A sum still undecided at its cap is reported as a tie (`Equal`), which the caller breaks by
//! the smaller window start: deterministic, the same on every thread count and tile shape, and
//! counted by `count::budget_fallbacks`. The plain cap also asserts in a debug build, since a
//! sum of plain roots is separated from zero long before it (reaching it means broken
//! arithmetic); the nested cap is a counted tie in every build, because a nested sum that is
//! exactly zero without cancelling term by term is decided only by running to its bound.
//!
//! Within the caps every sum is decided exactly, and a cap changes a result only for two distinct
//! sums that agree to 2^14 bits (nested) or 2^16 bits (plain) of their size. That is provably
//! unreachable for non-normalized means of up to 4 dimensions and for about 3 non-cancelled
//! z-normalized distances (the proved lower bound on such a sum's distance from zero needs fewer bits than the cap).
//! Beyond that it rests on the sum-of-square-roots gap conjecture: no construction of two such
//! sums that close is known, and the counter says if one ever arises from window distances.
//!
//! All arithmetic is in pooled integers, so a decision allocates only until the pool has grown
//! to its size.

use alloc::vec::Vec;
use core::cmp::Ordering;

use super::big::{Arith, Big};
use crate::nearest::ExactCorrelation;
use crate::spill::{Spilled, WideCorrelation};
use crate::wide::Wide;

/// `±√radicand / divisor`.
#[derive(Debug)]
struct Part {
    negative: bool,
    radicand: Big,
    divisor: Big,
}

/// `±√(2m(1 − ρ))` with `ρ = ±cross / √denominator`, irrational and not a sum of plain roots.
#[derive(Debug)]
struct Nested {
    negative: bool,
    rho_negative: bool,
    window: u64,
    cross: Big,
    denominator: Big,
}

/// The precision a refinement starts at, in bits after the point.
const FIRST_PRECISION: u64 = 64;

/// The precision past which a sum with nested roots is no longer refined, in bits after the point.
/// The bound of a sum of two nested roots (one against one) is about 5 000 bits at the widest
/// denominators, so every such sum is decided exactly; three or more may fall back to a tie.
pub(crate) const PRECISION_BUDGET: u64 = 1 << 14;

/// The precision past which a sum of plain roots that is not exactly zero is taken to be
/// beyond refining: no such sum is within `2^-65536` of zero, so this only ends a loop whose
/// arithmetic is broken.
const PLAIN_PRECISION_LIMIT: u64 = 1 << 16;

/// A signed sum of exact distances, and the pooled integers that decide its sign.
#[derive(Debug, Default)]
pub struct Terms {
    arith: Arith,
    parts: Vec<Part>,
    nested: Vec<Nested>,
}

impl Terms {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Forgets the terms, keeping the pooled numbers.
    pub(crate) fn clear(&mut self) {
        for part in self.parts.drain(..) {
            self.arith.put(part.radicand);
            self.arith.put(part.divisor);
        }
        for nested in self.nested.drain(..) {
            self.arith.put(nested.cross);
            self.arith.put(nested.denominator);
        }
    }

    fn push_part(&mut self, negative: bool, radicand: Big, divisor: Big) {
        if radicand.is_zero() {
            self.arith.put(radicand);
            self.arith.put(divisor);
        } else {
            self.parts.push(Part {
                negative,
                radicand,
                divisor,
            });
        }
    }

    /// `±√radicand`.
    pub(crate) fn push_root(&mut self, negative: bool, radicand: u128) {
        let radicand = self.arith.get_u128(radicand);
        let divisor = self.arith.get_u128(1);
        self.push_part(negative, radicand, divisor);
    }

    /// `±value · numerator / denominator` for a positive normal `value`, to compare a computed
    /// distance with the exact one.
    #[cfg(test)]
    pub(crate) fn push_scaled(
        &mut self,
        negative: bool,
        value: f64,
        numerator: u64,
        denominator: u64,
    ) {
        let bits = value.to_bits();
        let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
        let exponent = ((bits >> 52) & 0x7ff) as i64 - 1075;
        let mut scaled = self
            .arith
            .get_u128(u128::from(mantissa) * u128::from(numerator));
        let mut radicand = self.arith.get();
        radicand.mul(&scaled, &scaled);
        scaled.set_u128(u128::from(denominator));
        if exponent >= 0 {
            radicand.shl(2 * exponent as u64);
        } else {
            scaled.shl(exponent.unsigned_abs());
        }
        self.push_part(negative, radicand, scaled);
    }

    /// `±` the z-normalized distance of a pair of windows of length `window` with `correlation`.
    pub(crate) fn push_znorm(
        &mut self,
        negative: bool,
        window: usize,
        correlation: ExactCorrelation,
    ) {
        let m = window as u128;
        match correlation {
            ExactCorrelation::BothConstant => {}
            ExactCorrelation::OneConstant => self.push_root(negative, m),
            ExactCorrelation::Valid {
                sign,
                cross_squared,
                denominator,
                ..
            } => {
                let cross = self.arith.get_u128(cross_squared.isqrt());
                let denominator = self.arith.get_u128(denominator);
                self.push_correlated(negative, window, sign < 0, cross, denominator);
            }
        }
    }

    /// [`Self::push_znorm`] for a correlation the cold route holds on 256 bits.
    pub(crate) fn push_znorm_wide(
        &mut self,
        negative: bool,
        window: usize,
        correlation: WideCorrelation,
    ) {
        let m = window as u128;
        match correlation {
            Spilled::BothConstant => {}
            Spilled::OneConstant => self.push_root(negative, m),
            Spilled::Valid {
                n,
                row_variance,
                neighbor_variance,
            } => {
                let cross = self.big_of(&n.unsigned_abs());
                let (row, neighbor) = (self.big_of(&row_variance), self.big_of(&neighbor_variance));
                let mut denominator = self.arith.get();
                denominator.mul(&row, &neighbor);
                self.arith.put(row);
                self.arith.put(neighbor);
                self.push_correlated(negative, window, n.is_negative(), cross, denominator);
            }
        }
    }

    /// `value`, nonnegative, as a pooled integer.
    fn big_of(&mut self, value: &Wide<4>) -> Big {
        let mut big = self.arith.get();
        let mut limb = self.arith.get();
        for &word in value.limbs().iter().rev() {
            big.shl(64);
            limb.set_u128(u128::from(word));
            big.add(&limb);
        }
        self.arith.put(limb);
        big
    }

    /// `±√(2m(1 − ρ))` for windows of length `window`, `ρ = ±cross/√denominator` (negative when
    /// `rho_negative`): a plain root, a difference of two, or a nested root.
    fn push_correlated(
        &mut self,
        negative: bool,
        window: usize,
        rho_negative: bool,
        cross: Big,
        denominator: Big,
    ) {
        let m = window as u128;
        if cross.is_zero() {
            self.arith.put(cross);
            self.arith.put(denominator);
            return self.push_root(negative, 2 * m);
        }
        let mut root = self.arith.get();
        let mut product = self.arith.get();
        if self.arith.exact_sqrt(&mut root, &denominator) {
            // ρ = ±cross/root, so the distance is √(2m·root·(root ∓ cross)) / root.
            let mut factor = self.arith.get();
            factor.set(&root);
            if rho_negative {
                factor.add(&cross);
            } else {
                factor.sub(&cross);
            }
            let scale = self.arith.get_u128(2 * m);
            product.mul(&scale, &root);
            let mut radicand = self.arith.get();
            radicand.mul(&product, &factor);
            for big in [product, factor, scale, cross, denominator] {
                self.arith.put(big);
            }
            return self.push_part(negative, radicand, root);
        }
        let mut gap = self.arith.get();
        gap.mul(&cross, &cross);
        let mut under = self.arith.get();
        under.set(&denominator);
        under.sub(&gap);
        product.mul(&denominator, &under);
        let mut sqrt = self.arith.get();
        if !self.arith.exact_sqrt(&mut sqrt, &product) {
            for big in [root, product, gap, under, sqrt] {
                self.arith.put(big);
            }
            self.nested.push(Nested {
                negative,
                rho_negative,
                window: window as u64,
                cross,
                denominator,
            });
            return;
        }
        // With P = √(D(D − N²)): the distance is √x ∓ √y, x = m(D + P)/D, y = m(D − P)/D.
        let (mut plus, mut minus) = (self.arith.get(), self.arith.get());
        under.set(&denominator);
        gap.set(&under);
        gap.add(&sqrt);
        plus.mul(&under, &gap);
        gap.set(&under);
        gap.sub(&sqrt);
        minus.mul(&under, &gap);
        let factor = self.arith.get_u128(m);
        let mut first = self.arith.get();
        let mut second = self.arith.get();
        first.mul(&plus, &factor);
        second.mul(&minus, &factor);
        let mut other_divisor = self.arith.get();
        other_divisor.set(&denominator);
        for big in [root, under, gap, product, sqrt, plus, minus, factor, cross] {
            self.arith.put(big);
        }
        self.push_part(negative, first, denominator);
        self.push_part(negative ^ !rho_negative, second, other_divisor);
    }

    /// The sign of the sum: `Less` if it is below zero.
    pub(crate) fn sign(&mut self) -> Ordering {
        if self.parts.is_empty() && self.nested.is_empty() {
            return Ordering::Equal;
        }
        let cap = self.zero_bound();
        // The loop ends: a sum of plain roots is separated from zero long before the plain limit,
        // and a sum with nested roots stops at its bound or the budget. Reaching the plain limit
        // with a sum that is not zero means the arithmetic is broken, which a debug build reports
        // and a release build ends as a counted tie rather than spin.
        let limit = if self.nested.is_empty() {
            cap.min(PLAIN_PRECISION_LIMIT)
        } else {
            cap.min(PRECISION_BUDGET)
        };
        let mut precision = FIRST_PRECISION;
        let mut tested_zero = false;
        loop {
            if let Some(order) = self.enclosed_sign(precision) {
                return order;
            }
            if self.nested.is_empty() && !tested_zero {
                tested_zero = true;
                if self.parts_cancel() {
                    return Ordering::Equal;
                }
            }
            if precision >= limit {
                debug_assert!(
                    !self.nested.is_empty(),
                    "a sum of plain roots that is not zero was not separated from zero"
                );
                if self.nested.is_empty() || cap > PRECISION_BUDGET {
                    super::count::budget_fallback();
                }
                return Ordering::Equal;
            }
            precision *= 2;
        }
    }

    /// The precision past which an enclosure around zero means the sum is zero, for a sum with
    /// nested roots; `u64::MAX` when no precision a machine could use is enough.
    fn zero_bound(&self) -> u64 {
        let terms = (self.parts.len() + self.nested.len()) as u64;
        let degree_bits = self.parts.len() as u64 + 2 * self.nested.len() as u64;
        if degree_bits >= 40 {
            return u64::MAX;
        }
        let mut divisor_bits = 0u64;
        let mut magnitude_bits = 0u64;
        for part in &self.parts {
            divisor_bits += part.divisor.bit_len();
            magnitude_bits = magnitude_bits.max(part.radicand.bit_len().div_ceil(2));
        }
        for nested in &self.nested {
            let bits = nested.denominator.bit_len();
            divisor_bits += bits;
            magnitude_bits =
                magnitude_bits.max(bits + 2 + u64::from(64 - nested.window.leading_zeros()));
        }
        let conjugate_bits = divisor_bits + magnitude_bits + (64 - terms.leading_zeros()) as u64;
        ((1u64 << degree_bits).saturating_mul(conjugate_bits)).saturating_add(divisor_bits + 1)
    }

    /// The sign if the enclosure of the sum at `precision` bits excludes zero.
    fn enclosed_sign(&mut self, precision: u64) -> Option<Ordering> {
        let arith = &mut self.arith;
        let mut sums = [arith.get(), arith.get(), arith.get(), arith.get()];
        let [plus_low, plus_high, minus_low, minus_high] = &mut sums;
        let (mut low, mut high) = (arith.get(), arith.get());
        let (mut scratch, mut other, mut third) = (arith.get(), arith.get(), arith.get());
        for part in &self.parts {
            // ⌊√(R·4^p)⌋ over the divisor, rounded outward.
            scratch.set(&part.radicand);
            scratch.shl(2 * precision);
            arith.isqrt(&mut other, &scratch);
            arith.div_rem(&mut low, &mut third, &other, &part.divisor);
            other.add(&part.divisor);
            arith.div_rem(&mut high, &mut third, &other, &part.divisor);
            if part.negative {
                minus_low.add(&low);
                minus_high.add(&high);
            } else {
                plus_low.add(&low);
                plus_high.add(&high);
            }
        }
        for nested in &self.nested {
            enclose_nested(arith, nested, precision, &mut low, &mut high);
            if nested.negative {
                minus_low.add(&low);
                minus_high.add(&high);
            } else {
                plus_low.add(&low);
                plus_high.add(&high);
            }
        }
        let order = if plus_low.compare(minus_high) == Ordering::Greater {
            Some(Ordering::Greater)
        } else if plus_high.compare(minus_low) == Ordering::Less {
            Some(Ordering::Less)
        } else {
            None
        };
        for big in sums.into_iter().chain([low, high, scratch, other, third]) {
            arith.put(big);
        }
        order
    }

    /// Whether the plain parts sum to exactly zero: by class, every class's rational coefficient
    /// is zero.
    fn parts_cancel(&mut self) -> bool {
        let arith = &mut self.arith;
        let mut representatives: Vec<usize> = Vec::new();
        // Per class: the sums of `q/c` over its positive and over its negative parts, as
        // numerator and denominator pairs.
        let mut sums: Vec<[Big; 4]> = Vec::new();
        let (mut product, mut root, mut numerator, mut denominator, mut scratch) = (
            arith.get(),
            arith.get(),
            arith.get(),
            arith.get(),
            arith.get(),
        );
        for (index, part) in self.parts.iter().enumerate() {
            let mut class = None;
            let mut cross = arith.get();
            for (c, &representative) in representatives.iter().enumerate() {
                product.mul(&part.radicand, &self.parts[representative].radicand);
                if arith.exact_sqrt(&mut root, &product) {
                    cross.set(&root);
                    class = Some(c);
                    break;
                }
            }
            let class = class.unwrap_or_else(|| {
                representatives.push(index);
                cross.set(&part.radicand);
                let (zero, one) = (arith.get(), arith.get_u128(1));
                let (zero_b, one_b) = (arith.get(), arith.get_u128(1));
                sums.push([zero, one, zero_b, one_b]);
                representatives.len() - 1
            });
            // sums[class][side] += cross / divisor: n/d + q/c = (n·c + q·d)/(d·c).
            let side = if part.negative { 2 } else { 0 };
            let [num, den] = &mut sums[class][side..side + 2] else {
                unreachable!("a class holds four numbers");
            };
            numerator.mul(num, &part.divisor);
            scratch.mul(&cross, den);
            numerator.add(&scratch);
            denominator.mul(den, &part.divisor);
            core::mem::swap(num, &mut numerator);
            core::mem::swap(den, &mut denominator);
            arith.put(cross);
        }
        let mut zero = true;
        for [pos_num, pos_den, neg_num, neg_den] in &sums {
            product.mul(pos_num, neg_den);
            scratch.mul(neg_num, pos_den);
            zero &= product.compare(&scratch) == Ordering::Equal;
        }
        for big in [product, root, numerator, denominator, scratch] {
            arith.put(big);
        }
        for class in sums {
            for big in class {
                arith.put(big);
            }
        }
        zero
    }
}

/// `low ≤ term · 2^precision ≤ high` for `√(2m(1 − ρ))`, `ρ = ±cross/√D`.
fn enclose_nested(
    arith: &mut Arith,
    nested: &Nested,
    precision: u64,
    low: &mut Big,
    high: &mut Big,
) {
    let (mut under, mut root_low, mut root_high) = (arith.get(), arith.get(), arith.get());
    let (mut numerator, mut remainder) = (arith.get(), arith.get());
    let (mut ratio_low, mut ratio_high, mut one) = (arith.get(), arith.get(), arith.get());
    // √D at scale 2^p, then |ρ| at scale 2^p, both rounded outward.
    under.set(&nested.denominator);
    under.shl(2 * precision);
    arith.isqrt(&mut root_low, &under);
    root_high.set(&root_low);
    root_high.add_one();
    numerator.set(&nested.cross);
    numerator.shl(2 * precision);
    arith.div_rem(&mut ratio_low, &mut remainder, &numerator, &root_high);
    arith.div_rem(&mut ratio_high, &mut remainder, &numerator, &root_low);
    if !remainder.is_zero() {
        ratio_high.add_one();
    }
    one.set_u128(1);
    one.shl(precision);
    for ratio in [&mut ratio_low, &mut ratio_high] {
        if ratio.compare(&one) == Ordering::Greater {
            ratio.set(&one);
        }
    }
    // 2m(1 ∓ |ρ|) at scale 2^p: its square root at scale 2^p is √(that · 2^p).
    let mut factor = arith.get_u128(2 * u128::from(nested.window));
    let (bound_low, bound_high) = if nested.rho_negative {
        (&ratio_low, &ratio_high)
    } else {
        (&ratio_high, &ratio_low)
    };
    for (bound, out) in [(bound_low, &mut *low), (bound_high, &mut *high)] {
        under.set(&one);
        if nested.rho_negative {
            under.add(bound);
        } else {
            under.sub(bound);
        }
        numerator.mul(&under, &factor);
        numerator.shl(precision);
        arith.isqrt(out, &numerator);
    }
    high.add_one();
    factor.clear();
    for big in [
        under, root_low, root_high, numerator, remainder, ratio_low, ratio_high, one, factor,
    ] {
        arith.put(big);
    }
}

#[cfg(test)]
mod nested_vectors;

#[cfg(test)]
pub(super) mod vectors;

#[cfg(test)]
mod tests;
