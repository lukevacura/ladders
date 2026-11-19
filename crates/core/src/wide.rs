//! A fixed-width integer on `L` 64-bit limbs, no allocation: the arithmetic of exact products and
//! of the cold route's cross terms. Addition, subtraction and multiplication wrap at `64·L` bits
//! (two's complement, so one type serves unsigned and signed values); comparison is unsigned.

use core::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wide<const L: usize>([u64; L]);

impl<const L: usize> Wide<L> {
    pub const ZERO: Self = Self([0; L]);

    pub(crate) const fn from_u64(value: u64) -> Self {
        let mut limbs = [0; L];
        limbs[0] = value;
        Self(limbs)
    }

    pub(crate) const fn from_u128(value: u128) -> Self {
        let mut limbs = [0; L];
        limbs[0] = value as u64;
        limbs[1] = (value >> 64) as u64;
        Self(limbs)
    }

    /// `value` in two's complement, sign-extended to every limb.
    pub(crate) const fn from_i128(value: i128) -> Self {
        let fill = if value < 0 { u64::MAX } else { 0 };
        let mut limbs = [fill; L];
        limbs[0] = value as u64;
        limbs[1] = (value >> 64) as u64;
        Self(limbs)
    }

    /// The same value on `M ≥ L` limbs, zero-extended.
    pub(crate) fn widen<const M: usize>(&self) -> Wide<M> {
        let mut limbs = [0; M];
        limbs[..L].copy_from_slice(&self.0);
        Wide(limbs)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0.iter().all(|&limb| limb == 0)
    }

    /// The top bit, the sign of a two's complement value.
    pub(crate) fn is_negative(&self) -> bool {
        self.0[L - 1] >> 63 == 1
    }

    pub fn wrapping_add(&self, other: &Self) -> Self {
        let mut out = [0u64; L];
        let mut carry = false;
        for (slot, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(&other.0)) {
            let (sum, first) = a.overflowing_add(b);
            let (sum, second) = sum.overflowing_add(u64::from(carry));
            *slot = sum;
            carry = first | second;
        }
        Self(out)
    }

    pub(crate) fn wrapping_sub(&self, other: &Self) -> Self {
        let mut out = [0u64; L];
        let mut borrow = false;
        for (slot, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(&other.0)) {
            let (difference, first) = a.overflowing_sub(b);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            *slot = difference;
            borrow = first | second;
        }
        Self(out)
    }

    pub(crate) fn wrapping_neg(&self) -> Self {
        Self::ZERO.wrapping_sub(self)
    }

    /// The low `64·L` bits of the product.
    pub fn wrapping_mul(&self, other: &Self) -> Self {
        let mut out = [0u64; L];
        for (i, &a) in self.0.iter().enumerate() {
            if a == 0 {
                continue;
            }
            let mut carry = 0u128;
            for j in 0..L - i {
                let cell = u128::from(a) * u128::from(other.0[j]) + u128::from(out[i + j]) + carry;
                out[i + j] = cell as u64;
                carry = cell >> 64;
            }
        }
        Self(out)
    }

    /// The magnitude of a two's complement value.
    pub(crate) fn unsigned_abs(&self) -> Self {
        if self.is_negative() {
            self.wrapping_neg()
        } else {
            *self
        }
    }

    /// The value with these limbs, lowest first.
    pub(crate) const fn from_limbs(limbs: [u64; L]) -> Self {
        Self(limbs)
    }

    /// The limbs, lowest first.
    pub(crate) fn limbs(&self) -> &[u64; L] {
        &self.0
    }

    /// The lowest 64 bits.
    pub(crate) fn low_u64(&self) -> u64 {
        self.0[0]
    }

    /// The number of bits below the highest set one; `0` for zero.
    pub(crate) fn bit_length(&self) -> u32 {
        for (i, &limb) in self.0.iter().enumerate().rev() {
            if limb != 0 {
                return 64 * i as u32 + 64 - limb.leading_zeros();
            }
        }
        0
    }

    /// The unsigned value rounded to the nearest `f64`, ties to even.
    pub fn to_f64(&self) -> f64 {
        let bits = self.bit_length();
        if bits <= 64 {
            return self.0[0] as f64;
        }
        let shift = bits - 64;
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let mut top = self.0[limb] >> offset;
        if offset > 0 && limb + 1 < L {
            top |= self.0[limb + 1] << (64 - offset);
        }
        let mut sticky = offset > 0 && self.0[limb] << (64 - offset) != 0;
        sticky |= self.0[..limb].iter().any(|&x| x != 0);
        top |= u64::from(sticky);
        (top as f64) * f64::from_bits((1023 + u64::from(shift)) << 52)
    }

    /// The two's complement value rounded to the nearest `f64`, ties to even.
    pub fn to_f64_signed(&self) -> f64 {
        let magnitude = self.unsigned_abs().to_f64();
        if self.is_negative() {
            -magnitude
        } else {
            magnitude
        }
    }
}

impl Wide<4> {
    /// `a · b`, exactly.
    pub fn product(a: u128, b: u128) -> Self {
        let (a_low, a_high) = (u128::from(a as u64), a >> 64);
        let (b_low, b_high) = (u128::from(b as u64), b >> 64);
        let (low_low, low_high) = (a_low * b_low, a_low * b_high);
        let (high_low, high_high) = (a_high * b_low, a_high * b_high);
        let middle = (low_low >> 64) + u128::from(low_high as u64) + u128::from(high_low as u64);
        let upper =
            (low_high >> 64) + (high_low >> 64) + u128::from(high_high as u64) + (middle >> 64);
        let top = (high_high >> 64) + (upper >> 64);
        Self([low_low as u64, middle as u64, upper as u64, top as u64])
    }
}

impl<const L: usize> PartialOrd for Wide<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const L: usize> Ord for Wide<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

/// `x` rounded to the nearest `f64`, ties to even, without the compiler's slow 128-bit
/// conversion: the top 64 bits with a sticky bit, rounded by the hardware's `u64` conversion.
#[inline]
pub(crate) fn i128_to_f64(x: i128) -> f64 {
    let magnitude = x.unsigned_abs();
    let value = if magnitude >> 64 == 0 {
        (magnitude as u64) as f64
    } else {
        let shift = 64 - magnitude.leading_zeros();
        let top = (magnitude >> shift) as u64 | u64::from(magnitude << (128 - shift) != 0);
        (top as f64) * f64::from_bits((1023 + u64::from(shift)) << 52)
    };
    if x < 0 { -value } else { value }
}

#[cfg(test)]
mod tests;
