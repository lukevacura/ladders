//! Unsigned integers of any size, for the exact decisions between sums of square roots.
//!
//! Numbers are little-endian `u32` limbs with no leading zero limb. Every operation writes into a
//! destination the caller owns, and [`Arith`] keeps the numbers an algorithm needs in a pool, so
//! after the first decision of a size nothing allocates: a number put back keeps its capacity.

use alloc::vec::Vec;
use core::cmp::Ordering;

#[derive(Clone, Debug, Default)]
pub(crate) struct Big {
    limbs: Vec<u32>,
}

impl Big {
    pub(crate) fn clear(&mut self) {
        self.limbs.clear();
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    pub(crate) fn set_u128(&mut self, value: u128) {
        self.limbs.clear();
        self.limbs
            .extend((0..4).map(|i| (value >> (32 * i)) as u32));
        self.trim();
    }

    pub(crate) fn set(&mut self, other: &Self) {
        self.limbs.clear();
        self.limbs.extend_from_slice(&other.limbs);
    }

    pub(crate) fn bit_len(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(top) => 32 * (self.limbs.len() as u64 - 1) + u64::from(32 - top.leading_zeros()),
        }
    }

    pub(crate) fn compare(&self, other: &Self) -> Ordering {
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }

    /// `self += other`.
    pub(crate) fn add(&mut self, other: &Self) {
        if self.limbs.len() < other.limbs.len() {
            self.limbs.resize(other.limbs.len(), 0);
        }
        let mut carry = 0u64;
        for (i, limb) in self.limbs.iter_mut().enumerate() {
            let sum =
                u64::from(*limb) + u64::from(other.limbs.get(i).copied().unwrap_or(0)) + carry;
            *limb = sum as u32;
            carry = sum >> 32;
        }
        if carry != 0 {
            self.limbs.push(carry as u32);
        }
    }

    /// `self += 1`.
    pub(crate) fn add_one(&mut self) {
        for limb in &mut self.limbs {
            let (sum, overflow) = limb.overflowing_add(1);
            *limb = sum;
            if !overflow {
                return;
            }
        }
        self.limbs.push(1);
    }

    /// `self -= other`; `other` must not exceed `self`.
    pub(crate) fn sub(&mut self, other: &Self) {
        debug_assert!(self.compare(other) != Ordering::Less);
        let mut borrow = 0i64;
        for (i, limb) in self.limbs.iter_mut().enumerate() {
            let difference =
                i64::from(*limb) - i64::from(other.limbs.get(i).copied().unwrap_or(0)) - borrow;
            *limb = difference as u32;
            borrow = i64::from(difference < 0);
        }
        self.trim();
    }

    /// `self = a · b`.
    pub(crate) fn mul(&mut self, a: &Self, b: &Self) {
        self.limbs.clear();
        if a.is_zero() || b.is_zero() {
            return;
        }
        self.limbs.resize(a.limbs.len() + b.limbs.len(), 0);
        for (i, &x) in a.limbs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &y) in b.limbs.iter().enumerate() {
                let cell = u64::from(self.limbs[i + j]) + u64::from(x) * u64::from(y) + carry;
                self.limbs[i + j] = cell as u32;
                carry = cell >> 32;
            }
            self.limbs[i + b.limbs.len()] = carry as u32;
        }
        self.trim();
    }

    pub(crate) fn shl(&mut self, bits: u64) {
        if self.is_zero() {
            return;
        }
        let (limbs, shift) = ((bits / 32) as usize, (bits % 32) as u32);
        if shift != 0 {
            let mut carry = 0u32;
            for limb in &mut self.limbs {
                let next = *limb >> (32 - shift);
                *limb = (*limb << shift) | carry;
                carry = next;
            }
            if carry != 0 {
                self.limbs.push(carry);
            }
        }
        self.limbs.splice(0..0, core::iter::repeat_n(0, limbs));
    }

    pub(crate) fn shr(&mut self, bits: u64) {
        let (limbs, shift) = ((bits / 32) as usize, (bits % 32) as u32);
        if limbs >= self.limbs.len() {
            self.limbs.clear();
            return;
        }
        self.limbs.drain(..limbs);
        if shift != 0 {
            let mut carry = 0u32;
            for limb in self.limbs.iter_mut().rev() {
                let next = *limb << (32 - shift);
                *limb = (*limb >> shift) | carry;
                carry = next;
            }
        }
        self.trim();
    }

    /// The low 128 bits.
    #[cfg(test)]
    pub(crate) fn low_u128(&self) -> u128 {
        (0..4).fold(0u128, |acc, i| {
            acc | u128::from(self.limbs.get(i).copied().unwrap_or(0)) << (32 * i)
        })
    }
}

/// Numbers to work with, and the buffers division needs.
#[derive(Debug, Default)]
pub(crate) struct Arith {
    pool: Vec<Big>,
    dividend: Vec<u32>,
    divisor: Vec<u32>,
}

impl Arith {
    /// A zero from the pool.
    pub(crate) fn get(&mut self) -> Big {
        self.pool.pop().unwrap_or_default()
    }

    /// A number `value` from the pool.
    pub(crate) fn get_u128(&mut self, value: u128) -> Big {
        let mut big = self.get();
        big.set_u128(value);
        big
    }

    /// Returns a number to the pool, keeping its capacity.
    pub(crate) fn put(&mut self, mut big: Big) {
        big.clear();
        self.pool.push(big);
    }

    /// `quotient = ⌊a / b⌋` and `remainder = a mod b`, for `b` not zero.
    pub(crate) fn div_rem(&mut self, quotient: &mut Big, remainder: &mut Big, a: &Big, b: &Big) {
        assert!(!b.is_zero(), "division by zero");
        if a.compare(b) == Ordering::Less {
            quotient.clear();
            remainder.set(a);
            return;
        }
        if let [only] = b.limbs[..] {
            let divisor = u64::from(only);
            quotient.limbs.clear();
            quotient.limbs.resize(a.limbs.len(), 0);
            let mut rest = 0u64;
            for i in (0..a.limbs.len()).rev() {
                let cell = (rest << 32) | u64::from(a.limbs[i]);
                quotient.limbs[i] = (cell / divisor) as u32;
                rest = cell % divisor;
            }
            quotient.trim();
            remainder.set_u128(u128::from(rest));
            return;
        }
        let n = b.limbs.len();
        let m = a.limbs.len() - n;
        let shift = b.limbs[n - 1].leading_zeros();
        let (vn, un) = (&mut self.divisor, &mut self.dividend);
        vn.clear();
        vn.extend_from_slice(&b.limbs);
        un.clear();
        un.extend_from_slice(&a.limbs);
        un.push(0);
        if shift != 0 {
            for i in (1..n).rev() {
                vn[i] = (vn[i] << shift) | (vn[i - 1] >> (32 - shift));
            }
            vn[0] <<= shift;
            for i in (1..un.len()).rev() {
                un[i] = (un[i] << shift) | (un[i - 1] >> (32 - shift));
            }
            un[0] <<= shift;
        }
        quotient.limbs.clear();
        quotient.limbs.resize(m + 1, 0);
        const BASE: u64 = 1 << 32;
        for j in (0..=m).rev() {
            let top = (u64::from(un[j + n]) << 32) | u64::from(un[j + n - 1]);
            let mut estimate = top / u64::from(vn[n - 1]);
            let mut rest = top - estimate * u64::from(vn[n - 1]);
            loop {
                if estimate >= BASE
                    || estimate * u64::from(vn[n - 2]) > ((rest << 32) | u64::from(un[j + n - 2]))
                {
                    estimate -= 1;
                    rest += u64::from(vn[n - 1]);
                    if rest < BASE {
                        continue;
                    }
                }
                break;
            }
            let mut carry = 0i64;
            for i in 0..n {
                let product = estimate * u64::from(vn[i]);
                let t = i64::from(un[i + j]) - carry - (product & 0xFFFF_FFFF) as i64;
                un[i + j] = t as u32;
                carry = (product >> 32) as i64 - (t >> 32);
            }
            let t = i64::from(un[j + n]) - carry;
            un[j + n] = t as u32;
            quotient.limbs[j] = estimate as u32;
            if t < 0 {
                quotient.limbs[j] = quotient.limbs[j].wrapping_sub(1);
                let mut carry = 0u64;
                for i in 0..n {
                    let sum = u64::from(un[i + j]) + u64::from(vn[i]) + carry;
                    un[i + j] = sum as u32;
                    carry = sum >> 32;
                }
                un[j + n] = un[j + n].wrapping_add(carry as u32);
            }
        }
        quotient.trim();
        remainder.limbs.clear();
        remainder.limbs.extend_from_slice(&un[..n]);
        remainder.trim();
        remainder.shr(u64::from(shift));
    }

    /// `root = ⌊√a⌋`.
    pub(crate) fn isqrt(&mut self, root: &mut Big, a: &Big) {
        if a.is_zero() {
            root.clear();
            return;
        }
        // Newton from above: 2^⌈bits/2⌉ is at least √a, and the iterates fall to ⌊√a⌋.
        root.set_u128(1);
        root.shl(a.bit_len().div_ceil(2));
        let mut quotient = self.get();
        let mut remainder = self.get();
        let mut next = self.get();
        // Newton from above halves the error's bits at each step, so it ends within the bit
        // length; the bound only stops a broken division from spinning.
        let mut converged = false;
        for _ in 0..=a.bit_len() + 1 {
            self.div_rem(&mut quotient, &mut remainder, a, root);
            next.set(&quotient);
            next.add(root);
            next.shr(1);
            if next.compare(root) != Ordering::Less {
                converged = true;
                break;
            }
            core::mem::swap(root, &mut next);
        }
        debug_assert!(converged, "the square root did not converge");
        self.put(quotient);
        self.put(remainder);
        self.put(next);
    }

    /// Whether `a` is the square of an integer; if so, `root` is that integer.
    pub(crate) fn exact_sqrt(&mut self, root: &mut Big, a: &Big) -> bool {
        self.isqrt(root, a);
        let mut square = self.get();
        square.mul(root, root);
        let exact = square.compare(a) == Ordering::Equal;
        self.put(square);
        exact
    }
}

#[cfg(test)]
mod tests;
