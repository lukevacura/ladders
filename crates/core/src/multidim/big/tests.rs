use super::{Arith, Big};
use alloc::vec::Vec;
use core::cmp::Ordering;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A limb that is often an extreme one, where carries and borrows live.
    fn limb(&mut self) -> u32 {
        match self.next() % 6 {
            0 => 0,
            1 => u32::MAX,
            2 => 1,
            3 => 0x8000_0000,
            _ => self.next() as u32,
        }
    }

    fn big(&mut self, limbs: usize) -> Big {
        let mut big = Big {
            limbs: (0..limbs).map(|_| self.limb()).collect(),
        };
        big.trim();
        big
    }
}

fn big(value: u128) -> Big {
    let mut big = Big::default();
    big.set_u128(value);
    big
}

#[test]
fn small_values_agree_with_u128() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut arith = Arith::default();
    for _ in 0..4000 {
        let a = u128::from(rng.next()) << (rng.next() % 64) | u128::from(rng.next());
        let b = (u128::from(rng.next()) >> (rng.next() % 64)).max(1);
        let (x, y) = (big(a), big(b));
        assert_eq!(x.compare(&y), a.cmp(&b));
        let mut sum = x.clone();
        sum.add(&y);
        if let Some(want) = a.checked_add(b) {
            assert_eq!(sum.low_u128(), want);
            assert!(sum.bit_len() <= 128);
        }
        if a >= b {
            let mut difference = x.clone();
            difference.sub(&y);
            assert_eq!(difference.low_u128(), a - b);
        }
        let (small_a, small_b) = (a as u64, b as u64);
        let mut product = Big::default();
        product.mul(&big(u128::from(small_a)), &big(u128::from(small_b)));
        assert_eq!(
            product.low_u128(),
            u128::from(small_a) * u128::from(small_b)
        );
        let shift = rng.next() % 100;
        let mut shifted = x.clone();
        shifted.shl(shift);
        shifted.shr(shift);
        assert_eq!(shifted.compare(&x), Ordering::Equal);
        let (mut quotient, mut remainder) = (Big::default(), Big::default());
        arith.div_rem(&mut quotient, &mut remainder, &x, &y);
        assert_eq!(quotient.low_u128(), a / b);
        assert_eq!(remainder.low_u128(), a % b);
        let mut root = Big::default();
        arith.isqrt(&mut root, &x);
        assert_eq!(root.low_u128(), a.isqrt());
    }
}

#[test]
fn bit_length_and_zero() {
    assert_eq!(big(0).bit_len(), 0);
    assert!(big(0).is_zero());
    assert_eq!(big(1).bit_len(), 1);
    assert_eq!(big(u128::MAX).bit_len(), 128);
    let mut power = big(1);
    power.shl(1000);
    assert_eq!(power.bit_len(), 1001);
    power.shr(1000);
    assert_eq!(power.low_u128(), 1);
    power.shr(1);
    assert!(power.is_zero());
    let mut zero = big(0);
    zero.shl(70);
    assert!(zero.is_zero());
}

#[test]
fn division_reconstructs_its_dividend() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut arith = Arith::default();
    let (mut quotient, mut remainder) = (Big::default(), Big::default());
    for round in 0..30_000 {
        let divisor = loop {
            let candidate = rng.big(1 + round % 9);
            if !candidate.is_zero() {
                break candidate;
            }
        };
        let limbs = 1 + (rng.next() % 14) as usize;
        let dividend = rng.big(limbs);
        arith.div_rem(&mut quotient, &mut remainder, &dividend, &divisor);
        assert_eq!(remainder.compare(&divisor), Ordering::Less);
        let mut rebuilt = Big::default();
        rebuilt.mul(&quotient, &divisor);
        rebuilt.add(&remainder);
        assert_eq!(rebuilt.compare(&dividend), Ordering::Equal, "round {round}");
    }
}

#[test]
fn division_handles_the_estimate_correction_cases() {
    // Dividends and divisors built so the estimated quotient digit is too large, including
    // the rare case that needs the add-back.
    let mut arith = Arith::default();
    let (mut quotient, mut remainder) = (Big::default(), Big::default());
    let cases: [(&[u32], &[u32]); 5] = [
        (&[3, 0, 0x8000_0000], &[1, 0, 0x2000_0000]),
        (&[0, 0xffff_fffe, 0x8000_0000], &[0xffff_ffff, 0x8000_0000]),
        (&[0, 0, 0, 0x8000_0000, 0x7fff_ffff], &[1, 0, 0x8000_0000]),
        (&[0xffff_ffff; 6], &[0xffff_ffff, 0xffff_ffff, 0xffff_ffff]),
        (&[0, 0, 0, 0, 1], &[0xffff_ffff, 0xffff_ffff]),
    ];
    for (dividend, divisor) in cases {
        let (dividend, divisor) = (
            Big {
                limbs: dividend.to_vec(),
            },
            Big {
                limbs: divisor.to_vec(),
            },
        );
        arith.div_rem(&mut quotient, &mut remainder, &dividend, &divisor);
        assert_eq!(remainder.compare(&divisor), Ordering::Less);
        let mut rebuilt = Big::default();
        rebuilt.mul(&quotient, &divisor);
        rebuilt.add(&remainder);
        assert_eq!(rebuilt.compare(&dividend), Ordering::Equal);
    }
}

#[test]
fn square_root_brackets_its_argument() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut arith = Arith::default();
    let mut root = Big::default();
    for round in 0..4000 {
        let value = rng.big(1 + round % 12);
        arith.isqrt(&mut root, &value);
        let mut square = Big::default();
        square.mul(&root, &root);
        assert_ne!(square.compare(&value), Ordering::Greater);
        root.add_one();
        square.mul(&root, &root);
        assert_eq!(square.compare(&value), Ordering::Greater);
    }
}

#[test]
fn an_exact_square_is_recognised() {
    let mut rng = Rng(77);
    let mut arith = Arith::default();
    let mut root = Big::default();
    for round in 0..500 {
        let base = rng.big(1 + round % 8);
        let mut square = Big::default();
        square.mul(&base, &base);
        assert!(arith.exact_sqrt(&mut root, &square));
        assert_eq!(root.compare(&base), Ordering::Equal);
        square.add_one();
        if !base.is_zero() {
            assert!(!arith.exact_sqrt(&mut root, &square));
        }
    }
}

#[test]
fn numbers_returned_to_the_pool_keep_their_capacity() {
    let mut arith = Arith::default();
    let mut number = arith.get_u128(u128::MAX);
    number.shl(4000);
    let capacity = number.limbs.capacity();
    arith.put(number);
    let again = arith.get();
    assert!(again.is_zero());
    assert_eq!(again.limbs.capacity(), capacity);
}

#[test]
fn shifts_of_many_limbs_round_trip() {
    let mut rng = Rng(5);
    for round in 0..500 {
        let value = rng.big(1 + round % 10);
        let shift = rng.next() % 200;
        let mut moved = value.clone();
        moved.shl(shift);
        let expected_bits = if value.is_zero() {
            0
        } else {
            value.bit_len() + shift
        };
        assert_eq!(moved.bit_len(), expected_bits);
        moved.shr(shift);
        assert_eq!(moved.compare(&value), Ordering::Equal);
        let limbs: Vec<u32> = moved.limbs.clone();
        assert!(limbs.last().copied().unwrap_or(1) != 0);
    }
}
