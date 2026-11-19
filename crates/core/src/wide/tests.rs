extern crate std;

use super::{Wide, i128_to_f64};
use core::cmp::Ordering;
use num_bigint::{BigInt, BigUint};
use std::string::ToString;
use std::vec::Vec;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A limb that is often an edge value, so carries and borrows chain across limbs.
    fn limb(&mut self) -> u64 {
        match self.next() % 6 {
            0 => 0,
            1 => u64::MAX,
            2 => 1 << 63,
            3 => (1 << 63) - 1,
            4 => 1,
            _ => self.next(),
        }
    }

    fn wide<const L: usize>(&mut self) -> Wide<L> {
        let mut limbs = [0u64; L];
        for limb in &mut limbs {
            *limb = self.limb();
        }
        Wide(limbs)
    }
}

fn unsigned<const L: usize>(value: &Wide<L>) -> BigUint {
    let bytes: Vec<u8> = value.0.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

fn modulus<const L: usize>() -> BigUint {
    BigUint::from(1u8) << (64 * L)
}

/// The nearest `f64` to `value`, ties to even: Rust's decimal parser rounds correctly.
fn nearest_f64(value: &BigUint) -> f64 {
    value.to_string().parse().expect("a decimal integer parses")
}

fn check_arithmetic<const L: usize>(seed: u64) {
    let mut rng = Rng(seed);
    let modulus = modulus::<L>();
    for _ in 0..3000 {
        let (a, b) = (rng.wide::<L>(), rng.wide::<L>());
        let (big_a, big_b) = (unsigned(&a), unsigned(&b));
        assert_eq!(unsigned(&a.wrapping_add(&b)), (&big_a + &big_b) % &modulus);
        assert_eq!(
            unsigned(&a.wrapping_sub(&b)),
            (&modulus + &big_a - &big_b) % &modulus
        );
        assert_eq!(unsigned(&a.wrapping_mul(&b)), (&big_a * &big_b) % &modulus);
        assert_eq!(unsigned(&a.wrapping_neg()), (&modulus - &big_a) % &modulus);
        assert_eq!(a.cmp(&b), big_a.cmp(&big_b));
        assert_eq!(a.cmp(&a), Ordering::Equal);
        assert_eq!(a.is_zero(), big_a == BigUint::from(0u8));
        assert_eq!(a.bit_length() as u64, big_a.bits());
        assert_eq!(a.is_negative(), big_a.bit(64 * L as u64 - 1));
    }
}

fn signed_of<const L: usize>(value: &Wide<L>) -> BigInt {
    let unsigned_value = BigInt::from(unsigned(value));
    if value.is_negative() {
        unsigned_value - BigInt::from(modulus::<L>())
    } else {
        unsigned_value
    }
}

#[test]
fn arithmetic_wraps_like_the_big_integer_on_every_width() {
    check_arithmetic::<2>(0x1234_5678_9ABC_DEF1);
    check_arithmetic::<4>(0x0FED_CBA9_8765_4321);
    check_arithmetic::<8>(0x9E37_79B9_7F4A_7C15);
}

#[test]
fn unsigned_abs_and_negation_follow_the_signed_value() {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let most_negative = Wide::<4>([0, 0, 0, 1 << 63]);
    for value in core::iter::once(most_negative).chain((0..3000).map(|_| rng.wide::<4>())) {
        if value == most_negative {
            // Its magnitude is one past the largest positive value: it keeps its own bits.
            assert_eq!(value.wrapping_neg(), value);
            assert_eq!(value.unsigned_abs(), value);
        } else {
            assert_eq!(signed_of(&value.wrapping_neg()), -signed_of(&value));
            assert_eq!(
                BigInt::from(unsigned(&value.unsigned_abs())),
                BigInt::from(signed_of(&value).magnitude().clone())
            );
        }
    }
}

#[test]
fn from_i128_sign_extends_to_every_limb() {
    let mut rng = Rng(77);
    for _ in 0..2000 {
        let value = (u128::from(rng.limb()) << 64 | u128::from(rng.limb())) as i128;
        assert_eq!(signed_of(&Wide::<2>::from_i128(value)), BigInt::from(value));
        assert_eq!(signed_of(&Wide::<4>::from_i128(value)), BigInt::from(value));
        assert_eq!(signed_of(&Wide::<8>::from_i128(value)), BigInt::from(value));
    }
    for value in [0i128, 1, -1, i128::MAX, i128::MIN] {
        assert_eq!(signed_of(&Wide::<4>::from_i128(value)), BigInt::from(value));
    }
}

#[test]
fn widen_keeps_the_value() {
    let mut rng = Rng(5);
    for _ in 0..500 {
        let value = rng.wide::<4>();
        assert_eq!(unsigned(&value.widen::<8>()), unsigned(&value));
        assert_eq!(unsigned(&value.widen::<4>()), unsigned(&value));
    }
}

#[test]
fn product_is_the_exact_product_of_two_u128() {
    let mut rng = Rng(11);
    for _ in 0..3000 {
        let a = u128::from(rng.limb()) << 64 | u128::from(rng.limb());
        let b = u128::from(rng.limb()) << 64 | u128::from(rng.limb());
        assert_eq!(
            unsigned(&Wide::product(a, b)),
            BigUint::from(a) * BigUint::from(b)
        );
    }
    assert_eq!(
        unsigned(&Wide::product(u128::MAX, u128::MAX)),
        BigUint::from(u128::MAX) * BigUint::from(u128::MAX)
    );
}

#[test]
fn arithmetic_matches_u128_where_both_fit() {
    let values: [u128; 7] = [
        0,
        1,
        3,
        u128::from(u64::MAX),
        1 << 100,
        (1 << 126) + 12345,
        u128::MAX,
    ];
    for &a in &values {
        for &b in &values {
            let (wide_a, wide_b) = (Wide::<4>::from_u128(a), Wide::<4>::from_u128(b));
            assert_eq!(wide_a.cmp(&wide_b), a.cmp(&b));
            if a >= b {
                assert_eq!(wide_a.wrapping_sub(&wide_b), Wide::from_u128(a - b));
            }
            if let Some(sum) = a.checked_add(b) {
                assert_eq!(wide_a.wrapping_add(&wide_b), Wide::from_u128(sum));
            }
            if let Some(product) = a.checked_mul(b) {
                assert_eq!(wide_a.wrapping_mul(&wide_b), Wide::from_u128(product));
            }
        }
    }
}

#[test]
fn to_f64_equals_the_u128_conversion() {
    let mut rng = Rng(21);
    for _ in 0..20_000 {
        let value = u128::from(rng.limb()) << 64 | u128::from(rng.limb());
        let shifted = value >> (rng.next() % 128);
        assert_eq!(
            Wide::<4>::from_u128(shifted).to_f64().to_bits(),
            (shifted as f64).to_bits()
        );
    }
}

/// Every significand alignment, with the discarded part exactly half, a hair under, a hair over,
/// and with only a far low limb set.
#[test]
fn to_f64_rounds_to_nearest_even_at_every_alignment() {
    let mut rng = Rng(31);
    for shift in 1usize..=(512 - 54) {
        for _ in 0..6 {
            let significand = (rng.next() >> 11) | (1 << 52);
            let mut value = BigUint::from(significand) << shift;
            let half = BigUint::from(1u8) << (shift - 1);
            for tail in [
                BigUint::from(0u8),
                half.clone(),
                &half - BigUint::from(1u8),
                &half + BigUint::from(1u8),
                BigUint::from(1u8),
            ] {
                let candidate = &value + &tail;
                let wide = from_big::<8>(&candidate);
                assert_eq!(
                    wide.to_f64().to_bits(),
                    nearest_f64(&candidate).to_bits(),
                    "{candidate}"
                );
            }
            value += BigUint::from(1u8) << (shift.saturating_sub(40));
            let wide = from_big::<8>(&value);
            assert_eq!(wide.to_f64().to_bits(), nearest_f64(&value).to_bits());
        }
    }
}

fn from_big<const L: usize>(value: &BigUint) -> Wide<L> {
    let mut bytes = value.to_bytes_le();
    bytes.resize(8 * L, 0);
    let mut limbs = [0u64; L];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks(8)) {
        *limb = u64::from_le_bytes(chunk.try_into().expect("eight bytes"));
    }
    Wide(limbs)
}

#[test]
fn to_f64_of_random_wide_values_is_correctly_rounded() {
    let mut rng = Rng(41);
    for _ in 0..4000 {
        let value = rng.wide::<8>();
        assert_eq!(
            value.to_f64().to_bits(),
            nearest_f64(&unsigned(&value)).to_bits()
        );
        let value = rng.wide::<4>();
        assert_eq!(
            value.to_f64_signed().to_bits(),
            signed_of(&value)
                .to_string()
                .parse::<f64>()
                .unwrap()
                .to_bits()
        );
    }
}

#[test]
fn to_f64_signed_is_the_signed_value_rounded() {
    for value in [
        0i128,
        1,
        -1,
        5,
        -5,
        i128::MAX,
        i128::MIN,
        1 << 100,
        -(1 << 100) - 1,
    ] {
        assert_eq!(
            Wide::<4>::from_i128(value).to_f64_signed().to_bits(),
            (value as f64).to_bits(),
            "{value}"
        );
    }
}

#[test]
fn above_128_bits_the_conversion_is_exact_for_powers_and_rounds_the_rest() {
    let square = Wide::<8>::from_u128(1 << 100).wrapping_mul(&Wide::from_u128(1 << 100));
    assert_eq!(square.to_f64(), 2f64.powi(200));
    let odd = Wide::<8>::from_u128((1 << 100) + 1).wrapping_mul(&Wide::from_u128((1 << 100) + 1));
    assert_eq!(odd.to_f64(), 2f64.powi(200) * (1.0 + 2f64.powi(-99)));
}

#[test]
fn i128_to_f64_equals_the_hardware_conversion() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..200_000 {
        let bits = (rng.next() % 128) as u32;
        let raw = (u128::from(rng.next()) << 64 | u128::from(rng.next())) >> (127 - bits.min(127));
        let mut value = raw as i128;
        if rng.next() & 1 == 1 {
            value = -value;
        }
        // Exact ties and near-ties: the bits just under a 53-bit significand.
        let tie = (value >> 1 << 1) | i128::from(rng.next() & 1 == 1);
        for v in [value, tie, value ^ (1 << (rng.next() % 100))] {
            assert_eq!(i128_to_f64(v).to_bits(), (v as f64).to_bits(), "{v}");
        }
    }
    for v in [
        0i128,
        1,
        -1,
        i128::MAX,
        i128::MIN,
        1 << 64,
        (1 << 64) + 1,
        -(1 << 100),
    ] {
        assert_eq!(i128_to_f64(v).to_bits(), (v as f64).to_bits(), "{v}");
    }
}
