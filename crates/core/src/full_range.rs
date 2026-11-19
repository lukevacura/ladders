//! The exact `i64 → f64` conversion over the whole `i64` range by two magic-constant halves, for
//! vector bodies that have no native conversion (AVX2): its constants, its proof, and a scalar
//! model of the same operation sequence on bit patterns, which the tests hold equal to `x as f64`
//! on every target and the AVX2 body (`kernel_avx2::to_f64_full`) mirrors line for line.
//!
//! The method, for one lane holding `x`, split as `x = hi · 2³² + lo` with `hi` the signed high
//! half (`−2³¹ ≤ hi < 2³¹`) and `lo` the unsigned low half (`0 ≤ lo < 2³²`):
//!
//! 1. **High.** Shift the lane right by 32 as an unsigned integer (the upper half becomes zero,
//!    the lower half holds `hi`'s bits) and xor it with [`HIGH_MAGIC_BITS`], the bits of
//!    `2⁸⁴ + 2⁶³`. The upper half becomes the exponent of `2⁸⁴`; the lower half, `hi`'s bits with
//!    the top one flipped, is the unsigned integer `hi + 2³¹`. In `[2⁸⁴, 2⁸⁵)` the fraction field
//!    counts units of `2³²`, so the pattern is the double `H = 2⁸⁴ + (hi + 2³¹) · 2³² =
//!    2⁸⁴ + 2⁶³ + hi · 2³²`, exactly (`hi + 2³¹ < 2³²` fits the 52-bit field and carries nothing
//!    into the exponent).
//! 2. **Low.** Keep the lane's lower half and replace its upper half by that of
//!    [`LOW_MAGIC_BITS`], the bits of `2⁵²`. In `[2⁵², 2⁵³)` the fraction field counts units of
//!    1, so the pattern is the double `L = 2⁵² + lo`, exactly.
//! 3. **Unbias.** `F = H − HIGH_MAGIC`, with [`HIGH_MAGIC`] `= 2⁸⁴ + 2⁶³ + 2⁵²` (its bits span
//!    `2⁸⁴` to `2⁵²`, 33 bits, so it is a double). Both operands lie in `[2⁸⁴, 2⁸⁵)`, within a
//!    factor of two of each other, so by Sterbenz's lemma the subtraction is exact:
//!    `F = hi · 2³² − 2⁵²` (also a double: `2³² · (hi − 2²⁰)` with `|hi − 2²⁰| < 2³²`).
//! 4. **Add.** `F + L` has the exact value `hi · 2³² − 2⁵² + 2⁵² + lo = x`. One IEEE addition
//!    returns its exact result correctly rounded to nearest, ties to even: the same rounding
//!    `x as f64` makes. Only this step rounds; `H`, `L` and `F` are exact. For `x = 0` the sum is
//!    `−2⁵² + 2⁵² = +0.0` (an exact zero sum of opposite signs is `+0.0` under round to nearest),
//!    as `0 as f64` is; for `x ≠ 0` it is not zero.
//!
//! So the result equals `x as f64` bit for bit on every `i64`, with no range bound, no
//! lane-crossing permute and no `i32` conversion. The steps are a separate subtract and add,
//! never fused, and there is no multiply to fuse with.

/// The bits of `2⁸⁴ + 2⁶³`: biased exponent `0x453` (`2⁸⁴`), fraction bit 31 (`2³¹` units of
/// `2³²`). Xor'd onto the high half, it biases `hi` by `2³¹` and supplies the exponent.
pub(crate) const HIGH_MAGIC_BITS: u64 = 0x4530_0000_8000_0000;

/// The bits of `2⁵²`: biased exponent `0x433`, fraction zero. Its upper half replaces the low
/// half's.
pub(crate) const LOW_MAGIC_BITS: u64 = 0x4330_0000_0000_0000;

/// `2⁸⁴ + 2⁶³ + 2⁵²`, the bias `H` carries above `hi · 2³² − 2⁵²`: exponent `0x453`, fraction
/// bits 31 (`2⁶³`) and 20 (`2⁵²`).
pub(crate) const HIGH_MAGIC: f64 = f64::from_bits(0x4530_0000_8010_0000);

/// The lower 32 bits of a lane, as the AVX2 body's blend keeps them from `x` (even dwords).
#[cfg(test)]
const LOW_HALF: u64 = 0xFFFF_FFFF;

/// The scalar model of the AVX2 body, step for step on the same bit patterns: `x as f64` for
/// every `i64` (the proof is in the module documentation).
#[cfg(test)]
pub(crate) fn model(x: i64) -> f64 {
    let lane = x as u64;
    // `_mm256_srli_epi64(x, 32)`, then `_mm256_xor_si256` with the high magic.
    let high = f64::from_bits((lane >> 32) ^ HIGH_MAGIC_BITS);
    // `_mm256_blend_epi32::<0b1010_1010>(x, low magic)`: even dwords from `x`, odd from the magic.
    let low = f64::from_bits((lane & LOW_HALF) | (LOW_MAGIC_BITS & !LOW_HALF));
    // `_mm256_sub_pd`, then `_mm256_add_pd`.
    (high - HIGH_MAGIC) + low
}

#[cfg(test)]
pub(crate) mod tests;
