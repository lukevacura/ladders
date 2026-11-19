//! The one ranking expression every kernel uses to approximate a pair's correlation in O(1):
//! [`rank_estimate`], and the near-tie band its producers' error bounds give.

/// `((n as f64) * ri) * rj`, left to right: an O(1) estimate of the exact correlation `ρ =
/// n/√(Vi·Vj)`, for `ri = 1/√Vi` and `rj = 1/√Vj`, each a correctly rounded square root
/// followed by a correctly rounded division.
///
/// Seven correctly rounded operations reach it: `n`'s conversion to `f64`; each window's
/// square root and its reciprocal division (two roundings per window, four in all); and the
/// two multiplications above. None of them is subnormal — a window's variance stays below
/// `2⁶³`, so its reciprocal square root is at least `2⁻³¹·⁵`, and a nonzero `n` converts to at
/// least `1.0` — so ordinary half-ULP rounding bounds apply throughout. Each rounding
/// contributes about as much error as the step depends on what came before it: roughly `1`
/// ULP from `n`'s own conversion, `2.5` ULP from each reciprocal square root (its sqrt's
/// rounding plus its division's, the first attenuated by the square root's shallower slope),
/// and `1` ULP from each product, for a first-order sum of `8` ULP, i.e. `2⁻⁵⁰`. That sum is
/// already this function's whole promised bound; what it leaves out — cross terms between two
/// already-tiny roundings, of order `2⁻¹⁰⁶` — is too small to spend any of that bound on.
pub fn rank_estimate(n: i64, ri: f64, rj: f64) -> f64 {
    ((n as f64) * ri) * rj
}

/// A producer of pair estimates, whatever computes them (a vector kernel, a scalar one, a device
/// buffer): `DELTA` bounds how far an estimate, or a block's maximum of them, lies from the exact
/// correlation it stands for, for every input the library accepts, on the sides `SIDES` names.
/// Every near-tie band and estimate margin reads it from the producer that made its estimates.
pub(crate) trait Producer {
    const DELTA: f64;
    /// Which side of the exact correlation `DELTA` bounds: both, unless the producer says
    /// otherwise.
    const SIDES: Sides = Sides::Both;
}

/// The sides of the exact correlation `ρ` a [`Producer`]'s value `v` is bounded on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sides {
    /// An estimate: `|v − ρ| ≤ DELTA`.
    Both,
    /// An upper bound (one-sided producers, ladder amendment): `v ≥ ρ − DELTA` for every pair the value covers,
    /// the producer's float evaluation margins inside its proof. A consumer skips a block only
    /// when `v` is below the floor of every row and column in it, each floor formed by
    /// `band_floor` under the bound of the producer whose estimates set it.
    Above,
}

impl Sides {
    /// Whether a value bounded on these sides is an upper bound of every exact correlation it
    /// covers.
    pub(crate) const fn bounds_above(self) -> bool {
        matches!(self, Self::Above)
    }
}

/// `2⁻⁴⁹`: the bound [`F64`] declares, twice the `2⁻⁵⁰` that [`rank_estimate`] is proved to keep.
pub(crate) const EPSILON: f64 = 1.0 / (1u64 << 49) as f64;

/// [`rank_estimate`] as a producer of estimates, whatever kernel (scalar or vector) evaluates it
/// in `f64`, operation for operation.
pub(crate) struct F64;

impl Producer for F64 {
    const DELTA: f64 = EPSILON;
}

/// The near-tie band's floor under a producer's `delta`: a pair whose estimate is below
/// `best − 2·delta` cannot be nearer than the pair `best` estimates.
#[inline(always)]
pub(crate) fn band_floor(best: f64, delta: f64) -> f64 {
    best - 2.0 * delta
}

#[cfg(test)]
mod tests;
