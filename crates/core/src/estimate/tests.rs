use super::rank_estimate;

/// Chosen so that `(n · ri) · rj` and `n · (ri · rj)` round to different bits (checked
/// below, not assumed): a small input like `(7, 0.1, 0.2)` rounds both orderings to the
/// same bits, so it would let a reassociated product through undetected.
#[test]
fn left_to_right_order_matches_the_written_expression() {
    let n: i64 = -68_661_172_291_171;
    let ri = 1.0 / libm::sqrt(834_911_684_991_698_f64);
    let rj = 1.0 / libm::sqrt(275_720_901_775_960_f64);
    let left_to_right = (n as f64) * ri * rj;
    let reassociated = (n as f64) * (ri * rj);
    assert_ne!(
        left_to_right.to_bits(),
        reassociated.to_bits(),
        "these inputs must actually distinguish the two operation orders"
    );
    // Pinned so a future edit can't silently reassociate the product: other kernels
    // (the NEON kernel, the narrow growing/batch path) must match these exact bits, not
    // just this bound.
    assert_eq!(rank_estimate(n, ri, rj), left_to_right);
}

/// The f64 producer's declared bound, pinned to a literal bit pattern independent of the
/// expression that defines it, and the bound every kernel evaluating its expression declares.
#[test]
fn the_f64_producer_declares_two_to_the_minus_49() {
    use super::{EPSILON, F64, Producer};
    assert_eq!(EPSILON.to_bits(), 0x3ce0000000000000);
    assert_eq!(F64::DELTA.to_bits(), EPSILON.to_bits());
    assert_eq!(crate::row::RankRow::DELTA.to_bits(), F64::DELTA.to_bits());
    assert_eq!(crate::row::WideRank::DELTA.to_bits(), F64::DELTA.to_bits());
}

/// The floor is `best − 2·delta`, exact where both are powers of two.
#[test]
fn a_band_floor_is_twice_delta_below_the_best() {
    use super::band_floor;
    assert_eq!(band_floor(0.0, 0.25), -0.5);
    assert_eq!(band_floor(1.0, 0.125), 0.75);
    assert_eq!(band_floor(0.5, 0.0), 0.5);
}
