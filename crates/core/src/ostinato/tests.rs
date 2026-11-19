use alloc::vec::Vec;

use super::{is_close, mean, pairwise_sum};

/// Values and the bits NumPy 2.5's `add.reduce` and `mean` return on them.
fn numpy_sums() -> [(usize, u64, u64); 10] {
    [
        (1, 0x3fbfb2c3741c8d88, 0x3fbfb2c3741c8d88),
        (7, 0x40f2ae03bf40b765, 0x40c55928da931abd),
        (8, 0x40f2ae72f1d6281e, 0x40c2ae72f1d6281e),
        (9, 0x40f3bef8b8d562eb, 0x40c18d4edd2f745f),
        (17, 0x4105314c5796e8b7, 0x40c3f229bbd9537f),
        (64, 0x41247d2afc0e568c, 0x40c47d2afc0e568c),
        (129, 0x4135e85989b215ec, 0x40c5bcdfca1dda38),
        (130, 0x413636064ac973a4, 0x40c5de8c1a5ff3d5),
        (257, 0x4145720b3f0883b6, 0x40c55cae90780baa),
        (1000, 0x41653255043d173e, 0x40c5b49066a4f4fc),
    ]
}

#[test]
fn sums_and_means_equal_numpys_on_bits() {
    let scale = [0.01, 0.1, 1.0, 10.0, 100.0];
    for (n, sum, average) in numpy_sums() {
        let values: Vec<f64> = (0..n)
            .map(|i| ((i * 2_654_435_761 + 12345) % 1_000_003) as f64 / 997.0 * scale[i % 5])
            .collect();
        assert_eq!(pairwise_sum(&values).to_bits(), sum, "sum of {n}");
        assert_eq!(mean(&values).to_bits(), average, "mean of {n}");
    }
}

#[test]
fn closeness_is_numpys_with_default_tolerances() {
    assert!(is_close(1.0, 1.0 + 0.9e-5));
    assert!(!is_close(1.0, 1.0 + 1.1e-5));
    assert!(is_close(0.0, 0.9e-8));
    assert!(!is_close(0.0, 1.1e-8));
    assert!(is_close(f64::INFINITY, f64::INFINITY));
    assert!(!is_close(f64::INFINITY, 1.0));
    assert!(!is_close(1.0, f64::INFINITY));
    assert!(!is_close(f64::NAN, f64::NAN));
}
