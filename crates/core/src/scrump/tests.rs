use super::chunks::chunk_ranges;
use super::{DiagonalSource, Explicit, RandomSource, ScrumpError, Shuffled, SplitMix64};
use alloc::vec::Vec;

/// The words of a fixed script, then zeros.
struct Script(Vec<u64>);

impl RandomSource for Script {
    fn next_u64(&mut self) -> u64 {
        if self.0.is_empty() {
            0
        } else {
            self.0.remove(0)
        }
    }
}

#[test]
fn splitmix64_is_the_reference_generator() {
    let mut zero = SplitMix64::new(0);
    assert_eq!(
        [zero.next_u64(), zero.next_u64(), zero.next_u64()],
        [
            0xe220_a839_7b1d_cdaf,
            0x6e78_9e6a_a1b9_65f4,
            0x06c4_5d18_8009_454f
        ]
    );
    let mut other = SplitMix64::new(1_234_567);
    assert_eq!(
        [other.next_u64(), other.next_u64(), other.next_u64()],
        [
            6_457_827_717_110_365_317,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423
        ]
    );
}

#[test]
fn below_rejects_the_words_that_would_bias_the_draw() {
    // 2^64 mod 3 is 1: only the word 0 lands below the threshold, so it is skipped.
    let mut source = Script(alloc::vec![0, 5]);
    assert_eq!(source.below(3), 0);
    assert!(source.0.is_empty());
    let mut source = Script(alloc::vec![u64::MAX]);
    assert_eq!(source.below(3), 2);
    assert_eq!(Script(alloc::vec![9]).below(0), 0);
    assert_eq!(Script(alloc::vec![u64::MAX]).below(1), 0);
    // A power of two never rejects.
    assert_eq!(Script(alloc::vec![u64::MAX]).below(1 << 63), (1 << 63) - 1);
}

#[test]
fn a_seed_fixes_the_order_on_every_platform() {
    let order = |seed, first, count| Shuffled::seeded(seed).order(first, count).unwrap();
    assert_eq!(order(42, 3, 10), [11, 6, 9, 8, 7, 3, 12, 5, 4, 10]);
    assert_eq!(order(0, -4, 8), [-3, -2, 2, 1, 0, -4, -1, 3]);
    assert_eq!(order(7, 1, 1), [1]);
    assert!(order(7, 1, 0).is_empty());
}

#[test]
fn a_shuffle_is_a_permutation_of_the_diagonals() {
    for (seed, first, count) in [(1, 0, 1), (2, 5, 2), (3, -20, 41), (4, 7, 300)] {
        let mut order = Shuffled::seeded(seed).order(first, count).unwrap();
        order.sort_unstable();
        let expected: Vec<i64> = (first..first + count as i64).collect();
        assert_eq!(order, expected);
    }
}

#[test]
fn a_custom_generator_drives_the_same_shuffle() {
    let words: Vec<u64> = {
        let mut reference = SplitMix64::new(42);
        (0..64).map(|_| reference.next_u64()).collect()
    };
    assert_eq!(
        Shuffled(Script(words)).order(3, 10).unwrap(),
        Shuffled::seeded(42).order(3, 10).unwrap()
    );
}

#[test]
fn an_explicit_order_must_be_a_permutation() {
    assert_eq!(
        Explicit(alloc::vec![5, 3, 4]).order(3, 3),
        Ok(alloc::vec![5, 3, 4])
    );
    for order in [
        alloc::vec![],
        alloc::vec![3, 4],
        alloc::vec![3, 4, 5, 6],
        alloc::vec![3, 4, 4],
        alloc::vec![2, 4, 5],
        alloc::vec![3, 4, 6],
        alloc::vec![i64::MIN, 4, 5],
        alloc::vec![3, i64::MAX, 5],
    ] {
        assert_eq!(
            Explicit(order.clone()).order(3, 3),
            Err(ScrumpError::NotAPermutation),
            "{order:?}"
        );
    }
    assert_eq!(Explicit(Vec::new()).order(3, 0), Ok(Vec::new()));
}

/// STUMPY 1.14.1's chunks for a self-join of `windows` windows at exclusion `e`, the order a
/// seeded `RandomState` permutation gave, and `percentage`.
fn stumpy_chunks(windows: u64, order: &[i64], percentage: f64) -> Vec<(usize, usize)> {
    chunk_ranges(order, |d| windows - d as u64, percentage)
        .unwrap()
        .into_iter()
        .map(|range| (range.start, range.end))
        .collect()
}

#[test]
fn chunks_are_stumpys() {
    let order = [6, 16, 10, 5, 9, 13, 7, 4, 17, 3, 19, 18, 12, 11, 15, 14, 8];
    assert_eq!(
        stumpy_chunks(20, &order, 0.1),
        [
            (0, 2),
            (2, 4),
            (4, 5),
            (5, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 14),
            (14, 16),
            (16, 17)
        ]
    );
    let order = [15, 7, 8, 3, 12, 6, 4, 13, 10, 17, 5, 14, 9, 19, 16, 18, 11];
    assert_eq!(
        stumpy_chunks(20, &order, 0.25),
        [(0, 4), (4, 7), (7, 11), (11, 17)]
    );
    let order = [
        22, 21, 16, 30, 19, 20, 17, 6, 5, 27, 18, 8, 26, 10, 11, 9, 24, 13, 15, 32, 23, 25, 4, 12,
        31, 7, 29, 28, 14,
    ];
    assert_eq!(
        stumpy_chunks(33, &order, 0.3333),
        [(0, 8), (8, 15), (15, 22), (22, 29)]
    );
    assert_eq!(stumpy_chunks(33, &order, 1.0), [(0, 29)]);
    let order = [
        15, 25, 31, 19, 24, 28, 21, 29, 23, 4, 7, 20, 14, 10, 16, 6, 8, 26, 17, 11, 13, 22, 32, 12,
        5, 9, 27, 18, 30,
    ];
    assert_eq!(stumpy_chunks(33, &order, 0.9), [(0, 16), (16, 29)]);
    let order = [10, 7, 2, 4, 3, 11, 9, 5, 8, 6];
    assert_eq!(stumpy_chunks(12, &order, 0.5), [(0, 5), (5, 10)]);
    let order = [8, 4, 5, 6, 3, 7];
    assert_eq!(stumpy_chunks(9, &order, 0.15), [(0, 2), (2, 6)]);
}

/// A share exactly half of the total: STUMPY (numba fastmath) divides as a product with the
/// reciprocal, which rounds `49 * (1 / 98)` to just under one half, so the boundary lands one
/// diagonal later than plain division puts it.
#[test]
fn a_boundary_at_an_exact_tie_follows_stumpys_reciprocal() {
    let order: Vec<i64> = (0..50).collect();
    let chunks: Vec<(usize, usize)> = chunk_ranges(&order, |d| if d == 0 { 49 } else { 1 }, 0.5)
        .unwrap()
        .into_iter()
        .map(|range| (range.start, range.end))
        .collect();
    assert_eq!(chunks, [(0, 2), (2, 50)]);
}

#[test]
fn a_tiny_percentage_ends_at_the_first_repeated_boundary() {
    let order = [
        36, 37, 27, 29, 12, 30, 13, 34, 49, 48, 44, 9, 40, 39, 6, 20, 25, 38, 23, 33, 43, 47, 24,
        28, 45, 11, 35, 46, 18, 8, 14, 42, 10, 32, 19, 17, 7, 31, 22, 21, 26, 41, 15, 16,
    ];
    let expected = [(0, 1), (1, 2), (2, 3), (3, 44)];
    assert_eq!(stumpy_chunks(50, &order, 0.01), expected);
    // Far below one chunk per diagonal every boundary is the first one: two chunks, from a
    // bounded amount of work.
    assert_eq!(stumpy_chunks(50, &order, 1e-300), [(0, 1), (1, 44)]);
    assert_eq!(
        stumpy_chunks(50, &order, f64::MIN_POSITIVE),
        [(0, 1), (1, 44)]
    );
}

#[test]
fn chunks_cover_the_order_once_whatever_the_percentage() {
    let order: Vec<i64> = (3..80).rev().collect();
    for percentage in [1.0, 0.99, 0.5, 0.34, 0.2, 0.05, 0.001] {
        let chunks = stumpy_chunks(90, &order, percentage);
        assert_eq!(chunks.first().map(|c| c.0), Some(0));
        assert_eq!(chunks.last().map(|c| c.1), Some(order.len()));
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "{percentage}");
        }
        assert!(chunks.iter().all(|c| c.0 < c.1), "{percentage}");
    }
}

#[test]
fn a_percentage_outside_zero_to_one_has_no_chunks() {
    for percentage in [
        0.0,
        -0.0,
        -1.0,
        1.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(
            chunk_ranges(&[1, 2], |_| 1, percentage),
            Err(ScrumpError::Percentage)
        );
    }
}

#[cfg(feature = "rand")]
mod rand {
    use super::*;
    use rand_core::{RngCore, SeedableRng};

    /// SplitMix64 as a `rand_core` generator.
    struct Mix(SplitMix64);

    impl RngCore for Mix {
        fn next_u32(&mut self) -> u32 {
            (self.0.next_u64() >> 32) as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0.next_u64()
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                let word = self.0.next_u64().to_le_bytes();
                chunk.copy_from_slice(&word[..chunk.len()]);
            }
        }
    }

    #[test]
    fn a_rand_core_generator_is_a_random_source_with_its_own_stream() {
        // The same words, so the same order as the built-in generator.
        assert_eq!(
            Shuffled(Mix(SplitMix64::new(42))).order(3, 10).unwrap(),
            Shuffled::seeded(42).order(3, 10).unwrap()
        );
        let mut generator = Mix(SplitMix64::new(5));
        assert_eq!(
            RandomSource::next_u64(&mut generator),
            SplitMix64::new(5).next_u64()
        );
    }

    #[test]
    fn a_seeded_rand_core_generator_gives_a_permutation_that_follows_its_seed() {
        let order = |seed: u64| {
            Shuffled(rand_xorshift::XorShiftRng::seed_from_u64(seed))
                .order(-5, 90)
                .unwrap()
        };
        let mut sorted = order(1);
        sorted.sort_unstable();
        assert_eq!(sorted, (-5..85).collect::<Vec<i64>>());
        assert_eq!(order(1), order(1));
        assert_ne!(order(1), order(2));
    }
}
