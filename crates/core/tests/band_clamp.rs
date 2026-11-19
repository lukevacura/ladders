#![cfg(not(any(loom, shuttle)))]
//! A huge requested band height, chunk width or tile width in the tiled self-join, top-k,
//! two-sided and AB-join entry points gives the bits of the default plan, without sizing scratch
//! by the request.

use core::num::NonZeroUsize;

use ladde::batch::{
    ab_join_report, ab_join_report_tiled, left_profile, left_profile_report_tiled,
    left_topk_profile, left_topk_profile_tiled, two_sided_profile, two_sided_profile_tiled,
};
use ladde::quantize::Scale;
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale() -> Scale {
    Scale::new(1e-3).unwrap()
}

fn series(n: usize, seed: u64, levels: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % levels) as f64
        })
        .collect()
}

fn shown<T: std::fmt::Debug + ?Sized>(value: &T) -> String {
    format!("{value:?}")
}

fn bands(rows: usize) -> [usize; 5] {
    [1, rows, rows + 1, 1 << 63, usize::MAX]
}

const CHUNK_TILES: [(usize, usize); 4] = [
    (1, 1),
    (8, 64),
    (1 << 63, 1 << 63),
    (usize::MAX, usize::MAX),
];

fn check_self_join(data: &[f64], m: usize) {
    let windows = data.len() + 1 - m;
    let expected = shown(&left_profile(data, nz(m), scale()));
    let expected_topk = shown(&left_topk_profile(data, nz(m), scale(), nz(1), nz(2)));
    let expected_two_sided = shown(&two_sided_profile(data, nz(m), scale(), nz(1)));
    for band in bands(windows) {
        for (chunk, tile) in CHUNK_TILES {
            let (band, chunk, tile) = (nz(band), nz(chunk), nz(tile));
            let (profile, _) =
                left_profile_report_tiled(data, nz(m), scale(), nz(2), band, chunk, tile);
            assert_eq!(shown(&profile), expected, "band {band} chunk {chunk}");
            let (topk, _) =
                left_topk_profile_tiled(data, nz(m), scale(), nz(2), band, chunk, tile, nz(2));
            assert_eq!(shown(&topk), expected_topk, "band {band} chunk {chunk}");
            let (two, _) = two_sided_profile_tiled(data, nz(m), scale(), nz(2), band, chunk, tile);
            assert_eq!(shown(&two), expected_two_sided, "band {band} chunk {chunk}");
        }
    }
}

fn check_ab_join(a: &[f64], b: &[f64], m: usize) {
    let n_a = a.len() + 1 - m;
    let (expected, _) = ab_join_report(a, b, nz(m), scale(), scale(), nz(1));
    let expected = shown(&expected);
    for band in bands(n_a) {
        for (chunk, tile) in CHUNK_TILES {
            let (profile, _) = ab_join_report_tiled(
                a,
                b,
                nz(m),
                scale(),
                scale(),
                nz(2),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            assert_eq!(shown(&profile), expected, "band {band} chunk {chunk}");
        }
    }
}

#[test]
fn a_band_of_two_to_the_63_rows_in_the_ab_join_gives_the_default_bits() {
    check_ab_join(&series(40, 3, 5), &series(33, 4, 5), 4);
}

#[test]
fn huge_bands_in_the_self_join_give_the_default_bits() {
    check_self_join(&series(60, 5, 6), 5);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]

    #[test]
    fn huge_shapes_keep_the_default_bits(
        n_a in 8usize..70,
        n_b in 8usize..70,
        m in 2usize..7,
        levels in 1u64..8,
        seed in any::<u64>(),
    ) {
        let a = series(n_a, seed, levels);
        let b = series(n_b, seed ^ 0x9e37, levels);
        check_ab_join(&a, &b, m);
        check_self_join(&a, m);
    }
}
