#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The tiled, threaded AB-join with `k` neighbors equals `naive_ab_join_topk` on bits at every
//! thread count and tiling: random A and B of different lengths and scales, gaps, flat runs, exact
//! repeats and heavy ties, B equal to A, one side too wide for an anchor; with `k` of one it is
//! `ab_join`.

use core::num::NonZeroUsize;

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{ab_join, ab_join_topk, ab_join_topk_tiled};
use ladde::entry::Entry;
use ladde::quantize::Scale;
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series};

type EntryBits = (u64, u64, u64);

const TILINGS: [(usize, usize, usize); 5] =
    [(1, 1, 1), (3, 2, 5), (8, 4, 4), (5, 3, 64), (64, 8, 64)];
const THREADS: [usize; 3] = [1, 2, 7];
const KS: [usize; 5] = [1, 2, 3, 8, 200];

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn entry_bits(e: Entry) -> EntryBits {
    (
        e.rho().to_bits(),
        e.squared_distance().to_bits(),
        e.index().get().unwrap_or(u64::MAX),
    )
}

fn bits(profile: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<EntryBits>)> {
    profile
        .iter()
        .map(|(row, entries)| (*row, entries.iter().copied().map(entry_bits).collect()))
        .collect()
}

/// The exact reference top-k join, in `bits`' shape: a window with no distance in the hot
/// passes' domain has one too.
fn exact(
    a: &[f64],
    b: &[f64],
    m: usize,
    scales: (Scale, Scale),
    k: usize,
) -> Vec<(u64, Vec<EntryBits>)> {
    spill_run::exact_join_topk(a, scales.0, b, scales.1, m, k)
        .into_iter()
        .enumerate()
        .map(|(row, kept)| {
            let kept = kept.into_iter().map(|(_, rho, dist, at)| (rho, dist, at));
            (row as u64, kept.collect())
        })
        .collect()
}

fn check_everywhere(a: &[f64], b: &[f64], m: usize, scales: (Scale, Scale), ks: &[usize]) {
    for &k in ks {
        let expected = exact(a, b, m, scales, k);
        for threads in THREADS {
            let default =
                ab_join_topk(a, b, nz(m), scales.0, scales.1, nz(threads), nz(k)).unwrap();
            assert_eq!(bits(&default), expected, "k {k}, default threads {threads}");
            for (band, chunk, tile) in TILINGS {
                let (profile, _) = ab_join_topk_tiled(
                    a,
                    b,
                    nz(m),
                    scales.0,
                    scales.1,
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                    nz(k),
                )
                .unwrap();
                assert_eq!(
                    bits(&profile),
                    expected,
                    "k {k}, threads {threads}, tiling {:?}",
                    (band, chunk, tile)
                );
            }
        }
    }
}

fn scales() -> impl Strategy<Value = Scale> {
    prop_oneof![Just(MARKET_TICK), Just(0.5), Just(1.0)].prop_map(scale)
}

proptest! {
    #![proptest_config(testkit::generators::seeded(16))]

    #[test]
    fn random_pairs_equal_naive_at_every_tiling(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        b in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        scale_a in scales(),
        scale_b in scales(),
    ) {
        check_everywhere(&a, &b, m, (scale_a, scale_b), &KS);
    }

    #[test]
    fn a_series_joined_with_itself_equals_naive_at_every_tiling(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        scale_a in scales(),
    ) {
        check_everywhere(&a, &a, m, (scale_a, scale_a), &[1, 3, 8]);
    }
}

#[test]
fn a_repeating_pair_ties_to_the_smaller_index() {
    let period = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0][i % 7])
            .collect()
    };
    check_everywhere(&period(40), &period(75), 5, (scale(1.0), scale(1.0)), &KS);
}

#[test]
fn constant_and_few_level_pairs_equal_naive() {
    let one = scale(1.0);
    check_everywhere(&[7.0; 30], &[7.0; 45], 4, (one, one), &KS);
    let levels = |n: usize, mul: usize| -> Vec<f64> {
        (0..n).map(|i| f64::from((i * mul / 5 % 3) as u8)).collect()
    };
    check_everywhere(&levels(50, 7), &levels(60, 3), 4, (one, one), &KS);
    let runs: Vec<f64> = (0..90).map(|k| f64::from(((k / 11) % 3) as u8)).collect();
    check_everywhere(&runs, &levels(70, 7), 5, (one, one), &KS);
}

#[test]
fn gaps_at_the_start_the_end_and_inside_equal_naive() {
    let mut a: Vec<f64> = (0..60).map(|k| f64::from((k * 7 % 11) as u8)).collect();
    a[..3].fill(f64::NAN);
    a[30] = f64::NAN;
    a[57..].fill(f64::NAN);
    let mut b: Vec<f64> = (0..50).map(|k| f64::from((k * 5 % 9) as u8)).collect();
    b[10..14].fill(f64::NAN);
    b[47..].fill(f64::NAN);
    check_everywhere(&a, &b, 4, (scale(1.0), scale(1.0)), &KS);
    check_everywhere(&b, &a, 4, (scale(1.0), scale(1.0)), &[1, 3]);
}

#[test]
fn a_side_too_wide_for_one_anchor_equals_naive() {
    let wide: Vec<f64> = (0..90)
        .map(|k| f64::from(((k * 7 % 13) as u8) % 5) + if k % 2 == 0 { 0.0 } else { 1.5e9 })
        .collect();
    let flat: Vec<f64> = (0..70).map(|k| f64::from((k * 3 % 8) as u8)).collect();
    let one = scale(1.0);
    check_everywhere(&wide, &flat, 4, (one, one), &[1, 3, 8]);
    check_everywhere(&flat, &wide, 4, (one, one), &[1, 3, 8]);
}

#[test]
fn k_of_one_is_the_ab_join_on_bits() {
    let a: Vec<f64> = (0..90).map(|k| f64::from((k * 5 % 17) as u8)).collect();
    let b: Vec<f64> = (0..70).map(|k| f64::from((k * 3 % 11) as u8)).collect();
    let one = scale(1.0);
    let single = ab_join(&a, &b, nz(5), one, one, nz(2));
    let topk = ab_join_topk(&a, &b, nz(5), one, one, nz(2), nz(1)).unwrap();
    assert_eq!(single.len(), topk.len());
    for ((row, entry), (topk_row, entries)) in single.iter().zip(&topk) {
        assert_eq!(row, topk_row);
        match entry.index().get() {
            None => assert!(entries.is_empty()),
            Some(_) => assert_eq!(
                entries.iter().copied().map(entry_bits).collect::<Vec<_>>(),
                [entry_bits(*entry)]
            ),
        }
    }
}

/// With `b` shorter than `k` windows every row keeps one entry per finite `b` window.
#[test]
fn fewer_b_windows_than_k_keeps_them_all() {
    let a: Vec<f64> = (0..40).map(|k| f64::from((k * 5 % 17) as u8)).collect();
    let b = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0];
    let one = scale(1.0);
    let topk = ab_join_topk(&a, &b, nz(4), one, one, nz(2), nz(50)).unwrap();
    assert!(topk.iter().all(|(_, entries)| entries.len() == 3));
    check_everywhere(&a, &b, 4, (one, one), &[2, 3, 50]);
}

#[test]
fn an_empty_side_has_no_neighbors_at_any_k() {
    let one = scale(1.0);
    let a: Vec<f64> = (0..20).map(f64::from).collect();
    for k in [1, 3] {
        let rows = ab_join_topk(&a, &[1.0, 2.0], nz(5), one, one, nz(2), nz(k)).unwrap();
        assert_eq!(rows.len(), 16);
        assert!(rows.iter().all(|(_, entries)| entries.is_empty()));
        assert!(
            ab_join_topk(&[1.0, 2.0], &a, nz(5), one, one, nz(2), nz(k))
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn a_huge_k_is_clamped_to_the_b_windows() {
    let a: Vec<f64> = (0..50).map(|k| f64::from((k * 5 % 17) as u8)).collect();
    let b: Vec<f64> = (0..40).map(|k| f64::from((k * 3 % 11) as u8)).collect();
    let one = scale(1.0);
    let huge = ab_join_topk(&a, &b, nz(4), one, one, nz(2), nz(usize::MAX)).unwrap();
    let all = ab_join_topk(&a, &b, nz(4), one, one, nz(2), nz(37)).unwrap();
    assert_eq!(bits(&huge), bits(&all));
}

/// Past the row accumulator's candidate cap, where exact ties are resolved down to the nearest
/// `k` while tiles merge: every pair ties, or a handful of distances repeat many times.
#[test]
fn ties_beyond_the_candidate_cap_resolve_to_the_exact_nearest_k() {
    let one = scale(1.0);
    let constant = vec![7.0; 400];
    let period_four: Vec<f64> = (0..420).map(|k| f64::from([0, 2, 1, 3][k % 4])).collect();
    for (a, b) in [
        (&constant, &constant),
        (&period_four, &constant),
        (&period_four, &period_four),
    ] {
        for k in [1, 3, 8] {
            let expected = exact(a, b, 4, (one, one), k);
            for threads in [1, 3] {
                for (band, chunk, tile) in [(16, 8, 32), (64, 64, 64), (430, 430, 430)] {
                    let (actual, _) = ab_join_topk_tiled(
                        a,
                        b,
                        nz(4),
                        one,
                        one,
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                        nz(k),
                    )
                    .unwrap();
                    assert_eq!(
                        bits(&actual),
                        expected,
                        "k {k}, threads {threads}, shape {:?}",
                        (band, chunk, tile)
                    );
                }
            }
        }
    }
}
