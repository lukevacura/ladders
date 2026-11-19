#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The z-normalized anytime profile after every update equals the exact reference restricted to
//! the diagonals visited so far, on bits, at every thread count; after the last update it equals
//! the full two-sided top-k profile. On series with gaps, flat runs, exact repeats and heavy ties,
//! for `k` below, at and above the candidate count, from any source of the order.

use core::num::NonZeroUsize;

use ladde::batch::two_sided_topk_profile;
use ladde::entry::{Entry, TwoSidedTopk};
use ladde::naive::naive_two_sided_topk_on;
use ladde::quantize::{Scale, to_ticks};
use ladde::scrump::{DiagonalSource, Explicit, Scrump, ScrumpError, Shuffled};
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series_and_m};

type EntryBits = (u64, u64, u64);
type RowBits = (u64, [EntryBits; 2], Vec<EntryBits>);

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn entry_bits(entry: Entry) -> EntryBits {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<RowBits> {
    profile
        .iter()
        .map(|(row, sides)| {
            (
                *row,
                [sides.left, sides.right].map(entry_bits),
                sides.nearest.iter().copied().map(entry_bits).collect(),
            )
        })
        .collect()
}

fn samples(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

const KS: [usize; 4] = [1, 2, 3, 200];
const THREADS: [usize; 3] = [1, 2, 7];

/// Every update of `source`'s order at `percentage` equals the reference over its visited
/// diagonals, and the last equals the full profile.
fn assert_anytime(
    series: &[f64],
    m: usize,
    scale: Scale,
    percentage: f64,
    ks: &[usize],
    order_of: &dyn Fn(i64, usize) -> Vec<i64>,
) {
    let samples = samples(series, scale);
    let Ok((first, count)) = Scrump::diagonals(series.len(), nz(m)) else {
        return;
    };
    for &k in ks {
        for threads in THREADS {
            let order = order_of(first, count);
            let mut scrump = Scrump::new(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(k),
                percentage,
                Explicit(order.clone()),
            )
            .unwrap();
            let empty = naive_two_sided_topk_on(&samples, m, nz(k), &[]);
            assert_eq!(
                bits(&scrump.profile().unwrap()),
                bits(&empty),
                "k {k}, threads {threads}: before any update"
            );
            let mut updates = 0;
            while !scrump.is_complete() {
                scrump.update().unwrap();
                updates += 1;
                assert_eq!(scrump.updates(), updates);
                let visited = scrump.visited();
                assert_eq!(visited, &order[..visited.len()]);
                let expected = naive_two_sided_topk_on(&samples, m, nz(k), visited);
                assert_eq!(
                    bits(&scrump.profile().unwrap()),
                    bits(&expected),
                    "k {k}, threads {threads}, update {updates} of {}, {} diagonals",
                    scrump.chunks(),
                    visited.len()
                );
            }
            assert_eq!(scrump.visited(), order.as_slice());
            let full = two_sided_topk_profile(series, nz(m), scale, nz(threads), nz(k)).unwrap();
            assert_eq!(
                bits(&scrump.profile().unwrap()),
                bits(&full),
                "k {k}, threads {threads}: after the last update"
            );
            let before = bits(&scrump.profile().unwrap());
            scrump.update().unwrap();
            assert_eq!(scrump.updates(), updates, "an update after the last");
            assert_eq!(bits(&scrump.profile().unwrap()), before);
        }
    }
}

fn shuffled(seed: u64) -> impl Fn(i64, usize) -> Vec<i64> {
    move |first, count| Shuffled::seeded(seed).order(first, count).unwrap()
}

fn assert_all(series: &[f64], m: usize, scale: Scale, seed: u64) {
    for percentage in [0.07, 0.3, 1.0] {
        assert_anytime(series, m, scale, percentage, &KS, &shuffled(seed));
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(12))]

    #[test]
    fn anytime_equals_the_restricted_reference_on_market_series(
        (series, m) in market_series_and_m(0..70),
        seed in any::<u64>(),
    ) {
        assert_all(&series, m, Scale::new(MARKET_TICK).unwrap(), seed);
    }
}

#[test]
fn a_repeating_series_ties_to_the_smaller_index() {
    let series: Vec<f64> = (0..60)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    assert_all(&series, 4, Scale::new(1.0).unwrap(), 1);
}

#[test]
fn a_constant_series_equals_the_reference() {
    assert_all(&[7.0; 50], 4, Scale::new(1.0).unwrap(), 2);
}

#[test]
fn constant_runs_between_levels_equal_the_reference() {
    let series: Vec<f64> = (0..150).map(|k| f64::from(((k / 13) % 3) as u8)).collect();
    assert_all(&series, 5, Scale::new(1.0).unwrap(), 3);
}

#[test]
fn period_k_repeats_equal_the_reference() {
    for period in [2, 4, 7] {
        let series: Vec<f64> = (0..90)
            .map(|k| f64::from(((k % period) * 5 % 7) as u8))
            .collect();
        assert_all(&series, 6, Scale::new(1.0).unwrap(), period as u64);
    }
}

#[test]
fn nan_runs_at_the_boundaries_equal_the_reference() {
    let scale = Scale::new(1.0).unwrap();
    for gaps in [(0..5, 70..80), (0..1, 79..80), (30..45, 46..47)] {
        let mut series: Vec<f64> = (0..80)
            .map(|k| f64::from(((k * k / 5 + k) % 11) as u8))
            .collect();
        for k in gaps.0.chain(gaps.1) {
            series[k] = f64::NAN;
        }
        assert_all(&series, 6, scale, 9);
    }
}

/// STUMPY 1.14.1's chunks for 20 windows at `m = 8` (so diagonals 3 to 19) in the order its
/// seeded permutation gave and `percentage = 0.1`: how many diagonals each update has visited.
#[test]
fn the_chunks_are_stumpys() {
    let series: Vec<f64> = (0..27).map(|k| f64::from((k * 5 % 11) as u8)).collect();
    let order = vec![6, 16, 10, 5, 9, 13, 7, 4, 17, 3, 19, 18, 12, 11, 15, 14, 8];
    let mut scrump = Scrump::new(
        &series,
        nz(8),
        Scale::new(1.0).unwrap(),
        nz(1),
        nz(1),
        0.1,
        Explicit(order.clone()),
    )
    .unwrap();
    let mut visited = Vec::new();
    while !scrump.is_complete() {
        scrump.update().unwrap();
        visited.push(scrump.visited().len());
    }
    assert_eq!(visited, [2, 4, 5, 7, 8, 9, 10, 14, 16, 17]);
    assert_eq!(scrump.chunks(), 10);
    assert_eq!(scrump.visited(), order.as_slice());
}

#[test]
fn explicit_orders_visit_in_that_order() {
    let series: Vec<f64> = (0..70)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let scale = Scale::new(1.0).unwrap();
    let (first, count) = Scrump::diagonals(series.len(), nz(5)).unwrap();
    let ascending: Vec<i64> = (first..first + count as i64).collect();
    let descending: Vec<i64> = ascending.iter().rev().copied().collect();
    for order in [ascending, descending] {
        assert_anytime(&series, 5, scale, 0.2, &[1, 3], &|_, _| order.clone());
        let mut scrump = Scrump::new(
            &series,
            nz(5),
            scale,
            nz(1),
            nz(1),
            0.2,
            Explicit(order.clone()),
        )
        .unwrap();
        while !scrump.is_complete() {
            scrump.update().unwrap();
        }
        assert_eq!(scrump.visited(), order.as_slice());
    }
}

#[test]
fn the_same_seed_gives_the_same_profile_on_any_thread_count() {
    let series: Vec<f64> = (0..300)
        .map(|k| f64::from(((k * k + 3 * k) % 29) as u8))
        .collect();
    let scale = Scale::new(1.0).unwrap();
    let run = |seed: u64, threads: usize| {
        let mut scrump = Scrump::new(
            &series,
            nz(8),
            scale,
            nz(threads),
            nz(2),
            0.25,
            Shuffled::seeded(seed),
        )
        .unwrap();
        scrump.update().unwrap();
        scrump.update().unwrap();
        (scrump.visited().to_vec(), bits(&scrump.profile().unwrap()))
    };
    let base = run(5, 1);
    for threads in [1, 2, 7] {
        assert_eq!(run(5, threads), base, "threads {threads}");
    }
    assert_ne!(run(6, 1).0, base.0);
}

#[test]
fn nothing_is_visited_before_the_first_update() {
    let series: Vec<f64> = (0..40).map(|k| f64::from((k * 3 % 7) as u8)).collect();
    let scrump = Scrump::new(
        &series,
        nz(4),
        Scale::new(1.0).unwrap(),
        nz(1),
        nz(2),
        0.5,
        Shuffled::seeded(1),
    )
    .unwrap();
    assert!(scrump.visited().is_empty());
    assert_eq!(scrump.updates(), 0);
    assert!(!scrump.is_complete());
    for (_, row) in scrump.profile().unwrap() {
        assert_eq!(row.left.index().get(), None);
        assert_eq!(row.right.index().get(), None);
        assert!(row.nearest.is_empty());
    }
}

#[test]
fn bad_input_is_an_error_never_a_panic_or_a_clamp() {
    let series: Vec<f64> = (0..40).map(|k| f64::from((k * 3 % 7) as u8)).collect();
    let scale = Scale::new(1.0).unwrap();
    let make = |m: usize, percentage: f64, source: &dyn Fn() -> Explicit| {
        Scrump::new(&series, nz(m), scale, nz(1), nz(1), percentage, source())
            .map(|scrump| scrump.chunks())
    };
    let (first, count) = Scrump::diagonals(series.len(), nz(4)).unwrap();
    let ascending = || Explicit((first..first + count as i64).collect());
    assert_eq!(make(4, 0.5, &ascending), Ok(2));
    for percentage in [
        f64::NAN,
        0.0,
        -0.0,
        -0.5,
        1.0000001,
        2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(
            make(4, percentage, &ascending),
            Err(ScrumpError::Percentage),
            "{percentage}"
        );
    }
    let orders: [(&str, Vec<i64>); 6] = [
        ("empty", vec![]),
        ("short", (first..first + count as i64 - 1).collect()),
        ("long", (first..first + count as i64 + 1).collect()),
        ("repeat", {
            let mut order: Vec<i64> = (first..first + count as i64).collect();
            order[1] = order[0];
            order
        }),
        ("below", {
            let mut order: Vec<i64> = (first..first + count as i64).collect();
            order[0] = first - 1;
            order
        }),
        ("above", {
            let mut order: Vec<i64> = (first..first + count as i64).collect();
            order[0] = first + count as i64;
            order
        }),
    ];
    for (name, order) in orders {
        assert_eq!(
            make(4, 0.5, &|| Explicit(order.clone())),
            Err(ScrumpError::NotAPermutation),
            "{name}"
        );
    }
    for m in [40, 41, 100] {
        assert_eq!(
            make(m, 0.5, &ascending),
            Err(ScrumpError::NoDiagonals),
            "m {m}"
        );
        assert_eq!(
            Scrump::diagonals(series.len(), nz(m)),
            Err(ScrumpError::NoDiagonals)
        );
    }
    assert_eq!(
        Scrump::new(&[], nz(4), scale, nz(1), nz(1), 0.5, Shuffled::seeded(1)).map(|_| ()),
        Err(ScrumpError::NoDiagonals)
    );
}
