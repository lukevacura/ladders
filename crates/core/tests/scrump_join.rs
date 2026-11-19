#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The anytime joins (z-normalized and non-normalized) after every update equal the exact join
//! restricted to the diagonals visited so far (for the z-normalized join, and to the pairs of the
//! windows with no distance in the hot passes, ranked up front), on bits, at every thread count;
//! after the last update they equal the full join. Series of different lengths, with gaps, flat
//! runs, exact repeats and heavy ties, for `k` below, at and above the candidate count.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use core::num::NonZeroUsize;

use ladde::batch::{ab_join_topk, nonnorm::ab_join};
use ladde::entry::Entry;
use ladde::naive::naive_ab_join_nonnorm_on;
use ladde::quantize::{Scale, to_ticks};
use ladde::scrump::{DiagonalSource, Explicit, ScraampJoin, ScrumpError, ScrumpJoin, Shuffled};
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

type EntryBits = (u64, u64, u64);

fn entry_bits(entry: Entry) -> EntryBits {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn bits(rows: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<EntryBits>)> {
    rows.iter()
        .map(|(row, entries)| (*row, entries.iter().copied().map(entry_bits).collect()))
        .collect()
}

fn samples(series: &[f64]) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, unit())).collect()
}

const THREADS: [usize; 3] = [1, 2, 7];

/// The z-normalized anytime join's reference after visiting `visited`, as [`bits`] lays it out.
fn anytime(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    visited: &[i64],
) -> Vec<(u64, Vec<EntryBits>)> {
    spill_run::anytime_join_topk((a, unit()), (b, unit()), m, k, visited)
        .into_iter()
        .enumerate()
        .map(|(row, kept)| {
            (
                row as u64,
                kept.into_iter().map(|e| (e.1, e.2, e.3)).collect(),
            )
        })
        .collect()
}

fn assert_znorm(a: &[f64], b: &[f64], m: usize, k: usize, percentage: f64, seed: u64) {
    let Ok((first, count)) = ScrumpJoin::diagonals(a.len(), b.len(), nz(m)) else {
        return;
    };
    for threads in THREADS {
        let order = Shuffled::seeded(seed).order(first, count).unwrap();
        let mut join = ScrumpJoin::new(
            a,
            b,
            nz(m),
            (unit(), unit()),
            nz(threads),
            nz(k),
            percentage,
            Explicit(order.clone()),
        )
        .unwrap();
        assert_eq!(
            bits(&join.profile().unwrap()),
            anytime(a, b, m, k, &[]),
            "k {k}, threads {threads}: before any update"
        );
        while !join.is_complete() {
            join.update().unwrap();
            let visited = join.visited();
            assert_eq!(visited, &order[..visited.len()]);
            assert_eq!(
                bits(&join.profile().unwrap()),
                anytime(a, b, m, k, visited),
                "k {k}, threads {threads}, update {} of {}",
                join.updates(),
                join.chunks()
            );
        }
        assert_eq!(join.visited(), order.as_slice());
        let full = ab_join_topk(a, b, nz(m), unit(), unit(), nz(threads), nz(k)).unwrap();
        let done = bits(&join.profile().unwrap());
        assert_eq!(
            done,
            bits(&full),
            "k {k}, threads {threads}: the last update"
        );
        join.update().unwrap();
        assert_eq!(
            bits(&join.profile().unwrap()),
            done,
            "an update after the last"
        );
    }
}

fn assert_nonnorm(a: &[f64], b: &[f64], m: usize, percentage: f64, seed: u64) {
    let Ok((first, count)) = ScraampJoin::diagonals(a.len(), b.len(), nz(m)) else {
        return;
    };
    let (sa, sb) = (samples(a), samples(b));
    let rows = |profile: &[(u64, ladde::nonnorm::NonNormNeighbor)]| nonnorm::batch_rows(profile);
    let expected = |visited: &[i64]| -> Vec<nonnorm::Row> {
        naive_ab_join_nonnorm_on(&sa, &sb, m, visited)
            .into_iter()
            .map(|(row, entry)| nonnorm::entry_row(row, entry))
            .collect()
    };
    for threads in THREADS {
        let order = Shuffled::seeded(seed).order(first, count).unwrap();
        let mut join = ScraampJoin::new(
            a,
            b,
            nz(m),
            unit(),
            nz(threads),
            percentage,
            Explicit(order.clone()),
        )
        .unwrap();
        assert_eq!(rows(&join.profile().unwrap()), expected(&[]));
        while !join.is_complete() {
            join.update().unwrap();
            assert_eq!(
                rows(&join.profile().unwrap()),
                expected(join.visited()),
                "threads {threads}, update {} of {}",
                join.updates(),
                join.chunks()
            );
        }
        assert_eq!(join.visited(), order.as_slice());
        let full = ab_join(a, b, nz(m), unit(), nz(threads));
        let done = rows(&join.profile().unwrap());
        assert_eq!(done, rows(&full), "threads {threads}: the last update");
        join.update().unwrap();
        assert_eq!(rows(&join.profile().unwrap()), done);
    }
}

fn assert_all(a: &[f64], b: &[f64], m: usize, seed: u64) {
    for percentage in [0.07, 0.3, 1.0] {
        for k in [1, 2, 200] {
            assert_znorm(a, b, m, k, percentage, seed);
        }
        assert_nonnorm(a, b, m, percentage, seed);
    }
}

fn walk(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|k| f64::from(((k * k + salt * k / 3 + salt) % 17) as u8))
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded(10))]

    #[test]
    fn joins_equal_the_restricted_reference(
        a in prop_oneof![nonnorm::series(), nonnorm::tie_series()],
        b in prop_oneof![nonnorm::series(), nonnorm::tie_series()],
        m in 1usize..=6,
        seed in any::<u64>(),
    ) {
        assert_all(&a, &b, m, seed);
    }
}

#[test]
fn series_of_different_lengths_equal_the_reference() {
    for (la, lb) in [(40, 70), (70, 40), (60, 60), (25, 8)] {
        assert_all(&walk(la, 3), &walk(lb, 5), 5, la as u64);
    }
}

#[test]
fn ties_and_gaps_equal_the_reference() {
    let flat = vec![7.0; 40];
    let runs: Vec<f64> = (0..60).map(|k| f64::from(((k / 11) % 3) as u8)).collect();
    let period: Vec<f64> = (0..50)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    let mut gapped = walk(55, 2);
    for k in [0, 1, 2, 27, 28, 52, 53, 54] {
        gapped[k] = f64::NAN;
    }
    for (a, b) in [
        (&flat, &runs),
        (&runs, &period),
        (&period, &period),
        (&gapped, &runs),
        (&runs, &gapped),
    ] {
        assert_all(a, b, 4, 1);
    }
}

#[test]
fn a_series_joined_with_itself_meets_itself() {
    let a = walk(50, 4);
    assert_znorm(&a, &a, 6, 3, 0.2, 2);
    assert_nonnorm(&a, &a, 6, 0.2, 2);
}

#[test]
fn bad_input_is_an_error_never_a_panic_or_a_clamp() {
    let (a, b) = (walk(40, 1), walk(30, 2));
    let (first, count) = ScrumpJoin::diagonals(a.len(), b.len(), nz(5)).unwrap();
    assert_eq!((first, count), (-35, 40 - 5 + 1 + 30 - 5 + 1 - 1));
    let ascending = || Explicit((first..first + count as i64).collect());
    let make = |percentage: f64, order: Explicit, m: usize| {
        ScrumpJoin::new(
            &a,
            &b,
            nz(m),
            (unit(), unit()),
            nz(1),
            nz(1),
            percentage,
            order,
        )
        .map(|join| join.chunks())
    };
    assert!(make(0.5, ascending(), 5).is_ok());
    for percentage in [f64::NAN, 0.0, -0.5, 1.0000001, f64::INFINITY] {
        assert_eq!(
            make(percentage, ascending(), 5),
            Err(ScrumpError::Percentage)
        );
    }
    for order in [
        vec![],
        (first..first + count as i64 - 1).collect(),
        (first..first + count as i64 + 1).collect(),
        std::iter::repeat_n(first, count).collect::<Vec<_>>(),
        (first - 1..first + count as i64 - 1).collect(),
    ] {
        assert_eq!(
            make(0.5, Explicit(order), 5),
            Err(ScrumpError::NotAPermutation)
        );
    }
    for m in [31, 40, 100] {
        assert_eq!(
            make(0.5, ascending(), m),
            Err(ScrumpError::NoDiagonals),
            "m {m}"
        );
    }
    assert_eq!(
        ScrumpJoin::diagonals(0, 30, nz(5)),
        Err(ScrumpError::NoDiagonals)
    );
    assert_eq!(
        ScraampJoin::diagonals(40, 4, nz(5)),
        Err(ScrumpError::NoDiagonals)
    );
}

/// STUMPY 1.14.1's chunks for an AB-join of 12 against 9 windows (`m = 4`) at `percentage = 0.2`:
/// the diagonals each update has visited, in a fixed order.
#[test]
fn the_chunks_are_stumpys() {
    let (a, b) = (walk(15, 1), walk(12, 2));
    let (first, count) = ScrumpJoin::diagonals(a.len(), b.len(), nz(4)).unwrap();
    assert_eq!((first, count), (-11, 20));
    let order: Vec<i64> = (0..count as i64)
        .map(|at| first + (at * 7) % count as i64)
        .collect();
    let mut join = ScrumpJoin::new(
        &a,
        &b,
        nz(4),
        (unit(), unit()),
        nz(1),
        nz(1),
        0.2,
        Explicit(order.clone()),
    )
    .unwrap();
    let mut visited = Vec::new();
    while !join.is_complete() {
        join.update().unwrap();
        visited.push(join.visited().len());
    }
    assert_eq!(visited, STUMPY_JOIN_CHUNKS);
}

/// From `core._get_array_ranges(core._count_diagonal_ndist(order, 4, 15, 12), 5, True)` of
/// STUMPY 1.14.1 on the order of the test above.
const STUMPY_JOIN_CHUNKS: [usize; 5] = [5, 9, 13, 17, 20];
