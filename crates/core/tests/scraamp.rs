#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The non-normalized anytime profile after every update equals the exact reference restricted
//! to the diagonals visited so far, on bits, at every thread count; after the last update it
//! equals the full non-normalized two-sided profile. On series of ties, gaps at every boundary,
//! ticks around the `u64` bound, from any source of the order.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::batch::nonnorm::two_sided_profile;
use ladde::naive::naive_two_sided_nonnorm_on;
use ladde::scrump::{DiagonalSource, Explicit, Scraamp, ScrumpError, Shuffled};
use nonnorm::*;
use proptest::prelude::*;

const THREADS: [usize; 3] = [1, 2, 7];

fn naive_sides_on(series: &[f64], m: usize, diagonals: &[i64]) -> Vec<Sides> {
    naive_two_sided_nonnorm_on(&samples(series, unit()), m, diagonals)
        .into_iter()
        .map(|(row, sides)| {
            let rows = [sides.left, sides.right, sides.nearest].map(|e| entry_row(row, e));
            (row, rows.map(|(_, squared, index)| (squared, index)))
        })
        .collect()
}

fn assert_anytime(
    series: &[f64],
    m: usize,
    percentage: f64,
    order_of: &dyn Fn(i64, usize) -> Vec<i64>,
) {
    let Ok((first, count)) = Scraamp::diagonals(series.len(), nz(m)) else {
        return;
    };
    for threads in THREADS {
        let order = order_of(first, count);
        let mut scraamp = Scraamp::new(
            series,
            nz(m),
            unit(),
            nz(threads),
            percentage,
            Explicit(order.clone()),
        )
        .unwrap();
        assert_eq!(
            batch_sides(&scraamp.profile().unwrap()),
            naive_sides_on(series, m, &[]),
            "threads {threads}: before any update"
        );
        let mut updates = 0;
        while !scraamp.is_complete() {
            scraamp.update().unwrap();
            updates += 1;
            let visited = scraamp.visited();
            assert_eq!(visited, &order[..visited.len()]);
            assert_eq!(
                batch_sides(&scraamp.profile().unwrap()),
                naive_sides_on(series, m, visited),
                "threads {threads}, update {updates} of {}, {} diagonals",
                scraamp.chunks(),
                visited.len()
            );
        }
        assert_eq!(scraamp.visited(), order.as_slice());
        let full = two_sided_profile(series, nz(m), unit(), nz(threads));
        let finished = batch_sides(&scraamp.profile().unwrap());
        assert_eq!(
            finished,
            batch_sides(&full),
            "threads {threads}: after the last update"
        );
        scraamp.update().unwrap();
        assert_eq!(scraamp.updates(), updates, "an update after the last");
        assert_eq!(batch_sides(&scraamp.profile().unwrap()), finished);
    }
}

fn assert_all(series: &[f64], m: usize, seed: u64) {
    for percentage in [0.07, 0.3, 1.0] {
        assert_anytime(series, m, percentage, &|first, count| {
            Shuffled::seeded(seed).order(first, count).unwrap()
        });
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]

    #[test]
    fn anytime_equals_the_restricted_reference(
        series in prop_oneof![series(), tie_series()],
        m in 1usize..=6,
        seed in any::<u64>(),
    ) {
        assert_all(&series, m, seed);
    }
}

#[test]
fn nan_runs_at_the_boundaries_equal_the_reference() {
    for gaps in [(0..5, 70..80), (0..1, 79..80), (30..45, 46..47)] {
        let mut series: Vec<f64> = (0..80)
            .map(|k| f64::from(((k * k / 5 + k) % 11) as u8))
            .collect();
        for k in gaps.0.chain(gaps.1) {
            series[k] = f64::NAN;
        }
        assert_all(&series, 6, 9);
    }
}

#[test]
fn a_range_past_the_u64_bound_equals_the_reference() {
    let series: Vec<f64> = (0..90)
        .map(|k| {
            let step = f64::from(((k * 5 + k / 4) % 9) as u8);
            if (30..60).contains(&k) {
                step + (1u64 << 40) as f64
            } else {
                step
            }
        })
        .collect();
    assert_all(&series, 6, 4);
}

#[test]
fn explicit_orders_visit_in_that_order() {
    let series: Vec<f64> = (0..70)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let (first, count) = Scraamp::diagonals(series.len(), nz(5)).unwrap();
    let ascending: Vec<i64> = (first..first + count as i64).collect();
    let descending: Vec<i64> = ascending.iter().rev().copied().collect();
    for order in [ascending, descending] {
        assert_anytime(&series, 5, 0.2, &|_, _| order.clone());
    }
}

#[test]
fn the_same_seed_gives_the_same_profile_on_any_thread_count() {
    let series: Vec<f64> = (0..300)
        .map(|k| f64::from(((k * k + 3 * k) % 29) as u8))
        .collect();
    let run = |seed: u64, threads: usize| {
        let mut scraamp = Scraamp::new(
            &series,
            nz(8),
            unit(),
            nz(threads),
            0.25,
            Shuffled::seeded(seed),
        )
        .unwrap();
        scraamp.update().unwrap();
        scraamp.update().unwrap();
        (
            scraamp.visited().to_vec(),
            batch_sides(&scraamp.profile().unwrap()),
        )
    };
    let base = run(5, 1);
    for threads in THREADS {
        assert_eq!(run(5, threads), base, "threads {threads}");
    }
    assert_ne!(run(6, 1).0, base.0);
}

#[test]
fn bad_input_is_an_error_never_a_panic_or_a_clamp() {
    let series: Vec<f64> = (0..40).map(|k| f64::from((k * 3 % 7) as u8)).collect();
    let make = |m: usize, percentage: f64, order: Option<Vec<i64>>| {
        let (first, count) = Scraamp::diagonals(series.len(), nz(m)).unwrap_or((0, 0));
        let order = order.unwrap_or_else(|| (first..first + count as i64).collect());
        Scraamp::new(&series, nz(m), unit(), nz(1), percentage, Explicit(order))
            .map(|scraamp| scraamp.chunks())
    };
    assert_eq!(make(4, 0.5, None), Ok(2));
    for percentage in [f64::NAN, 0.0, -0.5, 1.0000001, f64::INFINITY] {
        assert_eq!(
            make(4, percentage, None),
            Err(ScrumpError::Percentage),
            "{percentage}"
        );
    }
    let (first, count) = Scraamp::diagonals(series.len(), nz(4)).unwrap();
    for order in [
        vec![],
        (first..first + count as i64 - 1).collect(),
        (first..first + count as i64 + 1).collect(),
        std::iter::repeat_n(first, count).collect(),
        (first - 1..first + count as i64 - 1).collect(),
    ] {
        assert_eq!(
            make(4, 0.5, Some(order.clone())),
            Err(ScrumpError::NotAPermutation),
            "{order:?}"
        );
    }
    for m in [40, 41, 100] {
        assert_eq!(make(m, 0.5, None), Err(ScrumpError::NoDiagonals), "m {m}");
    }
}
