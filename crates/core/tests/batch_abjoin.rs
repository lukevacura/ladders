#![cfg(not(any(loom, shuttle)))]
//! The tiled, threaded AB-join equals the direct exact reference on bits at every thread count and
//! tiling: random A and B of different lengths, levels and scales, gaps, flat runs, repeats across
//! the two, B equal to A, one side too wide for an anchor, and every forced route.

use core::num::NonZeroUsize;

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    BatchReport, ab_join, ab_join_report, ab_join_report_tiled, ab_join_single_threaded,
};
use ladde::entry::Entry;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series};

type Bits = (u64, u64, u64, u64);

/// The exact reference join: a window with no distance in the hot passes' domain has one too.
fn exact(a: &[f64], scale_a: Scale, b: &[f64], scale_b: Scale, m: usize) -> Vec<Bits> {
    spill_run::exact_join(a, scale_a, b, scale_b, m)
}

const TILINGS: [(usize, usize, usize); 5] =
    [(1, 1, 1), (3, 2, 5), (8, 4, 4), (5, 3, 64), (64, 8, 64)];
const THREADS: [usize; 3] = [1, 2, 7];

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn bits(profile: &[(u64, Entry)]) -> Vec<Bits> {
    profile
        .iter()
        .map(|&(row, e)| {
            (
                row,
                e.rho().to_bits(),
                e.squared_distance().to_bits(),
                e.index().get().unwrap_or(u64::MAX),
            )
        })
        .collect()
}

fn scales() -> impl Strategy<Value = Scale> {
    prop_oneof![Just(MARKET_TICK), Just(0.5), Just(1.0)].prop_map(scale)
}

/// A series at scale 1 jumping between levels around the one-anchor bound for `m`, with gaps.
fn jumpy(m: usize) -> impl Strategy<Value = Vec<f64>> {
    let bound = (i32::MAX as usize / m) as f64;
    let levels = [0.0, bound, 3.0 * bound, -2.0 * bound];
    prop::collection::vec((0usize..8, -4i32..=4, 0u8..10), 0..50).prop_map(move |steps| {
        let mut level = 0.0;
        steps
            .into_iter()
            .map(|(jump, noise, kind)| {
                if let Some(&next) = levels.get(jump) {
                    level = next;
                }
                if kind == 0 {
                    f64::NAN
                } else {
                    level + f64::from(noise)
                }
            })
            .collect()
    })
}

fn forces(seed: u64) -> Vec<RouteForce> {
    let none = RouteForce::default();
    vec![
        none,
        RouteForce { wide: true, ..none },
        RouteForce {
            per_tile: true,
            anchor_seed: seed | 1,
            ..none
        },
    ]
}

/// Every thread count and tiling, and the default plan, against `expected`.
fn check_everywhere(
    a: &[f64],
    b: &[f64],
    m: usize,
    scales: (Scale, Scale),
    expected: &[Bits],
) -> Result<(), TestCaseError> {
    let (scale_a, scale_b) = scales;
    let m = nz(m);
    for threads in THREADS {
        let default = ab_join(a, b, m, scale_a, scale_b, nz(threads));
        prop_assert_eq!(&bits(&default), expected, "default threads {}", threads);
        for (band, chunk, tile) in TILINGS {
            let (profile, _) = ab_join_report_tiled(
                a,
                b,
                m,
                scale_a,
                scale_b,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            prop_assert_eq!(
                &bits(&profile),
                expected,
                "threads {} tiling {:?}",
                threads,
                (band, chunk, tile)
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn random_pairs_equal_naive_at_every_tiling(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        b in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        scale_a in scales(),
        scale_b in scales(),
    ) {
        let expected = exact(&a, scale_a, &b, scale_b, m);
        check_everywhere(&a, &b, m, (scale_a, scale_b), &expected)?;
    }

    #[test]
    fn a_series_joined_with_itself_equals_naive_at_every_tiling(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        scale_a in scales(),
    ) {
        let expected = exact(&a, scale_a, &a, scale_a, m);
        check_everywhere(&a, &a, m, (scale_a, scale_a), &expected)?;
    }

    #[test]
    fn repeats_across_the_two_sides_tie_to_the_smaller_index(
        m in 1usize..=4,
        a in (1usize..=4).prop_flat_map(|m| market_series(8..30, m)),
        cut in 0usize..8,
        copies in 1usize..4,
    ) {
        let piece = a[cut..].to_vec();
        let b: Vec<f64> = (0..copies).flat_map(|_| piece.iter().copied()).collect();
        let tick = scale(MARKET_TICK);
        let expected = exact(&a, tick, &b, tick, m);
        check_everywhere(&a, &b, m, (tick, tick), &expected)?;
    }

    #[test]
    fn a_span_too_wide_for_one_side_equals_naive_at_every_tiling(
        m in 1usize..=6,
        wide in (1usize..=6).prop_flat_map(jumpy),
        flat in (1usize..=6).prop_flat_map(|m| market_series(0..45, m)),
        wide_first in any::<bool>(),
    ) {
        let (one, tick) = (scale(1.0), scale(MARKET_TICK));
        let (a, b, scales) = if wide_first {
            (&wide, &flat, (one, tick))
        } else {
            (&flat, &wide, (tick, one))
        };
        let expected = exact(a, scales.0, b, scales.1, m);
        check_everywhere(a, b, m, scales, &expected)?;
    }

    #[test]
    fn every_forced_route_equals_naive(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(jumpy),
        b in (1usize..=6).prop_flat_map(jumpy),
        seed in any::<u64>(),
    ) {
        let one = scale(1.0);
        let expected = exact(&a, one, &b, one, m);
        for force in forces(seed) {
            let _forced = forced_route(force);
            let (profile, report) = ab_join_report_tiled(
                &a, &b, nz(m), one, one, nz(2), nz(3), nz(2), nz(5),
            );
            if force.wide {
                prop_assert_eq!(report.wide_tiles, report.tiles);
            }
            prop_assert_eq!(&bits(&profile), &expected, "{:?}", force);
        }
    }
}

/// A random walk of `n` ticks at a market level, with ties and a few gaps.
fn walk(n: usize, seed: u64, level: f64) -> Vec<f64> {
    let mut state = seed | 1;
    let mut x = level;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            x += ((state >> 58) as i64 % 3 - 1) as f64 * 0.01;
            if i % 97 == 41 { f64::NAN } else { x }
        })
        .collect()
}

#[test]
fn threads_change_no_bit_on_a_series_long_enough_to_run_threaded() {
    let _unforced = forced_route(RouteForce::default());
    let tick = scale(MARKET_TICK);
    let (a, b) = (walk(3200, 7, 100.0), walk(2700, 11, 100.5));
    let m = nz(24);
    let reference = bits(&ab_join_single_threaded(&a, &b, m, tick, tick));
    for threads in [1usize, 2, 3, 7, 16] {
        let (profile, report) = ab_join_report(&a, &b, m, tick, tick, nz(threads));
        assert_eq!(bits(&profile), reference, "threads {threads}");
        assert_eq!(report.wide_tiles, 0);
        if threads > 1 && cfg!(feature = "parallel") {
            assert!(report.tiles > 1, "threads {threads} ran {report:?}");
        }
    }
}

#[test]
fn a_wide_side_is_counted_and_changes_no_bit() {
    let tick = scale(MARKET_TICK);
    let (a, b) = (walk(900, 3, 100.0), walk(700, 5, 100.0));
    let m = nz(16);
    let reference = bits(&ab_join_single_threaded(&a, &b, m, tick, tick));
    let wide = RouteForce {
        wide: true,
        ..RouteForce::default()
    };
    let _forced = forced_route(wide);
    let (profile, report) =
        ab_join_report_tiled(&a, &b, m, tick, tick, nz(3), nz(8), nz(4), nz(16));
    assert_eq!(bits(&profile), reference);
    assert!(report.tiles > 1);
    assert_eq!(report.wide_tiles, report.tiles);
}

#[test]
fn one_thread_is_one_tile_and_an_empty_side_has_none() {
    let _unforced = forced_route(RouteForce::default());
    let tick = scale(MARKET_TICK);
    let a = walk(60, 1, 100.0);
    let (profile, report) = ab_join_report(&a, &a, nz(8), tick, tick, nz(1));
    assert_eq!(profile.len(), 53);
    assert_eq!(report, BatchReport::of(1, 0));

    let short = [1.0, 2.0];
    for threads in [1, 4] {
        let (rows, report) = ab_join_report(&a, &short, nz(8), tick, tick, nz(threads));
        assert_eq!(rows.len(), 53);
        assert!(rows.iter().all(|(_, e)| e.index().get().is_none()));
        assert_eq!(report.tiles, 0);
        assert!(ab_join(&short, &a, nz(8), tick, tick, nz(threads)).is_empty());
    }
}

#[test]
fn exact_ties_across_the_sides_go_to_the_smaller_b_index_at_every_tiling() {
    let tick = scale(1.0);
    let period = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0][i % 7])
            .collect()
    };
    let (a, b) = (period(40), period(75));
    let m = nz(5);
    for threads in THREADS {
        for (band, chunk, tile) in TILINGS {
            let (profile, _) = ab_join_report_tiled(
                &a,
                &b,
                m,
                tick,
                tick,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            for (row, entry) in &profile {
                assert_eq!(entry.index().get(), Some(row % 7), "row {row}");
                assert_eq!(entry.squared_distance().to_bits(), 0.0f64.to_bits());
            }
        }
    }
}
