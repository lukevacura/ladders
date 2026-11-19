#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The AB-join tile engines over a subset of the diagonals equal the exact reference over exactly
//! those pairs, on bits, at every thread count and tile shape and under every forced route,
//! whichever diagonals the subset holds, and cut only the tiles that hold one of them.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/route.rs"]
mod route;

use ladde::batch::ab_join_topk_over_tiled;
use ladde::batch::nonnorm::ab_join_over_tiled;
use ladde::entry::Entry;
use ladde::naive::{naive_ab_join_nonnorm_on, naive_ab_join_topk_on};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use proptest::prelude::*;
use route::{forces, nz, samples};

type EntryBits = (u64, u64, u64);

fn bits(rows: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<EntryBits>)> {
    let one = |e: Entry| {
        (
            e.rho().to_bits(),
            e.squared_distance().to_bits(),
            e.index().get().unwrap_or(u64::MAX),
        )
    };
    rows.iter()
        .map(|(row, es)| (*row, es.iter().copied().map(one).collect()))
        .collect()
}

/// The subsets the tests pick from, over diagonals `-(wa - 1)..wb`: none, one, runs, every
/// other, a block, a pseudo-random third and half, all, repeats, and numbers that are none.
fn subsets(wa: usize, wb: usize, salt: u64) -> Vec<Vec<i64>> {
    let all: Vec<i64> = (1 - wa as i64..wb as i64).collect();
    let mut state = salt | 1;
    let mut random = |per_mille: u64| -> Vec<i64> {
        all.iter()
            .copied()
            .filter(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) % 1000 < per_mille
            })
            .collect()
    };
    vec![
        vec![],
        vec![all[0]],
        vec![all[all.len() - 1]],
        vec![0],
        all.iter().copied().step_by(2).collect(),
        all.iter().copied().skip(2).take(7).collect(),
        random(300),
        random(500),
        all.clone(),
        all.iter().rev().chain(all.iter()).copied().collect(),
        vec![
            -(wa as i64),
            wb as i64,
            wb as i64 + 9,
            i64::MIN,
            i64::MAX,
            -(wa as i64) - 5,
        ],
    ]
}

const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)];
const THREADS: [usize; 2] = [1, 3];

fn assert_znorm(a: &[f64], b: &[f64], m: usize, scale: (Scale, Scale), k: usize, salt: u64) {
    let wins = |s: &[f64]| (s.len() + 1).saturating_sub(m);
    if wins(a) == 0 || wins(b) == 0 {
        return;
    }
    let (sa, sb) = (samples(a, scale.0), samples(b, scale.1));
    for diagonals in subsets(wins(a), wins(b), salt) {
        let expected = bits(&naive_ab_join_topk_on(&sa, &sb, m, nz(k), &diagonals));
        for threads in THREADS {
            for (band, chunk, tile) in SHAPES {
                let (rows, _) = ab_join_topk_over_tiled(
                    a,
                    b,
                    nz(m),
                    scale.0,
                    scale.1,
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                    nz(k),
                    &diagonals,
                )
                .unwrap();
                assert_eq!(
                    bits(&rows),
                    expected,
                    "k {k}, threads {threads}, shape {:?}, {} diagonals",
                    (band, chunk, tile),
                    diagonals.len()
                );
            }
        }
    }
}

fn assert_nonnorm(a: &[f64], b: &[f64], m: usize, salt: u64) {
    let unit = Scale::new(1.0).unwrap();
    let wins = |s: &[f64]| (s.len() + 1).saturating_sub(m);
    if wins(a) == 0 || wins(b) == 0 {
        return;
    }
    let (sa, sb) = (samples(a, unit), samples(b, unit));
    for diagonals in subsets(wins(a), wins(b), salt) {
        let expected: Vec<nonnorm::Row> = naive_ab_join_nonnorm_on(&sa, &sb, m, &diagonals)
            .into_iter()
            .map(|(row, entry)| nonnorm::entry_row(row, entry))
            .collect();
        for threads in THREADS {
            for (band, chunk, tile) in SHAPES {
                let (rows, _) = ab_join_over_tiled(
                    a,
                    b,
                    nz(m),
                    unit,
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                    &diagonals,
                );
                assert_eq!(
                    nonnorm::batch_rows(&rows),
                    expected,
                    "threads {threads}, shape {:?}, {} diagonals",
                    (band, chunk, tile),
                    diagonals.len()
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(8))]

    #[test]
    fn znorm_join_subsets_equal_the_reference(
        (a, m, sa) in route::cases(),
        (b, _, sb) in route::cases(),
        salt in any::<u64>(),
    ) {
    let _serial = forced_route(RouteForce::default());
        for k in [1, 3] {
            assert_znorm(&a, &b, m, (sa, sb), k, salt);
        }
    }

    #[test]
    fn znorm_join_subsets_equal_the_reference_on_every_route(
        (a, m, sa) in route::cases(),
        (b, _, sb) in route::cases(),
        seed in any::<u64>(),
    ) {
        for force in forces(seed) {
            let _forced = forced_route(force);
            assert_znorm(&a, &b, m, (sa, sb), 2, seed);
        }
    }

    #[test]
    fn nonnorm_join_subsets_equal_the_reference(
        a in prop_oneof![nonnorm::series(), nonnorm::tie_series()],
        b in prop_oneof![nonnorm::series(), nonnorm::tie_series()],
        m in 1usize..=6,
        salt in any::<u64>(),
    ) {
    let _serial = forced_route(RouteForce::default());
        assert_nonnorm(&a, &b, m, salt);
    }
}

#[test]
fn znorm_join_subsets_equal_the_reference_on_ties_and_gaps() {
    let _serial = forced_route(RouteForce::default());
    let scale = (Scale::new(1.0).unwrap(), Scale::new(1.0).unwrap());
    let flat = vec![7.0; 40];
    let runs: Vec<f64> = (0..60).map(|k| f64::from(((k / 11) % 3) as u8)).collect();
    let period: Vec<f64> = (0..50)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    let mut gapped: Vec<f64> = (0..55)
        .map(|k| f64::from(((k * k / 5 + k) % 11) as u8))
        .collect();
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
        for k in [1, 2, 200] {
            assert_znorm(a, b, 5, scale, k, 7);
        }
        assert_nonnorm(a, b, 5, 7);
    }
}

/// The tiles a join subset run cuts, worked out from the definition.
fn expected_tiles(
    (wa, wb): (usize, usize),
    (band_rows, tile_lanes): (usize, usize),
    diagonals: &[i64],
) -> usize {
    let slots: Vec<usize> = diagonals
        .iter()
        .filter_map(|&d| usize::try_from((wb as i64 - 1).checked_sub(d)?).ok())
        .collect();
    let mut tiles = 0;
    let mut start = 0;
    while start < wa {
        let end = (start + band_rows).min(wa);
        let lanes_end = end + wb - 1;
        let kept = (start..lanes_end)
            .step_by(tile_lanes)
            .filter(|&qa| {
                slots
                    .iter()
                    .any(|&s| (qa..(qa + tile_lanes).min(lanes_end)).contains(&s))
            })
            .count();
        tiles += kept.max(1);
        start = end;
    }
    tiles
}

#[test]
fn only_the_tiles_holding_a_visited_diagonal_are_cut() {
    let _serial = forced_route(RouteForce::default());
    let (a, b): (Vec<f64>, Vec<f64>) = (
        (0..60)
            .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
            .collect(),
        (0..45)
            .map(|k| f64::from(((k * 5 + k / 2) % 11) as u8))
            .collect(),
    );
    let (m, scale) = (5, Scale::new(1.0).unwrap());
    let wins = (a.len() + 1 - m, b.len() + 1 - m);
    for diagonals in subsets(wins.0, wins.1, 3) {
        for (band, chunk, tile) in [(4, 2, 4), (7, 3, 6), (200, 5, 9)] {
            let (_, znorm) = ab_join_topk_over_tiled(
                &a,
                &b,
                nz(m),
                scale,
                scale,
                nz(1),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(1),
                &diagonals,
            )
            .unwrap();
            let (_, plain) = ab_join_over_tiled(
                &a,
                &b,
                nz(m),
                scale,
                nz(1),
                nz(band),
                nz(chunk),
                nz(tile),
                &diagonals,
            );
            // A tile is a whole number of chunks wide, and a band is at most the rows.
            let tile = tile.div_ceil(chunk) * chunk;
            let want = expected_tiles(wins, (band, tile), &diagonals);
            assert_eq!(znorm.tiles, want, "{:?}", (band, tile));
            assert_eq!(plain.tiles, want, "{:?}", (band, tile));
        }
    }
}

#[test]
fn a_forced_wide_join_subset_run_runs_every_tile_wide() {
    use ladde::route::RouteForce;
    let a: Vec<f64> = (0..60)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let b: Vec<f64> = (0..45)
        .map(|k| f64::from(((k * 5 + k / 2) % 11) as u8))
        .collect();
    let scale = Scale::new(1.0).unwrap();
    let some: Vec<i64> = (-20..30).step_by(3).collect();
    let run = || {
        ab_join_topk_over_tiled(
            &a,
            &b,
            nz(5),
            scale,
            scale,
            nz(1),
            nz(8),
            nz(4),
            nz(8),
            nz(1),
            &some,
        )
        .unwrap()
        .1
    };
    let natural = {
        let guard = forced_route(RouteForce::default());
        let report = run();
        assert_eq!(guard.fired(), Default::default());
        report
    };
    assert!(natural.tiles > 0);
    assert_eq!(natural.wide_tiles, 0);
    let guard = forced_route(RouteForce {
        wide: true,
        ..RouteForce::default()
    });
    let wide = run();
    assert_eq!(wide.tiles, natural.tiles);
    assert_eq!(wide.wide_tiles, wide.tiles);
    assert!(guard.fired().wide_switches > 0);
}
