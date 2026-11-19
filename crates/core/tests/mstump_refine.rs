#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The nearest window of a multi-dimensional profile follows the exact means, not the rounding of
//! their `f64` values: on windows whose two-dimension means agree to 1e-13 (equal as `f64`) and on
//! tie-heavy z-normalized inputs, the batch profile at any thread count, tile shape and route
//! (with the estimate screen or without) and the reference both equal the neighbors an independent exact computation (integers and 400-digit
//! decimals, in Python) gives. On random walks no mean needs the exact decision.

#[path = "common/multi.rs"]
mod multi;
#[path = "common/refine_cases.rs"]
mod refine_cases;

use ladde::batch::multi::{maamp_report_tiled, mstump_report_tiled};
use ladde::multidim::{DimensionDistance, MultiProfile, NonFinitePolicy, exact_decisions};
use ladde::naive::{naive_multi_profile, naive_multi_profile_nonnorm};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use multi::*;
use refine_cases::{Case, NONNORM, ZNORM};
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn neighbors<D: DimensionDistance>(profile: &MultiProfile<D>) -> Vec<Vec<u64>> {
    profile
        .index
        .iter()
        .map(|row| row.iter().map(|i| i.get().unwrap_or(u64::MAX)).collect())
        .collect()
}

/// The estimate screen, and the forced wide route on which every cell takes the exact path.
fn routes() -> [RouteForce; 2] {
    [
        RouteForce::default(),
        RouteForce {
            wide: true,
            ..RouteForce::default()
        },
    ]
}

fn expected(case: &Case) -> Vec<Vec<u64>> {
    case.expected.iter().map(|row| row.to_vec()).collect()
}

fn samples(case: &Case) -> Vec<Vec<Option<i64>>> {
    case.dimensions
        .iter()
        .map(|series| ticks(series, unit()))
        .collect()
}

/// Runs `case` on every thread count and tile shape: the exact decision fires, and the
/// neighbors are the exact ones.
fn check_nonnorm(case: &Case) {
    let n = case.dimensions[0].len();
    let owned = samples(case);
    let refs: Vec<&[Option<i64>]> = owned.iter().map(Vec::as_slice).collect();
    // Each route holds the guard through its reference too: the exact decisions are counted
    // process-wide.
    for (force, threads) in routes().into_iter().flat_map(|f| THREADS.map(|t| (f, t))) {
        let _forced = forced_route(force);
        let reference = naive_multi_profile_nonnorm(
            &refs,
            case.m,
            case.include,
            case.discords,
            NonFinitePolicy::Missing,
        )
        .unwrap();
        assert_eq!(neighbors(&reference), expected(case), "reference");
        for (band, chunk, tile) in shapes(n) {
            let before = exact_decisions();
            let (profile, _) = maamp_report_tiled(
                case.dimensions,
                nz(case.m),
                unit(),
                case.include,
                case.discords,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            assert!(exact_decisions() > before, "the exact path ran");
            assert_eq!(neighbors(&profile), expected(case), "{threads} {band}");
            assert_eq!(profile, reference);
        }
    }
}

fn check_znorm(case: &Case) {
    let n = case.dimensions[0].len();
    let scales = vec![unit(); case.dimensions.len()];
    let owned = samples(case);
    let refs: Vec<&[Option<i64>]> = owned.iter().map(Vec::as_slice).collect();
    // Each route holds the guard through its reference too: the exact decisions are counted
    // process-wide.
    for (force, threads) in routes().into_iter().flat_map(|f| THREADS.map(|t| (f, t))) {
        let _forced = forced_route(force);
        let reference = naive_multi_profile(
            &refs,
            case.m,
            case.include,
            case.discords,
            NonFinitePolicy::Missing,
        )
        .unwrap();
        assert_eq!(neighbors(&reference), expected(case), "reference");
        for (band, chunk, tile) in shapes(n) {
            let before = exact_decisions();
            let (profile, _) = mstump_report_tiled(
                case.dimensions,
                nz(case.m),
                &scales,
                case.include,
                case.discords,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            assert!(exact_decisions() > before, "the exact path ran");
            assert_eq!(neighbors(&profile), expected(case), "{threads} {band}");
            assert_eq!(profile, reference);
        }
    }
}

#[test]
fn non_normalized_means_equal_in_f64_are_ordered_exactly() {
    assert!(NONNORM.len() >= 16);
    for case in NONNORM {
        check_nonnorm(case);
    }
}

#[test]
fn z_normalized_ties_and_near_ties_are_ordered_exactly() {
    assert!(ZNORM.len() >= 16);
    for case in ZNORM {
        check_znorm(case);
    }
}

#[test]
fn the_two_candidates_of_a_near_tie_are_not_always_the_first() {
    // The windows are built so that exactly one of the two candidates is nearer, whichever
    // comes first: both outcomes occur, so no rule such as "the smaller index" decides them.
    let winners: Vec<u64> = NONNORM.iter().map(|case| case.expected[1][0]).collect();
    assert!(winners.contains(&2) && winners.contains(&5), "{winners:?}");
}

#[test]
fn walks_of_continuous_steps_never_reach_the_exact_decision() {
    let series = |seed: u64, n: usize| -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut level = 0.0;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                level += (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                level
            })
            .collect()
    };
    let _serial = forced_route(RouteForce::default());
    let before = exact_decisions();
    let dimensions: Vec<Vec<f64>> = (1..=3).map(|seed| series(seed, 150)).collect();
    let refs = slices(&dimensions);
    let scale = Scale::new(1e-6).unwrap();
    let scales = vec![scale; 3];
    mstump_report_tiled(
        &refs,
        nz(10),
        &scales,
        &[],
        false,
        nz(2),
        nz(8),
        nz(2),
        nz(6),
    )
    .unwrap();
    maamp_report_tiled(&refs, nz(10), scale, &[], false, nz(2), nz(8), nz(2), nz(6)).unwrap();
    assert_eq!(exact_decisions(), before);
}

/// The near-tie windows of `blocks` distinct cases one after another: a series whose exact
/// decisions grow with its length.
fn many_near_ties(blocks: usize) -> Vec<Vec<f64>> {
    let mut dimensions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    for case in NONNORM.iter().step_by(2).cycle().take(blocks) {
        for (dimension, source) in dimensions.iter_mut().zip(case.dimensions) {
            dimension.extend_from_slice(&source[2..5]);
            dimension.extend_from_slice(&source[5..8]);
        }
    }
    dimensions
}

#[test]
fn exact_decisions_allocate_for_the_tiles_not_for_each_decision() {
    let run = |blocks: usize| {
        let dimensions = many_near_ties(blocks);
        let refs = slices(&dimensions);
        let _serial = forced_route(RouteForce::default());
        let before = exact_decisions();
        let ((report, decisions), allocations) = allocations_in(|| {
            let (_, report) =
                maamp_report_tiled(&refs, nz(2), unit(), &[], false, nz(1), nz(2), nz(1), nz(2))
                    .unwrap();
            (report, exact_decisions() - before)
        });
        (allocations, report.tiles, decisions)
    };
    let (few, tiles_few, decisions_few) = run(5);
    let (many, tiles_many, decisions_many) = run(40);
    assert!(
        tiles_many > 4 * tiles_few,
        "{tiles_few} tiles, then {tiles_many}"
    );
    assert!(
        decisions_many > 2 * decisions_few,
        "{decisions_few}, then {decisions_many}"
    );
    let (per_few, per_many) = (few / tiles_few, many / tiles_many);
    assert!(
        per_many <= 24 && per_many <= per_few + 2,
        "{few} allocations in {tiles_few} tiles, {many} in {tiles_many}, {decisions_many} decisions"
    );
}

/// The matches of a motif are ranked by the exact means too: a candidate whose two nearest
/// windows have means equal in `f64` lists the exactly nearer first, whichever has the smaller
/// start.
#[test]
fn motif_matches_are_ranked_by_the_exact_means() {
    use ladde::index::SampleIndex;
    use ladde::motif::{MultiMotifParams, mmotifs_nonnorm};
    use ladde::pattern::MaxDistance;

    let _serial = forced_route(RouteForce::default());
    let mut nearer_is_later = 0;
    for case in NONNORM.iter().filter(|case| !case.discords) {
        let windows = case.dimensions[0].len() - 1;
        let mut distance = vec![f64::INFINITY; windows];
        distance[0] = 0.0;
        let rows = vec![distance; 2];
        let mut neighbors = vec![SampleIndex::NONE; windows];
        neighbors[0] = SampleIndex::new(2).unwrap();
        let index = vec![neighbors; 2];
        let params = MultiMotifParams {
            min_neighbors: 0,
            max_distance: MaxDistance::Value(1e9),
            cutoffs: ladde::motif::Cutoffs::All(1e9),
            max_matches: Some(2),
            k: Some(1),
            ..MultiMotifParams::default()
        };
        let motifs =
            mmotifs_nonnorm(case.dimensions, nz(2), unit(), &rows, &index, &params).unwrap();
        let starts: Vec<u64> = motifs[0].matches.iter().map(|&(start, _)| start).collect();
        let nearer = case.expected[1][0];
        assert_eq!(starts, vec![0, nearer], "{case:?}", case = case.dimensions);
        nearer_is_later += usize::from(nearer == 5);
    }
    assert!(nearer_is_later > 0);
}

/// Flat and constant windows have distances no pair can change (zero, or the square root of the
/// window length), so means of them tie by count and no exact search runs, and the profile still
/// equals the reference.
#[test]
fn flat_and_constant_series_tie_without_any_exact_search() {
    let _serial = forced_route(RouteForce::default());
    for flat_from in [0usize, 45, 61] {
        let dimensions: Vec<Vec<f64>> = (0..4u64)
            .map(|a| {
                (0..60u64)
                    .map(|t| {
                        if t as usize >= flat_from.min(60) {
                            3.0
                        } else {
                            ((t * 7 + a * 13) % 11) as f64
                        }
                    })
                    .collect()
            })
            .collect();
        let refs = slices(&dimensions);
        let owned: Vec<_> = dimensions.iter().map(|s| ticks(s, unit())).collect();
        let tick_refs: Vec<&[Option<i64>]> = owned.iter().map(Vec::as_slice).collect();
        let scales = vec![unit(); 4];
        let expected_z =
            naive_multi_profile(&tick_refs, 6, &[], false, NonFinitePolicy::Missing).unwrap();
        let expected_n =
            naive_multi_profile_nonnorm(&tick_refs, 6, &[], false, NonFinitePolicy::Missing)
                .unwrap();
        for (band, chunk, tile) in shapes(60) {
            let before = exact_decisions();
            let (z, _) = mstump_report_tiled(
                &refs,
                nz(6),
                &scales,
                &[],
                false,
                nz(2),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            let (n, _) = maamp_report_tiled(
                &refs,
                nz(6),
                unit(),
                &[],
                false,
                nz(2),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            assert_eq!(z, expected_z, "{flat_from} {band}");
            assert_eq!(n, expected_n, "{flat_from} {band}");
            if flat_from == 0 {
                assert_eq!(
                    exact_decisions(),
                    before,
                    "a constant series needs no search"
                );
            }
        }
    }
}

/// Dimensions of a short period over a few levels: exact ties between windows that no flat or
/// constant rule decides, so most tiles make many exact decisions.
fn periodic_dimensions(n: usize) -> Vec<Vec<f64>> {
    (0..3usize)
        .map(|a| {
            (0..n)
                .map(|t| ((t % 5) * (a + 2) % 7) as f64 + (t % 5 == a) as u8 as f64)
                .collect()
        })
        .collect()
}

/// Exact decisions are comparable to or more than the tiles, and allocations still follow the
/// tiles: one allocation per decision would multiply the allocations per tile by the decisions
/// per tile.
#[test]
fn many_exact_decisions_per_tile_allocate_nothing_more_per_decision() {
    let _serial = forced_route(RouteForce::default());
    let run = |n: usize, normalized: bool| {
        let dimensions = periodic_dimensions(n);
        let refs = slices(&dimensions);
        let scales = vec![unit(); 3];
        let before = exact_decisions();
        let (report, allocations) = allocations_in(|| {
            if normalized {
                mstump_report_tiled(
                    &refs,
                    nz(8),
                    &scales,
                    &[],
                    false,
                    nz(1),
                    nz(4),
                    nz(2),
                    nz(8),
                )
                .unwrap()
                .1
            } else {
                maamp_report_tiled(
                    &refs,
                    nz(8),
                    unit(),
                    &[],
                    false,
                    nz(1),
                    nz(16),
                    nz(8),
                    nz(64),
                )
                .unwrap()
                .1
            }
        });
        (allocations, report.tiles, exact_decisions() - before)
    };
    // Non-normalized windows that repeat exactly are at distance zero, which needs no search.
    let normalized = true;
    let (few, tiles_few, decisions_few) = run(200, normalized);
    let (many, tiles_many, decisions_many) = run(800, normalized);
    assert!(
        tiles_many > 4 * tiles_few,
        "{tiles_few} tiles, then {tiles_many}"
    );
    assert!(
        decisions_few >= tiles_few && decisions_many >= tiles_many,
        "{normalized}: {decisions_few} decisions in {tiles_few} tiles, \
         {decisions_many} in {tiles_many}"
    );
    // Fewer allocations than decisions, by a margin: one per decision would pass this.
    assert!(
        2 * few < decisions_few && 2 * many < decisions_many,
        "{normalized}: {few} allocations for {decisions_few} decisions, {many} for {decisions_many}"
    );
    let (per_few, per_many) = (few / tiles_few, many / tiles_many);
    assert!(
        per_many <= 24 && per_many <= per_few + 2,
        "{normalized}: {few} allocations in {tiles_few} tiles, {many} in {tiles_many}, \
         {decisions_many} decisions"
    );
}
