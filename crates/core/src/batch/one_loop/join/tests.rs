//! The AB-join's entry: its cut of the rectangle into tiles, the condition under which the route
//! runs, and what it does where it declines.

use super::{Layout, cut_tiles};
use crate::batch::one_loop::dense::valid_cells_over;
use crate::batch::pass::Shape;

/// Every tile of a band holds slots of its rows only (inside the band's own range), the tiles of
/// a band are disjoint and cover every slot a row of the band has, and every range is aligned
/// to the tile width, the same in every band: the pairs of all tiles add up to the rectangle's.
#[test]
fn the_tiles_cover_every_cell_of_the_rectangle_once() {
    for (n_a, n_b) in [
        (1, 1),
        (2, 9),
        (9, 2),
        (16, 17),
        (33, 64),
        (100, 40),
        (257, 300),
    ] {
        for plan in [
            (1, 1, 3),
            (16, 16, 16),
            (16, 24, 48),
            (64, 64, 128),
            (1000, 64, 4096),
        ] {
            let slots = n_a + n_b - 1;
            let layout = Layout::new(plan, slots);
            let tiles = cut_tiles((n_a, n_b), &layout).expect("small cut");
            let mut cells = 0;
            let mut seen = alloc::vec![0u8; n_a * n_b];
            for tile in &tiles {
                assert_eq!(tile.qa % layout.tile_lanes, 0, "tile {} aligned", tile.qa);
                assert!(tile.qb - tile.qa <= layout.tile_lanes && tile.qa < tile.qb);
                assert_eq!(tile.rows.start % layout.side, 0);
                assert!(tile.rows.end - tile.rows.start <= layout.side);
                assert_eq!(tile.band, tile.rows.start / layout.side);
                cells += valid_cells_over(Shape::Join { n_b }, &tile.rows, &(tile.qa..tile.qb));
                for row in tile.rows.clone() {
                    for slot in tile.qa..tile.qb {
                        if row <= slot && slot < row + n_b {
                            seen[row * n_b + (row + n_b - 1 - slot)] += 1;
                        }
                    }
                }
            }
            assert_eq!(cells, n_a * n_b, "{n_a}x{n_b} {plan:?}");
            assert!(seen.iter().all(|&c| c == 1), "{n_a}x{n_b} {plan:?}");
            // The tiles of a band ascend by slot and by ordinal.
            for pair in tiles.windows(2) {
                if pair[0].band == pair[1].band {
                    assert_eq!(pair[1].qa, pair[0].qa + layout.tile_lanes);
                    assert_eq!(pair[1].column, pair[0].column + 1);
                } else {
                    assert_eq!(pair[1].band, pair[0].band + 1);
                    assert_eq!(pair[1].column, 0);
                }
            }
        }
    }
}

#[cfg(not(any(loom, shuttle)))]
mod routed {
    use super::super::{profile, taken};
    use crate::batch::ab_join::ab_join_on;
    use crate::batch::plan::TilePlan;
    use crate::batch::quantized::Quantized;
    use crate::entry::Entry;
    use crate::quantize::Scale;
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;

    fn nz(x: usize) -> NonZeroUsize {
        NonZeroUsize::new(x).unwrap()
    }

    /// The join takes the route wherever it is not forced off, over every diagonal, with
    /// coefficients on both sides, whatever the anchors.
    #[test]
    fn the_join_takes_the_route_unless_it_is_forced_off() {
        use crate::route::{OneLoop, RouteForce, forced_route};
        for (route, expect) in [
            (OneLoop::Off, false),
            (OneLoop::Auto, true),
            (OneLoop::On, true),
        ] {
            let _guard = forced_route(RouteForce {
                one_loop: route,
                ..RouteForce::default()
            });
            assert_eq!(taken(false, true), expect, "{route:?}");
            assert!(!taken(true, true), "{route:?}: some diagonals");
            assert!(!taken(false, false), "{route:?}: no coefficients");
        }
    }

    fn walk(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed | 1;
        let mut price = 100.0;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                price += ((state >> 33) % 7) as f64 - 3.0;
                price
            })
            .collect()
    }

    fn bits(profile: &[(u64, Entry)]) -> Vec<(u64, u64, u64, u64)> {
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

    /// A run without coefficients is declined with nothing touched, and one with them returns the
    /// bits the plain tiled pass returns, in the tiles it cut.
    #[test]
    fn the_entry_declines_without_coefficients_and_otherwise_equals_the_tiled_path() {
        use crate::route::{OneLoop, RouteForce, SurvivorForce, forced_route};
        let (m, scale) = (nz(24), Scale::new(1.0).unwrap());
        let (a, b) = (walk(1, 700), walk(2, 520));
        let plan = |q: &Quantized| TilePlan {
            k: nz(2),
            survivor: q.filter,
            ..TilePlan::fixed(nz(32), nz(16), nz(64))
        };
        {
            let _guard = forced_route(RouteForce {
                one_loop: OneLoop::Off,
                survivor: SurvivorForce::Off,
                ..RouteForce::default()
            });
            let (qa, qb) = (
                Quantized::build(&a[..], m, scale),
                Quantized::build(&b[..], m, scale),
            );
            assert!(qa.paa.is_none() && qb.paa.is_none());
            let mut results = Vec::new();
            assert!(profile((&qa, &qb), m, plan(&qa), nz(1), None, &mut results).is_none());
            assert!(results.is_empty(), "nothing touched");
        }
        let (qa, qb) = {
            let _guard = forced_route(RouteForce {
                survivor: SurvivorForce::Paa,
                ..RouteForce::default()
            });
            (
                Quantized::build(&a[..], m, scale),
                Quantized::build(&b[..], m, scale),
            )
        };
        assert!(qa.paa.is_some() && qb.paa.is_some());
        let ran = |route: OneLoop| {
            let _guard = forced_route(RouteForce {
                one_loop: route,
                survivor: SurvivorForce::Paa,
                ..RouteForce::default()
            });
            let ((flat, k), report) = ab_join_on(
                &qa,
                &qb,
                m,
                nz(1),
                16,
                Some(TilePlan::fixed(nz(32), nz(16), nz(64))),
                nz(2),
                Vec::new(),
                None,
            );
            (bits(&flat), k, report.tiles)
        };
        let (off, k, off_tiles) = ran(OneLoop::Off);
        let (on, on_k, on_tiles) = ran(OneLoop::On);
        assert_eq!(on, off);
        assert_eq!(on_k, k);
        assert!(on_tiles > 0 && off_tiles > 0);
        let mut direct = Vec::new();
        let _guard = forced_route(RouteForce {
            one_loop: OneLoop::On,
            survivor: SurvivorForce::Paa,
            ..RouteForce::default()
        });
        let report = profile((&qa, &qb), m, plan(&qa), nz(1), None, &mut direct).expect("taken");
        assert_eq!(bits(&direct), off);
        assert_eq!(report.tiles, on_tiles);
    }
}
