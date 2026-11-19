extern crate std;

use super::super::plan::{TileCut, TilePlan};
use super::super::tile::{join_tile_is_wide, triangle_tile_is_wide};
use crate::kernel_neon::GROUP_LANES;
use crate::quantize::{SPAN_SLACK, fast_ticks};
use core::num::NonZeroUsize;
use std::vec::Vec;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn ramp(len: usize, step: i64) -> Vec<i64> {
    (0..len as i64).map(|k| k * step).collect()
}

/// The smallest cell the route cuts, one row by one kernel group, spans `m + GROUP_LANES`
/// samples, no more than a clean span of the default scale: on a ramp whose every
/// `m + SPAN_SLACK` samples fit one anchor no tile takes the exact pass, and on one whose
/// `m + GROUP_LANES` samples do not, some tile does.
#[test]
fn every_cell_fits_where_every_clean_span_does() {
    let _serial = crate::route::forced_route(crate::route::RouteForce::default());
    const { assert!(GROUP_LANES <= SPAN_SLACK) };
    let len = 2000;
    for m in [4usize, 16, 64, 200] {
        let reach = 2 * fast_ticks(nz(m));
        let fitting = ramp(len, reach / (m + SPAN_SLACK - 1) as i64);
        let cell_too_wide = ramp(len, reach / (m + GROUP_LANES - 1) as i64 + 1);
        let n = len + 1 - m;
        for (band_rows, chunk_lanes, tile_lanes) in [(64, 256, 1024), (7, 64, 192), (300, 512, 512)]
        {
            let plan = TilePlan::fixed(nz(band_rows), nz(chunk_lanes), nz(tile_lanes));
            let triangle = TileCut::triangle(n, m, plan, None).tiles;
            let wide = |ticks: &[i64]| {
                triangle
                    .iter()
                    .filter(|tile| triangle_tile_is_wide(ticks, nz(m), tile, chunk_lanes))
                    .count()
            };
            assert_eq!(wide(&fitting), 0, "triangle, m {m}");
            assert!(wide(&cell_too_wide) > 0, "triangle, m {m}");
            let join = TileCut::join(n, n, plan, None).tiles;
            let wide = |ticks: &[i64]| {
                join.iter()
                    .filter(|tile| join_tile_is_wide(ticks, ticks, nz(m), tile, chunk_lanes))
                    .count()
            };
            assert_eq!(wide(&fitting), 0, "join, m {m}");
            assert!(wide(&cell_too_wide) > 0, "join, m {m}");
        }
    }
}
