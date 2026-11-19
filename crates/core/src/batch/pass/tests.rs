use super::{
    BLOCK_ROWS, Block, FrontGrowBuffer, NarrowColumns, Nearest, RowAcc, Shape, advance_chunk,
    block_scalars, compute_visits,
};
use crate::batch::quantized::Quantized;
use crate::cross_term::CrossTerms;
use crate::entry::Entry;
use crate::quantize::Scale;
use core::num::NonZeroUsize;
use std::format;
use std::string::String;
use std::vec;
use std::vec::Vec;

/// How the slots of a chunk reach the row loop.
#[derive(Clone, Copy)]
enum Via {
    /// [`advance_chunk`]: the plain pass over the whole chunk.
    Chunk,
    /// [`compute_visits`] over the chunk as one visit.
    Leaf,
    /// [`compute_visits`] over the chunk as two adjacent visits.
    Halves,
    /// [`compute_visits`] over the chunk's first and last quarter only, the rest left out.
    Edges,
}

/// A random walk with a few repeated stretches, from a fixed seed.
fn walk(n: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut x = 0.0;
    (0..n)
        .map(|k| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if k % 97 >= 80 {
                return x;
            }
            x += f64::from((state >> 40) as u32 % 2001) / 1000.0 - 1.0;
            x
        })
        .collect()
}

/// The self-join profile of [`walk`] through the first `blocks` blocks of rows, one at a time, every chunk of `chunk_lanes`
/// slots reaching the row loop `via`: each row's entry as text, and the pairs computed.
fn profile(
    via: Via,
    (n, m, chunk_lanes): (usize, usize, usize),
    blocks: usize,
) -> (Vec<String>, usize) {
    let series = walk(n);
    let m_nz = NonZeroUsize::new(m).unwrap();
    let quantized = Quantized::build(&series[..], m_nz, Scale::new(1.0).unwrap());
    let anchor = quantized.narrow_anchor(m_nz).expect("one anchor covers it");
    let (ticks, sums) = quantized.narrow_columns(anchor, m_nz);
    let cols = NarrowColumns {
        ticks: &ticks,
        sum: &sums,
        r: &quantized.r,
        bias: &quantized.bias,
        variance: &quantized.variance,
        conversion: quantized.range.conversion(),
    };
    let windows = &quantized.windows;
    let exclusion = crate::exclusion::zone(m);
    let k = NonZeroUsize::MIN;
    let mut qt = FrontGrowBuffer::new();
    let mut tile = vec![0.0; chunk_lanes];
    let mut accs: Vec<RowAcc> = (0..BLOCK_ROWS).map(|_| RowAcc::new()).collect();
    let mut entries: Vec<(u64, Entry)> = Vec::new();
    let mut computed = 0;
    let mut first = exclusion + 1;
    let mut done = 0;
    while first < windows.len() && done < blocks {
        done += 1;
        let rows = first..(first + BLOCK_ROWS).min(windows.len());
        for _ in rows.clone() {
            qt.grow_front();
        }
        let live = qt.as_mut_slice();
        let slots = live.len();
        let scalars = block_scalars(&cols, windows, &rows, m);
        for (acc, row) in accs.iter_mut().zip(rows.clone()) {
            acc.begin(&windows[row], k, windows.len());
        }
        let block = Block {
            cols: &cols,
            base: 0,
            pairs: CrossTerms::of(cols.ticks, m),
            rows: rows.clone(),
            scalars: &scalars,
            m,
            shape: Shape::Triangle { exclusion },
            slots,
        };
        for qa in (0..slots).step_by(chunk_lanes) {
            let qb = (qa + chunk_lanes).min(slots);
            let states = &mut live[slots - qb..slots - qa];
            let mid = qa + (qb - qa) / 2;
            let quarter = (qb - qa) / 4;
            computed += match via {
                Via::Chunk => {
                    advance_chunk(&block, (qa, qb), states, &mut accs, &mut tile, None);
                    0
                }
                Via::Leaf => compute_visits(
                    &block,
                    (qa, qb),
                    states,
                    &mut accs,
                    &mut tile,
                    None,
                    core::slice::from_ref(&(qa..qb)),
                ),
                Via::Halves => compute_visits(
                    &block,
                    (qa, qb),
                    states,
                    &mut accs,
                    &mut tile,
                    None,
                    &[qa..mid, mid..qb],
                ),
                Via::Edges => compute_visits(
                    &block,
                    (qa, qb),
                    states,
                    &mut accs,
                    &mut tile,
                    None,
                    &[qa..qa + quarter, qb - quarter..qb],
                ),
            };
        }
        for (acc, row) in accs.iter_mut().zip(rows.clone()) {
            let mut nearest = Nearest::with_k(&windows[row], m, k, windows.len());
            acc.finish(&mut nearest);
            nearest.push_kept(row as u64, k, &mut entries);
        }
        first = rows.end;
    }
    (entries.iter().map(|e| format!("{e:?}")).collect(), computed)
}

#[test]
fn the_leaf_compute_over_a_whole_chunk_is_the_plain_pass() {
    for shape in [(420, 16, 40), (333, 8, 64), (260, 24, 1000)] {
        let (plain, _) = profile(Via::Chunk, shape, usize::MAX);
        let (leaf, computed) = profile(Via::Leaf, shape, usize::MAX);
        assert_eq!(leaf, plain, "{shape:?}");
        assert!(computed > 0);
        let (halves, split) = profile(Via::Halves, shape, usize::MAX);
        assert_eq!(halves, plain, "two visits, {shape:?}");
        assert_eq!(split, computed, "the same pairs are computed");
    }
}

#[test]
fn the_leaf_compute_computes_only_the_visits_it_is_given() {
    // One block, so no later block reads the states the left-out slots did not advance.
    let (_, whole) = profile(Via::Leaf, (420, 16, 40), 1);
    let (_, edges) = profile(Via::Edges, (420, 16, 40), 1);
    assert!(
        edges < whole,
        "the slots left out of the visits are not computed ({edges} of {whole})"
    );
}
