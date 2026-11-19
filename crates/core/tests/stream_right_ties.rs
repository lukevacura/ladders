#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The right side of the z-normalized streams on series made of ties: constants, repeats of a
//! short period, few levels and flat runs between levels. Equal distances go to the smaller
//! index, in the lists of both sides and after a checkpoint, on every route.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::quantize::Scale;
use ladde::route::forced_route;
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::spill_ref::{self, Bits};
use route::{bits, forces, nz};

fn ties() -> impl Strategy<Value = (Vec<f64>, usize)> {
    let series = prop_oneof![
        (20usize..70, -3i32..3).prop_map(|(n, level)| vec![f64::from(level); n]),
        (20usize..70, 2usize..8, prop::collection::vec(0u8..4, 8)).prop_map(
            |(n, period, shape)| (0..n)
                .map(|k| f64::from(shape[k % period]))
                .collect::<Vec<_>>()
        ),
        prop::collection::vec(0u8..3, 20..70)
            .prop_map(|levels| levels.into_iter().map(f64::from).collect::<Vec<_>>()),
        (prop::collection::vec((0u8..3, 1usize..12), 3..9)).prop_map(|runs| {
            runs.into_iter()
                .flat_map(|(level, len)| vec![f64::from(level); len])
                .collect::<Vec<_>>()
        }),
        (20usize..70, 2usize..6).prop_map(|(n, gap)| {
            (0..n)
                .map(|k| {
                    if k % (gap + 3) == 0 {
                        f64::NAN
                    } else {
                        f64::from((k % 3) as u8)
                    }
                })
                .collect::<Vec<_>>()
        }),
    ];
    (series, 2usize..7)
}

fn ticks_and_presence(series: &[f64]) -> (Vec<i64>, Vec<bool>) {
    let mut last = 0;
    let mut ticks = Vec::new();
    let mut had = Vec::new();
    for &x in series {
        if x.is_finite() {
            last = x as i64;
        }
        ticks.push(last);
        had.push(x.is_finite());
    }
    (ticks, had)
}

fn read(window: u64, lists: &[ladde::entry::Entry]) -> Vec<Bits> {
    lists.iter().map(|&e| bits(window, e)).collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn equal_distances_go_to_the_smaller_index_on_both_sides(
        (series, m) in ties(),
        seed in any::<u64>(),
        extra in 0usize..30,
        k in prop_oneof![Just(1usize), 2usize..9],
    ) {
        let scale = Scale::new(1.0).unwrap();
        let (ticks, had) = ticks_and_presence(&series);
        let right = spill_ref::right_topk(&ticks, &had, m, k);
        let both = spill_ref::two_sided_topk(&ticks, &had, m, k);
        let w = m + extra;
        let held_both = spill_ref::two_sided_topk_where(&ticks, &had, m, k, &|i, j| i.abs_diff(j) <= extra);
        let mut out = Vec::new();
        for force in forces(seed) {
            let _forced = forced_route(force);
            let mut growing = GrowingStream::with_right_side(nz(m), scale, nz(k));
            let mut bounded = BoundedStream::with_right_side(nz(m), nz(w), scale, nz(k)).unwrap();
            for &x in &series {
                let _ = (growing.update(x), bounded.update(x));
            }
            for window in growing.right_windows() {
                growing.right_neighbors_into(window, &mut out);
                prop_assert_eq!(read(window, &out), right[window as usize].clone(), "{:?} right {}", force, window);
                growing.two_sided_neighbors_into(window, &mut out);
                prop_assert_eq!(read(window, &out), both[window as usize].2.clone(), "{:?} both {}", force, window);
            }
            for window in bounded.right_windows() {
                bounded.right_neighbors_into(window, &mut out);
                prop_assert_eq!(read(window, &out), right[window as usize].clone(), "{:?} bounded right {}", force, window);
                bounded.two_sided_neighbors_into(window, &mut out);
                prop_assert_eq!(read(window, &out), held_both[window as usize].2.clone(), "{:?} bounded both {}", force, window);
            }
        }
    }
}
