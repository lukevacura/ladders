#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized streams that look right keep, for each window they hold, its `k` nearest
//! later neighbors and its `k` nearest over both sides, equal to a direct scan over the held
//! samples, on ties and gaps; their own neighbors do not change.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::exclusion::zone;
use ladde::nonnorm::NonNormNeighbor;
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use nonnorm::*;
use proptest::prelude::*;

type Pair = (u128, u64);

fn pairs(list: &[NonNormNeighbor]) -> Vec<Pair> {
    list.iter()
        .map(|n| (n.squared_ticks(), n.index().get().unwrap()))
        .collect()
}

/// For each window, its right and two-sided lists by a direct scan, over pairs at most `reach`
/// windows apart.
fn scan(samples: &[Option<i64>], m: usize, k: usize, reach: usize) -> Vec<(Vec<Pair>, Vec<Pair>)> {
    let windows = samples.len().saturating_sub(m - 1);
    let complete = |j: usize| samples[j..j + m].iter().all(Option::is_some);
    let squared = |i: usize, j: usize| -> u128 {
        (0..m)
            .map(|t| {
                let d = i128::from(samples[i + t].unwrap()) - i128::from(samples[j + t].unwrap());
                (d * d) as u128
            })
            .sum()
    };
    (0..windows)
        .map(|j| {
            if !complete(j) {
                return (Vec::new(), Vec::new());
            }
            let mut right = Vec::new();
            let mut both = Vec::new();
            for i in 0..windows {
                if i.abs_diff(j) <= zone(m) || i.abs_diff(j) > reach || !complete(i) {
                    continue;
                }
                both.push((squared(i, j), i as u64));
                if i > j {
                    right.push((squared(i, j), i as u64));
                }
            }
            right.sort();
            both.sort();
            right.truncate(k);
            both.truncate(k);
            (right, both)
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn the_growing_stream_looking_right_equals_a_scan(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
        k in prop_oneof![Just(1usize), 2usize..9],
    ) {
        let held = samples(&series, unit());
        let expected = scan(&held, m, k, usize::MAX);
        let mut stream = GrowingNonNormStream::with_right_side(nz(m), unit(), nz(k));
        let mut plain = GrowingNonNormStream::with_k(nz(m), unit(), nz(k));
        let mut out = Vec::new();
        for &x in &series {
            let a = stream.update_topk(x).map(|(r, n)| (r, pairs(n)));
            let b = plain.update_topk(x).map(|(r, n)| (r, pairs(n)));
            prop_assert_eq!(a, b);
        }
        prop_assert_eq!(stream.right_windows().end as usize, expected.len());
        for window in stream.right_windows() {
            stream.right_neighbors_into(window, &mut out);
            prop_assert_eq!(&pairs(&out), &expected[window as usize].0, "right {}", window);
            stream.two_sided_neighbors_into(window, &mut out);
            prop_assert_eq!(&pairs(&out), &expected[window as usize].1, "both {}", window);
        }
    }

    #[test]
    fn the_bounded_stream_looking_right_equals_a_scan(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
        k in prop_oneof![Just(1usize), 2usize..9],
        extra in 0usize..30,
    ) {
        let held = samples(&series, unit());
        let expected = scan(&held, m, k, extra);
        let mut stream =
            BoundedNonNormStream::with_right_side(nz(m), nz(m + extra), unit(), nz(k)).unwrap();
        let mut out = Vec::new();
        for &x in &series {
            let _ = stream.update(x);
        }
        for window in stream.right_windows() {
            stream.right_neighbors_into(window, &mut out);
            prop_assert_eq!(&pairs(&out), &expected[window as usize].0, "right {}", window);
            stream.two_sided_neighbors_into(window, &mut out);
            prop_assert_eq!(&pairs(&out), &expected[window as usize].1, "both {}", window);
        }
    }
}
