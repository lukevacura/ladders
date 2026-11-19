#![cfg(feature = "parallel")]

//! Peak heap use of the anytime profiles on series where every pair ties exactly stays linear in
//! the series length: doubling it at most about doubles the peak, over several updates and
//! the profile read after each. Its own test binary, with a process-wide peak-tracking allocator
//! and a single test so nothing else allocates meanwhile. Every run uses one thread, so tiles run
//! in a fixed order and the peak is deterministic.

#[path = "common/peak_alloc.rs"]
mod peak_alloc;

use core::num::NonZeroUsize;

use ladde::quantize::Scale;
use ladde::scrump::{Scraamp, ScraampJoin, Scrump, ScrumpJoin, Shuffled};
use peak_alloc::peak_of;

type Make = fn(usize) -> Vec<f64>;

#[test]
fn peak_heap_is_linear_in_the_series_under_exact_ties() {
    let m = NonZeroUsize::new(8).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let k = NonZeroUsize::new(4).unwrap();
    let threads = NonZeroUsize::MIN;
    let series: [(&str, Make); 2] = [
        ("constant", |n| vec![3.0; n]),
        ("period four", |n| {
            (0..n).map(|i| [0.0, 2.0, 1.0, 3.0][i % 4]).collect()
        }),
    ];
    for (name, make) in series {
        let z = |n: usize| {
            let series = make(n);
            peak_of(|| {
                let mut scrump =
                    Scrump::new(&series, m, scale, threads, k, 0.5, Shuffled::seeded(1)).unwrap();
                while !scrump.is_complete() {
                    scrump.update().unwrap();
                    assert_eq!(scrump.profile().unwrap().len(), n - 7);
                }
            })
        };
        let plain = |n: usize| {
            let series = make(n);
            peak_of(|| {
                let mut scraamp =
                    Scraamp::new(&series, m, scale, threads, 0.5, Shuffled::seeded(1)).unwrap();
                while !scraamp.is_complete() {
                    scraamp.update().unwrap();
                    assert_eq!(scraamp.profile().unwrap().len(), n - 7);
                }
            })
        };
        let z_join = |n: usize| {
            let series = make(n);
            peak_of(|| {
                let mut join = ScrumpJoin::new(
                    &series,
                    &series,
                    m,
                    (scale, scale),
                    threads,
                    k,
                    0.5,
                    Shuffled::seeded(1),
                )
                .unwrap();
                while !join.is_complete() {
                    join.update().unwrap();
                    assert_eq!(join.profile().unwrap().len(), n - 7);
                }
            })
        };
        let plain_join = |n: usize| {
            let series = make(n);
            peak_of(|| {
                let mut join = ScraampJoin::new(
                    &series,
                    &series,
                    m,
                    scale,
                    threads,
                    0.5,
                    Shuffled::seeded(1),
                )
                .unwrap();
                while !join.is_complete() {
                    join.update().unwrap();
                    assert_eq!(join.profile().unwrap().len(), n - 7);
                }
            })
        };
        for (which, run) in [
            ("scrump", &z as &dyn Fn(usize) -> usize),
            ("scraamp", &plain),
            ("scrump join", &z_join),
            ("scraamp join", &plain_join),
        ] {
            let (small, large) = (run(3_000), run(6_000));
            assert!(
                large <= small * 11 / 4,
                "{which} {name}: peak {large} bytes at 6000 vs {small} at 3000"
            );
            assert!(
                large < 6_000 * 1536 * (k.get() + 1),
                "{which} {name}: peak {large} bytes"
            );
        }
    }
}
