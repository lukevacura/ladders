//! The AB-join driver's pieces: the lane catch-up over the join's slots, where a lane is born at
//! a row and ends at another.

#[cfg(not(any(loom, shuttle)))]
use crate::batch::one_loop::Counts;
#[cfg(not(any(loom, shuttle)))]
use crate::batch::one_loop::band::catch_up_over;
#[cfg(not(any(loom, shuttle)))]
use crate::batch::one_loop::store::Carry;
#[cfg(not(any(loom, shuttle)))]
use crate::batch::pass::Shape;
#[cfg(not(any(loom, shuttle)))]
use crate::cross_term::CrossTerms;
#[cfg(not(any(loom, shuttle)))]
use crate::route::{CatchUpForce, RouteForce, forced_route};
#[cfg(not(any(loom, shuttle)))]
use alloc::vec::Vec;

/// A lane of the join caught up to a row, through any number of earlier catch-ups and under either
/// branch of the step-or-restart rule, holds the state a fresh exact start gives it: the pair one
/// row back, one column back. The lanes live at a row are the slots `row..row + n_b - 1`.
#[cfg(not(any(loom, shuttle)))]
#[test]
fn a_caught_up_join_lane_equals_a_fresh_exact_start() {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut ticks = |len: usize| -> Vec<i32> {
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 2001) as i32 - 1000
            })
            .collect()
    };
    let m = 8usize;
    let (a, b) = (ticks(70), ticks(50));
    let (n_a, n_b) = (a.len() - m + 1, b.len() - m + 1);
    let top = n_a + n_b - 1;
    let shape = Shape::Join { n_b };
    let pairs = CrossTerms::new(&a[..], 0, &b[..], 0, m);
    let live = |row: usize| row..top.min(row + n_b - 1);
    for force in [CatchUpForce::Rule, CatchUpForce::Step, CatchUpForce::Resume] {
        let _guard = forced_route(RouteForce {
            catch_up: force,
            ..RouteForce::default()
        });
        for (r0, r1) in [(10, 11), (12, 20), (16, 40), (33, 34), (20, 60)] {
            let mut counts = Counts::default();
            let mut fresh = Carry::try_new(top).unwrap();
            catch_up_over(
                &pairs,
                &mut fresh,
                (0, top, shape),
                (live(r1), r1),
                &mut counts,
            );
            let mut walked = Carry::try_new(top).unwrap();
            catch_up_over(
                &pairs,
                &mut walked,
                (0, top, shape),
                (live(r0), r0),
                &mut counts,
            );
            catch_up_over(
                &pairs,
                &mut walked,
                (0, top, shape),
                (live(r1), r1),
                &mut counts,
            );
            for s in live(r1) {
                assert_eq!(walked.through[s], r1);
                assert_eq!(fresh.through[s], r1);
                let column = r1 + n_b - 1 - s;
                assert!(column >= 1, "a live lane continues");
                let exact = pairs.at(r1 - 1, column - 1);
                assert_eq!(fresh.qt[top - 1 - s], exact, "{force:?} fresh, slot {s}");
                assert_eq!(
                    walked.qt[top - 1 - s],
                    exact,
                    "{force:?} rows {r0}->{r1} slot {s}"
                );
            }
        }
    }
}
