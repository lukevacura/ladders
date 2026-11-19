#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The narrow start sums and resume products take their AVX2 body on x86_64 when the CPU has it,
//! and the scalar fold when forced or elsewhere; a batch join gives the same bits both ways. The
//! counter stays 0 on every other target.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{left_profile_report_tiled, two_sided_profile};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use route::*;

/// Whether the AVX2 route can fire here: on x86_64 with AVX2 detected, and nowhere else.
fn avx2_expected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if ladde::kernel_avx2::detected() {
            return true;
        }
        eprintln!("skipped: this CPU has no AVX2, so the route cannot fire");
    }
    false
}

fn series(n: usize) -> Vec<f64> {
    let mut state = 0x5eed_57a2_7500_0004u64;
    (0..n)
        .map(|_| (testkit::rng::splitmix64(&mut state) % 4001) as f64 - 2000.0)
        .collect()
}

#[test]
fn a_join_fires_the_route_and_the_force_silences_it() {
    let expect = avx2_expected();
    let (t, m) = (series(700), nz(48));
    let run = || {
        two_sided_profile(&t[..], m, Scale::new(1.0).unwrap(), nz(1))
            .iter()
            .map(|&(row, e)| (row, e.left.rho().to_bits(), e.right.rho().to_bits()))
            .collect::<Vec<_>>()
    };
    let guard = forced_route(RouteForce::default());
    let natural = run();
    assert_eq!(guard.avx2_start_sums() > 0, expect);
    drop(guard);
    let guard = forced_route(RouteForce {
        scalar_start_sum: true,
        ..RouteForce::default()
    });
    let scalar = run();
    assert_eq!(guard.avx2_start_sums(), 0);
    assert_eq!(natural, scalar);
}

#[test]
fn a_tiled_left_join_takes_the_route() {
    let expect = avx2_expected();
    let (t, m) = (series(900), nz(33));
    let run = || {
        left_profile_report_tiled(
            &t[..],
            m,
            Scale::new(1.0).unwrap(),
            nz(1),
            nz(5),
            nz(7),
            nz(64),
        )
        .0
    };
    let guard = forced_route(RouteForce::default());
    let natural = profile_bits(&run());
    assert_eq!(guard.avx2_start_sums() > 0, expect);
    drop(guard);
    let guard = forced_route(RouteForce {
        scalar_start_sum: true,
        ..RouteForce::default()
    });
    assert_eq!(profile_bits(&run()), natural);
    assert_eq!(guard.avx2_start_sums(), 0);
}
