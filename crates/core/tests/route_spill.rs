#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The spill route in the route harness: forced on, off and left to the cost model, on series
//! where the model engages, declines, or has nothing to do. Every route equals the exact
//! reference (a window with no distance gets one on every route), the report says which route
//! ran, and the forced-route counters move exactly when a forced route overrides the model.

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    SpillReport, ab_join_spill_report, left_profile_spill_report, two_sided_spill_report,
};
use ladde::route::{RouteForce, SpillForce, forced_route};
use spill_run::*;

fn force(spill: SpillForce) -> RouteForce {
    RouteForce {
        spill,
        ..RouteForce::default()
    }
}

struct Scenario {
    name: &'static str,
    series: Vec<f64>,
    m: usize,
    /// What the cost model says on its own; `None` for a series it is never asked about.
    natural: Option<bool>,
}

fn run(scenario: &Scenario) -> (Vec<Bits>, SpillReport) {
    let (entries, _, report) =
        left_profile_spill_report(&scenario.series, nz(scenario.m), unit(), nz(2));
    (bits(&entries), report)
}

/// `series` with `count` excursions of `len` samples, `every` samples apart, each `shift` up: the
/// windows inside one are finite today although they hold outliers.
fn excursions(
    mut series: Vec<f64>,
    count: usize,
    len: usize,
    every: usize,
    shift: f64,
) -> Vec<f64> {
    for k in 0..count {
        for x in &mut series[30 + k * every..30 + k * every + len] {
            *x += shift;
        }
    }
    series
}

fn scenarios() -> Vec<Scenario> {
    let many: Vec<(usize, f64)> = (0..30).map(|k| (13 * k + 5, 4.0e11)).collect();
    vec![
        Scenario {
            name: "one outlier",
            series: with_spikes(walk(1200, 1), &[(600, 1.0e12)]),
            m: 20,
            natural: Some(true),
        },
        Scenario {
            name: "a few outliers",
            series: with_spikes(
                walk(1200, 2),
                &[(200, 5.0e11), (700, -5.0e11), (701, 3.0e9)],
            ),
            m: 20,
            natural: Some(true),
        },
        Scenario {
            name: "many outliers",
            series: with_spikes(walk(400, 5), &many),
            m: 20,
            natural: Some(true),
        },
        Scenario {
            name: "many excursions",
            series: excursions(walk(800, 6), 10, 40, 70, 3.0e11),
            m: 20,
            natural: Some(false),
        },
        Scenario {
            name: "clean",
            series: walk(500, 9),
            m: 20,
            natural: None,
        },
    ]
}

#[test]
fn every_route_is_the_exact_profile_and_the_model_picks_which_ran() {
    for scenario in scenarios() {
        let (name, series, m) = (scenario.name, &scenario.series, scenario.m);
        let exact = reference(series, m, unit());

        let guard = forced_route(force(SpillForce::Off));
        let (off, report) = run(&scenario);
        assert_eq!(off, exact, "{name}: off");
        assert!(!report.engaged, "{name}: off");
        let natural = report.plan.map(|plan| plan.natural);
        assert_eq!(natural, scenario.natural, "{name}");
        assert_eq!(
            guard.fired().spill_forced_off,
            usize::from(natural == Some(true)),
            "{name}"
        );
        assert_eq!(guard.fired().spill_forced_on, 0);
        drop(guard);

        let guard = forced_route(force(SpillForce::On));
        let (on, report) = run(&scenario);
        assert_eq!(on, exact, "{name}: on");
        assert_eq!(report.engaged, natural.is_some(), "{name}: on");
        assert_eq!(
            guard.fired().spill_forced_on,
            usize::from(natural == Some(false)),
            "{name}"
        );
        assert_eq!(guard.fired().spill_forced_off, 0);
        drop(guard);

        let _guard = forced_route(force(SpillForce::Auto));
        let (auto, report) = run(&scenario);
        assert_eq!(auto, exact, "{name}: auto");
        assert_eq!(report.plan.map(|plan| plan.natural), natural, "{name}");
        assert_eq!(report.engaged, natural == Some(true), "{name}");
    }
}

#[test]
fn a_window_with_no_distance_gets_one_whatever_the_route() {
    // A print of 1e13 leaves every window holding it past the exact domain.
    let m = 20;
    let series = with_spikes(walk(1200, 4), &[(600, 1.0e13)]);
    for spill in [SpillForce::Off, SpillForce::Auto, SpillForce::On] {
        let _guard = forced_route(force(spill));
        let (entries, _, report) = left_profile_spill_report(&series, nz(m), unit(), nz(1));
        for (row, (_, entry)) in entries.iter().enumerate().take(601).skip(581) {
            assert!(
                entry.index().get().is_some(),
                "{spill:?}: row {row} has a neighbor"
            );
        }
        assert!(report.spilled_windows >= m, "{spill:?}");
        assert!(report.walked.cells > 0, "{spill:?}");
    }
}

#[test]
fn the_model_never_decides_which_windows_have_a_distance() {
    let m = 20;
    let series = with_spikes(walk(1200, 6), &[(300, 2.0e12), (900, -3.0e12)]);
    let exact = reference(&series, m, unit());
    for spill in [SpillForce::Off, SpillForce::On] {
        let _guard = forced_route(force(spill));
        let (got, _, _) = left_profile_spill_report(&series, nz(m), unit(), nz(3));
        assert_eq!(bits(&got), exact, "{spill:?}");
    }
}

#[test]
fn forced_on_fires_only_where_the_model_declined() {
    let guard = forced_route(force(SpillForce::On));
    for scenario in scenarios() {
        let before = guard.fired().spill_forced_on;
        run(&scenario);
        assert_eq!(
            guard.fired().spill_forced_on - before,
            usize::from(scenario.natural == Some(false)),
            "{}",
            scenario.name
        );
    }
}

/// Which profile a scenario runs.
#[derive(Clone, Copy, Debug)]
enum Family {
    TwoSided,
    JoinRows,
    JoinColumns,
}

/// One run of `family` over `scenario`: the bits of the rows it ranks and the report.
fn run_family(family: Family, scenario: &Scenario) -> (Vec<Bits>, SpillReport) {
    let (series, m) = (&scenario.series, scenario.m);
    let other = walk(700, 17);
    match family {
        Family::TwoSided => {
            let (profile, _, report) = two_sided_spill_report(series, nz(m), unit(), nz(2));
            let nearest = profile
                .iter()
                .map(|&(row, sides)| {
                    let e = sides.nearest;
                    (
                        row,
                        e.rho().to_bits(),
                        e.squared_distance().to_bits(),
                        e.index().get().unwrap_or(u64::MAX),
                    )
                })
                .collect();
            (nearest, report)
        }
        Family::JoinRows => {
            let (entries, _, report) =
                ab_join_spill_report(series, &other, nz(m), unit(), unit(), nz(2));
            (bits(&entries), report)
        }
        Family::JoinColumns => {
            let (entries, _, report) =
                ab_join_spill_report(&other, series, nz(m), unit(), unit(), nz(2));
            (bits(&entries), report)
        }
    }
}

/// The exact answer `run_family` is held to.
fn exact_family(family: Family, scenario: &Scenario) -> Vec<Bits> {
    let (series, m) = (&scenario.series, scenario.m);
    let other = walk(700, 17);
    match family {
        Family::TwoSided => exact_two_sided(series, unit(), m)
            .into_iter()
            .map(|sides| sides[2])
            .collect(),
        Family::JoinRows => exact_join(series, unit(), &other, unit(), m),
        Family::JoinColumns => exact_join(&other, unit(), series, unit(), m),
    }
}

#[test]
fn the_model_picks_the_route_of_two_sided_profiles_and_joins_and_every_route_is_exact() {
    for family in [Family::TwoSided, Family::JoinRows, Family::JoinColumns] {
        for scenario in scenarios() {
            let exact = exact_family(family, &scenario);
            for spill in [SpillForce::Off, SpillForce::On, SpillForce::Auto] {
                let guard = forced_route(force(spill));
                let (got, report) = run_family(family, &scenario);
                let name = format!("{family:?}, {}, {spill:?}", scenario.name);
                assert_eq!(got, exact, "{name}");
                let natural = report.plan.map(|plan| plan.natural);
                if spill == SpillForce::Auto {
                    assert_eq!(natural, scenario.natural, "{name}");

                    assert_eq!(report.engaged, natural == Some(true), "{name}");
                }
                if spill == SpillForce::On {
                    assert_eq!(report.engaged, natural.is_some(), "{name}");
                    let declined = usize::from(natural == Some(false));
                    assert_eq!(guard.fired().spill_forced_on, declined, "{name}");
                }
                if spill == SpillForce::Off {
                    assert!(!report.engaged, "{name}");
                    let engaged = usize::from(natural == Some(true));
                    assert_eq!(guard.fired().spill_forced_off, engaged, "{name}");
                }
            }
        }
    }
}
