#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Multi-dimensional motifs equal a plain, independent statement of STUMPY's `mmotifs` procedure
//! (exact rational orders for one dimension, the f64 mean for more, the description-length
//! reference for `k` and the subspace): on inputs of exact ties, gaps at either end and `include`,
//! over a grid of parameters, with profiles from the reference, from the batch engine at every
//! thread count, and made by hand.

#[path = "common/exact_pair.rs"]
mod exact_pair;
#[path = "common/mmotifs_ref.rs"]
mod mmotifs_ref;
#[path = "common/multi.rs"]
mod multi;

use core::cmp::Ordering;

use exact_pair::{Rho, cmp_rho, pair};
use ladde::batch::multi::mstump;
use ladde::index::SampleIndex;
use ladde::motif::{Cutoffs, MultiMotifParams, mmotifs};
use ladde::multidim::{MultiError, NonFinitePolicy};
use ladde::naive::{Ticks, naive_mdl, naive_multi_profile};
use ladde::pattern::MaxDistance;
use ladde::quantize::Scale;
use ladde::stats::Series;
use ladde::subspace::{Bits, Subspace};
use mmotifs_ref::{Plain, RefMetric, grid, plain, reference, rows};
use multi::*;
use proptest::prelude::*;

struct Setup<'a> {
    series: &'a [Vec<f64>],
    ticks: Vec<Vec<Option<i64>>>,
    /// The caller's flags for each dimension's windows, if any.
    flags: Option<Vec<Vec<bool>>>,
    m: usize,
}

impl Setup<'_> {
    fn flags_of(&self, a: usize) -> Option<&[bool]> {
        self.flags.as_ref().map(|flags| flags[a].as_slice())
    }

    fn ticks_of(&self, a: usize) -> Ticks<'_> {
        match self.flags_of(a) {
            Some(flags) => Ticks::with_constant_windows(&self.ticks[a], flags),
            None => Ticks::from(&self.ticks[a]),
        }
    }

    fn series_of(&self, a: usize) -> Series<'_> {
        match self.flags_of(a) {
            Some(flags) => {
                Series::with_constant_windows(&self.series[a], nz(self.m), flags).unwrap()
            }
            None => Series::from(&self.series[a]),
        }
    }
}

/// A pair's exact order, nearer smaller.
#[derive(Clone, Copy)]
struct Key(Rho);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_rho(other.0, self.0)
    }
}

impl RefMetric for Setup<'_> {
    type Key = Key;

    fn dimensions(&self) -> usize {
        self.series.len()
    }

    fn samples(&self) -> usize {
        self.series[0].len()
    }

    fn m(&self) -> usize {
        self.m
    }

    fn part(&self, a: usize, window: usize, start: usize) -> Option<(f64, Key)> {
        pair(self.ticks_of(a), self.m, window, start).map(|(rho, d2)| (libm::sqrt(d2), Key(rho)))
    }

    fn caller_units(&self, mean: f64) -> f64 {
        mean
    }

    fn describe(
        &self,
        starts: &[usize],
        neighbors: &[usize],
        include: &[usize],
    ) -> Result<(Vec<f64>, Vec<Subspace>), MultiError> {
        let set: Vec<(Series, Scale)> = (0..self.series.len())
            .map(|a| (self.series_of(a), unit()))
            .collect();
        naive_mdl(
            &set,
            nz(self.m),
            starts,
            neighbors,
            include,
            false,
            Bits::EIGHT,
        )
    }
}

impl Setup<'_> {
    /// The reference motifs; every series is quantized at the unit scale.
    fn motifs(
        &self,
        _scales: &[Scale],
        distance: &[Vec<f64>],
        index: &[Vec<SampleIndex>],
        params: &MultiMotifParams<'_>,
    ) -> Result<Vec<Plain>, MultiError> {
        reference(self, distance, index, params)
    }
}

fn library(
    setup: &Setup<'_>,
    scales: &[Scale],
    distance: &[Vec<f64>],
    index: &[Vec<SampleIndex>],
    params: &MultiMotifParams<'_>,
) -> Result<Vec<Plain>, MultiError> {
    let set: Vec<(Series, Scale)> = (0..setup.series.len())
        .map(|a| (setup.series_of(a), scales[a]))
        .collect();
    mmotifs(&set, nz(setup.m), distance, index, params)
        .map(|found| found.iter().map(plain).collect())
}

/// Every parameter set agrees with the reference; returns how many had a motif.
fn assert_equals_reference(input: &Input) -> usize {
    let scales = vec![unit(); input.dimensions.len()];
    let samples: Vec<_> = input
        .dimensions
        .iter()
        .map(|series| ticks(series, unit()))
        .collect();
    let refs: Vec<&[Option<i64>]> = samples.iter().map(Vec::as_slice).collect();
    let profile = naive_multi_profile(
        &refs,
        input.m,
        &input.include,
        input.discords,
        NonFinitePolicy::Missing,
    )
    .unwrap();
    let (distance, index) = rows(&profile);
    let setup = Setup {
        series: &input.dimensions,
        ticks: samples.clone(),
        flags: None,
        m: input.m,
    };
    let d = input.dimensions.len();
    let reach = libm::sqrt(input.m as f64);
    let per_row: Vec<f64> = (0..d).map(|k| reach * (k as f64) / d as f64).collect();
    let mut nonempty = 0;
    for params in grid(&input.include, reach, d, &per_row) {
        let want = setup.motifs(&scales, &distance, &index, &params).unwrap();
        let got = library(&setup, &scales, &distance, &index, &params).unwrap();
        assert_eq!(got, want, "{input:?} {params:?}");
        nonempty += usize::from(!got.is_empty());
    }
    nonempty
}

#[test]
fn inputs_equal_the_reference() {
    let (cases, nonempty) = (core::cell::Cell::new(0usize), core::cell::Cell::new(0usize));
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(12))
        .run(&input(8), |input| {
            prop_assume!(input.dimensions[0].len() >= input.m);
            nonempty.set(nonempty.get() + assert_equals_reference(&input));
            cases.set(cases.get() + 1);
            Ok(())
        })
        .unwrap();
    assert!(
        nonempty.get() >= cases.get(),
        "{} in {}",
        nonempty.get(),
        cases.get()
    );
}

#[test]
fn tie_inputs_equal_the_reference_and_find_motifs() {
    let nonempty: usize = tie_inputs()
        .iter()
        .step_by(4)
        .map(assert_equals_reference)
        .sum();
    assert!(nonempty > 60, "{nonempty}");
}

#[test]
fn profiles_of_every_thread_count_give_one_answer() {
    for input in tie_inputs().iter().step_by(7) {
        let scales = vec![unit(); input.dimensions.len()];
        let dimensions = slices(&input.dimensions);
        let setup = Setup {
            series: &input.dimensions,
            ticks: input.dimensions.iter().map(|s| ticks(s, unit())).collect(),
            flags: None,
            m: input.m,
        };
        let params = MultiMotifParams {
            max_motifs: 4,
            include: &input.include,
            ..MultiMotifParams::default()
        };
        let mut answers = Vec::new();
        for threads in THREADS {
            let profile = mstump(
                &dimensions,
                nz(input.m),
                &scales,
                &input.include,
                input.discords,
                nz(threads),
            )
            .unwrap();
            let (distance, index) = rows(&profile);
            answers.push(library(&setup, &scales, &distance, &index, &params).unwrap());
        }
        assert_eq!(answers[0], answers[1]);
        assert_eq!(answers[0], answers[2]);
    }
}

/// A profile made by hand, full of equal values and arbitrary neighbors (including none and a
/// stale one), agrees with the reference: the procedure takes `P` and `I` as given.
#[test]
fn hand_made_profiles_equal_the_reference() {
    let n = 50;
    let m = 4;
    let mut series: Vec<Vec<f64>> = (0..3)
        .map(|a| kind_series([3, 4, 0][a], n, 5 + a as u64))
        .collect();
    // Candidates the profile points at can have gaps in the subspace's dimensions.
    series[2][10..20].fill(f64::NAN);
    series[0][38..44].fill(f64::NAN);
    let windows = n - m + 1;
    let scales = vec![unit(); 3];
    let setup = Setup {
        ticks: series.iter().map(|s| ticks(s, unit())).collect(),
        flags: None,
        series: &series,
        m,
    };
    let mut state = 12345u64;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    for round in 0..12 {
        let levels = 1 + round % 4;
        let distance: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                (0..windows)
                    .map(|_| match next(9) {
                        0 => f64::INFINITY,
                        _ => next(levels) as f64 * 0.5 + 0.5,
                    })
                    .collect()
            })
            .collect();
        let index: Vec<Vec<SampleIndex>> = (0..3)
            .map(|_| {
                (0..windows)
                    .map(|_| match next(10) {
                        0 => SampleIndex::NONE,
                        _ => SampleIndex::new(next(windows as u64)).unwrap(),
                    })
                    .collect()
            })
            .collect();
        let per_row = [1.0, 1.5, 2.0];
        for params in grid(&[], 3.0, 3, &per_row).iter() {
            let want = setup.motifs(&scales, &distance, &index, params).unwrap();
            let got = library(&setup, &scales, &distance, &index, params).unwrap();
            assert_eq!(got, want, "{params:?}");
        }
    }
}

fn walk_input() -> (Vec<Vec<f64>>, usize) {
    ((0..3).map(|a| kind_series(0, 60, 40 + a)).collect(), 5)
}

fn walk_profile(series: &[Vec<f64>], m: usize) -> (Vec<Vec<f64>>, Vec<Vec<SampleIndex>>) {
    let refs = slices(series);
    rows(&mstump(&refs, nz(m), &vec![unit(); series.len()], &[], false, nz(1)).unwrap())
}

#[test]
fn refused_inputs_and_empty_results() {
    let (series, m) = walk_input();
    let (distance, index) = walk_profile(&series, m);
    let scales = vec![unit(); 3];
    let setup = Setup {
        ticks: series.iter().map(|s| ticks(s, unit())).collect(),
        flags: None,
        series: &series,
        m,
    };
    let run = |distance: &[Vec<f64>], index: &[Vec<SampleIndex>], params: &MultiMotifParams<'_>| {
        library(&setup, &scales, distance, index, params)
    };
    let defaults = MultiMotifParams::default();
    assert!(run(&distance, &index, &defaults).is_ok());
    let k = |k| MultiMotifParams {
        k: Some(k),
        ..defaults
    };
    assert_eq!(run(&distance, &index, &k(3)), Err(MultiError::KOutOfRange));
    assert_eq!(
        run(&distance, &index, &k(usize::MAX)),
        Err(MultiError::KOutOfRange)
    );
    assert!(run(&distance, &index, &k(2)).is_ok());
    let short = [0.0, 1.0];
    let params = MultiMotifParams {
        cutoffs: Cutoffs::PerRow(&short),
        ..defaults
    };
    assert_eq!(
        run(&distance, &index, &params),
        Err(MultiError::CutoffCount)
    );
    assert_eq!(
        run(&distance[..2], &index, &defaults),
        Err(MultiError::ProfileShape)
    );
    assert_eq!(
        run(&distance, &index[..2], &defaults),
        Err(MultiError::ProfileShape)
    );
    let mut shorter = distance.clone();
    shorter[1].pop();
    assert_eq!(
        run(&shorter, &index, &defaults),
        Err(MultiError::ProfileShape)
    );
    let mut stray = index.clone();
    stray[2][3] = SampleIndex::new(distance[0].len() as u64).unwrap();
    assert_eq!(
        run(&distance, &stray, &defaults),
        Err(MultiError::ProfileShape)
    );
    let include = [3];
    let params = MultiMotifParams {
        include: &include,
        ..defaults
    };
    assert_eq!(
        run(&distance, &index, &params),
        Err(MultiError::IncludeOutOfRange)
    );
    let long = MultiMotifParams::default();
    let set: Vec<(&[f64], Scale)> = series.iter().map(|s| (s.as_slice(), unit())).collect();
    assert_eq!(
        mmotifs(&set, nz(61), &distance, &index, &long),
        Err(MultiError::WindowTooLong)
    );
    assert_eq!(
        mmotifs::<&[f64]>(&[], nz(m), &[], &[], &long),
        Err(MultiError::NoDimensions)
    );
    let mut unequal = set.clone();
    unequal[1].0 = &series[1][..59];
    assert_eq!(
        mmotifs(&unequal, nz(m), &distance, &index, &long),
        Err(MultiError::UnequalLengths)
    );

    // No motifs: no candidate has a distance, nothing may have zero motifs asked, a negative
    // `max_distance` leaves no match.
    let infinite = vec![vec![f64::INFINITY; distance[0].len()]; 3];
    assert_eq!(run(&infinite, &index, &defaults), Ok(vec![]));
    let none = MultiMotifParams {
        max_motifs: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..defaults
    };
    assert_eq!(run(&distance, &index, &none), Ok(vec![]));
    // A NaN in a row is that row's smallest value, as NumPy's `argmin` has it, and a candidate
    // without a distance ends the search: the same profile without it finds motifs.
    let all = MultiMotifParams {
        k: Some(1),
        min_neighbors: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..defaults
    };
    assert!(!run(&distance, &index, &all).unwrap().is_empty());
    let mut not_a_number = distance.clone();
    not_a_number[1][distance[1].len() / 2] = f64::NAN;
    assert_eq!(run(&not_a_number, &index, &all), Ok(vec![]));
    let negative = MultiMotifParams {
        max_distance: MaxDistance::Value(-1.0),
        cutoffs: Cutoffs::All(f64::INFINITY),
        max_motifs: usize::MAX,
        ..defaults
    };
    assert_eq!(run(&distance, &index, &negative), Ok(vec![]));
    let zero_matches = MultiMotifParams {
        max_matches: Some(0),
        min_neighbors: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        max_motifs: usize::MAX,
        ..defaults
    };
    assert_eq!(run(&distance, &index, &zero_matches), Ok(vec![]));
}

/// Every motif of a series, `max_motifs` past what exists: it ends, with motifs outside each
/// other's zones.
#[test]
fn all_motifs_end_and_stay_apart() {
    let (series, m) = walk_input();
    let (distance, index) = walk_profile(&series, m);
    let scales = vec![unit(); 3];
    let setup = Setup {
        ticks: series.iter().map(|s| ticks(s, unit())).collect(),
        flags: None,
        series: &series,
        m,
    };
    let params = MultiMotifParams {
        max_motifs: usize::MAX,
        min_neighbors: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        max_matches: None,
        ..MultiMotifParams::default()
    };
    let got = library(&setup, &scales, &distance, &index, &params).unwrap();
    assert_eq!(
        got,
        setup.motifs(&scales, &distance, &index, &params).unwrap()
    );
    let starts: Vec<u64> = got.iter().map(|motif| motif.0[0].0).collect();
    for (a, &x) in starts.iter().enumerate() {
        for &y in &starts[a + 1..] {
            assert!(
                x.abs_diff(y) > ladde::exclusion::zone(m) as u64,
                "{starts:?}"
            );
        }
    }
    assert!(got.len() > 3);
}

/// A cutoff equal to a candidate's own distance admits it; one just below does not.
#[test]
fn the_cutoff_admits_a_distance_equal_to_it() {
    let (series, m) = walk_input();
    let (distance, index) = walk_profile(&series, m);
    let scales = vec![unit(); 3];
    let setup = Setup {
        ticks: series.iter().map(|s| ticks(s, unit())).collect(),
        flags: None,
        series: &series,
        m,
    };
    let best = |k: usize| distance[k].iter().copied().fold(f64::INFINITY, f64::min);
    for k in 0..3 {
        let at = |cutoff: f64| MultiMotifParams {
            k: Some(k),
            min_neighbors: 0,
            cutoffs: Cutoffs::All(cutoff),
            max_distance: MaxDistance::Value(f64::INFINITY),
            ..MultiMotifParams::default()
        };
        let admitted = library(&setup, &scales, &distance, &index, &at(best(k))).unwrap();
        assert_eq!(admitted.len(), 1, "k {k}");
        let below = f64::from_bits(best(k).to_bits() - 1);
        assert!(
            library(&setup, &scales, &distance, &index, &at(below))
                .unwrap()
                .is_empty()
        );
        let max = MultiMotifParams {
            max_distance: MaxDistance::Value(best(k)),
            cutoffs: Cutoffs::All(f64::INFINITY),
            ..at(0.0)
        };
        assert_eq!(
            library(&setup, &scales, &distance, &index, &max)
                .unwrap()
                .len(),
            1
        );
        let below_max = MultiMotifParams {
            max_distance: MaxDistance::Value(below),
            ..max
        };
        assert!(
            library(&setup, &scales, &distance, &index, &below_max)
                .unwrap()
                .is_empty()
        );
    }
}

/// One dimension: the motifs are the single-dimensional ones' matches, equal to the reference.
#[test]
fn one_dimension_and_a_window_as_long_as_the_series() {
    let series = vec![kind_series(3, 30, 3)];
    let scales = vec![unit()];
    let samples = vec![ticks(&series[0], unit())];
    let profile = naive_multi_profile(
        &[samples[0].as_slice()],
        6,
        &[],
        false,
        NonFinitePolicy::Missing,
    )
    .unwrap();
    let (distance, index) = rows(&profile);
    let setup = Setup {
        flags: None,
        ticks: samples,
        series: &series,
        m: 6,
    };
    for params in grid(&[], 3.0, 1, &[1.0]) {
        let want = setup.motifs(&scales, &distance, &index, &params).unwrap();
        assert_eq!(
            library(&setup, &scales, &distance, &index, &params).unwrap(),
            want
        );
    }
    // m = n: one window, no neighbor, so nothing to find.
    let short = vec![vec![1.0, 2.0, 4.0, 3.0]];
    let setup = Setup {
        flags: None,
        ticks: vec![ticks(&short[0], unit())],
        series: &short,
        m: 4,
    };
    let params = MultiMotifParams::default();
    assert_eq!(
        library(
            &setup,
            &scales,
            &[vec![f64::INFINITY]],
            &[vec![SampleIndex::NONE]],
            &params
        ),
        Ok(vec![])
    );
}

/// The scale a series is quantized at does not change a z-normalized distance: the same ticks at
/// another scale give the same motifs, bit for bit.
#[test]
fn the_scale_is_not_part_of_a_znormalized_distance() {
    let (series, m) = walk_input();
    let (distance, index) = walk_profile(&series, m);
    let setup = Setup {
        ticks: series.iter().map(|s| ticks(s, unit())).collect(),
        flags: None,
        series: &series,
        m,
    };
    let scaled: Vec<Vec<f64>> = series
        .iter()
        .map(|s| s.iter().map(|&x| x * 0.25).collect())
        .collect();
    let scale = Scale::new(0.25).unwrap();
    let set: Vec<(&[f64], Scale)> = scaled.iter().map(|s| (s.as_slice(), scale)).collect();
    let params = MultiMotifParams {
        max_motifs: 5,
        min_neighbors: 0,
        max_matches: Some(4),
        max_distance: MaxDistance::Value(f64::INFINITY),
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..MultiMotifParams::default()
    };
    let got: Vec<Plain> = mmotifs(&set, nz(m), &distance, &index, &params)
        .unwrap()
        .iter()
        .map(plain)
        .collect();
    let unit_scale = library(&setup, &[unit(); 3], &distance, &index, &params).unwrap();
    assert!(got.len() >= 2 && got[0].0.len() == 4);
    assert_eq!(got, unit_scale);
}

/// Runs of gaps longer than a window at the very start, in the middle and at the very end of
/// different dimensions: windows touching them have no distance in that dimension.
#[test]
fn gap_runs_equal_the_reference() {
    let n = 64;
    let mut dimensions: Vec<Vec<f64>> = (0..3)
        .map(|a| kind_series([3, 0, 5][a], n, 30 + a as u64))
        .collect();
    dimensions[0][..9].fill(f64::NAN);
    dimensions[1][24..40].fill(f64::NAN);
    dimensions[2][n - 10..].fill(f64::NAN);
    let input = Input {
        dimensions,
        m: 6,
        include: vec![1],
        discords: false,
    };
    let found = assert_equals_reference(&input);
    assert!(found > 1, "{found}");
}

/// Two windows at an exactly equal distance whose f64 roots differ in the last bit are ordered by
/// their start, not by that noise (windows 3 and 9 of this series against window 11).
#[test]
fn exactly_equal_distances_are_ordered_by_start_not_by_rounding() {
    let series = vec![vec![
        2.0, 0.0, 4.0, 0.0, 3.0, 3.0, 2.0, 2.0, 4.0, 2.0, 1.0, 2.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0,
        1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 4.0, 4.0,
    ]];
    let m = 3;
    let windows = series[0].len() - m + 1;
    let mut distance = vec![vec![5.0; windows]];
    distance[0][11] = 0.0;
    let mut neighbors = vec![SampleIndex::new(0).unwrap(); windows];
    neighbors[11] = SampleIndex::new(20).unwrap();
    let index = vec![neighbors];
    let setup = Setup {
        flags: None,
        ticks: vec![ticks(&series[0], unit())],
        series: &series,
        m,
    };
    let params = MultiMotifParams {
        cutoffs: Cutoffs::All(9.0),
        max_distance: MaxDistance::Value(3.0),
        max_matches: None,
        min_neighbors: 0,
        ..MultiMotifParams::default()
    };
    let got = library(&setup, &[unit()], &distance, &index, &params).unwrap();
    assert_eq!(
        got,
        setup.motifs(&[unit()], &distance, &index, &params).unwrap()
    );
    let starts: Vec<u64> = got[0].0.iter().map(|&(start, _)| start).collect();
    let position = |start| starts.iter().position(|&s| s == start).unwrap();
    assert!(position(3) < position(9), "{starts:?}");
}

/// A caller's flags for which windows are constant reach the subspace choice and the matches: with
/// flags per dimension (the default rule, none constant, all constant, spread-out ones) the library
/// equals the reference given the same flags, on a profile made with the same flags.
#[test]
fn flagged_windows_equal_the_reference() {
    let mut state = 99u64;
    let mut nonempty = 0;
    for input in tie_inputs().iter().step_by(5) {
        let samples: Vec<_> = input.dimensions.iter().map(|s| ticks(s, unit())).collect();
        let windows = input.dimensions[0].len() + 1 - input.m;
        for which in 0..4 {
            let flags: Vec<Vec<bool>> = input
                .dimensions
                .iter()
                .enumerate()
                .map(|(a, series)| {
                    (0..windows)
                        .map(|i| match (which + a) % 4 {
                            0 => series[i..i + input.m].iter().all(|x| *x == series[i]),
                            1 => false,
                            2 => true,
                            _ => {
                                state = state
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                (state >> 33).is_multiple_of(3)
                            }
                        })
                        .collect()
                })
                .collect();
            let reference_ticks: Vec<Ticks> = samples
                .iter()
                .zip(&flags)
                .map(|(t, f)| Ticks::with_constant_windows(t, f))
                .collect();
            let profile = naive_multi_profile(
                &reference_ticks,
                input.m,
                &input.include,
                input.discords,
                NonFinitePolicy::Missing,
            )
            .unwrap();
            let (distance, index) = rows(&profile);
            let setup = Setup {
                series: &input.dimensions,
                ticks: samples.clone(),
                flags: Some(flags),
                m: input.m,
            };
            let scales = vec![unit(); input.dimensions.len()];
            let d = input.dimensions.len();
            let reach = libm::sqrt(input.m as f64);
            let per_row: Vec<f64> = (0..d).map(|k| reach * (k as f64) / d as f64).collect();
            for params in grid(&input.include, reach, d, &per_row) {
                let want = setup.motifs(&scales, &distance, &index, &params).unwrap();
                let got = library(&setup, &scales, &distance, &index, &params).unwrap();
                assert_eq!(got, want, "flags {which} {params:?}");
                nonempty += usize::from(!got.is_empty());
            }
        }
    }
    assert!(nonempty > 20, "{nonempty}");
}
