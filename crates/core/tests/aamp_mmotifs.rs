#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Non-normalized multi-dimensional motifs (STUMPY's `aamp_mmotifs`) equal a plain, independent
//! statement of the procedure on exact integer ticks: squared distances are integers, every order
//! is on the f64 mean of their roots (ties of one dimension on the integer), the description
//! length reference chooses `k` and the subspace; on inputs of exact ties and gaps, over a grid of
//! parameters, at a scale other than one, with profiles from the reference and from the batch
//! engine at every thread count; and the two dispatches see different distances.

#[path = "common/mmotifs_ref.rs"]
mod mmotifs_ref;
#[path = "common/multi.rs"]
mod multi;

use ladde::batch::multi::maamp;
use ladde::index::SampleIndex;
use ladde::motif::{Cutoffs, MultiMotifParams, mmotifs, mmotifs_nonnorm};
use ladde::multidim::{MultiError, NonFinitePolicy};
use ladde::naive::{naive_maamp_mdl, naive_multi_profile_nonnorm};
use ladde::quantize::Scale;
use ladde::subspace::{Bits, Subspace};
use mmotifs_ref::{Plain, RefMetric, grid, plain, reference, rows};
use multi::*;
use proptest::prelude::*;

struct Setup {
    series: Vec<Vec<f64>>,
    ticks: Vec<Vec<Option<i64>>>,
    scale: Scale,
    m: usize,
}

impl Setup {
    /// `dimensions`, every sample multiplied by `scale` (exactly, a power of two), so the ticks at
    /// that scale are the integers they were.
    fn new(dimensions: &[Vec<f64>], scale: f64, m: usize) -> Self {
        let scale = Scale::new(scale).unwrap();
        let series: Vec<Vec<f64>> = dimensions
            .iter()
            .map(|d| d.iter().map(|&x| x * scale.get()).collect())
            .collect();
        Self {
            ticks: series.iter().map(|s| ticks(s, scale)).collect(),
            series,
            scale,
            m,
        }
    }

    fn refs(&self) -> Vec<&[f64]> {
        self.series.iter().map(Vec::as_slice).collect()
    }

    fn library(
        &self,
        distance: &[Vec<f64>],
        index: &[Vec<SampleIndex>],
        params: &MultiMotifParams<'_>,
    ) -> Result<Vec<Plain>, MultiError> {
        mmotifs_nonnorm(
            &self.refs(),
            nz(self.m),
            self.scale,
            distance,
            index,
            params,
        )
        .map(|found| found.iter().map(plain).collect())
    }

    /// The profile the batch reference gives, in the caller's units.
    fn profile(&self, include: &[usize], discords: bool) -> (Vec<Vec<f64>>, Vec<Vec<SampleIndex>>) {
        let refs: Vec<&[Option<i64>]> = self.ticks.iter().map(Vec::as_slice).collect();
        let profile =
            naive_multi_profile_nonnorm(&refs, self.m, include, discords, NonFinitePolicy::Missing)
                .unwrap();
        let (mut distance, index) = rows(&profile);
        for x in distance.iter_mut().flatten() {
            *x *= self.scale.get();
        }
        (distance, index)
    }
}

impl RefMetric for Setup {
    type Key = u128;

    fn dimensions(&self) -> usize {
        self.series.len()
    }

    fn samples(&self) -> usize {
        self.series[0].len()
    }

    fn m(&self) -> usize {
        self.m
    }

    fn part(&self, a: usize, window: usize, start: usize) -> Option<(f64, u128)> {
        let take = |at: usize| -> Option<Vec<i64>> {
            self.ticks[a][at..at + self.m].iter().copied().collect()
        };
        let (x, y) = (take(window)?, take(start)?);
        let lowest = x.iter().chain(&y).copied().min().unwrap();
        let highest = x.iter().chain(&y).copied().max().unwrap();
        let range = (i128::from(highest) - i128::from(lowest)) as u128;
        range
            .checked_mul(range)
            .and_then(|squared| squared.checked_mul(self.m as u128))?;
        let squared: u128 = x
            .iter()
            .zip(&y)
            .map(|(&p, &q)| {
                let d = i128::from(p) - i128::from(q);
                (d * d) as u128
            })
            .sum();
        Some((libm::sqrt(squared as f64), squared))
    }

    fn caller_units(&self, mean: f64) -> f64 {
        mean * self.scale.get()
    }

    fn describe(
        &self,
        starts: &[usize],
        neighbors: &[usize],
        include: &[usize],
    ) -> Result<(Vec<f64>, Vec<Subspace>), MultiError> {
        naive_maamp_mdl(
            &self.refs(),
            nz(self.m),
            starts,
            neighbors,
            include,
            false,
            Bits::EIGHT,
        )
    }
}

/// Every parameter set agrees with the reference at the scales; returns how many had a motif.
fn assert_equals_reference(input: &Input, scales: &[f64]) -> usize {
    let d = input.dimensions.len();
    let reach = libm::sqrt(input.m as f64);
    let mut nonempty = 0;
    for &scale in scales {
        let setup = Setup::new(&input.dimensions, scale, input.m);
        let (distance, index) = setup.profile(&input.include, input.discords);
        // Thresholds in the caller's units, around where distances of small integer series lie.
        let per_row: Vec<f64> = (0..d)
            .map(|k| 4.0 * scale * (k as f64) / d as f64)
            .collect();
        for params in grid(&input.include, 4.0 * scale * reach, d, &per_row) {
            let want = reference(&setup, &distance, &index, &params).unwrap();
            let got = setup.library(&distance, &index, &params).unwrap();
            assert_eq!(got, want, "{input:?} scale {scale} {params:?}");
            nonempty += usize::from(!got.is_empty());
        }
    }
    nonempty
}

#[test]
fn inputs_equal_the_reference() {
    let (cases, nonempty) = (core::cell::Cell::new(0usize), core::cell::Cell::new(0usize));
    proptest::test_runner::TestRunner::new(testkit::generators::seeded(10))
        .run(&input(9), |input| {
            prop_assume!(input.dimensions[0].len() >= input.m);
            nonempty.set(nonempty.get() + assert_equals_reference(&input, &[1.0, 0.25]));
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
        .map(|input| assert_equals_reference(input, &[1.0, 0.5]))
        .sum();
    assert!(nonempty > 60, "{nonempty}");
}

#[test]
fn profiles_of_every_thread_count_give_one_answer() {
    for input in tie_inputs().iter().step_by(7) {
        let setup = Setup::new(&input.dimensions, 1.0, input.m);
        let params = MultiMotifParams {
            max_motifs: 4,
            include: &input.include,
            ..MultiMotifParams::default()
        };
        let mut answers = Vec::new();
        for threads in THREADS {
            let profile = maamp(
                &slices(&input.dimensions),
                nz(input.m),
                setup.scale,
                &input.include,
                input.discords,
                nz(threads),
            )
            .unwrap();
            let (distance, index) = rows(&profile);
            answers.push(setup.library(&distance, &index, &params).unwrap());
        }
        assert_eq!(answers[0], answers[1]);
        assert_eq!(answers[0], answers[2]);
    }
}

/// Dimensions that are scaled copies of one shape: z-normalized, every dimension is the same
/// distance from itself, so the metrics' motifs differ in which windows match and how far; the
/// dispatch must give each its own.
#[test]
fn the_two_metrics_see_different_distances() {
    let shape = [3.0, -7.0, 5.0, 1.0, -2.0, 8.0, -4.0];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for (k, c) in [
        (1.0, 0.0),
        (3.0, 24.0),
        (-2.0, 5.0),
        (5.0, -11.0),
        (2.0, 3.0),
    ] {
        a.extend(shape.iter().map(|&x| k * x + c));
        a.extend([1.0, 4.0, 1.0]);
        b.extend(shape.iter().map(|&x| 2.0 * k * x - c));
        b.extend([4.0, 1.0, 4.0]);
    }
    let dimensions = vec![a, b];
    let m = 7;
    let setup = Setup::new(&dimensions, 1.0, m);
    let (distance, index) = setup.profile(&[], false);
    let params = MultiMotifParams {
        max_motifs: 3,
        max_matches: None,
        min_neighbors: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..MultiMotifParams::default()
    };
    let nonnorm = setup.library(&distance, &index, &params).unwrap();
    assert_eq!(
        nonnorm,
        reference(&setup, &distance, &index, &params).unwrap()
    );
    // The z-normalized procedure over the same profile and series.
    let set: Vec<(&[f64], Scale)> = dimensions
        .iter()
        .map(|d| (d.as_slice(), Scale::new(1.0).unwrap()))
        .collect();
    let znorm: Vec<Plain> = mmotifs(&set, nz(m), &distance, &index, &params)
        .unwrap()
        .iter()
        .map(plain)
        .collect();
    assert!(!nonnorm.is_empty() && !znorm.is_empty());
    assert_ne!(nonnorm, znorm);
}

#[test]
fn edges_and_refusals() {
    let dimensions = vec![
        kind_series(0, 60, 1),
        kind_series(3, 60, 2),
        kind_series(0, 60, 3),
    ];
    let setup = Setup::new(&dimensions, 1.0, 5);
    let (distance, index) = setup.profile(&[], false);
    let defaults = MultiMotifParams::default();
    assert!(setup.library(&distance, &index, &defaults).is_ok());
    let k = MultiMotifParams {
        k: Some(3),
        ..defaults
    };
    assert_eq!(
        setup.library(&distance, &index, &k),
        Err(MultiError::KOutOfRange)
    );
    let include = [3];
    let params = MultiMotifParams {
        include: &include,
        ..defaults
    };
    assert_eq!(
        setup.library(&distance, &index, &params),
        Err(MultiError::IncludeOutOfRange)
    );
    assert_eq!(
        setup.library(&distance[..2], &index, &defaults),
        Err(MultiError::ProfileShape)
    );
    let short = [1.0];
    let params = MultiMotifParams {
        cutoffs: Cutoffs::PerRow(&short),
        ..defaults
    };
    assert_eq!(
        setup.library(&distance, &index, &params),
        Err(MultiError::CutoffCount)
    );
    let infinite = vec![vec![f64::INFINITY; distance[0].len()]; 3];
    assert_eq!(setup.library(&infinite, &index, &defaults), Ok(vec![]));
    let all = MultiMotifParams {
        max_motifs: usize::MAX,
        min_neighbors: 0,
        cutoffs: Cutoffs::All(f64::INFINITY),
        ..defaults
    };
    let got = setup.library(&distance, &index, &all).unwrap();
    assert_eq!(got, reference(&setup, &distance, &index, &all).unwrap());
    assert!(got.len() > 2);
    let none = MultiMotifParams {
        max_matches: Some(0),
        ..all
    };
    assert_eq!(setup.library(&distance, &index, &none), Ok(vec![]));
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
    let found = assert_equals_reference(&input, &[1.0, 0.5]);
    assert!(found > 2, "{found}");
}
