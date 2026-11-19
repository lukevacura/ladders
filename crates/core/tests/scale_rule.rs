#![cfg(not(any(loom, shuttle)))]

//! The scales rule 1 picks, pinned on fixed inputs and compared on bits.
//!
//! A change to any expected value in this file needs a new `SCALE_RULE`: the table and the
//! number change in one commit, and the release notes say which inputs moved. Inputs are built
//! from integer arithmetic and single correctly rounded float operations (no `sin`, no `powf`),
//! by a generator written out here, so nothing but the rule can move a value.

use core::num::NonZeroUsize;

use ladde::quantize::{
    FAST_TICKS_NUMERATOR, OUTLIER_WINDOW_SHARE, SCALE_RULE, SPAN_SLACK, Scale, ScaleChoice,
};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// SplitMix64, written out: the inputs must not move when a library generator does.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A whole number in `-reach..=reach`.
    fn whole(&mut self, reach: u64) -> f64 {
        (self.next() % (2 * reach + 1)) as f64 - reach as f64
    }

    /// A multiple of `2⁻⁵³` in `[-0.5, 0.5)`.
    fn step(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// `2^e`, exact by construction.
fn two(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// `n` samples, each one float operation `op` on a whole number within `reach` of zero.
fn grid(seed: u64, n: usize, reach: u64, op: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut mix = Mix(seed);
    (0..n).map(|_| op(mix.whole(reach))).collect()
}

/// A random walk of `n` samples on no decimal grid: the running sum of steps in `[-0.5, 0.5)`.
fn walk(seed: u64, n: usize) -> Vec<f64> {
    let mut mix = Mix(seed);
    let mut level = 0.0;
    (0..n)
        .map(|_| {
            level += mix.step();
            level
        })
        .collect()
}

/// `series` with `spike` added at every `every`-th sample, from sample `every / 2`.
fn spiked(mut series: Vec<f64>, every: usize, spike: f64) -> Vec<f64> {
    for x in series.iter_mut().skip(every / 2).step_by(every) {
        *x += spike;
    }
    series
}

/// `series` with a NaN run, one infinity and one negative infinity inside it.
fn gapped(mut series: Vec<f64>) -> Vec<f64> {
    let n = series.len();
    for x in &mut series[n / 3..n / 3 + 5] {
        *x = f64::NAN;
    }
    series[n / 2] = f64::INFINITY;
    series[2 * n / 3] = f64::NEG_INFINITY;
    series
}

/// One pinned input: its name, the series that share the scale, the window length, and the
/// scale the default rule and the finest rule each pick (`None`: no scale).
type Case = (&'static str, Vec<Vec<f64>>, usize, Option<f64>, Option<f64>);

fn cases() -> Vec<Case> {
    let one = |series: Vec<f64>| vec![series];
    let scaled = |series: Vec<f64>, by: f64| series.iter().map(|x| x * by).collect::<Vec<f64>>();
    let cents = |seed: u64, n: usize| grid(seed, n, 100_000, |k| k / 100.0);
    vec![
        // Data on a grid gets the coarsest candidate that holds it.
        ("cents", one(cents(1, 600)), 16, Some(0.01), Some(0.01)),
        (
            "whole numbers",
            one(grid(2, 600, 1000, |k| k)),
            16,
            Some(1.0),
            Some(1.0),
        ),
        (
            "halves",
            one(grid(3, 600, 1000, |k| k * 0.5)),
            16,
            Some(0.5),
            Some(0.5),
        ),
        (
            "steps of 2.5",
            one(grid(4, 600, 1000, |k| k * 2.5)),
            16,
            Some(2.5),
            Some(2.5),
        ),
        (
            "steps of 20",
            one(grid(5, 600, 1000, |k| k * 20.0)),
            16,
            Some(20.0),
            Some(20.0),
        ),
        (
            "thousands",
            one(grid(6, 600, 1000, |k| k * 1000.0)),
            16,
            Some(1000.0),
            Some(1000.0),
        ),
        (
            "millionths",
            one(grid(7, 600, 1000, |k| k / 1e6)),
            16,
            Some(1e-6),
            Some(1e-6),
        ),
        (
            "steps of 1e-12",
            one(grid(8, 600, 1000, |k| k / 1e12)),
            16,
            Some(1e-12),
            Some(1e-12),
        ),
        (
            "a binary grid",
            one(grid(9, 600, 1 << 20, |k| k * two(-20))),
            16,
            Some(two(-20)),
            Some(two(-20)),
        ),
        (
            "cents, long windows",
            one(cents(11, 3000)),
            1024,
            Some(0.01),
            Some(0.01),
        ),
        (
            "cents with gaps",
            one(gapped(cents(12, 600))),
            16,
            Some(0.01),
            Some(0.01),
        ),
        ("a constant", one(vec![0.1; 300]), 16, Some(0.1), Some(0.1)),
        // Zeros lie on every grid, so they get the first candidate.
        ("zeros", one(vec![0.0; 300]), 16, Some(5e9), Some(5e9)),
        // A grid the rule does not take: its windows are too wide for the kernel at it, or its
        // samples lie past 2^40 ticks.
        (
            "a binary grid too fine for its windows",
            one(grid(9, 600, 1 << 30, |k| k * two(-20))),
            16,
            Some(two(-17)),
            Some(two(-17)),
        ),
        (
            "cents past 2^40 ticks",
            one(grid(10, 600, 100_000, |k| 1e11 + k / 100.0)),
            16,
            Some(two(-16)),
            Some(two(-16)),
        ),
        // Data on no grid: the fast-path bound, which m moves.
        (
            "a walk, m 8",
            one(walk(13, 4096)),
            8,
            Some(two(-25)),
            Some(two(-28)),
        ),
        (
            "a walk, m 64",
            one(walk(13, 4096)),
            64,
            Some(two(-22)),
            Some(two(-24)),
        ),
        (
            "a walk, m 512",
            one(walk(13, 4096)),
            512,
            Some(two(-18)),
            Some(two(-20)),
        ),
        (
            "a walk, m 2048",
            one(walk(13, 4096)),
            2048,
            Some(two(-16)),
            Some(two(-17)),
        ),
        (
            "a walk shorter than its span",
            one(walk(14, 100)),
            16,
            Some(two(-25)),
            Some(two(-27)),
        ),
        (
            "a walk with gaps",
            one(gapped(walk(15, 4096))),
            64,
            Some(two(-22)),
            Some(two(-24)),
        ),
        (
            "a large walk",
            one(scaled(walk(16, 4096), two(40))),
            64,
            Some(two(18)),
            Some(two(16)),
        ),
        (
            "a small walk",
            one(scaled(walk(17, 4096), two(-40))),
            64,
            Some(two(-62)),
            Some(two(-64)),
        ),
        // Outliers: excused in up to one clean window in 16 (here 192 of 4033, then 256).
        (
            "a walk with spikes in under one window in 16",
            one(spiked(walk(18, 4096), 1500, 1e9)),
            64,
            Some(two(-22)),
            Some(two(2)),
        ),
        (
            "a walk with spikes in just over one window in 16",
            one(spiked(walk(18, 4096), 1000, 1e9)),
            64,
            Some(two(4)),
            Some(two(2)),
        ),
        (
            "a walk with a spike in every window",
            one(spiked(walk(19, 4096), 40, 1e9)),
            64,
            Some(two(4)),
            Some(two(2)),
        ),
        // Several series sharing one scale.
        (
            "two walks",
            vec![walk(21, 2048), walk(22, 1024)],
            64,
            Some(two(-22)),
            Some(two(-24)),
        ),
        (
            "a walk and one a million times larger",
            vec![walk(23, 2048), scaled(walk(24, 2048), 1e6)],
            64,
            Some(two(-3)),
            Some(two(-4)),
        ),
        (
            "cents and halves",
            vec![cents(25, 600), grid(26, 300, 1000, |k| k * 0.5)],
            16,
            Some(0.01),
            Some(0.01),
        ),
        (
            "cents and a walk",
            vec![cents(27, 600), walk(28, 600)],
            16,
            Some(two(-17)),
            Some(two(-18)),
        ),
        (
            "a walk and a series shorter than m",
            vec![walk(29, 600), walk(30, 10)],
            16,
            Some(two(-25)),
            Some(two(-27)),
        ),
        // No scale.
        ("shorter than m", one(walk(31, 15)), 16, None, None),
        ("no finite sample", one(vec![f64::NAN; 64]), 16, None, None),
    ]
}

#[test]
fn rule_one_picks_these_scales() {
    assert_eq!(SCALE_RULE, 1, "a new rule needs a new table below");
    assert_eq!(FAST_TICKS_NUMERATOR, (1 << 31) - 1);
    assert_eq!(SPAN_SLACK, 128);
    assert_eq!(OUTLIER_WINDOW_SHARE, 16);
    for (name, parts, m, fast, finest) in cases() {
        let parts: Vec<&[f64]> = parts.iter().map(Vec::as_slice).collect();
        let picked = |choice: ScaleChoice| choice.resolve_each(&parts, nz(m)).map(Scale::get);
        let bits = |scale: Option<f64>| scale.map(f64::to_bits);
        assert_eq!(
            bits(picked(ScaleChoice::Fast)),
            bits(fast),
            "{name}: the default rule picked {:?}, pinned {fast:?}",
            picked(ScaleChoice::Fast),
        );
        assert_eq!(
            bits(picked(ScaleChoice::Finest)),
            bits(finest),
            "{name}: the finest rule picked {:?}, pinned {finest:?}",
            picked(ScaleChoice::Finest),
        );
        if let [one] = parts[..] {
            assert_eq!(bits(Scale::infer(one, nz(m)).map(Scale::get)), bits(fast));
            assert_eq!(
                bits(Scale::infer_finest(one, nz(m)).map(Scale::get)),
                bits(finest)
            );
        }
    }
}
