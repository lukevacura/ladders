use super::*;
use alloc::{format, vec};
use proptest::prelude::*;

fn entry_bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn quantized(series: &[f64], m: usize) -> Quantized {
    Quantized::build(
        series,
        NonZeroUsize::new(m).unwrap(),
        Scale::new(1.0).unwrap(),
    )
}

/// `row`'s entry for `neighbor` alone, as a side reports it.
fn side_entry(q: &Quantized, m: usize, row: usize, neighbor: usize) -> Entry {
    let mut nearest = Nearest::new(&q.windows[row], m);
    let index = SampleIndex::new(neighbor as u64).unwrap();
    offer_entry(
        q,
        m,
        row,
        &mut nearest,
        Entry::new(0.0, 0.0, index).unwrap(),
    );
    nearest.finish()
}

/// The nearer of the two by one exact reduction over both, ties to the smaller index.
fn reference(q: &Quantized, m: usize, row: usize, left: Entry, right: Entry) -> Entry {
    let mut nearest = Nearest::new(&q.windows[row], m);
    for entry in [left, right] {
        offer_entry(q, m, row, &mut nearest, entry);
    }
    nearest.finish()
}

/// `entry` with its estimate replaced by `rho`.
fn with_rho(entry: Entry, rho: f64) -> Entry {
    Entry::new(rho, entry.squared_distance(), entry.index()).unwrap()
}

/// Checks every row's choice over left and right neighbor pairs of `series` against the exact
/// reduction: with the true estimates, with equal estimates (the exact order decides), and with
/// estimates set clearly apart (the estimate decides). Returns how many pairs tied exactly.
fn check_series(series: &[f64], m: usize) -> usize {
    let q = quantized(series, m);
    let past = crate::exclusion::zone(m) + 1;
    let count = q.windows.len();
    let mut ties = 0;
    for row in 0..count {
        for l in (0..(row + 1).saturating_sub(past)).step_by(3) {
            for r in (row + past..count).step_by(3) {
                let (left, right) = (side_entry(&q, m, row, l), side_entry(&q, m, row, r));
                if left.index().get().is_none() || right.index().get().is_none() {
                    continue;
                }
                let expected = reference(&q, m, row, left, right);
                let got = nearest_of(&q, None, m, row, left, right);
                assert_eq!(entry_bits(got), entry_bits(expected), "row {row}: {l}, {r}");
                if entry_bits(left).0 == entry_bits(right).0
                    && entry_bits(left).1 == entry_bits(right).1
                {
                    ties += 1;
                    assert_eq!(got.index(), left.index(), "row {row}: a tie, {l}, {r}");
                }
                let (same_l, same_r) = (with_rho(left, 0.25), with_rho(right, 0.25));
                let got = nearest_of(&q, None, m, row, same_l, same_r);
                assert_eq!(got.index(), expected.index(), "row {row}: equal, {l}, {r}");
                let (high_l, low_r) = (with_rho(left, 0.9), with_rho(right, -0.9));
                let got = nearest_of(&q, None, m, row, high_l, low_r);
                assert_eq!(got.index(), left.index(), "row {row}: left clear, {l}, {r}");
                let (low_l, high_r) = (with_rho(left, -0.9), with_rho(right, 0.9));
                let got = nearest_of(&q, None, m, row, low_l, high_r);
                assert_eq!(
                    got.index(),
                    right.index(),
                    "row {row}: right clear, {l}, {r}"
                );
                let exact = nearer_side(&q, m, row, left, right).unwrap();
                assert_eq!(entry_bits(exact.entry()), entry_bits(expected));
            }
        }
    }
    ties
}

#[test]
fn exact_ties_go_to_the_left_side() {
    let series: Vec<f64> = (0..60)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    assert!(check_series(&series, 4) > 0);
}

#[test]
fn constant_runs_and_scaled_copies_choose_as_the_exact_order_does() {
    let mut series: Vec<f64> = (0..70).map(|k| f64::from((k * 7 % 11) as u8)).collect();
    series[10..25].fill(5.0);
    series[40..52].fill(2.0);
    for k in 0..8 {
        series[55 + k] = 2.0 * series[k];
    }
    assert!(check_series(&series, 5) > 0);
}

#[test]
fn an_empty_side_yields_the_other() {
    let series: Vec<f64> = (0..40).map(|k| f64::from((k * 5 % 7) as u8)).collect();
    let m = 4;
    let q = quantized(&series, m);
    let (row, l, r) = (15, 2, 30);
    let (left, right) = (side_entry(&q, m, row, l), side_entry(&q, m, row, r));
    let none = Entry::NONE;
    let pick = |a, b| entry_bits(nearest_of(&q, None, m, row, a, b));
    assert_eq!(pick(left, none), entry_bits(left));
    assert_eq!(pick(none, right), entry_bits(right));
    assert_eq!(pick(none, none), entry_bits(none));
    assert!(nearer_side(&q, m, row, none, none).is_none());
    let one = nearer_side(&q, m, row, none, right).unwrap();
    assert_eq!(entry_bits(one.entry()), entry_bits(right));
}

proptest! {
    #![proptest_config(testkit::generators::seeded(16))]

    #[test]
    fn few_level_series_choose_as_the_exact_order_does(
        levels in prop::collection::vec(0u8..3, 24..48),
        m in 3usize..7,
    ) {
        let series: Vec<f64> = levels.iter().map(|&x| f64::from(x)).collect();
        check_series(&series, m);
    }
}

// ---------- the serial tail on threads ----------

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// Series that stress the tail: exact repeats, flat stretches, NaN gaps at the start, in the
/// middle (across several spans) and at the end, and a tick-valued walk.
fn tail_series(n: usize) -> Vec<(&'static str, Vec<f64>)> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let walk: Vec<f64> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from((state >> 59) as u8 % 5)
        })
        .collect();
    let period: Vec<f64> = (0..n)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5, 9][k % 7]))
        .collect();
    let mut flat = walk.clone();
    for start in (0..n).step_by(700) {
        flat[start..(start + 450).min(n)].fill(2.0);
    }
    let mut gaps = walk.clone();
    gaps[..n / 5].fill(f64::NAN);
    gaps[(n / 2).saturating_sub(300)..(n / 2 + 900).min(n)].fill(f64::NAN);
    for start in (1000..n).step_by(1021) {
        gaps[start..(start + 5).min(n)].fill(f64::NAN);
    }
    gaps[n.saturating_sub(40)..].fill(f64::NAN);
    vec![
        ("constant", vec![7.0; n]),
        ("period", period),
        ("flat", flat),
        ("gaps", gaps),
        ("all missing", vec![f64::NAN; n]),
        ("walk", walk),
    ]
}

fn assert_same_quantized(label: &str, a: &Quantized, b: &Quantized) {
    assert_eq!(a.ticks, b.ticks, "{label}: ticks");
    assert_eq!(a.sum, b.sum, "{label}: sums");
    assert_eq!(a.variance, b.variance, "{label}: variances");
    assert_eq!(a.windows, b.windows, "{label}: windows");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.r), bits(&b.r), "{label}: r");
    assert_eq!(bits(&a.bias), bits(&b.bias), "{label}: bias");
    assert_eq!(a.range, b.range, "{label}: cross-term bound");
}

#[test]
fn quantizing_on_threads_is_quantizing_on_one() {
    let scale = Scale::new(1.0).unwrap();
    for m in [3usize, 64, 700] {
        for (name, series) in tail_series(9000) {
            let one = Quantized::build(&series, nz(m), scale);
            for threads in [2, 3, 8, 64] {
                let many = Quantized::build_on(&series, nz(m), scale, nz(threads), Engine::Ungated);
                assert_same_quantized(&format!("{name} m{m} t{threads}"), &one, &many);
            }
        }
    }
}

#[test]
fn quantizing_a_short_series_on_threads_is_quantizing_on_one() {
    let scale = Scale::new(1.0).unwrap();
    for (n, m) in [
        (0usize, 3usize),
        (2, 3),
        (3, 3),
        (4, 3),
        (2049, 3),
        (2050, 3),
        (4098, 2049),
    ] {
        for (name, series) in tail_series(n) {
            let one = Quantized::build(&series, nz(m), scale);
            let many = Quantized::build_on(&series, nz(m), scale, nz(8), Engine::Ungated);
            assert_same_quantized(&format!("{name} n{n} m{m}"), &one, &many);
        }
    }
}

#[test]
fn quantizing_with_constant_flags_on_threads_is_quantizing_on_one() {
    let (n, m) = (9000, nz(5));
    let scale = Scale::new(1.0).unwrap();
    let flags: Vec<bool> = (0..n + 1 - m.get())
        .map(|w| w % 3 == 0 || w % 1001 < 4)
        .collect();
    for (name, series) in tail_series(n) {
        let series = Series::with_constant_windows(&series, m, &flags).unwrap();
        let one = Quantized::build(series, m, scale);
        let many = Quantized::build_on(series, m, scale, nz(8), Engine::Ungated);
        assert_same_quantized(name, &one, &many);
    }
}

#[test]
fn the_right_side_finishes_the_same_on_any_thread_count() {
    let m = nz(8);
    let scale = Scale::new(1.0).unwrap();
    for (name, series) in tail_series(3700) {
        let run = |threads: usize| {
            let (profile, _) =
                two_sided_profile_tiled(&series, m, scale, nz(threads), nz(64), nz(64), nz(256));
            profile
                .iter()
                .map(|(row, t)| {
                    (
                        *row,
                        entry_bits(t.left),
                        entry_bits(t.right),
                        entry_bits(t.nearest),
                    )
                })
                .collect::<Vec<_>>()
        };
        let one = run(1);
        for threads in [2, 3, 8] {
            assert_eq!(run(threads), one, "{name}: {threads} threads");
        }
    }
}

#[test]
fn the_top_k_right_side_finishes_the_same_on_any_thread_count() {
    let m = nz(8);
    let scale = Scale::new(1.0).unwrap();
    for (name, series) in tail_series(3700) {
        let run = |threads: usize| {
            let (profile, _) = two_sided_topk_profile_tiled(
                &series,
                m,
                scale,
                nz(threads),
                nz(64),
                nz(64),
                nz(256),
                nz(3),
            )
            .unwrap();
            format!("{profile:?}")
        };
        let one = run(1);
        for threads in [2, 3, 8] {
            assert_eq!(run(threads), one, "{name}: {threads} threads");
        }
    }
}
