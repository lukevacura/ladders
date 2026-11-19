#![cfg(feature = "oracle")]

//! A caller's flag for which windows are constant replaces the equal-ticks rule in every
//! z-normalized batch engine, equal on bits to the exact reference given the same flags, on tie
//! heavy series with gaps; flagging exactly the constant windows changes no bit, and a window
//! flagged constant is at distance zero from another and `√m` from any other.

use core::num::NonZeroUsize;

use ladde::batch::{
    ab_join, ab_join_topk, left_profile, left_topk_profile, two_sided_profile,
    two_sided_topk_profile,
};
use ladde::entry::Entry;
use ladde::naive::{
    Ticks, naive_ab_join, naive_ab_join_topk, naive_left_profile, naive_left_topk, naive_two_sided,
    naive_two_sided_topk,
};
use ladde::quantize::{Scale, to_ticks};
use ladde::stats::{ConstantFlagsLength, Series};

type Bits = (u64, u64, u64);

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn bits(entry: Entry) -> Bits {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

fn rows(profile: &[(u64, Entry)]) -> Vec<(u64, Bits)> {
    profile.iter().map(|&(row, e)| (row, bits(e))).collect()
}

fn topk_rows(profile: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<Bits>)> {
    profile
        .iter()
        .map(|(row, es)| (*row, es.iter().copied().map(bits).collect()))
        .collect()
}

fn ticks(series: &[f64]) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, unit())).collect()
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

/// Tie-heavy and gapped series: constant, constant runs between levels, small periods, few
/// levels, exact repeats, NaN runs at the start, middle and end.
fn series_set() -> Vec<Vec<f64>> {
    let mut seed = 7u64;
    let mut random = |n: usize, levels: u64| -> Vec<f64> {
        (0..n).map(|_| (lcg(&mut seed) % levels) as f64).collect()
    };
    let mut runs = random(40, 3);
    for k in [4..11, 18..26, 30..34] {
        let level = (k.start % 3) as f64;
        for i in k {
            runs[i] = level;
        }
    }
    let mut gapped = random(44, 4);
    for i in (0..3).chain(20..23).chain(41..44) {
        gapped[i] = f64::NAN;
    }
    let mut gapped_flat = runs.clone();
    gapped_flat[9] = f64::NAN;
    gapped_flat[27] = f64::NAN;
    let repeats: Vec<f64> = random(10, 50).into_iter().cycle().take(40).collect();
    vec![
        vec![5.0; 30],
        runs,
        gapped,
        gapped_flat,
        (0..36).map(|i| (i % 2) as f64).collect(),
        (0..36).map(|i| (i % 4) as f64).collect(),
        (0..42).map(|i| (i % 7) as f64).collect(),
        random(38, 2),
        repeats,
        random(45, 1000),
    ]
}

fn window_flags(series: &[f64], m: usize, f: impl Fn(&[f64]) -> bool) -> Vec<bool> {
    series.windows(m).map(f).collect()
}

fn is_flat(w: &[f64]) -> bool {
    w.iter().all(|x| *x == w[0])
}

/// The flag sets every series is run with: the default rule written out, nothing constant,
/// everything constant, a spread-out predicate that marks non-constant windows constant, and
/// the default rule inverted.
fn flag_sets(series: &[f64], m: usize) -> Vec<Vec<bool>> {
    let mut seed = (series.len() * 31 + m) as u64;
    let n = series.len() - m + 1;
    vec![
        window_flags(series, m, is_flat),
        vec![false; n],
        vec![true; n],
        (0..n).map(|_| lcg(&mut seed).is_multiple_of(3)).collect(),
        (0..n).map(|_| lcg(&mut seed).is_multiple_of(2)).collect(),
        window_flags(series, m, |w| !is_flat(w)),
        window_flags(series, m, |w| {
            let (lo, hi) = w
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            hi - lo <= 1.0
        }),
    ]
}

#[test]
fn every_batch_engine_equals_the_reference_under_any_flags() {
    for series in series_set() {
        let samples = ticks(&series);
        for m in [3usize, 4, 5, 8] {
            if series.len() < m + 3 {
                continue;
            }
            for flags in flag_sets(&series, m) {
                let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
                let reference = Ticks::with_constant_windows(&samples, &flags);
                for threads in [1, 3] {
                    assert_eq!(
                        rows(&left_profile(flagged, nz(m), unit())),
                        rows(&naive_left_profile(reference, m)),
                    );
                    for k in [1, 2, 5] {
                        assert_eq!(
                            topk_rows(&left_topk_profile(
                                flagged,
                                nz(m),
                                unit(),
                                nz(threads),
                                nz(k)
                            )),
                            topk_rows(&naive_left_topk(reference, m, nz(k))),
                        );
                    }
                    let sides = two_sided_profile(flagged, nz(m), unit(), nz(threads));
                    let expected = naive_two_sided(reference, m);
                    assert_eq!(sides.len(), expected.len());
                    for ((row, got), (_, want)) in sides.iter().zip(&expected) {
                        let row = *row;
                        assert_eq!(
                            [got.left, got.right, got.nearest].map(bits),
                            [want.left, want.right, want.nearest].map(bits),
                            "row {row}"
                        );
                    }
                    for k in [1, 3] {
                        let got =
                            two_sided_topk_profile(flagged, nz(m), unit(), nz(threads), nz(k))
                                .unwrap();
                        let want = naive_two_sided_topk(reference, m, nz(k));
                        for ((_, g), (_, w)) in got.iter().zip(&want) {
                            assert_eq!(bits(g.left), bits(w.left));
                            assert_eq!(bits(g.right), bits(w.right));
                            assert_eq!(
                                g.nearest.iter().copied().map(bits).collect::<Vec<_>>(),
                                w.nearest.iter().copied().map(bits).collect::<Vec<_>>(),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ab_join_takes_each_sides_flags() {
    let set = series_set();
    for (a, b) in set.iter().zip(set.iter().skip(1).chain(set.first())) {
        let m = 4;
        let (sa, sb) = (ticks(a), ticks(b));
        let a_flags = flag_sets(a, m);
        let b_flags = flag_sets(b, m);
        for (fa, fb) in a_flags.iter().zip(b_flags.iter().rev()) {
            let flagged_a = Series::with_constant_windows(a, nz(m), fa).unwrap();
            let flagged_b = Series::with_constant_windows(b, nz(m), fb).unwrap();
            let ref_a = Ticks::with_constant_windows(&sa, fa);
            let ref_b = Ticks::with_constant_windows(&sb, fb);
            for threads in [1, 3] {
                assert_eq!(
                    rows(&ab_join(
                        flagged_a,
                        flagged_b,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads)
                    )),
                    rows(&naive_ab_join(ref_a, ref_b, m)),
                );
                // One side flagged, the other the default rule.
                assert_eq!(
                    rows(&ab_join(
                        flagged_a,
                        b.as_slice(),
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads)
                    )),
                    rows(&naive_ab_join(ref_a, &sb, m)),
                );
                assert_eq!(
                    rows(&ab_join(
                        a.as_slice(),
                        flagged_b,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads)
                    )),
                    rows(&naive_ab_join(&sa, ref_b, m)),
                );
                for k in [1, 3] {
                    let got = ab_join_topk(
                        flagged_a,
                        flagged_b,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads),
                        nz(k),
                    )
                    .unwrap();
                    assert_eq!(
                        topk_rows(&got),
                        topk_rows(&naive_ab_join_topk(ref_a, ref_b, m, nz(k))),
                    );
                }
            }
        }
    }
}

#[test]
fn flagging_exactly_the_constant_windows_changes_no_bit() {
    for series in series_set() {
        for m in [3usize, 5] {
            if series.len() < m + 3 {
                continue;
            }
            let flags = window_flags(&series, m, is_flat);
            let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
            assert_eq!(
                rows(&left_profile(flagged, nz(m), unit())),
                rows(&left_profile(series.as_slice(), nz(m), unit())),
            );
            let with = two_sided_profile(flagged, nz(m), unit(), nz(2));
            let without = two_sided_profile(series.as_slice(), nz(m), unit(), nz(2));
            for ((_, w), (_, o)) in with.iter().zip(&without) {
                assert_eq!(
                    [w.left, w.right, w.nearest].map(bits),
                    [o.left, o.right, o.nearest].map(bits)
                );
            }
        }
    }
}

#[test]
fn flagged_windows_are_zero_apart_and_root_m_from_the_rest() {
    let series: Vec<f64> = (0..40).map(|i| ((i * 7 + i * i) % 11) as f64).collect();
    let m = 6;
    let n = series.len() - m + 1;
    let mut flags = vec![false; n];
    flags[1] = true;
    flags[20] = true;
    let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
    let sides = two_sided_profile(flagged, nz(m), unit(), nz(1));
    assert_eq!(sides[1].1.nearest.squared_distance(), 0.0);
    assert_eq!(sides[1].1.nearest.index().get(), Some(20));
    assert_eq!(sides[20].1.nearest.squared_distance(), 0.0);
    assert_eq!(sides[20].1.nearest.index().get(), Some(1));
    // With one flagged window and no other, the flagged one is at m from every other window.
    flags[20] = false;
    let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
    let sides = two_sided_profile(flagged, nz(m), unit(), nz(1));
    assert_eq!(sides[1].1.nearest.squared_distance(), m as f64);
    assert_eq!(
        sides[1].1.nearest.index().get(),
        Some(ladde::exclusion::zone(m) as u64 + 2)
    );
}

#[test]
fn an_unflagged_equal_window_is_uncorrelated_with_everything() {
    let mut series: Vec<f64> = (0..30).map(|i| ((i * 5 + i * i) % 9) as f64).collect();
    for x in &mut series[8..16] {
        *x = 4.0;
    }
    let m = 5;
    let flags = vec![false; series.len() - m + 1];
    let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
    let sides = two_sided_profile(flagged, nz(m), unit(), nz(1));
    // Window 9 is all 4s; every correlation with it is zero, so its distance is √(2m) unless
    // some other window is anti-correlated, which it cannot be: nothing is below zero.
    assert_eq!(sides[9].1.nearest.rho(), 0.0);
    assert_eq!(sides[9].1.nearest.squared_distance(), (2 * m) as f64);
}

#[test]
fn a_flag_on_a_window_with_a_gap_is_ignored() {
    let mut series: Vec<f64> = (0..24).map(|i| ((i * 3 + i * i) % 7) as f64).collect();
    series[10] = f64::NAN;
    let m = 4;
    let flags = vec![true; series.len() - m + 1];
    let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
    let sides = two_sided_profile(flagged, nz(m), unit(), nz(1));
    for (row, (_, side)) in sides.iter().enumerate().take(11).skip(7) {
        assert_eq!(bits(side.nearest), bits(Entry::NONE), "row {row}");
    }
    assert_eq!(sides[0].1.nearest.squared_distance(), 0.0);
}

#[test]
fn flags_of_the_wrong_length_are_refused() {
    let series = vec![1.0; 10];
    for found in [0usize, 6, 8, 9] {
        let flags = vec![false; found];
        assert_eq!(
            Series::with_constant_windows(&series, nz(4), &flags).err(),
            Some(ConstantFlagsLength { expected: 7, found })
        );
    }
    let flags = vec![true; 7];
    assert!(Series::with_constant_windows(&series, nz(4), &flags).is_ok());
    // A series shorter than the window has no windows, so only empty flags fit.
    assert!(Series::with_constant_windows(&series[..3], nz(4), &[]).is_ok());
    assert_eq!(
        Series::with_constant_windows(&series[..3], nz(4), &[true]).err(),
        Some(ConstantFlagsLength {
            expected: 0,
            found: 1
        })
    );
}

/// A random walk with steps up to a few billion ticks, so the whole series fits no narrow anchor
/// and tiles take the exact pass.
fn wide_walk(n: usize, seed: u64) -> Vec<f64> {
    let mut seed = seed;
    let mut level = 0i64;
    (0..n)
        .map(|_| {
            level += (lcg(&mut seed) % 5) as i64 * 300_000_000 - 600_000_000;
            level as f64
        })
        .collect()
}

#[test]
fn tiled_and_wide_passes_read_the_flags_like_the_rest() {
    use ladde::batch::{
        ab_join_report_tiled, ab_join_topk_tiled, left_profile_report_tiled,
        two_sided_profile_tiled,
    };
    for series in [wide_walk(150, 3), series_set().remove(1).repeat(4)] {
        let m = 4;
        let samples = ticks(&series);
        for flags in flag_sets(&series, m) {
            let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
            let reference = Ticks::with_constant_windows(&samples, &flags);
            let left = rows(&naive_left_profile(reference, m));
            let sides = naive_two_sided(reference, m);
            let join = rows(&naive_ab_join(reference, reference, m));
            let join_topk = topk_rows(&naive_ab_join_topk(reference, reference, m, nz(2)));
            for (band, chunk, tile) in [(1, 1, 1), (7, 3, 9), (40, 8, 16)] {
                for threads in [1, 4] {
                    let (got, report) = left_profile_report_tiled(
                        flagged,
                        nz(m),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                    );
                    assert_eq!(rows(&got), left);
                    assert!(report.tiles > 0);
                    if series.len() == 150 {
                        assert!(report.wide_tiles > 0, "the walk is too wide for any anchor");
                    }
                    let (got, _) = two_sided_profile_tiled(
                        flagged,
                        nz(m),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                    );
                    for ((_, g), (_, w)) in got.iter().zip(&sides) {
                        assert_eq!(
                            [g.left, g.right, g.nearest].map(bits),
                            [w.left, w.right, w.nearest].map(bits)
                        );
                    }
                    let (got, _) = ab_join_report_tiled(
                        flagged,
                        flagged,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                    );
                    assert_eq!(rows(&got), join);
                    let (topk, _) = ab_join_topk_tiled(
                        flagged,
                        flagged,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                        nz(2),
                    )
                    .unwrap();
                    assert_eq!(topk_rows(&topk), join_topk);
                }
            }
        }
    }
}
