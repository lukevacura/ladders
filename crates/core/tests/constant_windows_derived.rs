#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! A caller's flags for which windows are constant reach everything built on the batch profile:
//! the multi-dimensional profile, subspaces and description lengths, the pan profile, pattern
//! matching, and both streams (with their checkpoints), each equal to the exact reference given
//! the same flags; flags that say what the default rule says change no bit.

#[path = "common/exact_pair.rs"]
mod exact_pair;
#[path = "common/multi.rs"]
mod multi;

use ladde::batch::multi::mstump;
use ladde::batch::{discords, left_profile, two_sided_profile};
use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::mdl::mdl;
use ladde::multidim::NonFinitePolicy;
use ladde::naive::{
    Ticks, naive_left_profile, naive_mdl, naive_multi_profile, naive_subspace, naive_two_sided,
};
use ladde::pan::PanProfile;
use ladde::pattern::{MaxDistance, distance_profile, find_matches};
use ladde::quantize::{Scale, to_ticks};
use ladde::stats::Series;
use ladde::stream::GrowingStream;
use ladde::subspace::{Bits, Options, subspace};
use multi::{bits, nz, ticks, unit};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

/// Flags for every window of `series`: the default rule, none constant, all constant, a window
/// whose samples stay within a band, pseudo-random ones.
fn flags_of(series: &[f64], m: usize, which: usize, seed: &mut u64) -> Vec<bool> {
    (0..(series.len() + 1).saturating_sub(m))
        .map(|i| {
            let w = &series[i..i + m];
            match which % 5 {
                0 => w.iter().all(|x| *x == w[0]),
                1 => false,
                2 => true,
                3 => {
                    let (lo, hi) = w
                        .iter()
                        .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                    hi - lo <= 3.0
                }
                _ => lcg(seed).is_multiple_of(3),
            }
        })
        .collect()
}

#[test]
fn the_multi_dimensional_profile_reads_each_dimensions_flags() {
    let mut seed = 5u64;
    for input in multi::tie_inputs() {
        let m = input.m;
        let flags: Vec<Vec<Vec<bool>>> = (0..5)
            .map(|which| {
                input
                    .dimensions
                    .iter()
                    .enumerate()
                    .map(|(a, series)| flags_of(series, m, which + a, &mut seed))
                    .collect()
            })
            .collect();
        for flags in flags {
            let samples: Vec<_> = input.dimensions.iter().map(|s| ticks(s, unit())).collect();
            let reference: Vec<Ticks> = samples
                .iter()
                .zip(&flags)
                .map(|(t, f)| Ticks::with_constant_windows(t, f))
                .collect();
            let expected = naive_multi_profile(
                &reference,
                m,
                &input.include,
                input.discords,
                NonFinitePolicy::Missing,
            )
            .unwrap();
            let flagged: Vec<Series> = input
                .dimensions
                .iter()
                .zip(&flags)
                .map(|(s, f)| Series::with_constant_windows(s, nz(m), f).unwrap())
                .collect();
            let scales = vec![unit(); flagged.len()];
            for threads in [1, 3] {
                let got = mstump(
                    &flagged,
                    nz(m),
                    &scales,
                    &input.include,
                    input.discords,
                    nz(threads),
                )
                .unwrap();
                assert_eq!(bits(&got), bits(&expected), "{input:?}");
            }
        }
    }
}

#[test]
fn subspaces_and_description_lengths_read_each_dimensions_flags() {
    let mut seed = 9u64;
    for input in multi::tie_inputs().into_iter().step_by(3) {
        let m = input.m;
        let n = input.dimensions[0].len();
        for which in 0..5 {
            let flags: Vec<Vec<bool>> = input
                .dimensions
                .iter()
                .enumerate()
                .map(|(a, s)| flags_of(s, m, which + a, &mut seed))
                .collect();
            let set: Vec<(Series, Scale)> = input
                .dimensions
                .iter()
                .zip(&flags)
                .map(|(s, f)| (Series::with_constant_windows(s, nz(m), f).unwrap(), unit()))
                .collect();
            let options = Options {
                include: &input.include,
                discords: input.discords,
                bits: Bits::EIGHT,
                non_finite: NonFinitePolicy::Missing,
            };
            let dimensions = set.len();
            for pair in [(0, n - m), (2, 9), (4, 4 + m + 1)] {
                for k in 0..dimensions {
                    let got = subspace(&set, nz(m), pair, k, &options);
                    let want = naive_subspace(
                        &set,
                        nz(m),
                        pair,
                        k,
                        &input.include,
                        input.discords,
                        Bits::EIGHT,
                    );
                    assert_eq!(got, want, "{pair:?} k {k} flags {which}");
                }
            }
            let firsts: Vec<usize> = (0..dimensions).map(|a| (a * 3 + 1) % (n - m)).collect();
            let seconds: Vec<usize> = (0..dimensions).map(|a| (a * 5 + 4) % (n - m)).collect();
            let got = mdl(&set, nz(m), &firsts, &seconds, &options).unwrap();
            let (sizes, subspaces) = naive_mdl(
                &set,
                nz(m),
                &firsts,
                &seconds,
                &input.include,
                input.discords,
                Bits::EIGHT,
            )
            .unwrap();
            assert_eq!(
                got.bit_sizes
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                sizes.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
            assert_eq!(got.subspaces, subspaces);
        }
    }
}

#[test]
fn the_pan_profile_reads_flags_for_each_window_size() {
    let mut seed = 21u64;
    let series: Vec<f64> = (0..48)
        .map(|i| {
            if (10..22).contains(&i) {
                2.0
            } else {
                (lcg(&mut seed) % 4) as f64
            }
        })
        .collect();
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, unit())).collect();
    for which in 0..5 {
        let mut pan = PanProfile::new(&series, 4, Some(9), 1, Some(unit()), nz(2)).unwrap();
        let mut sizes = Vec::new();
        while let Some(m) = pan.next_window_size() {
            let flags = flags_of(&series, m, which, &mut seed);
            assert!(pan.update_with(Some(&flags)).unwrap());
            sizes.push((m, flags));
        }
        assert!(!pan.update_with(Some(&[])).unwrap());
        for (row, (m, flags)) in pan.rows().iter().zip(&sizes) {
            assert_eq!(row.window, *m);
            let want = naive_two_sided(Ticks::with_constant_windows(&samples, flags), *m);
            let got: Vec<(u64, u64)> = row
                .distances
                .iter()
                .zip(&row.indices)
                .map(|(d, i)| (d.to_bits(), i.get().unwrap_or(u64::MAX)))
                .collect();
            let expected: Vec<(u64, u64)> = want
                .iter()
                .map(|(_, sides)| {
                    (
                        libm::sqrt(sides.nearest.squared_distance()).to_bits(),
                        sides.nearest.index().get().unwrap_or(u64::MAX),
                    )
                })
                .collect();
            assert_eq!(got, expected, "m {m} flags {which}");
        }
    }
    let mut pan = PanProfile::new(&series, 4, Some(5), 1, Some(unit()), nz(1)).unwrap();
    assert!(pan.update_with(Some(&[true])).is_err());
}

#[test]
fn pattern_matching_reads_the_series_and_the_query_flag() {
    let mut seed = 33u64;
    let series: Vec<f64> = (0..60)
        .map(|i| {
            if (20..30).contains(&i) {
                1.0
            } else {
                (lcg(&mut seed) % 5) as f64
            }
        })
        .collect();
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, unit())).collect();
    let m = 6;
    let query: Vec<f64> = vec![1.0, 4.0, 2.0, 2.0, 3.0, 0.0];
    for which in 0..5 {
        let flags = flags_of(&series, m, which, &mut seed);
        let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
        for query_flag in [false, true] {
            let query_flags = [query_flag];
            let q = Series::with_constant_windows(&query, nz(m), &query_flags).unwrap();
            let got = distance_profile(q, flagged, unit()).unwrap();
            let query_ticks: Vec<Option<i64>> =
                query.iter().map(|&x| to_ticks(x, unit())).collect();
            for (i, entry) in got.iter().enumerate() {
                let want = exact_pair::cross_pair(
                    Ticks::with_constant_windows(&samples, &flags),
                    Ticks::with_constant_windows(&query_ticks, &query_flags),
                    m,
                    i,
                    0,
                );
                match want {
                    Some((_, d2)) => assert_eq!(entry.squared_distance().to_bits(), d2.to_bits()),
                    None => assert_eq!(entry.index().get(), None),
                }
            }
            let matches =
                find_matches(q, flagged, unit(), MaxDistance::Value(1.5), Some(5)).unwrap();
            for (start, entry) in &matches {
                assert_eq!(
                    entry.squared_distance().to_bits(),
                    got[*start as usize].squared_distance().to_bits()
                );
            }
        }
    }
}

fn entry_bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

#[test]
fn both_streams_read_the_flag_of_each_window_and_checkpoint_it() {
    let mut seed = 77u64;
    let mut series: Vec<f64> = (0..70).map(|_| (lcg(&mut seed) % 3) as f64).collect();
    series[15..27].fill(2.0);
    series[40] = f64::NAN;
    let m = 5;
    for which in 0..5 {
        let flags = flags_of(&series, m, which, &mut seed);
        let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
        let want: Vec<(u64, u64, u64)> = left_profile(flagged, nz(m), unit())
            .iter()
            .map(|&(_, e)| entry_bits(e))
            .collect();
        let flag_of = |k: usize| (k + 1 >= m).then(|| flags[k + 1 - m]);

        let mut growing = GrowingStream::new(nz(m), unit());
        let mut bounded = BoundedStream::new(nz(m), nz(series.len()), unit()).unwrap();
        let (mut from_growing, mut from_bounded) = (Vec::new(), Vec::new());
        for (k, &x) in series.iter().enumerate() {
            if k == 33 {
                growing = GrowingStream::restore(&growing.checkpoint()).unwrap();
                bounded = BoundedStream::restore(&bounded.checkpoint()).unwrap();
            }
            if let Some((_, entry)) = growing.update_flagged(x, flag_of(k)) {
                from_growing.push(entry_bits(entry));
            }
            if let Some((_, entry)) = bounded.update_flagged(x, flag_of(k)) {
                from_bounded.push(entry_bits(entry));
            }
        }
        assert_eq!(from_growing, want, "growing, flags {which}");
        assert_eq!(from_bounded, want, "bounded, flags {which}");
    }
}

#[test]
fn the_default_flags_change_no_bit_in_discords_or_the_profiles() {
    let mut seed = 3u64;
    let mut series: Vec<f64> = (0..50).map(|_| (lcg(&mut seed) % 4) as f64).collect();
    series[8..20].fill(1.0);
    let m = 5;
    let flags = flags_of(&series, m, 0, &mut seed);
    let flagged = Series::with_constant_windows(&series, nz(m), &flags).unwrap();
    let plain: Vec<_> = discords(series.as_slice(), nz(m), unit(), nz(2), 3);
    let with: Vec<_> = discords(flagged, nz(m), unit(), nz(2), 3);
    assert_eq!(plain.len(), with.len());
    for (a, b) in plain.iter().zip(&with) {
        assert_eq!(a.index(), b.index());
        assert_eq!(
            a.squared_distance().to_bits(),
            b.squared_distance().to_bits()
        );
    }
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, unit())).collect();
    assert_eq!(
        left_profile(flagged, nz(m), unit())
            .iter()
            .map(|&(_, e)| entry_bits(e))
            .collect::<Vec<_>>(),
        naive_left_profile(&samples, m)
            .iter()
            .map(|&(_, e)| entry_bits(e))
            .collect::<Vec<_>>(),
    );
    let _ = two_sided_profile(flagged, nz(m), unit(), nz(1));
}
