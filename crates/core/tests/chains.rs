#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Chains equal a line-by-line port of STUMPY's `atsc` and `allc` on any left and right indices
//! that point their own way, and the series entry point equals them over the exact two-sided
//! profile's indices.

use core::num::NonZeroUsize;
use std::collections::{BTreeSet, VecDeque};

use ladde::chain::{AllChains, ChainInputError, all_chains, anchored_chain, chains_from_series};
use ladde::index::SampleIndex;
use ladde::naive::naive_two_sided;
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn indices(raw: &[i64]) -> Vec<SampleIndex> {
    raw.iter()
        .map(|&i| {
            u64::try_from(i)
                .ok()
                .and_then(SampleIndex::new)
                .unwrap_or(SampleIndex::NONE)
        })
        .collect()
}

fn stumpy_atsc(il: &[i64], ir: &[i64], mut j: usize) -> Vec<usize> {
    let mut chain = VecDeque::from([j]);
    for _ in 0..il.len() {
        if ir[j] == -1 || il[ir[j] as usize] != j as i64 {
            break;
        }
        j = ir[j] as usize;
        chain.push_back(j);
    }
    chain.into_iter().collect()
}

fn argmax_first(l: &[i64]) -> usize {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best
}

/// STUMPY's `allc`, with its set of tuples returned sorted by first window.
fn stumpy_allc(il: &[i64], ir: &[i64]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut l = vec![1i64; il.len()];
    let mut set = BTreeSet::new();
    for i in 0..il.len() {
        if l[i] == 1 {
            let mut j = i;
            let mut chain = vec![j];
            for _ in 0..il.len() {
                if ir[j] == -1 || il[ir[j] as usize] != j as i64 {
                    break;
                }
                j = ir[j] as usize;
                l[j] = -1;
                l[i] += 1;
                chain.push(j);
            }
            set.insert(chain);
        }
    }
    let longest = if il.is_empty() {
        vec![]
    } else {
        stumpy_atsc(il, ir, argmax_first(&l))
    };
    (set.into_iter().collect(), longest)
}

/// Forward-pointing indices where most right neighbors are linked back, so chains are long.
fn forward_indices(n: usize, picks: &[(u32, u32, u8)]) -> (Vec<i64>, Vec<i64>) {
    let mut il = vec![-1i64; n];
    let mut ir = vec![-1i64; n];
    for i in 0..n {
        let (a, b, kind) = picks[i];
        if i + 1 < n && kind % 8 != 0 {
            ir[i] = (i + 1 + a as usize % (n - i - 1).min(3)) as i64;
        }
        if i > 0 && kind % 5 == 0 {
            il[i] = (b as usize % i) as i64;
        }
    }
    for i in 0..n {
        let t = ir[i];
        if t >= 0 && !picks[i].2.is_multiple_of(3) && il[t as usize] == -1 {
            il[t as usize] = i as i64;
        }
    }
    (il, ir)
}

fn chain_set(found: &AllChains) -> Vec<Vec<usize>> {
    let mut chains = found.chains.clone();
    chains.sort();
    chains
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn chains_equal_stumpys_on_forward_indices(
        n in 0usize..40,
        picks in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u8>()), 40),
    ) {
        let (il, ir) = forward_indices(n, &picks);
        let (left, right) = (indices(&il), indices(&ir));
        let (expected_set, expected_longest) = stumpy_allc(&il, &ir);
        let found = all_chains(&left, &right).unwrap();
        prop_assert_eq!(&found.chains, &expected_set);
        prop_assert_eq!(&found.longest, &expected_longest);
        for anchor in 0..n {
            prop_assert_eq!(anchored_chain(&left, &right, anchor).unwrap(), stumpy_atsc(&il, &ir, anchor));
        }
        let mut seen = vec![0u8; n];
        for chain in &found.chains {
            for &w in chain {
                seen[w] += 1;
            }
            for pair in chain.windows(2) {
                prop_assert_eq!(ir[pair[0]], pair[1] as i64);
                prop_assert_eq!(il[pair[1]], pair[0] as i64);
            }
        }
        prop_assert!(seen.iter().all(|&c| c == 1));
    }
}

#[test]
fn the_stumpy_documentation_example() {
    let il = [-1, -1, 0, 1, 0];
    let ir = [4, 3, 4, -1, -1];
    let (left, right) = (indices(&il), indices(&ir));
    let found = all_chains(&left, &right).unwrap();
    assert_eq!(found.chains, vec![vec![0, 4], vec![1, 3], vec![2]]);
    assert_eq!(found.longest, vec![0, 4]);
    assert_eq!(anchored_chain(&left, &right, 1).unwrap(), vec![1, 3]);
    assert_eq!(anchored_chain(&left, &right, 2).unwrap(), vec![2]);
}

#[test]
fn equal_longest_chains_resolve_to_the_smaller_first_window() {
    let (left, right) = (indices(&[-1, -1, 0, 1]), indices(&[2, 3, -1, -1]));
    assert_eq!(all_chains(&left, &right).unwrap().longest, vec![0, 2]);
}

#[test]
fn a_link_needs_the_neighbor_to_point_back() {
    let (left, right) = (indices(&[-1, -1, 0]), indices(&[2, 2, -1]));
    let found = all_chains(&left, &right).unwrap();
    assert_eq!(found.chains, vec![vec![0, 2], vec![1]]);
}

#[test]
fn no_windows_have_no_chains() {
    let found = all_chains(&[], &[]).unwrap();
    assert!(found.chains.is_empty() && found.longest.is_empty());
    assert_eq!(
        anchored_chain(&[], &[], 0),
        Err(ChainInputError::OutOfRange)
    );
}

#[test]
fn indices_that_cannot_belong_to_one_series_are_refused() {
    let none = SampleIndex::NONE;
    let at = |i| SampleIndex::new(i).unwrap();
    assert_eq!(
        all_chains(&[none], &[none, none]),
        Err(ChainInputError::LengthMismatch)
    );
    assert_eq!(
        all_chains(&[none, none], &[at(2), none]),
        Err(ChainInputError::OutOfRange)
    );
    assert_eq!(
        all_chains(&[none, none], &[none, at(7)]),
        Err(ChainInputError::OutOfRange)
    );
    assert_eq!(
        all_chains(&[none, none], &[at(0), none]),
        Err(ChainInputError::NotForward)
    );
    assert_eq!(
        all_chains(&[none, at(1)], &[none, none]),
        Err(ChainInputError::NotForward)
    );
    assert_eq!(
        anchored_chain(&[none], &[none], 1),
        Err(ChainInputError::OutOfRange)
    );
}

#[test]
fn each_refusal_says_what_is_wrong() {
    let messages: BTreeSet<String> = [
        ChainInputError::LengthMismatch,
        ChainInputError::OutOfRange,
        ChainInputError::NotForward,
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    assert_eq!(messages.len(), 3);
    assert!(messages.iter().all(|message| message.len() > 10));
}

fn assert_series_chains(series: &[f64], m: usize, scale: Scale) {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let naive = naive_two_sided(&samples, m);
    let left: Vec<SampleIndex> = naive.iter().map(|(_, s)| s.left.index()).collect();
    let right: Vec<SampleIndex> = naive.iter().map(|(_, s)| s.right.index()).collect();
    let expected = all_chains(&left, &right).unwrap();
    let il: Vec<i64> = left
        .iter()
        .map(|i| i.get().map_or(-1, |i| i as i64))
        .collect();
    let ir: Vec<i64> = right
        .iter()
        .map(|i| i.get().map_or(-1, |i| i as i64))
        .collect();
    assert_eq!(
        (chain_set(&expected), expected.longest.clone()),
        stumpy_allc(&il, &ir)
    );
    for threads in [1, 3] {
        let found = chains_from_series(series, nz(m), scale, nz(threads));
        assert_eq!(found, expected, "threads {threads}");
    }
}

#[test]
fn series_chains_equal_the_reference_on_tie_heavy_and_gappy_series() {
    let unit = Scale::new(1.0).unwrap();
    let constant = vec![2.0; 50];
    assert_series_chains(&constant, 4, unit);
    for period in [2usize, 4, 7] {
        let repeating: Vec<f64> = (0..90).map(|k| ((k % period) * 3 % 5) as f64).collect();
        assert_series_chains(&repeating, 5, unit);
    }
    let mut state = 17u64;
    let few_levels: Vec<f64> = (0..120)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 60) % 3) as f64
        })
        .collect();
    assert_series_chains(&few_levels, 4, unit);
    let mut gappy: Vec<f64> = (0..100).map(|k| ((k * 7) % 11) as f64).collect();
    gappy[..3].fill(f64::NAN);
    gappy[40] = f64::NAN;
    gappy[96..].fill(f64::NAN);
    gappy[10..25].fill(4.0);
    assert_series_chains(&gappy, 5, unit);
}

#[test]
fn a_walk_has_chains_and_the_stumpy_example_series_matches() {
    let example = [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0];
    let found = chains_from_series(&example, nz(3), Scale::new(1e-6).unwrap(), nz(1));
    assert_eq!(found.chains, vec![vec![0, 4], vec![1, 3], vec![2]]);
    assert_eq!(found.longest, vec![0, 4]);
}
