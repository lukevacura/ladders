//! A plain statement of STUMPY's snippets after the segment profiles: the greedy pick over areas
//! summed pairwise as NumPy does, the fractions and the regimes.
#![allow(dead_code)]

use ladde::snippets::Snippets;

/// NumPy's pairwise sum, restated.
fn pairwise(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |a, &x| a + x);
    }
    if n <= 128 {
        let mut r: [f64; 8] = values[..8].try_into().unwrap();
        let mut i = 8;
        while i + 8 <= n {
            for (lane, &x) in r.iter_mut().zip(&values[i..i + 8]) {
                *lane += x;
            }
            i += 8;
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &values[i..] {
            sum += x;
        }
        return sum;
    }
    let half = n / 2 - (n / 2) % 8;
    pairwise(&values[..half]) + pairwise(&values[half..])
}

/// The `k` snippets of `t` (segments of `m`) from the segments' `profiles`, one per segment.
pub fn assemble(t: &[f64], m: usize, k: usize, profiles: &[Vec<f64>]) -> Snippets {
    let len = t.len() - m + 1;
    let mut running = vec![f64::INFINITY; len];
    let (mut chosen, mut areas) = (Vec::new(), Vec::new());
    for _ in 0..k {
        let lowered = |profile: &[f64], running: &[f64]| -> Vec<f64> {
            profile
                .iter()
                .zip(running)
                .map(|(&x, &q)| x.min(q))
                .collect()
        };
        let all: Vec<f64> = profiles
            .iter()
            .map(|p| pairwise(&lowered(p, &running)))
            .collect();
        let mut at = 0;
        for (i, &area) in all.iter().enumerate() {
            if area < all[at] {
                at = i;
            }
        }
        running = lowered(&profiles[at], &running);
        chosen.push(at);
        areas.push(all[at]);
    }
    let mut remaining: Vec<f64> = (0..len)
        .map(|p| {
            chosen
                .iter()
                .map(|&c| profiles[c][p])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let (mut fractions, mut regimes) = (Vec::new(), Vec::new());
    for (i, &c) in chosen.iter().enumerate() {
        let mask: Vec<bool> = (0..len).map(|p| profiles[c][p] <= remaining[p]).collect();
        fractions.push(mask.iter().filter(|&&x| x).count() as f64 / len as f64);
        for p in 0..len {
            if mask[p] {
                remaining[p] -= 1.0;
            }
        }
        let mut p = 0;
        while p < len {
            if mask[p] {
                let start = p;
                while p < len && mask[p] {
                    p += 1;
                }
                regimes.push([i, start, p]);
            } else {
                p += 1;
            }
        }
    }
    Snippets {
        m,
        profile_len: len,
        snippets: chosen
            .iter()
            .flat_map(|&c| t[c * m..(c + 1) * m].to_vec())
            .collect(),
        indices: chosen.iter().map(|&c| c * m).collect(),
        profiles: chosen.iter().flat_map(|&c| profiles[c].clone()).collect(),
        fractions,
        areas,
        regimes,
    }
}
