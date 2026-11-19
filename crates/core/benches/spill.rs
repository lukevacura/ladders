//! `spill` bench: the work behind the cost model's constants, one workload per process so an
//! outer instruction counter (`/usr/bin/time -l` reports instructions retired) can read each one:
//! `spill <mode> <n> <m> <reps>`, with `mode` one of `narrow` (today's left profile on a series
//! that fits one anchor), `wide` (the same on a series whose tiles all run the exact i64 pass; either
//! takes a `-two-sided` or `-join` suffix for that profile),
//! `cold` (the cold route over every window of a series in the 128-bit tier) and `cold256` (the
//! same in the 256-bit tier); `spill messy <left|two-sided|join> <n> <m> <prints> <reps>` runs that
//! profile of a cent walk with that many bad prints (a thousand times the price). It prints what
//! the run walked. With no arguments it runs each mode once at a small size.

use ladde::batch::{ab_join_report, left_profile_report, two_sided_profile};
use ladde::entry::Entry;
use ladde::quantize::{Scale, to_ticks};
use ladde::spill::{Survey, Width};
use std::hint::black_box;
use std::num::NonZeroUsize;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// An integer random walk, with `step` added every `every` samples and `spike` at the middle.
fn series(len: usize, step: f64, every: usize, spike: f64) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut value = 10_000.0;
    (0..len)
        .map(|i| {
            value += (splitmix64(&mut state) % 5) as f64 - 2.0;
            if i > 0 && i % every == 0 {
                value += step;
            }
            if i == len / 2 { value + spike } else { value }
        })
        .collect()
}

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).expect("nonzero")
}

/// The hot passes alone over `series`, as the left profile, the two-sided profile or a join with
/// a series half as long: tiles and tiles on the exact i64 pass.
fn hot(series: &[f64], other: &[f64], m: usize, family: &str) -> (usize, usize) {
    let scale = Scale::new(1.0).expect("a scale");
    match family {
        "two-sided" => {
            black_box(two_sided_profile(series, nz(m), scale, nz(1)));
            (0, 0)
        }
        "join" => {
            let (profile, report) = ab_join_report(series, other, nz(m), scale, scale, nz(1));
            black_box(profile);
            (report.tiles, report.wide_tiles)
        }
        _ => {
            let (profile, report) = left_profile_report(series, nz(m), scale, nz(1));
            black_box(profile);
            (report.tiles, report.wide_tiles)
        }
    }
}

fn cold(series: &[f64], m: usize) -> (Width, u64, u64) {
    let scale = Scale::new(1.0).expect("a scale");
    let ticks: Vec<i64> = series
        .iter()
        .map(|&x| to_ticks(x, scale).expect("in range"))
        .collect();
    let had = vec![true; ticks.len()];
    let windows = ticks.len() - m + 1;
    let survey = Survey::with_spilled(&ticks, &had, nz(m), vec![true; windows])
        .expect("memory")
        .expect("inside the exact domain");
    let mut entries: Vec<(u64, Entry)> = (0..windows as u64).map(|r| (r, Entry::NONE)).collect();
    let counts = survey.merge_left(&mut entries).expect("memory");
    black_box(entries);
    (survey.width(), counts.cells, counts.chains)
}

/// A walk of cents around 5000.00, with `prints` samples made `factor` times their price.
fn messy(len: usize, prints: usize, factor: f64, seed: u64) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64 ^ seed;
    let mut ticks = 500_000i64;
    let mut walk: Vec<f64> = (0..len)
        .map(|_| {
            ticks += (splitmix64(&mut state) % 5) as i64 - 2;
            ticks as f64 * 0.01
        })
        .collect();
    for _ in 0..prints {
        let at = (splitmix64(&mut state) % len as u64) as usize;
        walk[at] *= factor;
    }
    walk
}

/// `messy <family> <n> <m> <prints> <reps>`: the left profile (`left`), the two-sided profile
/// (`two-sided`) or the join with a clean walk half as long (`join`) of a cent walk with bad
/// prints, one thread.
fn run_messy(family: &str, n: usize, m: usize, prints: usize, reps: usize) {
    let data = messy(n, prints, 1000.0, 0);
    let other = messy(n / 2, 0, 1000.0, 7);
    let scale = Scale::new(0.01).expect("a scale");
    for _ in 0..reps {
        let (neighbors, tiles) = match family {
            "left" => {
                let (profile, report) = left_profile_report(&data, nz(m), scale, nz(1));
                black_box(&profile);
                let found = profile.iter().filter(|(_, e)| e.index().get().is_some());
                (
                    found.count(),
                    format!("{}/{}", report.wide_tiles, report.tiles),
                )
            }
            "two-sided" => {
                let profile = two_sided_profile(&data, nz(m), scale, nz(1));
                black_box(&profile);
                let found = profile
                    .iter()
                    .filter(|(_, s)| s.nearest.index().get().is_some());
                (found.count(), "-".to_string())
            }
            "join" => {
                let (profile, report) = ab_join_report(&data, &other, nz(m), scale, scale, nz(1));
                black_box(&profile);
                let found = profile.iter().filter(|(_, e)| e.index().get().is_some());
                (
                    found.count(),
                    format!("{}/{}", report.wide_tiles, report.tiles),
                )
            }
            other => panic!("unknown family {other}"),
        };
        println!(
            "messy {family} n {n} m {m} prints {prints} wide/tiles {tiles} windows with a neighbor {neighbors}"
        );
    }
}

fn run(mode: &str, n: usize, m: usize, reps: usize) {
    let (mode, family) = mode.split_once('-').unwrap_or((mode, "left"));
    let (steps, spike) = match mode {
        "narrow" => (0.0, 0.0),
        "wide" => (1.0e8, 0.0),
        "cold" => (0.0, 0.0),
        "cold256" => (0.0, 8.0e15),
        other => panic!("unknown mode {other}"),
    };
    let data = series(n, steps, 200, spike);
    let other = series(n / 2, steps, 200, 0.0);
    for _ in 0..reps {
        match mode {
            "narrow" | "wide" => {
                let (tiles, wide_tiles) = hot(&data, &other, m, family);
                println!("{mode} {family} n {n} m {m} tiles {tiles} wide {wide_tiles}");
            }
            _ => {
                let (width, cells, chains) = cold(&data, m);
                println!("{mode} n {n} m {m} cells {cells} chains {chains} {width:?}");
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    match args.as_slice() {
        [mode, family, n, m, prints, reps] if mode == "messy" => run_messy(
            family,
            n.parse().expect("n"),
            m.parse().expect("m"),
            prints.parse().expect("prints"),
            reps.parse().expect("reps"),
        ),
        [mode, n, m, reps] => run(
            mode,
            n.parse().expect("n"),
            m.parse().expect("m"),
            reps.parse().expect("reps"),
        ),
        _ => {
            run("narrow", 600, 20, 1);
            run("wide", 600, 20, 1);
            run("cold", 300, 12, 1);
            run("cold256", 3000, 2400, 1);
        }
    }
}
