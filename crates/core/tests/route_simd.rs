#![cfg(all(feature = "oracle", target_arch = "x86_64", not(any(loom, shuttle))))]
//! The AVX2 kernels and the scalar body give the same bits through every engine that runs them:
//! each run is made twice, as the CPU would take it and with the scalar body forced, and the two
//! must agree. Where the CPU has no AVX2 both runs are scalar, and each test says so.

#[path = "common/nonnorm.rs"]
mod nonnorm;
#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join, ab_join_report_tiled, left_profile, left_profile_report_tiled, nonnorm as nn_batch,
    two_sided_profile,
};
use ladde::bounded::BoundedStream;
use ladde::kernel_avx2::{
    Kernel, detected, fast_rows_taken, force_scalar, full_range_rows_taken, rows_taken,
};
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::quantize::{Scale, to_ticks};
use ladde::route::{ConversionForce, RouteForce, forced_route};
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::{Bits, bits, cases, nz, profile_bits, stream_bits};

fn announce() {
    if !detected() {
        eprintln!("skipped: this CPU has no AVX2, so both runs are scalar");
    }
}

/// `run` as the CPU takes it, then with the scalar body forced.
fn both<T>(run: impl Fn() -> T) -> (T, T) {
    let _serial = forced_route(RouteForce::default());
    let natural = run();
    let _scalar = force_scalar();
    (natural, run())
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn z_normalized_batch_and_streams_agree((series, m, scale) in cases(), extra in 0usize..40) {
        announce();
        let (natural, scalar) = both(|| {
            let left = profile_bits(&left_profile(&series, nz(m), scale));
            let two_sided: Vec<(u64, [Bits; 3])> = two_sided_profile(&series, nz(m), scale, nz(2))
                .iter()
                .map(|&(row, s)| (row, [s.left, s.right, s.nearest].map(|e| bits(0, e))))
                .collect();
            let mut growing = GrowingStream::new(nz(m), scale);
            let grown: Vec<_> = series.iter().filter_map(|&x| growing.update(x)).collect();
            let mut bounded = BoundedStream::new(nz(m), nz(m + extra), scale).expect("w >= m");
            let kept: Vec<_> = series.iter().filter_map(|&x| bounded.update(x)).collect();
            (left, two_sided, stream_bits(&grown), stream_bits(&kept))
        });
        prop_assert_eq!(natural, scalar);
    }

    #[test]
    fn non_normalized_batch_and_stream_agree(
        series in prop_oneof![nonnorm::series(), nonnorm::tie_series()],
        m in 1usize..=6,
    ) {
        announce();
        let (natural, scalar) = both(|| {
            let batch = nonnorm::batch_rows(&nn_batch::left_profile(
                &series, nz(m), nonnorm::unit(), nz(2),
            ));
            let mut stream = GrowingNonNormStream::new(nz(m), nonnorm::unit());
            let rows: Vec<_> = series
                .iter()
                .filter_map(|&x| stream.update(x))
                .map(nonnorm::stream_row)
                .collect();
            (batch, rows)
        });
        prop_assert_eq!(natural, scalar);
    }
}

/// Constant, periodic and few-level series, where every row ties.
#[test]
fn tie_heavy_series_agree() {
    announce();
    let scale = ladde::quantize::Scale::new(1.0).unwrap();
    let flat = vec![3.0; 90];
    let period: Vec<f64> = (0..120).map(|k| (k % 7) as f64).collect();
    let levels: Vec<f64> = (0..150).map(|k| ((k * k + 3 * k) % 3) as f64).collect();
    for series in [flat, period, levels] {
        for m in [2, 4, 7] {
            let (natural, scalar) = both(|| {
                let left = profile_bits(&left_profile(&series, nz(m), scale));
                let nn = nonnorm::batch_rows(&nn_batch::left_profile(&series, nz(m), scale, nz(1)));
                (left, nn)
            });
            assert_eq!(natural, scalar, "m = {m}");
        }
    }
}

/// Each engine's rows reach the AVX2 kernel when the CPU has it, and none do when the scalar body
/// is forced.
#[test]
fn rows_reach_the_avx2_kernel_exactly_when_it_is_allowed() {
    announce();
    let _serial = forced_route(RouteForce::default());
    let scale = ladde::quantize::Scale::new(1.0).unwrap();
    let series: Vec<f64> = (0..200).map(|k| ((k * 37) % 11) as f64).collect();
    let run = |kernel: Kernel| {
        let before = rows_taken(kernel);
        match kernel {
            Kernel::Narrow => {
                let _ = left_profile(&series, nz(8), scale);
                let mut stream = GrowingStream::new(nz(8), scale);
                for &x in &series {
                    let _ = stream.update(x);
                }
            }
            Kernel::NonNormalized => {
                let _ = nn_batch::left_profile(&series, nz(8), scale, nz(1));
                let mut stream = GrowingNonNormStream::new(nz(8), scale);
                for &x in &series {
                    let _ = stream.update(x);
                }
            }
        }
        rows_taken(kernel) - before
    };
    for kernel in [Kernel::Narrow, Kernel::NonNormalized] {
        assert_eq!(
            run(kernel) > 0,
            detected(),
            "{kernel:?} as the CPU takes it"
        );
        let _scalar = force_scalar();
        assert_eq!(run(kernel), 0, "{kernel:?} with the scalar body forced");
    }
}

/// Windows of 8 ticks whose `V = m·Q − S²` is `2⁵¹ − 1`, and of 3 ticks whose `V` is `2⁵¹`.
const JUST_INSIDE: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 587_366.0, -17_842_265.0];
const JUST_OUTSIDE: [f64; 3] = [0.0, 0.0, 33_554_432.0];

/// `pattern` twice among zeros, so two windows repeat each other exactly (their cross term is
/// the widest `V`). With `gap`, the first copy's second sample is missing (filled with the zero
/// before it): the widest window there is non-finite, and a valid row still meets it.
fn edge_series(pattern: &[f64], gap: bool) -> Vec<f64> {
    let mut series = vec![0.0; 5];
    series.extend_from_slice(pattern);
    series.extend(core::iter::repeat_n(0.0, 30));
    series.extend_from_slice(pattern);
    series.extend(core::iter::repeat_n(0.0, 9));
    if gap {
        series[6] = f64::NAN;
    }
    series
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

/// The z-normalized batch engines that rank through the row kernel.
const ENGINES: [&str; 5] = ["left", "left tiled", "two-sided", "join", "join tiled"];

/// One engine of [`ENGINES`] on `series`, as bits.
fn engine_bits(engine: &str, series: &[f64], m: usize, scale: Scale) -> Vec<Bits> {
    let other: Vec<f64> = series.iter().rev().map(|x| x + 3.0).collect();
    match engine {
        "left" => profile_bits(&left_profile(series, nz(m), scale)),
        "left tiled" => profile_bits(
            &left_profile_report_tiled(series, nz(m), scale, nz(4), nz(4), nz(2), nz(16)).0,
        ),
        "two-sided" => two_sided_profile(series, nz(m), scale, nz(3))
            .iter()
            .flat_map(|&(row, s)| [s.left, s.right, s.nearest].map(|e| bits(row, e)))
            .collect(),
        "join" => profile_bits(&ab_join(series, &other, nz(m), scale, scale, nz(1))),
        _ => profile_bits(
            &ab_join_report_tiled(
                series,
                &other,
                nz(m),
                scale,
                scale,
                nz(3),
                nz(4),
                nz(2),
                nz(16),
            )
            .0,
        ),
    }
}

/// Every engine of [`ENGINES`] on `series`, as bits.
fn batch_bits(series: &[f64], m: usize, scale: Scale) -> Vec<Vec<Bits>> {
    ENGINES
        .iter()
        .map(|engine| engine_bits(engine, series, m, scale))
        .collect()
}

/// The fast conversion runs exactly where the bound fits and AVX2 is there, the full-range one
/// exactly where it does not, in every engine and on the whole-series anchor, per-tile anchors
/// and re-anchored cells, and each run gives the bits of the full-range conversion forced
/// everywhere and of the scalar body: at `V = 2⁵¹ − 1` (the repeat's cross term is the largest
/// the fast conversion takes) and `2⁵¹`, with and without a gap in a widest window.
#[test]
fn the_edge_series_agree_on_every_conversion() {
    announce();
    let shapes = [
        RouteForce::default(),
        RouteForce {
            per_tile: true,
            ..RouteForce::default()
        },
        RouteForce {
            reanchor_cell: 2,
            ..RouteForce::default()
        },
    ];
    for (pattern, m, fits) in [(&JUST_INSIDE[..], 8, true), (&JUST_OUTSIDE[..], 3, false)] {
        for gap in [false, true] {
            let series = edge_series(pattern, gap);
            for engine in ENGINES {
                for shape in shapes {
                    let at = format!("m {m} gap {gap} {engine} {shape:?}");
                    let run = |conversion: ConversionForce| {
                        let _forced = forced_route(RouteForce {
                            conversion,
                            ..shape
                        });
                        let before = (fast_rows_taken(), full_range_rows_taken());
                        let out = engine_bits(engine, &series, m, unit());
                        let after = (fast_rows_taken(), full_range_rows_taken());
                        (out, after.0 - before.0, after.1 - before.1)
                    };
                    let (natural, fast_rows, full_rows) = run(ConversionForce::Auto);
                    assert_eq!(fast_rows > 0, fits && detected(), "{at}");
                    assert_eq!(full_rows > 0, !fits && detected(), "{at}");
                    assert_eq!(natural, run(ConversionForce::Fast).0, "{at}");
                    let (full, full_fast_rows, full_rows) = run(ConversionForce::FullRange);
                    assert_eq!((full_fast_rows, full_rows > 0), (0, detected()), "{at}");
                    assert_eq!(natural, full, "{at}");
                    let _forced = forced_route(shape);
                    let _scalar = force_scalar();
                    assert_eq!(natural, engine_bits(engine, &series, m, unit()), "{at}");
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    /// Random series, fitting or not: the natural conversion, the full-range one forced
    /// everywhere and the scalar body give the same bits.
    #[test]
    fn the_conversion_never_changes_a_bit((series, m, scale) in cases()) {
        announce();
        let run = |force: ConversionForce| {
            let _forced = forced_route(RouteForce {
                conversion: force,
                ..RouteForce::default()
            });
            batch_bits(&series, m, scale)
        };
        let natural = run(ConversionForce::Auto);
        prop_assert_eq!(&natural, &run(ConversionForce::FullRange));
        let _serial = forced_route(RouteForce::default());
        let _scalar = force_scalar();
        prop_assert_eq!(&natural, &batch_bits(&series, m, scale));
    }
}

/// The largest `V = m·Q − S²` over the windows of `series` at `scale`, exact.
fn widest_v(series: &[f64], m: usize, scale: Scale) -> u128 {
    let ticks: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    ticks
        .windows(m)
        .filter_map(|w| {
            let w: Option<Vec<i128>> = w.iter().map(|t| t.map(i128::from)).collect();
            let w = w?;
            let (s, q) = (
                w.iter().sum::<i128>(),
                w.iter().map(|t| t * t).sum::<i128>(),
            );
            Some((m as i128 * q - s * s) as u128)
        })
        .max()
        .unwrap_or(0)
}

/// Off-grid floats, with a gap, at the default scale: cross terms far past `2⁵³`, where the
/// conversion rounds.
fn wide_float_series(n: usize, salt: u64) -> Vec<f64> {
    let mut x = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut series: Vec<f64> = (0..n)
        .map(|k| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let walk = (k as f64 * 0.37).sin() * 40.0;
            walk + (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    series[n / 3] = f64::NAN;
    series
}

/// Series whose cross terms reach far past `2⁵³` (the fast conversion never fires and the
/// conversion rounds): every engine, on the whole-series anchor, per-tile anchors and
/// re-anchored cells, gives the same bits under the automatic route, every forced conversion
/// and the scalar body, and the full-range body runs on every row where AVX2 is there.
#[test]
fn wide_terms_agree_on_every_conversion() {
    announce();
    let shapes = [
        RouteForce::default(),
        RouteForce {
            per_tile: true,
            ..RouteForce::default()
        },
        RouteForce {
            reanchor_cell: 2,
            ..RouteForce::default()
        },
    ];
    for (m, salt) in [(3, 1), (8, 2), (24, 3)] {
        let series = wide_float_series(160, salt);
        let scale = Scale::infer(&series, nz(m)).expect("a scale");
        let widest = widest_v(&series, m, scale);
        assert!(
            widest > 1 << 56,
            "m {m}: V {widest} does not reach past 2^56"
        );
        for engine in ENGINES {
            for shape in shapes {
                let at = format!("m {m} {engine} {shape:?}");
                let run = |conversion: ConversionForce| {
                    let _forced = forced_route(RouteForce {
                        conversion,
                        ..shape
                    });
                    let before = (fast_rows_taken(), full_range_rows_taken());
                    let out = engine_bits(engine, &series, m, scale);
                    let after = (fast_rows_taken(), full_range_rows_taken());
                    (out, after.0 - before.0, after.1 - before.1)
                };
                let (natural, fast_rows, full_rows) = run(ConversionForce::Auto);
                assert_eq!((fast_rows, full_rows > 0), (0, detected()), "{at}");
                for force in [ConversionForce::Fast, ConversionForce::FullRange] {
                    let (out, fast_rows, full_rows) = run(force);
                    assert_eq!(natural, out, "{at} {force:?}");
                    assert_eq!(fast_rows, 0, "{at} {force:?}");
                    assert_eq!(full_rows > 0, detected(), "{at} {force:?}");
                }
                let _forced = forced_route(shape);
                let _scalar = force_scalar();
                assert_eq!(natural, engine_bits(engine, &series, m, scale), "{at}");
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]

    /// Random off-grid floats at the default scale: the automatic route, the full-range
    /// conversion forced everywhere, and the scalar body give the same bits in every engine.
    #[test]
    fn wide_random_terms_agree(salt in any::<u64>(), n in 40usize..200, m in 3usize..20) {
        announce();
        let series = wide_float_series(n, salt);
        let scale = Scale::infer(&series, nz(m)).expect("a scale");
        let run = |force: ConversionForce| {
            let _forced = forced_route(RouteForce {
                conversion: force,
                ..RouteForce::default()
            });
            batch_bits(&series, m, scale)
        };
        let natural = run(ConversionForce::Auto);
        prop_assert_eq!(&natural, &run(ConversionForce::FullRange));
        let _serial = forced_route(RouteForce::default());
        let _scalar = force_scalar();
        prop_assert_eq!(&natural, &batch_bits(&series, m, scale));
    }
}
