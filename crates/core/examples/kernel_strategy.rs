//! Scratch comparison of the hot row kernel's estimate-and-sweep strategies, for timing only.
//!
//! Nothing here is wired into the library. The bodies are benchmark-local copies of the narrow
//! row kernel that [`ladde::kernel_neon::rank_row_groups`] (the function the hot path
//! calls) runs with floors:
//!
//! * on aarch64 (`kernel_strategy/neon.rs`) the 1-row body copies `kernel_neon::rank_neon` and its
//!   `Lanes8::rank` step: widening `i32 x i32` products into `i64` with `vmull_s32`, the
//!   conversion `vcvtq_f64_s64`, the separate multiply and add chain `x * ri * r + bias` (never
//!   fused), four max accumulators, group maxima and the hit mask per 64 lanes;
//! * on x86_64 (`kernel_strategy/avx2.rs`) it copies `kernel_avx2::rank` and its `Lanes4::rank`
//!   step: `_mm256_mul_epi32` on sign-extended lanes, the full-range conversion `to_f64_full`
//!   (two 32-bit halves as exact doubles by magic constants, an exact subtract and one rounding
//!   add), the same chain, two max accumulators alternating per step; the AVX2 body is
//!   runtime-detected.
//!
//! The 1-row copy keeps production's loop shape (the step, the accumulators, the group close and
//! the returned row maximum) and leaves out only the scalar tail, which the harness's lane counts
//! (multiples of 64) never reach. The strategies run over the same data, interleaved, and are
//! timed per (row, lane) pair:
//!
//! * `f64 x1`: the 1-row copy, one row per pass over the columns;
//! * `f32 x1`: one row per pass, the estimate in `f32` from precomputed `f32` columns (`r`,
//!   `bias`, `floor`) into an `f32` output tile; the `i64` lane goes to `f32` through `f64`
//!   (`i64` to `f64` as the production kernel does it, full range on AVX2, then narrowed to
//!   `f32`);
//! * `f64 x2`: the 1-row copy's step for two rows per pass, each `qt` vector loaded and stored
//!   once for both rows, each row with its own accumulators and group reports;
//! * `f64 x4`: the same for four rows, the register-spill check;
//! * `prod x1`: `rank_row_groups` itself, the baseline every gain is measured against;
//! * `fast x1`, `fast x2`: `f64 x1` and `f64 x2` with the `i64` to `f64` conversion done by the
//!   magic constant (an integer add of the bits of `1.5 * 2^52`, then an `f64` subtract of it),
//!   equal to `as f64` bit for bit while `|x| < 2^51` (the proof is beside `to_f64_fast`). Both
//!   run as a per-series route: the fast body on a series whose every cross term is proved
//!   inside that range, the full-range body otherwise, chosen per row from a flag the compiler
//!   cannot see, so a cell where the route does not fire is timed on its real fallback path. On
//!   NEON, which converts in one instruction, they are timed as a control only.
//!
//! * `old x1`: `f64 x1` with every `i64` converted by the old production conversion on AVX2,
//!   emulated (both 32-bit halves through `i32` conversions after lane-crossing permutes), which
//!   full-range-convert replaced; the library no longer has it, and this example keeps its own
//!   copy only as the "old production" reference. It is reported, not decided. On NEON, whose
//!   conversion was and is native, it equals `f64 x1`.
//!
//! The `f32`, multi-row and fast-conversion bodies do not exist in the library. Every body
//! returns its hit masks and writes its output tile, and the sweeps fold the masks into a value
//! the timer's caller consumes, so the compiler cannot drop the floor tests or the stores.
//!
//! `--check` (and the unit tests of this example, and the start of every default run) proves the
//! timings compare equal work: on random inputs and on every dataset cell, the `f64 x1`, `f64 x2`
//! and `f64 x4` lanes, updated `qt`, group maxima and hit masks equal `rank_row_groups` bit for
//! bit; so do the `fast x1` and `fast x2` routes, which take the fast body on every input whose
//! cross terms lie within `2^51` (random inputs drawn inside it, and each dataset cell whose
//! terms do) and the full-range body on the others, so does `old x1`, and the fast
//! conversion equals `as f64` on the range's edges, on zero and one, and on random values, and
//! differs just outside it; `f32 x1` equals a scalar `f32` reference bit for bit and lies within
//! `6 * 2^-24 * |x * ri * r| + 2^-24 * |estimate|` of the production `f64` estimate. `--check` is
//! deterministic: it times nothing.
//!
//! The timing run also checks the copy against production: when the median ratio of `f64 x1` to
//! `prod x1` is more than 2% from one, the copy is not faithful and the run says the comparison
//! is void.
//!
//! The range report (`--range`, and the start of every timing run) measures, per dataset and
//! window length at the library's default scale (`Scale::infer`), the largest cross term the
//! kernel can meet: the maximum variance numerator `V = m * Q - S^2` over every window of the
//! ticks the kernel ranks, which bounds every cross term by Cauchy-Schwarz (`|N| <= sqrt(Vi*Vj)`)
//! and is attained at lag 0, beside the cheaper tick-range bound `m^2 * R^2 / 4` (`R` the widest
//! window's tick range), and whether each fits `2^51` (the fast conversion) and `2^53`.
//!
//! Decision rule. A whole body (`f32 x1`, `f64 x2`) is adopted on a target only if it beats
//! `prod x1` by at least 15% on grid m64 and float m64 and does not run slower than it (more than
//! 2% noise margin) on any ties or messy cell; of those that qualify, the fastest on the m64 cells
//! wins; if none does, the 1-row f64 body stays. A route is judged where it fires: it fires on a
//! timing cell when the range report fits `2^51` for that dataset and window length at the default
//! scale and the cell's own terms do; it is adopted only if it beats `prod x1` by at least 15% on
//! every m64 cell where it fires (grid and at least one of messy or ties among them), does not run
//! more than 2% slower on any cell where it does not fire, and `--check` shows equal bits. `fast
//! x1` rides the 1-row f64 body and `fast x2` the 2-row sweep, so a route is adopted only on top
//! of its body; the decision names both. Local runs are for basic confidence only: the x86_64 AVX2
//! run on the benchmark box decides.
//!
//! Run: `cargo run --release -p ladde --example kernel_strategy` (no arguments; `--quick`
//! for a smoke run, `--check` for the equality checks alone, `--range` for the range report alone,
//! `--range-file NAME=PATH` to add a raw little-endian `f64` series to the range report,
//! `--out FILE` for the JSON path).
//! The datasets are generated here with the shapes of the campaign's grid, float, ties and messy
//! series (not byte-identical to its files), quantized to ticks locally for timing only.
#![allow(unsafe_op_in_unsafe_fn)]
// The bodies index small fixed arrays by row and take the kernel's own long argument lists.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod harness {
    use super::body;
    use core::num::NonZeroUsize;
    use ladde::kernel_neon::{Conversion, GROUP_LANES, RowScalars, SPAN_LANES, rank_row_groups};
    use ladde::quantize::{Scale, to_ticks};
    use ladde::sanitize::{ForwardFill, Sanitized};
    use ladde::spill::outlier_ticks;
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::Instant;

    const DEFAULT_LANES: usize = 256;
    const MIN_GAIN: f64 = 0.15;
    const NOISE_MARGIN: f64 = 0.02;
    /// The most the 1-row copy may differ from production in time before the comparison is void.
    const FAITHFUL_MARGIN: f64 = 0.02;
    const VARIANTS: usize = 8;
    const F64X1: usize = 0;
    const F32X1: usize = 1;
    const F64X2: usize = 2;
    const F64X4: usize = 3;
    const PROD: usize = 4;
    const FASTX1: usize = 5;
    const FASTX2: usize = 6;
    const OLDX1: usize = 7;
    const NAMES: [&str; VARIANTS] = [
        "f64 x1", "f32 x1", "f64 x2", "f64 x4", "prod x1", "fast x1", "fast x2", "old x1",
    ];
    /// The node each candidate strategy would become.
    const NODES: [&str; VARIANTS] = [
        "",
        "f32-estimate",
        "two-row-sweep",
        "",
        "",
        "fast-conversion route",
        "fast-conversion route on the 2-row sweep",
        "",
    ];
    /// Exclusive bound on `|x|` for the fast conversion.
    const FAST_RANGE: u128 = 1 << 51;
    /// Exclusive bound on `|x|` for an exact `f64` recurrence (no conversion at all).
    const EXACT_F64: u128 = 1 << 53;
    const DATASETS: [&str; 4] = ["grid", "float", "messy", "ties"];
    const MS: [usize; 4] = [16, 64, 256, 1024];
    pub const RANGE_N: usize = 65_536;

    // ---- data ------------------------------------------------------------------------------

    /// The per-window columns the kernels read, `f32` mirrors included. Index = window.
    pub struct Cols {
        pub enter: Vec<i32>,
        pub leave: Vec<i32>,
        pub sum: Vec<i32>,
        pub r: Vec<f64>,
        pub bias: Vec<f64>,
        pub floor: Vec<f64>,
        pub r32: Vec<f32>,
        pub bias32: Vec<f32>,
        pub floor32: Vec<f32>,
    }

    impl Cols {
        pub fn len(&self) -> usize {
            self.enter.len()
        }
    }

    /// One timed cell: the columns, the row scalars of rows `0..`, and the lane state before
    /// row 1. Lane `l` of row `i` pairs window `i` with window `i + base + l`, so row `i` reads
    /// the columns from `i + base`.
    pub struct Cell {
        pub name: String,
        pub dataset: String,
        pub m: usize,
        pub lanes: usize,
        pub base: usize,
        pub cols: Cols,
        pub rows: Vec<RowScalars>,
        pub qt0: Vec<i64>,
        /// The largest variance numerator over the cell's windows: a bound on every cross term
        /// its sweep converts.
        pub max_v: u128,
        /// Whether the fast-conversion route takes this series: its bound proves every cross
        /// term inside the range. The timing run also requires the default-scale range report to
        /// fit, so the route is timed where it would fire in the library.
        pub route_fires: bool,
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + self.below((hi - lo + 1) as u64) as i64
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn walk(rng: &mut Rng, n: usize) -> Vec<i64> {
        let mut v = Vec::with_capacity(n);
        let mut t = 10_000;
        for _ in 0..n {
            v.push(t);
            t += rng.range(-2, 2);
        }
        v
    }

    fn series(dataset: &str, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng(seed);
        match dataset {
            "grid" => walk(&mut rng, n).iter().map(|&t| t as f64 * 0.01).collect(),
            "float" => {
                let mut x = 100.0;
                (0..n)
                    .map(|_| {
                        x += rng.unit() - 0.5;
                        x
                    })
                    .collect()
            }
            "ties" => {
                let mut out = Vec::with_capacity(n);
                while out.len() < n {
                    let length = rng.range(64, 1023) as usize;
                    let mut block = Vec::with_capacity(length + 8);
                    match rng.below(3) {
                        0 => {
                            while block.len() < length {
                                let level = rng.below(3) as f64;
                                let run = rng.range(1, 8) as usize;
                                block.extend(std::iter::repeat_n(level, run));
                            }
                        }
                        1 => {
                            let cycle: Vec<f64> = (0..7).map(|_| rng.below(3) as f64).collect();
                            block.extend((0..length).map(|i| cycle[i % 7]));
                        }
                        _ => block.extend(std::iter::repeat_n(rng.below(3) as f64, length)),
                    }
                    block.truncate(length);
                    out.extend(block);
                }
                out.truncate(n);
                out
            }
            "messy" => {
                let mut ticks = walk(&mut rng, n);
                for _ in 0..3 {
                    let start = rng.below((n - n / 32) as u64) as usize;
                    let held = ticks[start];
                    ticks[start..start + n / 64].fill(held);
                }
                let mut x: Vec<f64> = ticks.iter().map(|&t| t as f64 * 0.01).collect();
                const FACTORS: [f64; 11] = [
                    1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
                ];
                for _ in 0..10 {
                    let at = rng.below(n as u64) as usize;
                    x[at] = (x[at] * FACTORS[rng.below(11) as usize] / 0.01).round() * 0.01;
                }
                let mut gaps = vec![(0, 3), (n - 2, 2)];
                for _ in 0..4 {
                    gaps.push((
                        rng.below((n - 20) as u64) as usize,
                        rng.range(1, 20) as usize,
                    ));
                }
                for (start, length) in gaps {
                    x[start..start + length].fill(f64::NAN);
                }
                x
            }
            other => panic!("unknown dataset {other}"),
        }
    }

    /// Ticks at the finest power-of-two scale that keeps the 99th percentile of |x| within 2^18,
    /// clamped to +-2^19 so every kernel product and window sum fits (timing only; not the
    /// library's scale rule). Non-finite samples become tick 0 and mark their windows.
    fn quantize(x: &[f64]) -> Vec<i32> {
        let mut mags: Vec<f64> = x
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .collect();
        mags.sort_by(f64::total_cmp);
        let top = mags[mags.len() * 99 / 100].max(1e-9);
        let scale = 2f64.powi((262_144.0 / top).log2().floor() as i32);
        x.iter()
            .map(|v| {
                if v.is_finite() {
                    (v * scale).round().clamp(-524_288.0, 524_288.0) as i32
                } else {
                    0
                }
            })
            .collect()
    }

    fn seed_of(dataset: &str, n: usize) -> u64 {
        0x5EED ^ (n as u64) ^ ((dataset.len() as u64) << 40)
    }

    pub fn build_cell(dataset: &str, n: usize, m: usize, lanes: usize, rows: usize) -> Cell {
        let x = series(dataset, n, seed_of(dataset, n));
        let t = quantize(&x);
        let windows = n - m + 1;
        let base = m;
        assert!(
            rows + base + lanes + 16 <= windows,
            "series too short for the sweep"
        );
        let mut sum_p = vec![0i64; n + 1];
        let mut sq_p = vec![0i64; n + 1];
        let mut bad_p = vec![0u32; n + 1];
        for k in 0..n {
            sum_p[k + 1] = sum_p[k] + i64::from(t[k]);
            sq_p[k + 1] = sq_p[k] + i64::from(t[k]) * i64::from(t[k]);
            bad_p[k + 1] = bad_p[k] + u32::from(!x[k].is_finite());
        }
        let pad = 16;
        let len = windows + pad;
        let mut c = Cols {
            enter: vec![0; len],
            leave: vec![0; len],
            sum: vec![0; len],
            r: vec![0.0; len],
            bias: vec![0.0; len],
            floor: vec![0.9; len],
            r32: vec![0.0; len],
            bias32: vec![0.0; len],
            floor32: vec![0.9; len],
        };
        let mut ri = vec![1.0f64; windows];
        let mut max_v = 0u128;
        for j in 0..windows {
            let s = sum_p[j + m] - sum_p[j];
            let q = sq_p[j + m] - sq_p[j];
            let v = m as i128 * i128::from(q) - i128::from(s) * i128::from(s);
            max_v = max_v.max(v.unsigned_abs());
            c.enter[j] = t[j + m - 1];
            c.leave[j] = if j > 0 { t[j - 1] } else { 0 };
            c.sum[j] = s as i32;
            if bad_p[j + m] != bad_p[j] {
                c.bias[j] = f64::NEG_INFINITY;
            } else if v == 0 {
                c.bias[j] = 0.5;
            } else {
                c.r[j] = 1.0 / (v as f64).sqrt();
                ri[j] = c.r[j];
            }
        }
        for j in 0..len {
            c.r32[j] = c.r[j] as f32;
            c.bias32[j] = c.bias[j] as f32;
            c.floor32[j] = c.floor[j] as f32;
        }
        let row_scalars: Vec<RowScalars> = (0..windows)
            .map(|i| RowScalars {
                me: m as i32 * t[i + m - 1],
                ml: if i > 0 { m as i32 * t[i - 1] } else { 0 },
                si: c.sum[i],
                ri: ri[i],
                conversion: Conversion::FULL_RANGE,
            })
            .collect();
        let qt0 = (0..lanes)
            .map(|l| {
                (0..m).fold(0i64, |a, k| {
                    a.wrapping_add(
                        (m as i64).wrapping_mul(i64::from(t[k]) * i64::from(t[base + l + k])),
                    )
                })
            })
            .collect();
        Cell {
            name: format!("{dataset} m{m}"),
            dataset: dataset.to_string(),
            m,
            lanes,
            base,
            cols: c,
            rows: row_scalars,
            qt0,
            max_v,
            route_fires: max_v < FAST_RANGE,
        }
    }

    /// Random columns and rows with the three window kinds, for the equality check. With
    /// `in_range`, every cross term the sweep meets stays within `2^50` (the lane states start
    /// within `2^49`, and each row moves a lane by less than `2^37`), so the fast conversion is
    /// exact on it.
    fn random_cell(seed: u64, lanes: usize, rows: usize, in_range: bool) -> Cell {
        let mut rng = Rng(seed);
        let base = 3;
        let len = rows + base + lanes + 16;
        let mut c = Cols {
            enter: Vec::new(),
            leave: Vec::new(),
            sum: Vec::new(),
            r: Vec::new(),
            bias: Vec::new(),
            floor: Vec::new(),
            r32: Vec::new(),
            bias32: Vec::new(),
            floor32: Vec::new(),
        };
        for _ in 0..len {
            c.enter.push(rng.range(-1_000_000, 1_000_000) as i32);
            c.leave.push(rng.range(-1_000_000, 1_000_000) as i32);
            c.sum.push(rng.range(-20_000_000, 20_000_000) as i32);
            let (r, bias) = match rng.below(8) {
                0 => (0.0, 0.5),
                1 => (0.0, f64::NEG_INFINITY),
                _ => (1e-6 * (rng.below(1000) + 1) as f64, 0.0),
            };
            c.r.push(r);
            c.bias.push(bias);
            c.floor.push(-1.0 + 2.0 * rng.unit());
            c.r32.push(r as f32);
            c.bias32.push(bias as f32);
            c.floor32.push(*c.floor.last().unwrap() as f32);
        }
        let row_scalars = (0..=rows + 8)
            .map(|_| RowScalars {
                me: 64 * rng.range(-1000, 1000) as i32,
                ml: 64 * rng.range(-1000, 1000) as i32,
                si: rng.range(-20_000_000, 20_000_000) as i32,
                ri: 1e-6 * (rng.below(1000) + 1) as f64,
                conversion: Conversion::FULL_RANGE,
            })
            .collect();
        let qt0 = (0..lanes)
            .map(|_| {
                if in_range {
                    rng.range(-(1 << 49), 1 << 49)
                } else {
                    rng.next() as i64
                }
            })
            .collect();
        Cell {
            name: format!(
                "random lanes {lanes} seed {seed}{}",
                if in_range { " in range" } else { "" }
            ),
            dataset: "random".into(),
            m: 0,
            lanes,
            base,
            cols: c,
            rows: row_scalars,
            qt0,
            max_v: u128::MAX,
            route_fires: false,
        }
    }

    // ---- safe wrappers over the bodies -----------------------------------------------------

    /// The 1-row copy as `rank_row_groups` calls production: the lane slices offset to the row's
    /// first lag, their lengths asserted, the floors on, the hit mask returned.
    #[inline(never)]
    fn rank1<const CONV: u8>(
        qt: &mut [i64],
        cols: &Cols,
        col0: usize,
        row: RowScalars,
        scratch: &mut [f64],
        maxima: &mut [f64],
    ) -> u64 {
        let len = qt.len();
        let lanes = body::Lanes {
            enter: &cols.enter[col0..col0 + len],
            leave: &cols.leave[col0..col0 + len],
            sum: &cols.sum[col0..col0 + len],
            r: &cols.r[col0..col0 + len],
            bias: &cols.bias[col0..col0 + len],
        };
        let floors = &cols.floor[col0..col0 + len];
        assert!(len.is_multiple_of(GROUP_LANES) && len <= SPAN_LANES);
        assert_eq!(scratch.len(), len);
        assert_eq!(maxima.len(), len / GROUP_LANES);
        // SAFETY: the CPU has the body's instruction set (`main` checks `body::available`); every
        // lane slice, `scratch` and `floors` hold `len` elements, a multiple of the group size,
        // and `maxima` one slot per group, at most `SPAN_GROUPS`.
        unsafe { body::rank1::<true, true, CONV>(qt, &lanes, row, Some(scratch), floors, maxima).1 }
    }

    fn rows<const R: usize, const CONV: u8>(
        qt: &mut [i64],
        cols: &Cols,
        col0: usize,
        rows: &[RowScalars; R],
        outs: &mut [Vec<f64>; R],
        maxima: &mut [Vec<f64>; R],
    ) -> [u64; R] {
        let lanes = qt.len();
        assert!(lanes.is_multiple_of(GROUP_LANES) && lanes > 0 && lanes <= SPAN_LANES);
        assert!(col0 + R - 1 + lanes <= cols.len());
        for t in 0..R {
            assert_eq!(outs[t].len(), lanes);
            assert_eq!(maxima[t].len(), lanes / GROUP_LANES);
        }
        // SAFETY: the CPU has the body's instruction set (`main` checks `body::available`); `qt`
        // and every output tile hold `lanes` elements, a multiple of the group size, `maxima`
        // one slot per group, and the columns hold every index `col0 .. col0 + R - 1 + lanes`.
        unsafe { body::rows::<R, CONV>(qt, cols, col0, rows, outs, maxima) }
    }

    /// The fast-conversion route over one row: the fast body when the series' bound proves every
    /// cross term inside the range (`fires`), production's 1-row body otherwise.
    fn route1(
        fires: bool,
        qt: &mut [i64],
        cols: &Cols,
        col0: usize,
        row: RowScalars,
        scratch: &mut [f64],
        maxima: &mut [f64],
    ) -> u64 {
        if fires {
            rank1::<{ body::FAST }>(qt, cols, col0, row, scratch, maxima)
        } else {
            rank1::<{ body::PROD }>(qt, cols, col0, row, scratch, maxima)
        }
    }

    /// The fast-conversion route over `R` rows per pass, as [`route1`].
    fn route_rows<const R: usize>(
        fires: bool,
        qt: &mut [i64],
        cols: &Cols,
        col0: usize,
        rs: &[RowScalars; R],
        outs: &mut [Vec<f64>; R],
        maxima: &mut [Vec<f64>; R],
    ) -> [u64; R] {
        if fires {
            rows::<R, { body::FAST }>(qt, cols, col0, rs, outs, maxima)
        } else {
            rows::<R, { body::PROD }>(qt, cols, col0, rs, outs, maxima)
        }
    }

    fn row32(
        qt: &mut [i64],
        cols: &Cols,
        col0: usize,
        row: RowScalars,
        out: &mut [f32],
        maxima: &mut [f32],
    ) -> u64 {
        let lanes = qt.len();
        assert!(lanes.is_multiple_of(GROUP_LANES) && lanes > 0);
        assert!(col0 + lanes <= cols.len());
        assert_eq!(out.len(), lanes);
        assert_eq!(maxima.len(), lanes / GROUP_LANES);
        // SAFETY: as in `rows`, for one row.
        unsafe { body::row32(qt, cols, col0, row, out, maxima) }
    }

    // ---- equality checks -------------------------------------------------------------------

    /// Compares one row of a variant (name, row, output tile, group maxima, hit mask) with
    /// production.
    type Same64<'a> = dyn Fn(&str, usize, &[f64], &[f64], u64) -> Result<(), String> + 'a;

    fn f32_bound(x: i64, row: RowScalars, r: f64, estimate: f64) -> f64 {
        6.0 * (-24f64).exp2() * ((x as f64) * row.ri * r).abs() + (-24f64).exp2() * estimate.abs()
    }

    /// Runs rows `1..=nrows` (a multiple of 4) through the production kernel and every variant
    /// and compares them; `Err` names the first difference. The fast-conversion variants run as
    /// routes: the fast body when every cross term of those rows lies within its range,
    /// production's body otherwise; returns whether the fast body ran.
    pub fn check_cell(cell: &Cell, nrows: usize) -> Result<bool, String> {
        assert!(nrows.is_multiple_of(4) && nrows + 1 < cell.rows.len());
        let l = cell.lanes;
        let groups = l / GROUP_LANES;
        let c = &cell.cols;
        // Production: one rank_row_groups call per row.
        let mut qt_p = cell.qt0.clone();
        let mut est = vec![vec![0.0f64; l]; nrows + 1];
        let mut max = vec![vec![0.0f64; groups]; nrows + 1];
        let mut mask = vec![0u64; nrows + 1];
        for i in 1..=nrows {
            let at = i + cell.base;
            mask[i] = rank_row_groups(
                &mut qt_p,
                &c.enter[at..at + l],
                &c.leave[at..at + l],
                &c.sum[at..at + l],
                &c.r[at..at + l],
                &c.bias[at..at + l],
                cell.rows[i],
                &mut est[i],
                Some(&c.floor[at..at + l]),
                &mut max[i],
            );
        }
        let same64 = |what: &str,
                      i: usize,
                      out: &[f64],
                      mx: &[f64],
                      mk: u64|
         -> Result<(), String> {
            for (lane, (a, b)) in out.iter().zip(&est[i]).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "{}: {what} row {i} lane {lane}: {a:e} ({:#x}) != production {b:e} ({:#x})",
                        cell.name,
                        a.to_bits(),
                        b.to_bits()
                    ));
                }
            }
            if mx.iter().zip(&max[i]).any(|(a, b)| a != b) {
                return Err(format!(
                    "{}: {what} row {i}: group maxima differ",
                    cell.name
                ));
            }
            if mk != mask[i] {
                return Err(format!(
                    "{}: {what} row {i}: hit mask {mk:#x} != production {:#x}",
                    cell.name, mask[i]
                ));
            }
            Ok(())
        };
        let tile = || vec![0.0f64; l];
        let mx = || vec![0.0f64; groups];
        // Whether every cross term these rows meet lies within the fast conversion's range.
        let fast_applies = {
            let mut qs = cell.qt0.clone();
            let mut fits = true;
            for i in 1..=nrows {
                let (at, row) = (i + cell.base, cell.rows[i]);
                for (lane, q) in qs.iter_mut().enumerate() {
                    let col = at + lane;
                    *q = q
                        .wrapping_add(i64::from(row.me).wrapping_mul(i64::from(c.enter[col])))
                        .wrapping_sub(i64::from(row.ml).wrapping_mul(i64::from(c.leave[col])));
                    let x = q.wrapping_sub(i64::from(row.si).wrapping_mul(i64::from(c.sum[col])));
                    fits &= u128::from(x.unsigned_abs()) < FAST_RANGE;
                }
            }
            fits
        };
        // The 1-row copy, the fast-conversion route over it (the fast body when these rows'
        // terms fit, production's otherwise), and the old production conversion on every term.
        for variant in [F64X1, FASTX1, OLDX1] {
            let what = NAMES[variant];
            let mut qt = cell.qt0.clone();
            let (mut o, mut mm) = (tile(), mx());
            for i in 1..=nrows {
                let at = i + cell.base;
                let row = cell.rows[i];
                let h = match variant {
                    FASTX1 => route1(fast_applies, &mut qt, c, at, row, &mut o, &mut mm),
                    OLDX1 => rank1::<{ body::OLD }>(&mut qt, c, at, row, &mut o, &mut mm),
                    _ => rank1::<{ body::PROD }>(&mut qt, c, at, row, &mut o, &mut mm),
                };
                same64(what, i, &o, &mm, h)?;
            }
            if qt != qt_p {
                return Err(format!("{}: {what}: qt differs from production", cell.name));
            }
        }
        // The multi-row sweeps.
        fn sweep<const R: usize>(
            cell: &Cell,
            nrows: usize,
            what: &str,
            fast: bool,
            same64: &Same64,
        ) -> Result<Vec<i64>, String> {
            let (l, groups) = (cell.lanes, cell.lanes / GROUP_LANES);
            let mut qt = cell.qt0.clone();
            let mut outs: [Vec<f64>; R] = std::array::from_fn(|_| vec![0.0; l]);
            let mut maxima: [Vec<f64>; R] = std::array::from_fn(|_| vec![0.0; groups]);
            for i in (1..=nrows).step_by(R) {
                let rs: [RowScalars; R] = std::array::from_fn(|t| cell.rows[i + t]);
                let h = route_rows::<R>(
                    fast,
                    &mut qt,
                    &cell.cols,
                    i + cell.base,
                    &rs,
                    &mut outs,
                    &mut maxima,
                );
                for t in 0..R {
                    same64(what, i + t, &outs[t], &maxima[t], h[t])?;
                }
            }
            Ok(qt)
        }
        let swept = [
            (
                NAMES[F64X2],
                sweep::<2>(cell, nrows, NAMES[F64X2], false, &same64)?,
            ),
            (
                NAMES[F64X4],
                sweep::<4>(cell, nrows, NAMES[F64X4], false, &same64)?,
            ),
            (
                NAMES[FASTX2],
                sweep::<2>(cell, nrows, NAMES[FASTX2], fast_applies, &same64)?,
            ),
        ];
        for (what, qt) in swept {
            if qt != qt_p {
                return Err(format!("{}: {what}: qt differs from production", cell.name));
            }
        }
        // f32 x1 against the scalar f32 reference (bits) and the production estimate (bound).
        let mut qt = cell.qt0.clone();
        let mut qs = cell.qt0.clone();
        let (mut o32, mut m32) = (vec![0.0f32; l], vec![0.0f32; groups]);
        for i in 1..=nrows {
            let at = i + cell.base;
            let row = cell.rows[i];
            let h = row32(&mut qt, c, at, row, &mut o32, &mut m32);
            let ri32 = row.ri as f32;
            let mut want_mask = 0u64;
            for lane in 0..l {
                let col = at + lane;
                let q = qs[lane]
                    .wrapping_add(i64::from(row.me).wrapping_mul(i64::from(c.enter[col])))
                    .wrapping_sub(i64::from(row.ml).wrapping_mul(i64::from(c.leave[col])));
                qs[lane] = q;
                let x = q.wrapping_sub(i64::from(row.si).wrapping_mul(i64::from(c.sum[col])));
                let want = (x as f64) as f32 * ri32 * c.r32[col] + c.bias32[col];
                if o32[lane].to_bits() != want.to_bits() {
                    return Err(format!(
                        "{}: f32 x1 row {i} lane {lane}: {:e} != scalar f32 {want:e}",
                        cell.name, o32[lane]
                    ));
                }
                if want >= c.floor32[col] {
                    want_mask |= 1 << (lane / GROUP_LANES);
                }
                let e64 = est[i][lane];
                let ok = if e64 == f64::NEG_INFINITY {
                    want == f32::NEG_INFINITY
                } else {
                    (f64::from(want) - e64).abs() <= f32_bound(x, row, c.r[col], e64)
                };
                if !ok {
                    return Err(format!(
                        "{}: f32 x1 row {i} lane {lane}: {want:e} not within bound of f64 {e64:e}",
                        cell.name
                    ));
                }
            }
            for g in 0..groups {
                let want = o32[g * GROUP_LANES..(g + 1) * GROUP_LANES]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if m32[g] != want {
                    return Err(format!(
                        "{}: f32 x1 row {i}: group {g} maximum differs",
                        cell.name
                    ));
                }
            }
            if h != want_mask {
                return Err(format!(
                    "{}: f32 x1 row {i}: hit mask {h:#x} != {want_mask:#x}",
                    cell.name
                ));
            }
        }
        if qt != qt_p {
            return Err(format!("{}: f32 x1: qt differs from production", cell.name));
        }
        Ok(fast_applies)
    }

    /// The random-input check over several lane counts, on full-range lane states and on states
    /// within the fast conversion's range; returns how many cases passed.
    pub fn check_random() -> Result<usize, String> {
        let mut cases = 0;
        for (seed, lanes) in [(1, 64), (2, 192), (3, 256), (4, 1024), (5, 128), (6, 320)] {
            assert!(lanes <= SPAN_LANES);
            for in_range in [false, true] {
                let fast_checked = check_cell(&random_cell(seed, lanes, 16, in_range), 16)?;
                if in_range && !fast_checked {
                    return Err(format!(
                        "random seed {seed}: a cross term left the fast range"
                    ));
                }
                cases += 1;
            }
        }
        Ok(cases)
    }

    /// The fast conversion against `as f64`, bit for bit: the range's edges, zero, one, and
    /// random values of every magnitude inside it; and the first values outside it must differ,
    /// so the range is the proved one and no wider. Returns how many values were compared.
    pub fn check_fast_conversion() -> Result<usize, String> {
        let edge = (1i64 << 51) - 1;
        let mut inside = vec![
            0,
            1,
            -1,
            2,
            -2,
            edge,
            -edge,
            edge - 1,
            -edge + 1,
            -(1 << 51),
        ];
        let mut rng = Rng(0xFA57);
        for k in 0..4096 {
            inside.push(rng.range(-(1 << 51), edge));
            let bits = k % 52;
            inside.push(rng.range(-(1 << bits), (1 << bits) - 1));
        }
        while !inside.len().is_multiple_of(4) {
            inside.push(0);
        }
        for chunk in inside.chunks(4) {
            let x: [i64; 4] = chunk.try_into().expect("four lanes");
            for (v, f) in x.iter().zip(body::convert_fast(x)) {
                if f.to_bits() != (*v as f64).to_bits() {
                    return Err(format!(
                        "fast conversion of {v}: {f:e} ({:#x}) != as f64 ({:#x})",
                        f.to_bits(),
                        (*v as f64).to_bits()
                    ));
                }
            }
        }
        let outside = [(1i64 << 51) + 1, -(1 << 51) - 1, i64::MAX, i64::MIN];
        for (v, f) in outside.iter().zip(body::convert_fast(outside)) {
            if f.to_bits() == (*v as f64).to_bits() {
                return Err(format!(
                    "fast conversion of {v}, outside its range, equals as f64: the range is not pinned"
                ));
            }
        }
        Ok(inside.len() + outside.len())
    }

    /// The values the full-range conversion is pinned on: zero and one, the 32-bit split's
    /// boundaries, both ends of `2^51`, `2^53` and `i64`, and every halfway case (and its
    /// neighbors) around each power of two above `2^53`, where rounding to nearest, ties to
    /// even, differs from truncation and from ties away.
    fn full_range_edges() -> Vec<i64> {
        let mut values = Vec::new();
        let mut both = |v: i64| {
            values.push(v);
            values.push(v.wrapping_neg());
        };
        for v in [
            0,
            1,
            2,
            3,
            i64::MAX,
            i64::MAX - 1,
            (1 << 62) + 1,
            u32::MAX as i64,
        ] {
            both(v);
        }
        for p in [31, 32, 51, 52, 53] {
            for d in -1..=1 {
                both((1i64 << p) + d);
            }
        }
        for p in 53..=62 {
            let quarter = (1i64 << (p - 52)) / 4;
            for k in -8..=8 {
                for d in -1..=1 {
                    both((1i64 << p) + k * quarter.max(1) + d);
                }
            }
        }
        values.extend([i64::MIN, i64::MIN + 1, i64::MIN + 2]);
        values
    }

    /// The full-range conversion against `as f64`, bit for bit, on every value it is pinned on
    /// and on random values of every magnitude. Returns how many values were compared.
    pub fn check_full_conversion() -> Result<usize, String> {
        let mut values = full_range_edges();
        let mut rng = Rng(0xF011);
        for k in 0..8192 {
            values.push((rng.next() as i64) >> (k % 64));
        }
        while !values.len().is_multiple_of(4) {
            values.push(0);
        }
        for chunk in values.chunks(4) {
            let x: [i64; 4] = chunk.try_into().expect("four lanes");
            for (v, f) in x.iter().zip(body::convert_full(x)) {
                if f.to_bits() != (*v as f64).to_bits() {
                    return Err(format!(
                        "full-range conversion of {v}: {f:e} ({:#x}) != as f64 ({:#x})",
                        f.to_bits(),
                        (*v as f64).to_bits()
                    ));
                }
            }
        }
        Ok(values.len())
    }

    // ---- counters --------------------------------------------------------------------------

    #[derive(Clone, Copy, Default)]
    struct Reading {
        instr: Option<u64>,
        cycles: Option<u64>,
        tsc: Option<u64>,
    }

    #[cfg(target_os = "macos")]
    mod counters {
        use super::Reading;

        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
            fn getpid() -> i32;
        }

        pub struct Counters;

        impl Counters {
            pub fn open() -> (Self, String) {
                let c = Counters;
                let ok = c.read().instr.is_some();
                (
                    c,
                    if ok {
                        "proc_pid_rusage (instructions, cycles)".into()
                    } else {
                        "none".into()
                    },
                )
            }

            pub fn read(&self) -> Reading {
                // rusage_info_v4: a 16-byte uuid, then u64 fields; ri_instructions is the 30th
                // and ri_cycles the 31st of them.
                let mut buf = [0u64; 64];
                // SAFETY: the buffer is larger than rusage_info_v4 and aligned for u64.
                let status = unsafe { proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()) };
                if status != 0 {
                    return Reading::default();
                }
                Reading {
                    instr: Some(buf[2 + 29]),
                    cycles: Some(buf[2 + 30]),
                    tsc: None,
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    mod counters {
        use super::Reading;

        unsafe extern "C" {
            fn syscall(number: i64, ...) -> i64;
            fn read(fd: i32, buf: *mut core::ffi::c_void, count: usize) -> isize;
        }

        #[cfg(target_arch = "x86_64")]
        const SYS_PERF_EVENT_OPEN: i64 = 298;
        #[cfg(target_arch = "aarch64")]
        const SYS_PERF_EVENT_OPEN: i64 = 241;

        pub struct Counters {
            instr: Option<i32>,
            cycles: Option<i32>,
        }

        fn open_one(config: u64) -> Option<i32> {
            // perf_event_attr, version 0 (64 bytes): type 0 (hardware), size 64, the config, and
            // the flags word with exclude_kernel (bit 5) and exclude_hv (bit 6) set, counting
            // from open.
            let mut attr = [0u64; 8];
            attr[0] = 64 << 32;
            attr[1] = config;
            attr[5] = (1 << 5) | (1 << 6);
            // SAFETY: perf_event_open reads the 64-byte attr the pointer names.
            let fd =
                unsafe { syscall(SYS_PERF_EVENT_OPEN, attr.as_ptr(), 0i32, -1i32, -1i32, 8u64) };
            (fd >= 0).then_some(fd as i32)
        }

        fn value(fd: Option<i32>) -> Option<u64> {
            let mut v = 0u64;
            // SAFETY: reads at most eight bytes into `v`.
            let n = unsafe { read(fd?, (&raw mut v).cast::<core::ffi::c_void>(), 8) };
            (n == 8).then_some(v)
        }

        impl Counters {
            pub fn open() -> (Self, String) {
                let c = Counters {
                    instr: open_one(1),
                    cycles: open_one(0),
                };
                let kind = if c.instr.is_some() {
                    "perf_event (instructions, cycles)".to_string()
                } else {
                    "none (perf_event_open refused: no PMU in the guest or perf_event_paranoid)"
                        .to_string()
                };
                (c, kind)
            }

            pub fn read(&self) -> Reading {
                Reading {
                    instr: value(self.instr),
                    cycles: value(self.cycles),
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: rdtsc reads the time stamp counter; it has no preconditions.
                    tsc: Some(unsafe { core::arch::x86_64::_rdtsc() }),
                    #[cfg(not(target_arch = "x86_64"))]
                    tsc: None,
                }
            }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    mod counters {
        use super::Reading;

        pub struct Counters;

        impl Counters {
            pub fn open() -> (Self, String) {
                (Counters, "none".into())
            }

            pub fn read(&self) -> Reading {
                Reading::default()
            }
        }
    }

    use counters::Counters;

    // ---- timing ----------------------------------------------------------------------------

    struct Work {
        qt: Vec<i64>,
        o1: Vec<f64>,
        m1: Vec<f64>,
        o2: [Vec<f64>; 2],
        m2: [Vec<f64>; 2],
        o4: [Vec<f64>; 4],
        m4: [Vec<f64>; 4],
        o32: Vec<f32>,
        m32: Vec<f32>,
    }

    impl Work {
        fn new(lanes: usize) -> Self {
            let t = || vec![0.0f64; lanes];
            let g = || vec![0.0f64; lanes / GROUP_LANES];
            Work {
                qt: vec![0; lanes],
                o1: t(),
                m1: g(),
                o2: [t(), t()],
                m2: [g(), g()],
                o4: [t(), t(), t(), t()],
                m4: [g(), g(), g(), g()],
                o32: vec![0.0; lanes],
                m32: vec![0.0; lanes / GROUP_LANES],
            }
        }
    }

    /// Rows `1..=rows` through `R`-row passes; returns the last row's hit masks folded together.
    fn sweep_rows<const R: usize, const CONV: u8>(
        cell: &Cell,
        qt: &mut [i64],
        outs: &mut [Vec<f64>; R],
        maxima: &mut [Vec<f64>; R],
        nrows: usize,
    ) -> u64 {
        let mut acc = 0u64;
        for i in (1..=nrows).step_by(R) {
            let rs: [RowScalars; R] = std::array::from_fn(|t| cell.rows[i + t]);
            acc ^= rows::<R, CONV>(qt, &cell.cols, i + cell.base, &rs, outs, maxima)[R - 1];
        }
        acc
    }

    /// Rows `1..=rows` through one strategy; returns the hit masks folded together.
    fn sweep(variant: usize, cell: &Cell, w: &mut Work, rows: usize) -> u64 {
        let (c, base, l) = (&cell.cols, cell.base, cell.lanes);
        let mut acc = 0u64;
        match variant {
            F64X1 => {
                for i in 1..=rows {
                    let row = cell.rows[i];
                    acc ^=
                        rank1::<{ body::PROD }>(&mut w.qt, c, i + base, row, &mut w.o1, &mut w.m1);
                }
            }
            OLDX1 => {
                for i in 1..=rows {
                    let row = cell.rows[i];
                    acc ^=
                        rank1::<{ body::OLD }>(&mut w.qt, c, i + base, row, &mut w.o1, &mut w.m1);
                }
            }
            FASTX1 => {
                let fires = black_box(cell.route_fires);
                for i in 1..=rows {
                    acc ^= route1(
                        fires,
                        &mut w.qt,
                        c,
                        i + base,
                        cell.rows[i],
                        &mut w.o1,
                        &mut w.m1,
                    );
                }
            }
            F32X1 => {
                for i in 1..=rows {
                    acc ^= row32(&mut w.qt, c, i + base, cell.rows[i], &mut w.o32, &mut w.m32);
                }
            }
            F64X2 => {
                acc = sweep_rows::<2, { body::PROD }>(cell, &mut w.qt, &mut w.o2, &mut w.m2, rows)
            }
            FASTX2 => {
                let fires = black_box(cell.route_fires);
                for i in (1..=rows).step_by(2) {
                    let rs = [cell.rows[i], cell.rows[i + 1]];
                    acc ^=
                        route_rows::<2>(fires, &mut w.qt, c, i + base, &rs, &mut w.o2, &mut w.m2)
                            [1];
                }
            }
            F64X4 => {
                acc = sweep_rows::<4, { body::PROD }>(cell, &mut w.qt, &mut w.o4, &mut w.m4, rows)
            }
            _ => {
                for i in 1..=rows {
                    let at = i + base;
                    acc ^= rank_row_groups(
                        &mut w.qt,
                        &c.enter[at..at + l],
                        &c.leave[at..at + l],
                        &c.sum[at..at + l],
                        &c.r[at..at + l],
                        &c.bias[at..at + l],
                        cell.rows[i],
                        &mut w.o1,
                        Some(&c.floor[at..at + l]),
                        &mut w.m1,
                    );
                }
            }
        }
        acc
    }

    #[derive(Clone, Copy)]
    struct Sample {
        secs: f64,
        instr: Option<f64>,
        cycles: Option<f64>,
        tsc: Option<f64>,
    }

    fn measure(
        variant: usize,
        cell: &Cell,
        w: &mut Work,
        rows: usize,
        counters: &Counters,
    ) -> Sample {
        w.qt.copy_from_slice(&cell.qt0);
        let before = counters.read();
        let start = Instant::now();
        let hits = sweep(variant, cell, w, rows);
        let secs = start.elapsed().as_secs_f64();
        let after = counters.read();
        black_box(hits);
        black_box(&mut *w);
        let delta = |a: Option<u64>, b: Option<u64>| Some(b?.checked_sub(a?)? as f64);
        Sample {
            secs,
            instr: delta(before.instr, after.instr),
            cycles: delta(before.cycles, after.cycles),
            tsc: delta(before.tsc, after.tsc),
        }
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn quartile_spread(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() * 3 / 4] - s[s.len() / 4]) / s[s.len() / 2]
    }

    #[derive(Clone, Default)]
    pub struct Cost {
        ns: f64,
        ns_min: f64,
        spread: f64,
        instr: Option<f64>,
        cycles: Option<f64>,
        tsc: Option<f64>,
    }

    pub struct CellResult {
        pub dataset: String,
        pub m: usize,
        /// Whether every cross term of this timing cell lies within the fast conversion's range,
        /// so its fast variants computed the production values.
        pub cell_fits: bool,
        /// Whether the fast-conversion route took this cell (its fast body ran) in the timing.
        pub route_fires: bool,
        cost: [Cost; VARIANTS],
    }

    impl CellResult {
        /// The gain of `variant` over the production function.
        fn gain(&self, variant: usize) -> f64 {
            1.0 - self.cost[variant].ns / self.cost[PROD].ns
        }
    }

    fn time_cell(
        cell: &Cell,
        rows: usize,
        rounds: usize,
        counters: &Counters,
    ) -> Result<CellResult, String> {
        let mut w = Work::new(cell.lanes);
        // Warm-up, and the proof every strategy ends a sweep on the same lane state.
        let mut first: Option<Vec<i64>> = None;
        for variant in 0..VARIANTS {
            for _ in 0..2 {
                measure(variant, cell, &mut w, rows, counters);
            }
            match &first {
                None => first = Some(w.qt.clone()),
                Some(q) if *q != w.qt => {
                    return Err(format!(
                        "{}: {} ends on a different qt",
                        cell.name, NAMES[variant]
                    ));
                }
                Some(_) => {}
            }
        }
        let pairs = (rows * cell.lanes) as f64;
        let mut samples: [Vec<Sample>; VARIANTS] = Default::default();
        for round in 0..rounds {
            for k in 0..VARIANTS {
                let variant = (round + k) % VARIANTS;
                samples[variant].push(measure(variant, cell, &mut w, rows, counters));
            }
        }
        let cost = std::array::from_fn(|v| {
            let s = &samples[v];
            let per = |f: fn(&Sample) -> Option<f64>| {
                let mut x: Vec<f64> = s.iter().filter_map(f).collect();
                (!x.is_empty()).then(|| median(&mut x) / pairs)
            };
            let mut secs: Vec<f64> = s.iter().map(|x| x.secs).collect();
            let spread = quartile_spread(&secs);
            let ns_min = secs.iter().copied().fold(f64::INFINITY, f64::min) * 1e9 / pairs;
            Cost {
                ns: median(&mut secs) * 1e9 / pairs,
                ns_min,
                spread,
                instr: per(|x| x.instr),
                cycles: per(|x| x.cycles),
                tsc: per(|x| x.tsc),
            }
        });
        Ok(CellResult {
            dataset: cell.dataset.clone(),
            m: cell.m,
            cell_fits: cell.max_v < FAST_RANGE,
            route_fires: cell.route_fires,
            cost,
        })
    }

    // ---- range report ----------------------------------------------------------------------

    /// The largest cross term one series allows at one window length, at the default scale.
    pub struct RangeRow {
        pub name: String,
        pub m: usize,
        pub n: usize,
        pub scale: Option<f64>,
        /// The largest `V = m * Q - S^2` over every window of the ticks the kernel ranks as
        /// quantized (gaps forward-filled, leading gaps at tick 0): a bound on every cross term
        /// by Cauchy-Schwarz, attained at lag 0. `None` past `2^127`.
        pub max_v: Option<u128>,
        /// The same once the cold route stands in for the outlier ticks.
        pub max_v_stood_in: Option<u128>,
        /// The same over the clean windows only (no missing sample).
        pub max_v_clean: Option<u128>,
        /// `m^2 * R^2 / 4`, `R` the widest window's tick range: the cheaper bound a column
        /// builder knowing each window's tick range would give.
        pub tick_range_bound: Option<u128>,
    }

    fn below(v: Option<u128>, limit: u128) -> bool {
        v.is_some_and(|v| v < limit)
    }

    impl RangeRow {
        /// Where the fast route fires: "yes" when the series as quantized fits `2^51`, "cold
        /// route only" when only the stood-in ticks do, "no" otherwise.
        fn fires(&self) -> &'static str {
            if below(self.max_v, FAST_RANGE) {
                "yes"
            } else if below(self.max_v_stood_in, FAST_RANGE) {
                "cold route only"
            } else {
                "no"
            }
        }
    }

    /// Whether the fast route fires on `dataset` at window length `m` (no row: it does not).
    fn fires(ranges: &[RangeRow], dataset: &str, m: usize) -> bool {
        ranges
            .iter()
            .any(|r| r.name == dataset && r.m == m && r.fires() == "yes")
    }

    /// The largest variance numerator over the windows of `ticks`; `None` on overflow.
    fn max_variance(ticks: &[i64], m: usize, keep: impl Fn(usize) -> bool) -> Option<u128> {
        let mut s = 0i128;
        let mut q = 0i128;
        let mut best = 0u128;
        for (p, &t) in ticks.iter().enumerate() {
            let t = i128::from(t);
            s += t;
            q = q.checked_add(t.checked_mul(t)?)?;
            if p >= m {
                let old = i128::from(ticks[p - m]);
                s -= old;
                q -= old * old;
            }
            if p + 1 >= m && keep(p + 1 - m) {
                let v = (m as i128).checked_mul(q)?.checked_sub(s.checked_mul(s)?)?;
                best = best.max(v.unsigned_abs());
            }
        }
        Some(best)
    }

    /// `m^2 * R^2 / 4` for the widest window's tick range `R`; `None` on overflow.
    fn tick_range_bound(ticks: &[i64], m: usize) -> Option<u128> {
        let mut high = std::collections::VecDeque::<usize>::new();
        let mut low = std::collections::VecDeque::<usize>::new();
        let mut widest = 0u128;
        for p in 0..ticks.len() {
            while high.back().is_some_and(|&b| ticks[b] <= ticks[p]) {
                high.pop_back();
            }
            while low.back().is_some_and(|&b| ticks[b] >= ticks[p]) {
                low.pop_back();
            }
            high.push_back(p);
            low.push_back(p);
            if high[0] + m <= p {
                high.pop_front();
            }
            if low[0] + m <= p {
                low.pop_front();
            }
            if p + 1 >= m {
                widest = widest.max(u128::from(ticks[high[0]].abs_diff(ticks[low[0]])));
            }
        }
        let m = m as u128;
        (m * m)
            .checked_mul(widest.checked_mul(widest)?)
            .map(|v| v / 4)
    }

    /// One series at one window length: its default scale, ticks as the kernel ranks them, and
    /// the bounds of [`RangeRow`].
    pub fn range_row(name: &str, x: &[f64], m: usize) -> RangeRow {
        let mut row = RangeRow {
            name: name.to_string(),
            m,
            n: x.len(),
            scale: None,
            max_v: None,
            max_v_stood_in: None,
            max_v_clean: None,
            tick_range_bound: None,
        };
        let Some(scale) = NonZeroUsize::new(m).and_then(|mz| Scale::infer(x, mz)) else {
            return row;
        };
        row.scale = Some(scale.get());
        let mut fill = ForwardFill::default();
        let (ticks, had): (Vec<i64>, Vec<bool>) = x
            .iter()
            .map(|&v| match fill.push(to_ticks(v, scale)) {
                Sanitized::Present(t) => (t, true),
                Sanitized::Filled(t) => (t, false),
                Sanitized::Leading => (0, false),
            })
            .unzip();
        let mut missing = vec![0usize; ticks.len() + 1];
        for p in 0..ticks.len() {
            missing[p + 1] = missing[p] + usize::from(!had[p]);
        }
        row.max_v = max_variance(&ticks, m, |_| true);
        row.max_v_clean = max_variance(&ticks, m, |w| missing[w + m] == missing[w]);
        row.tick_range_bound = tick_range_bound(&ticks, m);
        // The cold route's stand-ins, as the batch path writes them: each outlier tick, and each
        // filled tick after the first measured one, becomes the last tick that is neither (the
        // first such tick for a leading run).
        if let Ok(outliers) = outlier_ticks(&ticks, &had, m) {
            let first = (0..ticks.len())
                .find(|&p| had[p] && !outliers[p])
                .map(|p| ticks[p]);
            let (mut last, mut seen) = (None, false);
            let stood: Vec<i64> = (0..ticks.len())
                .map(|p| {
                    if had[p] && !outliers[p] {
                        last = Some(ticks[p]);
                    }
                    let stand_in = last.or(first).unwrap_or(0);
                    let t = if outliers[p] || (!had[p] && seen) {
                        stand_in
                    } else {
                        ticks[p]
                    };
                    seen |= had[p];
                    t
                })
                .collect();
            row.max_v_stood_in = max_variance(&stood, m, |_| true);
        }
        row
    }

    /// Reads a raw little-endian `f64` series, its first `n` samples at most.
    fn read_series(path: &str, n: usize) -> Result<Vec<f64>, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Ok(bytes
            .chunks_exact(8)
            .take(n)
            .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
            .collect())
    }

    /// The range report: every dataset of the gate at `n` and the gate's window lengths, then
    /// every extra series.
    pub fn range_report(n: usize, files: &[(String, String)]) -> Result<Vec<RangeRow>, String> {
        let mut rows = Vec::new();
        for dataset in DATASETS {
            let x = series(dataset, n, seed_of(dataset, n));
            for m in MS {
                rows.push(range_row(dataset, &x, m));
            }
        }
        for (name, path) in files {
            let x = read_series(path, n)?;
            for m in MS {
                if m <= x.len() {
                    rows.push(range_row(name, &x, m));
                }
            }
        }
        Ok(rows)
    }

    fn log2_text(v: Option<u128>) -> String {
        match v {
            None => "overflow".into(),
            Some(0) => "0".into(),
            Some(v) => format!("2^{:.2}", (v as f64).log2()),
        }
    }

    pub fn range_table(rows: &[RangeRow]) -> String {
        let mut t = String::new();
        writeln!(
            t,
            "range report at the default scale (Scale::infer): max V over every window bounds every cross term (|N| <= sqrt(Vi*Vj)) and is attained at lag 0"
        )
        .unwrap();
        writeln!(
            t,
            "{:<14} {:>5} {:>7} {:>11} | {:>10} {:>10} {:>10} {:>12} | {:>6} {:>6} | fast route fires",
            "series", "m", "n", "scale", "max V", "stood in", "clean", "m^2 R^2/4", "<2^51", "<2^53"
        )
        .unwrap();
        for r in rows {
            writeln!(
                t,
                "{:<14} {:>5} {:>7} {:>11} | {:>10} {:>10} {:>10} {:>12} | {:>6} {:>6} | {}",
                r.name,
                r.m,
                r.n,
                r.scale.map_or("none".into(), |s| format!("{s:.3e}")),
                log2_text(r.max_v),
                log2_text(r.max_v_stood_in),
                log2_text(r.max_v_clean),
                log2_text(r.tick_range_bound),
                if below(r.max_v, FAST_RANGE) {
                    "yes"
                } else {
                    "no"
                },
                if below(r.max_v, EXACT_F64) {
                    "yes"
                } else {
                    "no"
                },
                r.fires()
            )
            .unwrap();
        }
        t
    }

    // ---- decision --------------------------------------------------------------------------

    pub struct Verdict {
        pub variant: usize,
        /// A per-series route (judged where it fires) rather than a whole body.
        pub route: bool,
        pub candidate: bool,
        pub qualifies: bool,
        pub grid_m64: Option<f64>,
        pub float_m64: Option<f64>,
        pub worst_ties_messy: Option<f64>,
        /// For a route: its smallest gain over the m64 cells where it fires.
        pub fired_m64_min: Option<f64>,
        /// For a route: its smallest gain over the cells where it does not fire.
        pub unfired_worst: Option<f64>,
        pub fires_on: Option<String>,
        pub reason: String,
    }

    fn gain_at(results: &[CellResult], dataset: &str, m: usize, variant: usize) -> Option<f64> {
        results
            .iter()
            .find(|r| r.dataset == dataset && r.m == m)
            .map(|r| r.gain(variant))
    }

    fn verdict_text(candidate: bool, why: &[String]) -> String {
        if !candidate {
            "a control on this target, not a candidate".into()
        } else if why.is_empty() {
            "meets the rule".into()
        } else {
            why.join("; ")
        }
    }

    /// A whole-body strategy: at least 15% over production on grid m64 and float m64, and not
    /// more than 2% slower on any ties or messy cell.
    fn body_verdict(results: &[CellResult], variant: usize) -> Verdict {
        let grid = gain_at(results, "grid", 64, variant);
        let float = gain_at(results, "float", 64, variant);
        let worst = results
            .iter()
            .filter(|r| r.dataset == "ties" || r.dataset == "messy")
            .map(|r| r.gain(variant))
            .reduce(f64::min);
        let mut why = Vec::new();
        match (grid, float) {
            (Some(g), Some(f)) => {
                if g < MIN_GAIN || f < MIN_GAIN {
                    why.push(format!(
                        "gain under {:.0}% on grid m64 ({:+.1}%) or float m64 ({:+.1}%)",
                        MIN_GAIN * 100.0,
                        g * 100.0,
                        f * 100.0
                    ));
                }
            }
            _ => why.push("no grid m64 or float m64 cell run".into()),
        }
        if let Some(w) = worst.filter(|&w| w < -NOISE_MARGIN) {
            why.push(format!(
                "slower than production on a ties or messy cell ({:+.1}%)",
                w * 100.0
            ));
        }
        Verdict {
            variant,
            route: false,
            candidate: true,
            qualifies: why.is_empty(),
            grid_m64: grid,
            float_m64: float,
            worst_ties_messy: worst,
            fired_m64_min: None,
            unfired_worst: None,
            fires_on: None,
            reason: verdict_text(true, &why),
        }
    }

    /// A per-series route, judged where it fires: at least 15% over production on every m64 cell
    /// where it fires, which must include grid and at least one of messy or ties; not more than
    /// 2% slower than production on any cell where it does not fire (timed through the route,
    /// its fallback included); equal bits are the run's precondition (`check_cell`).
    fn route_verdict(results: &[CellResult], variant: usize) -> Verdict {
        let fired = |d: &str| {
            results
                .iter()
                .any(|r| r.dataset == d && r.m == 64 && r.route_fires)
        };
        let fired_m64_min = results
            .iter()
            .filter(|r| r.m == 64 && r.route_fires)
            .map(|r| r.gain(variant))
            .reduce(f64::min);
        let unfired_worst = results
            .iter()
            .filter(|r| !r.route_fires)
            .map(|r| r.gain(variant))
            .reduce(f64::min);
        let mut why = Vec::new();
        if !fired("grid") || !(fired("messy") || fired("ties")) {
            why.push("does not fire on grid m64 and on messy or ties m64".to_string());
        }
        if let Some(g) = fired_m64_min.filter(|&g| g < MIN_GAIN) {
            why.push(format!(
                "gain under {:.0}% on an m64 cell where it fires ({:+.1}%)",
                MIN_GAIN * 100.0,
                g * 100.0
            ));
        }
        if let Some(w) = unfired_worst.filter(|&w| w < -NOISE_MARGIN) {
            why.push(format!(
                "slower than production on a cell where it does not fire ({:+.1}%)",
                w * 100.0
            ));
        }
        let on: Vec<String> = results
            .iter()
            .filter(|r| r.route_fires)
            .map(|r| format!("{} m{}", r.dataset, r.m))
            .collect();
        let candidate = body::FAST_IS_CANDIDATE;
        Verdict {
            variant,
            route: true,
            candidate,
            qualifies: candidate && why.is_empty(),
            grid_m64: gain_at(results, "grid", 64, variant),
            float_m64: gain_at(results, "float", 64, variant),
            worst_ties_messy: None,
            fired_m64_min,
            unfired_worst,
            fires_on: Some(if on.is_empty() {
                "none".into()
            } else {
                on.join(", ")
            }),
            reason: verdict_text(candidate, &why),
        }
    }

    /// The gate's decisions: each whole body and each route against production; the body to keep
    /// (the fastest on the m64 cells of the bodies that qualify, else the 1-row f64 body); the
    /// route on top of it, if the route for that body qualifies (fast x1 on the 1-row f64 body,
    /// fast x2 on the 2-row sweep; none on the f32 body); and whether four rows spill.
    pub fn decide(results: &[CellResult]) -> (Vec<Verdict>, String, String) {
        let verdicts = vec![
            body_verdict(results, F32X1),
            body_verdict(results, F64X2),
            route_verdict(results, FASTX1),
            route_verdict(results, FASTX2),
        ];
        let m64 = |v: &Verdict| {
            v.grid_m64.unwrap_or(f64::NEG_INFINITY) + v.float_m64.unwrap_or(f64::NEG_INFINITY)
        };
        let body = verdicts
            .iter()
            .filter(|v| !v.route && v.qualifies)
            .max_by(|a, b| m64(a).total_cmp(&m64(b)))
            .map(|v| v.variant);
        let route_for = match body {
            None => Some(FASTX1),
            Some(F64X2) => Some(FASTX2),
            Some(_) => None,
        };
        let route = route_for.filter(|&r| verdicts.iter().any(|v| v.variant == r && v.qualifies));
        let label = |v: usize| format!("{} ({})", NAMES[v], NODES[v]);
        let dropped: Vec<String> = [F32X1, F64X2]
            .into_iter()
            .filter(|&v| Some(v) != body)
            .map(label)
            .collect();
        let mut adopt = match body {
            None => "keep the 1-row f64 body".to_string(),
            Some(b) => format!("adopt {}", label(b)),
        };
        match route {
            Some(r) => write!(
                adopt,
                "; adopt the fast-conversion route on series whose range fits ({})",
                NAMES[r]
            )
            .unwrap(),
            None => adopt.push_str("; no fast-conversion route"),
        }
        write!(adopt, "; drop {} on this target", dropped.join(", ")).unwrap();
        let spill = {
            let n = results.len() as f64;
            let slower_than_two = results
                .iter()
                .filter(|r| r.cost[F64X4].ns > r.cost[F64X2].ns * 1.02)
                .count() as f64;
            let slower_than_one = results
                .iter()
                .filter(|r| r.cost[F64X4].ns > r.cost[F64X1].ns * 1.02)
                .count() as f64;
            format!(
                "4 rows slower than 2 rows in {slower_than_two:.0} of {n:.0} cells, slower than 1 row in {slower_than_one:.0}: {}",
                if slower_than_one > n / 2.0 {
                    "spills badly"
                } else if slower_than_two > n / 2.0 {
                    "two rows is the ceiling"
                } else {
                    "no spill penalty seen"
                }
            )
        };
        (verdicts, adopt, spill)
    }

    /// The old production conversion against today's, for reference only: `old x1` is the 1-row
    /// copy with every term converted by the emulated body full-range-convert replaced, `prod x1`
    /// the production kernel with the full-range conversion on every row. Nothing is decided.
    /// Returns its smallest m64 gain and the text.
    pub fn old_conversion_report(results: &[CellResult]) -> (Option<f64>, String) {
        let m64: Vec<(String, f64)> = results
            .iter()
            .filter(|r| r.m == 64)
            .map(|r| (r.dataset.clone(), r.gain(OLDX1)))
            .collect();
        let worst = m64.iter().map(|&(_, g)| g).reduce(f64::min);
        let worst_any = results.iter().map(|r| r.gain(OLDX1)).reduce(f64::min);
        let gains: Vec<String> = m64
            .iter()
            .map(|(d, g)| format!("{d} {:+.1}%", g * 100.0))
            .collect();
        let status = if body::FAST_IS_CANDIDATE {
            "reference (the old production conversion, emulated)"
        } else {
            "control (native conversion on this target, then and now)"
        };
        let text = format!(
            "{}: {status} (m64 gains over prod x1: {}; worst m64 {}, worst any cell {})",
            NAMES[OLDX1],
            if gains.is_empty() {
                "none run".to_string()
            } else {
                gains.join(", ")
            },
            percent(worst),
            percent(worst_any),
        );
        (worst, text)
    }

    /// The median ratio of the 1-row copy's time to production's, and whether it is faithful.
    fn faithfulness(results: &[CellResult]) -> (f64, bool) {
        let mut ratios: Vec<f64> = results
            .iter()
            .map(|r| r.cost[F64X1].ns / r.cost[PROD].ns)
            .collect();
        let ratio = median(&mut ratios);
        (ratio, (ratio - 1.0).abs() <= FAITHFUL_MARGIN)
    }

    // ---- reporting -------------------------------------------------------------------------

    fn machine_text(cmd: &str, args: &[&str]) -> String {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }

    fn cpu_model() -> String {
        if cfg!(target_os = "macos") {
            return machine_text("sysctl", &["-n", "machdep.cpu.brand_string"]);
        }
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name") || l.starts_with("Model"))
                    .and_then(|l| l.split(':').nth(1).map(|v| v.trim().to_string()))
            })
            .unwrap_or_default()
    }

    fn load_average() -> String {
        if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
            return s.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
        }
        machine_text("sysctl", &["-n", "vm.loadavg"])
            .trim_matches(|c| c == '{' || c == '}')
            .trim()
            .to_string()
    }

    fn json_str(s: &str) -> String {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c if (c as u32) < 0x20 => out.push(' '),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn json_num(v: Option<f64>) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{x:.6}"),
            _ => "null".into(),
        }
    }

    fn json_log2(v: Option<u128>) -> String {
        match v {
            Some(0) => "null".into(),
            v => json_num(v.map(|v| (v as f64).log2())),
        }
    }

    fn percent(v: Option<f64>) -> String {
        v.map_or("n/a".into(), |x| format!("{:+.1}%", x * 100.0))
    }

    pub struct Options {
        pub quick: bool,
        pub check_only: bool,
        pub range_only: bool,
        pub range_files: Vec<(String, String)>,
        pub out: Option<String>,
        pub lanes: usize,
        pub rounds: Option<usize>,
        pub rows: Option<usize>,
    }

    pub fn parse_args() -> Result<Options, String> {
        let mut o = Options {
            quick: false,
            check_only: false,
            range_only: false,
            range_files: Vec::new(),
            out: None,
            lanes: DEFAULT_LANES,
            rounds: None,
            rows: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
            match a.as_str() {
                "--quick" => o.quick = true,
                "--check" => o.check_only = true,
                "--range" => o.range_only = true,
                "--range-file" => {
                    let v = value("--range-file")?;
                    let (name, path) = v
                        .split_once('=')
                        .ok_or("--range-file takes NAME=PATH".to_string())?;
                    o.range_files.push((name.to_string(), path.to_string()));
                }
                "--out" => o.out = Some(value("--out")?),
                "--lanes" => {
                    o.lanes = value("--lanes")?
                        .parse()
                        .map_err(|e| format!("--lanes: {e}"))?
                }
                "--rounds" => {
                    o.rounds = Some(
                        value("--rounds")?
                            .parse()
                            .map_err(|e| format!("--rounds: {e}"))?,
                    )
                }
                "--rows" => {
                    o.rows = Some(
                        value("--rows")?
                            .parse()
                            .map_err(|e| format!("--rows: {e}"))?,
                    )
                }
                "-h" | "--help" => {
                    return Err("usage: kernel_strategy [--quick] [--check] [--range] [--range-file NAME=PATH]... [--out FILE] [--lanes N] [--rounds N] [--rows N]".into());
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if !o.lanes.is_multiple_of(GROUP_LANES) || o.lanes == 0 || o.lanes > SPAN_LANES {
            return Err(format!(
                "--lanes must be a multiple of {GROUP_LANES} up to {SPAN_LANES}"
            ));
        }
        Ok(o)
    }

    /// The whole run; returns the process exit code.
    pub fn run(o: &Options) -> i32 {
        if !body::available() {
            eprintln!(
                "kernel_strategy: this CPU lacks the instruction set of the {} bodies",
                body::TARGET
            );
            return 2;
        }
        // Equal work first: nothing is timed if a variant disagrees with the production kernel.
        match check_random()
            .and_then(|n| Ok((n, check_fast_conversion()?, check_full_conversion()?)))
        {
            Ok((n, k, f)) => println!(
                "check: {n} random-input cases equal the production kernel (f64, fast and old x1, fast x2, f64 x4 bits, f32 x1 reference and bound); fast conversion equals as f64 on {k} values, full-range conversion on {f}"
            ),
            Err(e) => {
                eprintln!("CHECK FAILED: {e}");
                return 1;
            }
        }
        let ranges = match range_report(RANGE_N, &o.range_files) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        };
        let (n, ms, rows, rounds): (usize, &[usize], usize, usize) = if o.quick {
            (16_384, &[64], 2048, 5)
        } else {
            (65_536, &MS, 16_384, 21)
        };
        let rows = o.rows.unwrap_or(rows);
        let rounds = o.rounds.unwrap_or(rounds);
        assert!(rows.is_multiple_of(4), "--rows must be a multiple of 4");
        let (counters, counter_kind) = Counters::open();
        let mut results = Vec::new();
        let mut table = String::new();
        let load_before = load_average();
        let short = [
            "f64x1", "f32x1", "f64x2", "f64x4", "prod", "fastx1", "fastx2", "oldx1",
        ];
        let shown = [F64X1, F32X1, F64X2, F64X4, FASTX1, FASTX2, OLDX1];
        let mut head = format!("{:<12} |", "cell");
        for s in short {
            write!(head, " {s:>7}").unwrap();
        }
        head.push_str(" |");
        for &v in &shown {
            write!(head, " {:>7}", short[v]).unwrap();
        }
        head.push_str(" |");
        for s in short {
            write!(head, " {:>7}", format!("i {s}")).unwrap();
        }
        writeln!(table, "{head} | fits | route | noisy").unwrap();
        writeln!(
            table,
            "{:<12} | ns per pair (median of {rounds}) | gain over prod x1 | instructions per pair | cell's terms < 2^51 | fast route fires (default-scale range and cell fit)",
            ""
        )
        .unwrap();
        for dataset in DATASETS {
            for &m in ms {
                let mut cell = build_cell(dataset, n, m, o.lanes, rows);
                cell.route_fires &= fires(&ranges, dataset, m);
                if let Err(e) = check_cell(&cell, 8) {
                    eprintln!("CHECK FAILED: {e}");
                    return 1;
                }
                let r = match time_cell(&cell, rows, rounds, &counters) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("CHECK FAILED: {e}");
                        return 1;
                    }
                };
                let mut line = format!("{:<12} |", cell.name);
                for v in 0..VARIANTS {
                    write!(line, " {:>7.3}", r.cost[v].ns).unwrap();
                }
                line.push_str(" |");
                for &v in &shown {
                    write!(line, " {:>+7.1}", r.gain(v) * 100.0).unwrap();
                }
                line.push_str(" |");
                for v in 0..VARIANTS {
                    let i = r.cost[v]
                        .instr
                        .map_or("n/a".to_string(), |x| format!("{x:.2}"));
                    write!(line, " {i:>7}").unwrap();
                }
                let noisy = r.cost.iter().any(|c| c.spread > 0.05);
                writeln!(
                    table,
                    "{line} | {:>4} | {:>5} | {}",
                    if r.cell_fits { "yes" } else { "no" },
                    if r.route_fires { "yes" } else { "no" },
                    if noisy { "yes" } else { "" }
                )
                .unwrap();
                results.push(r);
            }
        }
        let load_after = load_average();
        let (ratio, faithful) = faithfulness(&results);
        let (verdicts, adopt, spill) = decide(&results);
        let (old_worst, old_text) = old_conversion_report(&results);
        println!();
        print!("{}", range_table(&ranges));
        println!();
        print!("{table}");
        println!();
        println!(
            "target {}   lanes {}   rows per sweep {rows}   rounds {rounds}   n {n}",
            body::TARGET,
            o.lanes
        );
        println!(
            "cpu: {}   load average before/after: {load_before} / {load_after}",
            cpu_model()
        );
        println!("counters: {counter_kind}");
        println!(
            "faithfulness: the 1-row copy (f64 x1) runs at {ratio:.3}x production (median over cells; void beyond {:.0}%): {}",
            FAITHFUL_MARGIN * 100.0,
            if faithful {
                "faithful"
            } else {
                "VOID: the copy is not equivalent to production, so the gains of the bodies built from it are not evidence"
            }
        );
        for v in &verdicts {
            let status = if !v.candidate {
                "control"
            } else if v.qualifies {
                "QUALIFIES"
            } else {
                "does not qualify"
            };
            if v.route {
                println!(
                    "{} (route): {status} (worst m64 where it fires {}, worst where it does not {}; fires on: {}): {}",
                    NAMES[v.variant],
                    percent(v.fired_m64_min),
                    percent(v.unfired_worst),
                    v.fires_on.as_deref().unwrap_or("none"),
                    v.reason
                );
            } else {
                println!(
                    "{} (body): {status} (grid m64 {}, float m64 {}, worst ties/messy {}): {}",
                    NAMES[v.variant],
                    percent(v.grid_m64),
                    percent(v.float_m64),
                    percent(v.worst_ties_messy),
                    v.reason
                );
            }
        }
        println!("spill check: {spill}");
        if faithful {
            println!("{old_text}");
        } else {
            println!("VOID (the 1-row copy is not within 2% of production): {old_text}");
        }
        if faithful {
            println!("DECISION for {}: {adopt}", body::TARGET);
        } else {
            println!(
                "DECISION for {}: VOID (the 1-row copy is not within {:.0}% of production); it would read: {adopt}",
                body::TARGET,
                FAITHFUL_MARGIN * 100.0
            );
        }
        if o.quick {
            println!("(quick run: a smoke test, not evidence)");
        }

        let mut j = String::from("{\n");
        let mut kv = |k: &str, v: String| {
            writeln!(j, "  {}: {v},", json_str(k)).unwrap();
        };
        kv("harness", json_str("kernel_strategy"));
        kv("target", json_str(body::TARGET));
        kv("arch", json_str(std::env::consts::ARCH));
        kv("os", json_str(std::env::consts::OS));
        kv("cpu", json_str(&cpu_model()));
        kv("load_before", json_str(&load_before));
        kv("load_after", json_str(&load_after));
        kv("counters", json_str(&counter_kind));
        kv("local_over_production_f64x1", json_num(Some(ratio)));
        kv("faithful", faithful.to_string());
        kv("quick", o.quick.to_string());
        kv("n", n.to_string());
        kv("lanes", o.lanes.to_string());
        kv("rows_per_sweep", rows.to_string());
        kv("rounds", rounds.to_string());
        let mut range = String::from("[\n");
        for (i, r) in ranges.iter().enumerate() {
            writeln!(
                range,
                "    {{\"series\": {}, \"m\": {}, \"n\": {}, \"scale\": {}, \"log2_max_v\": {}, \"log2_max_v_stood_in\": {}, \"log2_max_v_clean\": {}, \"log2_tick_range_bound\": {}, \"fits_2_51\": {}, \"fits_2_53\": {}, \"fast_route_fires\": {}}}{}",
                json_str(&r.name),
                r.m,
                r.n,
                json_num(r.scale),
                json_log2(r.max_v),
                json_log2(r.max_v_stood_in),
                json_log2(r.max_v_clean),
                json_log2(r.tick_range_bound),
                below(r.max_v, FAST_RANGE),
                below(r.max_v, EXACT_F64),
                json_str(r.fires()),
                if i + 1 == ranges.len() { "" } else { "," }
            )
            .unwrap();
        }
        range.push_str("  ]");
        kv("range", range);
        let mut cells = String::from("[\n");
        for (i, r) in results.iter().enumerate() {
            let mut variants = String::new();
            for v in 0..VARIANTS {
                let c = &r.cost[v];
                write!(
                    variants,
                    "{}{}: {{\"ns_per_pair\": {}, \"ns_per_pair_min\": {}, \"spread\": {}, \"gain_vs_prod\": {}, \"effective_gain_vs_prod\": {}, \"instructions_per_pair\": {}, \"cycles_per_pair\": {}, \"tsc_per_pair\": {}}}",
                    if v == 0 { "" } else { ", " },
                    json_str(NAMES[v]),
                    json_num(Some(c.ns)), json_num(Some(c.ns_min)), json_num(Some(c.spread)),
                    json_num(Some(r.gain(v))), json_num(Some(r.gain(v))),
                    json_num(c.instr), json_num(c.cycles), json_num(c.tsc)
                )
                .unwrap();
            }
            writeln!(
                cells,
                "    {{\"dataset\": {}, \"m\": {}, \"cell_terms_fit_2_51\": {}, \"route_fires\": {}, \"variants\": {{{variants}}}}}{}",
                json_str(&r.dataset),
                r.m,
                r.cell_fits,
                r.route_fires,
                if i + 1 == results.len() { "" } else { "," }
            )
            .unwrap();
        }
        cells.push_str("  ]");
        kv("cells", cells);
        let mut dec = String::from("{\n");
        for v in &verdicts {
            writeln!(
                dec,
                "    {}: {{\"kind\": {}, \"candidate\": {}, \"qualifies\": {}, \"grid_m64\": {}, \"float_m64\": {}, \"worst_ties_messy\": {}, \"fired_m64_min\": {}, \"unfired_worst\": {}, \"fires_on\": {}, \"reason\": {}}},",
                json_str(NAMES[v.variant]),
                json_str(if v.route { "route" } else { "body" }),
                v.candidate,
                v.qualifies,
                json_num(v.grid_m64), json_num(v.float_m64), json_num(v.worst_ties_messy),
                json_num(v.fired_m64_min), json_num(v.unfired_worst),
                v.fires_on.as_deref().map_or("null".into(), json_str),
                json_str(&v.reason)
            )
            .unwrap();
        }
        writeln!(
            dec,
            "    \"old conversion\": {{\"faithful\": {}, \"worst_m64\": {}, \"text\": {}}},",
            faithful,
            json_num(old_worst),
            json_str(&old_text)
        )
        .unwrap();
        writeln!(dec, "    \"rule\": {},", json_str("a body is adopted only if >= 15% faster than prod x1 on grid m64 and float m64 and not slower (2% margin) on ties and messy, the fastest on the m64 cells of those that qualify; a route is adopted only if >= 15% faster than prod x1 on every m64 cell where it fires (grid and messy or ties among them), not slower (2% margin) on any cell where it does not fire, and equal on bits; fast x1 rides the 1-row f64 body, fast x2 the 2-row sweep; void if the 1-row copy is not within 2% of production"))
            .unwrap();
        write!(
            dec,
            "    \"spill\": {},\n    \"void\": {},\n    \"route\": {},\n    \"adopt\": {}\n  }}",
            json_str(&spill),
            !faithful,
            json_str(adopt.split("; ").nth(1).unwrap_or("")),
            json_str(&adopt)
        )
        .unwrap();
        kv("decision", dec);
        j.push_str("  \"end\": true\n}\n");
        let path = o
            .out
            .clone()
            .unwrap_or_else(|| format!("kernel-strategy-{}.json", std::env::consts::ARCH));
        match std::fs::write(&path, j) {
            Ok(()) => println!("wrote {path}"),
            Err(e) => {
                eprintln!("could not write {path}: {e}");
                return 1;
            }
        }
        0
    }

    pub fn check_all(quick: bool) -> Result<String, String> {
        let random = check_random()?;
        let values = check_fast_conversion()?;
        let full = check_full_conversion()?;
        let ms: &[usize] = if quick { &[16] } else { &MS };
        let (mut cells, mut fast_cells) = (0, 0);
        for dataset in DATASETS {
            for &m in ms {
                fast_cells += usize::from(check_cell(&build_cell(dataset, 16_384, m, 256, 64), 8)?);
                cells += 1;
            }
        }
        Ok(format!(
            "{random} random-input cases and {cells} dataset cells ({fast_cells} with the fast variants) equal the production kernel; the fast conversion equals as f64 on {values} values and differs just outside its range; the full-range conversion equals as f64 on {full} values"
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[path = "kernel_strategy/neon.rs"]
mod body;

#[cfg(target_arch = "x86_64")]
#[path = "kernel_strategy/avx2.rs"]
mod body;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn main() {
    let options = match harness::parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if options.range_only {
        match harness::range_report(harness::RANGE_N, &options.range_files) {
            Ok(rows) => print!("{}", harness::range_table(&rows)),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        return;
    }
    if options.check_only {
        if !body::available() {
            eprintln!(
                "kernel_strategy: this CPU lacks the instruction set of the {} bodies",
                body::TARGET
            );
            std::process::exit(2);
        }
        match harness::check_all(options.quick) {
            Ok(s) => {
                println!("check ({}): {s}", body::TARGET);
                println!(
                    "faithfulness of the 1-row copy to production in time is measured by the timing run only (void beyond 2%)"
                );
            }
            Err(e) => {
                eprintln!("CHECK FAILED: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    std::process::exit(harness::run(&options));
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn main() {
    eprintln!("kernel_strategy: only aarch64 (NEON) and x86_64 (AVX2) have bodies");
}

#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
mod tests {
    #[test]
    fn variants_equal_the_production_kernel() {
        if !super::body::available() {
            return;
        }
        let summary = super::harness::check_all(true)
            .expect("a variant disagrees with the production kernel");
        assert!(summary.contains("equal the production kernel"));
    }
}
