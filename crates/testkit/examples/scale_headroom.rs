//! How many bits of range the integer-tick form uses for several market inputs, so the
//! headroom left in `i128` (for raw ticks) and in `i64` (for `i32` deviations) can be read off.

struct Input {
    name: &'static str,
    level: f64,
    scale: f64,
}

const INPUTS: &[Input] = &[
    Input {
        name: "index/future price",
        level: 1e8,
        scale: 0.01,
    },
    Input {
        name: "equity price",
        level: 100.0,
        scale: 0.0001,
    },
    Input {
        name: "FX price",
        level: 1.1,
        scale: 1e-5,
    },
    Input {
        name: "log return",
        level: 0.2,
        scale: 1e-9,
    },
    Input {
        name: "VWAP",
        level: 1e4,
        scale: 1e-6,
    },
];

/// `m` values the window-size columns are reported at: 2^6, 2^10, 2^16.
const WINDOW_LOG2: [u32; 3] = [6, 10, 16];

/// Local price range this input's window is assumed to move across, as a fraction of `level`.
const LOCAL_RANGE_FRACTION: f64 = 0.05;

fn ticks(level: f64, scale: f64) -> i128 {
    (level / scale).round() as i128
}

fn bit_length(value: i128) -> u32 {
    let magnitude = value.unsigned_abs();
    u128::BITS - magnitude.leading_zeros()
}

/// Headroom, in bits, before `value` would no longer fit in a signed integer of `capacity` bits.
fn headroom(value: i128, capacity: u32) -> i32 {
    capacity as i32 - 1 - bit_length(value) as i32
}

fn main() {
    println!("Bits used by the raw-tick form (window-start-independent worst case: Q and QT");
    println!("are both bounded by m·x², since QT(i,j) ≤ √(Q(i)·Q(j)) ≤ Q when both windows");
    println!("share a magnitude bound):\n");
    println!(
        "{:<22}{:>10}{:>18}{:>18}{:>18}",
        "input", "|x| bits", "m=2^6 m·Q bits", "m=2^10 m·Q bits", "m=2^16 m·Q bits"
    );
    for input in INPUTS {
        let x = ticks(input.level, input.scale);
        let x_bits = bit_length(x);
        let cells: Vec<String> = WINDOW_LOG2
            .iter()
            .map(|&log2m| {
                let m: i128 = 1 << log2m;
                let q_bound = m * x * x; // Q(i) worst case: every sample at the same magnitude
                let mq_bound = m * q_bound;
                format!(
                    "{} (i128 headroom {})",
                    bit_length(mq_bound),
                    headroom(mq_bound, 128)
                )
            })
            .collect();
        println!(
            "{:<22}{:>10}{:>24}{:>24}{:>24}",
            input.name, x_bits, cells[0], cells[1], cells[2]
        );
    }

    println!(
        "\ni32/i64 headroom for i64_deviations, window range {:.0}% of level (one-sided bound,\n\
         deviation from a fixed reference taken equal to the full stated range):\n",
        LOCAL_RANGE_FRACTION * 100.0
    );
    println!(
        "{:<22}{:>14}{:>20}{:>22}{:>22}",
        "input", "dev bits", "fits i32", "m=2^16 m·Qdev bits", "i64 headroom"
    );
    for input in INPUTS {
        let x = ticks(input.level, input.scale);
        let dev = (LOCAL_RANGE_FRACTION * x as f64).round() as i128;
        let dev_bits = bit_length(dev);
        let fits_i32 = headroom(dev, 32) >= 0;
        let m: i128 = 1 << 16;
        let qdev_bound = m * dev * dev;
        let mqdev_bound = m * qdev_bound;
        println!(
            "{:<22}{:>14}{:>20}{:>22}{:>22}",
            input.name,
            dev_bits,
            fits_i32,
            bit_length(mqdev_bound),
            headroom(mqdev_bound, 64)
        );
    }

    println!(
        "\nBreakeven local-range fraction of level (one-sided deviation from a fixed reference)\n\
         at which each constraint first exhausts its range. `i32 |y|` does not depend on m;\n\
         `i64 m·Qdev` is shown at the bench's window sizes and at m = 2^16:\n"
    );
    println!(
        "{:<22}{:>12}{:>14}{:>14}{:>14}{:>14}",
        "input", "i32 |y|", "i64 m=16", "i64 m=64", "i64 m=256", "i64 m=2^16"
    );
    for input in INPUTS {
        let x = ticks(input.level, input.scale) as f64;
        let i32_breakeven = 100.0 * (1i128 << 31) as f64 / x;
        // m² · dev² < 2⁶³  ⇒  dev < 2^31.5 / m  ⇒  frac < 2^31.5 / (m · x)
        let i64_breakeven = |m: f64| 100.0 * ((1i128 << 63) as f64).sqrt() / (m * x);
        println!(
            "{:<22}{:>11.4}%{:>13.6}%{:>13.6}%{:>13.6}%{:>13.6}%",
            input.name,
            i32_breakeven,
            i64_breakeven(16.0),
            i64_breakeven(64.0),
            i64_breakeven(256.0),
            i64_breakeven(65536.0),
        );
    }

    println!(
        "\nLargest per-window σ (price units) before the wrapping forms (i64_wrapping,\n\
         i32dev_wrapping) lose exactness: both need only m·Σ(x−x̄)² = m²·σ_ticks² < 2⁶³, so\n\
         σ_ticks < 2^31.5/m — unlike the two tables above, this bound does not depend on\n\
         price level, only on local volatility:\n"
    );
    println!(
        "{:<22}{:>18}{:>18}{:>18}",
        "input", "m=2^6 sigma", "m=2^10 sigma", "m=2^16 sigma"
    );
    let sigma_ticks_max = |log2m: u32| -> f64 { 2f64.powf(31.5) / 2f64.powi(log2m as i32) };
    for input in INPUTS {
        let cells: Vec<String> = WINDOW_LOG2
            .iter()
            .map(|&log2m| format!("{:.6e}", sigma_ticks_max(log2m) * input.scale))
            .collect();
        println!(
            "{:<22}{:>18}{:>18}{:>18}",
            input.name, cells[0], cells[1], cells[2]
        );
    }
}
