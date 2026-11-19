//! The exact pair arithmetic the tests that restate a procedure share: a pair's correlation as
//! the exact rational its square is, and its f64 squared distance computed as the batch computes it.
#![allow(dead_code)]

use core::cmp::Ordering;

use ladde::naive::Ticks;

/// `sign * sqrt(num / den)`: a correlation as the exact rational its square is.
#[derive(Clone, Copy, Debug)]
pub struct Rho {
    pub sign: i8,
    pub num: u128,
    pub den: u128,
}

/// `a * b` as (high, low) halves of the 256-bit product.
fn widening_mul(a: u128, b: u128) -> (u128, u128) {
    let mask = u128::from(u64::MAX);
    let (a_hi, a_lo, b_hi, b_lo) = (a >> 64, a & mask, b >> 64, b & mask);
    let (lo_lo, lo_hi, hi_lo, hi_hi) = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi);
    let middle = (lo_lo >> 64) + (lo_hi & mask) + (hi_lo & mask);
    let low = (lo_lo & mask) | (middle << 64);
    let high = hi_hi + (lo_hi >> 64) + (hi_lo >> 64) + (middle >> 64);
    (high, low)
}

pub fn cmp_rho(a: Rho, b: Rho) -> Ordering {
    if a.sign != b.sign {
        return a.sign.cmp(&b.sign);
    }
    let by_square = widening_mul(a.num, b.den).cmp(&widening_mul(b.num, a.den));
    if a.sign < 0 {
        by_square.reverse()
    } else {
        by_square
    }
}

/// The pair of the window of `rows` at `row` and the window of `neighbors` at `neighbor`: its
/// exact correlation and its f64 squared distance, computed as the batch computes it with `row`
/// the row. `None` for a window with a gap or too volatile for the range check. A series given
/// with the caller's flags has each window constant as flagged; one without, when its ticks are
/// equal. Two constant windows are at distance zero and one constant window at `m`; an
/// unflagged window of equal ticks is uncorrelated with every other.
pub fn cross_pair<'a, 'b>(
    rows: impl Into<Ticks<'a>>,
    neighbors: impl Into<Ticks<'b>>,
    m: usize,
    row: usize,
    neighbor: usize,
) -> Option<(Rho, f64)> {
    let (rows, neighbors): (Ticks<'a>, Ticks<'b>) = (rows.into(), neighbors.into());
    let take = |ticks: Ticks<'_>, at: usize| {
        ticks.samples()[at..at + m]
            .iter()
            .copied()
            .collect::<Option<Vec<i64>>>()
    };
    let (window, query) = (take(rows, row)?, take(neighbors, neighbor)?);
    let width = m as i128;
    let stats = |t: &[i64]| {
        let s: i128 = t.iter().map(|&x| i128::from(x)).sum();
        let q: i128 = t.iter().map(|&x| i128::from(x) * i128::from(x)).sum();
        (s, width * q - s * s)
    };
    let ((sq, vq), (sx, vx)) = (stats(&query), stats(&window));
    // A window too volatile for the range check has no distance.
    if vq >= 1 << 63 || vx >= 1 << 63 {
        return None;
    }
    let two_m = 2.0 * m as f64;
    let flagged = |ticks: Ticks<'_>, at: usize, flat: bool| {
        ticks.constant_windows().map_or(flat, |flags| flags[at])
    };
    let (constant_q, constant_x) = (
        flagged(neighbors, neighbor, vq == 0),
        flagged(rows, row, vx == 0),
    );
    Some(match (constant_q, constant_x) {
        (true, true) => (
            Rho {
                sign: 1,
                num: 1,
                den: 1,
            },
            0.0,
        ),
        (false, false) if vq == 0 || vx == 0 => (
            Rho {
                sign: 0,
                num: 0,
                den: 1,
            },
            two_m,
        ),
        (false, false) => {
            let qt: i128 = query
                .iter()
                .zip(&window)
                .map(|(&a, &b)| i128::from(a) * i128::from(b))
                .sum();
            let n = width * qt - sq * sx;
            let ri = 1.0 / libm::sqrt(vx as f64);
            let rj = 1.0 / libm::sqrt(vq as f64);
            let rho = (((n as i64 as f64) * ri) * rj).clamp(-1.0, 1.0);
            let d2 = if rho > 0.0 {
                let d = (vq * vx - n * n) as u128;
                let r = ri * rj;
                (((two_m * (d as f64)) * r) * r) / (1.0 + rho)
            } else {
                two_m * (1.0 - rho)
            };
            (
                Rho {
                    sign: n.signum() as i8,
                    num: (n * n) as u128,
                    den: (vq * vx) as u128,
                },
                d2,
            )
        }
        _ => (
            Rho {
                sign: 1,
                num: 1,
                den: 4,
            },
            two_m * (1.0 - 0.5),
        ),
    })
}

/// [`cross_pair`] within one series.
pub fn pair<'a>(
    ticks: impl Into<Ticks<'a>>,
    m: usize,
    row: usize,
    neighbor: usize,
) -> Option<(Rho, f64)> {
    let ticks = ticks.into();
    cross_pair(ticks, ticks, m, row, neighbor)
}
