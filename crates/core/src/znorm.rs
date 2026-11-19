//! The z-normalized metric: the covariance recurrence on integer ticks.

use crate::kernel::Metric;
use crate::stats::Window;

/// The z-normalized metric.
pub struct ZNorm;

/// One window's state at a diagonal step: the ticks sliding it from the previous position, and
/// its own sum and validity.
#[derive(Clone, Copy)]
pub struct TickWindow {
    /// `x[i+m−1]`, the window's newest tick.
    pub entering: i64,
    /// `x[i−1]`, the tick the window just dropped.
    pub leaving: i64,
    pub stats: Window,
}

impl Metric for ZNorm {
    type Step = TickWindow;
    /// `QT`, wrapping mod 2⁶⁴.
    type Diagonal = i64;

    #[inline]
    fn start(a: &[i64], _a_step: &TickWindow, b: &[i64], _b_step: &TickWindow) -> i64 {
        crate::cross_term::wide(a, b)
    }

    #[inline]
    fn advance(previous: i64, a: &TickWindow, b: &TickWindow) -> i64 {
        previous
            .wrapping_sub(a.leaving.wrapping_mul(b.leaving))
            .wrapping_add(a.entering.wrapping_mul(b.entering))
    }
}

/// `N = m·QT − Sᵢ·Sⱼ`, wrapping. Every term here is a residue mod 2⁶⁴: combining residues with
/// wrapping ops and reading the result back as `i64` reconstructs the true integer exactly
/// whenever that integer itself fits `i64`, however far `QT`, `m·QT` or `Sᵢ·Sⱼ` individually
/// ran past it on the way there.
#[inline]
pub(crate) fn centered_cross(qt: i64, a: &TickWindow, b: &TickWindow, m: usize) -> i64 {
    (m as i64)
        .wrapping_mul(qt)
        .wrapping_sub(a.stats.sum.wrapping_mul(b.stats.sum))
}

#[cfg(test)]
mod tests;
