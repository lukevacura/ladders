//! `mpdist` and `aampdist`: the matrix profile distance ([`ladde::mpdist`]) between two
//! series, as STUMPY's `mpdist` and `aampdist` return it. As in STUMPY, `mpdist(...,
//! normalize=False)` is `aampdist`.

use ladde::mpdist::Selection;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aamp::{require_euclidean, shared_scale};
use crate::batch::{resolve_scale, resolve_threads};
use crate::constant::{flags_of, series as flagged};
use crate::convert::Floats;
use crate::err::{ScaleArg, window_length};

/// The selection `percentage` or the rank `k` names (`k` wins when given) among `positions` joined
/// distances; `names` are their keywords, for the messages. A rank past the last position and a
/// percentage outside `[0, 1]` are `ValueError`s, where STUMPY clamps them.
pub(crate) fn selection(
    percentage: f64,
    k: Option<i64>,
    positions: usize,
    (percentage_name, k_name): (&str, &str),
) -> PyResult<Selection> {
    let Some(k) = k else {
        return Selection::percentage(percentage).ok_or_else(|| {
            if percentage.is_nan() {
                return PyValueError::new_err(format!("{percentage_name} must not be NaN"));
            }
            let clipped = percentage.clamp(0.0, 1.0);
            PyValueError::new_err(format!(
                "{percentage_name}={percentage:?} is outside [0, 1]; STUMPY clips it to \
                 {clipped:?}; pass {percentage_name}={clipped:?} to get that"
            ))
        });
    };
    let rank = usize::try_from(k)
        .map(Selection::rank)
        .map_err(|_| PyValueError::new_err(format!("{k_name} must be at least 0, got {k}")))?;
    if rank.fits(positions) {
        return Ok(rank);
    }
    let last = positions.saturating_sub(1);
    Err(PyValueError::new_err(format!(
        "{k_name}={k} is past the last position ({last}) of the {positions} joined distances; \
         STUMPY uses {last}; pass {k_name}={last} to get that"
    )))
}

/// The number of joined distances of two series at window length `m`, or the `ValueError` for a
/// series without a window.
fn joined_positions(a: &[f64], b: &[f64], m: core::num::NonZeroUsize) -> PyResult<usize> {
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    match (windows(a.len()), windows(b.len())) {
        (0, _) | (_, 0) => Err(too_short()),
        (wa, wb) => Ok(wa + wb),
    }
}

const NAMES: (&str, &str) = ("percentage", "k");

/// The matrix profile distance between `T_A` and `T_B` at window length `m`, as STUMPY's
/// `mpdist`: the AB-join and BA-join nearest distances are concatenated and the value at sorted
/// position `k` is reported. `k` is `ceil(percentage * (len(T_A) + len(T_B)))`, lowered to the
/// last position as in STUMPY, or the given `k` (then `percentage` is ignored). Differences from
/// STUMPY, each a `ValueError`: a `percentage` outside `[0, 1]` (STUMPY clips it) and a `k` past
/// the last position, `len(T_A) + len(T_B) - 2m + 1` (STUMPY lowers it). When that value is
/// infinite the largest finite one before it is reported, or
/// `inf` when there is none. The order between distances is the exact one, so equal distances
/// never depend on rounding; STUMPY sorts rounded values.
///
/// The first seven arguments are STUMPY's, in its order; the rest are keyword-only.
/// `scale_a` and `scale_b` are each series' own tick size, inferred per series when `None` (as
/// `ab_join` does); `threads` means what it does there and the result does not depend on it.
/// With `normalize=False` this is `aampdist`, and `p` and `scale` are its; with the default
/// `normalize=True`, `p` is ignored, as in STUMPY, and `scale` must not be given (nor
/// `scale_a`/`scale_b` with `normalize=False`).
///
/// `T_A` and `T_B` take anything NumPy turns into a one-dimensional float array. `ValueError` for
/// a series shorter than `m`, a negative `k`, a NaN `percentage` (unless `k` is given), or an
/// inferable scale that does not exist. Releases the GIL while it runs.
///
/// `T_A_subseq_isconstant` and `T_B_subseq_isconstant` are STUMPY's: for each series, a boolean
/// array with one flag per window, or a function `f(a, w)` returning one (given the series,
/// non-finite samples as NaN, and `m`), saying which windows are constant, in place of "all
/// samples equal"; two constant windows are at distance 0 and a constant one is at `sqrt(m)`
/// from any other. With `normalize=False` they are accepted and ignored, as in STUMPY.
#[pyfunction]
#[pyo3(signature = (
    T_A, T_B, m, percentage=0.05, k=None, normalize=true, p=2.0, T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None, *, scale_a=None, scale_b=None, threads=None, scale=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn mpdist<'py>(
    py: Python<'py>,
    T_A: Floats<'py>,
    T_B: Floats<'py>,
    m: i64,
    percentage: f64,
    k: Option<i64>,
    normalize: bool,
    p: f64,
    T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    T_B_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scale_a: Option<ScaleArg>,
    scale_b: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    scale: Option<ScaleArg>,
) -> PyResult<f64> {
    if !normalize {
        if scale_a.is_some() || scale_b.is_some() {
            return Err(PyValueError::new_err(
                "scale_a and scale_b are for normalize=True; use scale",
            ));
        }
        return aampdist(py, T_A, T_B, m, percentage, k, p, scale, threads);
    }
    if scale.is_some() {
        return Err(PyValueError::new_err(
            "scale is for normalize=False; use scale_a and scale_b",
        ));
    }
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let (a, b) = (T_A.as_array().to_vec(), T_B.as_array().to_vec());
    let selection = selection(percentage, k, joined_positions(&a, &b, m)?, NAMES)?;
    let (scale_a, scale_b) = (
        resolve_scale(&a, m, scale_a)?,
        resolve_scale(&b, m, scale_b)?,
    );
    let (name_a, name_b) = ("T_A_subseq_isconstant", "T_B_subseq_isconstant");
    let (flags_a, flags_b) = (
        flags_of(py, T_A_subseq_isconstant, &a, m, name_a)?,
        flags_of(py, T_B_subseq_isconstant, &b, m, name_b)?,
    );
    let (input_a, input_b) = (
        flagged(&a, m, &flags_a, name_a)?,
        flagged(&b, m, &flags_b, name_b)?,
    );
    py.detach(|| ladde::mpdist::mpdist(input_a, input_b, m, scale_a, scale_b, threads, selection))
        .ok_or_else(too_short)
}

fn too_short() -> PyErr {
    PyValueError::new_err("T_A and T_B must each hold at least m samples")
}

/// STUMPY's `aampdist`: [`mpdist`] under the Euclidean distance on the raw values. Both series
/// are quantized to one `scale` (inferred from both together as `aamp` infers it when `None`),
/// and the distance is in the caller's units. `p` must be 2.0; any other value raises
/// `NotImplementedError`. A window with a gap has no distance and contributes infinity.
#[pyfunction]
#[pyo3(signature = (T_A, T_B, m, percentage=0.05, k=None, p=2.0, *, scale=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn aampdist<'py>(
    py: Python<'py>,
    #[allow(non_snake_case)] T_A: Floats<'py>,
    #[allow(non_snake_case)] T_B: Floats<'py>,
    m: i64,
    percentage: f64,
    k: Option<i64>,
    p: f64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<f64> {
    require_euclidean(p)?;
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let (a, b) = (T_A.as_array().to_vec(), T_B.as_array().to_vec());
    let selection = selection(percentage, k, joined_positions(&a, &b, m)?, NAMES)?;
    let scale = shared_scale(&a, Some(&b), m, scale)?;
    py.detach(|| ladde::mpdist::aampdist(&a, &b, m, scale, threads, selection))
        .ok_or_else(too_short)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mpdist, module)?)?;
    module.add_function(wrap_pyfunction!(aampdist, module)?)?;
    Ok(())
}
