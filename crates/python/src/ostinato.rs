//! `ostinato` and `aamp_ostinato`: the consensus motif of a set of series
//! ([`ladde::ostinato`]), in STUMPY's shape. As in STUMPY, `ostinato(..., normalize=False)`
//! is `aamp_ostinato`.

use core::num::NonZeroUsize;

use ladde::quantize::Scale;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aamp::{require_euclidean, shared_scale_of};
use crate::batch::{resolve_scale, resolve_threads};
use crate::constant::{flags_of_each, series as flagged};
use crate::convert::Floats;
use crate::err::{ScaleArg, window_length};

/// STUMPY's `ostinato` return: `(radius, Ts_idx, subseq_idx)`.
type OstinatoResult = (f64, usize, usize);

/// The consensus motif of the series in `Ts` at window length `m`, as STUMPY's `ostinato`:
/// `(radius, Ts_idx, subseq_idx)`, the window `subseq_idx` of series `Ts_idx` whose distances to
/// its nearest window in every series are at most `radius`, the smallest radius there is. STUMPY's
/// search and its central-motif pass are followed step for step; every order and comparison
/// between distances is the exact one, not on rounded values, and the nearest window of a series
/// is the exactly nearest, ties to the smaller start. The central-motif pass keeps STUMPY's
/// `isclose` tolerance and mean on the f64 distances. A set with no window meeting every series
/// gives `(inf, 0, 0)`, STUMPY's initial best; for a set whose every sample is
/// a gap STUMPY zero-fills the gaps and reports `(0.0, 0, 0)`, where a window with a gap has no
/// distance here. A series with no finite sample needs no scale, so none is inferred for it.
///
/// The first five arguments are STUMPY's, in its order; the rest are keyword-only.
/// `Ts_subseq_isconstant` is STUMPY's: a list with one entry per series, each `None`, a boolean
/// array with one flag per window of that series, or a function `f(a, w)` returning one (given
/// the series, non-finite samples as NaN, and `m`), saying which of its windows are constant, in
/// place of "all samples equal"; a single function applies to every series, and a
/// two-dimensional boolean array (equal-length series) gives a row per series. Two constant
/// windows are at distance 0 and a constant one is at `sqrt(m)` from any other. With
/// `normalize=False` it is accepted and ignored, as in STUMPY.
/// `scales` has one tick size per series, `None` (the default) or a `None` entry inferring that
/// series' own, as `ab_join` does; `threads` means what it does there and the result does not
/// depend on it. With `normalize=False` this is `aamp_ostinato`, and `p` and `scale` are its; with
/// the default `normalize=True`, `p` is ignored, as in STUMPY, and `scale` must not be given (nor
/// `scales` with `normalize=False`).
///
/// Each series takes anything NumPy turns into a one-dimensional float array. `ValueError` for an
/// empty `Ts`, a `scales` of another length, a series shorter than `m`, or a scale that cannot be
/// inferred. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (
    Ts, m, normalize=true, p=2.0, Ts_subseq_isconstant=None, *, scales=None, threads=None,
    scale=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn ostinato<'py>(
    py: Python<'py>,
    Ts: Vec<Floats<'py>>,
    m: i64,
    normalize: bool,
    p: f64,
    Ts_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scales: Option<Vec<Option<ScaleArg>>>,
    threads: Option<&Bound<'py, PyAny>>,
    scale: Option<ScaleArg>,
) -> PyResult<OstinatoResult> {
    if !normalize {
        if scales.is_some() {
            return Err(PyValueError::new_err(
                "scales is for normalize=True; use scale",
            ));
        }
        return aamp_ostinato(py, Ts, m, p, scale, threads);
    }
    if scale.is_some() {
        return Err(PyValueError::new_err(
            "scale is for normalize=False; use scales",
        ));
    }
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    if Ts.is_empty() {
        return Err(PyValueError::new_err("Ts must hold at least one series"));
    }
    let scales = scales.unwrap_or_else(|| vec![None; Ts.len()]);
    if scales.len() != Ts.len() {
        return Err(PyValueError::new_err(format!(
            "scales must have one entry per series ({}), got {}",
            Ts.len(),
            scales.len()
        )));
    }
    let series: Vec<Vec<f64>> = Ts.iter().map(|t| t.as_array().to_vec()).collect();
    let scales: Vec<Scale> = series
        .iter()
        .zip(scales)
        .map(|(s, scale)| scale_or_unit(s, m, scale))
        .collect::<PyResult<_>>()?;
    let name = "Ts_subseq_isconstant";
    let slices: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let flags = flags_of_each(py, Ts_subseq_isconstant, &slices, m, name)?;
    let set = slices
        .iter()
        .zip(&flags)
        .zip(scales)
        .map(|((s, f), scale)| Ok((flagged(s, m, f, name)?, scale)))
        .collect::<PyResult<Vec<_>>>()?;
    let found = py
        .detach(|| ladde::ostinato::ostinato(&set, m, threads))
        .ok_or_else(too_short)?;
    Ok((found.radius, found.series, found.window))
}

fn has_finite(series: &[f64]) -> bool {
    series.iter().any(|x| x.is_finite())
}

/// The scale `resolve_scale` gives, or the unit scale for a series with no finite sample at all
/// (it has no window to quantize, so no scale can be wrong for it).
fn scale_or_unit(series: &[f64], m: NonZeroUsize, scale: Option<ScaleArg>) -> PyResult<Scale> {
    if !matches!(scale, Some(ScaleArg::Tick(_))) && !has_finite(series) {
        return Ok(Scale::new(1.0).expect("one is a tick size"));
    }
    resolve_scale(series, m, scale)
}

fn too_short() -> PyErr {
    PyValueError::new_err("every series must hold at least m samples")
}

/// STUMPY's `aamp_ostinato`: [`ostinato`] under the Euclidean distance on the raw values. Every
/// series is quantized to one `scale` (inferred from all of them together as `aamp` infers it when
/// `None`), and the radius is in the caller's units. `p` must be 2.0; any other value raises
/// `NotImplementedError`. A window with a gap has no distance.
#[pyfunction]
#[pyo3(signature = (Ts, m, p=2.0, *, scale=None, threads=None))]
pub(crate) fn aamp_ostinato<'py>(
    py: Python<'py>,
    #[allow(non_snake_case)] Ts: Vec<Floats<'py>>,
    m: i64,
    p: f64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<OstinatoResult> {
    require_euclidean(p)?;
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    if Ts.is_empty() {
        return Err(PyValueError::new_err("Ts must hold at least one series"));
    }
    let series: Vec<Vec<f64>> = Ts.iter().map(|t| t.as_array().to_vec()).collect();
    let set: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let scale = if set.iter().any(|s| has_finite(s)) {
        shared_scale_of(&set, m, scale)?
    } else {
        scale_or_unit(&[], m, scale)?
    };
    let found = py
        .detach(|| ladde::ostinato::aamp_ostinato(&set, scale, m, threads))
        .ok_or_else(too_short)?;
    Ok((found.radius, found.series, found.window))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ostinato, module)?)?;
    module.add_function(wrap_pyfunction!(aamp_ostinato, module)?)?;
    Ok(())
}
