//! `mstump`: STUMPY's multi-dimensional matrix profile, built on the batch engine in
//! [`ladde::batch::multi`] and returned as NumPy arrays, releasing the GIL for the
//! computation itself.

use ladde::batch::multi as batch;
use ladde::multidim::{DimensionDistance, MultiError, MultiProfile};
use ladde::quantize::Scale;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::batch::{resolve_scale, resolve_threads};
use crate::constant::{flags_of_each, series as flagged};
use crate::convert::{FloatRows, Ints, int_vec};
use crate::err::{ScaleArg, multi_error, window_length};

/// STUMPY's `mstump` shape: `(P, I)`, each `(d, windows)`.
type MstumpResult<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

/// The `include` list as dimension numbers: negative entries are a `ValueError`.
fn include_list(include: Option<Ints<'_>>) -> PyResult<Vec<usize>> {
    include
        .as_ref()
        .map(int_vec)
        .unwrap_or_default()
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                PyValueError::new_err(format!(
                    "include entries must not be negative, got {dimension}"
                ))
            })
        })
        .collect()
}

/// The profile as `(P, I)` arrays of shape `(d, windows)`; `value` turns a mean into `P`'s entry.
fn arrays<'py, D: DimensionDistance>(
    py: Python<'py>,
    profile: &MultiProfile<D>,
    value: impl Fn(f64) -> f64,
) -> PyResult<MstumpResult<'py>> {
    let shape = [
        profile.distance.len(),
        profile.distance.first().map_or(0, Vec::len),
    ];
    let distances: Vec<f64> = profile
        .distance
        .iter()
        .flatten()
        .map(|mean| value(mean.to_f64()))
        .collect();
    let indices: Vec<i64> = profile
        .index
        .iter()
        .flatten()
        .map(|index| index.get().map_or(-1, |i| i as i64))
        .collect();
    Ok((
        numpy::PyArray1::from_vec(py, distances).reshape(shape)?,
        numpy::PyArray1::from_vec(py, indices).reshape(shape)?,
    ))
}

/// The multi-dimensional matrix profile of `t` (`d` series of one length, shape `(d, n)`), in the
/// shape of STUMPY's `mstump(T, m, include, discords)`: `(P, I)`, each `(d, n - m + 1)`. Row `k` is
/// the profile over the mean of the `k + 1` chosen dimensions' z-normalized Euclidean distances
/// (the nearest dimensions at each pair of windows; farthest first with `discords`), the nearest
/// window outside the exclusion zone, ties to the smaller index; a window with no neighbor has
/// index `-1` and distance `inf`. `include` lists dimensions that are always chosen first, in the
/// order given. A dimension whose window holds a non-finite sample has no distance for that
/// window, and is chosen last (even with `discords`, where STUMPY would choose it first).
///
/// `scales`, when given, is the tick size every sample is quantized to: one number for every
/// dimension, or one per dimension; when `None`, each dimension's is inferred as `left_profile`
/// infers it. `threads` means what it does in `left_profile`, and the result does not depend on
/// it. With `normalize=False` this is `maamp` (`scales` then none or one number). `p` must be
/// 2.0; any other value raises `NotImplementedError`. Releases the GIL while it runs.
///
/// `T_subseq_isconstant` is STUMPY's: a two-dimensional boolean array with one flag per window
/// of each dimension, or a function `f(a, w)` applied to every dimension, or a list with one
/// entry per dimension (`None`, a boolean array, or a function), saying which windows are
/// constant, in place of "all samples equal"; a function is given the dimension, non-finite
/// samples as NaN, and `m`, and returns one flag per window. With `normalize=False` it is
/// accepted and ignored, as in STUMPY.
#[pyfunction]
#[pyo3(signature = (T, m, include=None, discords=false, normalize=true, p=2.0, T_subseq_isconstant=None, *, scales=None, threads=None))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn mstump<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    m: i64,
    include: Option<Ints<'py>>,
    discords: bool,
    normalize: bool,
    p: f64,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scales: Option<&Bound<'py, PyAny>>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<MstumpResult<'py>> {
    if p != 2.0 {
        return Err(PyNotImplementedError::new_err(format!(
            "p must be 2.0, got {p}: other values of p are not available yet"
        )));
    }
    if !normalize {
        let scale = one_scale(scales)?;
        return nonnorm(py, T, m, include, discords, scale, threads);
    }
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let include = include_list(include)?;
    let series: Vec<Vec<f64>> = T
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    if series.first().is_some_and(|first| first.len() < m.get()) {
        return Err(PyValueError::new_err(format!(
            "T must have at least m = {m} samples in every dimension"
        )));
    }
    let scales = dimension_scales(&series, m, scales)?;
    let name = "T_subseq_isconstant";
    let dimensions: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let flags = flags_of_each(py, T_subseq_isconstant, &dimensions, m, name)?;
    let inputs = dimensions
        .iter()
        .zip(&flags)
        .map(|(s, f)| flagged(s, m, f, name))
        .collect::<PyResult<Vec<_>>>()?;
    let profile = py
        .detach(|| batch::mstump(&inputs, m, &scales, &include, discords, threads))
        .map_err(multi_error)?;
    arrays(py, &profile, |mean| mean)
}

/// The non-normalized profile, as `maamp` returns it: all dimensions share one tick size,
/// `scale` or, when `None`, the one inferred from every dimension together.
fn nonnorm<'py>(
    py: Python<'py>,
    t: FloatRows<'py>,
    m: i64,
    include: Option<Ints<'py>>,
    discords: bool,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<MstumpResult<'py>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let include = include_list(include)?;
    let series: Vec<Vec<f64>> = t
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    if series.first().is_some_and(|first| first.len() < m.get()) {
        return Err(PyValueError::new_err(format!(
            "T must have at least m = {m} samples in every dimension"
        )));
    }
    let scale = shared_scale(&series, m, scale)?;
    let dimensions: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let profile = py
        .detach(|| batch::maamp(&dimensions, m, scale, &include, discords, threads))
        .map_err(multi_error)?;
    arrays(py, &profile, |mean| mean * scale.get())
}

/// The multi-dimensional non-normalized matrix profile of `t` (`d` series of one length, shape
/// `(d, n)`), in the shape of STUMPY's `maamp(T, m, include, discords, p)`: `(P, I)`, each
/// `(d, n - m + 1)`, as `mstump` returns them but over the Euclidean distance on the raw values
/// (computed exactly on values quantized to `scale`, in the caller's units), the mean of the
/// nearest `k + 1` dimensions' distances in row `k`. All dimensions share one `scale`, because
/// their distances average in one unit; when `None`, it is inferred from every dimension
/// together as `aamp` infers it. A dimension's pair also has no distance when `m` times the
/// squared range of its ticks from the start through the later window reaches 2**128. `p` must
/// be 2.0; any other value raises `NotImplementedError`. `threads` means what it does in
/// `left_profile`, and the result does not depend on it. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (T, m, include=None, discords=false, p=2.0, *, scale=None, threads=None))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn maamp<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    m: i64,
    include: Option<Ints<'py>>,
    discords: bool,
    p: f64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<MstumpResult<'py>> {
    if p != 2.0 {
        return Err(PyNotImplementedError::new_err(format!(
            "p must be 2.0, got {p}: other values of p are not available yet"
        )));
    }
    nonnorm(py, T, m, include, discords, scale, threads)
}

/// The one tick size every dimension of a non-normalized input shares: `scale`, or the one
/// inferred as [`crate::aamp::shared_scale_of`] does: the common grid of all dimensions, else the
/// finest power of two at which each dimension fits on its own.
pub(crate) fn shared_scale(
    series: &[Vec<f64>],
    m: core::num::NonZeroUsize,
    scale: Option<ScaleArg>,
) -> PyResult<Scale> {
    let parts: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    crate::aamp::shared_scale_of(&parts, m, scale)
}

/// A `scales` argument that is `None` or the one number every dimension shares.
pub(crate) fn one_scale(scales: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ScaleArg>> {
    scales
        .map(|scales| {
            scales.extract::<ScaleArg>().map_err(|_| {
                PyValueError::new_err("normalize=False takes one scale for every dimension")
            })
        })
        .transpose()
}

/// One tick size per dimension: the number given for all, the list given, or each dimension's
/// own inferred one.
pub(crate) fn dimension_scales(
    series: &[Vec<f64>],
    m: core::num::NonZeroUsize,
    scales: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Scale>> {
    let Some(scales) = scales else {
        return series
            .iter()
            .map(|dimension| resolve_scale(dimension, m, None))
            .collect();
    };
    let listed: Vec<ScaleArg> = match scales.extract::<ScaleArg>() {
        Ok(one) => vec![one; series.len()],
        Err(_) => scales.extract().map_err(|_| {
            PyValueError::new_err(
                "scales must be None, a number, \"finest\" or a list of numbers or \"finest\"",
            )
        })?,
    };
    if listed.len() != series.len() {
        return Err(multi_error(MultiError::ScaleCount));
    }
    series
        .iter()
        .zip(listed)
        .map(|(dimension, scale)| resolve_scale(dimension, m, Some(scale)))
        .collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mstump, module)?)?;
    module.add_function(wrap_pyfunction!(maamp, module)?)?;
    Ok(())
}
