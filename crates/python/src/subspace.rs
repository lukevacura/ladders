//! `subspace`: the dimensions a motif pair is closest in ([`ladde::subspace`]), in
//! STUMPY's shape.

use ladde::multidim::NonFinitePolicy;
use ladde::quantize::Scale;
use ladde::subspace::{Bits, Options};
use numpy::PyArray1;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::batch::resolve_scale;
use crate::constant::{flags_of_each, series as flagged};
use crate::convert::FloatRows;
use crate::convert::{Ints, int_vec};
use crate::err::{ScaleArg, multi_error, non_negative, window_length};

/// The dimensions of `t` as owned rows.
pub(crate) fn rows(t: &FloatRows<'_>) -> Vec<Vec<f64>> {
    t.as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect()
}

/// Refuses the arguments STUMPY takes that this library does not.
pub(crate) fn reject_unsupported(
    discretize_func: Option<&Bound<'_, PyAny>>,
    normalize: bool,
    p: f64,
) -> PyResult<()> {
    if discretize_func.is_some_and(|f| !f.is_none()) {
        return Err(PyNotImplementedError::new_err(
            "discretize_func is not supported: the discretization is STUMPY's default one",
        ));
    }
    if !normalize && p != 2.0 {
        return Err(PyNotImplementedError::new_err(
            "only p = 2 is supported for normalize=False",
        ));
    }
    Ok(())
}

pub(crate) fn bits(n_bit: i64) -> PyResult<Bits> {
    u8::try_from(n_bit)
        .ok()
        .and_then(|bits| Bits::new(bits).ok())
        .ok_or_else(|| PyValueError::new_err("n_bit must be between 1 and 8"))
}

pub(crate) fn include_dimensions(include: Option<Ints<'_>>) -> PyResult<Vec<usize>> {
    include
        .as_ref()
        .map(int_vec)
        .unwrap_or_default()
        .into_iter()
        .map(|dimension| non_negative("include entries", dimension))
        .collect()
}

/// One tick size per dimension, inferred where not given.
pub(crate) fn resolve_scales(
    rows: &[Vec<f64>],
    m: core::num::NonZeroUsize,
    scales: Option<Vec<Option<ScaleArg>>>,
) -> PyResult<Vec<Scale>> {
    let scales = scales.unwrap_or_else(|| vec![None; rows.len()]);
    if scales.len() != rows.len() {
        return Err(PyValueError::new_err(format!(
            "scales must have one entry per dimension ({}), got {}",
            rows.len(),
            scales.len()
        )));
    }
    rows.iter()
        .zip(scales)
        .map(|(row, scale)| resolve_scale(row, m, scale))
        .collect()
}

/// What `subspace` and `maamp_subspace` share once their arguments are parsed.
#[allow(clippy::too_many_arguments)]
fn subspace_of<'py>(
    py: Python<'py>,
    t: &FloatRows<'py>,
    m: i64,
    subseq_idx: i64,
    nn_idx: i64,
    k: i64,
    include: Option<Ints<'py>>,
    discords: bool,
    n_bit: i64,
    normalize: bool,
    scales: Option<Vec<Option<ScaleArg>>>,
    strict: bool,
    constant: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let m = window_length(m)?;
    let bits = bits(n_bit)?;
    let include = include_dimensions(include)?;
    let pair = (
        non_negative("subseq_idx", subseq_idx)?,
        non_negative("nn_idx", nn_idx)?,
    );
    let k = non_negative("k", k)?;
    let rows = rows(t);
    let options = Options {
        include: &include,
        discords,
        bits,
        non_finite: NonFinitePolicy::Missing,
    };
    let found = if normalize {
        let scales = resolve_scales(&rows, m, scales)?;
        let name = "T_subseq_isconstant";
        let slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let flags = flags_of_each(py, constant, &slices, m, name)?;
        let set = slices
            .iter()
            .zip(&flags)
            .zip(scales)
            .map(|((s, f), scale)| Ok((flagged(s, m, f, name)?, scale)))
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(|| {
            #[cfg(feature = "test-hooks")]
            crate::hold::park_if_armed();
            ladde::subspace::subspace(&set, m, pair, k, &options)
        })
    } else {
        if scales.is_some() {
            return Err(PyValueError::new_err(
                "scales applies to normalize=True only",
            ));
        }
        let set: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        py.detach(|| {
            #[cfg(feature = "test-hooks")]
            crate::hold::park_if_armed();
            ladde::subspace::maamp_subspace(&set, m, pair, k, &options)
        })
    }
    .map_err(multi_error)?;
    if strict && !found.complete {
        return Err(PyValueError::new_err(
            "fewer than k + 1 dimensions have a distance for this pair (a window holds a non-finite sample, or its variance is out of range)",
        ));
    }
    Ok(PyArray1::from_vec(
        py,
        found.dimensions.into_iter().map(|d| d as i64).collect(),
    ))
}

/// The `k + 1` dimensions of `T` (one row per dimension; lists and integer arrays are
/// converted to `float64`) in which the windows of length `m` at `subseq_idx` and `nn_idx` are
/// closest, as STUMPY's `subspace`: each window is z-normalized and every sample put into one of
/// `2**n_bit` bins, a dimension's distance is the root of its squared bin differences, and the
/// dimensions are taken in order of that distance, ties to the smaller dimension; the `include`
/// dimensions come first, in order of their own distance. The bins are decided exactly on the
/// quantized samples, not on rounded z-values. With `normalize=False` the bins are STUMPY's
/// linear ones between the smallest and largest finite sample of `T` (`p` must be 2).
///
/// `discords=True` puts the farthest dimensions first, as the documentation says; STUMPY 1.14.1
/// returns the wrong dimensions there. A dimension whose window holds a non-finite sample, or
/// (z-normalized) whose variance reaches the integer limit (V >= 2**63, the variance numerator
/// of the tick values), has no distance and is treated as missing: it sorts last, in both modes,
/// rather than being zero-filled; when fewer than `k + 1` dimensions have a distance, the others
/// follow, and with `strict=True` that raises `ValueError` instead. A window whose standard
/// deviation is below `1e-7` is not scaled, as in STUMPY.
///
/// `scales` has one tick size per dimension, `None` (the default) or a `None` entry inferring
/// that dimension's own, as `ab_join` does; it applies to `normalize=True` only.
/// `discretize_func` other than `None` raises `NotImplementedError`, as does `p` other than 2
/// for `normalize=False`; a different `p` is ignored for `normalize=True`, as in STUMPY.
/// `T_subseq_isconstant` is STUMPY's: a two-dimensional boolean array with one flag per window of
/// each dimension, or a function `f(a, w)` applied to every dimension, or a list with one entry
/// per dimension (`None`, a boolean array, or a function), saying which windows are constant, in
/// place of "all samples equal"; a constant window is binned as all samples at the mean. With
/// `normalize=False` it is accepted and ignored. `ValueError` for `m < 3`, an `m` beyond `len(T)`, a `subseq_idx` or `nn_idx`
/// that is not a window start, a `k` that is not below the number of dimensions, an `include`
/// entry that is not a dimension, an `n_bit` outside 1..8, or an inferable scale that does not
/// exist. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (T, m, subseq_idx, nn_idx, k, include=None, discords=false, discretize_func=None, n_bit=8, normalize=true, p=2.0, T_subseq_isconstant=None, *, scales=None, strict=false))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn subspace<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    m: i64,
    subseq_idx: i64,
    nn_idx: i64,
    k: i64,
    include: Option<Ints<'py>>,
    discords: bool,
    discretize_func: Option<&Bound<'py, PyAny>>,
    n_bit: i64,
    normalize: bool,
    p: f64,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scales: Option<Vec<Option<ScaleArg>>>,
    strict: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    reject_unsupported(discretize_func, normalize, p)?;
    subspace_of(
        py,
        &T,
        m,
        subseq_idx,
        nn_idx,
        k,
        include,
        discords,
        n_bit,
        normalize,
        scales,
        strict,
        T_subseq_isconstant,
    )
}

/// `subspace(T, ..., normalize=False)` under STUMPY's name for it: the non-normalized subspace,
/// with `p` required to be 2. See `subspace`.
#[pyfunction]
#[pyo3(signature = (T, m, subseq_idx, nn_idx, k, include=None, discords=false, discretize_func=None, n_bit=8, p=2.0, *, strict=false))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn maamp_subspace<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    m: i64,
    subseq_idx: i64,
    nn_idx: i64,
    k: i64,
    include: Option<Ints<'py>>,
    discords: bool,
    discretize_func: Option<&Bound<'py, PyAny>>,
    n_bit: i64,
    p: f64,
    strict: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    reject_unsupported(discretize_func, false, p)?;
    subspace_of(
        py, &T, m, subseq_idx, nn_idx, k, include, discords, n_bit, false, None, strict, None,
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(subspace, module)?)?;
    module.add_function(wrap_pyfunction!(maamp_subspace, module)?)?;
    Ok(())
}
