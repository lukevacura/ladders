//! `mdl`: the description length of a motif pair at every subspace size ([`ladde::mdl`]),
//! in STUMPY's shape.

use ladde::multidim::NonFinitePolicy;
use ladde::subspace::Options;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constant::{flags_of_each, series as flagged};
use crate::convert::FloatRows;
use crate::convert::{Ints, int_vec};
use crate::err::{ScaleArg, multi_error, non_negative, window_length};
use crate::subspace::{bits, include_dimensions, reject_unsupported, resolve_scales, rows};

/// STUMPY's `mdl` return: `(bit_sizes, S)`.
type MdlResult<'py> = (Bound<'py, PyArray1<f64>>, Vec<Bound<'py, PyArray1<i64>>>);

fn starts(what: &str, values: &Ints<'_>) -> PyResult<Vec<usize>> {
    int_vec(values)
        .into_iter()
        .map(|value| non_negative(what, value))
        .collect()
}

/// What `mdl` and `maamp_mdl` share once their arguments are parsed.
#[allow(clippy::too_many_arguments)]
fn mdl_of<'py>(
    py: Python<'py>,
    t: &FloatRows<'py>,
    m: i64,
    subseq_idx: Ints<'py>,
    nn_idx: Ints<'py>,
    include: Option<Ints<'py>>,
    discords: bool,
    n_bit: i64,
    normalize: bool,
    scales: Option<Vec<Option<ScaleArg>>>,
    constant: Option<&Bound<'py, PyAny>>,
) -> PyResult<MdlResult<'py>> {
    let m = window_length(m)?;
    let bits = bits(n_bit)?;
    let include = include_dimensions(include)?;
    let (first, second) = (
        starts("subseq_idx", &subseq_idx)?,
        starts("nn_idx", &nn_idx)?,
    );
    let rows = rows(t);
    let options = Options {
        include: &include,
        discords,
        bits,
        non_finite: NonFinitePolicy::Missing,
    };
    let described = if normalize {
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
            ladde::mdl::mdl(&set, m, &first, &second, &options)
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
            ladde::mdl::maamp_mdl(&set, m, &first, &second, &options)
        })
    }
    .map_err(multi_error)?;
    let subspaces = described
        .subspaces
        .into_iter()
        .map(|subspace| {
            PyArray1::from_vec(
                py,
                subspace.dimensions.into_iter().map(|d| d as i64).collect(),
            )
        })
        .collect();
    Ok((PyArray1::from_vec(py, described.bit_sizes), subspaces))
}

/// The number of bits needed to describe the motif pair of each dimension count, as STUMPY's
/// `mdl`: for each `k`, the windows of length `m` at `subseq_idx[k]` and `nn_idx[k]` of `t` (one
/// row per dimension), usually the motif pair of the `(k + 1)`-dimensional matrix profile, are
/// binned as `subspace` bins them and described as one window plus the differences of the other
/// over the `k + 1` dimensions of their subspace: `n_bit * (2 * d * m - s * m) + s * m *
/// log2(distinct) + distinct * n_bit` with `s = k + 1` the subspace size, `d` the number of
/// dimensions and `distinct` the number of different bin differences. Returns `(bit_sizes, S)`,
/// the size for each `k` and a list of the subspaces, one array of `k + 1` dimensions for each `k`.
/// The logarithms are correctly rounded values, the integer terms exact.
///
/// `include`, `discords`, `n_bit`, `normalize`, `scales`, `discretize_func` and the rest mean
/// what they do for `subspace`, with the same differences from STUMPY: `discords=True` puts the
/// farthest dimensions first; a dimension whose window holds a non-finite sample has no distance
/// and sorts last rather than being zero-filled, and where that leaves a subspace holding such a
/// dimension its size is `inf` (so the smallest size never picks it). `ValueError` for `m < 3`,
/// an `m` beyond `len(T)`, `subseq_idx` or `nn_idx` of another length than the number of
/// dimensions or holding a number that is not a window start, an `include` entry that is not a
/// dimension, an `n_bit` outside 1..8, or an inferable scale that does not exist. Releases the GIL
/// while it runs. `T_subseq_isconstant` is STUMPY's, as for `subspace`.
#[pyfunction]
#[pyo3(signature = (T, m, subseq_idx, nn_idx, include=None, discords=false, discretize_func=None, n_bit=8, normalize=true, p=2.0, T_subseq_isconstant=None, *, scales=None))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn mdl<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    m: i64,
    subseq_idx: Ints<'py>,
    nn_idx: Ints<'py>,
    include: Option<Ints<'py>>,
    discords: bool,
    discretize_func: Option<&Bound<'py, PyAny>>,
    n_bit: i64,
    normalize: bool,
    p: f64,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scales: Option<Vec<Option<ScaleArg>>>,
) -> PyResult<MdlResult<'py>> {
    reject_unsupported(discretize_func, normalize, p)?;
    mdl_of(
        py,
        &T,
        m,
        subseq_idx,
        nn_idx,
        include,
        discords,
        n_bit,
        normalize,
        scales,
        T_subseq_isconstant,
    )
}

/// `mdl(T, ..., normalize=False)` under STUMPY's name for it, with `p` required to be 2. See
/// `mdl`.
#[pyfunction]
#[pyo3(signature = (T, m, subseq_idx, nn_idx, include=None, discords=false, discretize_func=None, n_bit=8, p=2.0))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn maamp_mdl<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    m: i64,
    subseq_idx: Ints<'py>,
    nn_idx: Ints<'py>,
    include: Option<Ints<'py>>,
    discords: bool,
    discretize_func: Option<&Bound<'py, PyAny>>,
    n_bit: i64,
    p: f64,
) -> PyResult<MdlResult<'py>> {
    reject_unsupported(discretize_func, false, p)?;
    mdl_of(
        py, &T, m, subseq_idx, nn_idx, include, discords, n_bit, false, None, None,
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mdl, module)?)?;
    module.add_function(wrap_pyfunction!(maamp_mdl, module)?)?;
    Ok(())
}
