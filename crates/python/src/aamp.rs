//! `aamp`: STUMPY's non-normalized matrix profile (the Euclidean distance on the raw values),
//! built on the batch engine in [`ladde::batch::nonnorm`] and returned as NumPy arrays,
//! releasing the GIL for the computation itself.

use ladde::batch::nonnorm as batch;
use ladde::nonnorm::NonNormNeighbor;
use ladde::quantize::Scale;
use numpy::PyArray1;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use std::num::NonZeroUsize;

use crate::batch::{capacity_error, resolve_k, resolve_scale_of, resolve_threads, topk_arrays_of};
use crate::convert::{Floats, nonnorm_index_and_distance};
use crate::err::{ScaleArg, bounded_cells, window_length};

/// Every non-normalized function takes STUMPY's `p`; only the Euclidean `p = 2` exists so far.
pub(crate) fn require_euclidean(p: f64) -> PyResult<()> {
    if p != 2.0 {
        return Err(PyNotImplementedError::new_err(format!(
            "p must be 2.0, got {p}: other values of p are not available yet"
        )));
    }
    Ok(())
}

/// STUMPY's column layout as four arrays: `(P, I, left_I, right_I)`; `P` and `I` are 1-D with `k =
/// 1` and `(windows, k)` otherwise.
type AampResult<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// The columns of a profile whose rows each hold a nearest neighbor and, for a self-join, its
/// left and right ones.
struct Columns {
    distances: Vec<f64>,
    indices: Vec<i64>,
    left: Vec<i64>,
    right: Vec<i64>,
}

impl Columns {
    fn with_capacity(rows: usize) -> Self {
        Self {
            distances: Vec::with_capacity(rows),
            indices: Vec::with_capacity(rows),
            left: Vec::with_capacity(rows),
            right: Vec::with_capacity(rows),
        }
    }

    fn push(&mut self, nearest: NonNormNeighbor, sides: (i64, i64), scale: Scale) {
        let (index, distance) = nonnorm_index_and_distance(nearest, scale);
        self.distances.push(distance);
        self.indices.push(index);
        self.left.push(sides.0);
        self.right.push(sides.1);
    }
}

/// The tick size `scale` names, or, when `None`, the one inferred from `a` and, for a join, `b`
/// together (the two series share one scale, because their distances compare in one unit).
pub(crate) fn shared_scale(
    a: &[f64],
    b: Option<&[f64]>,
    m: core::num::NonZeroUsize,
    scale: Option<ScaleArg>,
) -> PyResult<Scale> {
    match b {
        Some(b) => shared_scale_of(&[a, b], m, scale),
        None => shared_scale_of(&[a], m, scale),
    }
}

/// [`shared_scale`] for any number of series.
pub(crate) fn shared_scale_of(
    parts: &[&[f64]],
    m: core::num::NonZeroUsize,
    scale: Option<ScaleArg>,
) -> PyResult<Scale> {
    resolve_scale_of(parts, m, scale)
}

/// The non-normalized matrix profile of `T_A`, in the shape of STUMPY's `aamp(T_A, m, T_B,
/// ignore_trivial, p, k)`: `(P, I, left_I, right_I)`, one entry per window of `T_A`, oldest first.
/// The distance is the Euclidean distance on the raw values (`p = 2`), computed exactly on
/// values quantized to `scale`, in the caller's units.
///
/// With `T_B` of `None` (a self-join) `I` is the nearer of the left neighbor (`left_I`, earlier
/// windows outside the exclusion zone) and the right neighbor (`right_I`, later ones), ties to
/// the smaller index, and `P` is its distance; a window with no neighbor on a side gets index
/// `-1` there, and with none at all, distance `inf`. Looks ahead: the right side uses samples
/// after each window, so it is not a causal signal.
///
/// With a `T_B` that is not `T_A`, every window of `T_A` meets its nearest window of `T_B`, with
/// no exclusion zone, ties to the smaller index: `I` indexes the windows of `T_B`, and `left_I`
/// and `right_I` are `-1`, as in STUMPY. As in STUMPY, `ignore_trivial` only matters when `T_B`
/// equals `T_A` element for element: `True` (the default) makes it a self-join, `False` an
/// AB-join in which each window meets itself at distance 0.
///
/// `k` (default 1) is how many nearest neighbors each window keeps, as STUMPY's `aamp(T_A, m, T_B,
/// k=k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`, nearest first by exact squared
/// distance, ties to the smaller index, padded with `inf` and `-1` where a window has fewer than
/// `k` neighbors. A self-join ranks over both sides while `left_I` and `right_I` stay each side's
/// one nearest; an AB-join keeps the `k` nearest windows of `T_B`. `k < 1` raises `ValueError`, and
/// so does `windows * k` above 2**28.
///
/// `p` must be 2.0; any other value raises `NotImplementedError` (whole-number `p` come later).
/// `scale`, when given, is the tick size every sample is quantized to; when `None`, it is
/// the one both series share, on their common grid when they have one, else the finest power of
/// two at which each series fits on its own, as `left_profile` chooses for one. `threads` means
/// what it does in `left_profile`, and the result does not depend on it. A window with a
/// non-finite sample has no neighbor and is never one. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (T_A, m, T_B=None, ignore_trivial=true, p=2.0, k=1, *, scale=None, threads=None))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn aamp<'py>(
    py: Python<'py>,
    T_A: Floats<'py>,
    m: i64,
    T_B: Option<Floats<'py>>,
    ignore_trivial: bool,
    p: f64,
    k: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<AampResult<'py>> {
    require_euclidean(p)?;
    let m = window_length(m)?;
    let k = resolve_k(k)?;
    let threads = resolve_threads(threads)?;
    let a = T_A.as_array().to_vec();
    let b = T_B.map(|b| b.as_array().to_vec());
    let join_with = b.filter(|b| !(ignore_trivial && *b == a));
    let scale = shared_scale(&a, join_with.as_deref(), m, scale)?;

    if k != NonZeroUsize::MIN {
        return top_k(py, &a, join_with.as_deref(), m, scale, threads, k);
    }
    let columns = match join_with {
        None => {
            let profile = py.detach(|| batch::two_sided_profile(&a, m, scale, threads));
            let mut columns = Columns::with_capacity(profile.len());
            for (_, sides) in profile {
                let side = |n: NonNormNeighbor| nonnorm_index_and_distance(n, scale).0;
                columns.push(sides.nearest, (side(sides.left), side(sides.right)), scale);
            }
            columns
        }
        Some(b) => {
            let rows = py.detach(|| batch::ab_join(&a, &b, m, scale, threads));
            let mut columns = Columns::with_capacity(rows.len());
            for (_, nearest) in rows {
                columns.push(nearest, (-1, -1), scale);
            }
            columns
        }
    };
    Ok((
        PyArray1::from_vec(py, columns.distances).into_any(),
        PyArray1::from_vec(py, columns.indices).into_any(),
        PyArray1::from_vec(py, columns.left),
        PyArray1::from_vec(py, columns.right),
    ))
}

/// [`aamp`] with `k > 1`: the `k` nearest neighbors per window, as `(windows, k)` arrays.
fn top_k<'py>(
    py: Python<'py>,
    a: &[f64],
    b: Option<&[f64]>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: NonZeroUsize,
) -> PyResult<AampResult<'py>> {
    bounded_cells("k", (a.len() + 1).saturating_sub(m.get()), k.get())?;
    let convert = |neighbor: Option<NonNormNeighbor>| match neighbor {
        Some(neighbor) => nonnorm_index_and_distance(neighbor, scale),
        None => (-1, f64::INFINITY),
    };
    match b {
        None => {
            let rows = py
                .detach(|| batch::two_sided_topk_profile(a, m, scale, threads, k))
                .map_err(capacity_error)?;
            let (distances, indices) =
                topk_arrays_of(py, rows.iter().map(|(_, sides)| &sides.nearest), k, convert)?;
            let side = |n: NonNormNeighbor| nonnorm_index_and_distance(n, scale).0;
            let left = rows.iter().map(|(_, sides)| side(sides.left)).collect();
            let right = rows.iter().map(|(_, sides)| side(sides.right)).collect();
            Ok((
                distances,
                indices,
                PyArray1::from_vec(py, left),
                PyArray1::from_vec(py, right),
            ))
        }
        Some(b) => {
            let rows = py
                .detach(|| batch::ab_join_topk(a, b, m, scale, threads, k))
                .map_err(capacity_error)?;
            let (distances, indices) =
                topk_arrays_of(py, rows.iter().map(|(_, kept)| kept), k, convert)?;
            let none = vec![-1i64; rows.len()];
            Ok((
                distances,
                indices,
                PyArray1::from_vec(py, none.clone()),
                PyArray1::from_vec(py, none),
            ))
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(aamp, module)?)?;
    Ok(())
}
