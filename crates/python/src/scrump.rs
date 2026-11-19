//! `scrump` and `scraamp`: STUMPY's anytime matrix profile classes ([`ladde::scrump`]),
//! which visit a random subset of the diagonals and refine the profile with `update()`.

use ladde::entry::Entry;
use ladde::quantize::Scale;
use ladde::scrump::{Explicit, Scraamp, ScraampJoin, Scrump, ScrumpError, ScrumpJoin};
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::aamp::{require_euclidean, shared_scale};
use crate::batch::{resolve_scale, resolve_threads};
use crate::convert::{index_and_distance, nonnorm_index_and_distance};
use crate::err::{ScaleArg, bounded_cells, window_length};

/// `percentage`, `m` or the order that core refused, as the `ValueError` it raises.
fn scrump_error(error: ScrumpError) -> PyErr {
    PyValueError::new_err(match error {
        ScrumpError::Percentage => "percentage must be above 0 and at most 1",
        ScrumpError::NoDiagonals => {
            "the window size is too long: there is no pair of windows to visit"
        }
        ScrumpError::NotAPermutation => "the order is not a permutation of the diagonals",
        ScrumpError::Capacity => "the profile is too large to allocate; use a smaller k or series",
        // The enum is non-exhaustive: a variant this file does not know is named as core names it.
        _ => return PyValueError::new_err(format!("the input was refused: {error:?}")),
    })
}

/// STUMPY's own draw of the visiting order, `np.random.permutation(range(first, first + count))`
/// as `int64`, from NumPy's global state, so `np.random.seed(x)` gives the same order as STUMPY.
fn draw_order(py: Python<'_>, first: i64, count: usize) -> PyResult<Vec<i64>> {
    let range = py
        .import("builtins")?
        .getattr("range")?
        .call1((first, first + count as i64))?;
    let permuted = py
        .import("numpy")?
        .getattr("random")?
        .getattr("permutation")?
        .call1((range,))?
        .call_method1("astype", ("int64",))?;
    let order: PyReadonlyArray1<'_, i64> = permuted.extract()?;
    Ok(order.as_array().to_vec())
}

/// The anytime profile a Python object holds, whichever metric.
enum Held {
    ZNorm(Scrump),
    NonNorm(Scraamp, Scale),
    ZNormJoin(ScrumpJoin),
    NonNormJoin(ScraampJoin, Scale),
}

impl Held {
    fn update(&mut self) -> Result<(), ScrumpError> {
        match self {
            Self::ZNorm(held) => held.update(),
            Self::NonNorm(held, _) => held.update(),
            Self::ZNormJoin(held) => held.update(),
            Self::NonNormJoin(held, _) => held.update(),
        }
    }

    fn visited(&self) -> &[i64] {
        match self {
            Self::ZNorm(held) => held.visited(),
            Self::NonNorm(held, _) => held.visited(),
            Self::ZNormJoin(held) => held.visited(),
            Self::NonNormJoin(held, _) => held.visited(),
        }
    }

    fn updates(&self) -> usize {
        match self {
            Self::ZNorm(held) => held.updates(),
            Self::NonNorm(held, _) => held.updates(),
            Self::ZNormJoin(held) => held.updates(),
            Self::NonNormJoin(held, _) => held.updates(),
        }
    }

    fn chunks(&self) -> usize {
        match self {
            Self::ZNorm(held) => held.chunks(),
            Self::NonNorm(held, _) => held.chunks(),
            Self::ZNormJoin(held) => held.chunks(),
            Self::NonNormJoin(held, _) => held.chunks(),
        }
    }
}

/// STUMPY's four columns, flat: `P` and `I` as `windows * k` row-major values.
struct Columns {
    distances: Vec<f64>,
    indices: Vec<i64>,
    left: Vec<i64>,
    right: Vec<i64>,
}

/// An approximate matrix profile that improves with every `update()`, STUMPY's `scrump`.
///
/// Each `update()` visits the next share of the diagonals of the distance matrix, taken in a
/// random order, about `percentage` of all the distances per update, and `P_`, `I_`, `left_I_`
/// and `right_I_` are the exact profile over the diagonals visited so far: exactly the matrix
/// profile restricted to those pairs, on bits, whatever `threads` is. Once every diagonal is
/// visited they are the full profile, and further updates change nothing. Nothing is visited
/// before the first `update()`, so until then `P_` is `inf` and `I_` is `-1`, as in STUMPY,
/// except that with `normalize=True` the pairs of a window whose variance is too large for the
/// fast passes (every sample present) are ranked exactly when the object is made, so they are in
/// the profile from the start and the profile is exact over them and the diagonals visited.
///
/// The order is drawn exactly as STUMPY draws it, with `np.random.permutation` on NumPy's global
/// state, so `np.random.seed(x)` visits the same diagonals as STUMPY in the same chunks.
///
/// Differences from STUMPY: a `percentage` that is NaN, not above 0 or above 1 raises
/// `ValueError` (STUMPY clips one above 1 to a single chunk and fails on 0); `pre_scrump=True`
/// raises `NotImplementedError` (`s` is accepted and ignored without it, as in STUMPY);
/// `T_A_subseq_isconstant` and `T_B_subseq_isconstant` other than `None` raise
/// `NotImplementedError` (a window is constant when its samples are equal); exact ties go to the
/// smaller index whichever update finds them, where STUMPY keeps the neighbor an earlier update
/// found, and the distances are exact, so the two differ on near ties. With `normalize=False`
/// this is `scraamp` (as STUMPY re-routes it; the object is still a `scrump`), and `p` is its
/// `p`; with the default `normalize=True`, `p` is ignored, as in STUMPY.
///
/// A series `T_A` is self-joined unless `T_B` is given and differs from it. The self-join keeps
/// `k` nearest neighbors per window over both sides (`P_` and `I_` have shape `(windows, k)` for
/// `k > 1`, padded with `inf` and `-1`), and `left_I_` and `right_I_` each side's nearest. With a
/// `T_B` that differs from `T_A` (or `ignore_trivial=False`) it is an AB-join: each window of
/// `T_A` keeps its `k` nearest windows of `T_B`, with no exclusion zone, `I_` indexes `T_B`, and
/// `left_I_` and `right_I_` are all `-1`, as in STUMPY. `k < 1`, or `windows * k` above 2**28,
/// raises `ValueError`. Extra to STUMPY, keyword-only: `scale`, the tick size the series is
/// quantized to (inferred as `matrix_profile` infers it when `None`), `scale_b`, the tick size of
/// `T_B` in a z-normalized AB-join (each series takes its own, as in `ab_join`; `ValueError`
/// anywhere else), and `threads`, which means what it does in `matrix_profile`. `update()`
/// releases the GIL.
///
/// Also extra: `visited_`, the diagonals visited so far in order, `n_updates_`, the updates run,
/// and `n_chunks_`, the updates that visit every diagonal.
#[pyclass(module = "ladders", name = "scrump", subclass)]
pub struct PyScrump {
    held: Held,
    k: usize,
}

/// The arguments both classes share, once parsed.
struct Request<'a, 'py> {
    a: PyArrayLike1<'py, f64, AllowTypeChange>,
    m: i64,
    b: Option<PyArrayLike1<'py, f64, AllowTypeChange>>,
    ignore_trivial: bool,
    percentage: f64,
    k: i64,
    scale: Option<ScaleArg>,
    scale_b: Option<ScaleArg>,
    threads: Option<&'a Bound<'py, PyAny>>,
}

/// `k` as a count: at least 1, else `ValueError`.
fn resolve_k(k: i64) -> PyResult<core::num::NonZeroUsize> {
    usize::try_from(k)
        .ok()
        .and_then(core::num::NonZeroUsize::new)
        .ok_or_else(|| PyValueError::new_err(format!("k must be at least 1, got {k}")))
}

fn make<'py>(py: Python<'py>, request: Request<'_, 'py>, normalize: bool) -> PyResult<PyScrump> {
    let Request {
        a,
        m,
        b,
        ignore_trivial,
        percentage,
        k,
        scale,
        scale_b,
        threads,
    } = request;
    let m = window_length(m)?;
    let k = resolve_k(k)?;
    if !normalize && k.get() > 1 {
        return Err(PyNotImplementedError::new_err(
            "k above 1 is not available for normalize=False yet",
        ));
    }
    let threads = resolve_threads(threads)?;
    let a = a.as_array().to_vec();
    let b = b.map(|b| b.as_array().to_vec());
    if b.is_none() && !ignore_trivial {
        return Err(PyValueError::new_err(
            "ignore_trivial must be True for a self-join (T_B is None)",
        ));
    }
    let join_with = b.filter(|b| !(ignore_trivial && *b == a));
    if scale_b.is_some() && !(normalize && join_with.is_some()) {
        return Err(PyValueError::new_err(
            "scale_b applies only to a z-normalized AB-join",
        ));
    }
    let range = match &join_with {
        None => Scrump::diagonals(a.len(), m),
        Some(b) => ScrumpJoin::diagonals(a.len(), b.len(), m),
    };
    let (first, count) = range.map_err(scrump_error)?;
    bounded_cells("k", (a.len() + 1).saturating_sub(m.get()), k.get())?;
    let order = Explicit(draw_order(py, first, count)?);
    let held = match (normalize, join_with) {
        (true, None) => {
            let scale = resolve_scale(&a, m, scale)?;
            Held::ZNorm(
                Scrump::new(&a, m, scale, threads, k, percentage, order).map_err(scrump_error)?,
            )
        }
        (false, None) => {
            let scale = shared_scale(&a, None, m, scale)?;
            Held::NonNorm(
                Scraamp::new(&a, m, scale, threads, percentage, order).map_err(scrump_error)?,
                scale,
            )
        }
        (true, Some(b)) => {
            let scales = (resolve_scale(&a, m, scale)?, resolve_scale(&b, m, scale_b)?);
            Held::ZNormJoin(
                ScrumpJoin::new(&a, &b, m, scales, threads, k, percentage, order)
                    .map_err(scrump_error)?,
            )
        }
        (false, Some(b)) => {
            let scale = shared_scale(&a, Some(&b), m, scale)?;
            Held::NonNormJoin(
                ScraampJoin::new(&a, &b, m, scale, threads, percentage, order)
                    .map_err(scrump_error)?,
                scale,
            )
        }
    };
    Ok(PyScrump { held, k: k.get() })
}

impl PyScrump {
    fn columns(&self) -> PyResult<Columns> {
        let width = self.k;
        match &self.held {
            Held::ZNorm(held) => {
                let rows = held.profile().map_err(scrump_error)?;
                let mut columns = Columns {
                    distances: Vec::with_capacity(rows.len() * width),
                    indices: Vec::with_capacity(rows.len() * width),
                    left: Vec::with_capacity(rows.len()),
                    right: Vec::with_capacity(rows.len()),
                };
                for (_, sides) in &rows {
                    for slot in 0..width {
                        let entry = sides.nearest.get(slot).copied().unwrap_or(Entry::NONE);
                        let (index, distance) = index_and_distance(entry);
                        columns.indices.push(index);
                        columns.distances.push(distance);
                    }
                    columns.left.push(index_and_distance(sides.left).0);
                    columns.right.push(index_and_distance(sides.right).0);
                }
                Ok(columns)
            }
            Held::ZNormJoin(held) => {
                let rows = held.profile().map_err(scrump_error)?;
                let mut columns = Columns {
                    distances: Vec::with_capacity(rows.len() * width),
                    indices: Vec::with_capacity(rows.len() * width),
                    left: vec![-1; rows.len()],
                    right: vec![-1; rows.len()],
                };
                for (_, entries) in &rows {
                    for slot in 0..width {
                        let entry = entries.get(slot).copied().unwrap_or(Entry::NONE);
                        let (index, distance) = index_and_distance(entry);
                        columns.indices.push(index);
                        columns.distances.push(distance);
                    }
                }
                Ok(columns)
            }
            Held::NonNormJoin(held, scale) => {
                let rows = held.profile().map_err(scrump_error)?;
                let mut columns = Columns {
                    distances: Vec::with_capacity(rows.len()),
                    indices: Vec::with_capacity(rows.len()),
                    left: vec![-1; rows.len()],
                    right: vec![-1; rows.len()],
                };
                for (_, nearest) in &rows {
                    let (index, distance) = nonnorm_index_and_distance(*nearest, *scale);
                    columns.indices.push(index);
                    columns.distances.push(distance);
                }
                Ok(columns)
            }
            Held::NonNorm(held, scale) => {
                let rows = held.profile().map_err(scrump_error)?;
                let mut columns = Columns {
                    distances: Vec::with_capacity(rows.len()),
                    indices: Vec::with_capacity(rows.len()),
                    left: Vec::with_capacity(rows.len()),
                    right: Vec::with_capacity(rows.len()),
                };
                for (_, sides) in &rows {
                    let (index, distance) = nonnorm_index_and_distance(sides.nearest, *scale);
                    columns.indices.push(index);
                    columns.distances.push(distance);
                    columns
                        .left
                        .push(nonnorm_index_and_distance(sides.left, *scale).0);
                    columns
                        .right
                        .push(nonnorm_index_and_distance(sides.right, *scale).0);
                }
                Ok(columns)
            }
        }
    }

    /// `values` as STUMPY returns a top-`k` column: 1-D for `k` of one, else `(windows, k)`.
    fn shaped<'py, T: numpy::Element>(
        &self,
        py: Python<'py>,
        values: Vec<T>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let windows = values.len() / self.k;
        let flat = PyArray1::from_vec(py, values);
        if self.k == 1 {
            return Ok(flat.into_any());
        }
        Ok(flat.reshape([windows, self.k])?.into_any())
    }
}

#[pymethods]
impl PyScrump {
    #[new]
    #[pyo3(signature = (T_A, m, T_B=None, ignore_trivial=true, percentage=0.01, pre_scrump=false, s=None, normalize=true, p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None, *, scale=None, scale_b=None, threads=None))]
    #[allow(clippy::too_many_arguments, unused_variables, non_snake_case)]
    fn new<'py>(
        py: Python<'py>,
        T_A: PyArrayLike1<'py, f64, AllowTypeChange>,
        m: i64,
        T_B: Option<PyArrayLike1<'py, f64, AllowTypeChange>>,
        ignore_trivial: bool,
        percentage: f64,
        pre_scrump: bool,
        s: Option<i64>,
        normalize: bool,
        p: f64,
        k: i64,
        T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
        T_B_subseq_isconstant: Option<&Bound<'py, PyAny>>,
        scale: Option<ScaleArg>,
        scale_b: Option<ScaleArg>,
        threads: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        if pre_scrump {
            return Err(PyNotImplementedError::new_err(
                "pre_scrump is not available yet",
            ));
        }
        let constant = T_A_subseq_isconstant.is_some_and(|c| !c.is_none())
            || T_B_subseq_isconstant.is_some_and(|c| !c.is_none());
        if normalize && constant {
            return Err(PyNotImplementedError::new_err(
                "T_A_subseq_isconstant and T_B_subseq_isconstant are not supported: windows are constant by exact comparison",
            ));
        }
        if !normalize {
            require_euclidean(p)?;
            if constant {
                return Err(PyTypeError::new_err(
                    "T_A_subseq_isconstant and T_B_subseq_isconstant do not apply to normalize=False",
                ));
            }
        }
        make(
            py,
            Request {
                a: T_A,
                m,
                b: T_B,
                ignore_trivial,
                percentage,
                k,
                scale,
                scale_b,
                threads,
            },
            normalize,
        )
    }

    /// Visits the next share of the diagonals; does nothing once all are visited.
    fn update(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.held.update()).map_err(scrump_error)
    }

    #[getter(P_)]
    fn distances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.shaped(py, self.columns()?.distances)
    }

    #[getter(I_)]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.shaped(py, self.columns()?.indices)
    }

    #[getter(left_I_)]
    fn left<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_vec(py, self.columns()?.left))
    }

    #[getter(right_I_)]
    fn right<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_vec(py, self.columns()?.right))
    }

    #[getter(visited_)]
    fn visited<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, self.held.visited())
    }

    #[getter(n_updates_)]
    fn updates(&self) -> usize {
        self.held.updates()
    }

    #[getter(n_chunks_)]
    fn chunks(&self) -> usize {
        self.held.chunks()
    }
}

/// STUMPY's `scraamp`: [`scrump`](PyScrump) under the Euclidean distance on the raw values
/// (every `scrump` method and attribute), with `p` of 2.0 (any other raises
/// `NotImplementedError`) and `k` of 1 (a larger one raises `NotImplementedError` for now).
/// `P_` is in the series' units; `scale` is the tick size the series is quantized to, inferred
/// as `aamp` infers it when `None`. `pre_scraamp=True` raises `NotImplementedError`.
#[pyclass(module = "ladders", name = "scraamp", extends = PyScrump)]
pub struct PyScraamp;

#[pymethods]
impl PyScraamp {
    #[new]
    #[pyo3(signature = (T_A, m, T_B=None, ignore_trivial=true, percentage=0.01, pre_scraamp=false, s=None, p=2.0, k=1, *, scale=None, threads=None))]
    #[allow(clippy::too_many_arguments, unused_variables, non_snake_case)]
    fn new<'py>(
        py: Python<'py>,
        T_A: PyArrayLike1<'py, f64, AllowTypeChange>,
        m: i64,
        T_B: Option<PyArrayLike1<'py, f64, AllowTypeChange>>,
        ignore_trivial: bool,
        percentage: f64,
        pre_scraamp: bool,
        s: Option<i64>,
        p: f64,
        k: i64,
        scale: Option<ScaleArg>,
        threads: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        if pre_scraamp {
            return Err(PyNotImplementedError::new_err(
                "pre_scraamp is not available yet",
            ));
        }
        require_euclidean(p)?;
        let base = make(
            py,
            Request {
                a: T_A,
                m,
                b: T_B,
                ignore_trivial,
                percentage,
                k,
                scale,
                scale_b: None,
                threads,
            },
            false,
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyScrump>()?;
    module.add_class::<PyScraamp>()?;
    Ok(())
}
