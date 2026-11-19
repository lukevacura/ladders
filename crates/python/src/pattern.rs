//! Pattern matching: a [`PatternLibrary`] of equal-length queries, matched against a whole
//! series with `distance_profile` or against a stream with [`PatternStream`], and `match` /
//! `aamp_match`, STUMPY's whole-series matching of one query.

use ladde::entry::Entry;
use ladde::pattern::{
    MatchRules, MaxDistance, PatternLibrary as CoreLibrary, PatternStream as CoreStream, QueryId,
    RegisterError,
};
use ladde::quantize::Scale;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aamp::{require_euclidean, shared_scale};
use crate::batch::resolve_scale;
use crate::constant::{flags_of, series as flagged};
use crate::convert::{
    Floats, IndexDistanceArrays, entry_arrays, index_and_distance, nonnorm_index_and_distance,
};
use crate::err::{Remedy, ScaleArg, countable, require_none, tick_scale, tolerance, window_length};

fn register_error(error: RegisterError) -> PyErr {
    PyValueError::new_err(match error {
        RegisterError::WrongLength => "query must be exactly m samples long",
        RegisterError::NonFinite => "query must not contain NaN or infinite samples",
        RegisterError::Unquantizable => "no scale quantizes this query within the supported range",
        // The enum is non-exhaustive: a variant this file does not know is named as core names it.
        _ => return PyValueError::new_err(format!("the input was refused: {error:?}")),
    })
}

fn bad_query() -> PyErr {
    PyValueError::new_err("query must be non-empty and contain only finite samples")
}

/// Queries of one length `m`, matched against series quantized at `scale`.
#[pyclass(module = "ladders")]
pub struct PatternLibrary {
    inner: CoreLibrary,
    ids: Vec<QueryId>,
    queries: Vec<Vec<f64>>,
    m: usize,
    scale: Scale,
}

#[pymethods]
impl PatternLibrary {
    #[new]
    fn new(m: i64, scale: f64) -> PyResult<Self> {
        let m = window_length(m)?;
        let scale = tick_scale(scale)?;
        Ok(Self {
            inner: CoreLibrary::new(m, scale),
            ids: Vec::new(),
            queries: Vec::new(),
            m: m.get(),
            scale,
        })
    }

    /// Adds a query of exactly `m` finite samples and returns its id (0, 1, ... in
    /// registration order). Raises `ValueError` for a query of the wrong length, with a
    /// non-finite sample, or that no scale can quantize.
    fn register(&mut self, query: Floats<'_>) -> PyResult<usize> {
        let query = query.as_array().to_vec();
        let id = self.inner.register(&query).map_err(register_error)?;
        self.ids.push(id);
        self.queries.push(query);
        Ok(self.ids.len() - 1)
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }

    /// `(indices, distances)` of query `id` against every window of `series`: entry `i` is for
    /// `series[i : i + m]`. A window with a gap, or too volatile for the arithmetic, gets index
    /// `-1` and distance `inf`.
    fn distance_profile<'py>(
        &self,
        py: Python<'py>,
        id: i64,
        series: Floats<'py>,
    ) -> PyResult<IndexDistanceArrays<'py>> {
        let id = usize::try_from(id)
            .ok()
            .and_then(|i| self.ids.get(i))
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("no query with id {id}")))?;
        let series = series.as_array().to_vec();
        countable(&series, self.scale, Remedy::Library)?;
        let entries = py
            .detach(|| self.inner.distance_profile(id, &series))
            .expect("the id came from this library");
        Ok(entry_arrays(py, entries.into_iter()))
    }
}

/// `(indices, distances)` of `query` against every window of `series`, as `PatternLibrary`'s
/// `distance_profile` but for one ad hoc query (any non-empty length, finite samples).
/// `T_subseq_isconstant` and `Q_subseq_isconstant` are STUMPY's (see `match`): which windows of
/// the series, and whether the query, are constant.
#[pyfunction]
#[pyo3(signature = (query, series, scale, T_subseq_isconstant=None, Q_subseq_isconstant=None))]
#[allow(non_snake_case)]
pub fn distance_profile<'py>(
    py: Python<'py>,
    query: Floats<'py>,
    series: Floats<'py>,
    scale: f64,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    Q_subseq_isconstant: Option<&Bound<'py, PyAny>>,
) -> PyResult<IndexDistanceArrays<'py>> {
    let scale = tick_scale(scale)?;
    let query = query.as_array().to_vec();
    let series = series.as_array().to_vec();
    countable(&series, scale, Remedy::Pass)?;
    countable(&query, scale, Remedy::Pass)?;
    let m = core::num::NonZeroUsize::new(query.len()).ok_or_else(bad_query)?;
    let (name_t, name_q) = ("T_subseq_isconstant", "Q_subseq_isconstant");
    let (flags_t, flags_q) = (
        flags_of(py, T_subseq_isconstant, &series, m, name_t)?,
        flags_of(py, Q_subseq_isconstant, &query, m, name_q)?,
    );
    let (input_t, input_q) = (
        flagged(&series, m, &flags_t, name_t)?,
        flagged(&query, m, &flags_q, name_q)?,
    );
    let entries = py
        .detach(|| ladde::pattern::distance_profile(input_q, input_t, scale))
        .ok_or_else(bad_query)?;
    Ok(entry_arrays(py, entries.into_iter()))
}

/// STUMPY's `(distances, indices)` of the matches, nearest first.
type MatchArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>);

fn match_arrays<'py, H: Copy>(
    py: Python<'py>,
    matches: &[(u64, H)],
    convert: impl Fn(H) -> (i64, f64),
) -> MatchArrays<'py> {
    let (indices, distances): (Vec<i64>, Vec<f64>) =
        matches.iter().map(|&(_, hit)| convert(hit)).unzip();
    (
        PyArray1::from_vec(py, distances),
        PyArray1::from_vec(py, indices),
    )
}

fn max_distance_of(max_distance: Option<f64>) -> MaxDistance {
    max_distance.map_or(MaxDistance::Default, MaxDistance::Value)
}

fn max_matches_of(max_matches: Option<i64>) -> PyResult<Option<usize>> {
    max_matches
        .map(|k| {
            usize::try_from(k).map_err(|_| {
                PyValueError::new_err(format!("max_matches must be at least 0, got {k}"))
            })
        })
        .transpose()
}

fn bad_match_query() -> PyErr {
    PyValueError::new_err(
        "Q must be finite samples and no longer than T, T must be long enough to hold it, and query_idx must start a window of T",
    )
}

fn match_rules(atol: f64, query_idx: Option<i64>) -> PyResult<MatchRules> {
    let query_idx = query_idx
        .map(|at| {
            usize::try_from(at).map_err(|_| {
                PyValueError::new_err(format!("query_idx must not be negative, got {at}"))
            })
        })
        .transpose()?;
    Ok(MatchRules {
        atol: tolerance(atol)?,
        query_idx,
    })
}

/// The windows of `T` within `max_distance` of the query `Q`, as STUMPY's `match` finds them:
/// nearest first, none within `ceil(m/4)` of a nearer one, at most `max_matches` of them
/// (`None` keeps every one). `max_distance` defaults to `max(mean(D) - 2 std(D), min(D))` over
/// the finite distances `D` of the distance profile, so the nearest window always qualifies; it
/// must be a number (not a function). A window qualifies when its distance is at most
/// `max_distance + atol` (STUMPY's `atol`, default `1e-8`; a negative or NaN `atol` raises
/// `ValueError` where STUMPY would add it as it is). A window with a gap never matches.
///
/// `query_idx`, when `Q` is the window of `T` starting there, takes that window first at distance
/// zero, as STUMPY does (and it counts as zero in the default threshold); a `query_idx` that
/// starts no window of `T` raises `ValueError`. STUMPY warns when `Q` differs from that window;
/// this does not check. For `normalize=False` STUMPY does not force the distance to zero but
/// keeps the window's own, so there a `Q` that is not the window at `query_idx` gets zero here
/// and its true distance in STUMPY; for the window itself they agree.
///
/// Returns `(distances, indices)`, two arrays of one length in STUMPY's order (not its single
/// object array). Every order between windows is decided on exact distances, ties to the smaller
/// index. `Q` must be at least 3 finite samples and no longer than `T`. `scale` (keyword-only) is
/// the tick size `T` is quantized to, inferred from `T` as `matrix_profile` infers it when `None`.
/// With `normalize=False` this is `aamp_match`, and `p` is its `p`; with the default
/// `normalize=True`, `p` is ignored, as in STUMPY.
///
/// `Q` and `T` take anything NumPy turns into a one-dimensional float array (lists and integer
/// arrays included). `M_T`, `Sigma_T` (STUMPY's `Σ_T`, spelled in ASCII because Python cannot
/// read a builtin signature with other characters) and `T_subseq_isfinite` (precomputed window
/// statistics and finiteness, which STUMPY takes to save work) raise `NotImplementedError` when
/// given: the statistics are always computed exactly here. `max_distance` cannot be a function. With
/// `normalize=False`, STUMPY fills NaN and infinite samples of `T` with 0 before matching, so
/// windows with gaps can match and shift the default threshold; here a window holding a gap never
/// matches.
///
/// `T_subseq_isconstant` is STUMPY's: a boolean array with one flag per window of `T`, or a
/// function `f(a, w)` returning one (given `T`, non-finite samples as NaN, and `m = len(Q)`),
/// saying which windows are constant, in place of "all samples equal"; `Q_subseq_isconstant` is
/// the same for `Q`, whose one window it flags (an array of length 1, or a function). Two
/// constant windows are at distance 0 and a constant one is at `sqrt(m)` from any other. With
/// `normalize=False` both are accepted and ignored, as in STUMPY.
#[pyfunction(name = "match")]
#[pyo3(signature = (
    Q, T, M_T=None, Sigma_T=None, max_distance=None, max_matches=None, atol=1e-8, query_idx=None,
    normalize=true, p=2.0, T_subseq_isfinite=None, T_subseq_isconstant=None,
    Q_subseq_isconstant=None, *, scale=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn find_match<'py>(
    py: Python<'py>,
    Q: Floats<'py>,
    T: Floats<'py>,
    M_T: Option<&Bound<'py, PyAny>>,
    Sigma_T: Option<&Bound<'py, PyAny>>,
    max_distance: Option<f64>,
    max_matches: Option<i64>,
    atol: f64,
    query_idx: Option<i64>,
    normalize: bool,
    p: f64,
    T_subseq_isfinite: Option<&Bound<'py, PyAny>>,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    Q_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scale: Option<ScaleArg>,
) -> PyResult<MatchArrays<'py>> {
    require_none("M_T", M_T)?;
    require_none("Sigma_T", Sigma_T)?;
    if !normalize {
        return aamp_match(
            py,
            Q,
            T,
            T_subseq_isfinite,
            max_distance,
            max_matches,
            atol,
            query_idx,
            p,
            scale,
        );
    }
    require_none("T_subseq_isfinite", T_subseq_isfinite)?;
    let query = Q.as_array().to_vec();
    let series = T.as_array().to_vec();
    let m = window_length(query.len() as i64)?;
    let scale = resolve_scale(&series, m, scale)?;
    let max_matches = max_matches_of(max_matches)?;
    let rules = match_rules(atol, query_idx)?;
    let (name_t, name_q) = ("T_subseq_isconstant", "Q_subseq_isconstant");
    let (flags_t, flags_q) = (
        flags_of(py, T_subseq_isconstant, &series, m, name_t)?,
        flags_of(py, Q_subseq_isconstant, &query, m, name_q)?,
    );
    let (input_t, input_q) = (
        flagged(&series, m, &flags_t, name_t)?,
        flagged(&query, m, &flags_q, name_q)?,
    );
    let matches = py
        .detach(|| {
            ladde::pattern::find_matches_with(
                input_q,
                input_t,
                scale,
                max_distance_of(max_distance),
                max_matches,
                rules,
            )
        })
        .ok_or_else(bad_match_query)?;
    Ok(match_arrays(py, &matches, index_and_distance))
}

/// STUMPY's `aamp_match`: `match` under the Euclidean distance on the raw values. `Q` and `T` are
/// quantized to one `scale` (keyword-only; inferred from both together as `aamp` infers it when
/// `None`), so a `Q` off that grid is rounded to it; distances, `max_distance` and `atol` are in
/// the caller's units. `p` must be 2.0; any other value raises `NotImplementedError`, as does a
/// given `T_subseq_isfinite`. `atol` and `query_idx` are as in `match`.
#[pyfunction]
#[pyo3(signature = (
    Q, T, T_subseq_isfinite=None, max_distance=None, max_matches=None, atol=1e-8, query_idx=None,
    p=2.0, *, scale=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn aamp_match<'py>(
    py: Python<'py>,
    Q: Floats<'py>,
    T: Floats<'py>,
    T_subseq_isfinite: Option<&Bound<'py, PyAny>>,
    max_distance: Option<f64>,
    max_matches: Option<i64>,
    atol: f64,
    query_idx: Option<i64>,
    p: f64,
    scale: Option<ScaleArg>,
) -> PyResult<MatchArrays<'py>> {
    require_euclidean(p)?;
    require_none("T_subseq_isfinite", T_subseq_isfinite)?;
    let query = Q.as_array().to_vec();
    let series = T.as_array().to_vec();
    let m = window_length(query.len() as i64)?;
    let scale = shared_scale(&series, Some(&query), m, scale)?;
    let max_matches = max_matches_of(max_matches)?;
    let rules = match_rules(atol, query_idx)?;
    let matches = py
        .detach(|| {
            ladde::pattern::find_matches_nonnorm_with(
                &query,
                &series,
                scale,
                max_distance_of(max_distance),
                max_matches,
                rules,
            )
        })
        .ok_or_else(bad_match_query)?;
    Ok(match_arrays(py, &matches, |hit| {
        nonnorm_index_and_distance(hit, scale)
    }))
}

/// A library's queries matched against a stream one sample at a time: after each sample, one
/// distance per query for the window ending there. Until `m` samples have arrived, and for a
/// window with a gap, a distance is `inf` (as for `left_profile`'s unmatched rows).
///
/// The stream is quantized at the library's scale; the library may keep growing afterwards
/// without affecting this stream.
#[pyclass(module = "ladders")]
pub struct PatternStream {
    inner: CoreStream,
    queries: usize,
    scale: Scale,
}

#[pymethods]
impl PatternStream {
    #[new]
    fn new(library: &PatternLibrary) -> Self {
        let mut copy = CoreLibrary::new(
            core::num::NonZeroUsize::new(library.m).expect("m >= 3"),
            library.scale,
        );
        for query in &library.queries {
            copy.register(query)
                .expect("the query registered once registers again");
        }
        Self {
            inner: CoreStream::new(copy),
            queries: library.queries.len(),
            scale: library.scale,
        }
    }

    /// Folds in one sample; returns one distance per query, in registration order.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        x: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        countable(&[x], self.scale, Remedy::Restart)?;
        Ok(numpy::PyArray1::from_vec(py, self.push(x)))
    }

    /// Folds in every sample of `xs`; returns a `(len(xs), queries)` array whose row `t` is
    /// what `update` returns for sample `t`.
    fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        xs: Floats<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let xs = xs.as_array().to_vec();
        countable(&xs, self.scale, Remedy::Restart)?;
        let flat = py.detach(|| {
            let mut flat = Vec::with_capacity(xs.len() * self.queries);
            for &x in &xs {
                flat.extend(self.push(x));
            }
            flat
        });
        numpy::PyArray1::from_vec(py, flat).reshape([xs.len(), self.queries])
    }
}

impl PatternStream {
    fn push(&mut self, x: f64) -> Vec<f64> {
        let entries: &[Entry] = self.inner.push(x);
        if entries.is_empty() {
            return vec![f64::INFINITY; self.queries];
        }
        entries.iter().map(|&e| index_and_distance(e).1).collect()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PatternLibrary>()?;
    module.add_class::<PatternStream>()?;
    module.add_function(wrap_pyfunction!(distance_profile, module)?)?;
    module.add_function(wrap_pyfunction!(find_match, module)?)?;
    module.add_function(wrap_pyfunction!(aamp_match, module)?)?;
    Ok(())
}
