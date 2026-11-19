//! `mmotifs` and `aamp_mmotifs`: the multi-dimensional motifs of a series from its profile
//! ([`ladde::motif::mmotifs`], [`ladde::motif::mmotifs_nonnorm`]) in the shape
//! and padding of STUMPY's. As in STUMPY, `mmotifs(..., normalize=False)` is `aamp_mmotifs`.

use ladde::index::SampleIndex;
use ladde::motif::{Cutoffs, MultiMotif, MultiMotifParams};
use ladde::multidim::MultiError;
use ladde::pattern::MaxDistance;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::aamp::require_euclidean;
use crate::constant::{flags_of_each, series as flagged};
use crate::convert::Ints;
use crate::convert::{FloatRows, IntRows};
use crate::err::{ScaleArg, multi_error, tolerance, window_length};
use crate::motif::{count, motif_limit};
use crate::multi::{dimension_scales, one_scale, shared_scale};
use crate::subspace::{include_dimensions, reject_unsupported, rows};

/// STUMPY's `(motif_distances, motif_indices, motif_subspaces, motif_mdls)`.
type MmotifsResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyList>,
    Bound<'py, PyList>,
);

/// The arguments of `mmotifs` once parsed.
struct Request {
    m: core::num::NonZeroUsize,
    distance: Vec<Vec<f64>>,
    index: Vec<Vec<SampleIndex>>,
    min_neighbors: usize,
    max_distance: MaxDistance,
    cutoffs: Option<Vec<f64>>,
    cutoff: Option<f64>,
    max_matches: Option<usize>,
    max_motifs: usize,
    atol: f64,
    k: Option<i64>,
    include: Vec<usize>,
}

impl Request {
    /// `T`'s window length is what `P` leaves of it, as STUMPY infers `m`.
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        series: &[Vec<f64>],
        p: &FloatRows<'py>,
        i: &IntRows<'py>,
        min_neighbors: i64,
        max_distance: Option<f64>,
        cutoffs: Option<&Bound<'py, PyAny>>,
        max_matches: Option<i64>,
        max_motifs: i64,
        atol: f64,
        k: Option<i64>,
        include: Option<Ints<'py>>,
    ) -> PyResult<Self> {
        let n = series.first().map_or(0, Vec::len);
        let windows = p.as_array().ncols();
        if windows == 0 || windows > n {
            return Err(multi_error(MultiError::ProfileShape));
        }
        let (cutoff, cutoffs) = match cutoffs {
            None => (None, None),
            Some(cutoffs) => match cutoffs.extract::<f64>() {
                Ok(one) => (Some(one), None),
                Err(_) => (None, Some(cutoffs.extract::<Vec<f64>>()?)),
            },
        };
        Ok(Self {
            m: window_length((n - windows + 1) as i64)?,
            distance: p
                .as_array()
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            index: i
                .as_array()
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|&at| neighbor(at)).collect())
                .collect::<PyResult<_>>()?,
            min_neighbors: count("min_neighbors", min_neighbors)?,
            max_distance: max_distance.map_or(MaxDistance::Default, MaxDistance::Value),
            cutoffs,
            cutoff,
            max_matches: max_matches.map(|k| count("max_matches", k)).transpose()?,
            max_motifs: motif_limit(max_motifs)?,
            atol: tolerance(atol)?,
            k,
            include: include_dimensions(include)?,
        })
    }

    /// STUMPY indexes the subspaces with `k`, so a negative one counts from the last dimension.
    fn params(&self) -> PyResult<MultiMotifParams<'_>> {
        let dimensions = self.distance.len() as i64;
        let k = match self.k {
            None => None,
            Some(k) if k >= 0 => Some(k as usize),
            Some(k) if k + dimensions >= 0 => Some((k + dimensions) as usize),
            Some(_) => return Err(multi_error(MultiError::KOutOfRange)),
        };
        Ok(MultiMotifParams {
            min_neighbors: self.min_neighbors,
            max_distance: self.max_distance,
            cutoffs: match (&self.cutoffs, self.cutoff) {
                (Some(each), _) => Cutoffs::PerRow(each),
                (None, Some(one)) => Cutoffs::All(one),
                (None, None) => Cutoffs::Default,
            },
            max_matches: self.max_matches,
            max_motifs: self.max_motifs,
            atol: self.atol,
            k,
            include: &self.include,
        })
    }
}

/// An entry of `I`: a window start, or `-1` for none. STUMPY slices `T` from its end with any
/// other negative entry, which is a `ValueError` here.
fn neighbor(at: i64) -> PyResult<SampleIndex> {
    match u64::try_from(at) {
        Ok(start) => Ok(SampleIndex::new(start).unwrap_or(SampleIndex::NONE)),
        Err(_) if at == -1 => Ok(SampleIndex::NONE),
        Err(_) => Err(PyValueError::new_err(format!(
            "I holds {at}, a negative window start; STUMPY slices T from its end with it; pass -1 \
             for no neighbor, or len(T) + ({at}) for the window it reads"
        ))),
    }
}

/// The motifs as STUMPY's padded arrays and lists; with no motif the arrays have shape `(1, 0)`.
fn arrays<'py>(py: Python<'py>, motifs: &[MultiMotif]) -> MmotifsResult<'py> {
    let width = motifs.iter().map(|m| m.matches.len()).max().unwrap_or(0);
    let height = motifs.len().max(1);
    let mut distances = Array2::from_elem((height, width), f64::NAN);
    let mut indices = Array2::from_elem((height, width), -1_i64);
    for (row, motif) in motifs.iter().enumerate() {
        for (column, &(start, distance)) in motif.matches.iter().enumerate() {
            distances[[row, column]] = distance;
            indices[[row, column]] = start as i64;
        }
    }
    let subspaces = motifs
        .iter()
        .map(|motif| PyArray1::from_vec(py, motif.subspace.iter().map(|&a| a as i64).collect()));
    let sizes = motifs
        .iter()
        .map(|motif| PyArray1::from_vec(py, motif.bit_sizes.clone()));
    (
        PyArray2::from_owned_array(py, distances),
        PyArray2::from_owned_array(py, indices),
        PyList::new(py, subspaces).expect("a list of arrays"),
        PyList::new(py, sizes).expect("a list of arrays"),
    )
}

/// The multi-dimensional motifs of `T` (`d` series, shape `(d, n)`) from its multi-dimensional
/// matrix profile `P` and indices `I` (as `mstump` returns them, shape `(d, n - m + 1)`, so `m`
/// is what they leave of `n`), following STUMPY's `mmotifs`. Returns STUMPY's
/// `(motif_distances, motif_indices, motif_subspaces, motif_mdls)`: one row per motif, padded
/// with `nan` and `-1` to the longest, its matches' distances (the mean over its subspace of each
/// dimension's z-normalized distance) and starts, the motif's own window first at distance 0; the
/// dimensions of its subspace; and the description length for every `k`. With no motif, both
/// arrays have shape `(1, 0)`.
///
/// Each round takes, for every `k`, the window with the smallest remaining value of row `k` of
/// `P` (the smaller index on equal values) and its neighbor in `I`, takes `k` from `k` or, when
/// `None`, from the smallest description length (`mdl`, the smaller `k` on equal ones), and stops
/// at the first candidate whose `P` value exceeds `cutoffs[k]` (a number for every dimension, a
/// list, or by default `max(mean(P[k]) - 2 std(P[k]), min(P[k]))` over its finite values) or
/// `max_distance`. The candidate's matches are the windows within `max_distance` (by default
/// `max(mean(D) - 2 std(D), min(D))` of the candidate's distance profile `D`, plus `atol`) of it
/// in the mean over its subspace, nearest first, at most `max_matches` (`None` for all), none
/// within `ceil(m/4)` of another; it is a motif when it has more than `min_neighbors` of them.
/// Every matched window then leaves every row of `P` with the `ceil(m/4)` starts on each side of
/// it, whether or not the candidate was a motif. `include` lists dimensions every subspace holds.
///
/// With `normalize=False` this is `aamp_mmotifs`, as STUMPY re-routes it (`scales` is then `None`
/// or the one number every dimension is quantized to). `p` must be 2.0 then, and is ignored
/// otherwise. `T_subseq_isconstant` is STUMPY's: a two-dimensional boolean array with one flag
/// per window of each dimension, or a function `f(a, w)` applied to every dimension, or a list
/// with one entry per dimension (`None`, a boolean array, or a function), saying which windows
/// are constant, in place of "all samples equal"; it decides the subspaces and the matches, and
/// should be the one the profile `P` was made with. With `normalize=False` it is accepted and
/// ignored. A `max_motifs` below 1 is treated as 1,
/// with a warning. `scales` (for `normalize=True`) is a number for every dimension, a list, or
/// inferred per dimension as `mstump` infers it.
///
/// Differences from STUMPY 1.14.1: `T_subseq_isconstant` decides the subspaces and description
/// lengths as well as the matches, as `mdl(T_subseq_isconstant=...)` does, where STUMPY's
/// `mmotifs` calls `mdl` without forwarding it and so chooses subspaces and sizes as if no
/// window were flagged (confirmed in its source and by running it); every order is decided on exact
/// distances (ties to the smaller index), the default thresholds are summed in index order, and a negative `k` counts
/// from the last dimension as STUMPY's does; a window with a gap has no distance in its
/// dimension, so it is not a match of a window whose subspace holds that dimension, and a motif
/// candidate whose own window has a gap in its subspace matches only itself; a row whose
/// candidate has no neighbor (`I` negative) has an infinite description length (so it is never
/// the smallest, as `mdl` gives incomplete subspaces); a candidate with no matches (`max_matches`
/// of 0, a negative `max_distance`) still leaves the pool, where STUMPY never returns;
/// `max_distance` may be an int and must not be a function; `I` entries below 0 mean no neighbor.
/// `ValueError` for `P` and `I` that are not `(d, windows)`, an `I` entry that is not below the
/// number of windows, `cutoffs` that are not one per dimension, `k` or `include` entries that
/// are not dimensions, a `nan` `atol`, and everything `mdl` refuses. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (
    T, P, I, min_neighbors=1, max_distance=None, cutoffs=None, max_matches=Some(10),
    max_motifs=1, atol=1e-8, k=None, include=None, normalize=true, p=2.0,
    T_subseq_isconstant=None, *, scales=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn mmotifs<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    P: FloatRows<'py>,
    I: IntRows<'py>,
    min_neighbors: i64,
    max_distance: Option<f64>,
    cutoffs: Option<&Bound<'py, PyAny>>,
    max_matches: Option<i64>,
    max_motifs: i64,
    atol: f64,
    k: Option<i64>,
    include: Option<Ints<'py>>,
    normalize: bool,
    p: f64,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scales: Option<&Bound<'py, PyAny>>,
) -> PyResult<MmotifsResult<'py>> {
    reject_unsupported(None, normalize, p)?;
    let series = rows(&T);
    let request = Request::new(
        &series,
        &P,
        &I,
        min_neighbors,
        max_distance,
        cutoffs,
        max_matches,
        max_motifs,
        atol,
        k,
        include,
    )?;
    if normalize {
        let scales = dimension_scales(&series, request.m, scales)?;
        let name = "T_subseq_isconstant";
        let slices: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
        let flags = flags_of_each(py, T_subseq_isconstant, &slices, request.m, name)?;
        let set = slices
            .iter()
            .zip(&flags)
            .zip(scales)
            .map(|((s, f), scale)| Ok((flagged(s, request.m, f, name)?, scale)))
            .collect::<PyResult<Vec<_>>>()?;
        let params = request.params()?;
        let found = py
            .detach(|| {
                #[cfg(feature = "test-hooks")]
                crate::hold::park_if_armed();
                ladde::motif::mmotifs(&set, request.m, &request.distance, &request.index, &params)
            })
            .map_err(multi_error)?;
        return Ok(arrays(py, &found));
    }
    nonnorm(py, &series, &request, one_scale(scales)?)
}

/// STUMPY's `aamp_mmotifs`: the motifs under the Euclidean distance on the raw values, `mmotifs`
/// in every other respect. Values are quantized to the one `scale` every dimension shares
/// (inferred from every dimension together as `maamp` infers it when `None`); distances,
/// `max_distance` and `cutoffs` are in `T`'s units, and the subspaces are those of
/// `maamp_mdl`. `p` must be 2.0; any other value raises `NotImplementedError`. A pair of
/// windows in one dimension whose squared distance no 128-bit integer holds has no distance.
#[pyfunction]
#[pyo3(signature = (
    T, P, I, min_neighbors=1, max_distance=None, cutoffs=None, max_matches=Some(10),
    max_motifs=1, atol=1e-8, k=None, include=None, p=2.0, scale=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn aamp_mmotifs<'py>(
    py: Python<'py>,
    T: FloatRows<'py>,
    P: FloatRows<'py>,
    I: IntRows<'py>,
    min_neighbors: i64,
    max_distance: Option<f64>,
    cutoffs: Option<&Bound<'py, PyAny>>,
    max_matches: Option<i64>,
    max_motifs: i64,
    atol: f64,
    k: Option<i64>,
    include: Option<Ints<'py>>,
    p: f64,
    scale: Option<ScaleArg>,
) -> PyResult<MmotifsResult<'py>> {
    require_euclidean(p)?;
    let series = rows(&T);
    let request = Request::new(
        &series,
        &P,
        &I,
        min_neighbors,
        max_distance,
        cutoffs,
        max_matches,
        max_motifs,
        atol,
        k,
        include,
    )?;
    nonnorm(py, &series, &request, scale)
}

fn nonnorm<'py>(
    py: Python<'py>,
    series: &[Vec<f64>],
    request: &Request,
    scale: Option<ScaleArg>,
) -> PyResult<MmotifsResult<'py>> {
    let scale = shared_scale(series, request.m, scale)?;
    let set: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    let params = request.params()?;
    let found = py
        .detach(|| {
            #[cfg(feature = "test-hooks")]
            crate::hold::park_if_armed();
            ladde::motif::mmotifs_nonnorm(
                &set,
                request.m,
                scale,
                &request.distance,
                &request.index,
                &params,
            )
        })
        .map_err(multi_error)?;
    Ok(arrays(py, &found))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mmotifs, module)?)?;
    module.add_function(wrap_pyfunction!(aamp_mmotifs, module)?)?;
    Ok(())
}
