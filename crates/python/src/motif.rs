//! `motifs` and `aamp_motifs`: the whole-series batch motifs ([`ladde::motif::motifs`],
//! [`ladde::motif::motifs_nonnorm`]) in the shape and padding of STUMPY's `motifs` and
//! `aamp_motifs`. As in STUMPY, `motifs(..., normalize=False)` is `aamp_motifs`.

use ladde::motif::MotifParams;
use ladde::pattern::{MatchRules, MaxDistance};
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aamp::require_euclidean;
use crate::batch::{resolve_scale, resolve_threads};
use crate::constant::{flags_of, series as flagged};
use crate::convert::{Floats, index_and_distance, nonnorm_index_and_distance};
use crate::err::{ScaleArg, tolerance, window_length};

/// STUMPY's `(motif_distances, motif_indices)`.
type MotifArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

/// The window length STUMPY infers from a series of `n` samples and a profile of `profile_len`
/// entries, which must be a window length this library accepts.
fn window_of_profile(n: usize, profile_len: usize) -> PyResult<i64> {
    let m = (n + 1)
        .checked_sub(profile_len)
        .filter(|&m| m >= 1 && m <= n)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "P of {profile_len} entries does not fit T of {n} samples: its length must be \
                 len(T) - m + 1 for a window length m of at least 3 and at most len(T)"
            ))
        })?;
    Ok(m as i64)
}

pub(crate) fn count(name: &str, value: i64) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be at least 0, got {value}")))
}

/// The top motifs of `T` as STUMPY's `motifs` finds them from `T`'s matrix profile, with every
/// order between windows decided on exact distances (ties to the smaller index). `P` is STUMPY's
/// matrix profile, any one-dimensional array-like, but only its length is used: the window length
/// is `m = len(T) - len(P) + 1`, and the exact two-sided profile is computed here (with `scale` and
/// `threads`, keyword-only, as in `matrix_profile`) because the exact profile decides every order,
/// whatever values `P` holds. A `P` whose length gives an `m` below 3 or above `len(T)` raises
/// `ValueError`. Breaking change from earlier versions: the second argument was the window length
/// `m` (an int); it is now `P`, and `m` is its inferred length.
///
/// Candidates are windows by ascending nearest-neighbor distance. A candidate is a motif when it
/// has more than `min_neighbors` matches (itself first, at distance 0, then windows within
/// `max_distance`, nearest first, at most `max_matches`, none within `ceil(m/4)` of another);
/// every matched window then leaves the candidate pool with its exclusion zone. `max_distance`
/// defaults to `max(mean(D) - 2 std(D), min(D))` of the candidate's distance profile `D`, and
/// `cutoff`, the largest nearest-neighbor distance a candidate may have, to the same formula over
/// the profile; `max_matches=None` keeps every match. Thresholds are compared on the f64
/// distances; `atol` (default `1e-08`, as STUMPY's) is added to the `max_distance` each candidate's
/// matches are compared with, as in `match`, and, as in STUMPY, not to `cutoff`. `max_distance`
/// must be a number (not a function).
///
/// With `normalize=False` this is `aamp_motifs` (as STUMPY re-routes it), and `p` is its `p`;
/// with the default `normalize=True`, `p` is ignored, as in STUMPY.
///
/// Returns `(motif_distances, motif_indices)`, one row per motif, padded with `nan` and `-1` to
/// the longest row; when there is no motif, both have shape `(1, 0)` as STUMPY's do. Matches of
/// different motifs can overlap, as in STUMPY; motifs' own windows never do. A `max_motifs`
/// below 1 raises `ValueError` (STUMPY warns and sets it to 1).
///
/// `T_subseq_isconstant` is STUMPY's: a boolean array with one flag per window of `T`, or a
/// function `f(a, w)` returning one (given `T`, non-finite samples as NaN, and `m`), saying which
/// windows are constant, in place of "all samples equal"; it decides both the profile and the
/// matches. With `normalize=False` it is accepted and ignored, as in STUMPY.
#[pyfunction]
#[pyo3(signature = (
    T, P, min_neighbors=1, max_distance=None, cutoff=None, max_matches=Some(10), max_motifs=1,
    atol=1e-8, normalize=true, p=2.0, T_subseq_isconstant=None, *, scale=None, threads=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn motifs<'py>(
    py: Python<'py>,
    T: Floats<'py>,
    P: Floats<'py>,
    min_neighbors: i64,
    max_distance: Option<f64>,
    cutoff: Option<f64>,
    max_matches: Option<i64>,
    max_motifs: i64,
    atol: f64,
    normalize: bool,
    p: f64,
    T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<MotifArrays<'py>> {
    if !normalize {
        return aamp_motifs(
            py,
            T,
            P,
            min_neighbors,
            max_distance,
            cutoff,
            max_matches,
            max_motifs,
            atol,
            p,
            scale,
            threads,
        );
    }
    let m = window_length(window_of_profile(T.as_array().len(), P.as_array().len())?)?;
    let threads = resolve_threads(threads)?;
    let series = T.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let params = motif_params(
        min_neighbors,
        max_distance,
        cutoff,
        max_matches,
        max_motifs,
        atol,
    )?;
    let name = "T_subseq_isconstant";
    let flags = flags_of(py, T_subseq_isconstant, &series, m, name)?;
    let input = flagged(&series, m, &flags, name)?;
    let groups = py.detach(|| ladde::motif::motifs(input, m, scale, threads, &params));
    Ok(motif_arrays(py, &groups, index_and_distance))
}

/// STUMPY's `aamp_motifs`, from the series itself: the motifs under the Euclidean distance on the
/// raw values, `motifs` in every other respect (`P` and `atol` are as there).
/// Values are quantized to `scale` (keyword-only; inferred from `T` as `aamp` infers it when `None`); distances, `max_distance` and `cutoff` are in `T`'s
/// units. `p` must be 2.0; any other value raises `NotImplementedError`.
///
/// Gaps differ from STUMPY 1.14.1, whose `aamp_motifs` fills NaN and infinite samples of `T` with
/// 0 before matching, so windows with gaps can match and shift the default threshold; here a
/// window without a distance (one holding a gap) is never a candidate or a match.
#[pyfunction]
#[pyo3(signature = (
    T, P, min_neighbors=1, max_distance=None, cutoff=None, max_matches=Some(10), max_motifs=1,
    atol=1e-8, p=2.0, *, scale=None, threads=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn aamp_motifs<'py>(
    py: Python<'py>,
    T: Floats<'py>,
    P: Floats<'py>,
    min_neighbors: i64,
    max_distance: Option<f64>,
    cutoff: Option<f64>,
    max_matches: Option<i64>,
    max_motifs: i64,
    atol: f64,
    p: f64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<MotifArrays<'py>> {
    require_euclidean(p)?;
    let m = window_length(window_of_profile(T.as_array().len(), P.as_array().len())?)?;
    let threads = resolve_threads(threads)?;
    let series = T.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let params = motif_params(
        min_neighbors,
        max_distance,
        cutoff,
        max_matches,
        max_motifs,
        atol,
    )?;
    let groups = py.detach(|| ladde::motif::motifs_nonnorm(&series, m, scale, threads, &params));
    Ok(motif_arrays(py, &groups, |hit| {
        nonnorm_index_and_distance(hit, scale)
    }))
}

fn motif_params(
    min_neighbors: i64,
    max_distance: Option<f64>,
    cutoff: Option<f64>,
    max_matches: Option<i64>,
    max_motifs: i64,
    atol: f64,
) -> PyResult<MotifParams> {
    Ok(MotifParams {
        min_neighbors: count("min_neighbors", min_neighbors)?,
        max_distance: max_distance.map_or(MaxDistance::Default, MaxDistance::Value),
        cutoff,
        max_matches: max_matches.map(|k| count("max_matches", k)).transpose()?,
        max_motifs: motif_limit(max_motifs)?,
        match_rules: MatchRules {
            atol: tolerance(atol)?,
            query_idx: None,
        },
    })
}

/// `max_motifs`: at least 1. STUMPY warns and sets one below 1 to 1, which is a `ValueError` here.
pub(crate) fn motif_limit(max_motifs: i64) -> PyResult<usize> {
    usize::try_from(max_motifs)
        .ok()
        .filter(|&limit| limit >= 1)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "max_motifs={max_motifs} is below 1; STUMPY warns and sets it to 1; pass \
                 max_motifs=1 to get that"
            ))
        })
}

/// The matches of every motif as STUMPY's padded arrays, each match through `convert`.
fn motif_arrays<'py, H: Copy>(
    py: Python<'py>,
    groups: &[Vec<(u64, H)>],
    convert: impl Fn(H) -> (i64, f64),
) -> MotifArrays<'py> {
    let width = groups.iter().map(Vec::len).max().unwrap_or(0);
    let rows = groups.len().max(1);
    let mut distances = Array2::from_elem((rows, width), f64::NAN);
    let mut indices = Array2::from_elem((rows, width), -1_i64);
    for (row, group) in groups.iter().enumerate() {
        for (column, &(_, hit)) in group.iter().enumerate() {
            let (index, distance) = convert(hit);
            distances[[row, column]] = distance;
            indices[[row, column]] = index;
        }
    }
    (
        PyArray2::from_owned_array(py, distances),
        PyArray2::from_owned_array(py, indices),
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(motifs, module)?)?;
    module.add_function(wrap_pyfunction!(aamp_motifs, module)?)?;
    Ok(())
}
