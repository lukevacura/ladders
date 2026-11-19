//! `snippets` and `aampdist_snippets`: the snippets of a series ([`ladde::snippets`]),
//! z-normalized or not, in STUMPY's shape. As in STUMPY, `snippets(..., normalize=False)` is
//! `aampdist_snippets`.

use ladde::snippets::{InnerLength, SnippetsError};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aamp::require_euclidean;
use crate::batch::{resolve_scale, resolve_threads};
use crate::constant::{flags_of, series as flagged};
use crate::convert::Floats;
use crate::err::{ScaleArg, bounded_count, window_length};
use crate::mpdist::selection;

/// STUMPY's `snippets` return: `(snippets, indices, profiles, fractions, areas, regimes)`.
type SnippetsResult<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyAny>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyAny>,
);

fn error(error: SnippetsError, n: usize, arguments: &Arguments) -> PyErr {
    let Arguments { m, percentage, .. } = *arguments;
    PyValueError::new_err(match error {
        SnippetsError::SeriesTooShort => {
            format!("m = {m} is too large for a series of {n} samples: need m <= len(T) // 2")
        }
        SnippetsError::EmptyInnerWindow => {
            "the inner window length s must be at least 1 (check s and percentage)".to_owned()
        }
        SnippetsError::InnerWindowTooShort => {
            "the inner window length s must be at least 3 (check s and percentage)".to_owned()
        }
        SnippetsError::NotANumber => "percentage must not be NaN".to_owned(),
        SnippetsError::ShareOutOfRange => {
            let clipped = percentage.clamp(0.0, 1.0);
            format!(
                "percentage={percentage:?} is outside [0, 1]; STUMPY clips it to {clipped:?}; \
                 pass percentage={clipped:?} to get that"
            )
        }
        SnippetsError::InnerWindowAboveSegment => format!(
            "s={} is above m={m}; STUMPY uses s={m}; pass s={m} to get that",
            arguments.s.unwrap_or_default()
        ),
        SnippetsError::RankPastLast => {
            "mpdist_k is past the last position of the joined distances".to_owned()
        }
        SnippetsError::Capacity => {
            "k snippets of this series are too large to allocate: use a smaller k".to_owned()
        }
        // The enum is non-exhaustive: a variant this file does not know is named as core names it.
        _ => format!("the input was refused: {error:?}"),
    })
}

/// The `k` snippets of `t` for segment length `m`, as STUMPY's `snippets`:
/// `(snippets, indices, profiles, fractions, areas, regimes)`, `snippets` and `profiles` one row
/// per snippet, `regimes` rows of `(snippet, start, stop)`. Each segment's MPdist profile is built
/// at inner window length `s` (`s` if given, else `ceil(percentage * m)`); each profile value is
/// the entry at sorted position `mpdist_k` (else `ceil(mpdist_percentage * 2m)`, lowered to the
/// last position as in STUMPY) of the segment's and the window's nearest distances, as
/// `mpdist`, every order between distances the exact one, not on rounded values; the nearest
/// window of a window is the exactly nearest, ties to the smaller start. The greedy pick, areas,
/// fractions and regimes are STUMPY's on f64 values (areas summed pairwise as NumPy does, equal
/// areas to the smaller segment). Like STUMPY, a `k` above the number of segments picks segments
/// again.
///
/// Differences from STUMPY, each a `ValueError`: a `percentage` or `mpdist_percentage` outside
/// `[0, 1]` (STUMPY clips it), an `s` above `m` (STUMPY lowers it to `m`) and an `mpdist_k` past
/// the last of the `2(m - s + 1)` joined distances (STUMPY lowers it to the last).
///
/// `scale` is the tick size, inferred from `t` when `None`; `threads` means what it does for
/// `ab_join` and the result does not depend on it. `ValueError` for `m < 3`, `len(t) < 2m`, `k < 1`,
/// an inner window length `s` below 3, whether given or from a small `percentage` (STUMPY accepts 1
/// and 2 but spins for minutes there), a NaN `percentage` (unless `s` is given) or `mpdist_percentage` (unless
/// `mpdist_k` is given), a negative `mpdist_k`, or an inferable scale that does not exist.
/// Memory beyond the profiles: with `s` far below `m` on tie-heavy data the per-row sliding
/// deques keep every equal value, holding up to `(m - s + 1)²` entries per segment per thread
/// (about 1 GB at n = 8000, m = 2000, s = 3, one thread; per-thread peaks add).
/// Releases the GIL while it runs.
///
/// With `normalize=False` this is `aampdist_snippets` (as STUMPY re-routes it), and `p` is its
/// `p`; with the default `normalize=True`, `p` is accepted and ignored, as in STUMPY.
/// `mpdist_T_subseq_isconstant` is STUMPY's: a boolean array with one flag per window of the inner
/// length `s` (`len(T) - s + 1` of them), or a function `f(a, w)` returning one (given `T`,
/// non-finite samples as NaN, and `s`), saying which windows are constant, in place of "all
/// samples equal". With `normalize=False` it is accepted and ignored, as in STUMPY.
#[pyfunction]
#[pyo3(signature = (
    T, m, k, percentage=1.0, s=None, mpdist_percentage=0.05, mpdist_k=None, normalize=true,
    p=2.0, mpdist_T_subseq_isconstant=None, *, scale=None, threads=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn snippets<'py>(
    py: Python<'py>,
    T: Floats<'py>,
    m: i64,
    k: i64,
    percentage: f64,
    s: Option<i64>,
    mpdist_percentage: f64,
    mpdist_k: Option<i64>,
    normalize: bool,
    p: f64,
    mpdist_T_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<SnippetsResult<'py>> {
    if !normalize {
        require_euclidean(p)?;
    }
    compute(
        py,
        &T.as_array().to_vec(),
        Arguments {
            m,
            k,
            percentage,
            s,
            mpdist_percentage,
            mpdist_k,
        },
        scale,
        threads,
        (!normalize, mpdist_T_subseq_isconstant),
    )
}

/// STUMPY's `aampdist_snippets`: the snippets under the Euclidean distance on the raw values, in
/// `snippets`' shape and with its keywords (`snippets(..., normalize=False)` is this). Profiles
/// and `areas` are in `T`'s units, whose samples are quantized to `scale` (inferred from `T` as
/// `aamp` infers it when `None`); a window holding a NaN or infinite sample has no distance.
/// `p` must be 2.0; any other value raises `NotImplementedError`.
///
/// Like STUMPY, and unlike `snippets`, the last whole segment is not a candidate when `m` divides
/// `len(T)` (STUMPY pads `T` to a multiple of `m` and then takes one segment fewer than the
/// padded series holds), so `k` above the segment count picks the others again.
#[pyfunction]
#[pyo3(signature = (
    T, m, k, percentage=1.0, s=None, mpdist_percentage=0.05, mpdist_k=None, p=2.0, *, scale=None,
    threads=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn aampdist_snippets<'py>(
    py: Python<'py>,
    T: Floats<'py>,
    m: i64,
    k: i64,
    percentage: f64,
    s: Option<i64>,
    mpdist_percentage: f64,
    mpdist_k: Option<i64>,
    p: f64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<SnippetsResult<'py>> {
    require_euclidean(p)?;
    compute(
        py,
        &T.as_array().to_vec(),
        Arguments {
            m,
            k,
            percentage,
            s,
            mpdist_percentage,
            mpdist_k,
        },
        scale,
        threads,
        (true, None),
    )
}

/// The arguments both functions take as STUMPY does.
#[derive(Clone, Copy)]
struct Arguments {
    m: i64,
    k: i64,
    percentage: f64,
    s: Option<i64>,
    mpdist_percentage: f64,
    mpdist_k: Option<i64>,
}

fn compute<'py>(
    py: Python<'py>,
    series: &[f64],
    arguments: Arguments,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    (nonnorm, constant): (bool, Option<&Bound<'py, PyAny>>),
) -> PyResult<SnippetsResult<'py>> {
    let Arguments {
        m,
        k,
        percentage,
        s,
        mpdist_percentage,
        mpdist_k,
    } = arguments;
    let m_checked = window_length(m)?;
    let k = bounded_count("k", k)?;
    let threads = resolve_threads(threads)?;
    let inner = match s {
        Some(s) => InnerLength::Length(
            usize::try_from(s)
                .map_err(|_| PyValueError::new_err(format!("s must be non-negative, got {s}")))?,
        ),
        None => InnerLength::Share(percentage),
    };
    let m_len = m_checked.get();
    let inner_len = inner
        .resolve(m_len)
        .map_err(|e| error(e, series.len(), &arguments))?;
    if series.len() < 2 * m_len {
        return Err(error(
            SnippetsError::SeriesTooShort,
            series.len(),
            &arguments,
        ));
    }
    let selection = selection(
        mpdist_percentage,
        mpdist_k,
        2 * (m_len - inner_len.get() + 1),
        ("mpdist_percentage", "mpdist_k"),
    )?;
    let scale = resolve_scale(series, inner_len, scale)?;
    let k_nonzero = core::num::NonZeroUsize::new(k).expect("k is at least 1");
    let name = "mpdist_T_subseq_isconstant";
    let flags = match nonnorm {
        true => None,
        false => flags_of(py, constant, series, inner_len, name)?,
    };
    let input = flagged(series, inner_len, &flags, name)?;
    let found = py
        .detach(|| {
            if nonnorm {
                ladde::snippets::snippets_nonnorm(
                    series, m_checked, k_nonzero, scale, threads, inner, selection,
                )
            } else {
                ladde::snippets::snippets(
                    input, m_checked, k_nonzero, scale, threads, inner, selection,
                )
            }
        })
        .map_err(|e| error(e, series.len(), &arguments))?;
    let to_i64 = |values: &[usize]| values.iter().map(|&v| v as i64).collect::<Vec<_>>();
    let regimes: Vec<i64> = found.regimes.iter().flatten().map(|&v| v as i64).collect();
    Ok((
        PyArray1::from_vec(py, found.snippets)
            .reshape([k, found.m])?
            .into_any(),
        PyArray1::from_vec(py, to_i64(&found.indices)),
        PyArray1::from_vec(py, found.profiles)
            .reshape([k, found.profile_len])?
            .into_any(),
        PyArray1::from_vec(py, found.fractions),
        PyArray1::from_vec(py, found.areas),
        PyArray1::from_vec(py, regimes)
            .reshape([found.regimes.len(), 3])?
            .into_any(),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(snippets, module)?)?;
    module.add_function(wrap_pyfunction!(aampdist_snippets, module)?)?;
    Ok(())
}
