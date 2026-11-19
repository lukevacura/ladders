//! `aampi`: STUMPY's non-normalized streaming profile, built directly on whichever native
//! non-normalized stream `egress` picks (see [`Engine`]) — the same `update`, never a second way
//! of walking a series.

use core::num::NonZeroUsize;

use ladde::index::SampleIndex;
use ladde::narrow::RouteCounts;
use ladde::nonnorm::NonNormNeighbor;
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::quantize::Scale;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::aamp::{require_euclidean, shared_scale};
use crate::batch::resolve_k;
use crate::convert::{Floats, nonnorm_index_and_distance};
use crate::err::{
    Remedy, ScaleArg, checkpoint_error, countable, require_none, window_bound, window_length,
};
use crate::stumpi::{
    History, InferredScale, Side, bad, padded_arrays, put_history, seal, take, take_history,
    take_u64, unseal_versions,
};

/// Either native stream: every sample this wrapper folds in goes through *that* `update_many`,
/// the same one the stream exposes directly.
enum Engine {
    Growing(GrowingNonNormStream),
    Bounded(BoundedNonNormStream),
}

impl Engine {
    /// The engine's own `update_topk`: the newest window's kept neighbors, nearest first.
    fn update_topk(&mut self, x: f64) -> Option<(SampleIndex, &[NonNormNeighbor])> {
        match self {
            Engine::Growing(stream) => stream.update_topk(x),
            Engine::Bounded(stream) => stream.update_topk(x),
        }
    }

    fn k(&self) -> NonZeroUsize {
        match self {
            Engine::Growing(stream) => stream.k(),
            Engine::Bounded(stream) => stream.k(),
        }
    }

    fn route_counts(&self) -> RouteCounts {
        match self {
            Engine::Growing(stream) => stream.route_counts(),
            Engine::Bounded(stream) => stream.route_counts(),
        }
    }

    fn scale(&self) -> Scale {
        match self {
            Engine::Growing(stream) => stream.scale(),
            Engine::Bounded(stream) => stream.scale(),
        }
    }

    /// The held windows' lists of `side` as a history, `None` for a stream that does not look
    /// ahead.
    fn side(&self, side: Side) -> Option<History> {
        let scale = self.scale();
        let mut history = History::default();
        let mut out = Vec::new();
        macro_rules! walk {
            ($stream:expr) => {{
                if !$stream.looks_right() {
                    return None;
                }
                for window in $stream.right_windows() {
                    match side {
                        Side::Right => $stream.right_neighbors_into(window, &mut out),
                        Side::Both => $stream.two_sided_neighbors_into(window, &mut out),
                    }
                    history.push_pairs(
                        out.iter()
                            .map(|&neighbor| nonnorm_index_and_distance(neighbor, scale)),
                    );
                }
            }};
        }
        match self {
            Engine::Growing(stream) => walk!(stream),
            Engine::Bounded(stream) => walk!(stream),
        }
        Some(history)
    }
}

/// A causal streaming left profile under the Euclidean distance on the raw values, `aampi`-shaped:
/// `update(t)`, `left_P_`, `left_I_` (STUMPY's own names), and no `P_`/`I_` — this library never
/// looks ahead. Distances are in the caller's units, the square root of the exact squared
/// distance between the quantized windows.
///
/// With `k` above 1 each window keeps its `k` nearest earlier neighbors: `left_topk_P_` and
/// `left_topk_I_` hold them as arrays of shape `(windows, k)`, nearest first, padded with `inf`
/// and `-1` where fewer exist (STUMPY's layout for `P_` and `I_`, whose last row is the same
/// numbers), while `left_P_` and `left_I_` stay the nearest one, as in STUMPY.
///
/// `scale` is the tick size each sample is quantized to. `egress=True` (the default) bounds the
/// search to the trailing `w` samples; `egress=False` keeps every sample and takes no `w`. Unlike
/// STUMPY's, `left_P_` and `left_I_` keep growing with every window the stream completes, also
/// with `egress=True`.
///
/// Breaking change from earlier versions: the arguments follow STUMPY's (`T, m, egress, p, k,
/// mp`), so `scale` is keyword-only (`aampi(T, m, scale=...)`) and optional, and `w` defaults to
/// `len(T)` under `egress=True`.
#[pyclass(module = "ladders", name = "aampi")]
pub struct Aampi {
    engine: Engine,
    history: History,
    guard: Option<InferredScale>,
}

/// The wrapper's own checkpoint framing: magic, version, engine tag, the core checkpoint
/// (length-prefixed), then the `left_P_`/`left_I_` history. A stream keeping more than one
/// neighbor writes [`VERSION_TOPK`], which has `k` before the history and, in it, every window's
/// neighbor count and then all neighbors.
const MAGIC: &[u8; 4] = b"MPAI";
const VERSION: u16 = 2;
const VERSION_TOPK: u16 = 3;

impl Aampi {
    /// The held windows' lists of `side`, or a `ValueError` for a stream built without
    /// `look_ahead=True`.
    fn ahead(&self, side: Side) -> PyResult<History> {
        self.engine.side(side).ok_or_else(|| {
            PyValueError::new_err("this stream does not look ahead: build it with look_ahead=True")
        })
    }

    fn feed(&mut self, xs: &[f64]) {
        let scale = self.engine.scale();
        for &x in xs {
            if let Some((_, kept)) = self.engine.update_topk(x) {
                self.history.push_pairs(
                    kept.iter()
                        .map(|&neighbor| nonnorm_index_and_distance(neighbor, scale)),
                );
            }
        }
    }
}

#[pymethods]
impl Aampi {
    /// `T` is the warm-start history: run through the stream before this returns, so
    /// `left_P_`/`left_I_` already cover it.
    ///
    /// `egress=True` (the default) keeps the trailing `w` samples, `w` defaulting to `len(T)` as in
    /// STUMPY; `egress=False` keeps every sample and takes no `w`. `scale` is the tick size every
    /// sample is quantized to; when `None` it is inferred from `T` and `m` as `aamp` infers it, and
    /// a later sample off that grid is rounded to it. `p` must be 2.0 and `mp` must be `None`: any
    /// other value raises `NotImplementedError`. `k` is how many nearest earlier neighbors each
    /// window keeps (`left_topk_P_`, `left_topk_I_`); below 1 it is a `ValueError`.
    #[new]
    #[pyo3(signature = (T, m, egress=true, p=2.0, k=1, mp=None, *, scale=None, w=None, look_ahead=false))]
    #[allow(clippy::too_many_arguments, non_snake_case)]
    fn new(
        py: Python<'_>,
        T: Floats<'_>,
        m: i64,
        egress: bool,
        p: f64,
        k: i64,
        mp: Option<&Bound<'_, PyAny>>,
        scale: Option<ScaleArg>,
        w: Option<i64>,
        look_ahead: bool,
    ) -> PyResult<Self> {
        require_euclidean(p)?;
        let k = resolve_k(k)?;
        require_none("mp", mp)?;
        let m_nz = window_length(m)?;
        let history = T.as_array().to_vec();
        let inferred = !matches!(scale, Some(ScaleArg::Tick(_)));
        let scale = shared_scale(&history, None, m_nz, scale)?;
        let engine = if egress {
            let w = w.unwrap_or(history.len() as i64);
            let w_nz: NonZeroUsize = window_bound(w, m_nz)?;
            let stream = if look_ahead {
                BoundedNonNormStream::with_right_side(m_nz, w_nz, scale, k)
            } else {
                BoundedNonNormStream::with_k(m_nz, w_nz, scale, k)
            };
            Engine::Bounded(stream.ok_or_else(|| {
                PyValueError::new_err(
                    "k keeps more neighbors than can be reserved; use a smaller k",
                )
            })?)
        } else {
            if w.is_some() {
                return Err(PyValueError::new_err("w is only used when egress=True"));
            }
            Engine::Growing(if look_ahead {
                GrowingNonNormStream::with_right_side(m_nz, scale, k)
            } else {
                GrowingNonNormStream::with_k(m_nz, scale, k)
            })
        };

        let guard = inferred.then(|| InferredScale::new(scale, m_nz, false, &history));
        let mut this = Self {
            engine,
            history: History::default(),
            guard,
        };
        py.detach(|| this.feed(&history));
        Ok(this)
    }

    /// Folds in one causal sample, appending its window's entry to `left_P_`/`left_I_` if it
    /// completes one. Returns nothing, matching STUMPY's own `aampi.update`. A finite sample with
    /// no tick at the stream's scale (beyond 2**53 ticks) raises `ValueError` and is not fed;
    /// pass `scale=` for a stream that may leave the range of its history.
    fn update(&mut self, py: Python<'_>, t: f64) -> PyResult<()> {
        countable(&[t], self.engine.scale(), Remedy::Restart)?;
        if let Some(guard) = &mut self.guard {
            guard.admit(t)?;
        }
        py.detach(|| self.feed(&[t]));
        Ok(())
    }

    /// The tick size every sample is quantized to, given or inferred.
    #[getter(scale_)]
    fn scale_used(&self) -> f64 {
        self.engine.scale().get()
    }

    /// The windows' nearest later neighbors' distances, oldest window first, for the windows the
    /// stream holds; needs `look_ahead=True`.
    #[getter(right_P_)]
    fn right_p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let ahead = self.ahead(Side::Right)?;
        Ok(PyArray1::from_iter(
            py,
            ahead.nearest().map(|(_, distance)| distance),
        ))
    }

    /// The indices of [`Self::right_p`], global; `-1` where a window has no later neighbor.
    #[getter(right_I_)]
    fn right_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ahead = self.ahead(Side::Right)?;
        Ok(PyArray1::from_iter(
            py,
            ahead.nearest().map(|(index, _)| index),
        ))
    }

    /// Each held window's `k` nearest later neighbors' distances, shape `(windows, k)`, nearest
    /// first, `inf` where a window has fewer; needs `look_ahead=True`.
    #[getter(right_topk_P_)]
    fn right_topk_p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(padded_arrays(py, &self.ahead(Side::Right)?, self.engine.k().get())?.0)
    }

    /// [`Self::right_topk_p`]'s indices, `-1` where a window has fewer than `k` neighbors.
    #[getter(right_topk_I_)]
    fn right_topk_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(padded_arrays(py, &self.ahead(Side::Right)?, self.engine.k().get())?.1)
    }

    /// The held windows' nearest neighbors over both sides (STUMPY's `P_`): the earlier ones as
    /// the stream found them, the later ones so far; needs `look_ahead=True`.
    #[getter(two_sided_P_)]
    fn two_sided_p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let ahead = self.ahead(Side::Both)?;
        Ok(PyArray1::from_iter(
            py,
            ahead.nearest().map(|(_, distance)| distance),
        ))
    }

    /// The indices of [`Self::two_sided_p`], global (STUMPY's `I_`).
    #[getter(two_sided_I_)]
    fn two_sided_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ahead = self.ahead(Side::Both)?;
        Ok(PyArray1::from_iter(
            py,
            ahead.nearest().map(|(index, _)| index),
        ))
    }

    /// Each held window's `k` nearest neighbors over both sides, shape `(windows, k)`.
    #[getter(two_sided_topk_P_)]
    fn two_sided_topk_p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(padded_arrays(py, &self.ahead(Side::Both)?, self.engine.k().get())?.0)
    }

    /// [`Self::two_sided_topk_p`]'s indices.
    #[getter(two_sided_topk_I_)]
    fn two_sided_topk_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(padded_arrays(py, &self.ahead(Side::Both)?, self.engine.k().get())?.1)
    }

    /// The causal left profile distances seen so far, oldest first.
    #[getter(left_P_)]
    fn left_p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_iter(py, self.history.nearest().map(|(_, distance)| distance))
    }

    /// The causal left profile indices seen so far, oldest first; `-1` where a row has no
    /// neighbor.
    #[getter(left_I_)]
    fn left_i<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_iter(py, self.history.nearest().map(|(index, _)| index))
    }

    /// Each window's `k` nearest earlier neighbors' distances, shape `(windows, k)`, nearest
    /// first, `inf` where a window has fewer. Its last row is what STUMPY's `P_[-1]` holds after
    /// the same updates; the rows before it differ, since STUMPY revises them with later windows.
    #[getter(left_topk_P_)]
    fn left_topk_p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(padded_arrays(py, &self.history, self.engine.k().get())?.0)
    }

    /// [`Self::left_topk_p`]'s indices, `-1` where a window has fewer than `k` neighbors.
    #[getter(left_topk_I_)]
    fn left_topk_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(padded_arrays(py, &self.history, self.engine.k().get())?.1)
    }

    /// How the stream's updates split across routes, as a dict: `narrow_updates` and
    /// `wide_updates` (samples fed on each arithmetic), `switches_to_wide` and
    /// `switches_to_narrow`. Informational only: no entry depends on it, and a restored stream
    /// counts from zero.
    fn route_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counts = self.engine.route_counts();
        let dict = PyDict::new(py);
        dict.set_item("narrow_updates", counts.narrow_updates)?;
        dict.set_item("wide_updates", counts.wide_updates)?;
        dict.set_item("switches_to_wide", counts.switches_to_wide)?;
        dict.set_item("switches_to_narrow", counts.switches_to_narrow)?;
        Ok(dict)
    }

    /// The wrapper's whole state as bytes: the stream's core checkpoint plus the accumulated
    /// `left_P_`/`left_I_`, so `restore` brings the history back too.
    fn checkpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let (tag, core) = match &self.engine {
            Engine::Growing(stream) => (0u8, stream.checkpoint()),
            Engine::Bounded(stream) => (1u8, stream.checkpoint()),
        };
        let k = self.engine.k();
        let topk = k != NonZeroUsize::MIN;
        let version = if topk { VERSION_TOPK } else { VERSION };
        let mut buf = Vec::with_capacity(core.len() + 24 * self.history.windows() + 40);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.push(tag);
        buf.extend_from_slice(&(core.len() as u64).to_le_bytes());
        buf.extend_from_slice(&core);
        if topk {
            buf.extend_from_slice(&(k.get() as u64).to_le_bytes());
            self.history.put(&mut buf);
        } else {
            let (index, distance): (Vec<i64>, Vec<f64>) = self.history.nearest().unzip();
            put_history(&mut buf, &index, &distance);
        }
        seal(&mut buf);
        PyBytes::new(py, &buf)
    }

    /// A wrapper rebuilt from `checkpoint()` bytes, history included. Raises `ValueError` for
    /// any bytes that are not an aampi checkpoint or whose checksum does not match: it detects
    /// accidental corruption (bit rot, partial writes, truncation), not tampering.
    #[staticmethod]
    fn restore(data: &[u8]) -> PyResult<Self> {
        let (version, mut rest) = unseal_versions(data, MAGIC, &[VERSION, VERSION_TOPK])?;
        let tag = take(&mut rest, 1)?[0];
        let core_len = usize::try_from(take_u64(&mut rest)?).map_err(|_| bad("Truncated"))?;
        let core = take(&mut rest, core_len)?;
        let engine = match tag {
            0 => Engine::Growing(GrowingNonNormStream::restore(core).map_err(checkpoint_error)?),
            1 => Engine::Bounded(BoundedNonNormStream::restore(core).map_err(checkpoint_error)?),
            _ => return Err(bad("Inconsistent")),
        };
        let k = engine.k();
        let topk = version == VERSION_TOPK;
        if topk != (k != NonZeroUsize::MIN) || (topk && take_u64(&mut rest)? != k.get() as u64) {
            return Err(bad("Inconsistent"));
        }
        let history = if topk {
            History::take(&mut rest, k.get())?
        } else {
            let (index, distance) = take_history(&mut rest)?;
            History::from_nearest(index, distance)
        };
        Ok(Self {
            engine,
            history,
            guard: None,
        })
    }

    /// Pickle support: rebuilds through `restore` from the checkpoint bytes.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let restore = py.get_type::<Self>().getattr("restore")?;
        Ok((restore, (self.checkpoint(py),)))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Aampi>()?;
    Ok(())
}
