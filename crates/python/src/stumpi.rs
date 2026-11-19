//! `stumpi`: STUMPY's own shape for a streaming profile, built directly on whichever native
//! engine `egress` picks (see [`Engine`]) — no third way of walking a series.

use core::num::NonZeroUsize;
use ladde::bounded::BoundedStream as CoreBoundedStream;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::stream::GrowingStream as CoreGrowingStream;
use std::collections::VecDeque;

use ladde::quantize::{Scale, to_ticks};
use ladde::stats::{WindowKind, WindowStats};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::batch::{resolve_k, resolve_scale};
use crate::constant::{WindowPredicate, flags_of};
use crate::convert::{Floats, index_and_distance};
use crate::err::{
    Remedy, ScaleArg, checkpoint_error, countable, require_none, window_bound, window_length,
};

/// Either native engine, behind the one trait both already implement: every sample this
/// wrapper folds in goes through *that* `update`, the same one `GrowingStream`/`BoundedStream`
/// expose directly, never a reimplementation they could drift from.
enum Engine {
    Growing(CoreGrowingStream),
    Bounded(CoreBoundedStream),
}

impl Engine {
    /// The engine's own `update_topk_flagged`: the newest window's kept neighbors, nearest first.
    fn update_topk(&mut self, x: f64, constant: Option<bool>) -> Option<(SampleIndex, &[Entry])> {
        match self {
            Engine::Growing(stream) => stream.update_topk_flagged(x, constant),
            Engine::Bounded(stream) => stream.update_topk_flagged(x, constant),
        }
    }

    fn k(&self) -> NonZeroUsize {
        match self {
            Engine::Growing(stream) => stream.k(),
            Engine::Bounded(stream) => stream.k(),
        }
    }

    fn scale(&self) -> Scale {
        match self {
            Engine::Growing(stream) => stream.scale(),
            Engine::Bounded(stream) => stream.scale(),
        }
    }
}

/// Which of a window's lists a look-ahead stream is read for.
#[derive(Clone, Copy)]
pub(crate) enum Side {
    /// The nearest later neighbors.
    Right,
    /// The nearest neighbors over both sides.
    Both,
}

impl Engine {
    /// The held windows' lists of `side` as a history, `None` for a stream that does not look
    /// ahead.
    fn side(&self, side: Side) -> Option<History> {
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
                    history.push_row(&out);
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

/// The keyword the predicate comes in by, for the errors about it.
const PREDICATE: &str = "T_subseq_isconstant_func";

/// A causal streaming left matrix profile, `stumpi`-shaped: `update(t)`, `left_P_`, `left_I_`
/// (STUMPY's own names), and no `P_`/`I_` — this library never looks ahead unless asked: with
/// `look_ahead=True` the stream also keeps each window's nearest later neighbors, and
/// `right_P_`, `right_I_`, `two_sided_P_` and `two_sided_I_` (with `_topk_` forms) read them. A
/// window's entry there can change as later samples arrive, so nothing causal may read it.
///
/// With `k` above 1 each window keeps its `k` nearest earlier neighbors: `left_topk_P_` and
/// `left_topk_I_` hold them as arrays of shape `(windows, k)`, nearest first, padded with `inf`
/// and `-1` where fewer exist (STUMPY's layout for `P_` and `I_`, whose last row is the same
/// numbers), while `left_P_` and `left_I_` stay the nearest one, as in STUMPY.
///
/// `egress=True` (the default) bounds the search to the trailing `w` samples; `egress=False`
/// keeps every sample and takes no `w`. Unlike STUMPY's, `left_P_` and `left_I_` keep growing with
/// every window the stream completes, also with `egress=True`: the bound limits which windows can
/// be neighbors, not what is reported.
///
/// Breaking change from earlier versions: the arguments follow STUMPY's (`T, m, egress, normalize,
/// p, k, mp`), so `scale` is keyword-only (`stumpi(T, m, scale=...)`) and optional, and `w`
/// defaults to `len(T)` under `egress=True`.
#[pyclass(module = "ladders", name = "stumpi")]
pub struct Stumpi {
    engine: Engine,
    history: History,
    guard: Option<InferredScale>,
    predicate: Option<WindowPredicate>,
}

/// What each completed window kept: its neighbors, nearest first, as `(index, distance)`. Only
/// the neighbors a window has are stored, so a `k` beyond the windows that exist costs nothing
/// until `left_topk_P_` pads them to the shape `(windows, k)`.
#[derive(Default)]
pub(crate) struct History {
    index: Vec<i64>,
    distance: Vec<f64>,
    /// For each window, where its neighbors end in `index` and `distance`.
    ends: Vec<usize>,
}

impl History {
    pub(crate) fn windows(&self) -> usize {
        self.ends.len()
    }

    fn row(&self, window: usize) -> core::ops::Range<usize> {
        let start = window.checked_sub(1).map_or(0, |before| self.ends[before]);
        start..self.ends[window]
    }

    fn push_row(&mut self, kept: &[Entry]) {
        self.push_pairs(kept.iter().map(|&entry| index_and_distance(entry)));
    }

    /// Appends one window's neighbors as `(index, distance)`, nearest first.
    pub(crate) fn push_pairs(&mut self, kept: impl Iterator<Item = (i64, f64)>) {
        for (index, distance) in kept {
            self.index.push(index);
            self.distance.push(distance);
        }
        self.ends.push(self.index.len());
    }

    /// Each window's nearest neighbor, `-1` and `inf` for a window without one.
    pub(crate) fn nearest(&self) -> impl Iterator<Item = (i64, f64)> + '_ {
        (0..self.windows()).map(|window| {
            let row = self.row(window);
            if row.is_empty() {
                (-1, f64::INFINITY)
            } else {
                (self.index[row.start], self.distance[row.start])
            }
        })
    }

    /// `(windows, k)` indices and distances, padded with `-1` and `inf`; `None` when that many
    /// slots cannot be reserved.
    pub(crate) fn padded(&self, k: usize) -> Option<(Vec<i64>, Vec<f64>)> {
        let slots = self.windows().checked_mul(k)?;
        let mut index = Vec::new();
        let mut distance = Vec::new();
        index.try_reserve_exact(slots).ok()?;
        distance.try_reserve_exact(slots).ok()?;
        for window in 0..self.windows() {
            let row = self.row(window);
            let kept = row.len();
            index.extend_from_slice(&self.index[row.clone()]);
            distance.extend_from_slice(&self.distance[row]);
            index.resize(index.len() + (k - kept), -1);
            distance.resize(distance.len() + (k - kept), f64::INFINITY);
        }
        Some((index, distance))
    }

    /// The history of a one-neighbor stream checkpoint: one slot per window, `-1` for none.
    pub(crate) fn from_nearest(index: Vec<i64>, distance: Vec<f64>) -> Self {
        let mut history = Self::default();
        for (index, distance) in index.into_iter().zip(distance) {
            if index >= 0 {
                history.index.push(index);
                history.distance.push(distance);
            }
            history.ends.push(history.index.len());
        }
        history
    }

    /// The window count, each window's neighbor count, then every neighbor's index and distance.
    pub(crate) fn put(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(self.windows() as u64).to_le_bytes());
        for window in 0..self.windows() {
            buf.extend_from_slice(&(self.row(window).len() as u64).to_le_bytes());
        }
        put_history(buf, &self.index, &self.distance);
    }

    /// [`Self::put`]'s inverse over exactly the bytes that remain, every window keeping at most
    /// `k` neighbors.
    pub(crate) fn take(rest: &mut &[u8], k: usize) -> PyResult<Self> {
        let windows = usize::try_from(take_u64(rest)?).map_err(|_| bad("Truncated"))?;
        // The counts are checked against what remains before anything allocates.
        if windows.checked_mul(8).is_none_or(|n| n > rest.len()) {
            return Err(bad("Truncated"));
        }
        let mut ends = Vec::with_capacity(windows);
        let mut total = 0usize;
        for _ in 0..windows {
            let kept = usize::try_from(take_u64(rest)?).map_err(|_| bad("Inconsistent"))?;
            if kept > k {
                return Err(bad("Inconsistent"));
            }
            total = total.checked_add(kept).ok_or_else(|| bad("Inconsistent"))?;
            ends.push(total);
        }
        let (index, distance) = take_history(rest)?;
        if index.len() != total {
            return Err(bad("Inconsistent"));
        }
        Ok(Self {
            index,
            distance,
            ends,
        })
    }
}

/// The wrapper's own checkpoint framing: magic, version, engine tag, the core checkpoint
/// (length-prefixed), then the `left_P_`/`left_I_` history (count-prefixed, floats as bits),
/// then a checksum of every byte before it. A stream keeping more than one neighbor writes
/// [`VERSION_TOPK`], which has `k` before the history and, in it, every window's neighbor count
/// and then all neighbors (the nearest of each window alone is the version-2 history).
const MAGIC: &[u8; 4] = b"MPSI";
const VERSION: u16 = 2;
/// The version of a checkpoint of a stream with a predicate: the engine's checkpoint is
/// followed by the latest `m` samples the predicate was applied to.
const PREDICATE_VERSION: u16 = 3;
const VERSION_TOPK: u16 = 4;
const VERSION_TOPK_PREDICATE: u16 = 5;

/// Watches the samples fed to a stream whose scale was inferred from its warm start: a sample the
/// stream could not represent at that scale would silently become a gap (no neighbor, infinite
/// distance), so it is a `ValueError` instead, and the stream does not see it. Each sample is
/// checked on its own against the tick domain and, for the z-normalized stream, against the
/// running statistics of the window it completes, which the guard keeps as the stream does: a
/// check allocates nothing and costs the same whatever `m` is.
pub(crate) struct InferredScale {
    scale: Scale,
    m: NonZeroUsize,
    /// The running statistics of the latest `m` samples, when a window too volatile for the
    /// z-normalized arithmetic is also unrepresentable.
    stats: Option<WindowStats>,
    /// The latest `m` samples admitted, and how many of them are not finite.
    recent: VecDeque<f64>,
    gaps: usize,
    /// The newest tick, which a sample without one stands in for in `stats`.
    last_tick: i64,
    /// The sample the newest admitted one pushed out of `recent`, for [`Self::forget_last`].
    departed: Option<f64>,
}

impl InferredScale {
    pub(crate) fn new(scale: Scale, m: NonZeroUsize, z_normalized: bool, history: &[f64]) -> Self {
        let from = history.len().saturating_sub(m.get());
        let mut recent = VecDeque::with_capacity(m.get() + 1);
        recent.extend(history[from..].iter().copied());
        let mut this = Self {
            scale,
            m,
            stats: z_normalized.then(|| WindowStats::new(m)),
            recent,
            gaps: 0,
            last_tick: 0,
            departed: None,
        };
        this.replay();
        this
    }

    /// Rebuilds `stats`, the gap count and the newest tick from `recent`.
    fn replay(&mut self) {
        self.stats = self.stats.as_ref().map(|_| WindowStats::new(self.m));
        self.gaps = 0;
        self.last_tick = 0;
        for &x in &self.recent {
            let tick = to_ticks(x, self.scale).filter(|_| x.is_finite());
            self.gaps += usize::from(tick.is_none());
            self.last_tick = tick.unwrap_or(self.last_tick);
            if let Some(stats) = &mut self.stats {
                stats.push(self.last_tick, tick.is_some());
            }
        }
    }

    /// Records `x` as fed, or says why the stream could not take it.
    pub(crate) fn admit(&mut self, x: f64) -> PyResult<()> {
        let tick = to_ticks(x, self.scale).filter(|_| x.is_finite());
        let represented = tick.is_some() || !x.is_finite();
        let leaving = (self.recent.len() == self.m.get())
            .then(|| self.recent.front().copied())
            .flatten();
        let leaving_gap = leaving.is_some_and(|old| !old.is_finite());
        let gaps = self.gaps + usize::from(tick.is_none()) - usize::from(leaving_gap);
        let before = self.last_tick;
        let last_tick = tick.unwrap_or(before);
        let volatile = represented
            && self.stats.as_mut().is_some_and(|stats| {
                gaps == 0
                    && stats
                        .push(last_tick, tick.is_some())
                        .is_some_and(|w| w.kind == WindowKind::NonFinite)
            });
        if !represented || volatile {
            if volatile {
                // The statistics took the sample in; the refused one is not part of them.
                self.replay();
            }
            return Err(PyValueError::new_err(format!(
                "the sample {x} cannot be represented at the scale {} inferred from the history \
                 the stream started with: pass scale= (a coarser tick) when later samples may \
                 leave the range of the history",
                self.scale.get()
            )));
        }
        if let Some(stats) = &mut self.stats
            && gaps != 0
        {
            stats.push(last_tick, tick.is_some());
        }
        self.departed = leaving;
        if leaving.is_some() {
            self.recent.pop_front();
        }
        self.recent.push_back(x);
        self.gaps = gaps;
        self.last_tick = last_tick;
        Ok(())
    }

    /// Takes back the newest admitted sample, for one the stream then refused.
    pub(crate) fn forget_last(&mut self) {
        self.recent.pop_back();
        if let Some(old) = self.departed.take() {
            self.recent.push_front(old);
        }
        self.replay();
    }
}

pub(crate) fn bad(what: &str) -> PyErr {
    PyValueError::new_err(format!("invalid checkpoint: {what}"))
}

/// Takes `n` bytes off the front of `rest`, or fails if fewer remain.
pub(crate) fn take<'a>(rest: &mut &'a [u8], n: usize) -> PyResult<&'a [u8]> {
    if rest.len() < n {
        return Err(bad("Truncated"));
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

pub(crate) fn take_u64(rest: &mut &[u8]) -> PyResult<u64> {
    let bytes = take(rest, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes taken")))
}

/// The `left_P_`/`left_I_` history, count-prefixed, indices then distances as bits.
pub(crate) fn put_history(buf: &mut Vec<u8>, index: &[i64], distance: &[f64]) {
    buf.extend_from_slice(&(index.len() as u64).to_le_bytes());
    for index in index {
        buf.extend_from_slice(&index.to_le_bytes());
    }
    for distance in distance {
        buf.extend_from_slice(&distance.to_bits().to_le_bytes());
    }
}

/// [`put_history`]'s inverse, over exactly the bytes that remain.
pub(crate) fn take_history(rest: &mut &[u8]) -> PyResult<(Vec<i64>, Vec<f64>)> {
    let count = usize::try_from(take_u64(rest)?).map_err(|_| bad("Truncated"))?;
    // Exactly 16 bytes per row must remain: checked before anything allocates.
    if count.checked_mul(16) != Some(rest.len()) {
        return Err(bad(
            if count.checked_mul(16).is_some_and(|n| n > rest.len()) {
                "Truncated"
            } else {
                "Inconsistent"
            },
        ));
    }
    let (indices, distances) = rest.split_at(count * 8);
    let index = indices
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect();
    let distance = distances
        .chunks_exact(8)
        .map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().expect("8-byte chunk"))))
        .collect();
    Ok((index, distance))
}

/// Appends the checksum of every byte so far as the last eight bytes.
pub(crate) fn seal(buf: &mut Vec<u8>) {
    let sum = ladde::checkpoint::checksum(buf);
    buf.extend_from_slice(&sum.to_le_bytes());
}

/// `data` without its header and trailing checksum, once magic matches, the version is any of
/// `accepted` (also says which one it was) and the checksum over everything before it does. It
/// detects accidental corruption; it is not authentication.
pub(crate) fn unseal_versions<'a>(
    data: &'a [u8],
    magic: &[u8; 4],
    accepted: &[u16],
) -> PyResult<(u16, &'a [u8])> {
    let mut rest = data;
    if take(&mut rest, 4)? != magic {
        return Err(bad("BadMagic"));
    }
    let version = u16::from_le_bytes(take(&mut rest, 2)?.try_into().expect("2 bytes taken"));
    if !accepted.contains(&version) {
        return Err(bad("BadVersion"));
    }
    let body_len = data
        .len()
        .checked_sub(8)
        .filter(|&n| n >= 6)
        .ok_or_else(|| bad("Truncated"))?;
    let (body, stored) = data.split_at(body_len);
    if stored != ladde::checkpoint::checksum(body).to_le_bytes() {
        return Err(bad("BadChecksum"));
    }
    Ok((version, &body[6..]))
}

impl Stumpi {
    /// Folds every sample of `xs` through the engine, in order, and appends each completed
    /// window's neighbors to the history.
    fn feed(&mut self, xs: &[f64]) {
        for &x in xs {
            if let Some((_, kept)) = self.engine.update_topk(x, None) {
                self.history.push_row(kept);
            }
        }
    }

    /// [`Self::feed`] for a stream whose caller decides which windows are constant: the
    /// predicate (holding the GIL) is asked about each sample's window before the sample is
    /// folded in, and a predicate that fails leaves that sample out.
    fn feed_asking(&mut self, py: Python<'_>, xs: &[f64]) -> PyResult<()> {
        for &x in xs {
            let predicate = self.predicate.as_mut().expect("a predicate is being asked");
            let flag = predicate.push(py, x, PREDICATE)?;
            if let Some((_, kept)) = self.engine.update_topk(x, flag) {
                self.history.push_row(kept);
            }
        }
        Ok(())
    }

    /// The held windows' lists of `side`, or a `ValueError` for a stream built without
    /// `look_ahead=True`.
    fn ahead(&self, side: Side) -> PyResult<History> {
        self.engine.side(side).ok_or_else(|| {
            PyValueError::new_err("this stream does not look ahead: build it with look_ahead=True")
        })
    }

    /// `left_topk_P_` and `left_topk_I_`: the history padded to `(windows, k)`, or a
    /// `ValueError` when a `k` that large cannot be held for this many windows.
    fn padded_arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        padded_arrays(py, &self.history, self.engine.k().get())
    }
}

/// `(distances, indices)` of `history` padded to `(windows, k)`, or a `ValueError` when a `k` that
/// large cannot be held for this many windows.
pub(crate) fn padded_arrays<'py>(
    py: Python<'py>,
    history: &History,
    k: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (index, distance) = history.padded(k).ok_or_else(|| {
        PyValueError::new_err("k keeps more neighbors than can be reserved; use a smaller k")
    })?;
    let shape = [history.windows(), k];
    Ok((
        PyArray1::from_vec(py, distance).reshape(shape)?.into_any(),
        PyArray1::from_vec(py, index).reshape(shape)?.into_any(),
    ))
}

#[pymethods]
impl Stumpi {
    /// `T` is the warm-start history: run through the engine before this returns, so
    /// `left_P_`/`left_I_` already cover it.
    ///
    /// `egress=True` (the default) keeps the trailing `w` samples, `w` defaulting to `len(T)` as in
    /// STUMPY, where the length of the series stays fixed; `egress=False` keeps every sample and
    /// takes no `w`. `scale` is the tick size every sample is quantized to; when `None` it is
    /// inferred from `T` and `m` as `left_profile` infers it, and a later sample off that grid is
    /// rounded to it. `normalize=False` raises `NotImplementedError` (use `aampi`), `p` is ignored
    /// as in STUMPY, and `mp` raises `NotImplementedError`: this stream is always built from `T`.
    /// `k` is how many nearest earlier neighbors each window keeps (`left_topk_P_`,
    /// `left_topk_I_`); below 1 it is a `ValueError`.
    ///
    /// `T_subseq_isconstant_func` is STUMPY's: a function `f(a, w)` returning, for the array `a`
    /// (non-finite samples as NaN) and window length `w`, one boolean per window saying whether
    /// it is constant, in place of "all samples equal". It is applied to the whole of `T`, then,
    /// as each sample arrives, to the latest `m` samples. Holding the GIL, so a stream that has
    /// one does not release it while updating. A checkpoint of such a stream restores only with
    /// the function given again, and it cannot be pickled.
    #[new]
    #[pyo3(signature = (T, m, egress=true, normalize=true, p=2.0, k=1, mp=None, T_subseq_isconstant_func=None, *, scale=None, w=None, look_ahead=false))]
    #[allow(clippy::too_many_arguments, non_snake_case)]
    fn new(
        py: Python<'_>,
        T: Floats<'_>,
        m: i64,
        egress: bool,
        normalize: bool,
        #[allow(unused_variables)] p: f64,
        k: i64,
        mp: Option<&Bound<'_, PyAny>>,
        T_subseq_isconstant_func: Option<&Bound<'_, PyAny>>,
        scale: Option<ScaleArg>,
        w: Option<i64>,
        look_ahead: bool,
    ) -> PyResult<Self> {
        if !normalize {
            return Err(PyNotImplementedError::new_err(
                "normalize=False is the class aampi",
            ));
        }
        let k = resolve_k(k)?;
        require_none("mp", mp)?;
        let m_nz = window_length(m)?;
        let history = T.as_array().to_vec();
        let inferred = !matches!(scale, Some(ScaleArg::Tick(_)));
        let scale = resolve_scale(&history, m_nz, scale)?;
        let engine = if egress {
            let w = w.unwrap_or(history.len() as i64);
            let w_nz = window_bound(w, m_nz)?;
            let stream = if look_ahead {
                CoreBoundedStream::with_right_side(m_nz, w_nz, scale, k)
            } else {
                CoreBoundedStream::with_k(m_nz, w_nz, scale, k)
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
                CoreGrowingStream::with_right_side(m_nz, scale, k)
            } else {
                CoreGrowingStream::with_k(m_nz, scale, k)
            })
        };

        let guard = inferred.then(|| InferredScale::new(scale, m_nz, true, &history));
        let predicate = T_subseq_isconstant_func
            .filter(|function| !function.is_none())
            .map(|function| WindowPredicate::new(function, m_nz, PREDICATE))
            .transpose()?;
        let mut this = Self {
            engine,
            history: History::default(),
            guard,
            predicate,
        };
        match this.predicate.take() {
            None => py.detach(|| this.feed(&history)),
            Some(mut predicate) => {
                let flags = flags_of(py, T_subseq_isconstant_func, &history, m_nz, PREDICATE)?
                    .expect("a predicate was given");
                for (at, &x) in history.iter().enumerate() {
                    let flag = (at + 1 >= m_nz.get()).then(|| flags[at + 1 - m_nz.get()]);
                    if let Some((_, kept)) = this.engine.update_topk(x, flag) {
                        this.history.push_row(kept);
                    }
                }
                predicate.warm(&history);
                this.predicate = Some(predicate);
            }
        }
        Ok(this)
    }

    /// Folds in one causal sample, appending its window's entry to `left_P_`/`left_I_` if it
    /// completes one. Returns nothing, matching STUMPY's own `stumpi.update`. A finite sample with
    /// no tick at the stream's scale (beyond 2**53 ticks) raises `ValueError` and is not fed,
    /// where STUMPY, which works in floats, takes it; so does, when the stream inferred its
    /// `scale` from `T` and was not restored, a window too volatile for the arithmetic. Pass
    /// `scale=` for a stream that may leave the range of its history.
    fn update(&mut self, py: Python<'_>, t: f64) -> PyResult<()> {
        countable(&[t], self.engine.scale(), Remedy::Restart)?;
        if let Some(guard) = &mut self.guard {
            guard.admit(t)?;
        }
        if self.predicate.is_some() {
            let fed = self.feed_asking(py, &[t]);
            if fed.is_err()
                && let Some(guard) = &mut self.guard
            {
                // The predicate refused the sample, so the guard has not seen it either.
                guard.forget_last();
            }
            return fed;
        }
        py.detach(|| self.feed(&[t]));
        Ok(())
    }

    /// The tick size every sample is quantized to, given or inferred.
    #[getter(scale_)]
    fn scale(&self) -> f64 {
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

    /// The causal left matrix profile distances seen so far, oldest first: each window's nearest
    /// earlier neighbor.
    #[getter(left_P_)]
    fn left_p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_iter(py, self.history.nearest().map(|(_, distance)| distance))
    }

    /// The causal left matrix profile indices seen so far, oldest first; `-1` where a row has
    /// no neighbor.
    #[getter(left_I_)]
    fn left_i<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_iter(py, self.history.nearest().map(|(index, _)| index))
    }

    /// Each window's `k` nearest earlier neighbors' distances, shape `(windows, k)`, nearest
    /// first, `inf` where a window has fewer. Its last row is what STUMPY's `P_[-1]` holds after
    /// the same updates; the rows before it differ, since STUMPY revises them with later windows.
    #[getter(left_topk_P_)]
    fn left_topk_p<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.padded_arrays(py)?.0)
    }

    /// [`Self::left_topk_p`]'s indices, `-1` where a window has fewer than `k` neighbors.
    #[getter(left_topk_I_)]
    fn left_topk_i<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.padded_arrays(py)?.1)
    }

    /// How the engine's updates split across routes, as a dict: `narrow_updates` and
    /// `wide_updates` (samples fed on each arithmetic), `reanchors`, `switches_to_wide`,
    /// `switches_to_narrow`, `withheld_ticks` (samples kept from the hot arithmetic, their windows
    /// ranked by the exact cold route) and `cold_rows` (rows that ranked a pair by it).
    /// Informational only: no entry depends on it, and a restored stream counts from zero.
    fn route_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counts = match &self.engine {
            Engine::Growing(stream) => stream.route_counts(),
            Engine::Bounded(stream) => stream.route_counts(),
        };
        let dict = PyDict::new(py);
        dict.set_item("narrow_updates", counts.narrow_updates)?;
        dict.set_item("wide_updates", counts.wide_updates)?;
        dict.set_item("reanchors", counts.reanchors)?;
        dict.set_item("switches_to_wide", counts.switches_to_wide)?;
        dict.set_item("switches_to_narrow", counts.switches_to_narrow)?;
        dict.set_item("withheld_ticks", counts.withheld_ticks)?;
        dict.set_item("cold_rows", counts.cold_rows)?;
        Ok(dict)
    }

    /// The wrapper's whole state as bytes: the engine's core checkpoint plus the accumulated
    /// `left_P_`/`left_I_`, so `restore` brings the history back too.
    fn checkpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let (tag, core) = match &self.engine {
            Engine::Growing(stream) => (0u8, stream.checkpoint()),
            Engine::Bounded(stream) => (1u8, stream.checkpoint()),
        };
        let k = self.engine.k();
        let version = match (k == NonZeroUsize::MIN, self.predicate.is_some()) {
            (true, false) => VERSION,
            (true, true) => PREDICATE_VERSION,
            (false, false) => VERSION_TOPK,
            (false, true) => VERSION_TOPK_PREDICATE,
        };
        let topk = k != NonZeroUsize::MIN;
        let mut buf = Vec::with_capacity(core.len() + 24 * self.history.index.len() + 40);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.push(tag);
        buf.extend_from_slice(&(core.len() as u64).to_le_bytes());
        buf.extend_from_slice(&core);
        if let Some(predicate) = &self.predicate {
            let tail = predicate.tail();
            buf.extend_from_slice(&(tail.len() as u64).to_le_bytes());
            for x in tail {
                buf.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
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
    /// any bytes that are not a stumpi checkpoint or whose checksum does not match: it detects
    /// accidental corruption (bit rot, partial writes, truncation), not tampering.
    #[staticmethod]
    #[pyo3(signature = (data, T_subseq_isconstant_func=None))]
    #[allow(non_snake_case)]
    fn restore(data: &[u8], T_subseq_isconstant_func: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let function = T_subseq_isconstant_func.filter(|function| !function.is_none());
        let (version, mut rest) = unseal_versions(
            data,
            MAGIC,
            &[
                VERSION,
                PREDICATE_VERSION,
                VERSION_TOPK,
                VERSION_TOPK_PREDICATE,
            ],
        )?;
        let has_predicate = matches!(version, PREDICATE_VERSION | VERSION_TOPK_PREDICATE);
        let topk = matches!(version, VERSION_TOPK | VERSION_TOPK_PREDICATE);
        if has_predicate != function.is_some() {
            return Err(PyValueError::new_err(if has_predicate {
                "this checkpoint is of a stream with a T_subseq_isconstant_func: give it again"
            } else {
                "this checkpoint is of a stream without a T_subseq_isconstant_func"
            }));
        }
        let tag = take(&mut rest, 1)?[0];
        let core_len = usize::try_from(take_u64(&mut rest)?).map_err(|_| bad("Truncated"))?;
        let core = take(&mut rest, core_len)?;
        let engine = match tag {
            0 => Engine::Growing(CoreGrowingStream::restore(core).map_err(checkpoint_error)?),
            1 => Engine::Bounded(CoreBoundedStream::restore(core).map_err(checkpoint_error)?),
            _ => return Err(bad("Inconsistent")),
        };
        let m = match &engine {
            Engine::Growing(stream) => stream.window_length(),
            Engine::Bounded(stream) => stream.window_length(),
        };
        let predicate = match function {
            None => None,
            Some(function) => {
                let count = usize::try_from(take_u64(&mut rest)?).map_err(|_| bad("Truncated"))?;
                if count > m.get() || rest.len() < count.saturating_mul(8) {
                    return Err(bad("Truncated"));
                }
                let tail = take(&mut rest, count * 8)?
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_bits(u64::from_le_bytes(c.try_into().expect("8-byte chunk")))
                    })
                    .collect();
                Some(WindowPredicate::new(function, m, PREDICATE)?.with_tail(tail)?)
            }
        };
        let k = engine.k();
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
            predicate,
        })
    }

    /// Pickle support: rebuilds through `restore` from the checkpoint bytes.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        if self.predicate.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "a stumpi with a T_subseq_isconstant_func cannot be pickled: use checkpoint() and restore(data, T_subseq_isconstant_func)",
            ));
        }
        let restore = py.get_type::<Self>().getattr("restore")?;
        Ok((restore, (self.checkpoint(py),)))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Stumpi>()?;
    Ok(())
}
