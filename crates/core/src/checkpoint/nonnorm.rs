//! The non-normalized streams' checkpoints: the sanitized ticks still in reach and whether each
//! sample was present, and `k` when it is above one (as for the z-normalized streams). Every
//! diagonal is a function of those ticks, so none is written.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::{
    CheckpointError, Reader, get_k, get_m, get_scale, open_stream, put_bits, put_i64, put_k, put_m,
    put_scale, put_stream_header, put_u64, put_vec, seal, tick_in_range, within_allocation_bound,
};
use crate::quantize::Scale;

pub(crate) const KIND_GROWING: u8 = 2;
pub(crate) const KIND_BOUNDED: u8 = 3;

/// What a non-normalized checkpoint recovers, as plain data.
pub(crate) struct NonNormParts {
    pub(crate) m: NonZeroUsize,
    pub(crate) k: NonZeroUsize,
    /// The bounded stream's `w`; `0` for the growing one.
    pub(crate) horizon: usize,
    pub(crate) scale: Scale,
    /// Samples ever seen; `ticks` holds the newest `min(total, horizon)` of them (all, when
    /// growing).
    pub(crate) total: u64,
    pub(crate) ticks: Vec<i64>,
    pub(crate) present: Vec<bool>,
}

pub(crate) fn write_nonnorm_growing(
    m: NonZeroUsize,
    k: NonZeroUsize,
    scale: Scale,
    ticks: &[i64],
    present: &[bool],
) -> Vec<u8> {
    let mut buf = Vec::new();
    put_stream_header(&mut buf, KIND_GROWING, k);
    put_m(&mut buf, m);
    put_k(&mut buf, k);
    put_scale(&mut buf, scale);
    put_vec(&mut buf, ticks, put_i64);
    put_bits(&mut buf, present);
    seal(buf)
}

pub(crate) fn write_nonnorm_bounded(
    m: NonZeroUsize,
    k: NonZeroUsize,
    horizon: NonZeroUsize,
    scale: Scale,
    total: u64,
    ticks: &[i64],
    present: &[bool],
) -> Vec<u8> {
    let mut buf = Vec::new();
    put_stream_header(&mut buf, KIND_BOUNDED, k);
    put_m(&mut buf, m);
    put_k(&mut buf, k);
    put_u64(&mut buf, horizon.get() as u64);
    put_scale(&mut buf, scale);
    put_u64(&mut buf, total);
    put_vec(&mut buf, ticks, put_i64);
    put_bits(&mut buf, present);
    seal(buf)
}

pub(crate) fn read_nonnorm_growing(bytes: &[u8]) -> Result<NonNormParts, CheckpointError> {
    let (mut reader, topk) = open_stream(bytes, KIND_GROWING)?;
    let m = get_m(&mut reader)?;
    let k = get_k(&mut reader, topk)?;
    let scale = get_scale(&mut reader)?;
    let ticks = reader.vec_with(8, Reader::i64)?;
    let present = reader.bits()?;
    reader.finish()?;
    let parts = NonNormParts {
        m,
        k,
        horizon: 0,
        scale,
        total: ticks.len() as u64,
        ticks,
        present,
    };
    validate(&parts)?;
    Ok(parts)
}

pub(crate) fn read_nonnorm_bounded(bytes: &[u8]) -> Result<NonNormParts, CheckpointError> {
    let (mut reader, topk) = open_stream(bytes, KIND_BOUNDED)?;
    let m = get_m(&mut reader)?;
    let k = get_k(&mut reader, topk)?;
    let horizon = usize::try_from(reader.u64()?).map_err(|_| CheckpointError::Inconsistent)?;
    if horizon < m.get() {
        return Err(CheckpointError::Inconsistent);
    }
    if !within_allocation_bound(bytes.len(), m.get(), horizon - m.get()) {
        return Err(CheckpointError::Truncated);
    }
    let scale = get_scale(&mut reader)?;
    let total = reader.u64()?;
    let ticks = reader.vec_with(8, Reader::i64)?;
    let present = reader.bits()?;
    reader.finish()?;
    let parts = NonNormParts {
        m,
        k,
        horizon,
        scale,
        total,
        ticks,
        present,
    };
    validate(&parts)?;
    Ok(parts)
}

/// Only states a real stream reaches: every tick within `MAX_TICKS`, one flag per tick, the ticks
/// a run of forward-fill output, and the right count of them for `total`.
fn validate(p: &NonNormParts) -> Result<(), CheckpointError> {
    let bad = CheckpointError::Inconsistent;
    let growing = p.horizon == 0;
    let expected = if growing {
        p.total
    } else {
        p.total.min(p.horizon as u64)
    };
    if p.total >= 1 << 63
        || p.ticks.len() as u64 != expected
        || p.present.len() != p.ticks.len()
        || !p.ticks.iter().all(|&t| tick_in_range(t))
        || !fill_coherent(&p.ticks, &p.present, p.total == p.ticks.len() as u64)
    {
        return Err(bad);
    }
    Ok(())
}

/// Whether `ticks` with their presence flags are a run of forward-fill output: a missing sample
/// repeats the tick before it, or is `0` at the stream's very start (`from_start`).
fn fill_coherent(ticks: &[i64], present: &[bool], from_start: bool) -> bool {
    (0..ticks.len()).all(|k| match (present[k], k) {
        (true, _) => true,
        (false, 0) => !from_start || ticks[0] == 0,
        (false, _) => ticks[k] == ticks[k - 1],
    })
}
