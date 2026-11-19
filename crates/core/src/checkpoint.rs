//! The one place that reads and writes a stream's checkpoint bytes: magic, version,
//! stream kind, then every scalar and array [`crate::stream::GrowingStream`] or
//! [`crate::bounded::BoundedStream`] needs to resume bitwise identically. A ring or
//! front-growing buffer is written **logically** — its live elements, oldest first — never by
//! its physical layout, so a restore rebuilds one fresh through its own `new` and repeated
//! `push`, never by poking a private field.
//!
//! The last eight bytes are a checksum of everything before them, checked before any field
//! after the header is read. It detects accidental corruption (bit rot, partial writes,
//! truncation); it is not authentication: anyone can recompute it, and an altered checkpoint
//! with a valid checksum still passes the structural checks and can restore as a different
//! valid stream. Any single changed byte (equal length) is always detected; other damage with
//! probability about 1 − 2⁻⁶⁴.
//!
//! Every read here is bounds-checked against what is actually left in the buffer before it is
//! used, including before any length-prefixed array allocates anything (its declared length is
//! rejected unless enough bytes remain to hold that many elements), so a truncated or garbled
//! byte string can only ever produce [`CheckpointError`], never a panic. This module stays
//! at the level of bytes and plain data (`GrowingParts`/`BoundedParts`): a hidden field neither
//! stream type exposes (`WindowStats`'s own ring and deques, `ForwardFill`'s own last-seen
//! value) is rebuilt from plain data by `stream.rs`/`bounded.rs` themselves, through those
//! types' own public constructors — never by reading a private field from here.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::columns;
use crate::cross_term;
use crate::quantize::{MAX_TICKS, Scale};
use crate::stats::{Window, WindowKind, WindowStats};
use crate::znorm::TickWindow;

mod nonnorm;
mod right;
pub(crate) use nonnorm::{
    KIND_BOUNDED as KIND_NONNORM_BOUNDED, KIND_GROWING as KIND_NONNORM_GROWING,
};
pub(crate) use nonnorm::{
    read_nonnorm_bounded, read_nonnorm_growing, write_nonnorm_bounded, write_nonnorm_growing,
};
pub(crate) use right::{
    get_columns, get_squared_columns, open_right, put_columns, put_squared_columns, wrap_right,
};

const MAGIC: [u8; 4] = *b"MPCK";
const VERSION: u16 = 3;
/// A stream that keeps more than one neighbor per row: [`VERSION`]'s layout with `k` after `m`.
/// A stream of one keeps writing [`VERSION`], so its bytes are what they always were.
const VERSION_TOPK: u16 = 4;
/// A stream holding a window flagged constant whose variance numerator passes `2⁶³` (a window
/// its arrays hold as non-finite, whatever the flag): [`VERSION_TOPK`]'s layout with `k`
/// written whatever it is, then [`Spill`]. Any other stream keeps writing [`VERSION`] or
/// [`VERSION_TOPK`].
const VERSION_SPILL: u16 = 5;

/// A stream that looks right: a complete checkpoint of the same stream as it would be without
/// the right side, then the right side's lists. A stream that does not look right keeps writing
/// the other versions, so its bytes are what they always were.
const VERSION_RIGHT: u16 = 6;

pub(crate) const KIND_GROWING: u8 = 0;
pub(crate) const KIND_BOUNDED: u8 = 1;

/// Every way restoring a checkpoint's bytes can fail: a bad magic or version, a buffer that
/// ends before a declared field or array does, or a field that decodes to a shape no live
/// stream ever produces. `restore` returns one of these for any input rather than panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CheckpointError {
    /// The first four bytes are not this format's magic.
    BadMagic,
    /// The version field names a format this build does not read.
    BadVersion,
    /// The buffer ends before a declared field, tag or length-prefixed array does.
    Truncated,
    /// The trailing checksum does not match the bytes before it: the checkpoint was altered.
    BadChecksum,
    /// A field decodes to a shape no live stream ever produces: an unknown tag, a kind
    /// mismatch, or two arrays that must agree on length but do not.
    Inconsistent,
}

// ---------- primitive writer: every field is little-endian, floats as bits ----------

fn put_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_i64(buf: &mut Vec<u8>, v: i64) {
    put_u64(buf, v as u64);
}

fn put_f64(buf: &mut Vec<u8>, v: f64) {
    put_u64(buf, v.to_bits());
}

fn put_bool(buf: &mut Vec<u8>, v: bool) {
    put_u8(buf, u8::from(v));
}

/// A length prefix (`u64`), then `one` applied to every item, in order: every array in this
/// format is written this way — its live elements, oldest first — never by a ring's physical
/// layout or a front-growing buffer's own offset.
fn put_vec<T: Copy>(buf: &mut Vec<u8>, items: &[T], mut one: impl FnMut(&mut Vec<u8>, T)) {
    put_u64(buf, items.len() as u64);
    for &item in items {
        one(buf, item);
    }
}

/// A length prefix (`u64`), then the flags packed eight to a byte, lowest bit first.
fn put_bits(buf: &mut Vec<u8>, bits: &[bool]) {
    put_u64(buf, bits.len() as u64);
    for chunk in bits.chunks(8) {
        put_u8(
            buf,
            chunk
                .iter()
                .enumerate()
                .fold(0u8, |byte, (i, &bit)| byte | (u8::from(bit) << i)),
        );
    }
}

// ---------- bounds-checked reader ----------

/// A cursor over a checkpoint's bytes: every read either returns exactly the bytes it asked
/// for or [`CheckpointError::Truncated`], never a partial read or a panic.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self.pos.checked_add(n).ok_or(CheckpointError::Truncated)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(CheckpointError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    /// The whole buffer must have been consumed: trailing bytes are garbage, never ignored.
    pub(crate) fn finish(&self) -> Result<(), CheckpointError> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(CheckpointError::Inconsistent)
        }
    }

    pub(crate) fn tag(&mut self, expected: &[u8]) -> Result<(), CheckpointError> {
        if self.take(expected.len())? == expected {
            Ok(())
        } else {
            Err(CheckpointError::BadMagic)
        }
    }

    pub(crate) fn u8(&mut self) -> Result<u8, CheckpointError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, CheckpointError> {
        let bytes: [u8; 2] = self.take(2)?.try_into().expect("take(2) is 2 bytes");
        Ok(u16::from_le_bytes(bytes))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, CheckpointError> {
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("take(8) is 8 bytes");
        Ok(u64::from_le_bytes(bytes))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, CheckpointError> {
        Ok(self.u64()? as i64)
    }

    pub(crate) fn f64(&mut self) -> Result<f64, CheckpointError> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub(crate) fn bool(&mut self) -> Result<bool, CheckpointError> {
        Ok(self.u8()? != 0)
    }

    /// [`put_bits`]'s inverse; the declared count is rejected as truncated unless the bytes
    /// holding that many flags remain, and a set padding bit in the last byte is garbage.
    pub(crate) fn bits(&mut self) -> Result<Vec<bool>, CheckpointError> {
        let len = usize::try_from(self.u64()?).map_err(|_| CheckpointError::Truncated)?;
        let bytes = self.take(len.div_ceil(8))?;
        let bits: Vec<bool> = (0..len).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect();
        if len % 8 != 0 && bytes[len / 8] >> (len % 8) != 0 {
            return Err(CheckpointError::Inconsistent);
        }
        Ok(bits)
    }

    /// A length-prefixed array: the declared length is rejected as
    /// [`CheckpointError::Truncated`] before anything is allocated for it, unless at least
    /// `len * elem_size` bytes are still in the buffer — so a garbled length prefix can only
    /// ever allocate up to the input it actually came with, never the attacker-chosen count
    /// itself (the fuzz target's own reason to exist: remove this check and a crafted length
    /// asks the allocator for gigabytes instead of failing here).
    pub(crate) fn vec_with<T>(
        &mut self,
        elem_size: usize,
        mut one: impl FnMut(&mut Self) -> Result<T, CheckpointError>,
    ) -> Result<Vec<T>, CheckpointError> {
        let len = usize::try_from(self.u64()?).map_err(|_| CheckpointError::Truncated)?;
        let remaining = self.bytes.len() - self.pos;
        if elem_size == 0 || len > remaining / elem_size {
            return Err(CheckpointError::Truncated);
        }
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(one(self)?);
        }
        Ok(out)
    }
}

// ---------- header and checksum ----------

const CHECKSUM_LEN: usize = 8;

/// FNV-1a over `bytes`: each step is a bijection of the running hash, so any change confined
/// to one byte changes the checksum. Public (hidden) so a wrapper that frames a checkpoint
/// seals its own bytes with this same hash.
#[doc(hidden)]
pub fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn seal(mut body: Vec<u8>) -> Vec<u8> {
    let sum = checksum(&body);
    put_u64(&mut body, sum);
    body
}

/// `bytes` with its trailing checksum recomputed over everything before it: the way tests and
/// the fuzz target alter a checkpoint's fields without the checksum refusing it first.
#[doc(hidden)]
pub fn with_recomputed_checksum(bytes: &[u8]) -> Vec<u8> {
    seal(bytes[..bytes.len().saturating_sub(CHECKSUM_LEN)].to_vec())
}

/// A reader over `bytes` positioned after a valid header of `expected_kind` whose version is any
/// of `accepted` (also says which), limited to the bytes before the checksum, once that checksum
/// matches.
fn open_versions<'a>(
    bytes: &'a [u8],
    expected_kind: u8,
    accepted: &[u16],
) -> Result<(Reader<'a>, u16), CheckpointError> {
    let mut reader = Reader::new(bytes);
    let version = read_header(&mut reader, expected_kind, accepted)?;
    let body_len = bytes
        .len()
        .checked_sub(CHECKSUM_LEN)
        .filter(|&len| len >= reader.pos)
        .ok_or(CheckpointError::Truncated)?;
    let (body, stored) = bytes.split_at(body_len);
    if stored != checksum(body).to_le_bytes() {
        return Err(CheckpointError::BadChecksum);
    }
    reader.bytes = body;
    Ok((reader, version))
}

fn put_header(buf: &mut Vec<u8>, kind: u8) {
    put_header_versioned(buf, VERSION, kind);
}

fn put_header_versioned(buf: &mut Vec<u8>, version: u16, kind: u8) {
    buf.extend_from_slice(&MAGIC);
    put_u16(buf, version);
    put_u8(buf, kind);
}

fn read_header(
    reader: &mut Reader,
    expected_kind: u8,
    accepted: &[u16],
) -> Result<u16, CheckpointError> {
    reader.tag(&MAGIC)?;
    let version = reader.u16()?;
    if !accepted.contains(&version) {
        return Err(CheckpointError::BadVersion);
    }
    if reader.u8()? != expected_kind {
        return Err(CheckpointError::Inconsistent);
    }
    Ok(version)
}

/// The header of a stream's checkpoint keeping `k` neighbors per row, and `k` itself
/// after it when it is more than one.
fn put_stream_header(buf: &mut Vec<u8>, kind: u8, k: NonZeroUsize) {
    if k == NonZeroUsize::MIN {
        put_header(buf, kind);
    } else {
        put_header_versioned(buf, VERSION_TOPK, kind);
    }
}

/// [`put_stream_header`]'s `k`, written after `m`.
fn put_k(buf: &mut Vec<u8>, k: NonZeroUsize) {
    if k != NonZeroUsize::MIN {
        put_u64(buf, k.get() as u64);
    }
}

/// What a z-normalized stream's other arrays do not hold: for every live window, whether it is
/// flagged constant while its arrays hold it as non-finite (its variance numerator passes
/// `2⁶³`, and the cold route ranks it as constant).
#[derive(Clone, Default)]
pub(crate) struct Spill {
    pub(crate) constant: Vec<bool>,
}

impl Spill {
    /// Whether the stream needs [`VERSION_SPILL`] to hold it.
    fn holds_anything(&self) -> bool {
        self.constant.contains(&true)
    }
}

/// The header and `k` of a z-normalized stream's checkpoint, [`VERSION_SPILL`]'s when `spill`
/// holds anything.
fn put_spill_stream_header(buf: &mut Vec<u8>, kind: u8, k: NonZeroUsize, spill: &Spill) {
    if spill.holds_anything() {
        put_header_versioned(buf, VERSION_SPILL, kind);
    } else {
        put_stream_header(buf, kind, k);
    }
}

/// [`put_spill_stream_header`]'s `k`, written after `m`.
fn put_spill_k(buf: &mut Vec<u8>, k: NonZeroUsize, spill: &Spill) {
    if spill.holds_anything() {
        put_u64(buf, k.get() as u64);
    } else {
        put_k(buf, k);
    }
}

/// [`Spill`], when it holds anything, after the rest of the body.
fn put_spill(buf: &mut Vec<u8>, spill: &Spill) {
    if spill.holds_anything() {
        put_bits(buf, &spill.constant);
    }
}

/// A z-normalized stream's checkpoint, any version it writes: the reader and whether the
/// version carries [`Spill`].
fn open_spill_stream(
    bytes: &[u8],
    expected_kind: u8,
) -> Result<(Reader<'_>, u16), CheckpointError> {
    open_versions(
        bytes,
        expected_kind,
        &[VERSION, VERSION_TOPK, VERSION_SPILL],
    )
}

/// The `k` after `m` of a checkpoint of `version`.
fn get_spill_k(reader: &mut Reader, version: u16) -> Result<NonZeroUsize, CheckpointError> {
    if version != VERSION_SPILL {
        return get_k(reader, version == VERSION_TOPK);
    }
    let raw = usize::try_from(reader.u64()?).map_err(|_| CheckpointError::Inconsistent)?;
    NonZeroUsize::new(raw).ok_or(CheckpointError::Inconsistent)
}

/// [`put_spill`]'s inverse for a checkpoint of `version`: an empty [`Spill`] for an earlier
/// version, and one that holds nothing is not what a stream writes.
fn get_spill(reader: &mut Reader, version: u16) -> Result<Spill, CheckpointError> {
    if version != VERSION_SPILL {
        return Ok(Spill::default());
    }
    let spill = Spill {
        constant: reader.bits()?,
    };
    if !spill.holds_anything() {
        return Err(CheckpointError::Inconsistent);
    }
    Ok(spill)
}

/// Whether `spill` fits a stream of `windows` live windows, the arrays of each `stored`: one
/// flag per window, set only on a window held as non-finite (an earlier version's empty
/// `spill` fits any stream). The stream checks that each flagged window's ticks are present.
fn spill_coherent(spill: &Spill, windows: usize, stored: impl Fn(usize) -> Window) -> bool {
    spill.constant.is_empty()
        || (spill.constant.len() == windows
            && spill
                .constant
                .iter()
                .enumerate()
                .all(|(at, &flag)| !flag || stored(at).kind == WindowKind::NonFinite))
}

/// [`open_versions`] for a stream's checkpoint, which keeps `k` neighbors per row.
fn open_stream(bytes: &[u8], expected_kind: u8) -> Result<(Reader<'_>, bool), CheckpointError> {
    let (reader, version) = open_versions(bytes, expected_kind, &[VERSION, VERSION_TOPK])?;
    Ok((reader, version == VERSION_TOPK))
}

/// The `k` after `m`: one for a [`VERSION`] checkpoint, else the stored count, which is always
/// more than one (a stream of one writes [`VERSION`]).
fn get_k(reader: &mut Reader, topk: bool) -> Result<NonZeroUsize, CheckpointError> {
    if !topk {
        return Ok(NonZeroUsize::MIN);
    }
    let raw = usize::try_from(reader.u64()?).map_err(|_| CheckpointError::Inconsistent)?;
    NonZeroUsize::new(raw)
        .filter(|&k| k != NonZeroUsize::MIN)
        .ok_or(CheckpointError::Inconsistent)
}

// ---------- shared structural encodings ----------

fn put_window(buf: &mut Vec<u8>, w: Window) {
    put_i64(buf, w.sum);
    match w.kind {
        WindowKind::NonFinite => put_u8(buf, 0),
        WindowKind::Constant => put_u8(buf, 1),
        WindowKind::Valid {
            variance_numerator,
            inv_sqrt,
        } => {
            put_u8(buf, 2);
            put_u64(buf, variance_numerator);
            put_f64(buf, inv_sqrt);
        }
    }
}

fn get_window(reader: &mut Reader) -> Result<Window, CheckpointError> {
    let sum = reader.i64()?;
    let kind = match reader.u8()? {
        0 => WindowKind::NonFinite,
        1 => WindowKind::Constant,
        2 => WindowKind::Valid {
            variance_numerator: reader.u64()?,
            inv_sqrt: reader.f64()?,
        },
        _ => return Err(CheckpointError::Inconsistent),
    };
    Ok(Window { sum, kind })
}

fn put_tick_window(buf: &mut Vec<u8>, t: TickWindow) {
    put_i64(buf, t.entering);
    put_i64(buf, t.leaving);
    put_window(buf, t.stats);
}

fn get_tick_window(reader: &mut Reader) -> Result<TickWindow, CheckpointError> {
    Ok(TickWindow {
        entering: reader.i64()?,
        leaving: reader.i64()?,
        stats: get_window(reader)?,
    })
}

/// `entering` (8) + `leaving` (8) + `Window`'s own smallest encoding (sum `i64` plus a 1-byte
/// tag, 9): every `TickWindow` is at least this many bytes, whichever `WindowKind` it holds.
const TICK_WINDOW_MIN_SIZE: usize = 8 + 8 + 9;

fn put_scale(buf: &mut Vec<u8>, s: Scale) {
    put_f64(buf, s.get());
}

fn get_scale(reader: &mut Reader) -> Result<Scale, CheckpointError> {
    Scale::new(reader.f64()?).ok_or(CheckpointError::Inconsistent)
}

fn put_m(buf: &mut Vec<u8>, m: NonZeroUsize) {
    put_u64(buf, m.get() as u64);
}

fn get_m(reader: &mut Reader) -> Result<NonZeroUsize, CheckpointError> {
    let raw = usize::try_from(reader.u64()?).map_err(|_| CheckpointError::Inconsistent)?;
    NonZeroUsize::new(raw).ok_or(CheckpointError::Inconsistent)
}

// ---------- GrowingStream ----------

/// A borrowed view of every field [`crate::stream::GrowingStream::checkpoint`] writes: built
/// fresh from `&self`, never stored.
pub(crate) struct GrowingSnapshot<'a> {
    pub(crate) m: NonZeroUsize,
    pub(crate) k: NonZeroUsize,
    pub(crate) scale: Scale,
    pub(crate) seen_real_sample: bool,
    pub(crate) ticks: &'a [i64],
    pub(crate) sum: &'a [i64],
    pub(crate) r: &'a [f64],
    pub(crate) bias: &'a [f64],
    pub(crate) variance: &'a [u64],
    pub(crate) qt: &'a [i64],
    pub(crate) had_tick: &'a [bool],
    pub(crate) spill: Spill,
}

/// Everything [`read_growing`] recovers, as plain data: `stream.rs`'s own `restore` turns this
/// into a [`crate::stream::GrowingStream`] by rebuilding its rings, its front-growing buffer,
/// and its hidden `WindowStats`/`ForwardFill` state through their own public constructors.
pub(crate) struct GrowingParts {
    pub(crate) m: NonZeroUsize,
    pub(crate) k: NonZeroUsize,
    pub(crate) scale: Scale,
    pub(crate) seen_real_sample: bool,
    pub(crate) ticks: Vec<i64>,
    pub(crate) sum: Vec<i64>,
    pub(crate) r: Vec<f64>,
    pub(crate) bias: Vec<f64>,
    pub(crate) variance: Vec<u64>,
    pub(crate) qt: Vec<i64>,
    pub(crate) had_tick: Vec<bool>,
    pub(crate) spill: Spill,
}

pub(crate) fn write_growing(snapshot: GrowingSnapshot) -> Vec<u8> {
    let mut buf = Vec::new();
    put_spill_stream_header(&mut buf, KIND_GROWING, snapshot.k, &snapshot.spill);
    put_m(&mut buf, snapshot.m);
    put_spill_k(&mut buf, snapshot.k, &snapshot.spill);
    put_scale(&mut buf, snapshot.scale);
    put_bool(&mut buf, snapshot.seen_real_sample);
    put_vec(&mut buf, snapshot.ticks, put_i64);
    put_vec(&mut buf, snapshot.sum, put_i64);
    put_vec(&mut buf, snapshot.r, put_f64);
    put_vec(&mut buf, snapshot.bias, put_f64);
    put_vec(&mut buf, snapshot.variance, put_u64);
    put_vec(&mut buf, snapshot.qt, put_i64);
    put_bits(&mut buf, snapshot.had_tick);
    put_spill(&mut buf, &snapshot.spill);
    seal(buf)
}

pub(crate) fn read_growing(bytes: &[u8]) -> Result<GrowingParts, CheckpointError> {
    let (mut reader, version) = open_spill_stream(bytes, KIND_GROWING)?;
    let m = get_m(&mut reader)?;
    let k = get_spill_k(&mut reader, version)?;
    if !within_allocation_bound(bytes.len(), m.get(), 0) {
        return Err(CheckpointError::Truncated);
    }
    let scale = get_scale(&mut reader)?;
    let seen_real_sample = reader.bool()?;
    let ticks = reader.vec_with(8, Reader::i64)?;
    let sum = reader.vec_with(8, Reader::i64)?;
    let r = reader.vec_with(8, Reader::f64)?;
    let bias = reader.vec_with(8, Reader::f64)?;
    let variance = reader.vec_with(8, Reader::u64)?;
    if sum.len() != r.len() || r.len() != bias.len() || bias.len() != variance.len() {
        return Err(CheckpointError::Inconsistent);
    }
    let qt = reader.vec_with(8, Reader::i64)?;
    let had_tick = reader.bits()?;
    let spill = get_spill(&mut reader, version)?;

    if sum.is_empty() {
        if ticks.len() >= m.get() {
            return Err(CheckpointError::Inconsistent);
        }
    } else {
        let expected = sum
            .len()
            .checked_add(m.get())
            .and_then(|v| v.checked_sub(1))
            .ok_or(CheckpointError::Inconsistent)?;
        if ticks.len() != expected {
            return Err(CheckpointError::Inconsistent);
        }
    }

    reader.finish()?;

    let parts = GrowingParts {
        m,
        k,
        scale,
        seen_real_sample,
        ticks,
        sum,
        r,
        bias,
        variance,
        qt,
        had_tick,
        spill,
    };
    validate_growing(&parts)?;
    Ok(parts)
}

// ---------- BoundedStream ----------

/// A borrowed view of every field [`crate::bounded::BoundedStream::checkpoint`] writes: built
/// fresh from `&self`, never stored.
pub(crate) struct BoundedSnapshot<'a> {
    pub(crate) m: NonZeroUsize,
    pub(crate) k: NonZeroUsize,
    pub(crate) max_offset: usize,
    pub(crate) scale: Scale,
    pub(crate) seen_real_sample: bool,
    pub(crate) rows: u64,
    pub(crate) ticks: &'a [i64],
    pub(crate) windows: &'a [TickWindow],
    pub(crate) diagonals: &'a [i64],
    pub(crate) had_tick: &'a [bool],
    pub(crate) spill: Spill,
}

/// Everything [`read_bounded`] recovers, as plain data: `bounded.rs`'s own `restore` turns this
/// into a [`crate::bounded::BoundedStream`] by rebuilding its rings and its hidden
/// `WindowStats`/`ForwardFill` state through their own public constructors.
pub(crate) struct BoundedParts {
    pub(crate) m: NonZeroUsize,
    pub(crate) k: NonZeroUsize,
    pub(crate) max_offset: usize,
    pub(crate) scale: Scale,
    pub(crate) seen_real_sample: bool,
    pub(crate) rows: u64,
    pub(crate) ticks: Vec<i64>,
    pub(crate) windows: Vec<TickWindow>,
    pub(crate) diagonals: Vec<i64>,
    pub(crate) had_tick: Vec<bool>,
    pub(crate) spill: Spill,
}

pub(crate) fn write_bounded(snapshot: BoundedSnapshot) -> Vec<u8> {
    let mut buf = Vec::new();
    put_spill_stream_header(&mut buf, KIND_BOUNDED, snapshot.k, &snapshot.spill);
    put_m(&mut buf, snapshot.m);
    put_spill_k(&mut buf, snapshot.k, &snapshot.spill);
    put_u64(&mut buf, snapshot.max_offset as u64);
    put_scale(&mut buf, snapshot.scale);
    put_bool(&mut buf, snapshot.seen_real_sample);
    put_u64(&mut buf, snapshot.rows);
    put_vec(&mut buf, snapshot.ticks, put_i64);
    put_vec(&mut buf, snapshot.windows, put_tick_window);
    put_vec(&mut buf, snapshot.diagonals, put_i64);
    put_bits(&mut buf, snapshot.had_tick);
    put_spill(&mut buf, &snapshot.spill);
    seal(buf)
}

pub(crate) fn read_bounded(bytes: &[u8]) -> Result<BoundedParts, CheckpointError> {
    let (mut reader, version) = open_spill_stream(bytes, KIND_BOUNDED)?;
    let m = get_m(&mut reader)?;
    let k = get_spill_k(&mut reader, version)?;
    let max_offset = usize::try_from(reader.u64()?).map_err(|_| CheckpointError::Inconsistent)?;
    if !within_allocation_bound(bytes.len(), m.get(), max_offset) {
        return Err(CheckpointError::Truncated);
    }
    let scale = get_scale(&mut reader)?;
    let seen_real_sample = reader.bool()?;
    let rows = reader.u64()?;
    let ticks = reader.vec_with(8, Reader::i64)?;
    let windows = reader.vec_with(TICK_WINDOW_MIN_SIZE, get_tick_window)?;
    let diagonals = reader.vec_with(8, Reader::i64)?;
    if diagonals.len() != max_offset {
        return Err(CheckpointError::Inconsistent);
    }
    let had_tick = reader.bits()?;
    let spill = get_spill(&mut reader, version)?;

    reader.finish()?;

    let parts = BoundedParts {
        m,
        k,
        max_offset,
        scale,
        seen_real_sample,
        rows,
        ticks,
        windows,
        diagonals,
        had_tick,
        spill,
    };
    validate_bounded(&parts)?;
    Ok(parts)
}

// ---------- semantic validation: only states a real stream can reach ----------

/// What restore may allocate beyond the bytes it was given: a fixed slack, plus this many bytes
/// per input byte. The rings `WindowStats` and the bounded stream size from `m` and
/// `max_offset` alone, so a few claimed bytes of garbage must not buy gigabytes.
const RESTORE_SLACK_BYTES: u128 = 4 << 20;
const RESTORE_BYTES_PER_INPUT_BYTE: u128 = 64;
/// `WindowStats`' ring and both deques, per unit of `m`.
const BYTES_PER_WINDOW_LENGTH: u128 = 96;
/// The bounded stream's rings and arrays, its cold route's included, per unit of `max_offset`.
const BYTES_PER_OFFSET: u128 = 400;

/// Whether a checkpoint of `input_len` bytes may claim `m` and `max_offset`: a claim of more
/// state than the bytes (plus the slack) could hold is a truncated checkpoint.
fn within_allocation_bound(input_len: usize, m: usize, max_offset: usize) -> bool {
    let claimed = m as u128 * BYTES_PER_WINDOW_LENGTH + (max_offset as u128 + 1) * BYTES_PER_OFFSET;
    claimed <= RESTORE_SLACK_BYTES + RESTORE_BYTES_PER_INPUT_BYTE * input_len as u128
}

fn tick_in_range(tick: i64) -> bool {
    tick.unsigned_abs() <= MAX_TICKS as u64
}

/// Whether `ticks` with their presence flags `had` are a run of `ForwardFill` output: nothing
/// present yet means the placeholder `0`, a filled tick repeats the one before it, and
/// `seen_real_sample` agrees with the flags. `from_start` says `ticks[0]` is the stream's first.
fn presence_coherent(
    seen_real_sample: bool,
    ticks: &[i64],
    had: &[bool],
    from_start: bool,
) -> bool {
    if had.len() != ticks.len() {
        return false;
    }
    if !seen_real_sample {
        return ticks.iter().all(|&t| t == 0) && had.iter().all(|&h| !h);
    }
    if from_start && !had.iter().any(|&h| h) {
        return false;
    }
    let mut seen_before = from_start.then_some(false);
    for k in 0..ticks.len() {
        if had[k] {
            seen_before = Some(true);
            continue;
        }
        let ok = match (k, seen_before) {
            (0, _) => seen_before.is_none() || ticks[0] == 0,
            (_, Some(false)) => ticks[k] == 0,
            _ => ticks[k] == ticks[k - 1],
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Replays every tick with its presence through the one `WindowStats` that classifies windows,
/// calling `check(i, window)` for each completed window in order (`i` counts from 0). `None`
/// when any check fails; otherwise the rebuilt stats, whose hidden ring and deques depend only
/// on the trailing `m` ticks.
pub(crate) fn replay_stats(
    m: NonZeroUsize,
    ticks: &[i64],
    had: &[bool],
    mut check: impl FnMut(usize, Window) -> bool,
) -> Option<WindowStats> {
    let mut stats = WindowStats::new(m);
    for (k, (&tick, &had)) in ticks.iter().zip(had).enumerate() {
        if let Some(window) = stats.push(tick, had)
            && !check(k + 1 - m.get(), window)
        {
            return None;
        }
    }
    Some(stats)
}

/// Every invariant `GrowingStream::update` relies on, or `Inconsistent`: the state must be one
/// a real stream reaches, so a restored stream can do nothing `new` + `update` could not.
///  - every tick is within `MAX_TICKS`, and the presence flags are a run of
///    `ForwardFill` output (one per tick);
///  - replaying every tick and flag through `WindowStats` completes exactly the stored windows:
///    sum, kind (missing samples included) and columns, each in the one canonical encoding;
///  - `qt[j]` is the absolute `QT` of (newest row, `j`).
fn validate_growing(p: &GrowingParts) -> Result<(), CheckpointError> {
    let bad = CheckpointError::Inconsistent;
    let m = p.m.get();
    if !p.ticks.iter().all(|&t| tick_in_range(t))
        || !presence_coherent(p.seen_real_sample, &p.ticks, &p.had_tick, true)
    {
        return Err(bad);
    }

    let windows_match = |j: usize, window: Window| {
        [window, window.under(Some(true)), window.under(Some(false))]
            .iter()
            .any(|window| {
                let (r, bias, variance) = columns::encode(window);
                p.sum.get(j) == Some(&window.sum)
                    && p.r.get(j).map(|v| v.to_bits()) == Some(r.to_bits())
                    && p.bias.get(j).map(|v| v.to_bits()) == Some(bias.to_bits())
                    && p.variance.get(j) == Some(&variance)
            })
    };
    replay_stats(p.m, &p.ticks, &p.had_tick, windows_match).ok_or(bad)?;
    let stored = |j: usize| columns::window_of(p.sum[j], p.r[j], p.bias[j], p.variance[j]);
    if !spill_coherent(&p.spill, p.sum.len(), stored) {
        return Err(bad);
    }

    let exclusion = crate::exclusion::zone(m);
    let newest_row = p.sum.len().checked_sub(1);
    let count = newest_row.map_or(0, |row| row.saturating_sub(exclusion));
    if p.qt.len() != count {
        return Err(bad);
    }

    if let Some(row) = newest_row {
        for (j, &qt) in p.qt.iter().enumerate() {
            if qt != cross_term::wide(&p.ticks[row..row + m], &p.ticks[j..j + m]) {
                return Err(bad);
            }
        }
    }
    Ok(())
}

/// Every invariant `BoundedStream::update` relies on, or `Inconsistent` (same principle as
/// [`validate_growing`]):
///  - `rows` is below `2⁶³`; every tick is within `MAX_TICKS`;
///  - the rings hold exactly what `rows` completed windows leave: `min(rows + m − 1, w)` ticks
///    with one presence flag each (a run of `ForwardFill` output), `min(rows, max_offset + 1)`
///    windows, `max_offset` diagonals;
///  - replaying the live ticks and flags through `WindowStats` completes exactly the stored
///    windows (sum and kind); each window's entering tick is the last tick it covers and its
///    leaving tick the one before it (or the unread placeholder `0`);
///  - each live diagonal is the absolute `QT` of (newest row, row − d), every other slot zero.
fn validate_bounded(p: &BoundedParts) -> Result<(), CheckpointError> {
    let bad = CheckpointError::Inconsistent;
    let m = p.m.get();
    if p.rows >= 1 << 63 || !p.ticks.iter().all(|&t| tick_in_range(t)) {
        return Err(bad);
    }
    let w = p.max_offset.checked_add(m).ok_or(bad)?;
    let capacity = p.max_offset.checked_add(1).ok_or(bad)?;
    let total = if p.rows == 0 {
        if p.ticks.len() >= m {
            return Err(bad);
        }
        p.ticks.len() as u64
    } else {
        p.rows + m as u64 - 1
    };
    if p.ticks.len() as u64 != total.min(w as u64)
        || p.windows.len() as u64 != p.rows.min(capacity as u64)
    {
        return Err(bad);
    }

    // Global index of `ticks[0]`; the oldest window starts exactly there.
    let base = total - p.ticks.len() as u64;
    if !presence_coherent(p.seen_real_sample, &p.ticks, &p.had_tick, base == 0)
        || !spill_coherent(&p.spill, p.windows.len(), |at| p.windows[at].stats)
    {
        return Err(bad);
    }
    let window_count = p.windows.len() as u64;
    let windows_match = |i: usize, window: Window| {
        p.windows
            .get(i)
            .is_some_and(|tw| tw.stats.is_flagged_form_of(window))
    };
    replay_stats(p.m, &p.ticks, &p.had_tick, windows_match).ok_or(bad)?;
    for (start, tw) in p.windows.iter().enumerate() {
        let row = p.rows - window_count + start as u64;
        let covered = p.ticks.get(start..start + m).ok_or(bad)?;
        let leaving_ok = if row == 0 || p.max_offset == 0 {
            tw.leaving == 0
        } else if start > 0 {
            tw.leaving == p.ticks[start - 1]
        } else {
            tick_in_range(tw.leaving)
        };
        if tw.entering != covered[m - 1] || !leaving_ok {
            return Err(bad);
        }
    }

    let exclusion = crate::exclusion::zone(m);
    let live_hi = p
        .rows
        .checked_sub(1)
        .map(|row| row.min(p.max_offset as u64));
    for (slot, &qt) in p.diagonals.iter().enumerate() {
        let d = p.max_offset - slot;
        let live = live_hi.is_some_and(|hi| d > exclusion && d as u64 <= hi);
        let expected = if live {
            let row_start = p.ticks.len() - m;
            let neighbor = row_start.checked_sub(d).ok_or(bad)?;
            cross_term::wide(
                &p.ticks[row_start..row_start + m],
                &p.ticks[neighbor..neighbor + m],
            )
        } else {
            0
        };
        if qt != expected {
            return Err(bad);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use proptest::prelude::*;
    use std::format;
    use std::vec::Vec as StdVec;

    // ---------- primitive round trip ----------

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn primitives_round_trip(
            a in any::<u8>(), b in any::<u16>(), d in any::<u64>(),
            f in any::<i64>(), g in any::<u64>(), h in any::<bool>(),
        ) {
            let mut buf = Vec::new();
            put_u8(&mut buf, a);
            put_u16(&mut buf, b);
            put_u64(&mut buf, d);
            put_i64(&mut buf, f);
            put_f64(&mut buf, f64::from_bits(g));
            put_bool(&mut buf, h);

            let mut reader = Reader::new(&buf);
            prop_assert_eq!(reader.u8().unwrap(), a);
            prop_assert_eq!(reader.u16().unwrap(), b);
            prop_assert_eq!(reader.u64().unwrap(), d);
            prop_assert_eq!(reader.i64().unwrap(), f);
            prop_assert_eq!(reader.f64().unwrap().to_bits(), g);
            prop_assert_eq!(reader.bool().unwrap(), h);
        }

        #[test]
        fn vec_i64_round_trips(items in prop::collection::vec(any::<i64>(), 0..64)) {
            let mut buf = Vec::new();
            put_vec(&mut buf, &items, put_i64);
            let mut reader = Reader::new(&buf);
            let back = reader.vec_with(8, Reader::i64).unwrap();
            prop_assert_eq!(back, items);
        }
    }

    // ---------- truncated/garbled input never panics, always errs ----------

    #[test]
    fn every_prefix_of_a_valid_buffer_either_parses_or_errs_never_panics() {
        let mut buf = Vec::new();
        put_u8(&mut buf, 7);
        put_u64(&mut buf, 0xDEAD_BEEF_u64);
        put_vec(&mut buf, &[1i64, 2, 3, 4, 5], put_i64);
        put_f64(&mut buf, 1.5);

        for len in 0..=buf.len() {
            let mut reader = Reader::new(&buf[..len]);
            let _ = reader.u8();
            let _ = reader.u64();
            let _ = reader.vec_with(8, Reader::i64);
            let _ = reader.f64();
        }
    }

    #[test]
    fn a_length_prefix_past_the_buffer_errs_truncated_not_oom() {
        let mut buf = Vec::new();
        put_u64(&mut buf, u64::MAX);
        let mut reader = Reader::new(&buf);
        assert_eq!(
            reader.vec_with(8, Reader::i64),
            Err(CheckpointError::Truncated)
        );
    }

    #[test]
    fn empty_input_errs_rather_than_panics() {
        let growing: Result<GrowingParts, CheckpointError> = read_growing(&[]);
        assert_eq!(growing.err(), Some(CheckpointError::Truncated));
        let bounded: Result<BoundedParts, CheckpointError> = read_bounded(&[]);
        assert_eq!(bounded.err(), Some(CheckpointError::Truncated));
    }

    #[test]
    fn any_altered_byte_of_a_checkpoint_is_refused_by_its_checksum() {
        let bytes = growing_for("gapped").checkpoint();
        for at in 7..bytes.len() {
            let mut altered = bytes.clone();
            altered[at] ^= 0x10;
            assert_eq!(
                GrowingStream::restore(&altered).err(),
                Some(CheckpointError::BadChecksum),
                "byte {at}"
            );
        }
        let mut with_trailing_byte = bytes.clone();
        with_trailing_byte.push(0);
        assert_eq!(
            GrowingStream::restore(&with_trailing_byte).err(),
            Some(CheckpointError::BadChecksum)
        );
    }

    #[test]
    fn checksum_changes_with_any_single_changed_byte_and_with_length() {
        let bytes: StdVec<u8> = (0..64u8).collect();
        let sum = checksum(&bytes);
        for at in 0..bytes.len() {
            let mut altered = bytes.clone();
            altered[at] ^= 0x01;
            assert_ne!(checksum(&altered), sum, "byte {at}");
        }
        assert_ne!(checksum(&bytes[..63]), sum);
        assert_eq!(checksum(&[]), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn an_old_version_is_refused_before_its_checksum_is_read() {
        let mut bytes = bounded_for("steady").checkpoint();
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            BoundedStream::restore(&bytes).err(),
            Some(CheckpointError::BadVersion)
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let buf: StdVec<u8> = StdVec::from(*b"XXXX\x01\x00\x00");
        let result: Result<GrowingParts, CheckpointError> = read_growing(&buf);
        assert_eq!(result.err(), Some(CheckpointError::BadMagic));
    }

    // ---------- adversarial state that passes every structural check ----------

    /// A ramp-up `BoundedStream` checkpoint (`m = 4`, `w = 7`, no window completed yet) whose
    /// three ticks are `ticks`: lengths agree everywhere, so only a semantic check can refuse it.
    fn bounded_bytes_with_ticks(ticks: &[i64]) -> StdVec<u8> {
        let present = [true; 3];
        write_bounded(BoundedSnapshot {
            m: NonZeroUsize::new(4).unwrap(),
            k: NonZeroUsize::MIN,
            max_offset: 3,
            scale: Scale::new(1.0).unwrap(),
            seen_real_sample: true,
            rows: 0,
            ticks,
            windows: &[],
            diagonals: &[0; 3],
            had_tick: &present,
            spill: Spill::default(),
        })
    }

    #[test]
    fn ticks_spanning_the_whole_i64_range_are_refused_before_update_can_overflow() {
        // `update(1e12)` would reach `maintain_narrow_range`'s `hi - lo` over these.
        let bytes = bounded_bytes_with_ticks(&[i64::MIN, i64::MAX, 0]);
        let result = crate::bounded::BoundedStream::restore(&bytes);
        assert_eq!(result.err(), Some(CheckpointError::Inconsistent));
    }

    #[test]
    fn ticks_far_past_the_narrow_bound_restore_on_the_exact_arithmetic() {
        // Past twice the anchor's bound at `m = 4`, and inside the limit past which a tick is
        // withheld instead.
        let bytes = bounded_bytes_with_ticks(&[0, 3 << 29, 0]);
        let stream = crate::bounded::BoundedStream::restore(&bytes).unwrap();
        assert_eq!(stream.path(), crate::narrow::Path::Wide);
    }

    #[test]
    fn a_tick_past_the_withholding_limit_restores_withheld_and_narrow() {
        let bytes = bounded_bytes_with_ticks(&[0, 1 << 40, 0]);
        let stream = crate::bounded::BoundedStream::restore(&bytes).unwrap();
        assert_eq!(stream.path(), crate::narrow::Path::Narrow);
        assert_eq!(stream.checkpoint(), bytes);
    }

    #[test]
    fn growing_ticks_past_the_tick_bound_are_refused() {
        let bytes = write_growing(GrowingSnapshot {
            m: NonZeroUsize::new(4).unwrap(),
            k: NonZeroUsize::MIN,
            scale: Scale::new(1.0).unwrap(),
            seen_real_sample: true,
            ticks: &[i64::MIN, i64::MAX, 0],
            sum: &[],
            r: &[],
            bias: &[],
            variance: &[],
            qt: &[],
            had_tick: &[true; 3],
            spill: Spill::default(),
        });
        let result = crate::stream::GrowingStream::restore(&bytes);
        assert_eq!(result.err(), Some(CheckpointError::Inconsistent));
    }

    // ---------- one corrupted field at a time: a genuine checkpoint stops restoring ----------

    use crate::bounded::BoundedStream;
    use crate::stream::GrowingStream;

    fn growing_snapshot(p: &GrowingParts) -> GrowingSnapshot<'_> {
        GrowingSnapshot {
            m: p.m,
            k: p.k,
            scale: p.scale,
            seen_real_sample: p.seen_real_sample,
            ticks: &p.ticks,
            sum: &p.sum,
            r: &p.r,
            bias: &p.bias,
            variance: &p.variance,
            qt: &p.qt,
            had_tick: &p.had_tick,
            spill: p.spill.clone(),
        }
    }

    fn bounded_snapshot(p: &BoundedParts) -> BoundedSnapshot<'_> {
        BoundedSnapshot {
            m: p.m,
            k: p.k,
            max_offset: p.max_offset,
            scale: p.scale,
            seen_real_sample: p.seen_real_sample,
            rows: p.rows,
            ticks: &p.ticks,
            windows: &p.windows,
            diagonals: &p.diagonals,
            had_tick: &p.had_tick,
            spill: p.spill.clone(),
        }
    }

    /// A kind other than `window`'s own, to stand in for an older window's corrupted kind.
    fn other_kind(window: Window) -> Window {
        let kind = match window.kind {
            WindowKind::Valid { .. } => WindowKind::NonFinite,
            WindowKind::NonFinite => WindowKind::Constant,
            WindowKind::Constant => WindowKind::NonFinite,
        };
        Window { kind, ..window }
    }

    fn set_growing_window(p: &mut GrowingParts, j: usize, window: Window) {
        let (r, bias, variance) = columns::encode(&window);
        (p.sum[j], p.r[j], p.bias[j], p.variance[j]) = (window.sum, r, bias, variance);
    }

    fn series_for(name: &str) -> StdVec<f64> {
        let b = (i64::from(i32::MAX) / 4) as f64;
        let mut walk = StdVec::from([
            1.0, -2.0, 3.0, 0.0, -1.0, 2.0, -3.0, 1.0, 0.0, -2.0, 2.0, 1.0, -1.0, 3.0,
        ]);
        match name {
            "warm_up" => walk.truncate(3),
            "gapped" => walk[6] = f64::NAN,
            "late_gap" => walk[11] = f64::NAN,
            "re_anchor" => walk.extend([b + 1000.0, b + 1001.0, b + 999.0, b + 1002.0, b + 998.0]),
            "wide" => walk.extend([
                3.0 * b,
                3.0 * b + 1.0,
                3.0 * b - 1.0,
                3.0 * b + 2.0,
                3.0 * b + 3.0,
            ]),
            _ => {}
        }
        walk
    }

    const SCENARIOS: [&str; 5] = ["warm_up", "steady", "gapped", "re_anchor", "wide"];

    fn growing_for(name: &str) -> GrowingStream {
        let mut stream =
            GrowingStream::new(NonZeroUsize::new(4).unwrap(), Scale::new(1.0).unwrap());
        for x in series_for(name) {
            let _ = stream.update(x);
        }
        stream
    }

    fn bounded_for(name: &str) -> BoundedStream {
        let (m, w) = (NonZeroUsize::new(4).unwrap(), NonZeroUsize::new(7).unwrap());
        let mut stream = BoundedStream::new(m, w, Scale::new(1.0).unwrap()).unwrap();
        for x in series_for(name) {
            let _ = stream.update(x);
        }
        stream
    }

    /// Each field edit, applied alone to a genuine checkpoint, must stop it restoring.
    fn assert_growing_refused(
        label: &str,
        bytes: &[u8],
        edit: impl FnOnce(&mut GrowingParts),
        checked: &mut usize,
    ) {
        let mut parts = read_growing(bytes).unwrap();
        edit(&mut parts);
        let corrupted = write_growing(growing_snapshot(&parts));
        assert!(GrowingStream::restore(&corrupted).is_err(), "{label}");
        *checked += 1;
    }

    fn assert_bounded_refused(
        label: &str,
        bytes: &[u8],
        edit: impl FnOnce(&mut BoundedParts),
        checked: &mut usize,
    ) {
        let mut parts = read_bounded(bytes).unwrap();
        edit(&mut parts);
        let corrupted = write_bounded(bounded_snapshot(&parts));
        assert!(BoundedStream::restore(&corrupted).is_err(), "{label}");
        *checked += 1;
    }

    #[test]
    fn every_corrupted_growing_field_is_refused() {
        let mut checked = 0;
        for name in SCENARIOS {
            let stream = growing_for(name);
            let bytes = stream.checkpoint();
            let parts = read_growing(&bytes).unwrap();
            let windows = parts.sum.len();
            for j in 0..parts.qt.len() {
                let label = format!("{name}: qt[{j}]");
                assert_growing_refused(
                    &label,
                    &bytes,
                    |p| p.qt[j] = p.qt[j].wrapping_add(1),
                    &mut checked,
                );
            }
            // With no window completed yet, any tick values are a valid warm-up.
            for k in (0..parts.ticks.len()).filter(|_| windows > 0) {
                let label = format!("{name}: tick {k}");
                assert_growing_refused(&label, &bytes, |p| p.ticks[k] += 1, &mut checked);
            }
            for j in 0..windows {
                let label = format!("{name}: window {j} sum");
                assert_growing_refused(&label, &bytes, |p| p.sum[j] ^= 1, &mut checked);
                let label = format!("{name}: window {j} r");
                assert_growing_refused(
                    &label,
                    &bytes,
                    |p| p.r[j] = f64::from_bits(p.r[j].to_bits() ^ 1),
                    &mut checked,
                );
                let label = format!("{name}: window {j} bias");
                assert_growing_refused(
                    &label,
                    &bytes,
                    |p| p.bias[j] = f64::from_bits(p.bias[j].to_bits() ^ 1),
                    &mut checked,
                );
                let label = format!("{name}: window {j} variance");
                assert_growing_refused(&label, &bytes, |p| p.variance[j] ^= 1, &mut checked);
                // The kind of any window, older ones included, is proven from its own ticks.
                let label = format!("{name}: window {j} kind");
                assert_growing_refused(
                    &label,
                    &bytes,
                    |p| {
                        let own = columns::window_of(p.sum[j], p.r[j], p.bias[j], p.variance[j]);
                        set_growing_window(p, j, other_kind(own));
                    },
                    &mut checked,
                );
            }
        }
        assert!(checked > 300, "only {checked} corruptions ran");
    }

    #[test]
    fn a_presence_flag_changes_which_kind_an_older_window_must_have() {
        // One missing sample (tick 6, filled with tick 5's value): every window over it is
        // non-finite only because of that gap. Calling it present makes them all valid.
        let bytes = growing_for("gapped").checkpoint();
        let parts = read_growing(&bytes).unwrap();
        assert!(!parts.had_tick[6] && parts.ticks[6] == parts.ticks[5]);
        let mut checked = 0;
        assert_growing_refused(
            "gap called present",
            &bytes,
            |p| p.had_tick[6] = true,
            &mut checked,
        );

        let bytes = bounded_for("late_gap").checkpoint();
        assert_bounded_refused(
            "bounded gap called present",
            &bytes,
            |p| {
                let last = p.had_tick.iter().rposition(|&h| !h).unwrap();
                p.had_tick[last] = true;
            },
            &mut checked,
        );
    }

    #[test]
    fn every_corrupted_bounded_field_is_refused() {
        let mut checked = 0;
        for name in SCENARIOS {
            let stream = bounded_for(name);
            let bytes = stream.checkpoint();
            let parts = read_bounded(&bytes).unwrap();
            let count = parts.windows.len();
            for i in 0..parts.diagonals.len() {
                let label = format!("{name}: diagonals[{i}]");
                assert_bounded_refused(
                    &label,
                    &bytes,
                    |p| p.diagonals[i] = p.diagonals[i].wrapping_add(1),
                    &mut checked,
                );
            }
            // With no window completed yet, any tick values are a valid warm-up.
            for k in (0..parts.ticks.len()).filter(|_| count > 0) {
                let label = format!("{name}: tick {k}");
                assert_bounded_refused(&label, &bytes, |p| p.ticks[k] += 1, &mut checked);
            }
            for i in 0..count {
                assert_bounded_refused(
                    &format!("{name}: window {i} sum"),
                    &bytes,
                    |p| p.windows[i].stats.sum ^= 1,
                    &mut checked,
                );
                assert_bounded_refused(
                    &format!("{name}: window {i} kind"),
                    &bytes,
                    |p| p.windows[i].stats = other_kind(p.windows[i].stats),
                    &mut checked,
                );
                assert_bounded_refused(
                    &format!("{name}: window {i} entering"),
                    &bytes,
                    |p| p.windows[i].entering += 1,
                    &mut checked,
                );
                // The tick before the oldest live window is outside the ring: nothing proves it.
                let evicted_before =
                    i == 0 && parts.rows > 0 && parts.ticks.len() as u64 != parts.rows + 3;
                if !evicted_before {
                    assert_bounded_refused(
                        &format!("{name}: window {i} leaving"),
                        &bytes,
                        |p| p.windows[i].leaving += 1,
                        &mut checked,
                    );
                }
            }
        }
        assert!(checked > 90, "only {checked} corruptions ran");
    }

    #[test]
    fn a_checkpoint_claiming_more_state_than_its_bytes_hold_is_truncated() {
        // Genuine bytes with only `m` rewritten: a few dozen bytes cannot back 2^40 of window.
        let mut bytes = growing_for("warm_up").checkpoint();
        bytes[7..15].copy_from_slice(&(1u64 << 40).to_le_bytes());
        let bytes = with_recomputed_checksum(&bytes);
        assert_eq!(
            GrowingStream::restore(&bytes).err(),
            Some(CheckpointError::Truncated)
        );
        let mut bytes = bounded_for("warm_up").checkpoint();
        bytes[7..15].copy_from_slice(&(1u64 << 40).to_le_bytes());
        let bytes = with_recomputed_checksum(&bytes);
        assert_eq!(
            BoundedStream::restore(&bytes).err(),
            Some(CheckpointError::Truncated)
        );
        // And the same bytes with a window the slack covers restore fine.
        let bytes = growing_for("warm_up").checkpoint();
        assert!(GrowingStream::restore(&bytes).is_ok());
    }

    #[test]
    fn trailing_bytes_and_set_padding_bits_are_refused() {
        let mut bytes = growing_for("steady").checkpoint();
        bytes.insert(bytes.len() - 8, 0);
        let bytes = with_recomputed_checksum(&bytes);
        assert_eq!(
            GrowingStream::restore(&bytes).err(),
            Some(CheckpointError::Inconsistent)
        );

        // Three flags occupy bits 0..3 of one byte; a set bit 7 is not a flag.
        let mut encoded = StdVec::new();
        put_u64(&mut encoded, 3);
        put_u8(&mut encoded, 0b1000_0111);
        assert_eq!(
            Reader::new(&encoded).bits(),
            Err(CheckpointError::Inconsistent)
        );
        let mut encoded = StdVec::new();
        put_bits(&mut encoded, &[true, false, true]);
        assert_eq!(
            Reader::new(&encoded).bits(),
            Ok(StdVec::from([true, false, true]))
        );
    }

    #[test]
    fn a_tick_no_forward_fill_could_have_produced_is_refused() {
        // Everything else is exactly right for four ticks with no sample ever present (one
        // non-finite window, summing to 5), but a stream that has seen nothing writes only the
        // placeholder 0.
        let snapshot = |ticks: &[i64], seen_real_sample: bool, had: &[bool]| {
            write_growing(GrowingSnapshot {
                m: NonZeroUsize::new(4).unwrap(),
                k: NonZeroUsize::MIN,
                scale: Scale::new(1.0).unwrap(),
                seen_real_sample,
                ticks,
                sum: &[ticks.iter().sum()],
                r: &[0.0],
                bias: &[f64::NEG_INFINITY],
                variance: &[0],
                qt: &[],
                had_tick: had,
                spill: Spill::default(),
            })
        };
        assert_eq!(
            GrowingStream::restore(&snapshot(&[0, 0, 0, 5], false, &[false; 4])).err(),
            Some(CheckpointError::Inconsistent)
        );
        // The same shape from a placeholder-only run is an ordinary state.
        assert!(GrowingStream::restore(&snapshot(&[0, 0, 0, 0], false, &[false; 4])).is_ok());
    }
}
