//! The right side of a stream that looks right, in its checkpoint: the stream's own checkpoint
//! as it would be without it, then each held window's lists, nearest first. A restored stream's
//! lists are the ones it saved, so it goes on to the same lists as one that never stopped.

use alloc::vec::Vec;

use super::{
    CheckpointError, MAGIC, Reader, VERSION_RIGHT, open_versions, put_f64, put_header_versioned,
    put_u8, put_u64, put_vec, seal,
};
use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::nearest::{ExactCorrelation, Held, variance_product};
use crate::right::{ColdItem, Column};
use crate::right_nonnorm::Column as SquaredColumn;
use crate::spill::Spilled;
use crate::wide::Wide;

/// A right-side checkpoint opened: the stream's own checkpoint, and the reader at the lists.
pub(crate) struct Wrapped<'a> {
    pub(crate) inner: &'a [u8],
    pub(crate) reader: Reader<'a>,
}

/// `bytes` opened as a checkpoint of `kind` that carries the right side, or `None` when it is
/// another version.
pub(crate) fn open_right(bytes: &[u8], kind: u8) -> Result<Option<Wrapped<'_>>, CheckpointError> {
    let is_right = bytes.len() >= 6
        && bytes[..4] == MAGIC
        && u16::from_le_bytes([bytes[4], bytes[5]]) == VERSION_RIGHT;
    if !is_right {
        return Ok(None);
    }
    let (mut reader, _) = open_versions(bytes, kind, &[VERSION_RIGHT])?;
    let len = usize::try_from(reader.u64()?).map_err(|_| CheckpointError::Truncated)?;
    let inner = reader.take(len)?;
    Ok(Some(Wrapped { inner, reader }))
}

/// `inner`, a stream's own checkpoint of `kind`, followed by what `put` writes.
pub(crate) fn wrap_right(kind: u8, inner: &[u8], put: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut buf = Vec::new();
    put_header_versioned(&mut buf, VERSION_RIGHT, kind);
    put_u64(&mut buf, inner.len() as u64);
    buf.extend_from_slice(inner);
    put(&mut buf);
    seal(buf)
}

fn put_entry(buf: &mut Vec<u8>, entry: Entry) {
    put_f64(buf, entry.rho());
    put_f64(buf, entry.squared_distance());
    put_u64(buf, entry.index().get().unwrap_or(u64::MAX));
}

fn get_entry(reader: &mut Reader) -> Result<Entry, CheckpointError> {
    let (rho, distance) = (reader.f64()?, reader.f64()?);
    let index = SampleIndex::new(reader.u64()?).ok_or(CheckpointError::Inconsistent)?;
    Entry::new(rho, distance, index).ok_or(CheckpointError::Inconsistent)
}

/// A cold item's smallest encoding: its entry and the tag of its correlation.
const COLD_ITEM_MIN_SIZE: usize = 24 + 1;

fn put_u128(buf: &mut Vec<u8>, value: u128) {
    put_u64(buf, value as u64);
    put_u64(buf, (value >> 64) as u64);
}

fn get_u128(reader: &mut Reader) -> Result<u128, CheckpointError> {
    let low = u128::from(reader.u64()?);
    Ok(low | u128::from(reader.u64()?) << 64)
}

fn put_wide(buf: &mut Vec<u8>, value: Wide<4>) {
    for &limb in value.limbs() {
        put_u64(buf, limb);
    }
}

fn get_wide(reader: &mut Reader) -> Result<Wide<4>, CheckpointError> {
    Ok(Wide::from_limbs([
        reader.u64()?,
        reader.u64()?,
        reader.u64()?,
        reader.u64()?,
    ]))
}

fn put_held(buf: &mut Vec<u8>, held: Held) {
    match held {
        Held::Hot(ExactCorrelation::BothConstant) => put_u8(buf, 0),
        Held::Hot(ExactCorrelation::OneConstant) => put_u8(buf, 1),
        Held::Hot(ExactCorrelation::Valid {
            sign,
            cross_squared,
            row_variance,
            neighbor_variance,
            ..
        }) => {
            put_u8(buf, 2);
            put_u8(buf, sign as u8);
            put_u128(buf, cross_squared);
            put_u64(buf, row_variance);
            put_u64(buf, neighbor_variance);
        }
        Held::Cold(Spilled::BothConstant) => put_u8(buf, 3),
        Held::Cold(Spilled::OneConstant) => put_u8(buf, 4),
        Held::Cold(Spilled::Valid {
            n,
            row_variance,
            neighbor_variance,
        }) => {
            put_u8(buf, 5);
            put_wide(buf, n);
            put_wide(buf, row_variance);
            put_wide(buf, neighbor_variance);
        }
    }
}

fn get_held(reader: &mut Reader) -> Result<Held, CheckpointError> {
    Ok(match reader.u8()? {
        0 => Held::Hot(ExactCorrelation::BothConstant),
        1 => Held::Hot(ExactCorrelation::OneConstant),
        2 => {
            let sign = reader.u8()? as i8;
            let cross_squared = get_u128(reader)?;
            let (row_variance, neighbor_variance) = (reader.u64()?, reader.u64()?);
            if !(-1..=1).contains(&sign) || row_variance >= 1 << 63 || neighbor_variance >= 1 << 63
            {
                return Err(CheckpointError::Inconsistent);
            }
            Held::Hot(ExactCorrelation::Valid {
                sign,
                cross_squared,
                row_variance,
                neighbor_variance,
                denominator: variance_product(row_variance, neighbor_variance),
            })
        }
        3 => Held::Cold(Spilled::BothConstant),
        4 => Held::Cold(Spilled::OneConstant),
        5 => Held::Cold(Spilled::Valid {
            n: get_wide(reader)?,
            row_variance: get_wide(reader)?,
            neighbor_variance: get_wide(reader)?,
        }),
        _ => return Err(CheckpointError::Inconsistent),
    })
}

fn put_cold_item(buf: &mut Vec<u8>, item: ColdItem) {
    put_entry(buf, item.entry);
    put_held(buf, item.held);
}

fn get_cold_item(reader: &mut Reader) -> Result<ColdItem, CheckpointError> {
    Ok(ColdItem {
        entry: get_entry(reader)?,
        held: get_held(reader)?,
    })
}

/// The held windows' lists, oldest window first.
pub(crate) fn put_columns(buf: &mut Vec<u8>, columns: Vec<Column>) {
    put_u64(buf, columns.len() as u64);
    for column in columns {
        put_vec(buf, &column.right, put_cold_item);
        put_vec(buf, &column.both, put_cold_item);
    }
}

/// [`put_columns`]' inverse for a stream keeping `k` neighbors, over exactly `windows` windows.
pub(crate) fn get_columns(
    reader: &mut Reader,
    k: usize,
    windows: usize,
) -> Result<Vec<Column>, CheckpointError> {
    let columns = reader.vec_with(16, |reader| {
        let right = reader.vec_with(COLD_ITEM_MIN_SIZE, get_cold_item)?;
        let both = reader.vec_with(COLD_ITEM_MIN_SIZE, get_cold_item)?;
        if right.len() > k || both.len() > k {
            return Err(CheckpointError::Inconsistent);
        }
        Ok(Column { right, both })
    })?;
    if columns.len() != windows {
        return Err(CheckpointError::Inconsistent);
    }
    Ok(columns)
}

/// The held windows' lists of a non-normalized stream, as pairs of squared distance and neighbor.
pub(crate) fn put_squared_columns(buf: &mut Vec<u8>, columns: Vec<SquaredColumn>) {
    let put_pair = |buf: &mut Vec<u8>, (squared, index): (u128, u64)| {
        put_u128(buf, squared);
        put_u64(buf, index);
    };
    put_u64(buf, columns.len() as u64);
    for (right, both) in columns {
        put_vec(buf, &right, put_pair);
        put_vec(buf, &both, put_pair);
    }
}

/// [`put_squared_columns`]' inverse for a stream keeping `k` neighbors, over `windows` windows.
pub(crate) fn get_squared_columns(
    reader: &mut Reader,
    k: usize,
    windows: usize,
) -> Result<Vec<SquaredColumn>, CheckpointError> {
    let pair = |reader: &mut Reader| Ok::<_, CheckpointError>((get_u128(reader)?, reader.u64()?));
    let columns = reader.vec_with(16, |reader| {
        let right = reader.vec_with(24, pair)?;
        let both = reader.vec_with(24, pair)?;
        if right.len() > k || both.len() > k {
            return Err(CheckpointError::Inconsistent);
        }
        Ok((right, both))
    })?;
    if columns.len() != windows {
        return Err(CheckpointError::Inconsistent);
    }
    Ok(columns)
}
