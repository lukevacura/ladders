//! STUMPY batch `stump` outputs written by `scripts/golden_stumpy.py`, and multi-dimensional cases
//! written by `scripts/golden_mstump.py` (exact `mstump` indices beside STUMPY's output), loaded
//! bit-exactly.

use std::fs;
use std::path::PathBuf;

/// One series and STUMPY's profiles of it; a missing neighbor is `u64::MAX`.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: String,
    pub m: usize,
    pub series: Vec<f64>,
    pub left_p: Vec<f64>,
    pub left_i: Vec<u64>,
    pub p: Vec<f64>,
    pub i: Vec<u64>,
}

const EXTENSION: &str = "golden";

fn fixtures_dir() -> PathBuf {
    // timing-ok: a compile-time path, fixed per build
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Every committed fixture's name, sorted.
pub fn names() -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(fixtures_dir())
        .expect("fixtures directory is readable")
        .map(|entry| entry.expect("fixture entry is readable").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == EXTENSION))
        .map(|path| path.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Panics on a missing or malformed file, since a fixture is test input.
pub fn load(name: &str) -> Fixture {
    let path = fixtures_dir().join(format!("{name}.{EXTENSION}"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let field = |key: &str| -> Vec<&str> {
        text.lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix(' '))
            .unwrap_or_else(|| panic!("{name}: no `{key}` line"))
            .split_ascii_whitespace()
            .collect()
    };
    let parsed = |key: &str, parse: fn(&str) -> Option<u64>| -> Vec<u64> {
        field(key)
            .into_iter()
            .map(|token| parse(token).unwrap_or_else(|| panic!("{name}: `{key}` has `{token}`")))
            .collect()
    };
    let f64s = |key: &str| {
        parsed(key, parse_bits)
            .into_iter()
            .map(f64::from_bits)
            .collect()
    };
    let [m] = parsed("m", parse_decimal)[..] else {
        panic!("{name}: `m` is not one value");
    };
    Fixture {
        name: name.to_owned(),
        m: m.try_into()
            .unwrap_or_else(|_| panic!("{name}: `m` overflows usize")),
        series: f64s("series"),
        left_p: f64s("left_p"),
        left_i: parsed("left_i", parse_decimal),
        p: f64s("p"),
        i: parsed("i", parse_decimal),
    }
}

/// One multi-dimensional case: `d` series of equal length, the exact `mstump` indices computed
/// independently of this library (the expected answer), and STUMPY's `mstump` output beside them
/// (a comparison). A missing neighbor is `u64::MAX`. Each `[k]` holds one value per window.
#[derive(Clone, Debug)]
pub struct MultiFixture {
    pub name: String,
    pub m: usize,
    pub discords: bool,
    pub include: Vec<usize>,
    pub series: Vec<Vec<f64>>,
    /// The exact indices for the series quantized at 2^-26: each distance's order decided on
    /// integers, a mean's order by high precision where f64 cannot tell, ties to the smaller start.
    pub exact_i: Vec<Vec<u64>>,
    /// STUMPY's distances, rounded by its own arithmetic: a comparison within a tolerance.
    pub stumpy_p: Vec<Vec<f64>>,
    /// STUMPY's indices: a comparison, equal to `exact_i` wherever no two means nearly tie.
    pub stumpy_i: Vec<Vec<u64>>,
}

const MULTI_EXTENSION: &str = "mgolden";

/// Every committed multi-dimensional fixture's name, sorted.
pub fn multi_names() -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(fixtures_dir())
        .expect("fixtures directory is readable")
        .map(|entry| entry.expect("fixture entry is readable").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == MULTI_EXTENSION))
        .map(|path| path.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Panics on a missing or malformed file, since a fixture is test input.
pub fn load_multi(name: &str) -> MultiFixture {
    let path = fixtures_dir().join(format!("{name}.{MULTI_EXTENSION}"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let field = |key: &str| -> Vec<&str> {
        text.lines()
            .find_map(|line| {
                let rest = line.strip_prefix(key)?;
                (rest.is_empty() || rest.starts_with(' ')).then_some(rest)
            })
            .unwrap_or_else(|| panic!("{name}: no `{key}` line"))
            .split_ascii_whitespace()
            .collect()
    };
    let parsed = |key: &str, parse: fn(&str) -> Option<u64>| -> Vec<u64> {
        field(key)
            .into_iter()
            .map(|token| parse(token).unwrap_or_else(|| panic!("{name}: `{key}` has `{token}`")))
            .collect()
    };
    let f64s = |key: &str| -> Vec<f64> {
        parsed(key, parse_bits)
            .into_iter()
            .map(f64::from_bits)
            .collect()
    };
    let single = |key: &str| -> usize {
        let [value] = parsed(key, parse_decimal)[..] else {
            panic!("{name}: `{key}` is not one value");
        };
        value.try_into().expect("a fixture count fits usize")
    };
    let d = single("d");
    let m = single("m");
    let series: Vec<Vec<f64>> = (0..d).map(|a| f64s(&format!("series{a}"))).collect();
    let windows = series[0].len() - m + 1;
    let exact_i = parsed("exact_i", parse_decimal);
    let p = f64s("stumpy_p");
    let i = parsed("stumpy_i", parse_decimal);
    assert_eq!(
        exact_i.len(),
        d * windows,
        "{name}: `exact_i` is d x windows"
    );
    assert_eq!(p.len(), d * windows, "{name}: `stumpy_p` is d x windows");
    assert_eq!(i.len(), d * windows, "{name}: `stumpy_i` is d x windows");
    MultiFixture {
        name: name.to_owned(),
        m,
        discords: single("discords") == 1,
        include: parsed("include", parse_decimal)
            .into_iter()
            .map(|a| a as usize)
            .collect(),
        series,
        exact_i: exact_i.chunks(windows).map(<[u64]>::to_vec).collect(),
        stumpy_p: p.chunks(windows).map(<[f64]>::to_vec).collect(),
        stumpy_i: i.chunks(windows).map(<[u64]>::to_vec).collect(),
    }
}

/// `None` unless exactly 16 hex digits; `from_str_radix` alone would accept a sign.
fn parse_bits(hex: &str) -> Option<u64> {
    let digits_only = hex.len() == 16 && hex.bytes().all(|b| b.is_ascii_hexdigit());
    digits_only.then(|| u64::from_str_radix(hex, 16).ok())?
}

/// `None` unless only decimal digits; `parse` alone would accept a sign.
fn parse_decimal(decimal: &str) -> Option<u64> {
    let digits_only = !decimal.is_empty() && decimal.bytes().all(|b| b.is_ascii_digit());
    digits_only.then(|| decimal.parse().ok())?
}

#[cfg(test)]
mod tests {
    use super::{load, names, parse_bits, parse_decimal};

    #[test]
    fn encoding_roundtrips_bitwise() {
        let specials = [
            f64::NAN.to_bits(),
            0x7ff8_0000_dead_beef, // NaN with a payload
            0xfff8_0000_0000_0000, // negative NaN
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            (-0.0f64).to_bits(),
            0.0f64.to_bits(),
            1,                                    // smallest subnormal
            f64::MIN_POSITIVE.to_bits() - 1,      // largest subnormal
            (-f64::MIN_POSITIVE / 3.0).to_bits(), // negative subnormal
        ];
        for bits in specials {
            assert_eq!(parse_bits(&format!("{bits:016x}")), Some(bits));
        }
        assert_eq!(parse_decimal(&u64::MAX.to_string()), Some(u64::MAX));
        assert_eq!(parse_bits("+fff000000000000"), None);
        assert_eq!(parse_decimal("+1"), None);
    }

    #[test]
    fn multi_golden_roundtrip() {
        let names = super::multi_names();
        assert!(!names.is_empty());
        for name in names {
            let fixture = super::load_multi(&name);
            let windows = fixture.series[0].len() - fixture.m + 1;
            assert_eq!(fixture.stumpy_p.len(), fixture.series.len(), "{name}");
            assert_eq!(fixture.exact_i.len(), fixture.series.len(), "{name}");
            for ((p, i), exact) in fixture
                .stumpy_p
                .iter()
                .zip(&fixture.stumpy_i)
                .zip(&fixture.exact_i)
            {
                assert_eq!(
                    (p.len(), i.len(), exact.len()),
                    (windows, windows, windows),
                    "{name}"
                );
                for ((&d, &j), &e) in p.iter().zip(i).zip(exact) {
                    assert_eq!(j == u64::MAX, d == f64::INFINITY, "{name}");
                    assert!(j == u64::MAX || j < windows as u64, "{name}");
                    assert!(e == u64::MAX || e < windows as u64, "{name}");
                }
            }
        }
    }

    #[test]
    fn golden_roundtrip() {
        let names = names();
        assert!(!names.is_empty());
        for name in names {
            let fixture = load(&name);
            let windows = fixture.series.len() - fixture.m + 1;
            for profile in [&fixture.left_p, &fixture.p] {
                assert_eq!(profile.len(), windows, "{name}");
            }
            for indices in [&fixture.left_i, &fixture.i] {
                assert_eq!(indices.len(), windows, "{name}");
            }
            let exclusion = ladde::exclusion::zone(fixture.m) as u64;
            for (k, (&j, &d)) in fixture.left_i.iter().zip(&fixture.left_p).enumerate() {
                assert_eq!(j == u64::MAX, d == f64::INFINITY, "{name}[{k}]");
                assert!(j == u64::MAX || j + exclusion < k as u64, "{name}[{k}]");
            }
            for (k, (&j, &d)) in fixture.i.iter().zip(&fixture.p).enumerate() {
                assert_eq!(j == u64::MAX, d == f64::INFINITY, "{name}[{k}]");
                assert!(j == u64::MAX || j < windows as u64, "{name}[{k}]");
            }
        }
    }
}
