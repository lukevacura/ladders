//! `Entry` carries no ranking of its own: it cannot be compared with `<`.

#[test]
fn entry_cannot_be_ordered() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/compile-fail/entry_not_ord.rs");
}

/// A k-dimensional distance is built from per-dimension distances, never from a bare `f64`.
#[test]
fn mean_distance_does_not_convert_from_f64() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/compile-fail/mean_distance_from_f64.rs");
}
