use super::SampleIndex;

#[test]
fn sentinel_value_is_unconstructible() {
    assert_eq!(SampleIndex::new(u64::MAX), None);
    assert_eq!(SampleIndex::NONE.get(), None);
    assert_eq!(
        SampleIndex::new(u64::MAX - 1).unwrap().get(),
        Some(u64::MAX - 1)
    );
    assert!(SampleIndex::new(0).unwrap() < SampleIndex::NONE);
}
