#![cfg(loom)]

use ladde::adapter::published;
use ladde::sync::thread;

/// Any length a reader observes implies every entry below it is present and
/// equal to what the writer pushed; exhaustive over interleavings and over the
/// weak-memory reorderings the chosen atomic orderings permit. One reader thread
/// races the writer, and the main thread's read after joining the writer races
/// that reader, so two readers are concurrent without a second spawned one.
#[test]
fn loom_reader_sees_full_prefix() {
    loom::model(|| {
        let (mut publisher, reader) = published::<u64>(2).unwrap();
        let final_reader = reader.clone();

        let writer = thread::spawn(move || {
            publisher.push(1).unwrap();
            publisher.push(2).unwrap();
        });
        let racing_reader = thread::spawn(move || {
            let first = reader.snapshot();
            for (i, value) in first.iter().enumerate() {
                assert_eq!(*value, (i + 1) as u64);
            }
            // Published lengths never shrink: a later snapshot extends an earlier one.
            assert!(reader.snapshot().len() >= first.len());
        });

        writer.join().unwrap();
        // The join orders both pushes before this read: the whole prefix is visible, not
        // merely a consistent (possibly empty) one.
        assert_eq!(final_reader.snapshot(), vec![1, 2]);
        racing_reader.join().unwrap();
    });
}
