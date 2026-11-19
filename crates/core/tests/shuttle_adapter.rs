#![cfg(shuttle)]

use ladde::adapter::published;
use ladde::sync::thread;

/// A longer run, many reader threads, randomized scheduling: whatever prefix a
/// reader observes, it matches what the writer pushed, in order.
#[test]
fn shuttle_many_readers_consistent_prefix() {
    shuttle::check_random_with_seed(
        || {
            const VALUES: u64 = 64;
            let (mut publisher, reader) = published::<u64>(VALUES as usize).unwrap();
            let final_reader = reader.clone();
            let readers: Vec<_> = (0..6).map(|_| reader.clone()).collect();

            let writer = thread::spawn(move || {
                for value in 1..=VALUES {
                    publisher.push(value).unwrap();
                }
            });
            let readers: Vec<_> = readers
                .into_iter()
                .map(|reader| {
                    thread::spawn(move || {
                        for (i, &value) in reader.snapshot().iter().enumerate() {
                            assert_eq!(value, (i + 1) as u64);
                        }
                    })
                })
                .collect();

            writer.join().unwrap();
            // The join orders every push before this read: the whole prefix is visible.
            let expected: Vec<u64> = (1..=VALUES).collect();
            assert_eq!(final_reader.snapshot(), expected.as_slice());
            for reader in readers {
                reader.join().unwrap();
            }
        },
        testkit::generators::TEST_SEED,
        200,
    );
}
