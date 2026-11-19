use ladde::entry::Entry;
use ladde::index::SampleIndex;

fn main() {
    let index = SampleIndex::new(0).unwrap();
    let a = Entry::new(0.0, 1.0, index).unwrap();
    let b = Entry::new(0.0, 2.0, index).unwrap();
    let _ = a < b;
}
