/// A global sample count from the stream's first sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleIndex(u64);

impl SampleIndex {
    /// "No valid neighbor"; orders after every real index.
    pub const NONE: Self = Self(u64::MAX);

    /// `None` for `u64::MAX`, which only [`Self::NONE`] may hold.
    pub const fn new(index: u64) -> Option<Self> {
        if index == u64::MAX {
            None
        } else {
            Some(Self(index))
        }
    }

    pub const fn get(self) -> Option<u64> {
        if self.is_none() { None } else { Some(self.0) }
    }

    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }
}

#[cfg(test)]
mod tests;
