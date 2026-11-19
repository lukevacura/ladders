/// Replaces each missing sample with the last present one, for the arithmetic only.
#[derive(Clone, Copy, Debug)]
pub struct ForwardFill<T> {
    last: Option<T>,
}

impl<T> Default for ForwardFill<T> {
    // `derive(Default)` would wrongly require `T: Default`; absence of a sample does not.
    fn default() -> Self {
        Self { last: None }
    }
}

/// A sample as the arithmetic sees it; a substituted value is reachable only through
/// [`Sanitized::Filled`].
#[must_use]
#[derive(Clone, Copy, Debug)]
pub enum Sanitized<T> {
    /// Missing, and no sample has been present yet.
    Leading,
    Present(T),
    /// The last present sample, standing in for a missing one.
    Filled(T),
}

impl<T: Copy> ForwardFill<T> {
    pub fn push(&mut self, sample: Option<T>) -> Sanitized<T> {
        match sample {
            Some(value) => {
                self.last = Some(value);
                Sanitized::Present(value)
            }
            None => match self.last {
                Some(last) => Sanitized::Filled(last),
                None => Sanitized::Leading,
            },
        }
    }
}

#[cfg(test)]
mod tests;
