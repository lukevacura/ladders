# exact-ref: an independent exact reference

Pure Python (`int`, `fractions.Fraction`, `math.isqrt`). It never imports the Rust crate or the
Python package, so it shares no code with the library it checks. A check computes the answer from
integer ticks here and compares it with what an engine returned.

It is O(n² m) per profile in exact arithmetic, so it suits series of up to about a thousand
samples. It is not a performance tool.

## What it covers

`quantize`, window kinds, exact correlation and squared distance, the left, right and two-sided
profiles, top-k left, AB-join, discords, and comparison helpers. Motifs are not covered: their
defaults are thresholds on float64 values, not a pure function of exact values.

## Rules

These are STUMPY 1.14.1's conventions, which the library follows.

- **Input.** Tick lists, with `None` for a missing sample. `quantize(x, scale)` is
  round-half-even of `x / scale` in f64; NaN, infinity or |ticks| > 2^53 give `None`.
- **Window kinds.** Any `None` makes a window non-finite. Otherwise V = m·Σt² − (Σt)²: V ≥ 2^63
  is non-finite, max − min = 0 is constant, and the rest are finite. A non-finite window has no
  answer and is never a candidate.
- **Correlation.** Constant against constant has ρ = 1 (d = 0). Constant against non-constant has
  ρ = 1/2 (d² = m). Otherwise ρ = c / √(V_a V_b) with c = m·Σxy − S_a S_b, and d² = 2m(1 − ρ).
- **Exactness.** `Rho` keeps (sign, c², V_a V_b) as integers and orders by exact
  cross-multiplication, never in f64, so scaled copies tie exactly. `dist_f64` is the correctly
  rounded square root.
- **Ties.** The larger ρ wins; equal ρ goes to the smaller index.
- **Exclusion zone (self-join).** e = ⌈m/4⌉, and a candidate needs |i − j| > e. Left: j < i − e.
  Right: j > i + e. Two-sided: the better of the two, left on ties.
- **Top-k left.** The k best left candidates, nearest first, ties to the smaller index.
- **AB-join.** Each window of A against every window of B, with no exclusion zone and ties to the
  smaller B index.
- **Discords.** From the two-sided profile: windows that have a neighbor, by exact distance,
  largest first; a window is taken if it is more than e from every window already taken.

## Use

```python
import exact_ref as X
ticks = X.quantize_series(series, scale)
nearest, left, right = X.two_sided_profile(ticks, m)
assert X.compare_profile(I, P, nearest, m) == []
```

Profiles return `(index, Rho)` per window, or `None` where there is no neighbor. Compare indices
exactly. Compare distances within a tolerance (`within_tol`, 1e-14 relative): an engine publishes
the square root of an f64 squared distance, so its last bits can differ from the correctly rounded
value by a few ulp.

`python -m pytest test_exact_ref.py` runs the unit tests, plus a cross-check against `ladders` on
100 random cases when that package and numpy are installed.
