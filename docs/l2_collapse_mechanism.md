# Why the `L2` code's affine frontier collapses to 1

Measured on the committed `results/e5_weights/` (25 models: `L2`/`L4` at `p ∈ {0.01, 0.02}`,
five seeds each, plus five random-code controls), `d = 50`, `F = 100`.

This started as a check on Stage A's first cell, which reported `kappa_min = 1.00` for the `L2`
cell at `d = 50`, replicating E5. Theorem: feature `i` is separable at sparsity `s` iff
`kappa_i > s`. So `kappa = 1.00` says some feature is not separable even at `s = 1` — a single
active feature. That is a strong claim and it has a mechanism, but **not the one I first
guessed.**

## The first guess was wrong

At `s = 1` the unrelaxed collision is `Phi e_i = Phi e_j`, i.e. duplicate columns. And duplicates
are there:

| loss | seeds with a pair at `|cos| > 0.99` | total such pairs | coherence max |
|---|---|---|---|
| `L2` | 9/10 | 34 | 0.865 – 0.9999 |
| `L4` | 0/10 | 0 | 0.222 – 0.305 |
| random | 0/5 | 0 | 0.474 – 0.536 |

But one `L2` seed (`p=0.02`, seed 4) has **no** pair above 0.99 and still has `kappa = 1.000`.
So duplication cannot be the mechanism. It is a co-occurring symptom.

## The actual mechanism: unequal column norms

`kappa_i = 1` requires `Phi_{-i} z = phi_i` with `z >= 0`, `sum z <= 1`, `||z||_inf <= 1` — that
is, `phi_i` lies inside the convex hull of the other columns (the relaxed `s = 1` inactive hull
is the simplex). No duplicate is needed; a *short* column gets there on its own.

Inspecting the minimiser for the lowest-leverage feature:

| model | feature | `kappa` | `‖phi_i‖` | median `‖phi‖` | support of `z` | max `|cos|` to the columns used |
|---|---|---|---|---|---|---|
| `L2` p=0.01 s0 | 61 | 1.0004 | **0.068** | 0.429 | 50 | 0.65 |
| `L2` p=0.02 s4 | 79 | 1.0021 | **0.047** | 0.257 | 50 | 0.50 |
| `L4` p=0.01 s0 | 66 | 4.6200 | 0.774 | 0.808 | 51 | 0.23 |

The `L2` minimiser is a **spread combination of fifty columns**, none of them close in direction
to `phi_i`, with total weight only 0.49–0.65. It works because `phi_i` is five to fourteen times
shorter than a typical column. `L4`'s norms sit in a narrow band (0.74–0.89) and its minimiser
needs `sum z+ = 4.62` against `sum |z-| = 3.62` — a genuinely expensive representation.

So: **the `L2` loss leaves some feature directions almost unrepresented, and an almost
unrepresented direction falls inside the hull of the rest.** That is the collapse.

## What this means for the interpretation of `kappa`

Worth stating in the paper, because it cuts against a naive reading. A `kappa_i` near 1 has two
quite different possible causes:

1. the feature is represented but confusable with others (adversarial geometry);
2. the feature is barely represented at all (a near-zero column).

Operationally these are the same — no affine probe can read either — and that is exactly what an
interface capacity should report. But they are different *facts about the model*, and `kappa`
alone does not distinguish them. Reporting the column norm alongside `kappa` does, cheaply, and
we should.

This also sharpens why `R_geom` and `kappa` agree on `L2`: leverage `h_i` is small precisely for
short columns, so Corollary (failure threshold) fires on the same features. The three
measurements — `R_geom ≈ 19`, `h_min ≈ 0.008`, `kappa = 1.00` — are three views of one fact.

## Status

- Every number here is computed from committed weights by the snippets recorded in this
  session; it is a re-analysis, not a new campaign.
- It is `d = 50`, `F = 100`, ten `L2` models. Stage A's `L2` cell at `d = 50` reproduces
  `kappa_min = 1.00` independently. Whether the norm-collapse mechanism holds at `d = 100` and
  `d = 200` is a Stage A question and is **not** answered yet.
- Not yet in the manuscript. It belongs with the Stage A results, and the interpretation caveat
  above belongs in the discussion of `kappa`.
