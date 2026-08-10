# Decision log

Append-only. Each entry records a decision, when it was taken, what evidence was available at
the time, and what it commits us to. Entries are never edited after the fact; corrections are
added as new entries that reference the old one.

The point of this file is to make the order of events auditable. A referee should be able to
verify that the analysis plan predates the results it was applied to.

---

## D1 — 2026-08-08 — Pre-registration: the affine-parity outcome is a result, not a failure

**Status:** binding
**Evidence available:** E1–E6 as reported in the current manuscript. No new experiment from the
scaled campaign (J1–J8) has been run. No affine probe stronger than the three fixed readouts of
E5 has ever been fitted. No width beyond `d=50` has been trained.

**Commitment, recorded verbatim in the form agreed with the principal investigator:**

> If a properly trained, validation-selected affine decoder matches the network, we will report
> that result as a central finding rather than searching for a different metric that restores a
> nonlinear advantage.

**What this rules out.** Once the locked test sets of J1 are opened we will not: change the
primary estimand; change the sparsity grid; introduce a new summary statistic chosen because it
favours the network; restrict the reported readout family; or reclassify the affine oracle as
"not a fair baseline" after seeing that it wins.

**Pivot text authorised in advance**, to be used if the outcome is affine parity:

> The apparent separation between linear and nonlinear decoding disappears once analog
> reconstruction is distinguished from optimal affine support recovery.

and, if the robust affine frontier (G2) is informative enough to carry it:

> Selected linear readouts substantially underestimate the support-recovery capacity of the
> representation; the relevant hierarchy is analog reconstruction → affine support recovery →
> genuinely nonlinear recovery.

**Symmetry clause.** If instead the network robustly beats the affine optimum, that is the
stronger story and is reported as such. Both outcomes are publishable. The one thing that is
not acceptable is designing the analysis so that the network is guaranteed to win.

---

## D2 — 2026-08-08 — G2 is reformulated as a hard-margin programme

**Status:** binding
**Evidence:** the ball-constrained formulation `γ = ½ max_{‖w‖₂≤1}[a_i + L_{s−1} − U_s]` was
found to be degenerate before any implementation was built. The gap is positively homogeneous
of degree one and `w = 0` is feasible, so the optimum over the ball equals
`max(0, optimum over the sphere)` and can never be negative — the certificate of affine
insufficiency it was supposed to provide is unreachable. Confirmed numerically: on
non-separable codes the optimiser drives `w → 0` and returns ≈ 0.

**Decision.** Adopt `minimise ½‖w‖₂²` subject to a worst-case gap of at least 1. Feasible ⟺
affinely separable over every support of size `s`; the robust margin is `1/‖w*‖₂`;
infeasibility yields a Farkas certificate.

**Nomenclature, fixed here to avoid a signed scalar that hides the distinction:**

| Case | Reported quantity |
|---|---|
| Feasible | **robust affine margin** `γ_i*(s) = 1/‖w*‖₂ > 0` |
| Infeasible | **affine infeasibility certificate**, with the witnessing support pair |

We do not force a single signed scalar when the convex formulation distinguishes the two cases
better.

**Independent verification required before any use.** The certificate must be a mathematical
artefact checkable outside the solver that produced it: residuals recomputed in a fresh
process, and the witnessing supports exhibited explicitly. "The solver said infeasible" is not
evidence.

---

## D3 — 2026-08-08 — G2 must stand on its own, independently of nonlinearity

**Status:** design constraint

The robust affine frontier is **not** to be built as an instrument for proving that
nonlinearity is necessary. Its value is that it characterises the affine limit of a code,
whatever the empirical outcome turns out to be. If D1's affine-parity outcome materialises, G2
is the object that explains *why*, and it remains the paper's main methodological contribution.

Consequence for implementation: `robust_affine_frontier` takes a code and returns the frontier.
It has no argument, no branch and no output that refers to a trained network.

---

## D4 — 2026-08-08 — Theory scope is closed at G1 + G2 + G4

**Status:** binding until the core experiments are complete

In scope: the code-specific analog optimum and its leverage decomposition (G1); the robust
affine frontier (G2); the distribution-aware extension (G4).

Out of scope, and not to be reopened while the critical path is open: sign-rank lower bounds; a
sharp `s = Θ(d/log F)` converse; a new family of Welch-type bounds; any further frame-theory
programme.

**Acceptability clause.** If the novelty audit concludes that G1 is close to known results on
canonical duals, that G2 is a new formulation and algorithm rather than a deep theorem, and
that G4 is a known identity newly integrated, that is an acceptable outcome. The framework plus
the experimental evidence about trained systems is the contribution. We will not convert a
publishable systems paper into an open-ended search for a larger theorem.

---

## D5 — 2026-08-08 — Venue decision is deferred to a gate

**Status:** binding

Neurocomputing is the primary target and the work is designed to clear its standard. The final
choice among Neurocomputing / TMLR / Neural Networks is deferred until G1 and G2 are closed and
audited, the multi-width campaign has run, the optimal affine baseline is in, Boolean vs native
is characterised, and the second-system pilot has resolved.

No irreversible journal-specific framing before that gate. In particular the title, the abstract
and the emphasis of the contribution stay provisional.

---

## D6 — 2026-08-08 — The invented second task is withdrawn

**Status:** executed

The `square` / `abs` elementwise targets added to the toy model on 2026-08-08 are removed. They
were an invented protocol with no primary-source grounding, and a second task chosen by us to
be convenient is not independent evidence of generality.

Replacement: a Universal-AND pilot from primary source under a hard one-working-day budget,
with automatic fallback to a second *architecture* on the current task (two hidden layers) if
any abandonment criterion triggers. Abandonment criteria, agreed in advance: the original result
does not reproduce; the code needs substantial reconstruction; it is unclear which
representation should be diagnosed; integrating the diagnostic would require altering the
original protocol; or the work would consume several days before yielding interpretable
information.

If the fallback is used, the manuscript states plainly that it demonstrates architectural
generality, not task generality.

---

## D7 — 2026-08-08 — E6 is demoted to a scalability stress test

**Status:** executed in the plan, pending manuscript edit

Four widths cannot discriminate `d/log d` from the alternatives — the competing fits were
already reported and do not separate. E6 is retained as an algorithmic scalability
demonstration and consistency check. The scaling-law narrative and the `d/(16 ln d)` reference
curve are withdrawn from the results, the latter because its constant is inherited and not
derived anywhere in this work.

---

## D8 — 2026-08-08 — Provenance is mandatory for every run from now on

**Status:** binding

No experiment is run from an unidentified working tree. Every run manifest records the git
commit and the dirty status of the tree; if the tree is dirty this is recorded explicitly
alongside the list of modified files, rather than silently omitted.

Repository handling until further notice: local commits are made so that the state is clean and
reproducible; there is **no** public push, **no** definitive tag and **no** release or
visibility change without explicit approval. The immutable tag cited by the paper is created
only after the Phase-0 review.

---

## D9 — 2026-08-10 — G2 verdict: STRONG GO, on the collision-radius characterisation

**Status:** binding
**Evidence:** derivation plus exhaustive validation, ~20 000 (feature, sparsity) pairs against
brute-force strict separation of the two convex hulls over their full vertex sets. Zero
mismatches, on random codes and on constructed degenerate cases.

**Result.** Affine separability of feature `i` at sparsity `s` holds iff
`rho_i > 2 min(s, F-s) - 1`, where
`rho_i = min{ ||z||_1 : Phi_{-i} z = phi_i, 1^T z = 1, ||z||_inf <= 1 }`. One linear programme
per feature decides every sparsity, so the full frontier of a code costs `F` LPs instead of one
convex programme per (feature, sparsity) pair.

**Why the verdict moved from the first pass.** The first audit concluded "useful methodological
specialisation" because the machinery — Bertsimas–Sim, Farkas, cutting planes — is all standard.
The collision-radius statement is a mathematical result rather than a repackaging, and it is not
in the five nearest bodies of work: Euclidean/weighted superimposed codes give distinctness of
subset sums and rate bounds for families, not a per-code affine threshold; Donoho–Tanner
neighborliness is asymptotic, for random projections, and governs ell-1 recovery; the null-space
/ recoverable-supports line is per-matrix but again ell-1; group testing's `k`-separability is
Boolean-OR algebra.

**Caveats recorded so they are not lost later.** The proof is elementary — a Minkowski-difference
argument plus a counting identity — and must be presented as a proposition, not a theorem.
Quantitative group testing was not reached inside the time box and is the last place an
equivalent could exist; it must be read before submission.

**Two corrections to the first pass, both applied.** The Farkas certificate witnesses a convex
combination of active states equal to a convex combination of inactive states, not a single pair
of supports; `n_supports_involved` now records how many states are needed. And the compact
route's margin lower-bounded the exact one because it optimised a different objective, not
because the formulations differ; the margin is now checked for equality against the definition
— the minimum-norm solution over every `(A, B)` constraint — rather than against another
reformulation.

---

## D10 — 2026-08-10 — Theory is closed

**Status:** binding. Supersedes the scope in D4 only by adding the consequence package below;
nothing new is opened.

The consequence package is derived, implemented and validated (41 tests in
`tests/test_frontier_consequences.py`, 225 in the suite):

| Item | Result | Novelty |
|---|---|---|
| C1 | `rho_i >= 1/mu_i`; `mu_i < 1/(2s-1)` suffices, refining the classical `mu < 1/(2s)` | none as mathematics; places the classical condition as a corollary |
| C2 | `rho_min <= q - 1 <= d + 1`, hence `s_aff,robust <= ceil(d/2)` | the form is the classical spark condition, transported to the affine case: `q = spark([Phi; 1^T])` |
| C3 | `[I,I]` vs `[I,H]`: identical frame operator, leverage and analog optimum; `s_aff` 0 vs `Omega(sqrt d)` | elementary construction; earns the hierarchy its place |
| C4 | `delta_i(s)` is the hull distance and `gamma_i(s) = delta_i(s)/2`, with a certified noise tolerance | none — hard-margin SVM duality; fixes the factor of two |
| C5 | strict-integer `s_max` with tolerance; monotonicity restricted to `s <= floor(F/2)` | correction, not a result |

**Verdict unchanged: A — STRONG GO.** The collision radius is the single mathematical novelty.
The consequences are worth stating for interpretability and for the connection to spark and
coherence, and are claimed at that size.

**Closed.** No sign-rank, no random-code asymptotics, no further theoretical project before the
Stage A campaign. The one remaining literature debt is quantitative group testing, which stays
on the pre-submission checklist rather than the critical path.

**Next, in order:** `probes.py`, the G1 reanalysis of existing results, then J0 on the Spark.
