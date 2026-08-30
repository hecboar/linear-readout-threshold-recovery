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

---

## D11 — 2026-08-10 — The G1 reanalysis reverses the reading of E5's attainment ratios

**Status:** finding, to be carried into the manuscript and replicated in the campaign
**Evidence:** `docs/g1_reanalysis.md`, from `results/e5_weights/` (E5 re-run for weights, 186 min
CPU, 25/25 models matched against the committed diagnostics to 8e-10 relative — the residual is
float32 weight storage, not solver noise).

**What the published number actually measured.** `R_readout(pinv) = 1.000000` in every cell, by
construction: the calibrated pseudoinverse *is* the code-specific optimum (G1). So both
"L4 sits 1.0006 above the floor" and "L2 sits 19.0 above the floor" are statements about the
**leverage heterogeneity of the code** and say nothing whatever about any decoder. The
manuscript currently presents them as if they characterised the model.

**The split, per cell:**

| cell | `R_geom` | `R_readout(wout)` | leverage range | CV | limited by |
|---|---|---|---|---|---|
| L4 p=0.01 | 1.0006 | 1.0254 | 0.480–0.519 | 0.017 | neither |
| L2 p=0.01 | 19.0052 | 1.0053 | 0.009–0.994 | 0.867 | geometry |
| L4 p=0.02 | 1.0004 | 1.0249 | 0.484–0.518 | 0.014 | neither |
| L2 p=0.02 | 29.1032 | 1.0089 | 0.009–0.992 | 0.952 | geometry |
| random | 1.0170 | 1.0000 | 0.393–0.626 | 0.092 | neither |

**Two readings that change.**

1. **`L2` is geometry-limited, and its own decoder is near-optimal for the code it built** —
   within 0.5–0.9% of the best readout that code admits. It is not decoding crudely; it is doing
   close to the best linear thing available given a code whose leverage runs from 0.009 to 0.994.
   Minimum leverage near 0.01 means some features sit almost outside the row space: effectively
   dead directions. That is the quantitative form of the reading that `L2` does not compute in
   superposition, and it is sharper than the current `frac_relu_clipped` evidence.
2. **`L4` is the reverse.** Its code is near-ideal — leverage essentially uniform, CV 0.017 — but
   its trained decoder leaves about 2.5% on the table against its own code's optimum, which in
   relative terms is *worse* than `L2`'s decoder. The 1.0006 was never evidence that `L4`'s
   decoder is good.

**Consistency check.** `R_readout(wout) = 1.0000` exactly for the random control because
`random_code_baseline` sets `W_out = pinv(Phi)`, so its "own decoder" is the code-specific
optimum by construction. That is audit finding P1 resurfacing and it confirms the decomposition
behaves as intended.

**What this does not license.** Five seeds at one width, unchanged. The optimisation underneath
is Capon/MVDR and is classical (see the novelty audit); the split is arithmetic on it. This
reinterprets an existing number and adds no new evidence.

**Carried forward.** `R_readout(wout)` is the quantity the scaled campaign should track: it asks
whether training produced a decoder that is good *for the code it built*, which is a different
question from whether the code is good, and the two answers point opposite ways here.

---

## D12 — 2026-08-11 — Theory reopened briefly, and the hierarchy is now an implication

**Status:** finding. Supersedes the "theory closed" clause of D10 for these five items only;
nothing further is opened.

Five extensions were worked through (`docs/theory_extensions.md`). Four conclusions from the
first pass were corrected by the external review and all four corrections are adopted:

| item | first pass | corrected |
|---|---|---|
| E2 | "the affine level is now distribution-aware" | **overclaimed.** Two laws with the same support give the same frontier, so it is a *support-level worst-case* property. Both levels are induced by one law `P`: analog through `C = E[aa^T]`, affine through `supp(P)` |
| E3 | lower bound `rho_i >= sqrt(h_i/(1-h_i))`, nearly vacuous | **wrong direction.** The needed bound is *upper*, via Sherman-Morrison: `min ||z||_2^2 = h_i/(1-h_i)`, hence `kappa_i <= 1 + sqrt((F-1)h_i/(1-h_i))`. My original conjecture routed through coherence, which cannot work at all — `rho >= 1/mu` is a lower bound, so large `mu` puts no ceiling on `rho` |
| E4b | "equality essentially never attained" | **explained.** For the *pure* `ell_1` problem `[I,H]` attains it exactly (`sqrt(d) = 1/mu`, verified at four widths); the affine row `1^T z = 1` is what breaks it, which accounts for the measured 1.018 |
| E5 | "no object exists; structural" | **half right.** No *frontier*, but a finite Radon witness of at most about `d+2` states, derived from G2's own certificate. Stronger and implementable |

**The result.** `h_i <= (s-1)^2/((F-1)+(s-1)^2)` implies feature `i` is not affinely separable at
sparsity `s`. Verified on 967 predicted failures with no counterexample. Combined with `[I, I]` —
uniform leverage, `kappa = 1` — the hierarchy is **strictly one-way**: bad analog geometry implies
a bad affine frontier, and the converse fails.

**It predicts the trained models.** At `F = 100` the `s = 2` threshold is `0.01`; `L2` has
`h_min = 0.0091` and measured `kappa = 1.001`, i.e. separable at `s = 1` and lost at `s = 2`,
exactly as the bound requires. `L4` has `h_min = 0.4822` and `kappa = 4.672`. This is the only
place the theory touches a trained network, and it holds on five seeds at one width — which makes
replicating the arrow across widths a Stage A deliverable rather than a new project.

**Also adopted:** `|S| <= s` becomes the primary formulation (E1). Dropping `1^T z = 1` is what
makes E3 provable, so E1 and E3 are one reform. Monotonicity now holds by construction and the
`s > F/2` artefact is gone.

**Not opened:** the `1-delta` probabilistic affine frontier, sign-rank, and any bound on hidden
units. E5's witness extractor is authorised only as a one-day build when Stage A is under way.

---

## D13 — 2026-08-11 — A seed names the same code on every device

**Status:** binding

`scripts/check_env_gpu.py` refused to clear the DGX Spark, reporting a relative weight
difference of 2.07 after twenty float64 steps. That was not a numerical fault: a CUDA generator
and a CPU generator produce different streams for the same seed, so the check compared two
different random problems and could never have passed on any accelerator. On identical inputs
the arithmetic agrees to 0.0 exactly, measured on the machine.

The false alarm concealed a real defect. `train_toy_models_batched` drew the *initialisation*
from a device generator, so `seed 0` named a different `W_in` on GPU than on CPU. Every
theoretical quantity in this work is a function of `W_in` — leverage, `kappa`, `R_geom` — so the
theory and the measurement would have described different objects.

**Decided.** The initialisation is always drawn on the CPU stream. On CPU the batches continue
that same generator, so a CPU run stays byte-identical to `train_toy_model` and every committed
result still reproduces; only the accelerator changes, and it changes to agree.

**Deliberately not decided the other way.** The per-step batches stay on the compute device.
Drawing 32M uniforms per step on the host costs about two orders of magnitude more than the
training step it feeds, which would produce exactly the host-bound idle GPU that J0 exists to
detect. So a run is reproducible from its seed *on the same device class*, `device` is recorded
in every run record, and the docstring says this instead of claiming more. The `cpu_data` flag
forces full device-independence and exists so the gate can test arithmetic in isolation.

---

## D14 — 2026-08-11 — The repository is venue-neutral, and internal work is separated from the release

**Status:** executed for the renames; the pruning is pending approval

Seven top-level paths named the target journal, and the venue is not decided until after Stage
A. Renaming is hygiene rather than framing, so it was done now: `paper/` became
`paper/`, and the six venue-named documents moved into `internal/` under neutral names, joined
by the submission material that has no place in a public repository — cover letter, reviewer
candidates, prepared rebuttals, external-review prompt.

**On the earlier version.** The artefacts stay and the labels go. Preserving a superseded
manuscript is normal practice; naming the venues it passed through, and publishing our own
enumerated defect list, is not, and neither belongs in the scientific record. The README
now states the substantive fact — an audit found numerical claims in the earlier appendix that
its own published code does not reproduce — and points at `docs/decision_log.md` for the
registered predictions, which is the part that is evidence rather than confession.

**Open.** `superseded-submission/` is preserved
intact by instruction. Either keep it as it is or exclude it from the
public release. `internal/` must not ship; removing it costs no reproducibility, since it holds
no code, no configuration and no results.

---

## D15 — 2026-08-11 — E6's demotion is executed in the manuscript

**Status:** executed, closing D7

D7 decided this on 2026-08-08 and the manuscript still carried both the scaling-law reading and
the `d/(16 ln d)` reference curve. Now: the section is retitled a scalability stress test and
opens by saying it lacks the resolution to measure a scaling law; the reference curve is gone
from the figure, caption, prose and generated macros.

The fit is *kept*, because the reason for the withdrawal is that three shapes are
indistinguishable on four widths, and the `R^2` values are the evidence for that. It now leads
with the non-identifiability and claims no law, neither for `d/ln d` nor against it.

The abstract had to change with it. It claimed the `L4` model "reaches" the floor, which D11
showed is a statement about leverage evenness containing nothing about the decoder — so the
abstract was selling a result the paper had already retracted. Rewritten around the two
criteria, the code's own floor, the per-feature affine capacity and the one-way implication. At
250 words against a 250-word limit it cost the `O(Fd^2)` framing and the Hänni recast, both of
which survive in the contributions. The highlights had the same defect and the scalability
bullet is dropped, which is what demoting E6 means.

---

## D16 — 2026-08-11 — Worker pools spawn, and long phases must report progress

**Status:** binding

Stage A's `d=100` cell ran for two and a half hours on work that takes nineteen minutes. Two
separate defects, and the second is why the first went unnoticed.

**The pool must spawn.** `map_trials` exports `OMP_NUM_THREADS` and *then* creates the pool, which
only works if the child imports NumPy afterwards. `ProcessPoolExecutor` forks by default on
Linux, and a forked child imports nothing — it inherits the parent's already-initialised OpenBLAS
pool, sized by `_pin_threads_before_numpy` for the whole machine. Ten workers therefore ran
eighteen BLAS threads each: about 180 threads on 20 cores, load average 133, and each worker
progressing at roughly two cores while the rest of its threads spun. This is exactly the failure
`_pin_threads_before_numpy` documents, in a code path that had never been exercised until `e7`
started calling `map_trials`. Fixed by passing a spawn context. Verified rather than assumed:
parent `OMP=18`, children `OMP=2` with three OS threads each.

Spawning also avoids inheriting a CUDA context across `fork`, which is undefined for a parent
that has already trained on the GPU. It requires every experiment script to keep its
`if __name__ == "__main__"` guard; all eight have one, and that is now load-bearing.

**A long phase must report progress.** The cell printed nothing between starting and finishing,
so a twentyfold slowdown was invisible for two and a half hours. `map_trials` already had an
`on_done` hook that nobody used. Every completed model now logs the count, the elapsed time and
an ETA. The rule: no phase that can run for more than a few minutes may be silent.

**Also recorded: two of my own wrong turns.** I first blamed the probe fitting's scaling, on a
local measurement showing a 48x jump from `d=50` to `d=100`. On the Spark the same measurement is
linear (4.4 / 9.2 / 22.0 ms per step), so the pathology was the development machine and the
inference from it was wrong. And I estimated the campaign at 6–15 hours, then 3–4 days, from
extrapolations of a single unfinished cell. Both were avoidable by measuring one `diagnose` on the
target machine, which takes four minutes and settles it.

---

## D17 — 2026-08-12 — Both pre-registered estimands are reported, including where they disagree

**Status:** binding

Stage A's two pre-registered estimands disagree at the largest width, unanimously and in opposite
directions, on the `L4` cells:

| | network wins on `s95` | network wins on recovery AUC |
|---|---|---|
| `d=50` | 0/20 | 0/20 |
| `d=100` | 0/20 | 0/20 |
| `d=200` | 0/20 | **20/20** |

The recovery curves cross at about `s = 11` at `d=200`: the affine probe is better up to there and
the network is better beyond it. The interpolated `s95` gap at that width is 7.998 against 8.292,
not the full level the floored statistic suggests.

**Decided.** Report both, and report the reversal as the finding.

**Why this is not the manoeuvre D1 forbids.** D1 rules out "a new summary statistic chosen because
it favours the network". Recovery AUC is not new: `curve_auc` is the selection objective already
written in `configs/e7.json` and the analysis plan names it co-primary. Reporting only `s95` would
be equally selective in the other direction, and reporting only AUC would be exactly the
substitution D1 prohibits. D1 did not anticipate the two disagreeing; this entry resolves that
case rather than reinterpreting D1.

**What it means substantively, stated as a hypothesis and not a conclusion.** The affine probe is
better near the 95% crossing and the network is better in the tail. If that survives scrutiny it
is a sharper claim than either estimand alone, because it says *where* the nonlinearity earns
anything rather than whether it does.

**Process note.** The first summary of Stage A given to the principal investigator said the network
never beats the probe in any of the 120 trained models. That was true of `s95` and false of AUC,
and it came from reading the automated gate, which only computes `s95`. The gate reports its own
criteria; it does not notice when they are incomplete.

**Resolved, 2026-08-14, against the hypothesis above.** The decision was recorded and then not
carried out: until today the manuscript reported `s95` only and did not mention the AUC anywhere, so
"report both" existed in this log and nowhere a reader could see. It is now in Section 10.5 with a
figure showing both.

The hypothesis does not survive the other two widths. On the matched post-ReLU comparison of the
`L4` cells, the AUC difference is **+0.0101 at d=50, -0.0164 at d=100 and -0.0198 at d=200**, every
cell unanimous across all 20 seeds and no interval covering zero. So the network's edge away from the
crossing exists only at the smallest width and reverses at the two larger ones — the opposite of a
nonlinear advantage that strengthens with scale. "The network is better in the tail" was read off
`d=50`, which was the width the reversal happened to favour.

The claim in the paper is now the width-dependence itself, stated as something three widths cannot
explain, and nothing is built on it. Hedging the hypothesis when it was written is what made this
cheap to correct; had it gone into the abstract it would not have been.

---

## D19 — The frozen arm's `R_readout` is a trade-off, not a failed optimisation

**Date:** 2026-08-14.

E8's frozen arm ends at `R_readout` = 1.92 at `d=100` and 1.93 at `d=200`, stable across width. The
manuscript read that as an optimisation result: "gradient descent finds a good decoder only when it
may also move the code; held to a fixed random `Phi`, on this objective, it does not." That sentence
was written, committed, and is now withdrawn. It had no support.

**Why it was wrong.** `R_readout` scores cross-talk against the code-specific optimum, which
`pinv(Phi)` attains **by construction, for every code**. So "the code admits a near-optimal readout"
is a tautology and cannot separate a failure from a trade-off. The question the claim needed answered
is whether the arm's own objective prefers its `W_out` to that optimum, and nothing in the record
answered it — `final_mse` is `null` in the E8 run records.

**The check, and a wrong one worth recording.** The first attempt compared `R_readout` of `wout`
against `ls` and found 1.92 against 1.045, which looked decisive and was not: `ls` is fit on Boolean
states and reads the linear representation `Phi b`, so it shares neither the objective nor the input
with the trained readout. The second attempt (`scripts/frozen_readout_tradeoff.py`) scores both
readouts under the loss the arm trained on, in its own post-ReLU representation, at the training
sparsity, on a held-out seed.

**Result.** The frozen `W_out` beats `pinv(Phi)` on that loss by 4.39 / 4.70 / 4.50 at
`d = 50 / 100 / 200`, on 10 of 10 models at every width. The cross-talk is bought, not lost. The
untrained arm returns 1.00, as it must, since its readout *is* `pinv(Phi)` — that is the scoring
check.

**Decided.** State the divergence of the two objectives on a fixed code, and make the contrast about
co-adaptation: the joint arm holds `R_readout` = 1.02 *and* a task advantage of 9.71 at `d=200`, so
it does not trade one against the other, while the frozen arm can buy the second only with the
first. The joint arm's task advantage grows with width (6.59 → 8.40 → 9.71) where the frozen arm's is
flat. This is a stronger claim than the withdrawn one and it is the one the data supports.

**Pattern.** This is the fourth claim in this project withdrawn because the quantity measured was not
the quantity the sentence was about — after the `s95` gate read alone, the ReLU-robustness claim in
non-comparable units, and KD1's accuracy-selected threshold. In each case a ratio looked large and
the denominator was not what the prose assumed. The check that catches it is always the same: score
the alternative under the objective the thing was actually optimising.

---

## D20 — The trained-versus-untrained margin is reported as closing, not as growing

**Date:** 2026-08-14, on the evidence of stage C.

Stage C added `d=400` to test an extrapolation and returned a second result that costs us more than
the first one gains. Both are in Section 10.5 and the second is now a limitation.

**What the prediction bought.** The exact frontier over `d = 50, 100, 200` fits `d^0.4202` and
predicts 10.759 at `d=400`. Registered in `configs/e7_stageC.json` before the run; measured 10.510,
an error of -2.31%. The fit was redone on exact values only, so the test does not mix the exact
statistic with the subset one.

**What it cost.** The untrained control's frontier grows *faster* than the trained one:
exponents 0.4101 against 0.4977 fitted over the four widths, and per doubling the trained arm
decelerates (0.4228, 0.4176, 0.3873) while the untrained arm accelerates (0.4865, 0.4941, 0.5139).
The ratio falls at every step — 1.4195, 1.3582, 1.2882, 1.1800 — and the margin in sparsity levels
peaks at `d=200` and falls: 1.3260, 1.5866, 1.7974, 1.6029.

**Decided.** Report the closing margin as a finding and as the sharpest limitation on "training buys
the code", state the claim as holding at every width measured rather than as a property of training,
and decline the extrapolated crossing at `d ~ 3000` as outside the measured range. Name the two
candidate explanations — a real ceiling, or the fixed 50 000-step budget giving the widest models the
least optimisation per parameter — and say which experiment separates them.

**A correction of my own reassurance, recorded because the reasoning error is the reusable part.**
When the subset numbers first showed the ratio falling, I checked the absolute gap, found it rising
(1.33, 1.59, 1.80), and told the principal investigator the claim was safe because sparsity levels
and not ratios are the operational unit. That was true of three widths and false of four. Choosing
the reading that survives, and only then arguing that it is the meaningful one, is the same move as
picking a summary statistic after seeing the result — the thing D1 exists to forbid. The right
response was to report both readings, which is what the paper now does, and to notice that they
disagreed until they agreed in the direction I had argued away.

**Why the exact frontier was mandatory before writing any of this.** The subset shortcut's error is
asymmetric between the arms. It inflates the trained arm by a growing amount (+0.0025, +0.0118,
+0.0206, +0.0398) while the untrained arm's subset is exact at every width, because there the
lowest-leverage feature really is the `argmin` (rank 1-2, found in every model). So a margin computed
from subset values is biased *in favour of training* by an amount that grows with width — exactly the
axis of the claim. Correcting it moves the numbers against us, which is why it was not optional.

**Two limitations retired or corrected by this stage.** "Beyond `d=200` the frontier is again a subset
estimate" asserted the `d=400` sweep was a hundredfold more solver time and therefore unaffordable;
it ran in about two hours on ten CPU workers. That entry is corrected in place rather than deleted,
and `all_feature_frontier.py` now records its own duration so the next such claim is checkable rather
than remembered. The leverage limitation is sharpened with the `d=400` numbers: the shortcut finds the
true `argmin` in 2 of 10 trained models, with the true one at leverage rank 131 by median.

**Count.** This is the fifth limitation this project has had to retire or rewrite because later work
overtook it, and the fifth claim withdrawn or reframed after measurement. Both counts are worth
keeping visible: they are the argument for the run-record discipline, not an embarrassment.

---

## D21 — Target TMLR, and what the port cost us

**Decided** 2026-08-22. The manuscript is rewritten for TMLR (`paper/tmlr.tex`), and the
`elsarticle` version is retired to `paper/legacy/main_elsevier.tex` rather than deleted.

**Why TMLR.** Its two acceptance criteria are that the claims are supported by the evidence and that
some subset of readers would be interested. Neither asks for novelty of mechanism, which is the
weakest part of this work: the leverage identity is elementary and we claim no priority for it. What
we have instead is a measurement no one appears to have made — that every released sparse-autoencoder
dictionary we could find is worse conditioned for linear readout than a random code of its own shape,
with a matched randomly-initialised control locating the cause — and a record of how hard the claims
were pushed. Both are things TMLR's criteria reward and a novelty-first venue does not.

**What the port changed in substance, not only in format.** Three things, all of which made the paper
weaker before they made it more defensible.

1. *The dictionaries became the headline and the toy campaigns became the intervention.* Previously
   the campaigns led and the dictionaries were a late section. The campaigns cannot establish
   anything about a deployed model, so leading with them invited the reviewer's first objection.
   Leading with 40+ released decoders and demoting 400 trained networks to "the part where we control
   the variables" is the honest ordering.

2. *The decoder comparison was demoted to a subsection with a stated ceiling*
   (`sec:probes`). The probe's hypothesis class contains the network's own output layer, because
   `_design` appends only an intercept. So whenever the network scores higher, the gap bounds our
   fitting procedure and says nothing about what an affine map can express. This was identified after
   the campaigns were complete. The result is reported and nothing is built on it. The experiment
   that would settle it — refit the probe warm-started at the network's own `(W_out, theta)` — is
   named in the text and not run, which is a stated gap rather than a silent one.

3. *Stage B and the untrained arm entered the manuscript.* Neither was reported before. The untrained
   arm turned out to be the answer to the obvious objection against the whole comparison: under the
   identical protocol, an untrained code's own readout loses to the fitted probe by 1.0 to 2.6
   sparsity levels, unanimously, at all four widths. A protocol rigged for the network could not
   produce that. It is now the paragraph the decoder subsection turns on, and it was sitting unused
   in `results/` the whole time.

**Two readings withdrawn, and where.** The KD6 refit killed the claim that the two pre-registered
estimands disagree — they agree at all four widths — and the claim that a pre-ReLU probe beats the
network by a full sparsity level, which is now within a quarter of a level on trained codes and
reverses sign at `d=400`. Both are recorded in `paper/sections/app_withdrawn.tex`, an appendix of the
paper rather than a note in this repository. That appendix now carries all six withdrawals in one
place, with the cause of each, because a reader weighing the surviving claims is entitled to the rate
at which we found our own errors and the direction they pointed.

**Count.** Seven claims withdrawn or reframed after measurement, five limitations retired or
rewritten. Two of the seven were not defects in code but readings chosen after the fact for being the
stronger claim, which no additional test would have caught. That distinction is the reason
`app_withdrawn.tex` groups them by cause rather than listing them.

**Not decided here.** The repository name and public URL, whether `internal/` ships in the release,
and whether the 204 MB `.git` history is acceptable for a public push. All three are the principal
investigator's calls and all three are blockers for the anonymised archive TMLR wants attached.

---

## D22 — What the external adversarial review changed

**Decided** 2026-08-29, after two independent adversarial reviews of the submitted manuscript.

**The reviews were not equal and the difference is instructive.** One verified: it reproduced every
number from the released JSON, found seven factual errors in the manuscript, and reached
*accept with minor revision*. The other produced a mostly template-generated claim inventory,
verified nothing — its own log records that every generation and checking script failed to run in
its sandbox — and reached *reject*. The reject verdict rests on real conceptual objections about
what the metric can support, but a decision reached without executing a single check is not a
calibrated decision, and it labelled the notation table and the reproduction instructions as
overstated claims. We take the findings from both and the verdict from neither.

**Seven confirmed errors, all fixed.** Stage B's design (KD7), the probe-ceiling objective (KD8), a
stale pre-correction conclusion still asserting the tie count the results section had already
replaced, a depth claim false in two of five suites, a separability claim contradicted by our own
exact frontier at three of four widths, a readout ratio attributed to the wrong decoder, and an
abstract that called sixteen MLP down-projections sparse autoencoders. Every one is verified against
the data before being accepted; one reviewer finding — a systematic ceiling gap in the L2 arm — was
an artefact of our own defective objective and disappears once KD8 is fixed, so it is not adopted.

**One finding we could refute with a pointer, and should have pre-empted.** Both reviews raised
column normalisation, and one demonstrated that shrinking five columns of an i.i.d. code moves its
ratio further than any dictionary excess we report. The code has always applied `unit_cols` to every
dictionary and every control; the manuscript never said so. That is a documentation failure with the
same consequence as a methodological one, since a reader cannot distinguish them. Section 5 now
declares the gauge before anything is measured.

**One gift.** The repeated-basis counterexample — stack k copies of an orthonormal basis and every
leverage is exactly d/F, so the code attains the floor while being maximally unidentifiable — is
correct and is now in the paper. It was offered as a refutation; it is a sharper statement of our own
deflationary thesis than the i.i.d. control we had. Where it does bite is the wording: a ratio
invariant to every invertible left transform cannot be called "conditioning for linear readout", and
that phrase is gone.

**What we are not doing.** The reject verdict asks for activation-weighted endpoints, structured
nulls, multiple base-model seeds, and a nested selection experiment. Those would make a better paper
and they are the honest answer to "what would change your mind". They are also a different paper, and
the one we have states what it measures and what it does not. The dead-latent objection is the part
of that critique we can neither dismiss nor resolve from decoder matrices alone, so it is now a
limitation in its own right rather than a clause inside another one.

**On sequencing.** We waited for these reviews before posting the arXiv v2. Three of the seven errors
are in v1 as well, and two we would have introduced into v2. That decision is worth keeping.

---

## D23 — The second review round, and what it caught

**Decided** 2026-08-29. The round-two verification review returned *accept with minor revision*,
unchanged from round one, with six items. All six are applied.

**It was worth running, and the reason is the one we predicted.** Every item it found is in text
written while fixing round one. Two are substantive. The manuscript's conclusion had its integer
corrected and the inference that integer carried left standing, so it still asserted that the
nonlinearity does not out-decode a linear probe — which the results section had already replaced with
a three-way ordering. And a new macro cited the wrong arm: the SmolLM2 depth minimum came out at
1.0038 from the randomly-initialised dictionaries, whose labels also begin "SmolLM2", when the
trained suite's minimum is 1.0105 at layer 3. Both are the class of error the number checker cannot
see, because in both cases the value is faithfully equal to something — just not to what the sentence
says it is.

**Where the reviewer was right about the symptom and wrong about the cause, and why that mattered.**
It found that the ceiling analysis's reproduction check verified only because its scope excluded the
one campaign where it failed, by 0.7 of a sparsity level, and diagnosed a seed mismatch between the
audit and stage B. The seeds are right: the recomputed network score matches the campaign record in
all 220 models of all 22 cells. The actual cause was that the audit selected the probe over all three
threshold policies while the matched comparison it audits uses only the global one — KD9, and the
eighth instance of this project's recurring trap, committed for the third time inside the
probe-ceiling analysis. Chasing the reviewer's diagnosis rather than the symptom would have found
nothing; taking the symptom seriously and finding our own cause fixed it. With the filter corrected
the reproduction is exact in every cell, so the claim is widened rather than qualified.

**One correction to the reviewer, recorded because it cuts the other way.** Its round-one finding of
a systematic ceiling gap in the L2 arm was an artefact of our defective objective, and its round-two
recount of the corrected figure gives 52 of 110 where we had written 54. Both are in the change
manifest we sent, not in the paper, which says "essentially zero" and is right. We have not
propagated either number to the manuscript.

**The count is now nine defects and eight withdrawn readings.** Three rounds of adversarial review
have each found something the previous round's fixes introduced or left. That is the argument for
external review over more self-checking, and it is stated in the withdrawn-readings appendix in those
terms, because a reader deciding what to trust should know the shape of the process and not only its
output.

---

## D24 — What ships in the public release

**Decided** 2026-08-31, resolving the item D22 left open.

**`internal/` does not ship, and is purged from the history rather than merely deleted at the tip.**
It held the venue-selection playbook, the cover letter, the reviewer candidate list, the prepared
rebuttals, and the packets sent to the external reviewers. None of it is code, configuration or
results, so removing it costs no reproducibility — which is what its own README had claimed all
along while the directory sat tracked in every commit. The public repository had exactly one commit
at this point, so purging from history was free; it would not have been a month later.

**The weights and raw results stay in the repository.** They are 407 MB of the 424, and the
alternative — a light repository with the heavy artefacts as release assets — only saves anything if
they are purged from the history too, which would leave commits whose messages announce results they
no longer add. Having chosen to keep the history for what it documents, that incoherence costs more
than the convenience is worth. The largest single file is 58 MB, under GitHub's 100 MB limit, and the
README documents `git clone --filter=blob:none` for readers who want the code without the binaries.

**`superseded-submission/` is preserved intact**, by standing instruction. It
is a deliberate record rather than an
oversight, and it is left for the principal investigator to decide whether to keep it.

**The commit history is kept and rewritten rather than squashed.** The 101 messages document nine
defects and eight withdrawn readings, which is the same record the manuscript's own appendix rests
on; squashing would discard it. The rewrite strips the assistant co-author trailers and normalises
author and committer to a single identity.

**The README says the preprint is superseded.** arXiv:2605.01192 carries the earlier title, two
theorem statements that are too broad as written, and two readings later withdrawn. Pointing readers
at it without saying so would be the same failure the defect register exists to prevent, so the
README says it plainly and points at `docs/known_defects.md`. Posting the revision is the next task.
