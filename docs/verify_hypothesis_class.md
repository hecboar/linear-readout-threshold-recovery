# Verification: does the affine probe class contain the network's decoder?

**Verdict: CONTAINED.** The trained network's decision rule is a strict special case of the
post-ReLU affine probe class. The claim as stated is correct: "network beats best fitted affine
probe" cannot be evidence that the nonlinearity, or anything else, adds *expressive* decoding
power over affine maps of the post-ReLU state. The gap is a property of the fitting-and-selection
pipeline, not of the hypothesis class.

Everything below was verified by reading the code, not the summary I was given. "Confirmed" means
read directly in source; "inferred" is marked as such.

## 1. Evidence: both sides consume the identical vector

**The representation.** `src/lrtr/splits.py:240-266` (`representations`): builds
`R = sum_j amplitude_j * W_in[:, support_j]` (i.e. `Phi b`), adds `rep_noise` if the split carries
it, then returns `np.maximum(R, 0.0)` when `post_relu=True`. The amplitudes, noise, and supports
are properties of the *split object*, so any two scorers handed the same split see byte-identical
inputs. No per-feature normalisation, no standardisation fitted on train, no clipping, no
rescaling of `W_in` anywhere on this path. Confirmed.

**The probe.** `src/lrtr/probes.py:69-70` (`_design`): appends a column of ones — the intercept,
nothing else. `probes.py:163-167` (`score_probe`): score is
`_design(representations(W_in, split, post_relu=(rep=="post"))) @ W`. So a post-representation
probe's score for feature `i` is `w_i . ReLU(Phi b) + c_i` — an affine function of exactly the
post-ReLU vector, with one intercept per feature. Fitting (`fit_probe`, `probes.py:149-160`;
`_fit_ridge` `:88-103`; `_fit_margin_family` `:106-146`) builds the same design matrix and never
transforms it; the intercept column is unpenalised in every family (`:99`, `:142`). Confirmed.

**The network.** `src/lrtr/probes.py:396-401` (`evaluate_network`): score is
`representations(W_in, split, post_relu=True) @ W_out.T`, thresholded at scalar `theta` inside
`_exact_recovery` (`:174-177`, decision `Z >= theta`). Identically in
`scripts/primary_comparison.py:84-100` (`network_curve`). Confirmed.

**Containment, exactly.** The network's rule for feature `i` is
`w_out_i . ReLU(Phi b) >= theta`. The probe class realises it by setting `W[:d, i] = w_out_i` and
intercept `W[d, i] = 0` under the same scalar threshold — or intercept `-theta` under threshold 0.
Either way the network's decision rule is one point in the probe class; the probe class is strictly
larger (arbitrary per-feature intercepts give it effective per-feature thresholds the network's
single scalar does not have). Confirmed by inspection of the two score functions.

**Same states, same metric, same thresholding machinery.** In `experiments/e7_scaled_toy.py`
(`diagnose`, lines 192-221) one `bundle` is built once and passed to both `probe_profile` and
`evaluate_network`, and everything funnels into the single `evaluate_scores`
(`probes.py:349-380`), which computes exact recovery per state via `_exact_recovery`. Confirmed.

**What would have broken it, and does not.** I looked for: standardisation of the design
(absent), a scaling of `W_in` differing between callers (absent — both take the same array),
`rep_noise`/amplitudes applied to one side only (impossible — they live on the split), a
nonlinearity in probe scoring (absent), different test sets (absent — same `bundle.test_by_s`).
Any one of these would have voided the containment.

## 2. One asymmetry that exists, and where it is fixed

In `experiments/e7_scaled_toy.py:219` the network is scored only at the fixed `theta = 0.5`,
while `probe_profile` sweeps threshold policies including validation-selected ones. That is a
*threshold-treatment* asymmetry (it favours the probes), not a hypothesis-class one; it is
exactly what `scripts/primary_comparison.py` removes — there the network's global threshold is
selected on validation by the same `select_thresholds` the probes use
(`primary_comparison.py:89-91`), the probes are restricted to the post-ReLU representation
(`:139-143`), and differences are paired per seed. In the matched comparison neither side has
anything the other lacks except what is intrinsic: the probes retain the per-feature intercept
(class strictly larger, favouring probes); the network retains nothing outside the class.

## 3. `fixed_readouts["wout"]` is NOT the network's decoder path — confirmed

`src/lrtr/toymodel.py`, `_readouts` (docstring: "All three are read out from `Phi b`, never from
`ReLU(Phi b)`"): `pinv`, `wout`, `ls` are all pre-ReLU readouts, and `wout` is the raw `W_out`
matrix applied to the *linear* representation. Moreover `evaluate_fixed_readout`
(`probes.py:404-419`) scores with `post_relu=False` **and gain-normalises** `G` to unit diagonal
first, so what is evaluated is not even raw `W_out`. The initial misreading is corrected: the
network's own decoder is the `network` block (`evaluate_network`), not
`fixed_readouts["wout"]`.

Sanity numbers from `results/e7_stageC/raw/cell_relu_L4_p0.01_d400.json`, first diagnosis, no
fitting: network `s95 = 14`; `fixed_readouts` wout/pinv/ls `s95 = 14/12/11`; best post-ReLU
probe `s95 = 13` (`ridge_post_global`); best pre-ReLU probe `s95 = 14` (`ridge_pre_global`);
18 probe configurations present. Consistent with everything above — and note the best *pre*-ReLU
probe ties the network here, which by itself argues against any "ReLU adds power" narrative in
this cell.

## 4. What the comparison does and does not measure

Four distinct things, which the wording of the results section must not conflate:

- **(a) Hypothesis class.** Affine maps of `ReLU(Phi b)` with per-feature intercept. The
  network's decoder is inside it. Therefore the class gap is zero *by construction* and no
  network-vs-probe result can speak to expressiveness of this class.
- **(b) Fitting objective.** The probes minimise convex surrogates against one-hot supports:
  penalised least squares (ridge), logistic loss, squared hinge — none of which is exact
  recovery. The network's weights come from a different objective altogether: end-to-end
  gradient training of `W_out ReLU(W_in x)` on the reconstruction task, never on recovery.
- **(c) Model selection.** Penalty chosen from a finite grid (`DEFAULT_RIDGE_GRID`,
  `probes.py:66`) by validation `curve_auc` (`select_probe`, `:271-316`); threshold chosen on
  validation with `theta_fixed` always a scored candidate (`select_thresholds`, `:180-253`).
- **(d) Reported metric.** Exact recovery of the full support on the locked test set, and its
  95% crossing `s95` (`evaluate_scores`, `:349-380`).

**What CAN be claimed** from "network beats best fitted affine probe" (matched, post-ReLU,
same threshold policy): supervised fitting with these families, this surrogate-loss set, this
regularisation grid, this sample budget and this selection procedure *fails to recover* a point
of the affine class as good — on metric (d) — as the point the network's own training found. It
is a statement about what fitting achieves, i.e. an optimisation/estimation/selection gap
(b)+(c) measured on (d). It can legitimately be phrased as: the network's representation contains
an affine decoding of quality X, demonstrated by the network itself, that the probe protocol does
not find.

**What CANNOT be claimed:** that the nonlinearity adds decoding power beyond affine maps of the
post-ReLU state; that the network implements anything outside the affine class over its own
representation; that the gap would survive a better fitter, a larger grid, more probe data, or a
fitting objective aligned with exact recovery. The gap is an upper bound on nothing and a lower
bound only on the suboptimality of this particular probe protocol.

The question the phrase "the nonlinearity adds decoding power" actually asks is answered by a
*different* comparison that already exists in the pipeline: post-ReLU probes versus pre-ReLU
probes on the same states (`primary_comparison.py` reports the pre-ReLU column as "a different
question"), and by the pre-ReLU `fixed_readouts` / Welch-floor apparatus. If a claim about the
ReLU is wanted, it must rest there — and the stage C sanity numbers above (pre-ReLU ridge tying
the network) suggest caution even then.

**Inferred, not read:** that the network's training loss is reconstruction under the native
input distribution — from `train_toy_model` (`toymodel.py:73`, "y_hat = W_out ReLU(W_in x)"
under L2/L4) and `sample_task_batch`; I did not audit the training loop line by line. Nothing in
the containment argument depends on it.
