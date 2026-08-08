# Phase-0 claim and protocol audit

The number audit (`current_number_audit.md`) checks that the manuscript reports what the raw
records contain. It cannot detect an error present in *both* — a defect in the protocol rather
than in the reporting. That is what this document is for.

Every item was verified against the code and the raw records, not against the manuscript's own
description of itself.

**Scope:** E5, the trained-network campaign, since that is where the contested claims live.
E1–E4 verify theorems and are structurally simpler; E6 is being demoted (decision D7).

---

## Findings

### P1 — The "own decoder" comparison was not like with like — **CORRECTED**

**Severity: material.** Same category of defect as the mixed-readout comparison caught earlier,
one level deeper and therefore harder to see.

`random_code_baseline` returns `W_out = pinv(Phi)`. The random control has no trained decoder,
so its `wout` column *is* its `pinv` column — verified identical to machine precision on every
control seed (e.g. seed 0: 1.018611 vs 1.018611).

The manuscript said $L^4$ was "farther under the model's own decoder (1.0260 versus 1.0170)".
That compares a **trained decoder** against a **pseudoinverse**. The 1.0170 is not the control's
own decoder, because the control does not have one.

**Action taken:** the sentence now reports the trained decoder's ratio without pairing it
against the control, and states explicitly why no pairing is available. The conclusion —
comparable to a random code, not better — is unaffected, since it rests on the two readouts
that *do* admit a matched comparison.

**Carried forward:** in the scaled campaign the control must either be given a decoder trained
the same way, or the `wout` row must be omitted from control tables entirely. A column that
silently changes meaning between conditions is worse than a missing one.

### P2 — Probe fitting and evaluation states are drawn from the same stream

**Severity: minor for the current numbers, blocking for the new campaign.**

`diagnose_model` draws the least-squares fitting set and then the evaluation supports from the
same generator, sequentially. They are independent draws, so there is no systematic leakage,
but there is coincidental overlap:

| sparsity | distinct supports | fit states | eval trials | expected coincidences |
|---|---|---|---|---|
| s=3 | 161 700 | 4 096 | 400 | ~10.1 (≈2.5% of eval) |
| s=4 | 3 921 225 | 4 096 | 400 | ~0.4 |

At `s=3` about one evaluation state in forty was also a fitting state. The effect on the
reported numbers is small and favours the `ls` readout — that is, it works *against* the
paper's original hypothesis, not for it — but it is not a defensible design.

**Carried forward:** the scaled campaign uses three disjoint splits (100k fit / 20k validation /
10k locked test) with membership enforced, not merely probabilistically unlikely.

### P3 — The least-squares readout is fitted at one sparsity and evaluated at eight

`s_fit = median(sparsities) = 4`, while evaluation runs over `s = 1..8`. The `ls` readout is
therefore mismatched at every sparsity except one, which understates it away from `s=4` and
makes "best of three readouts" a weaker baseline than it appears.

**Carried forward:** the new design reports both a **global** affine interface fitted on a
mixture of sparsities and a **per-sparsity oracle**, the latter labelled explicitly as an upper
envelope rather than an implementable interface.

### P4 — The feature-coverage repair perturbs the fitting distribution

To guarantee every feature appears at least once, uncovered features are forced into arbitrary
fitting columns. Those columns then have sparsity `s_fit + 1`. At `F=100`, `n_fit=4096`,
`s_fit=4` coverage is essentially always achieved anyway, so the repair almost never fires —
but the mechanism is silent when it does.

**Carried forward:** log how many repairs fired per model; a nonzero count at larger `F` is a
signal that `n_fit` is too small, not something to paper over.

### P5 — State-level intervals are not seed-level evidence

Wilson intervals in `rows` are computed over 400 state-level trials **within one trained
model**. They describe uncertainty about that model, not about the population of trained
models. With 5 seeds, all nine Mann–Whitney tests return `p = 0.007937`, the smallest value
attainable at `n=5` per group — reported honestly in the current text, but it means the tests
carry no information beyond "the groups do not overlap".

**Carried forward:** the trained seed is the replication unit; inference is by paired bootstrap
over seeds. State-level intervals are retained as descriptive only.

### P6 — Evaluation is off the training distribution

Training uses `x = mask ⊙ U[−1,1]`; evaluation uses Boolean `1_S`. This is deliberate and
disclosed — Boolean states are the object of the theory — but it means E5 measures what the
representation *can* support, not what the network *does* on its own inputs. The two are not
the same claim.

**Carried forward:** six evaluation distributions, with probes refitted per distribution
(decision recorded in the plan, §5 J6). Before fixing them, the exact task target must be
inspected: "active" has to be defined coherently with a ReLU target, which is not obvious for
negative amplitudes.

### P7 — A single fixed threshold

`θ = 0.5` throughout, for the network and for every linear interface. It is the natural
midpoint for unit-gain Boolean states, but it is not tuned, and an affine decoder with a
validation-selected or feature-specific threshold could do better. The current comparison
therefore does not establish that the network beats the *best* affine rule; only that it beats
these rules at this threshold.

**Carried forward:** report both a fixed `θ = 0.5` where semantically appropriate and a
validation-calibrated threshold, including feature-specific variants.

### P8 — Nested supports induce within-trial correlation (E2/E6)

`recovery_trial_streaming` uses nested supports `S_1 ⊂ … ⊂ S_max`. Each marginal is exactly
uniform, so every point of `p_rec(s)` is unbiased; points within a trial are correlated by
construction. Already disclosed in the manuscript. No action — it affects the joint, not the
marginals, and the reported quantities are marginal.

---

## Checks that came back clean

- **The comparison is paired within a trial.** `diagnose_model` draws one support per trial and
  scores the network and every calibrated interface on that same state. This is a genuine
  strength of the existing design and is preserved.
- **No floor violation anywhere.** 25 runs × 3 readouts, zero violations, no attainment ratio
  below 1, no row below the proved uniform-support energy floor.
- **Stored `s95` matches the curves.** All 25 runs × 4 decoders recompute exactly.
- **Paired seeds by construction.** `L2` and `L4` at the same seed share initialisation and data
  stream exactly; verified in float64. Only the loss differs. This is the design the scaled
  campaign is built on.

---

## Items from the external review, checked rather than assumed

| Alleged defect | Status |
|---|---|
| Algorithm 1 forms the full `F×F` interface despite the `O(Fd²)` claim | **Was true; fixed.** `Lemma (mean-square cross-talk from d×d statistics)` added, `statistics="mean_sq"` path implemented and tested against the dense computation to `1e-10`. The max statistic is now separately declared as `O(F²d)`. |
| Algorithm 2 regenerates random codes instead of accepting a fixed code | **Was true; fixed.** Split into `fixed_code_separation_profile` (the diagnostic) and `random_code_scaling_experiment` (the ensemble study). E2/E6 numbers unchanged — same routine, renamed. |
| `p_train = 0.02`: `W_out` has `s95 = 4` on every seed | **Was false; corrected.** Measured `[3, 4, 4, 3, 4]`. The manuscript now says the best linear readout is ahead on 3 of 5 seeds, tied on the rest, and the figure is a generated macro. |
| The abstract compares the `L4` pseudoinverse ratio against the `L2` least-squares ratio | **Was true; fixed.** Now a within-readout comparison. |
| The public repository may not contain the claimed implementation | **Open.** See `repository_gap_audit.md`. |
