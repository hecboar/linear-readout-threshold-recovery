# Known defects

Every defect found in this work, what it invalidated, and what was done about it. A defect's status
changes only when the fix is committed and verified, not when it is understood, and no entry is
deleted once fixed: the record of what was wrong is worth more than a short file, because the pattern
across entries is the useful part. Twice here the same trap was hit in two different code paths.

All five entries are currently fixed. The file was called *Known defects, open* while some were not;
it is kept as a register rather than a queue.

---

## KD1 — The native-distribution detection threshold is selected by accuracy

**Status: FIXED** 2026-08-12 in `lrtr.distributional.native_distribution_profile`. It now selects
on the metric being reported (`f1_selected`), keeps the old accuracy-selected point visible beside
it rather than deleting it, and adds a **threshold-free** measure: per state, take the `k` largest
scores where `k` is that state's own number of active coordinates, and ask how often the active set
is recovered exactly. `scripts/native_comparison.py` recomputes it over all 180 saved models; no
retraining was needed.

The corrected numbers reverse the reading. On one `L4` model at `d=50`, F1 goes from network 0.197
against post-ReLU probe 0.182 (both artefacts of the accuracy criterion) to network **0.422**
against 0.398, and the threshold-free top-`k` gives network **0.851** against 0.840. The pre-ReLU
probe leads on both (0.764 and 0.923), which is the same pre/post split the Boolean path shows.

**Found:** 2026-08-12, while checking whether "the affine probe beats the network" survives on the
network's own input distribution.

**Where:** `lrtr.distributional.native_distribution_profile`, the inner `detection()` helper. It
sweeps a quantile grid and takes `argmax` of **accuracy** on the training split.

**Why it is wrong.** The base rate is `p`, about 1%. At that base rate the accuracy-optimal
threshold sits close to "predict everything off", so each decoder is evaluated at whatever
conservative operating point its own score distribution happens to produce. Measured on a trained
`L4` model at `d=50`, `F=100`:

| decoder | threshold | precision | recall | F1 |
|---|---|---|---|---|
| network output | 0.629 | 1.000 | 0.103 | 0.187 |
| affine probe, pre-ReLU | 0.118 | 0.704 | 0.752 | **0.727** |
| affine probe, post-ReLU | 0.426 | 1.000 | 0.093 | 0.170 |

The criterion is identical for all three, so there is no asymmetry of *method*. But the F1 gap
measures where each score distribution places its accuracy optimum, not how well each decodes.
Note that the two probes differ from each other by the same factor as probe-versus-network, which
is the tell.

**What it invalidates.** Any decoder comparison drawn from the `native.detection` block. The
analog half of the same profile is unaffected: it is exact via `tr(A^T A C)` and involves no
threshold.

**What it does not invalidate.** The Boolean audit, which is what the headline result rests on.
There the threshold is selected on the objective actually reported (`lrtr.probes.select_thresholds`
documents this trap and avoids it), the splits are enforced disjoint, and
`tests/test_probes.py` poisons the test set to check structurally that no selection sees it. The
`L4` `d=50` result — best affine probe `s95 = 4` against the network's `3`, on 20 of 20 seeds —
comes from that path.

**Fix.** Select the threshold on the metric being reported, as the Boolean path does. Cheap: this
profile is recomputed from saved weights with no retraining, so Stage A's cells can be re-analysed
rather than re-run.

**Deliberately not fixed yet** — the reasoning at the time, kept as the record and since overtaken.
Changing it mid-campaign would have left the finished cells and the remaining ones measured under
different criteria, so it waited for Stage A to end. Stage A ended, and
`scripts/refresh_native_blocks.py` recomputed the block for all 180 saved models at once; the
per-cell records in `results/e7/raw/` now carry `kd1_fixed: true`.

**Precedent.** This is the second time this exact trap has been hit in this repository. The first
was per-feature Boolean thresholds tuned on raw accuracy, which produced "predict everything off";
it was measured, diagnosed and fixed by selecting on the reported objective. The lesson did not
propagate to the native path.

---

## KD2 — The headline comparison gave the probe a threshold-tuning advantage

**Status: FIXED in the manuscript** 2026-08-13; **the campaign gate is labelled, not repaired.**
Section 10.5 leads with `tab:primary`, the paired matched comparison from
`scripts/primary_comparison.py`: same input, same single validation-selected global threshold on both
sides, bootstrap intervals within seed. `Figure 5(a,b)` shows it under both pre-registered estimands.
The stage report still prints the old `0%` line, because it is the number the campaign computed and
deleting it would rewrite history; it now carries an `UNMATCHED` tag, a paragraph naming the three
asymmetries, and a pointer here and to `primary_comparison.py`. A reader of `results/` cannot now
mistake it for a decoder comparison, which was the actual hazard.

**Found:** 2026-08-12, while checking whether the audit packet contained enough to answer its own
question about fairness.

**Where:** `experiments/e7_scaled_toy.py`, the stage report. `s95(model)` comes from
`evaluate_network(..., theta=cfg["theta"])` — a **fixed** 0.5, never tuned. `s95(best probe)` is the
maximum over all eighteen probe configurations, two thirds of which use a validation-selected
threshold (`global` or `per_feature`). The gate then reports "the network beats the best affine probe
on 0% of trained models", which compares an untuned decoder against the best of a tuned family.

**Corrected comparison.** Giving the network the same three policies, selected on the same validation
split (`scripts/network_threshold_policies.py`, output in
`results/e7/derived/network_threshold_policies.json`):

| cell | policy | network | best probe | winner |
|---|---|---|---|---|
| L4 d=50 | fixed | 3 | 2 | network |
| L4 d=50 | global | 3 | 4 | probe |
| L4 d=50 | per_feature | 3 | 3 | tie |
| L4 d=100 | fixed | 4 | 5 | probe |
| L4 d=100 | global | 5 | 5 | tie |
| L4 d=100 | per_feature | 5 | 5 | tie |
| L4 d=200 | fixed | 7 | 8 | probe |
| L4 d=200 | global | 6 | 7 | probe |
| L4 d=200 | per_feature | 5 | 7 | probe |

Network 1, tie 3, probe 5, against the gate's 0-for-120.

**What it changes.** The direction of the headline survives at `d=200`, where the probe wins under
all three policies. It does not survive at `d=50`, where the network wins the fixed-threshold
comparison, and at `d=100` the matched-policy result is a tie under both tuned policies. So the
finding is width-dependent: the probe's advantage grows with width. That is a sharper claim than the
flat one, and it is the one the data supports.

**Not a D1 violation.** D1 forbids introducing a new summary statistic because it favours the
network. `s95` is unchanged, the thresholds are selected by the same routine the probes use, and all
nine matched comparisons are reported including the five the probe wins. Equalising a treatment
asymmetry in both directions is not the manoeuvre D1 rules out; leaving it unequalised would have
been the mirror-image error.

**Also checked and clear.** Choosing the probe *configuration* by test `s95` rather than validation
is an optimistic bias in how the summary was computed. Quantified: selecting on validation and
reporting the test number gives the same medians (4, 5, 8), so it does not move the result.

**Fix.** Report the matched-policy matrix rather than a single gate percentage, and stop comparing an
untuned decoder against the best of a tuned family.

---

## KD3 — Two theorems are false as stated, and one proposition draws the wrong conclusion

**Status: FIXED** 2026-08-12, in `paper/main.tex` and `src/lrtr/affine_frontier.py`, with the
counterexamples added as regression tests in `tests/test_arrow_counterexamples.py`. Kept here
because the record of what was wrong is worth more than a clean file.

**Found:** 2026-08-12 by external adversarial audit, verified independently against the
implementation. Full record with counterexamples in `docs/external_audit_2026_08_12.md`.

`thm:arrow` and `cor:failthresh` are stated for all `h_i < 1` while their shared proof only
establishes them for `h_i <= 1/2`; explicit unit-norm counterexamples at `d=2, F=3` give
`kappa = inf` against finite bounds. `prop:oneway` concludes that `kappa_i = 1` leaves a feature
separable at `s = 1`, when `thm:frontier` requires `kappa_i > s` and the correct answer is 0.
`thm:codefloor` claims the row-wise optimum coincides with the algebraic pseudoinverse; it is the
gain-calibrated one, `D_h^{-1} Phi^+`.

**Not affected:** every measurement. `e3_bound_respected` applies the corollary at `h.min()` only,
and the largest `h_min` across all 180 Stage A models is 0.4885, inside the valid region. The tests
that reported no violations sampled only that region, which is why they passed — a test gap, not a
false test.

**Fix.** Add the `h_i <= 1/2` hypothesis, define `kappa_i = +infinity` when the LP is infeasible and
handle it in the frontier statement, correct `prop:oneway` to "separable at no positive sparsity",
delete the pseudoinverse clause, and add the counterexamples as regression tests so the gap that let
this through is closed.

---

## KD4 — The probe-versus-network headline compares different decoder inputs

**Status: FIXED** 2026-08-13. The primary comparison holds the decoder input fixed to the post-ReLU
state the network's output layer actually reads, and the pre-ReLU probe is reported separately as
what it is: a gap between *representations*, not between decoders. The manuscript says so in those
words, and the `theta = 0.5` scale complaint below is answered by matching a single
validation-selected global threshold on both sides rather than by asserting a common scale.

**Found:** 2026-08-12, same audit, verified per seed.

The network reads `ReLU(Phi b)`. Two thirds of the probe configurations read the pre-ReLU state
`Phi b`, which the ReLU has not truncated, and "best probe" maximises over both. Holding the decoder
input fixed to the post-ReLU state and matching threshold policies gives **network 2, tie 5,
probe 2** across the nine comparisons, against KD2's 1/3/5 and the gate's 0-for-120.

**What it changes.** The claim "an affine probe beats the network" is not supported once the input is
held fixed. A different claim is: the information is more affinely accessible before the ReLU than
after it. That is a statement about what the nonlinearity discards, and it needs its own framing
rather than being folded into a decoder comparison.

**Related.** `theta = 0.5` is not a common scale across ridge, logistic and squared-hinge outputs, so
E7's "same handicap" phrasing is wrong: `ridge_*_fixed` reaches `s95 = 0` at two widths, which is a
scale artefact. The matched policy should be the validation-selected global threshold, which adds one
calibration parameter to each decoder.


---

## KD5 — Presentation defects the audit named, now fixed

**Status: FIXED** 2026-08-12.

The untrained control's `s95(model)` printed as `0.0` in the stage report, and the audit read it as a
decoder that failed. There is no network in that arm, so the column is not a measurement; it now
prints `n/a`, and its `R_readout` prints "1 by construction" rather than `1.0000`, since the baseline
sets its own decoder to `pinv(Phi)`. The gate was already computed over `loss_kind != "random"`, so
no number was ever contaminated -- this was a reading hazard, not an arithmetic one.

The manuscript asserted "enforced-disjoint supports" without saying that it cannot hold at the
smallest sparsities: at `s = 1` only `F` supports exist, against thousands of requested states.
Section 10.5 now states what the sampler does -- test split served first, smallest sparsities capped
at what the space contains, remainder distributed proportionally, exhausted sparsities and rejected
draws recorded, an empty split raised rather than returned -- and gives the measured consequence,
that the `s = 1` test set holds 50 states at `F = 100` and that `s` in `{1, 2}` are flagged
exhausted, so their intervals are wider than the nominal budget implies.
