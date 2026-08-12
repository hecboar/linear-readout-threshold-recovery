# Known defects, open

Defects found and not yet fixed, with what they invalidate. A defect leaves this file only when
the fix is committed and verified, not when it is understood.

---

## KD1 — The native-distribution detection threshold is selected by accuracy

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

**Deliberately not fixed yet.** Changing it mid-campaign would leave the finished cells and the
remaining ones measured under different criteria. It waits for Stage A to end.

**Precedent.** This is the second time this exact trap has been hit in this repository. The first
was per-feature Boolean thresholds tuned on raw accuracy, which produced "predict everything off";
it was measured, diagnosed and fixed by selecting on the reported objective. The lesson did not
propagate to the native path.

---

## KD2 — The headline comparison gave the probe a threshold-tuning advantage

**Found:** 2026-08-12, while checking whether the audit packet contained enough to answer its own
question about fairness. Not yet fixed in the reporting code.

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
