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
