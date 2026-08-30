# Known defects

Every defect found in this work, what it invalidated, and what was done about it. A defect's status
changes only when the fix is committed and verified, not when it is understood, and no entry is
deleted once fixed: the record of what was wrong is worth more than a short file, because the pattern
across entries is the useful part. **Five of these nine are the same trap in five different code
paths**: a threshold or operating point chosen by maximising something, with no check that the
choice beats the baseline it replaced. That pattern is the most useful thing in this file.

All nine are fixed and all nine sets of affected measurements have been recomputed. KD7 and KD8 were
found by an external adversarial review of the submitted manuscript, after the first six had been
written up; KD9 by a second round of the same review, of the fixes for the first two — the most
useful fact in this file. The file was
called *Known defects, open* while some were not; it is kept as a register rather than a queue. The
manuscript-level consequences — which readings were withdrawn and what caused each — are in
`paper/sections/app_withdrawn.tex`, which is an appendix of the paper rather than a file only a
reader of this repository would find.

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

**Status: FIXED** 2026-08-12, in `paper/legacy/main_elsevier.tex` and `src/lrtr/affine_frontier.py`, with the
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

---

## KD6 — The "matched" threshold policy never evaluated the threshold it was matching against

**Status: FIXED** 2026-08-14 in `lrtr.probes.select_thresholds`, with
`tests/test_threshold_selection.py` asserting the post-condition and reproducing the old failure.
**The affected measurements have been recomputed**: `scripts/refresh_probe_blocks.py` refitted every
probe from the saved `W_in` on all three stages, `scripts/primary_comparison.py` recomputed the
matched comparison, and the per-cell records carry `kd6_fixed: true`.

Two readings did not survive the refit and are withdrawn in the manuscript rather than replaced:
that the two pre-registered estimands disagree (they agree at all four widths once the threshold is
selected correctly), and that a pre-ReLU probe beats the network by a full sparsity level (it is
within a quarter of a level on trained codes and the sign reverses at `d=400`). Both are recorded in
`paper/sections/app_withdrawn.tex`, items 1 and 2. The correction moved *both* sides of the
comparison, which is why it is trustworthy: it was not a change that could only help one arm.

**Found:** 2026-08-14 by an internal adversarial audit, verified per model at four widths.

**Where.** `select_thresholds`, the `global` and `per_feature` branches. The candidate set was
`np.quantile(Z_val, linspace(0.01, 0.999, n_grid))` and `best` was initialised to `-1.0`, with
`theta_fixed` as the initial `best_theta`. Since any attainable score exceeds `-1.0`, the first grid
point always displaced `theta_fixed`: **it was never scored.**

**Why a quantile grid is the wrong candidate set here.** The scores are bimodal — most coordinates
inactive and near zero, a few active and near one — so the quantiles crowd into the two modes and
sample the decision region between them sparsely or not at all. The routine then returns the best of
a bad set and reports it as a maximum.

**Measured.** On the `L4` cells, comparing the selected threshold against `theta = 0.5` on the
validation objective the selection claims to maximise:

| `d` | selected `theta` | validation objective, selected | at `theta=0.5` | `theta=0.5` wins |
|---|---|---|---|---|
| 50 | 0.4834 | 0.4766 | 0.4867 | 20/20 |
| 200 | 0.3613 | 0.4453 | 0.6280 | 20/20 |
| 400 | 0.5981 | 0.5056 | 0.7490 | 10/10 |

100% of models at every width. The cost in `s95` grows with width: none at `d=50`, one level at
`d=200`, **five levels at `d=400`** (8.0 against 13.0).

**What it invalidates.** Everything downstream of a selected threshold: the probes' `s95` and
recovery AUC under `global` and `per_feature`, the matched comparison of
`scripts/primary_comparison.py`, the claim that the affine probe's advantage grows with width, the
AUC reversal, and the `d=400` pre-ReLU gap. Note the direction is not simply "the network was
robbed": the same routine sets the probes' thresholds, so both sides were handicapped and the
recomputation could land anywhere.

**What it does not invalidate.** Anything computed without a selected threshold: `R_geom`, the exact
frontier `kappa_min` and everything built on it (the rate, the out-of-sample prediction, the closing
margin), E8's arms and its readout trade-off, the `L2`/`L4` separation, the untrained control's
proximity to the floor, and all of E1-E6. Those are functions of the code, not of a decision rule.

**Fix.** The candidate set is now the quantile grid, plus a uniform grid across the observed score
range, plus `theta_fixed`, which is scored first so that ties resolve toward the registered value.
The post-condition — the returned threshold is never worse than `theta_fixed` on validation — is
asserted by tests, one of which re-implements the old search and requires it to fail the case.

**Precedent, and this one is mine.** This defect was introduced by the fix for KD2. KD2 was an
asymmetry that favoured the probe; the correction was to give both decoders "the same
validation-selected threshold", and the routine chosen to do that was broken in a way that favoured
the probe again. That is the third time a threshold-selection criterion has produced a wrong
comparison in this repository, after per-feature accuracy tuning and after KD1's accuracy-selected
operating point. The lesson that did not propagate: a selection routine needs a post-condition
against the baseline it replaces, not just an objective to maximise.

---

## KD7 — A campaign run at a different feature load was compared as if it were the same design

**Status: FIXED** 2026-08-29, in the manuscript and, more importantly, in the derived files.

**Found:** 2026-08-29 by an external adversarial review of the submitted manuscript, verified against
`results/*/derived/probe_ceiling.json`, which records `F`.

**Where.** Stage B's `p = 0.01` cells are `F = 4d`; stage A is `F = 2d` everywhere. The manuscript
described them as exact repeats of stage A at `d in {100, 200}`, read their larger decoder
differences (+0.70, +1.00 against +0.35, +0.35) as a replication whose magnitude happened to differ,
and attributed that difference to the imprecision of a cell mean over ten seeds. It is a design
difference.

**What it invalidated.** The claim that doubling the training sparsity raises the frontier — called
"the cleanest small result in the campaign" — compared stage B's `p = 0.02` cell at `F = 400` against
its own `p = 0.01` cell at `F = 800`. At matched load the comparison reverses and the per-model
ranges are disjoint: `kappa_min = 7.6479 [7.5971, 7.7063]` at `p = 0.02` against stage A's
`8.0352 [7.9971, 8.0787]` at `p = 0.01`, both at `F = 400`, `d = 200`. The claim is withdrawn; the
manuscript now reports the null.

**What it did not invalidate.** The `p = 0.01` cells are a genuine feature-load axis, and the
loss–geometry separation and the sign structure of the decoder comparison hold at twice the load.
Stage B is reframed as that, which is a better experiment than the replication it was described as.

**Why no check could catch it.** Neither `primary_comparison.json` nor `all_feature_frontier.json`
recorded `F`. The number that distinguishes the two designs was absent from the files every table and
macro is generated from, so `check_manuscript_numbers.py` could verify every value and still not see
that two different designs were being compared. **This is the fix that matters**: both writers now
record `F`, the existing files are backfilled from the saved weights, and `tab_stageb` carries `F/d`
as its first column after `d`.

**Precedent.** This is the same class as KD1, KD2 and KD6 — a quantity compared against a baseline
without checking that the baseline is what it is claimed to be — but one level up, at the design
rather than in the code. Tests do not reach it. Recording the distinguishing variable does.

---

## KD8 — The probe-ceiling analysis maximised one objective and scored another

**Status: FIXED** 2026-08-29 in `scripts/probe_ceiling.py`.

**Found:** 2026-08-29, same external review.

**Where.** `select_probe` chooses a configuration by `trapezoid(ys, xs) / (max(xs) - min(xs))` over
the validation recovery curve. `probe_ceiling._curve` computed the plain mean over grid points, and
its docstring asserted that the mean *was* the campaign's selection objective. On a uniform integer
grid the two differ by `[(y_1 + y_n)/2 - mean] / (n - 1)`, the same order as the smallest gap the
script reports.

**What it changes.** Nothing is withdrawn. Recomputed under the correct functional the result
strengthens: the selection still misses `W_net` in 80 of 80 `L4` models at `d >= 100`, and the gaps
grow from 0.0163/0.0294/0.0638 to 0.0222/0.0368/0.0715 at `d = 100/200/400`. The untrained control is
still 0 of 90. Two readings move: `d = 50` is no longer indistinguishable from zero and is no longer
offered as a consistency check, and the `L2` arm — which under the wrong functional showed a spurious
systematic gap — now sits at zero, which is the better outcome.

**Precedent.** This is the seventh instance of the trap in KD1, KD2 and KD6, and it was committed
while writing the section that documents the other six. Knowing the failure mode is not the same as
being immune to it, which is the argument for external review rather than for more self-checking.


---

## KD9 — The ceiling analysis selected the probe over three threshold policies, not the matched one

**Status: FIXED** 2026-08-29 in `scripts/probe_ceiling.py`.

**Found:** 2026-08-29 by the second round of external adversarial review, from the symptom rather
than the cause: the reviewer noticed that `\CeilReproMaxDelta` verified as written only because its
scope excluded stage B, where the worst cell missed by 0.7 of a sparsity level, and diagnosed a seed
mismatch. The seeds were right — `net_test_at_fixed_theta` matches `campaign_network_s95` in all 220
models of all 22 cells — and the filter was wrong.

**Where.** `_cell` chose the probe as `max(val_criterion)` over every `_post_` key of
`probes_global`, which spans all three threshold policies. `primary_comparison.py` restricts to
`key.endswith("_global")`, the matched policy the comparison is defined by. In most cells the global
policy wins on validation anyway and the two agree; in stage B's untrained cell at `d = 200` it does
not, in 6 of 10 models.

**What it invalidated.** Nothing in the manuscript, because `probe_ceiling.py` is an audit of the
comparison rather than a source for it. But the audit was comparing against a probe the paper never
reports — a slightly stronger one, so the reported ceiling gaps were if anything conservative — and
its reproduction claim was scoped to the two campaigns where it happened to hold.

**What changed after the fix.** The reproduction is now exact across all 22 cells of all three
campaigns, so the claim in Section~\ref{sec:probes} is widened rather than qualified. The headline
counts are unchanged: 80 of 80 `L4` models at `d >= 100`, 0 of 90 untrained. Two stage B gaps moved
slightly (0.0362 to 0.0261, 0.0529 to 0.0520).

**Precedent.** The eighth instance of the trap in KD1, KD2, KD6 and KD8, and the third committed
inside the probe-ceiling analysis itself. The pattern is now specific enough to state as a rule: any
comparison against a "best" of a family must name the family, and the name must be checked against
the one the thing being audited uses.
