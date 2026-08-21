# Independent check of the primary-comparison arithmetic

Scope: the probe-side aggregation and internal consistency of
`results/{e7,e7_stageB,e7_stageC}/derived/primary_comparison.json`, recomputed
directly from the `probes_global` blocks in `results/<campaign>/raw/cell_*.json`
(all records carry `kd6_fixed: true`), independent of
`scripts/primary_comparison.py`. The network side is not rescored here; anything
that needs the network's per-seed s95 is checked only for internal consistency.

## Verdict: partly

The probe-side aggregation is keyed correctly, the bootstrap summaries are
internally consistent, and every reported number I could recompute matches the
data on disk. Two things need attention before the results section is rewritten:

1. **The probe's "best configuration" is chosen on the test metric, not on
   validation** (`scripts/primary_comparison.py:143-144`). This is conservative
   for the headline (it can only flatter the probe), but it changes one L4
   number materially: under validation selection, stage A L4 `d=100` flips from
   **-0.15 to +0.35**. The reported sign there is an artifact of letting the
   probe pick its family on test data.
2. **The stage B derived file contains two rows labeled only `(L4, d=200)`**
   (p=0.01 → +0.90 and p=0.02 → +0.40) with no `p` field to tell them apart.
   Nothing is collapsed or double-counted, but the quoted stage B summary
   "+0.40 at d=200" is the p=0.02 cell; the p=0.01 cell says **+0.90**.
   Also, stage B's L4 `d=50` cell is p=0.02, not p=0.01.

## 1. Key filtering (correct)

The filter at `scripts/primary_comparison.py:142` is
`f"_{tag}_" in key and key.endswith("_" + POLICY)` with `tag="post"`,
`POLICY="global"`. The actual keys in every record's `probes_global` are the 18
strings `{ridge,logistic,svm}_{pre,post}_{fixed,global,per_feature}` (e.g.
`ridge_post_global`, `svm_pre_per_feature`). I verified, for every diagnosis in
all 22 raw cell files across the three campaigns, that the substring filter
selects exactly `{ridge,logistic,svm}_post_global` and that this set equals the
set selected by the structured fields `representation == "post"` and
`policy == "global"`. No false catches (`_per_feature` does not end in
`_global`; `_post_` appears only in post keys) and no misses. All 18 configs
present in every record.

## 2. Max over configurations: taken on TEST, not validation

`scripts/primary_comparison.py:143` takes `max(pg[key]["s95"] for key in ks)` —
the maximum of the three families' **test** s95 — and line 144 independently
takes the max of their test `recovery_auc`. Two issues:

- The stated policy ("a single validation-selected global threshold") is true of
  each configuration's *threshold*, but the *family* (ridge vs logistic vs svm)
  is selected on the test metric itself.
- The s95 max and the AUC max are independent argmaxes, so the "probe" can be a
  different model for the two metrics in the same seed.

The records contain `val_criterion` (validation curve-AUC) and
`s95_validation`, so a clean validation-side selection is possible. Recomputing
with the family chosen by max `val_criterion` per seed (network per-seed values
held fixed, which is exact since only the probe term changes):

| cell | reported s95 diff | val-selected s95 diff | probe gain from test-max |
|---|---|---|---|
| e7 L4 d=50 | +0.00 | +0.00 | 0.00 |
| e7 L4 d=100 | **-0.15** | **+0.35** | +0.50 |
| e7 L4 d=200 | +0.35 | +0.35 | 0.00 |
| e7_stageC L4 d=400 | +2.00 | +2.00 | 0.00 |
| e7 random d=200 | -2.80 | -2.30 | +0.50 |
| e7_stageB L4 p0.01 d=200 | +0.90 | +1.00 | +0.10 |
| e7_stageB random d=100 | -1.90 | -1.80 | +0.10 |
| e7_stageC random d=400 | -2.70 | -2.60 | +0.10 |

All other cells: zero gain. So the test-set max only ever **helps the probe**;
switching to validation selection makes every affected difference grow in the
network's favor. The headline L4 numbers are conservative as reported, except
that the one negative L4 entry (-0.15 at d=100) becomes +0.35 — at d=100 the
test-max picks whichever of two families hit s95=5 on test (val selection picks
s95=4 in 10 of 20 seeds). AUC diffs move by at most +0.0013 (e7 random d=200)
and by +0.0008 for e7 L4 d=50 (+0.0045 → +0.0053).

## 3. Bootstrap and count consistency (all pass)

For all 22 rows across the three files: `n_positive + n_zero + n_negative ==
n_seeds`, and the mean lies inside `[ci_low, ci_high]`. Equality-with-zero is
exact-float but s95 values are integers, so ties are well defined.

Stage C L4 d=400, mean +2.00 with 10/0/0: this does **not** imply every model
differs by exactly +2 — the bootstrap CI is [+1.5, +2.5], which would collapse
to [+2.0, +2.0] if all ten diffs were 2. The per-seed diffs vary; the mean is
2.0 by integer coincidence. This is plausible given the probe values on disk:
the probe's test-max s95 at d=400 is [13,13,13,12,13,13,12,12,13,13] (mean
12.7, spread 12-13), implying a network mean s95 of 14.7 with per-seed diffs
mixing 1s, 2s and 3s. Nothing suspicious.

Cross-checks of the specific figures quoted: e7 L4 means +0.00/-0.15/+0.35 with
0/20/0, 0/17/3, 6/14/0 — confirmed in the file, and -0.15 = -3/20 is consistent
with three losses of 1; +0.35 = 7/20 with 6 wins requires one win of +2 (fine,
diffs are integers). AUC diffs +0.00448, +0.02099, +0.03795, +0.06990 match the
quoted +0.0045/+0.0210/+0.0379/+0.0699.

## 4. Stage B (exists, not collapsed, but ambiguous)

`results/e7_stageB/raw/` holds ten cells: L2 at (p0.01,d100), (p0.01,d200),
(p0.02,d200), (p0.02,d50); L4 at (p0.01,d100), (p0.01,d200), (p0.02,d200),
(p0.02,d50); random at (p0.01,d100), (p0.01,d200). `discover_cells()`
(`scripts/primary_comparison.py:64-75`) iterates the files, so nothing is
collapsed: the derived file has ten rows, including two `(L2,200)` and two
`(L4,200)` rows in filename-sort order (p0.01 before p0.02). But the output row
(`scripts/primary_comparison.py:145`) records only `loss` and `d`, not `p` — the
file is unambiguous only by row order. The L4 stage B rows are:

- d=100 (p0.01): +0.70, 7/3/0 — matches the quote.
- d=200 (p0.01): **+0.90, 9/1/0** — not in the quote.
- d=200 (p0.02): +0.40, 4/6/0 — the quoted "+0.40 at d=200".
- d=50 (p0.02): +0.00, 0/10/0 — quoted as d=50, but it is the p=0.02 cell.

Probe-side recomputation for all ten stage B cells matches the file.

## 5. Other observations

- No mis-keyed or double-counted aggregation found. Every raw cell has a
  matching weights file (no silent skips), each derived row's `n_seeds` equals
  the raw record's diagnosis count (20 for e7, 10 for stages B and C), and the
  probe-side per-seed values recomputed from `probes_global` are consistent
  with every derived mean and count I could bound.
- The independent argmax for s95 and AUC (point 2) means "the best probe" is
  not one model. Harmless for the conclusion (both maxima flatter the probe)
  but worth a sentence in the methods text.
- The `random`-loss rows go the other way (probe beats network by 1-2.8 steps
  of s95) in every campaign; the L4 advantage is specific to the trained code.

## Not verified (requires rescoring)

The network side: per-seed network s95/AUC come from rescoring saved weights
inside the script (`network_curve`, `scripts/primary_comparison.py:84-98`),
including its own validation-selected global threshold. I checked only that the
implied network means (reported diff + recomputed probe mean) are sane, e.g.
stage C L4: 12.7 + 2.0 = 14.7. The bootstrap CIs themselves (20k resamples)
were checked for containment of the mean, not re-simulated.
