# Adversarial audit: does the directional bias persist in the surviving claims?

**Date:** 2026-08-14. **Scope:** the claims currently standing in `paper/legacy/main_elsevier.tex`, attacked for
selection-after-the-fact bias in the estimand, statistic, aggregation, subset, or framing. Nothing in
`results/` was modified; every number below was recomputed from the raw records or from saved weights
with read-only scripts.

## Verdict

**Yes, the bias persists, and in the paper's single most load-bearing empirical claim it is not a
nuance but a reversal.** The E7 decoder-comparison block — "tie in 119 of 120", "the probe's
advantage grows with width", the AUC sign-reversal narrative, and the d=400 continuation — rests on a
threshold-selection routine that assigns the network a "validation-selected" threshold which is
*worse on the validation objective used to select it* than the fixed θ=0.5 it replaces, by an amount
that grows with width. At the network's pre-registered fixed threshold, the network **beats the best
of all eighteen probe configurations in 10 of 10 models at d=400** (the campaign's own stage C gate
and report table already say so: `results/e7_stageC/run_record.json:107`,
`results/e7_stageC/stage_C_report.md`), and its recovery-AUC lead over the best post-ReLU probe
**grows monotonically across all four widths**. The published direction — a probe advantage that
grows with scale — is the artifact; the suppressed direction — a network advantage that emerges with
scale — is what the data at the natural operating point supports. This is the same failure mode as
KD1–KD5 (the quantity measured is not the quantity the sentence is about; the reading that favours
the current thesis is the one that survives), and this time it favours the post-pivot thesis of
affine parity exactly as the withdrawn claims favoured the pre-pivot thesis of nonlinear advantage.
I am confident in the mechanism and the numbers (every one traces to the campaign's own raw records
or to a one-model recomputation reported below); I am *not* claiming the reversal is the final
answer — the probes might also gain from a repaired threshold selection — but no version of the
matched comparison that fixes the defect can produce the published direction at d=400. The remaining
surviving claims (loss decides the interface, near-floor geometry is generic, the E8 arms, the
frontier prediction, the closing margin) are substantially sound, with the smaller directional
defects listed below.

---

## Findings, most severe first

### F1 (critical): the matched comparison's network threshold fails its own selection objective, and the E7 direction claims reverse without it

**The defect.** `select_thresholds` (`src/lrtr/probes.py:205–212`), policy `global`, maximises
validation mixture exact-recovery over 64 quantiles of the pooled score distribution. It initialises
`best = -1.0` (line 207), so the `theta_fixed` it receives is **never evaluated** — the function can
only return a grid quantile, even when θ=0.5 dominates every grid point. The grid spans quantiles
0.01–0.999 of *all* scores pooled; the fraction of active coordinates in the validation mixture falls
from 5.5% at d=50 to 1.8% at d=400 (sparsity grids 1..10 at F=100 up to 1..28 at F=800), so with
~1.57% grid spacing the number of grid points inside the active/inactive transition region falls to
roughly one at d=400. The result, measured on seed 0 of each L4 cell (scratchpad
`threshold_mechanism.py`, reproducible from saved weights):

| d | selected θ | val mixture recovery at selected θ | at θ=0.5 | test s95 at selected θ | at θ=0.5 |
|---|---|---|---|---|---|
| 50 | 0.482 | 0.4825 | 0.4874 | 3 | 3 |
| 100 | 0.388 | 0.4344 | 0.5467 | 5 | 4 |
| 200 | 0.361 | 0.4479 | 0.6291 | 6 | 7 |
| 400 | 0.601 | **0.5047** | **0.7507** | **8** | **14** |

The "one validation-selected global threshold" the matched comparison gives the network is dominated
by the registered fixed threshold *on the very criterion used to select it*, and the damage grows
with width — at d=400 it costs the network about 5.5 sparsity levels. This violates the repository's
own KD1 rule ("select the threshold on the metric being reported", `docs/known_defects.md`) twice
over: the selection objective (mixture recovery) is not the reported estimand (s95 / AUC), and the
selection is not even optimal for its own objective. The probes pass through the same routine but
their score scales place the quantile grid where they need it — at d=400 the probes lose nothing
(post-ReLU best: fixed 9, global 9), while the network loses six levels.

**What the data says at the network's registered operating point** (θ=0.5, `configs/e7.json`;
network values are the campaign's own raw records, `results/e7/raw/`, `results/e7_stageC/raw/`,
probe values are the best over configurations as recorded, i.e. every probe keeps its tuned
threshold — a treatment biased *against* the network):

| d | net minus best of ALL 18 probe configs, s95 (net wins) | net minus best post-ReLU probe, AUC | net minus best pre-ReLU probe, s95 |
|---|---|---|---|
| 50 | −1.00 (0/20) | +0.0184 | −1.00 (0/20) |
| 100 | −0.80 (0/20) | +0.0444 | −0.80 (0/20) |
| 200 | −0.70 (0/20) | +0.1108 | −0.70 (0/20) |
| 400 | **+1.90 (10/10)** | **+0.2291** | **+1.90 (10/10)** |

The network's deficit shrinks monotonically and flips to a unanimous win at d=400 — against probes
that keep all three of the advantages KD2 removed. Its AUC lead grows at every width. The full
policy-by-policy matrix over all 70 L4 models (recomputed from saved weights, scratchpad
`matched_policy_audit.py` / `matched_policy_audit.json`; the `global` column reproduces
`results/*/derived/primary_comparison.json` exactly) gives medians:

| d | net fixed | net global | net per-feat | probe fixed | probe global | probe per-feat |
|---|---|---|---|---|---|---|
| 50 | 3 | 3 | 3 | 2 | 3 | 3 |
| 100 | 4 | 5 | 5 | 4 | 5 | 4 |
| 200 | 7 | 6 | 5 | 7 | 6 | 6 |
| 400 | **13** | 8 | 6 | 9 | 9 | 7 |

Under fixed–fixed matching the trajectory is +1, 0, +0.3, +4.2 (network never loses); under the
published global–global it is 0, 0, −0.05, −1.00. **Every direction claim in the E7 block is a
function of which column was published**, and the published column is the one in which the
network's threshold is broken.

**What this invalidates or endangers:**
- "the probe's advantage grows with width" and "the matched decoder comparison continues in the
  direction the smaller widths established, more sharply" (`paper/legacy/main_elsevier.tex:2239–2243`);
- the AUC paragraph's story that the network's edge "exists only at the smallest width and reverses
  at the two larger ones — the opposite of a nonlinear advantage that strengthens with scale"
  (`paper/legacy/main_elsevier.tex:2049–2068`, and D17's 2026-08-14 resolution, which was reasoned on the same
  broken numbers): at fixed θ the network's AUC lead *is* a nonlinear advantage that strengthens
  with scale, +0.018 → +0.044 → +0.111 → +0.229;
- contribution 7 (`paper/legacy/main_elsevier.tex:251–258`), Discussion reading 2 (`paper/legacy/main_elsevier.tex:2415–2422`), and
  the Conclusion's "negative result" (`paper/legacy/main_elsevier.tex:2604–2607`);
- the d=400 pre-ReLU gap of −3.3 levels (`results/e7_stageC/derived/primary_comparison.json`): at
  fixed θ the network beats even the best pre-ReLU probe 10/10 at d=400. (The representation-level
  claim — pre-ReLU probe beats post-ReLU probe, 12 vs 9 — survives; the network-vs-pre-probe
  numbers do not.)

**What survives at d≤200:** approximate parity is real. Under fixed–fixed the network is +1 at d=50
and ties at 100/200; under global–global it ties. The registered outcome "a validation-selected
affine decoder matches the network" is defensible *at the three stage A widths*. It is the width
trend, and everything stage C was said to show about the decoder comparison, that reverses.

**The record already contained the contrary evidence and it was argued away.** The stage C report
table prints s95(model) 13.0 against s95(best probe) 11.0; the gate records the network beating the
best probe in 50% of trained models (10/10 L4); and `scripts/primary_comparison.py:46` dismisses it
("stage C's unmatched gate reverses sign against stage A's, and a gate is not evidence either way")
with a warning written for the stage A direction — "three asymmetries, all favouring the probe" is a
reason a network *loss* is uninformative, and a reason a network *win* is highly informative. D1's
symmetry clause ("if instead the network robustly beats the affine optimum, that is the stronger
story and is reported as such") required this to be chased, not filtered.

**What would settle it:** re-run the matched comparison with a threshold selection that (i) includes
θ_fixed in the candidate set (one-line fix at `src/lrtr/probes.py:207`), and (ii) selects on the
reported estimand, for both sides symmetrically; report the result whatever it says. One open
question cuts both ways: the probes get 300 SGD steps at every width (`configs/e7_stageC.json`), so
at F=800 the "best affine probe" may be undertrained — which would weaken both the published tie and
my reversal, and should be checked with a converged probe before any direction claim is reinstated.

### F2 (high): "tie in 119 of 120" — half the sample cannot resolve a difference

60 of the 119 ties are L2 models in which *both* decoders sit at s95 = 0 at every width (verified
per model from `results/e7/raw/cell_relu_L2_*.json`: network fixed s95 = 0 and best post-ReLU probe
s95 = 0 in all 60). A tie between two decoders that both fail at s=1 is not evidence that "the
network does not out-decode an affine probe"; it is the absence of resolution. The informative
sample is the 60 L4 models. The composition is disclosed nowhere the claim is made
(`paper/legacy/main_elsevier.tex:254`, `2045`, `2417`, `2606`). The headline should say 59 of 60, on the models
where the statistic can move at all — and after F1, not even that without a repaired threshold.

### F3 (moderate): the d=400 AUC is on disk, unreported, and disagrees with the published growth claim even inside the matched framework

Section 10.5 reports matched AUC at d=50/100/200 (+0.0101/−0.0164/−0.0198) and then, for d=400,
reports only s95 (−1.00) as continuing the trend "more sharply". The matched d=400 AUC exists in
`results/e7_stageC/derived/primary_comparison.json`: **−0.0185**, i.e. *smaller in magnitude than at
d=200*. Under the paper's own co-primary estimand — the one Section 10.5 itself calls the finer
instrument ("$s_{95}$ is a floored crossing point and therefore blunt") — the probe's matched
advantage stopped growing between d=200 and d=400. The growth claim uses the blunt statistic at the
one width where the fine one contradicts it. (Subsumed by F1, but it shows the selection pattern
operating independently of the code defect.)

### F4 (moderate): "some feature is not separable even at s=1" is false at three of four widths

`paper/legacy/main_elsevier.tex:2113–2115` (and contribution 8 at 263–264) glosses the L2 collapse as "the collision
frontier collapses to 1 — by Theorem 5 some feature is not separable even at s=1". Per-model
κ_min from `results/e7/derived/all_feature_frontier.json` and
`results/e7_stageC/all_feature_frontier.json`: at d=50, 19/20 L2 models have κ_min = 1.0 exactly
(not separable at s=1 — the gloss holds); at d=100 the minimum over models is 1.00008, at d=200
1.0077, at d=400 1.065 — **every model separable at s=1** and lost at s=2. The sentence is also
internally inconsistent (a frontier of 1 means separable *at* 1) and contradicts the paper's own
theory prediction, which says the L2 code loses separability *at s=2*
(`paper/legacy/main_elsevier.tex:1984–1990`) — a prediction the data confirms exactly. The dramatised version is
strictly more favourable to "the loss decides the interface" than the true one. The true one is
still strong; state it.

### F5 (moderate): "We registered that number" — the registered number was 10.79, not 10.759

`configs/e7_stageC.json` (committed `ab68f01`, before the run; run record confirms the tree was
clean at that commit) registers the extrapolation **10.79**, from the subset-frontier fit. The
manuscript (`paper/legacy/main_elsevier.tex:2228–2231`) reports exponent 0.4202 and prediction **10.759**, computed
in `scripts/make_numbers.py:477–483` from the *exact* frontier — a fit performed after the run
(the comment says "fit on the three widths that existed when d=400 was committed to", which is true
of the widths and not of the fit). Measured 10.510: error −2.31% against the post-hoc fit, −2.60%
against the registered number. The exact-value fit is the methodologically better one (D20's
reasoning is sound), but attaching the word "registered" to it claims a sharper out-of-sample test
than was made. Honest text: registered 10.79 from the subset fit (−2.60%); the cleaner exact-value
fit, computed afterwards, gives 10.759 (−2.31%).

### F6 (low–moderate): the Conclusion mislabels the L2 number and uses the larger of the two

`paper/legacy/main_elsevier.tex:2596–2597`: "The trained $L^4$ network's 1.0006 and the $L^2$ baseline's factor
**28.8135** are statements about leverage profiles." 28.8135 is `\EfiveLtwoRatioLsMean` — the ratio
under the *fitted least-squares readout*, which is R_geom × R_readout(ls); the leverage-profile
statement is the pseudoinverse ratio **19.0052**, which the abstract (line 93) correctly pairs with
1.0006. The paper's own E5 text says the LS number measures "how far from optimal that readout is"
(`paper/legacy/main_elsevier.tex:1894`). The Conclusion both mislabels the quantity and picks the larger number for
the contrast.

### F7 (low): the Conclusion states the frozen-arm frontier comparison as evidence after 10.5 concedes it is a consistency check

κ_min is a function of the code alone, so "a frozen random code with a fully trained readout keeps
an untrained code's frontier" is true by construction. Section 10.5 says so in terms
(`paper/legacy/main_elsevier.tex:2163–2167`: "a consistency check rather than a result — but it is the point"); the
Conclusion (`paper/legacy/main_elsevier.tex:2610–2612`) restates it bare, as the second positive finding, with the
by-construction caveat dropped. The non-tautological content of "training buys the code" is
joint-vs-untrained (the code moves) plus the R_readout/task-loss trade-off, which is honestly
reported; the three-arm frontier table adds no evidence for the κ half and the Conclusion should not
lean on it.

### F8 (low): "paired bootstrap over models" is not paired

`paper/legacy/main_elsevier.tex:2253–2255` describes the d=200→d=400 margin drop (0.1940 [0.0903, 0.2818]) as a
"paired bootstrap over models". The code (`scripts/make_numbers.py:494–500`) resamples the four
groups independently — necessarily, since the models at the two widths are different models. The
interval is fine as an unpaired two-group bootstrap; the label claims a pairing that does not exist.

---

## What I checked and could not break

- **Traceability.** `python scripts/check_manuscript_numbers.py` passes: all 288 generated macros
  trace to `results/`, and the manuscript uses all 288. Every number I recomputed from raw JSON
  (κ_min means per cell, margin gaps and ratios, subset overestimates, argmin leverage ranks, the
  primary-comparison diffs, the E8 task gains) matched its macro. There is no typed number that
  fails provenance; the defects above are all in *which* number was chosen, not whether it is real.
- **Claim 3 (the loss decides the interface).** R_geom: L4 1.0006/1.0003/1.0002/1.0001 vs L2
  18.56/24.98/29.87/32.31 vs random 1.019/1.010/1.005/1.003, at every width, per-model spread
  checked. Robust to any aggregation I tried; the only defect is F4's s=1 gloss. Sound.
- **Claim 4 (near-floor geometry is generic).** Untrained codes at R_geom ≈ 1.002–1.019 and within
  1.3–1.8 sparsity levels of the L4 frontier; "one to two levels" is accurate; the missing
  tight-frame control is disclosed as a limitation (`paper/legacy/main_elsevier.tex:2535–2539`). Sound, and stated
  *against* the paper's interest.
- **Claims 5–6 (E8, the trade-off).** `scripts/frozen_readout_tradeoff.py` scores both readouts
  under the arm's own training loss, in its own representation, on a held-out seed (777001, disjoint
  from all campaign seeds); the untrained arm returns exactly 1.00 as the scoring check; the
  frozen-arm gains 4.39/4.70/4.50 and joint 6.59/8.40/9.71 reproduce from
  `results/e8/derived/frozen_readout_tradeoff.json`. I looked for an aggregation that reverses the
  joint-grows/frozen-flat contrast (means vs medians of per-model ratios vs ratio of medians) and
  did not find one. Two caveats, neither fatal: the gains are ratios to *different* baselines
  (each arm's own code's pinv), so the cross-arm growth comparison is of relative not absolute
  quantities; and D19's reframing is honestly recorded. The E8 threshold artifact of F1 does not
  touch these numbers (no thresholds involved). Sound.
- **Claim 7 (the frontier rate).** The fit arithmetic reproduces (subset fit → 10.79, exact fit →
  0.4202 / 10.759; measured 10.510); the exponent-drift hedge ("a local rate rather than a law") is
  appropriate; F5 is about the registration wording only. Sound apart from F5.
- **Claim 8 (the closing margin).** Reported in both units, with the fall at d=400 exceeding the
  bootstrap noise, a figure showing both readings, the budget confound named, and the crossing
  declined. I checked whether it is *under*stated: Limitation 15's "stops growing" is slightly
  softer than "peaks and falls", but the falling numbers and interval are in the same sentence, and
  the d=400 subset-vs-exact asymmetry (which biased the margin *toward* training) was corrected
  before the claim was made. This is the best-handled result in the paper. Sound.
- **KD1–KD5.** Spot-checked as fixed: the stage C report prints `n/a` and "1 by construction" for
  the untrained arm; split exhaustion at s∈{1,2} is disclosed in 10.5; the arrow corollary is
  applied at h_min ≤ 1/2 everywhere (largest h_min 0.4885); the matched comparison replaced the
  gate in the manuscript. The five withdrawn claims are indeed withdrawn.
- **Abstract and highlights.** I expected caveat decay here and did not find it: the abstract makes
  no E7 decoder claim at all, correctly pairs 1.0006 with 19.0052 (both pinv), and sells the
  deflationary reading; the highlights are theory-only plus E4, whose "gradient-optimised codes
  reach the floor" refers to direct cross-talk optimisation, not the trained task models. Clean.
- **Seed counts and comparability.** d=400 has 10 models per arm vs 20 at d≤200; disclosed
  (`\ScSeeds`, `paper/legacy/main_elsevier.tex:2230–2232`, 2158 for E8's separate d=200 run). The mixed-n margin
  bootstrap is valid as computed. No hidden n-asymmetry found beyond what is stated.
- **E5 and the G1 reanalysis.** The "uncomfortable" reading (the attainment ratio measures the code,
  not the decoder; L2 has the *better* decoder for its own code) is reported at full strength,
  including the failed pre-registered prediction. Sound.

## How to reproduce the F1 numbers

All read-only; scripts are in this session's scratchpad and inline in the audit transcript:
network fixed-θ values are the campaign's own `network.s95` / `network.recovery_auc` fields in
`results/e7/raw/cell_relu_L4_*.json` and `results/e7_stageC/raw/cell_relu_L4_p0.01_d400.json`;
probe values are the recorded `probes_global` blocks; the matched global/per-feature network values
are recomputed from `results/*/weights/relu_L4_*.npz` with the exact `primary_comparison.py`
machinery (the global column reproduces `derived/primary_comparison.json` to the model). The
one-model mechanism check (selected θ vs 0.5 on the validation objective itself) needs only
`select_thresholds` and one rebuilt split bundle per width.
