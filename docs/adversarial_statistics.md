# Adversarial statistical review: the inferential machinery of Section 10.5

**Scope.** The bootstraps in `scripts/primary_comparison.py` and `scripts/make_numbers.py`, the
floored `s95` estimand, the trained-minus-untrained margin drop, the power-law fits and the
out-of-sample prediction, multiplicity, and censoring. Everything below was recomputed from the raw
and derived JSON in `results/`; nothing is taken from the prose. Date: 2026-08-14.

## Verdict

**Accept as stated:** the margin-drop interval (0.1940, [0.0903, 0.2818]) — it reproduces exactly
from `per_model[].kappa_min_full`, is seed-stable to the third decimal, and survives replacement of
the percentile bootstrap by a Welch-t analysis (z = 3.77); the per-width margin intervals and their
non-overlap ([1.748, 1.845] at d=200 vs [1.530, 1.695] at d=400); the s95 tie table, the AUC
intervals and their unanimity counts; the frontier means at all four widths; the absence of grid
censoring anywhere in E7 stages A/B/C and E8, which I scanned exhaustively and could not break.

**Require restatement:** (1) the word "paired" for the margin-drop bootstrap and the Figure 7b
intervals — the arms are independently trained groups of unequal size and nothing is paired;
(2) the sentence "we registered that number" about the prediction 10.759 — the registered number
was 10.79, and 10.759 was recomputed on different (exact-frontier) inputs after the measurement
existed; (3) the zero-width "95% intervals" [0.00, 0.00] in Table `tab:primary`; (4) the Figure 7a
caption "both codes' frontiers follow a power law", which asserts the claim E6 was demoted for; (5)
the -2.31% headline, which needs its prediction interval and the alternative-shape context printed
next to it.

**Reject until fixed:** the two hardcoded limitation numbers "0.87 [0.63, 1.10]" and
"2.03 [1.77, 2.27]" (`paper/main.tex:2530`), whose backing artifact
`results/e7/derived/relu_frontier_gap.json` does not exist in the tree; and the total silence of
the manuscript on Stage B — 100 trained models in 10 cells (`results/e7_stageB/raw/`) that bear
directly on the generality of the margin claim and appear nowhere.

None of the findings reverses a conclusion. Several weaken the confidence the prose places in the
two most quotable numbers.

---

## Findings, by severity

### 1. The "registered" prediction is not the number that was registered — WRONG AS WRITTEN, cheap fix

`paper/main.tex:2229-2231`: "extrapolates to \ScPredicted{} [10.759] at d=400. We registered that
number, then ran the width." The registered artifact is `configs/e7_stageC.json:2` (committed
`ab68f01`, 2026-08-14 12:05 UTC; the run started 12:56 UTC per
`results/e7_stageC/run_record.json`), and it registers **10.79**, fitted on the *subset* frontier
values 4.515, 6.031, 8.069. The reported 10.759 is a refit on the *exact* frontier values 4.487,
6.015, 8.035 (`scripts/make_numbers.py:476-482`), performed after the measurement existed. The
substitution moves the error from -2.60% to -2.31%, i.e. in the paper's favour. `docs/decision_log.md`
D20 discloses the redo ("The fit was redone on exact values only"); the manuscript does not.

The exact Stage-A frontier existed on 2026-08-13 (commit `d38f4b1`), a day before registration, so
the registration *could* have used exact values and did not. The defensible statement is: "we
registered 10.79 from subset values (error -2.60%); refitting on the exact values the margin
analysis uses gives 10.759 (error -2.31%)." One sentence. As written, the flagship out-of-sample
claim attributes post-hoc numbers to a pre-registration, which is the one thing a referee will not
forgive if they diff the config against the text.

### 2. Two limitation numbers have no backing artifact — REJECT until regenerated

`paper/main.tex:2528-2532` quotes the ReLU frontier gap "0.87 levels [0.63, 1.10]" and "2.03
[1.77, 2.27]" as literals, not macros. `scripts/relu_frontier_gap.py:143` writes
`results/e7/derived/relu_frontier_gap.json`; that file is absent (`results/e7/derived/` holds four
files, none of them this one). `scripts/check_manuscript_numbers.py` passes (288 macros, all used)
because it checks macros, not literals — these two numbers are invisible to the discipline the
paper advertises at `main.tex:~2361` ("every measured number in the text is a macro"). I could not
recompute or falsify them. Fix: run the script, commit the artifact, emit macros; or delete the
numbers.

### 3. Stage B is a file drawer — REQUIRE a sentence or a table

`results/e7_stageB/raw/` holds 10 cells, 100 models, at F=4d and p=0.02 — run, summarised
(`e7_stage_B.json`), and never mentioned in `paper/main.tex` (grep finds nothing) or in any
decision-log entry. It is not neutral filler: at F=4d the trained-minus-untrained frontier margin
(subset values, which are *biased toward training*) is 0.92 at d=100 and 0.93 at d=200, against
1.59 and 1.80 at F=2d. The margin the headline discusses is roughly halved, and flat rather than
growing, at double the load. That is consistent with the closing-margin story, so reporting it
costs nothing and omitting it looks like selection. Either report Stage B or state in the paper
that it exists and why it is excluded.

### 4. "Paired bootstrap" describes an unpaired procedure — RESTATE, one word

`paper/main.tex:2254` ("paired bootstrap over models gives a drop of...") and the Figure 7b caption
(`main.tex:2288-2289`, "paired bootstrap intervals over models"). The procedure
(`scripts/make_numbers.py:490-500`) resamples the trained and untrained arms independently — as it
must: the arms are separately trained models, 20 per arm at d≤200 and 10 at d=400, with no pairing
key. The interval is a valid independent two-sample percentile bootstrap and my recomputation
confirms it ([0.0903, 0.2818]; fresh streams give [0.090, 0.283]; Welch-t gives [0.086, 0.303],
z = 3.77, Satterthwaite df ≈ 17.9). So the *inference stands*; the *label* is false, and it matters
because Table `tab:primary` uses "paired" correctly for a genuinely paired design two pages
earlier — a referee who notices will re-derive everything, as I did. Note also the reported 0.1940
is the bootstrap-distribution mean; the plug-in estimate is 0.1945 (report the plug-in). The
non-overlap of the two widths' intervals (`main.tex:2255`) is true and, since non-overlap of 95%
intervals is conservative, it is not being asked to carry more than it can; the direct drop
interval is the operative inference and it is sound.

One honest caveat the text should carry: the 200-vs-400 pair was selected after the peak was seen.
Against the three adjacent-width comparisons available, a Bonferroni factor of 3 leaves
p ≈ 5×10⁻⁴ — the finding survives, so say so rather than hide the selection (D20 already confesses
the gap-vs-ratio version of this).

### 5. Zero-width intervals, and what "119 of 120 ties" is evidence of — RESTATE

Five of the six trained cells in Table `tab:primary` print "0.00 [0.00, 0.00]"
(`paper/generated/numbers.tex`, `results/e7/derived/primary_comparison.json`). A percentile
bootstrap over 20 identical zeros returns a zero-width interval *by construction* — it would do so
at n=3 — and printing it under the header "95% interval" invites reading instrument floor as
infinite precision. `s95` lives on an integer grid (`src/lrtr/threshold.py:169-187`, floored,
contiguous-prefix); a tie means |Δ| < 1 level *as resolved by this grid and these test budgets*,
not Δ = 0. The co-primary AUC proves the differences exist and are sub-level: unanimous 20/0/0 in
*both* directions depending on width (+0.0101 at d=50; -0.0164, -0.0198 at d=100, 200), no interval
covering zero. The section text is honest about this ("$s_{95}$ is a floored crossing point and
therefore blunt", `main.tex:2050-2051`); the contribution bullet at `main.tex:253-254` ("the two
tie on $s_{95}$ in 119 of 120") and the table are not. Fix: replace [0.00, 0.00] with "all 20
models tie (grid resolution 1 level)", and phrase the headline as "indistinguishable at the
resolution of one sparsity level, with sub-level differences in both directions under the AUC" —
that is equivalence-at-a-stated-resolution, which the data supports, not equivalence.

The bootstrap itself (`scripts/primary_comparison.py:101-111`) is correct where it has work to do:
paired within seed (genuinely — same model, network vs probe), resampling models, which is the
right independent unit; 20,000 reps is ample. With 19 zeros and one -1 it degenerates to a scaled
binomial ([-0.15, 0.00] at L4 d=200), which is crude but not wrong.

### 6. The power-law machinery on four points, against the paper's own E6 standard — CORRECT BUT OVERSTATED in the figure

E6 was demoted (D7, `main.tex:2296` ff.) because four points cannot discriminate growth shapes.
Section 10.5 then fits `d^0.4101` and `d^0.4977` on four points, quotes both exponents to four
decimals with no uncertainty, and draws their crossing at d≈3039 in Figure 7a. Recomputed:
bootstrap (over models) sampling CIs on the exponents are tiny — [0.408, 0.412] and [0.491, 0.504],
difference bounded away from zero (P(diff ≤ 0) < 5×10⁻⁵) — so *given the power-law form* the rate
difference is real. But the form is exactly what four points cannot give: on the trained arm
`a·d^{1/3}+c` fits *better* than the power law (R² 0.99994 vs 0.99944), and on the untrained arm
`a·√d+c` matches it (0.99984 vs 0.99979). Under these equally-good shapes the crossing moves from
3039 to 1494–2795. The crossing's bootstrap CI [2458, 3973] therefore understates the real
uncertainty by ignoring the dominant (model-form) component.

The *prose* handles this correctly ("Four widths spanning a factor of eight cannot establish where
two rates cross", `main.tex:2258-2259`; "local rate rather than a law", 2233). The Figure 7a
caption does not: "Both codes' frontiers follow a power law over the four widths"
(`main.tex:2285`) is precisely the sentence E6 was punished for, and the panel's x-axis extends to
1.35× the crossing, so most of its width is extrapolation. Fix: caption says "are consistent with a
power law, among other shapes four points cannot separate", and either drop the crossing line or
annotate the 1494–3039 range. The claims that do the argumentative work — the drop, and "the
untrained exponent is larger" — are either model-free or robust, so nothing downstream falls.

### 7. What -2.31% is and is not — CORRECT BUT NEEDS ITS CONTEXT PRINTED

Recomputed from three points (log-log OLS, 1 residual df): prediction 10.759, 95% *prediction
interval* [10.40, 11.13] (t₁ = 12.71 on in-sample scatter); measured 10.510 — inside. Alternative
two-parameter shapes fitted on the same three points predict: linear in d, 12.74 (-17.5% error);
d/log d, 12.16 (-13.6%); linear in log d, 9.73 (+8.1%); a+b√d, 10.99 (-4.3%); the three-point
exact interpolant quadratic in log d, 10.55 (-0.35%). So -2.31% is *not* what any smooth increasing
function would have given — it beats the obvious non-power alternatives cleanly — but it is also
not unique: any concave shape with local exponent ≈0.42 lands within a few percent, and a 3-param
interpolant beats it. Two things the text should add: (i) the measured mean is known to
SE ≈ 0.014 (sd 0.044, n=10), so the -0.249 shortfall is ≈18 standard errors — the test that
"validates" the rate simultaneously *rejects* an exact power law, which is the paper's own
drifting-exponent story (per-doubling exponents 0.423, 0.418, 0.387) told with inferential force;
say it as a strength. (ii) The prediction interval, so the reader can see the test had teeth
(±3.4%, and four of five alternative shapes fall outside it). With finding 1 fixed, this number
survives.

### 8. Multiplicity — MOSTLY IMMATERIAL, acknowledged only once

Counted: Stage A alone computes 36 bootstrap intervals in `primary_comparison.json` (9 cells × 2
estimands × 2 representations), the probe selection searches 18 configurations × 6 penalties per
model per representation (legitimately absorbed by validation selection, and leakage-tested by
`tests/test_probes.py`), plus 4 margins, 4 ratios, 1 drop, 2 exponents, 1 crossing, 1 prediction,
~30 E8 numbers, 6 task gains, and two derived analyses (`network_threshold_policies.json`,
`native_comparison.json`) mostly unreported. The manuscript acknowledges the absence of correction
exactly once, for E5's nine Mann-Whitney tests (`main.tex:1853`). Nothing comparable for E7/E8.

Where it does not matter: nearly every reported effect is a unanimous sign pattern (20/20 or 10/10,
sign-test p ≤ 2⁻¹⁰ each) with intervals far from zero — no plausible correction touches them. Where
it deserves one sentence: the drop (data-selected pair, finding 4) and the AUC direction-reversal
(three intervals; unanimous, survives trivially). Also unreported and mildly awkward:
`network_threshold_policies.json` shows the network's own s95 at L4 d=200 is *non-monotone* in
threshold-policy complexity (fixed 7, global 6, per-feature 5) — per-feature thresholds overfit
validation — which is worth a footnote since the primary comparison's "global" choice is doing
quiet work there.

### 9. Small accuracy items — one line each

- `main.tex:2231` and Figure 6 caption: E8 figure plots *medians* (`scripts/make_figures.py:365`),
  the E8 table quotes *means* (`make_numbers.py:445-450`); with n=10 they differ little, but the
  caption and table should name their own summary.
- E8 task gains (`frozen_readout_tradeoff.json`) are ratios of medians with no interval; the 10/10
  unanimity carries the claim, but the 4-decimal means in the E8 table have no uncertainty at n=10.
- Percentile bootstrap at n=10 per arm runs slightly narrow (vs Welch-t [0.086, 0.303]); harmless
  here because the conclusion is identical, but a BCa or t-interval would be the defensible default
  at this n.

---

## What I could not break

- **The drop.** 0.1940 [0.0903, 0.2818] reproduces exactly from the two frontier JSONs with the
  code's own seed; five other seeds move the bounds by < 0.002; fresh independent streams, Welch-t
  and Satterthwaite df all agree; the plug-in drop is 0.1945; the only outlier in the d=400 data
  (untrained model at 8.558) *works against* the finding — removing it enlarges the drop. The margin
  values 1.3260, 1.5866, 1.7974, 1.6029 and ratios 1.4195, 1.3582, 1.2882, 1.1800 all reproduce.
- **Censoring.** I scanned every `s95` (network, all probe configurations, fixed readouts, native)
  in every raw cell of E7 stages A, B, C and E8 against `eval_s_max` (10/14/20/28): zero values at
  or above ceiling. E6 (s95 = 1, 2, 5, 8 vs grid maxima 5, 7, 9, 14), E2 and E5 are likewise
  uncensored. The one genuinely censored quantity — the sampled post-ReLU frontier at d ∈ {100, 200}
  — the manuscript itself declines to report (`main.tex:2530-2532`), which is the correct handling;
  its d=50 numbers are finding 2's problem, not a censoring one.
- **The tie count.** 119/120 checks: 20+20+20+20+19+20, the single exception one L4 d=200 model
  where the probe wins by one level. The pairing in `primary_comparison.py` is genuine (same model,
  same splits, both sides), the resampled unit (models) is the independent one, and the AUC
  unanimity counts and intervals all reproduce from the derived JSON.
- **The pipeline.** `check_manuscript_numbers.py` passes: 288 macros generated, 288 used, disk file
  identical to regeneration. Every Section 10.5 number I recomputed independently (gaps, ratios,
  exponents 0.4101/0.4977, crossing 3039, prediction 10.759/-2.31%) matches the macro file to the
  printed precision.

The statistical core of the new work — training buys frontier at every measured width and the
margin falls from d=200 to d=400 by more than sampling noise — is correct and robustly supported.
What needs the week is wording and provenance (findings 1, 2, 4, 5) and one act of disclosure
(finding 3), not new computation.
