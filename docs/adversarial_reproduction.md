# Adversarial reproduction audit

Referee audit of whether the artefact supports the manuscript. Performed 2026-08-14 on a clean
checkout of `phase0-reproducibility` at `934abdd`, Windows 11, Python 3.12.10, the pinned NumPy
2.4.4 / SciPy 1.17.1 / torch 2.13.0+cpu. Every command listed was actually run; every modified
tracked file was restored with `git checkout --` and the final `git status --short` is at the end.

## Verdict

**Not acceptable as it stands; acceptable after specific, small repairs.** The core of the claim is
real: the committed results regenerate the manuscript's macros, tables and figures deterministically
on this machine, the three derived-analysis scripts I was asked to test reproduce their committed
outputs to the byte (one modulo git's CRLF smudging), the smoke path runs end to end, and even the
two cheap campaigns I retrained from scratch (E1, E3) reproduce their committed raw files
byte-identically. That is far better than most artefacts.

But three findings each individually block the headline claim "every number in the manuscript
traces to results/":

1. a manuscript result whose backing file **was never committed** (the ReLU frontier gap),
2. the checker's guarantee is materially narrower than what it and the README say (tables invisible,
   undefined-macro check covers only 3 of ~30 macro prefixes, and hand-typed measured literals exist),
3. the provenance of E1–E6 — the six campaigns behind most of the paper's numbers — records a commit
   that contains **none of the code that ran**, with no dirty flag to say so.

All three are repairable in days: commit the missing file, extend the checker, macro-ify or footnote
the literals, and re-run E1–E6 once from a clean HEAD (I verified E1 and E3 already reproduce; E2,
E4, E5, E6 cost ~5.7 h total per the cost table).

---

## Findings, by severity

### Major

**M1. A quoted result has no committed artefact and is outside the macro system.**
`paper/legacy/main_elsevier.tex:2530` states the ReLU frontier gap: "At `d=50` that difference is `0.87` levels
`[0.63,1.10]` for `L4` and `2.03` `[1.77,2.27]` for an untrained code." These are typed literals, not
macros, and the file the producing script writes — `results/e7/derived/relu_frontier_gap.json`
(`scripts/relu_frontier_gap.py:139`) — does not exist and has never been committed
(`git log -- results/e7/derived/relu_frontier_gap.json` is empty), even though the script is wired
into `scripts/run_all.sh:108`. On a clean checkout, `run_all.sh` would *create* this file for the
first time. I ran the script; it is deterministic (fixed seeds, `relu_frontier_gap.py:116`) and the
`d=50` cells reproduce the manuscript exactly:
```
L4     d=50  gap=+0.87 [+0.63,+1.10]
random d=50  gap=+2.03 [+1.77,+2.27]
```
So the numbers are honest — but a referee cannot know that without an hour of CPU, and the checker
cannot see them at all. Related, same paragraph (`paper/legacy/main_elsevier.tex:2529`): "it overestimates the frontier —
by two to eight levels here". Measured looseness at `d=50` is +2.87 (L4), +1.93 (random), **+0.13
(L2)** — two of the three cells sit at or below the claimed lower edge. I stopped the run after the
three `d=50` cells (the only ones the manuscript quotes numerically); the `d=100/200` cells may
stretch the range upward, but as committed the range is unverifiable and partially contradicted.
*Fix: run the script to completion, commit the JSON, emit the four quoted values as macros, and
recompute or scope the "two to eight" range.*

**M2. The checker does not check what it says, demonstrated with two green false-positives.**
`scripts/check_manuscript_numbers.py` makes three claims (docstring, lines 4–11). Tested adversarially:

- *A falsified measured number passes.* The generated tables (`paper/tables/*.tex`) carry literal
  values and are `\input` by the manuscript (`paper/legacy/main_elsevier.tex:1760,1819,1939,2358,2382,2392`), but the
  checker reads only `paper/legacy/main_elsevier.tex` (`TEX_SOURCES`, line 26). I edited `tab_e1_floor.tex` to say
  `9.999` where the results say `1.999`, ran the checker: **exit 0, all green**. (Restored.) Only a
  full `run_all.sh` re-generation would catch this, and nothing diffs the tables the way
  `run_all.sh:114-119` diffs `results/e7/raw/`.
- *An undefined generated macro passes.* The undefined-use check (lines 65–67) filters by the prefix
  list `Eone…Esix, Dur, Env`, but `make_numbers.py` emits 288 macros across ~30 prefixes including
  `Eseven*`, `Eeight*`, `Prim*`, `Margin*`, `Sc*`, `Full*`, `Gone*`, `Subset*`, `Auc*`, `SAL*`,
  `SAR*`. I appended `\EeightBogusNumber{}` to `paper/legacy/main_elsevier.tex` and ran the checker: **exit 0**. (Restored.
  LaTeX would fail later, but the checker's stated guarantee — "every macro the manuscript uses is
  actually defined" — is false for the majority of the macro namespace, including everything E7/E8.)
- *Inherent, worth stating:* the checker verifies name-level traceability, not semantics. Swapping
  `\EoneTiedRatioMin` and `\EoneTiedRatioMax` in the prose keeps both "used" and stays green. No
  automated check can fully close this, but the README's "no number in the manuscript is typed by
  hand" should not imply it.

**M3. E1–E6 provenance is void as recorded.**
`docs/decision_log.md` D8 (2026-08-08) mandates commit + dirty status + dirty-file list. The six
main CPU campaigns ran **2026-08-07**, one day before D8, and their `run_record.json` files carry
only a bare `environment.git_commit` — no `dirty` flag, no file list. Worse, that commit is
`33016a6`, the repository's **initial commit of 2026-05-09, "Initial release of synthetic
illustration code"**, which contains none of `experiments/`, `src/lrtr/`, or `configs/`. The tree
that produced the paper's E1–E6 numbers is therefore unidentifiable from the records.
*Mitigation I performed:* re-ran the two cheap campaigns from HEAD
(`python experiments/e1_welch_floor.py --threads 2`, `python experiments/e3_linear_energy.py
--threads 2`); both regenerate `results/e1/raw/e1_rows.json` and `results/e3/raw/e3_rows.json`
**byte-identically** (git reported only `run_record.json` as modified; restored). So the archive
matches today's code for E1 and E3. E2 (67 min), E4 (170 min), E5 (76 min) and E6 remain unverified
by me for time reasons. *Fix: one clean re-run of E1–E6 from a tagged commit closes this
permanently; the code demonstrably reproduces at least two of them exactly.*

### Moderate

**Mo1. `refresh_native_blocks.py` is not the no-op the README promises, off its birth machine.**
README: "It is deterministic and idempotent, so on a clean checkout it must produce no diff."
On this machine it rewrote `results/e7/raw/cell_relu_L2_p0.01_d100.json` with 139 leaf diffs — all
last-ulp floating-point noise except `topk_hit_rate` 0.38653→0.38658 (a few states flipping rank
under BLAS reduction-order differences). A reader running `run_all.sh` on their own hardware will
trip the drift WARNING at `run_all.sh:116` and end with modified tracked files, for a benign reason
the warning text does not mention. Also, its docstring's "costs seconds" is false: it had processed
4 of 9 cells when I killed it at 10 minutes. (The kill left one file modified; verified
content-equivalent to ulp against my pre-run backup and restored via `git checkout`.) *Fix: compare
with a numeric tolerance instead of `git diff --quiet`, or say in README that byte-stability holds
only on the archiving machine — the README's own "Determinism caveat" paragraph already knows this
about figures.*

**Mo2. Measured literals typed in prose, invisible to the checker.** Found by regex sweep of
`paper/legacy/main_elsevier.tex` (`grep -nE '[0-9]+\.[0-9]+'` minus macro lines):
- `paper/legacy/main_elsevier.tex:254, 2417, 2606` — "119 of 120": hand-summed from the six `\Prim*PostTies` macros
  (20+20+20+20+19+20). If a re-run changed one cell the macros would update and the checker would
  stay green while "119" went stale. Three occurrences, one in the introduction.
- `paper/legacy/main_elsevier.tex:2530` — the four ReLU-gap numbers (see M1).
- `paper/legacy/main_elsevier.tex:2529` — "two to eight levels" (see M1).
- `paper/legacy/main_elsevier.tex:2512` — "about $0.24$\,s at `d=200`": a measured timing with no backing artefact
  anywhere in `results/`.
- `paper/legacy/main_elsevier.tex:264, 2536` — "$\approx1.01$" for the untrained code's `R_geom`; a macro carrying this
  measurement exists (`\EeightRandomHundredRgeom` = 1.0100) but is not used here.
- `paper/legacy/main_elsevier.tex:901` — "sits $0.06\%$ above" is phrased as a hypothetical but is recognisably the
  measured E5/E7 `L4` ratio (1.0006); borderline, flagging for the author to decide.

**Mo3. Stale counts in the documentation.** README:51 and `run_all.sh:15` say "all 238 generated
numbers"; `make_numbers.py` writes **288** macros and the checker confirms 288 used. README says the
test suite runs in ~75 s; it is 350 tests in **179 s** on this 14-core machine (and a `run_all.sh`
non-smoke pass is ~6 h of E1–E6 retraining plus ~45+ min of derived analysis — "CPU-only" is true,
"minutes" describes only the derived step, and only partially: `primary_comparison.py` took 14.2 min,
`network_threshold_policies.py` 15.5 min, `frozen_readout_tradeoff.py` 56 s here).

**Mo4. Older run records carry corrupted dirty-file names.** The first entry of `dirty_files` lost
its leading character in `results/e0_smoke` (`xperiments/e0_benchmark.py`), `results/e5_weights`
(`esults/logs/e5_weights.log`) and `results/e7_stageB` (`esults/e7/raw/...`). The parsing bug is
fixed and regression-tested (`tests/test_runlog_provenance.py`, which documents exactly this
defect), but the corrupted names remain in the committed records with no annotation. Note also
`results/e0_smoke` records a run with the *experiment script itself* dirty — smoke-only, but it is
the one record where the dirty file is code rather than results.

### Minor

- `results/results.rar` (793 KB) is a tracked archive whose first entry is `_audit_original` —
  apparently duplicating the already-tracked `results/_audit_original/`. Unexplained by the README's
  layout section; no run record accounts for it.
- `results/g1/g1_reanalysis.json` has no run record (`reanalyse_g1.py` writes none); the E7/E8
  `derived/` files likewise have none. For deterministic re-analyses of recorded runs this is
  defensible, but D8 says "every run".
- The `unused` check reads only `paper/legacy/main_elsevier.tex` (plus a `paper/sections/` that does not exist), so a
  macro used solely inside a generated table would be falsely flagged unused. None currently is.
- `check_manuscript_numbers.py:44-45` builds `d_disk` with a spurious `and` expression; harmless
  (both operands truthy on the failure path) but it is not doing what it looks like it does.

---

## Test-suite assessment (remit 3)

The suite is 350 tests, all passing here (179 s). It is **alive where it looks**: mutating the Welch
floor formula (`src/lrtr/codes.py:35`, `(F-1)` → `F`) was caught by **9 tests** across
`test_theory.py` and `test_theory_g124.py`. The KD3 lesson was genuinely absorbed for the theory:
`tests/test_arrow_counterexamples.py` pins the exact counterexamples (infeasible LP → κ=∞, the
`h ≤ 1/2` cap, duplicate columns), and `test_probes.py` structurally poisons test splits against the
KD1/KD2 threshold-leak trap. `test_batched_training.py` protects the batched-equals-individual claim
in float64 as the README says.

**Where it is blind, and this is the KD3 pattern again at larger scale:**

- **The L2/L4 distinction — the paper's central experimental contrast — has no test.** I set
  `exponent = 2` unconditionally in both training paths (`src/lrtr/toymodel.py:88,177`), so "L4"
  silently trains L2. **All 350 tests pass** (verified, then reverted). The batched-vs-individual
  test parametrises over both losses but only checks batched≡single, which my mutation preserves.
  Every E7/E8 claim about what the L4 objective does differently rests on code no test
  distinguishes from its negation. One assertion — train tiny L2 and L4 models and assert their
  `R_geom` separate, or simply that the two losses produce different weights — would close it.
- **Nothing under `scripts/` is tested.** No test imports the derived-analysis scripts; the matched
  threshold policy of `primary_comparison.py`, the policy sweep of `network_threshold_policies.py`,
  the tradeoff arithmetic of `frozen_readout_tradeoff.py` — the analyses KD1/KD2/KD4 were about —
  are protected only by determinism and the macro chain, not by any test of their logic.
- **No test reads `results/`**, so no committed empirical claim (frontier growth with width, the
  119/120 tie rate, E8's gain factors) is guarded against archive/code drift except through
  `check_manuscript_numbers.py`, whose gaps are M2.
- A decoder mutation (`threshold_decode` `>=` → `>`, `src/lrtr/threshold.py:32`) also survived all
  350 tests, but ties at exactly θ have measure zero for continuous scores, so I score that as
  benign rather than blindness.

---

## What I verified and found sound

- `python scripts/check_manuscript_numbers.py` → exit 0; 288 macros generated, 288 used; the
  byte-compare against `build().render()` genuinely regenerates from `results/` on every call, so
  any edit to a raw result file *is* caught unless `numbers.tex` is regenerated in the same breath.
- `python scripts/make_numbers.py` and `python scripts/make_tables.py` → **byte-identical** to the
  committed `paper/generated/numbers.tex` and all six `paper/tables/*.tex` (verified with `cmp`
  against pre-run copies).
- `python scripts/make_figures.py` → all seven committed PDFs reproduce **content-identically**;
  the only diff is the embedded `/CreationDate`. Consistent with the README's determinism caveat.
- `python scripts/frozen_readout_tradeoff.py` (56 s) → **byte-identical** to
  `results/e8/derived/frozen_readout_tradeoff.json`.
- `python scripts/primary_comparison.py` (14.2 min) → **byte-identical** to
  `results/e7/derived/primary_comparison.json`.
- `python scripts/network_threshold_policies.py` (15.5 min) → identical to
  `results/e7/derived/network_threshold_policies.json` modulo git's `core.autocrlf=true` CRLF
  smudging (blob is LF, content equal line-for-line after `tr -d '\r'`).
- `bash scripts/run_all.sh --smoke` → completes end to end, exit 0, and correctly routes all output
  to `*_smoke` dirs and `results/logs/smoke/` — it touched nothing tracked. (Smoke dirs are
  untracked prior outputs; I backed them up and restored them afterwards anyway.)
- Re-running E1 and E3 from HEAD reproduces their committed raw JSONs **byte-identically**.
- `scripts/relu_frontier_gap.py` reproduces the manuscript's `d=50` gap numbers exactly (see M1 —
  the numbers are right; the artefact is missing).
- Ordering/nondeterminism review of the emit path: `make_numbers.py` and `make_tables.py` iterate
  literal dicts and sorted globs only; no set-ordering or filesystem-ordering hazards found. The one
  order dependency (`all_feature_frontier` before `refresh_native_blocks`) is documented and honoured
  in `run_all.sh:101-104`.
- Provenance, post-D8: every run record from 2026-08-08 onward carries commit, branch, dirty flag
  and dirty-file list; all recorded commits exist in this repository and are ancestors of HEAD; the
  dirty trees of `e7` (stage A), `e7_stageB` and `e8` list only results files (self-output during
  resumed campaigns), which is disclosed and benign. The E7 stage-C record (run 2026-08-14 on the
  Linux/aarch64 CUDA box at `ab68f01`, clean tree) is exemplary.
- The cost table `paper/tables/tab_cost.tex` matches the run records line by line (E1 0.6, E2 67.5,
  E3 0.2, E4 169.6, E5 76.1, E7-A 306.1, E7-B 193.5, E7-C 103.8, E8 77.0, E8-d200 123.2 min; total
  1147.5 checks). E6's 29.9 min is correctly taken from the per-width `timings_seconds` rather than
  the 0.3 s resume-skip record, and both `make_numbers.py:85-90` and the table caption disclose the
  resume caveat honestly.

## Commands to repeat the key checks

```bash
python scripts/check_manuscript_numbers.py                       # green baseline
sed -i 's/128 & 1.999/128 \& 9.999/' paper/tables/tab_e1_floor.tex
python scripts/check_manuscript_numbers.py                       # still green = M2a
git checkout -- paper/tables/tab_e1_floor.tex
printf '\\EeightBogusNumber{}\n' >> paper/legacy/main_elsevier.tex
python scripts/check_manuscript_numbers.py                       # still green = M2b
git checkout -- paper/legacy/main_elsevier.tex
git log --oneline -- results/e7/derived/relu_frontier_gap.json   # empty = M1
python -c "import json; d=json.load(open('results/e1/run_record.json')); print(d['environment']['git_commit'])"
git log -1 --format='%ci %s' 33016a6                             # M3
# mutation: sed 's/exponent = 2 if loss_kind == "L2" else 4/exponent = 2/' src/lrtr/toymodel.py
# then: python -m pytest -q tests/   -> 350 passed; revert with git checkout
```

## Final tree state

`git status --short` after all checks and restorations:

```
?? docs/adversarial_directional_bias.md
?? docs/adversarial_statistics.md
?? docs/adversarial_reproduction.md
```

The two other untracked `docs/adversarial_*.md` files appeared during this session from concurrent
audit sessions and are not mine; every tracked file I touched (`paper/legacy/main_elsevier.tex`,
`paper/tables/tab_e1_floor.tex`, `paper/figures/*.pdf`, `paper/generated/numbers.tex`,
`src/lrtr/toymodel.py`, `src/lrtr/threshold.py`, `src/lrtr/codes.py`,
`results/e1/run_record.json`, `results/e3/run_record.json`,
`results/e7/raw/cell_relu_L2_p0.01_d100.json`,
`results/e7/derived/network_threshold_policies.json`) was verified restored to the committed state,
and the derived files rewritten byte-identically by their own scripts required no restoration.
Untracked smoke outputs were backed up before `run_all.sh --smoke` and copied back afterwards. The
`relu_frontier_gap.json` my run would have created was never written (run stopped after the quoted
cells were verified).
