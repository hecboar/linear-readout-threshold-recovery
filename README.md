# Near-Floor Geometry Is Generic: Leverage Dispersion in Trained Overcomplete Codes

Reference implementation, experimental campaigns and manuscript sources for:

> **Near-Floor Geometry Is Generic: Leverage Dispersion in Trained Overcomplete Codes**
> H. Borobia, E. Seguí-Mas, G. Tormo-Carbó.

**On the preprint.** [arXiv:2605.01192](https://arxiv.org/abs/2605.01192) is an earlier version of
this work, under its earlier title. It is superseded by the manuscript in `paper/`, and a revised
arXiv version is in preparation. Two of its theorem statements are too broad as written and two of
its readings did not survive later measurement; `docs/known_defects.md` records every defect found
during this work and `paper/sections/app_withdrawn.tex` records what each one cost. Cite the
repository, or wait for the revision, rather than the current preprint.

**Getting just the code.** The repository commits the trained weights and raw results deliberately,
so a clone is large. To skip the binaries until you touch them:

```bash
git clone --filter=blob:none https://github.com/hecboar/linear-readout-threshold-recovery
```

The repository contains the **Interface Diagnostic**, a computable procedure that takes an
overcomplete code — hand-built, gradient-optimised, or read off a trained network — and
returns how close its gain-normalised linear readout sits to a Welch-type floor, the sparsity
range over which a threshold decoder recovers Boolean states exactly, and a witness separating
*analog reconstruction* from *support recovery*.

**Every number in the manuscript can be recomputed on a CPU.** Campaigns E1-E6 need no
accelerator and hide any that is present. E7 and E8 do need one to *train*, but every model they
produce is committed under `results/*/weights/`, and every figure
the paper draws from them is computed from those weights by the CPU-only analysis step of
`scripts/run_all.sh`. So retraining is opt-in (`--with-campaigns`) and is only necessary for a
reader who doubts the training itself rather than the analysis of it. See `SPARK_CAMPAIGN.md`.

---

## Quickstart

```bash
git clone https://github.com/hecboar/linear-readout-threshold-recovery.git
cd linear-readout-threshold-recovery

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m pytest -q tests/          # 370 tests, ~105 s
```

Run one campaign end to end in seconds to check the installation:

```bash
python experiments/e2_threshold_recovery.py --smoke
```

Reproduce everything — experiments, figures, tables, generated numbers and the PDF:

```bash
bash scripts/run_all.sh             # Windows: powershell -File scripts\run_all.ps1
bash scripts/run_all.sh --smoke     # same shape in minutes, to check the plumbing
```

That reproduces E1-E6, the CPU analysis of the committed E7 and E8 models, every figure and table,
all 409 generated numbers, and the PDF. Add `--with-campaigns` to retrain E7 and E8 as well; that
needs a GPU and takes days.

---

## The method in three lines

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from lrtr.codes import random_unit_code
from lrtr.diagnostic import (fixed_code_separation_profile, interface_floor_diagnostic,
                             random_code_scaling_experiment)

Phi = random_unit_code(d=128, F=2048, rng=np.random.default_rng(0))

# Algorithm 1. `statistics="mean_sq"` runs in O(F d^2) without forming the F x F interface;
# the default "full" also returns the maximum statistic, which costs O(F^2 d).
print(interface_floor_diagnostic(Phi, statistics="mean_sq")["ratio_mean_sq"])  # >= 1, Thm 4.1

# Algorithm 2 -- the diagnostic: this code, held fixed, with random supports.
print(fixed_code_separation_profile(Phi, sparsities=[1, 2, 3], trials=200,
                                    seed=0)["separation_witness"])

# Algorithm 3 -- NOT a diagnosis of any code: a fresh random code per trial, for scaling.
print(random_code_scaling_experiment(d=128, F=128**2, sparsities=[1, 2, 3],
                                     trials=50, master_seed=0)["s95"])
```

The distinction between the last two matters. `fixed_code_separation_profile` is what you run on
a code you have — a designed one, or the effective code of a trained network — and it is the
only one that says anything about that code. `random_code_scaling_experiment` redraws the code
every trial, so it characterises the `(d, F)` ensemble; it is what E2 and E6 use to study how
the separation scales with width, and it streams the code block by block so it reaches
`F = 2^20`. Table 3 of the manuscript maps every equation to the function that implements it.

---

## Reproducing each experiment

Each command writes `results/<id>/run_record.json` with the exact command, configuration,
seeds, environment, wall-clock duration and exit status, plus raw results under
`results/<id>/raw/`. Append `--smoke` to any of them for a fast reduced run.

| Campaign | Question | Command | Measured wall clock |
|---|---|---|---|
| E1 | Do calibrated interfaces respect the floor, and how close do random codes get? | `python experiments/e1_welch_floor.py` | see `results/e1/run_record.json` |
| E2 | Threshold recovery and the separation at `F = d²` | `python experiments/e2_threshold_recovery.py` | see `results/e2/run_record.json` |
| E3 | Average linear-readout energy under both sparse-state models | `python experiments/e3_linear_energy.py` | see `results/e3/run_record.json` |
| E4 | Do *learned* codes attain the floor? | `python experiments/e4_optimized_codes.py` | see `results/e4/run_record.json` |
| E5 | Does the separation appear inside a *trained* network? | `python experiments/e5_trained_toy.py` | see `results/e5/run_record.json` |
| E6 | How does the recovery threshold scale, up to `d = 1024`? | `python experiments/e6_threshold_scaling.py` | see `results/e6/run_record.json` |
| E7 | The same question as E5, across widths, feature loads, losses and training sparsities | `bash scripts/run_spark.sh` | accelerator campaign, see `SPARK_CAMPAIGN.md` |
| E8 | Which half of the training does the work: the code, or the readout? | `python experiments/e8_frozen_encoder.py --device cuda --widths 50 100 --seeds 10` | accelerator campaign |

Measured durations are reported in Table 8 of the manuscript and are read from the run records
— they are not estimates. E4 is by far the most expensive campaign (168 optimisation runs of
20 000 Adam steps each); E6 is the most memory-sensitive.

Useful flags, common to every campaign:

```
--config PATH     alternative JSON configuration (defaults to configs/<id>.json)
--out-dir PATH    alternative output directory
--threads N       BLAS/torch threads (default: cpu_count - 2)
--workers N       worker processes for trial-level parallelism
--smoke           reduced configuration, runs end to end in seconds
--device DEV      torch device for campaigns that train (`cpu` default, `cuda` for E7)
--resume          reuse per-cell checkpoints already in the output directory
```

E1-E6 are CPU-only by construction: they hide the GPU from the process so that a stray
accelerator cannot make a published run irreproducible on a CPU-only machine. Passing
`--device cuda` opts back in, and only E7 and E8 do.

### E7, E8 and the accelerator

E1-E6 run on a laptop. E7 and E8 do not. E7 exists because the single-width, five-seed evidence of
E5 is single-width and five-seed, and it runs in three stages: **A**, the main grid
of 180 models over three widths and two losses; **B**, 100 models re-scoped after stage A to test the
two claims that survived it, at four times the feature load and at double the training sparsity; and
**C**, 30 models at `d = 400`, to test whether the frontier's growth over the first three widths is
a rate rather than just an increase. E8 adds 90 models in three arms — both parts trained, the code
frozen with only the readout trained, and neither trained — to separate what training the code
contributes from what training the readout contributes. Seeds within a cell are trained as one
batched computation -- at these widths a single model leaves an accelerator idle -- and
`tests/test_batched_training.py` asserts in float64 that this reproduces individually-trained
models exactly, so batching is a scheduling decision and not a modelling one.

Read `SPARK_CAMPAIGN.md` before starting. Run `python scripts/check_env_gpu.py` first: it
verifies that the torch build matches the card's architecture and then retrains the same model
on CPU and GPU in float64 and compares the weights, because an accelerator that runs but
returns different numbers is the failure mode that would otherwise go unnoticed.

E7 and E8 persist every trained model to `results/*/weights/*.npz`, so any of their numbers can be
recomputed without retraining. That is not a convenience. Three of the paper's findings — the
matched decoder comparison, the exact all-feature frontier, and the readout trade-off in E8's frozen
arm — were computed well after the runs, from these files alone, and one of them overturned a claim
the first analysis had made. The seven scripts that do it are wired into `scripts/run_all.sh` in
dependency order. E5 now does the same for future runs; the `results/e5/` already
in the repository predates that and has no weights, and re-running it elsewhere is not advised
— see the determinism warning below.

### Long runs and resuming

E6 writes its results per width and per chunk of trials. Re-running the command after an
interruption skips the trials already present and continues; completed results are never
overwritten. E7 checkpoints per cell and continues with `--resume`; because each seed keeps its
own generator, raising the seed count later leaves the already-trained seeds bit-identical. The
other campaigns are short enough to simply re-run.

### Re-analysing the trained models without retraining

Several of the paper's numbers are computed from the committed E7/E8 weights rather than during the
campaigns. They run on a CPU in minutes, in this order — `all_feature_frontier` must precede
`refresh_native_blocks`, which stamps the former's output into the per-cell records:

```bash
python scripts/all_feature_frontier.py 8      # exact frontier over all F features, not a subset
python scripts/refresh_native_blocks.py       # native detection with KD1 fixed; idempotent
python scripts/primary_comparison.py          # network vs affine probe, matched input and threshold
python scripts/native_comparison.py           # the same on the network's own input distribution
python scripts/network_threshold_policies.py  # the network under the probes' threshold policies
python scripts/relu_frontier_gap.py           # frontier before vs after the ReLU
python scripts/frozen_readout_tradeoff.py     # is E8's frozen readout losing cross-talk, or buying it?
```

`refresh_native_blocks.py` rewrites tracked files under `results/e7/raw/`. It is deterministic and
idempotent, so on a clean checkout it must produce no diff; `run_all.sh` warns if it does, because a
diff there means the analysis code and the archive have drifted apart.

### Tables, figures and the numbers in the paper

```bash
python scripts/make_figures.py            # -> paper/figures/*.pdf
python scripts/make_tables.py             # -> paper/tables/*.tex
python scripts/make_numbers.py            # -> paper/generated/numbers.tex
python scripts/check_manuscript_numbers.py
```

No number in the manuscript is typed by hand. Every measured quantity is a LaTeX macro emitted
from the raw result files; `check_manuscript_numbers.py` regenerates them, compares against
what is on disk, checks that the manuscript uses only defined macros and that no macro is left
unused, and exits non-zero on any mismatch.

### Building the manuscript

```bash
cd paper
latexmk -pdf tmlr.tex
```

`tmlr.tex` is the live manuscript, built from `paper/sections/*.tex` with the TMLR style
(`paper/tmlr.sty`, bundled). It is anonymous by default: add the `accepted` option to
`\usepackage{tmlr}` and `\input{sections/backmatter_camera_ready}` for a non-anonymous build.

`paper/legacy/main_elsevier.tex` is the earlier `elsarticle` version. It is **out of date** — two of
its readings did not survive the threshold defect KD6 — and is kept only as a record; see
`paper/legacy/README.md`. It is not built and no checker verifies its numbers.

---

## Repository layout

```text
src/lrtr/               the method
  codes.py              codes, Welch floors, harmonic tight frames, coherence
  interface.py          calibration, cross-talk statistics, closed-form linear energies
  threshold.py          threshold decoder; dense, streaming and fixed-code recovery trials
  diagnostic.py         Algorithms 1-3 (fixed-code diagnostic vs random-code ensemble)
  optimize.py           E4: gradient-optimised codes
  toymodel.py           E5/E7: compressed-computation toy model, batched training, diagnosis
  probes.py             E7: cross-validated affine probes and native-distribution evaluation
  stats.py              Wilson intervals, Mann-Whitney, Cliff's delta, scaling fit
  runlog.py             run records and environment capture
experiments/            one script per campaign (E1-E6 CPU-only; E7 opt-in accelerator)
configs/                one JSON per campaign, with a `smoke` override block
results/                raw results, run records and logs (committed)
scripts/                figures, tables, generated numbers, verification, run_all
tests/                  370 tests, including numerical verification of the theorems,
                        counterexamples to two of them that were once stated too broadly,
                        and regression tests for each defect in docs/known_defects.md
paper/                  manuscript sources, figures, tables, highlights
  tmlr.tex              the live manuscript (TMLR, anonymous by default)
  sections/             one file per section; every measured number is a generated macro
  generated/numbers.tex 409 macros, produced from results/ and verified against it
  legacy/               the retired elsarticle version, kept as a record
docs/                   decision log, defect register and supporting analyses
superseded-submission/
                        the earlier version, kept unmodified as a record
synthetic_illustrations.py, figures/, data/
                        the illustration script and outputs of that earlier submission
```

---

## Reproducibility

**Versions.** Pinned in `requirements.txt`; the exact versions used for the reported results
are Python 3.12.10, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.11.0, PyTorch 2.13.0 (CPU build).
Each run record stores the versions actually used.

**Seeds.** Every campaign takes a master seed from its configuration file and derives
per-configuration and per-trial seeds deterministically (`numpy.random.SeedSequence` with
`spawn_key`, and `torch.manual_seed` for the optimisation and training campaigns). E6's
streaming code generator is a pure function of `(master, trial, block)`, so any single trial
can be reproduced without replaying the others.

| Campaign | Master seed |
|---|---|
| E1 | 42 |
| E2 | 43 |
| E3 | 44 |
| E4 | per-variant seeds `0 … n-1` |
| E5 | per-seed `0 … 4`, evaluation seeds offset by 10000 |
| E6 | 46 |

**Data.** None required. All data are synthetic and generated from the recorded seeds.

**Compute.** CPU only. Thread limits are exported before NumPy is imported (setting them
afterwards is silently ignored by the BLAS runtime), and worker processes inherit an explicit
per-worker limit. By default two cores are reserved for the system.

**Numerical precision.** `float64` for the tests and for E1–E3; `float32` for E4, E5 and the
streaming code generation in E6, where the matrices are large and the reported ratios are
resolved well above the rounding error. Cross-checks between the two are in
`tests/test_threshold.py`.

**Determinism caveat.** Results are reproducible up to BLAS reduction order, which can differ
across machines and thread counts. The tests use tolerances rather than bit equality, and
figures may differ in the last displayed digit on a different platform.

---

## Relationship to the earlier version

This repository supersedes an earlier version of the work, preserved unmodified in
`superseded-submission/` together with the illustration script
(`synthetic_illustrations.py`) and figures it shipped.

The present version is not an edit of it. The theory is unchanged and correct, but the
contribution is now stated as a method and the experiments were rewritten from scratch, because
an audit found that several numerical claims in the earlier appendix are not reproduced by its
own published code. Every number in the current manuscript is generated from saved results by
`scripts/make_numbers.py` and checked against the manuscript by
`scripts/check_manuscript_numbers.py`.

The analysis plan, including the predictions registered before the results were in, is in
`docs/decision_log.md`.

---

## Citation

See `CITATION.cff`. Please cite the paper rather than the repository alone.

## License

MIT. See `LICENSE`.
