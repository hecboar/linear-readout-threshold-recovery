#!/usr/bin/env bash
# Reproduce every experiment, table, figure and manuscript number from a clean checkout.
#
# Usage:
#   bash scripts/run_all.sh                  # everything except the accelerator campaigns
#   bash scripts/run_all.sh --smoke          # same shape, minutes, for checking the plumbing
#   bash scripts/run_all.sh --with-campaigns # also retrain E7 (stages A, B, C) and E8: needs a GPU
#                                            # and takes days
#
# Options for --with-campaigns:  --device DEV (default cuda)   --workers N (default 10)
#
# Why the split. The campaigns train and diagnose several hundred networks; that needs an accelerator
# and most of a week. Everything the manuscript actually quotes is then computed from the *committed*
# weights and run records by the derived-analysis step below, which is CPU-only and takes minutes.
# So a reader with a laptop can regenerate all 238 generated numbers and the PDF, and only a reader
# who doubts the training itself needs the GPU. That is the point of committing the weights.
set -euo pipefail

SMOKE=""
CAMPAIGNS=0
DEVICE="cuda"
WORKERS=10
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke"; shift ;;
    --with-campaigns) CAMPAIGNS=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Every explicit --out-dir must carry the smoke suffix that prepare() would have added on its
# own. Passing --smoke together with a bare --out-dir writes reduced-fidelity output over
# committed results, which is the one way a reproduction script can destroy the thing it exists
# to verify.
SFX=""
[ -n "$SMOKE" ] && SFX="_smoke"

cd "$(dirname "$0")/.."
# Logs go to results/logs/smoke/ under --smoke. The tracked logs in results/logs/ belong to the
# real runs, and tee-ing a reduced run over them loses the record of the run the paper reports.
LOGS="results/logs"
[ -n "$SMOKE" ] && LOGS="results/logs/smoke"
mkdir -p "$LOGS"

echo "== environment =="
python --version
python -c "import numpy,scipy,matplotlib,torch;print('numpy',numpy.__version__,'scipy',scipy.__version__,'matplotlib',matplotlib.__version__,'torch',torch.__version__)"

echo "== tests =="
python -m pytest -q tests/

echo "== experiments E1-E6 (CPU) =="
for e in e1_welch_floor e2_threshold_recovery e3_linear_energy \
         e4_optimized_codes e5_trained_toy e6_threshold_scaling; do
  echo "-- $e"
  python "experiments/$e.py" $SMOKE 2>&1 | tee "$LOGS/${e}.log"
done

echo "== E5 again, keeping weights, for the G1 re-analysis =="
# reanalyse_g1.py needs the trained codes themselves, which the default E5 run does not keep.
python experiments/e5_trained_toy.py $SMOKE --out-dir results/e5_weights${SFX} \
  2>&1 | tee "$LOGS/e5_weights.log"
# Skipped under --smoke: reanalyse_g1.py reads results/e5_weights/ and writes results/g1/, both
# committed, so running it during a plumbing check would rewrite real output from a reduced run.
[ -z "$SMOKE" ] && python scripts/reanalyse_g1.py 2>&1 | tee "$LOGS/g1.log"

if [ "$CAMPAIGNS" = "1" ]; then
  echo "== E0: device capacity benchmark =="
  python experiments/e0_benchmark.py --device "$DEVICE" 2>&1 | tee "$LOGS/e0.log"

  echo "== E7 stage A: the main grid, 180 models (hours to days) =="
  python -u experiments/e7_scaled_toy.py $SMOKE --stage A --device "$DEVICE" --resume \
      --workers "$WORKERS" 2>&1 | tee "$LOGS/e7_stageA.log"

  echo "== E7 stage B: feature load and training sparsity, 100 models =="
  python -u experiments/e7_scaled_toy.py $SMOKE --config configs/e7_stageB.json --stage B \
      --device "$DEVICE" --workers "$WORKERS" --out-dir results/e7_stageB${SFX} \
      2>&1 | tee "$LOGS/e7_stageB.log"

  echo "== E7 stage C: d=400, testing the frontier rate the first three widths imply =="
  python -u experiments/e7_scaled_toy.py $SMOKE --config configs/e7_stageC.json --stage C \
      --device "$DEVICE" --workers "$WORKERS" --out-dir results/e7_stageC${SFX} \
      2>&1 | tee "$LOGS/e7_stageC.log"

  echo "== E8: trained / frozen-code / untrained arms =="
  # Two invocations with two run records, because d=200 was added after the first run and its
  # provenance is its own. Merging them on disk would claim one launch that never happened.
  python -u experiments/e8_frozen_encoder.py $SMOKE --device "$DEVICE" --widths 50 100 \
      --seeds 10 --workers "$WORKERS" 2>&1 | tee "$LOGS/e8.log"
  python -u experiments/e8_frozen_encoder.py $SMOKE --device "$DEVICE" --widths 200 \
      --seeds 10 --workers "$WORKERS" --out-dir results/e8_d200${SFX} 2>&1 | tee "$LOGS/e8_d200.log"
else
  echo "== E0, E7 and E8 skipped: pass --with-campaigns to retrain them =="
  echo "   the analysis below runs from the committed weights and needs no accelerator"
fi

if [ -z "$SMOKE" ]; then
  echo "== derived analysis (CPU, from the committed E7/E8 artefacts) =="
  # Order matters once: all_feature_frontier writes the exact frontier that refresh_native_blocks
  # stamps into the per-cell records.
  python scripts/all_feature_frontier.py 8       2>&1 | tee "$LOGS/all_feature_frontier.log"
  python scripts/refresh_native_blocks.py        2>&1 | tee "$LOGS/refresh_native.log"
  python scripts/primary_comparison.py           2>&1 | tee "$LOGS/primary_comparison.log"
  python scripts/native_comparison.py            2>&1 | tee "$LOGS/native_comparison.log"
  python scripts/network_threshold_policies.py   2>&1 | tee "$LOGS/threshold_policies.log"
  python scripts/relu_frontier_gap.py            2>&1 | tee "$LOGS/relu_frontier_gap.log"
  python scripts/frozen_readout_tradeoff.py      2>&1 | tee "$LOGS/frozen_tradeoff.log"

  # refresh_native_blocks rewrites tracked files in results/e7/raw/. On a clean checkout it is
  # deterministic and must produce no diff; if `git status` reports those files as modified after
  # this step, either the analysis code or the archive has drifted, and that is worth knowing.
  if git rev-parse --git-dir >/dev/null 2>&1; then
    if ! git diff --quiet -- results/e7/raw/ 2>/dev/null; then
      echo "WARNING: results/e7/raw/ changed. The refresh is meant to be a no-op on a clean" >&2
      echo "         checkout; inspect the diff before trusting anything downstream." >&2
    fi
  fi

  echo "== figures, tables and generated numbers =="
  python scripts/make_figures.py
  python scripts/make_tables.py
  python scripts/make_numbers.py
  python scripts/check_manuscript_numbers.py
  python scripts/check_manuscript_refs.py
  python scripts/check_abstract_length.py

  echo "== manuscript =="
  ( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
fi
echo "run_all: done"
