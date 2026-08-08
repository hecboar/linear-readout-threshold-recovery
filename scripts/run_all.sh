#!/usr/bin/env bash
# Reproduce every experiment, table and figure from a clean checkout.
# Usage:  bash scripts/run_all.sh [--smoke]
set -euo pipefail

SMOKE="${1:-}"
cd "$(dirname "$0")/.."
mkdir -p results/logs

echo "== environment =="
python --version
python -c "import numpy,scipy,matplotlib,torch;print('numpy',numpy.__version__,'scipy',scipy.__version__,'matplotlib',matplotlib.__version__,'torch',torch.__version__)"

echo "== tests =="
python -m pytest -q tests/

echo "== experiments (CPU only) =="
for e in e1_welch_floor e2_threshold_recovery e3_linear_energy \
         e4_optimized_codes e5_trained_toy e6_threshold_scaling; do
  echo "-- $e"
  python "experiments/$e.py" $SMOKE 2>&1 | tee "results/logs/${e}.log"
done

if [ -z "$SMOKE" ]; then
  echo "== figures, tables and generated numbers =="
  python scripts/make_figures.py
  python scripts/make_tables.py
  python scripts/make_numbers.py
  python scripts/check_manuscript_numbers.py
  python scripts/check_manuscript_refs.py

  echo "== manuscript =="
  ( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
fi
echo "run_all: done"
