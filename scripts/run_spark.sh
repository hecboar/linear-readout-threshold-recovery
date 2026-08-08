#!/usr/bin/env bash
# Accelerator campaign (E7), staged so that a mistake costs minutes rather than hours.
#
#   bash scripts/run_spark.sh            # the three stages in order
#   bash scripts/run_spark.sh full       # skip straight to the full grid
#   bash scripts/run_spark.sh resume     # continue an interrupted full grid
#
# Every stage writes to results/e7*/ with a run_record.json (command, seeds, environment,
# duration, exit status). The full grid checkpoints per cell, so an interruption at any point
# costs at most the cell in flight.
set -euo pipefail

cd "$(dirname "$0")/.."
STAGE="${1:-all}"
DEVICE="${LRTR_DEVICE:-cuda}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { printf '\n=== %s  [%s] ===\n' "$1" "$(stamp)"; }

banner "0. environment"
python scripts/check_env_gpu.py 2>&1 | tee "$LOGDIR/e7_env.log"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "smoke" ]; then
  banner "1. smoke (minutes) -- proves the pipeline end to end on the device"
  python experiments/e7_scaled_toy.py --smoke --device "$DEVICE" \
    2>&1 | tee "$LOGDIR/e7_smoke.log"
  python -m pytest tests/ -q 2>&1 | tee "$LOGDIR/e7_tests.log"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "pilot" ]; then
  banner "2. pilot -- one width, full step budget, to time the real thing"
  python - <<'PY'
import json, pathlib
cfg = json.loads(pathlib.Path("configs/e7.json").read_text())
cfg["widths"] = [cfg["widths"][0]]
cfg["tasks"] = [cfg["tasks"][0]]
cfg["losses"] = ["L4"]
cfg["train_sparsities"] = [cfg["train_sparsities"][0]]
cfg.pop("smoke", None)
pathlib.Path("configs/e7_pilot.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("wrote configs/e7_pilot.json")
PY
  python experiments/e7_scaled_toy.py --config configs/e7_pilot.json \
    --out-dir results/e7_pilot --device "$DEVICE" 2>&1 | tee "$LOGDIR/e7_pilot.log"
  banner "pilot done -- multiply its wall-clock by the cell count before committing"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "full" ] || [ "$STAGE" = "resume" ]; then
  banner "3. full grid"
  RESUME=""
  [ "$STAGE" = "resume" ] && RESUME="--resume"
  python experiments/e7_scaled_toy.py --device "$DEVICE" $RESUME \
    2>&1 | tee -a "$LOGDIR/e7_full.log"
fi

banner "done"
echo "raw results:  results/e7/raw/"
echo "weights:      results/e7/weights/   (every number is recomputable from these)"
echo "next:         python scripts/make_numbers.py && python scripts/check_manuscript_numbers.py"
