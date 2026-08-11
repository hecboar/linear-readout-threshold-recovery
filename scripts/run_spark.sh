#!/usr/bin/env bash
# Accelerator campaign (E7), staged so that a mistake costs minutes rather than hours.
#
#   bash scripts/run_spark.sh                # environment, smoke, then Stage A, and stop
#   bash scripts/run_spark.sh smoke          # just the end-to-end check on the device
#   bash scripts/run_spark.sh A              # a single stage (A, B or C)
#   bash scripts/run_spark.sh A --proceed    # run it and continue past its gate
#   bash scripts/run_spark.sh resume         # continue an interrupted stage
#
# The stages are gated on purpose. Stage A is a falsification test -- is the L2/L4 distinction
# real across width at all? -- and it writes a GO/NO-GO report and stops. Stage B costs several
# times as much and is only worth running if Stage A left a story alive, so `--proceed` has to
# be given deliberately, after reading the report. Stage C re-analyses the weights A and B
# already trained and trains nothing.
#
# Every stage writes to results/e7*/ with a run_record.json (command, seeds, environment,
# duration, exit status) and records the git commit and dirty status of the tree it ran from
# (D8). The grid checkpoints per cell, so an interruption costs at most the cell in flight.
set -euo pipefail

cd "$(dirname "$0")/.."
STAGE="${1:-default}"
shift || true
EXTRA=("$@")
DEVICE="${LRTR_DEVICE:-cuda}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { printf '\n=== %s  [%s] ===\n' "$1" "$(stamp)"; }

banner "0. environment"
python scripts/check_env_gpu.py 2>&1 | tee "$LOGDIR/e7_env.log"

run_smoke() {
  banner "smoke (minutes) -- proves the pipeline end to end on the device"
  python experiments/e7_scaled_toy.py --smoke --device "$DEVICE" \
    2>&1 | tee "$LOGDIR/e7_smoke.log"
  python -m pytest tests/ -q 2>&1 | tee "$LOGDIR/e7_tests.log"
}

run_stage() {
  local st="$1"; shift
  banner "stage $st"
  python experiments/e7_scaled_toy.py --stage "$st" --device "$DEVICE" "$@" \
    2>&1 | tee -a "$LOGDIR/e7_stage_$st.log"
  banner "stage $st done -- read results/e7/stage_${st}_report.md before going further"
}

case "$STAGE" in
  default)
    run_smoke
    run_stage A "${EXTRA[@]+"${EXTRA[@]}"}"
    ;;
  smoke)
    run_smoke
    ;;
  A|B|C|all)
    run_stage "$STAGE" "${EXTRA[@]+"${EXTRA[@]}"}"
    ;;
  resume)
    run_stage "${LRTR_STAGE:-A}" --resume "${EXTRA[@]+"${EXTRA[@]}"}"
    ;;
  *)
    echo "unknown argument '$STAGE'; expected one of: smoke A B C all resume" >&2
    exit 2
    ;;
esac

banner "done"
echo "raw results:  results/e7/raw/"
echo "weights:      results/e7/weights/   (every number is recomputable from these)"
echo "reports:      results/e7/stage_*_report.md"
echo "next:         python scripts/make_numbers.py && python scripts/check_manuscript_numbers.py"
