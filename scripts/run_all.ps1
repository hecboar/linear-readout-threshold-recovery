# Reproduce every experiment, table, figure and manuscript number from a clean checkout (Windows).
#
# Usage:
#   powershell -File scripts\run_all.ps1                  # everything except the accelerator campaigns
#   powershell -File scripts\run_all.ps1 -Smoke           # same shape, minutes, to check the plumbing
#   powershell -File scripts\run_all.ps1 -WithCampaigns   # also retrain E7 (A, B, C) and E8: needs a GPU
#
# Kept in step with scripts/run_all.sh. See that file for why the campaigns are opt-in: everything the
# manuscript quotes is computed from the committed weights by the CPU-only analysis step below, so a
# GPU is only needed by a reader who doubts the training itself rather than the analysis of it.
param(
  [switch]$Smoke,
  [switch]$WithCampaigns,
  [string]$Device = "cuda",
  [int]$Workers = 10
)
$ErrorActionPreference = "Stop"
# Every explicit --out-dir must carry the smoke suffix that prepare() would have added on its own.
# Passing --smoke with a bare --out-dir writes reduced-fidelity output over committed results, which
# is the one way a reproduction script can destroy the thing it exists to verify.
$sfx = if ($Smoke) { "_smoke" } else { "" }
Set-Location (Join-Path $PSScriptRoot "..")
# Logs go to results\logs\smoke\ under -Smoke. The tracked logs in results\logs\ belong to the real
# runs, and writing a reduced run over them loses the record of the run the paper reports. The bash
# twin tees its output; PowerShell does not, so here the directory is only created for parity.
$logs = if ($Smoke) { "results\logs\smoke" } else { "results\logs" }
New-Item -ItemType Directory -Force $logs | Out-Null

Write-Host "== environment =="
python --version
python -c "import numpy,scipy,matplotlib,torch;print('numpy',numpy.__version__,'scipy',scipy.__version__,'matplotlib',matplotlib.__version__,'torch',torch.__version__)"

Write-Host "== tests =="
python -m pytest -q tests/
if (-not $?) { throw "tests failed" }

$flag = if ($Smoke) { "--smoke" } else { "" }

Write-Host "== experiments E1-E6 (CPU) =="
foreach ($e in @("e1_welch_floor","e2_threshold_recovery","e3_linear_energy",
                 "e4_optimized_codes","e5_trained_toy","e6_threshold_scaling")) {
  Write-Host "-- $e"
  if ($flag) { python "experiments\$e.py" $flag } else { python "experiments\$e.py" }
  if (-not $?) { throw "$e failed" }
}

Write-Host "== E5 again, keeping weights, for the G1 re-analysis =="
# reanalyse_g1.py needs the trained codes themselves, which the default E5 run does not keep.
if ($flag) { python experiments\e5_trained_toy.py $flag --out-dir "results\e5_weights$sfx" }
else       { python experiments\e5_trained_toy.py --out-dir "results\e5_weights$sfx" }
if (-not $?) { throw "e5_weights failed" }
# Skipped under -Smoke: reanalyse_g1.py reads results\e5_weights\ and writes results\g1\, both
# committed, so running it during a plumbing check would rewrite real output from a reduced run.
if (-not $Smoke) { python scripts\reanalyse_g1.py; if (-not $?) { throw "g1 re-analysis failed" } }

if ($WithCampaigns) {
  Write-Host "== E0: device capacity benchmark =="
  python experiments\e0_benchmark.py --device $Device; if (-not $?) { throw "e0 failed" }

  Write-Host "== E7 stage A: the main grid, 180 models (hours to days) =="
  python -u experiments\e7_scaled_toy.py $flag --stage A --device $Device --resume --workers $Workers
  if (-not $?) { throw "e7 stage A failed" }

  Write-Host "== E7 stage B: feature load and training sparsity, 100 models =="
  python -u experiments\e7_scaled_toy.py $flag --config configs\e7_stageB.json --stage B `
      --device $Device --workers $Workers --out-dir "results\e7_stageB$sfx"
  if (-not $?) { throw "e7 stage B failed" }

  Write-Host "== E7 stage C: d=400, testing the frontier rate the first three widths imply =="
  python -u experiments\e7_scaled_toy.py $flag --config configs\e7_stageC.json --stage C `
      --device $Device --workers $Workers --out-dir "results\e7_stageC$sfx"
  if (-not $?) { throw "e7 stage C failed" }

  Write-Host "== E8: trained / frozen-code / untrained arms =="
  # Two invocations with two run records, because d=200 was added after the first run and its
  # provenance is its own. Merging them on disk would claim one launch that never happened.
  python -u experiments\e8_frozen_encoder.py $flag --device $Device --widths 50 100 `
      --seeds 10 --workers $Workers
  if (-not $?) { throw "e8 failed" }
  python -u experiments\e8_frozen_encoder.py $flag --device $Device --widths 200 `
      --seeds 10 --workers $Workers --out-dir "results\e8_d200$sfx"
  if (-not $?) { throw "e8 d=200 failed" }
} else {
  Write-Host "== E0, E7 and E8 skipped: pass -WithCampaigns to retrain them =="
  Write-Host "   the analysis below runs from the committed weights and needs no accelerator"
}

if (-not $Smoke) {
  Write-Host "== derived analysis (CPU, from the committed E7/E8 artefacts) =="
  # Order matters once: all_feature_frontier writes the exact frontier that refresh_native_blocks
  # stamps into the per-cell records.
  foreach ($Stage in @("e7", "e7_stageB", "e7_stageC")) {
    python scripts\all_feature_frontier.py 8 $Stage
    if (-not $?) { throw "all-feature frontier failed for $Stage" }
    # KD6: select_thresholds never scored theta_fixed, so every threshold-dependent probe block
    # written before 2026-08-14 is suspect. This refits them from the saved W_in -- no retraining --
    # and must run before primary_comparison, which reads those blocks.
    python scripts\refresh_probe_blocks.py 8 $Stage
    if (-not $?) { throw "probe refresh failed for $Stage" }
    python scripts\primary_comparison.py $Stage
    if (-not $?) { throw "primary comparison failed for $Stage" }
  }
  python scripts\refresh_native_blocks.py;       if (-not $?) { throw "native refresh failed" }
  python scripts\native_comparison.py;           if (-not $?) { throw "native comparison failed" }
  python scripts\network_threshold_policies.py;  if (-not $?) { throw "threshold policies failed" }
  python scripts\relu_frontier_gap.py;           if (-not $?) { throw "relu frontier gap failed" }
  python scripts\frozen_readout_tradeoff.py;     if (-not $?) { throw "frozen tradeoff failed" }

  # refresh_native_blocks rewrites tracked files in results\e7\raw\. On a clean checkout it is
  # deterministic and must produce no diff; a diff means the analysis code or the archive has drifted.
  git rev-parse --git-dir 2>$null | Out-Null
  if ($?) {
    git diff --quiet -- results/e7/raw/
    if (-not $?) {
      Write-Warning "results/e7/raw/ changed. The refresh is meant to be a no-op on a clean checkout;"
      Write-Warning "inspect the diff before trusting anything downstream."
    }
  }

  if ($env:WITH_SAE) {
    Write-Host "== released SAE dictionaries (needs network access to download checkpoints) =="
    python scripts\sae_diagnostic.py; if (-not $?) { throw "SAE diagnostic failed" }
  } else {
    Write-Host "== SAE dictionary analysis skipped: set WITH_SAE=1 to download and re-measure =="
    Write-Host "   results/sae/derived/sae_axes.json is committed, with every checkpoint's SHA-256"
  }

  Write-Host "== figures, tables and generated numbers =="
  python scripts\make_figures.py; if (-not $?) { throw "figures failed" }
  python scripts\make_tables.py;  if (-not $?) { throw "tables failed" }
  python scripts\make_numbers.py; if (-not $?) { throw "numbers failed" }
  python scripts\check_manuscript_numbers.py; if (-not $?) { throw "number check failed" }
  python scripts\check_manuscript_refs.py;    if (-not $?) { throw "reference check failed" }
  python scripts\check_abstract_length.py;    if (-not $?) { throw "abstract length check failed" }

  Write-Host "== manuscript =="
  Push-Location paper
  latexmk -pdf -interaction=nonstopmode tmlr.tex
  Pop-Location
}
Write-Host "run_all: done"
