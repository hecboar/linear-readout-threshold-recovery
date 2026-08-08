# Reproduce every experiment, table and figure from a clean checkout (Windows).
# Usage:  powershell -File scripts\run_all.ps1 [-Smoke]
param([switch]$Smoke)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force results\logs | Out-Null

Write-Host "== environment =="
python --version
python -c "import numpy,scipy,matplotlib,torch;print('numpy',numpy.__version__,'scipy',scipy.__version__,'matplotlib',matplotlib.__version__,'torch',torch.__version__)"

Write-Host "== tests =="
python -m pytest -q tests/
if (-not $?) { throw "tests failed" }

$flag = if ($Smoke) { "--smoke" } else { "" }
Write-Host "== experiments (CPU only) =="
foreach ($e in @("e1_welch_floor","e2_threshold_recovery","e3_linear_energy",
                 "e4_optimized_codes","e5_trained_toy","e6_threshold_scaling")) {
  Write-Host "-- $e"
  if ($flag) { python "experiments\$e.py" $flag } else { python "experiments\$e.py" }
  if (-not $?) { throw "$e failed" }
}

if (-not $Smoke) {
  Write-Host "== figures, tables and generated numbers =="
  python scripts\make_figures.py; if (-not $?) { throw "figures failed" }
  python scripts\make_tables.py;  if (-not $?) { throw "tables failed" }
  python scripts\make_numbers.py; if (-not $?) { throw "numbers failed" }
  python scripts\check_manuscript_numbers.py; if (-not $?) { throw "number check failed" }
  python scripts\check_manuscript_refs.py;    if (-not $?) { throw "reference check failed" }

  Write-Host "== manuscript =="
  Push-Location paper
  latexmk -pdf -interaction=nonstopmode main.tex
  Pop-Location
}
Write-Host "run_all: done"
