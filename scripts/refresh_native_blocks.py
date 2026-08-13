#!/usr/bin/env python3
"""Recompute the `native` block of every stored cell record, with KD1 fixed.

The per-cell records in `results/e7/raw/` were written before KD1 was found, so their `native`
blocks carry detection numbers selected by accuracy at a 1% base rate -- the operating point that is
close to "predict everything off". Those numbers have been retracted, and leaving them in the
archive means a reader of `raw/` gets a story the paper does not tell.

Only the `native` sub-block is recomputed. `diagnose` as a whole costs about 5.5 hours over the 180
models because of the probe fitting; this profile draws its own states from the saved weights and
costs seconds, so nothing else in the record is touched or needs to be.

The old values are not preserved in the file. They are preserved in git, which is what an archive
is for, and the commit that runs this says why.

Also stamps the exact all-feature frontier alongside the subset value rather than replacing it: the
subset number is what the campaign computed and the exact one supersedes it, and a reader should be
able to see both and the fact that they nearly agree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrtr.distributional import native_distribution_profile  # noqa: E402

RAW = ROOT / "results" / "e7" / "raw"
WEIGHTS = ROOT / "results" / "e7" / "weights"
CFG = json.loads((ROOT / "configs" / "e7.json").read_text(encoding="utf-8"))


def main() -> int:
    exact = {}
    af = ROOT / "results" / "e7" / "derived" / "all_feature_frontier.json"
    if af.exists():
        for c in json.loads(af.read_text(encoding="utf-8"))["cells"]:
            for k, pm in enumerate(c["per_model"]):
                exact[(c["loss"], c["d"], k)] = pm["kappa_min_full"]

    for cf in sorted(RAW.glob("cell_*.json")):
        rec = json.loads(cf.read_text(encoding="utf-8"))
        wf = WEIGHTS / f"{cf.stem.replace('cell_', '')}.npz"
        if not wf.exists():
            print(f"  {cf.name}: no weights, skipped", file=sys.stderr)
            continue
        z = np.load(wf, allow_pickle=True)
        loss, d = rec["loss"], rec["d"]
        for k, diag in enumerate(rec["diagnoses"]):
            prof = native_distribution_profile(
                z["W_in"][k].astype(np.float64), z["W_out"][k].astype(np.float64),
                p=diag.get("p_train") or 0.01,
                n_train=CFG["native_n_train"], n_test=CFG["native_n_test"],
                seed=90_000 + 13 * k)
            diag["native"] = prof
            diag["native"]["kd1_fixed"] = True
            key = (loss, d, k)
            if key in exact:
                aff = diag["theory"]["affine"]
                # Idempotent on purpose. A second run must not overwrite the subset value with the
                # exact one it was already replaced by -- that would silently destroy the number
                # the campaign actually computed, which is the one a reader needs to judge whether
                # the shortcut was sound.
                aff.setdefault("kappa_min_subset", aff["kappa_min"])
                aff["kappa_min_exact_all_features"] = exact[key]
                aff["kappa_min"] = exact[key]
                aff["is_upper_bound"] = False
                aff["superseded_note"] = (
                    "kappa_min is now the exact minimum over all F features "
                    "(scripts/all_feature_frontier.py); kappa_min_subset is what the campaign "
                    "computed over the lowest-leverage subset plus a random sample.")
        cf.write_text(json.dumps(rec, indent=2), encoding="utf-8", newline="\n")
        f1 = np.mean([x["native"]["detection"]["model"]["f1_selected"]["f1"]
                      for x in rec["diagnoses"]])
        tk = np.mean([x["native"]["detection"]["model"]["ranking"]["topk_exact"]
                      for x in rec["diagnoses"]])
        print(f"  {cf.name}: {len(rec['diagnoses'])} records refreshed  "
              f"network F1={f1:.4f} topk={tk:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
