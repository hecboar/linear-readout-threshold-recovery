#!/usr/bin/env python3
"""The corrected primary comparison: same decoder input, same threshold treatment, paired.

The stage gate compared the network at a fixed `theta = 0.5`, reading `ReLU(Phi b)`, against the
best of eighteen probe configurations, two thirds of which read the pre-ReLU state `Phi b` and two
thirds of which tune their threshold on validation. That mixes three separate asymmetries into one
number (KD2, KD4). This recomputes the comparison with all three removed:

* **same input.** The probe is restricted to the post-ReLU representation, which is the state the
  network's own output layer reads.
* **same threshold treatment.** Both sides get one validation-selected global threshold. That adds
  exactly one calibration parameter to each decoder, unlike `fixed` -- where `theta = 0.5` is not
  on a common scale across ridge, logistic and hinge outputs -- and unlike `per_feature`, which
  adds `F`.
* **paired, per seed.** Differences are computed within a seed and summarised with a bootstrap
  interval, rather than compared as medians of two independent columns.

The pre-ReLU comparison is reported alongside, as a *different* question: whether the ReLU discards
information that was affinely accessible in the preactivation.

Nothing is retrained. The network is rescored from the saved weights; the probe numbers are read
from the per-seed records the campaign already wrote.

Outputs `results/e7/derived/primary_comparison.json` and a markdown table on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrtr.probes import _exact_recovery, select_thresholds  # noqa: E402
from lrtr.splits import make_state_splits, representations  # noqa: E402
from lrtr.threshold import s95_from_curve  # noqa: E402

RAW = ROOT / "results" / "e7" / "raw"
CFG = json.loads((ROOT / "configs" / "e7.json").read_text(encoding="utf-8"))
POLICY = "global"                      # the matched policy; see the module docstring


def auc(ss: List[int], pr: List[float]) -> float:
    if len(ss) < 2:
        return float(pr[0]) if pr else float("nan")
    return float(np.trapezoid(pr, ss) / (max(ss) - min(ss)))


def network_curve(W_in, W_out, bundle, sparsities, policy):
    """The network's own thresholded output, with its threshold chosen like the probes'."""
    def scores(split):
        return representations(W_in, split, post_relu=True) @ W_out.T

    th = (CFG["theta"] if policy == "fixed"
          else select_thresholds(scores(bundle.val), bundle.val, policy,
                                 theta_fixed=CFG["theta"]))
    ss, pr = [], []
    for s in sparsities:
        sub = bundle.test_by_s[s]
        if len(sub):
            ss.append(s)
            pr.append(float(_exact_recovery(scores(sub), sub, th).mean()))
    return ss, pr


def boot(diffs: np.ndarray, reps: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap over seeds. Paired, so the resample is over models."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {"mean": float(diffs.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)),
            "n_positive": int((diffs > 0).sum()),
            "n_zero": int((diffs == 0).sum()),
            "n_negative": int((diffs < 0).sum())}


def main() -> int:
    cells = [(loss, d) for d in (50, 100, 200) for loss in ("L4", "L2", "random")]
    out: List[Dict[str, Any]] = []
    for loss, d in cells:
        name = f"relu_{loss}_p0.01_d{d}"
        wf = ROOT / "results" / "e7" / "weights" / f"{name}.npz"
        cf = RAW / f"cell_{name}.json"
        if not (wf.exists() and cf.exists()):
            print(f"  skipping {name}: missing artefacts", file=sys.stderr)
            continue
        z = np.load(wf, allow_pickle=True)
        recs = json.loads(cf.read_text(encoding="utf-8"))["diagnoses"]
        sparsities = z["sparsities"].tolist()
        F = int(z["W_in"].shape[2])
        ds95_post, dauc_post, ds95_pre, dauc_pre = [], [], [], []
        for k, seed in enumerate(z["seeds"].tolist()):
            W_in = z["W_in"][k].astype(np.float64)
            W_out = z["W_out"][k].astype(np.float64)
            bundle = make_state_splits(F=F, sparsities=sparsities,
                                       n_train=CFG["probe_n_train"], n_val=CFG["probe_n_val"],
                                       n_test=CFG["probe_n_test"],
                                       seed=(100_000 + 997 * k) + 1)
            ss, pr = network_curve(W_in, W_out, bundle, sparsities, POLICY)
            n_s95, n_auc = s95_from_curve(ss, pr), auc(ss, pr)
            pg = recs[k]["probes_global"]
            for tag, s95_acc, auc_acc in (("post", ds95_post, dauc_post),
                                          ("pre", ds95_pre, dauc_pre)):
                ks = [key for key in pg if f"_{tag}_" in key and key.endswith("_" + POLICY)]
                s95_acc.append(n_s95 - max(pg[key]["s95"] for key in ks))
                auc_acc.append(n_auc - max(pg[key]["recovery_auc"] for key in ks))
        row = {"loss": loss, "d": d, "policy": POLICY, "n_seeds": len(ds95_post),
               "network_minus_probe": {
                   "post_relu": {"s95": boot(np.array(ds95_post, float)),
                                 "auc": boot(np.array(dauc_post, float))},
                   "pre_relu": {"s95": boot(np.array(ds95_pre, float)),
                                "auc": boot(np.array(dauc_pre, float))}}}
        out.append(row)
        p = row["network_minus_probe"]["post_relu"]["s95"]
        print(f"  {loss:6s} d={d:<4d} post-ReLU s95 diff mean={p['mean']:+.2f} "
              f"[{p['ci_low']:+.2f},{p['ci_high']:+.2f}]  "
              f"net>probe {p['n_positive']}, tie {p['n_zero']}, probe>net {p['n_negative']}",
              flush=True)

    dest = ROOT / "results" / "e7" / "derived" / "primary_comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"policy": POLICY, "cells": out}, indent=2),
                    encoding="utf-8", newline="\n")

    print("\n| cell | input | s95 diff (net - probe) | 95% CI | net/tie/probe | AUC diff | 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        for tag, label in (("post_relu", "post-ReLU"), ("pre_relu", "pre-ReLU")):
            a = r["network_minus_probe"][tag]["s95"]
            b = r["network_minus_probe"][tag]["auc"]
            print(f"| {r['loss']} d={r['d']} | {label} | {a['mean']:+.2f} | "
                  f"[{a['ci_low']:+.2f}, {a['ci_high']:+.2f}] | "
                  f"{a['n_positive']}/{a['n_zero']}/{a['n_negative']} | {b['mean']:+.4f} | "
                  f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
    print(f"\nwrote {dest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
