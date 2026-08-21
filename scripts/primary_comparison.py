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

The probe's *configuration* is selected on validation, and the test-set maximum over families is
reported beside it. The first version used only the test maximum, which gives the probe a selection
advantage the network does not get -- worth half a sparsity level at `d=100`, enough to flip that
cell's sign. Both are kept: the validation-selected figure is the protocol, the test maximum is the
conservative bound on how good the probe could look.

Outputs `results/<campaign>/derived/primary_comparison.json` and a markdown table on stdout.
Usage: `primary_comparison.py [campaign]`, defaulting to stage A.
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

POLICY = "global"                      # the matched policy; see the module docstring

# Set by main() from the campaign argument. Stage A is the default because it is the campaign the
# headline rests on, but the matched comparison is the only fair one and every stage deserves it --
# stage C's unmatched gate reverses sign against stage A's, and a gate is not evidence either way.
CAMPAIGN = "e7"
RAW = ROOT / "results" / "e7" / "raw"
CFG: Dict[str, Any] = {}


def configure(campaign: str) -> None:
    global CAMPAIGN, RAW, CFG
    CAMPAIGN = campaign
    RAW = ROOT / "results" / campaign / "raw"
    # configs/e7.json for stage A, configs/e7_stageB.json for the directory results/e7_stageB.
    cfg_name = "e7.json" if campaign == "e7" else f"{campaign.replace('e7_', 'e7_')}.json"
    cfg_path = ROOT / "configs" / cfg_name
    if not cfg_path.exists():
        raise SystemExit(f"no config for campaign {campaign!r}: expected {cfg_path}")
    CFG = json.loads(cfg_path.read_text(encoding="utf-8"))


def discover_cells() -> List[Dict[str, Any]]:
    """Read the cells off disk rather than hardcoding them.

    Stage B varies the training sparsity, so a cell is not identified by (loss, d) alone; and a
    stage that has not finished should contribute the cells it has rather than nothing.
    """
    cells = []
    for cf in sorted(RAW.glob("cell_*.json")):
        rec = json.loads(cf.read_text(encoding="utf-8"))
        cells.append({"loss": rec["loss"], "d": rec["d"],
                      "name": cf.stem[len("cell_"):], "path": cf})
    return cells


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


def main(argv: List[str]) -> int:
    configure(argv[0] if argv else "e7")
    out: List[Dict[str, Any]] = []
    print(f"campaign {CAMPAIGN}, matched policy {POLICY!r}")
    for cell in discover_cells():
        loss, d, name = cell["loss"], cell["d"], cell["name"]
        p_train = float(name.split("_p")[1].split("_d")[0])
        wf = ROOT / "results" / CAMPAIGN / "weights" / f"{name}.npz"
        cf = cell["path"]
        if not wf.exists():
            print(f"  skipping {name}: no weights", file=sys.stderr)
            continue
        z = np.load(wf, allow_pickle=True)
        recs = json.loads(cf.read_text(encoding="utf-8"))["diagnoses"]
        sparsities = z["sparsities"].tolist()
        F = int(z["W_in"].shape[2])
        FIELDS = ("s95", "auc", "s95_test_max", "auc_test_max")
        post = {f: [] for f in FIELDS}
        pre = {f: [] for f in FIELDS}
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
            for tag, acc in (("post", post), ("pre", pre)):
                ks = [key for key in pg if f"_{tag}_" in key and key.endswith("_" + POLICY)]
                # The probe family is chosen on VALIDATION, by the criterion the campaign recorded
                # for that purpose. The first version took the maximum over families on the *test*
                # set, which hands the probe a selection advantage the network does not get: it was
                # worth 0.5 of a sparsity level at d=100, enough to flip that cell's sign. The
                # test-set maximum is still reported, separately and labelled, because as an
                # over-generous bound on the probe it is the conservative reading of the comparison.
                best = max(ks, key=lambda key: pg[key]["val_criterion"])
                acc["s95"].append(n_s95 - pg[best]["s95"])
                acc["auc"].append(n_auc - pg[best]["recovery_auc"])
                acc["s95_test_max"].append(n_s95 - max(pg[key]["s95"] for key in ks))
                acc["auc_test_max"].append(n_auc - max(pg[key]["recovery_auc"] for key in ks))
        def blk(acc):
            return {f: boot(np.array(acc[f], float)) for f in FIELDS}

        # p_train identifies the cell alongside (loss, d): stage B runs two training sparsities at
        # one width, so the pair alone collides there and the two rows were indistinguishable.
        row = {"loss": loss, "d": d, "p_train": p_train, "policy": POLICY,
               "n_seeds": len(post["s95"]),
               "probe_selection": "validation (val_criterion); test-set maximum reported alongside",
               "network_minus_probe": {"post_relu": blk(post), "pre_relu": blk(pre)}}
        out.append(row)
        p = row["network_minus_probe"]["post_relu"]["s95"]
        print(f"  {loss:6s} d={d:<4d} post-ReLU s95 diff mean={p['mean']:+.2f} "
              f"[{p['ci_low']:+.2f},{p['ci_high']:+.2f}]  "
              f"net>probe {p['n_positive']}, tie {p['n_zero']}, probe>net {p['n_negative']}",
              flush=True)

    dest = ROOT / "results" / CAMPAIGN / "derived" / "primary_comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"campaign": CAMPAIGN, "policy": POLICY, "cells": out},
                               indent=2),
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
    raise SystemExit(main(sys.argv[1:]))
