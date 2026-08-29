#!/usr/bin/env python3
"""Is the residual decoder gap a fitting artefact? Score the network's own map as a probe.

The affine probe reads the post-ReLU state through a design matrix that appends an intercept and
nothing else, so its hypothesis class *contains* the network's output layer: with

    W_net = [[W_out^T], [0]]                      (d+1, F), zero intercept

`score_probe(..., W_net, "post")` reproduces `evaluate_network` exactly. A probe configuration
attaining the network's score therefore exists in every cell, and whenever the network scores higher
what has been measured is our own selection failing to find it -- not a limit on what an affine map
can express. The manuscript states that ceiling; this script tests it, which is the difference
between a caveat and a result.

The test needs no fitting. The saved cell records already store `val_criterion` for every probe
configuration: the validation curve-AUC, which is exactly the objective the campaign's selection
maximises. So the only new number is that same objective evaluated at `W_net`. If the selected
configuration scores *below* the network's own map on it, the selection provably missed a better
member of its own class using data it was allowed to look at, and the gap is a search failure.

Three numbers per model, all from the campaign's own validation split, never the test split:

    net_val      the validation curve-AUC of W_net, thresholded like a probe (global, on validation)
    probe_val    the best post-ReLU probe's `val_criterion`, read from the saved record
    net_theta    the threshold that produced net_val, to separate the map from its operating point

and, for the record, the two test numbers those maps produce.

What this deliberately does NOT do: refit the probe from W_net. That asks a different and more
expensive question -- whether the surrogate objective pulls *away* from a map that recovers supports
-- and needs the full ridge grid and 300-step margin fits for every model, some hundred CPU-hours at
`d=400`. `fit_probe` accepts a `W0` starting point so that experiment can be run from the released
weights; `--with-refit` wires it up. It is not run for the paper and nothing here depends on it.

Usage: `probe_ceiling.py [workers] [--with-refit]`. Reads every campaign's committed weights.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _common import map_trials  # noqa: E402

from lrtr.probes import (DEFAULT_RIDGE_GRID, PROBE_FAMILIES, _exact_recovery,  # noqa: E402
                         fit_probe, score_probe, select_thresholds)
from lrtr.splits import make_state_splits  # noqa: E402

CAMPAIGNS = ("e7", "e7_stageB", "e7_stageC")


def boot(diffs: np.ndarray, reps: int = 20_000, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap over models, paired. Same routine as primary_comparison.boot."""
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(reps, len(diffs)))].mean(axis=1)
    return {"mean": float(diffs.mean()),
            "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)),
            "n_positive": int((diffs > 0).sum()),
            "n_zero": int((diffs == 0).sum()),
            "n_negative": int((diffs < 0).sum())}


def _net_as_probe(W_out: np.ndarray, d: int, F: int) -> np.ndarray:
    """`W_out` as a member of the post-ReLU probe class: `(d+1, F)` with a zero intercept."""
    W = np.zeros((d + 1, F))
    W[:d, :] = W_out.T
    return W


def _curve(W_in: np.ndarray, by_s: Dict[int, Any], W: np.ndarray, theta) -> float:
    """The campaign's selection objective, verbatim: normalised trapezoid over the validation curve.

    This must be the SAME functional `select_probe` maximises, or the comparison is between two
    different objectives and the word "missed" is unearned. `select_probe` computes
    `trapezoid(ys, xs) / (max(xs) - min(xs))`; an earlier version of this file used the plain mean
    over grid points and claimed in its docstring that the mean *was* the selection objective. It is
    not: on a uniform integer grid the two differ by endpoint reweighting of order
    `[(y_1 + y_n)/2 - mean] / (n - 1)`, which is the same order as the smallest gap this script
    reports. Found by an external adversarial review.
    """
    rows = [(s, float(_exact_recovery(score_probe(W_in, sp, W, "post"), sp, theta).mean()))
            for s, sp in sorted(by_s.items()) if len(sp)]
    if not rows:
        return float("nan")
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    if len(xs) < 2:
        return float(ys[0])
    return float(np.trapezoid(ys, xs) / (max(xs) - min(xs)))


def _s95(W_in: np.ndarray, test_by_s: Dict[int, Any], W: np.ndarray, theta) -> Dict[str, float]:
    rows = [(s, float(_exact_recovery(score_probe(W_in, sp, W, "post"), sp, theta).mean()))
            for s, sp in sorted(test_by_s.items()) if len(sp)]
    ok = [s for s, p in rows if p >= 0.95]
    return {"s95": float(max(ok)) if ok else 0.0,
            "auc": float(np.mean([p for _, p in rows])) if rows else float("nan")}


def one_model(W_in: np.ndarray, W_out: np.ndarray, cfg: Dict[str, Any], sparsities: List[int],
              seed: int, with_refit: bool = False) -> Dict[str, Any]:
    d, F = W_in.shape
    theta_fixed = float(cfg.get("theta", 0.5))
    # The campaign's own bundle: the seed offset of e7_scaled_toy.diagnose, so these are the very
    # states the published numbers were computed on rather than a fresh sample.
    bundle = make_state_splits(F=F, sparsities=sparsities, n_train=cfg["probe_n_train"],
                               n_val=cfg["probe_n_val"], n_test=cfg["probe_n_test"],
                               seed=seed + 1)
    W_net = _net_as_probe(W_out, d, F)
    theta_net = float(select_thresholds(score_probe(W_in, bundle.val, W_net, "post"), bundle.val,
                                        "global", theta_fixed=theta_fixed))
    out: Dict[str, Any] = {
        "seed": int(seed), "d": d, "F": F,
        "net_val_at_fixed_theta": _curve(W_in, bundle.val_by_s, W_net, theta_fixed),
        "net_val": _curve(W_in, bundle.val_by_s, W_net, theta_net),
        "net_theta": theta_net,
        "net_test_at_fixed_theta": _s95(W_in, bundle.test_by_s, W_net, theta_fixed),
        "net_test": _s95(W_in, bundle.test_by_s, W_net, theta_net),
    }
    if with_refit:
        # The expensive follow-up: does the surrogate objective pull away from W_net?
        grid = cfg.get("ridge_grid") or DEFAULT_RIDGE_GRID
        fit_kw = dict(steps=cfg.get("probe_steps", 300), batch=cfg.get("probe_batch", 4096))
        best: Optional[Dict[str, Any]] = None
        for fam in PROBE_FAMILIES:
            for lam, W in fit_probe(W_in, bundle.train, fam, "post", grid=grid, W0=W_net,
                                    **fit_kw).items():
                th = float(select_thresholds(score_probe(W_in, bundle.val, W, "post"), bundle.val,
                                             "global", theta_fixed=theta_fixed))
                val = _curve(W_in, bundle.val_by_s, W, th)
                if best is None or val > best["val"]:
                    best = {"family": fam, "penalty": float(lam), "theta": th, "val": val,
                            **_s95(W_in, bundle.test_by_s, W, th)}
        out["warm_refit"] = best
    return out


def _cell(campaign: str, arm: str, p: str, d: int, cfg: Dict[str, Any], workers: int,
          with_refit: bool) -> Optional[Dict[str, Any]]:
    wpath = ROOT / "results" / campaign / "weights" / f"relu_{arm}_p{p}_d{d}.npz"
    rpath = ROOT / "results" / campaign / "raw" / f"cell_relu_{arm}_p{p}_d{d}.json"
    if not (wpath.exists() and rpath.exists()):
        return None
    z = np.load(wpath)
    rec = json.loads(rpath.read_text(encoding="utf-8"))
    sparsities = [int(s) for s in z["sparsities"]]
    n = z["W_in"].shape[0]
    jobs = [(z["W_in"][i].astype(np.float64), z["W_out"][i].astype(np.float64), cfg, sparsities,
             100_000 + 997 * i, with_refit) for i in range(n)]
    t0 = time.time()
    rows = map_trials(one_model, jobs, workers=workers, threads_per_worker=2)

    # The saved selection, read rather than recomputed: whatever the campaign actually chose.
    for i, row in enumerate(rows):
        if row is None:
            continue
        g = rec["diagnoses"][i]
        # Restricted to the `global` threshold policy, which is the one the matched comparison
        # uses (primary_comparison.py filters the same way). Selecting over all three policies
        # compares against a probe the paper never reports: in most cells the global policy wins
        # on validation anyway and the two agree, but in stage B's untrained cell at d = 200 it
        # does not, and that single cell was the one place this script failed to reproduce
        # Table 1's differences. Found by an external adversarial review, which read the
        # discrepancy as a seed mismatch; the seeds are right and the filter was wrong.
        post = [v for k, v in g["probes_global"].items()
                if "_post_" in k and k.endswith("_global")]
        best = max(post, key=lambda v: v["val_criterion"])
        row["probe_val"] = float(best["val_criterion"])
        row["probe_family"] = f"{best['family']}_{best['policy']}"
        row["probe_test_s95"] = float(best["s95"])
        row["campaign_network_s95"] = float(g["network"]["s95"])

    rows = [r for r in rows if r is not None]
    gap = np.array([r["net_val"] - r["probe_val"] for r in rows], float)
    return {"campaign": campaign, "arm": arm, "p_train": float(p), "d": int(d),
            "F": int(z["W_in"].shape[2]), "n_models": len(rows),
            "duration_seconds": round(time.time() - t0, 1),
            "validation_gap_net_minus_probe": boot(gap),
            "models_where_selection_missed_the_network": int((gap > 0).sum()),
            "net_test_s95_mean": float(np.mean([r["net_test"]["s95"] for r in rows])),
            "net_test_s95_at_fixed_theta_mean":
                float(np.mean([r["net_test_at_fixed_theta"]["s95"] for r in rows])),
            "probe_test_s95_mean": float(np.mean([r["probe_test_s95"] for r in rows])),
            "per_model": rows}


def main(argv: List[str]) -> int:
    with_refit = "--with-refit" in argv
    argv = [a for a in argv if not a.startswith("--")]
    workers = int(argv[0]) if argv else 8
    for campaign in CAMPAIGNS:
        cfg_name = "e7.json" if campaign == "e7" else f"{campaign}.json"
        cfg = json.loads((ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))
        cells = []
        for wpath in sorted((ROOT / "results" / campaign / "weights").glob("relu_*.npz")):
            arm, p, dtag = wpath.stem.split("_")[1:]
            cell = _cell(campaign, arm, p[1:], int(dtag[1:]), cfg, workers, with_refit)
            if cell is None:
                continue
            cells.append(cell)
            g = cell["validation_gap_net_minus_probe"]
            print(f"  {campaign:11s} {arm:7s} p={cell['p_train']:.2f} d={cell['d']:3d}  "
                  f"val(net)-val(probe) {g['mean']:+.4f} "
                  f"[{g['ci_low']:+.4f},{g['ci_high']:+.4f}]  "
                  f"selection missed the network in "
                  f"{cell['models_where_selection_missed_the_network']}/{cell['n_models']}  "
                  f"({cell['duration_seconds']}s)", flush=True)
        out = ROOT / "results" / campaign / "derived" / "probe_ceiling.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"campaign": campaign, "with_refit": with_refit,
                                   "cells": cells}, indent=1), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
