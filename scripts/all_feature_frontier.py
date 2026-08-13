#!/usr/bin/env python3
"""The exact frontier over EVERY feature, removing the upper-bound caveat.

Stage A computed `kappa_min` over a subset -- the 24 lowest-leverage features plus 8 fixed random
ones -- so the reported minimum is an upper bound on the true minimum. The subset is justified by
Theorem 8.3, which identifies low leverage as what forces the frontier down, but that theorem is
sufficient and not necessary, so the subset can miss the true argmin. The external audit called a
full evaluation mandatory for a journal version, and J0's measured solver cost makes it affordable:
0.007 s per feature at `d=50`, 0.034 s at `d=100`, 0.236 s at `d=200`.

Reports, per cell, the true minimum, how far the subset estimate was from it, and where in the
leverage ordering the true argmin actually sits -- which is the number that says whether the
shortcut was sound.

Runs one linear programme per feature. HiGHS is CPU-only and no GPU solver exists in this stack, so
the lever is parallelism across models, not the accelerator.

Outputs `results/e7/derived/all_feature_frontier.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _common import map_trials  # noqa: E402

from lrtr.affine_frontier import collision_frontier  # noqa: E402
from lrtr.analog_optimum import leverage  # noqa: E402

CFG = json.loads((ROOT / "configs" / "e7.json").read_text(encoding="utf-8"))


def one_model(W_in: np.ndarray, n_low: int, n_random: int, seed: int) -> Dict[str, Any]:
    """Full frontier for one code, plus what the subset would have reported."""
    F = W_in.shape[1]
    h = leverage(W_in)
    full = collision_frontier(W_in, model="atmost")
    kappa = np.asarray(full["rho_hat"], dtype=float)      # one entry per feature, all F of them
    order = np.argsort(h)
    rng = np.random.default_rng(seed)
    subset = np.unique(np.concatenate([order[:n_low],
                                       rng.choice(F, size=min(n_random, F), replace=False)]))
    argmin_true = int(np.argmin(kappa))
    rank_of_true = int(np.flatnonzero(order == argmin_true)[0])
    return {"kappa_min_full": float(kappa.min()),
            "kappa_min_subset": float(kappa[subset].min()),
            "subset_overestimate": float(kappa[subset].min() - kappa.min()),
            "subset_found_true_argmin": bool(argmin_true in subset),
            "leverage_rank_of_true_argmin": rank_of_true,
            "n_features": int(F), "n_subset": int(len(subset)),
            "leverage_of_true_argmin": float(h[argmin_true])}


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_low = CFG.get("frontier_n_low", 24)
    n_rand = CFG.get("frontier_n_random", 8)
    rows: List[Dict[str, Any]] = []
    for d in (50, 100, 200):
        for loss in ("L4", "L2", "random"):
            wf = ROOT / "results" / "e7" / "weights" / f"relu_{loss}_p0.01_d{d}.npz"
            if not wf.exists():
                continue
            z = np.load(wf, allow_pickle=True)
            jobs = [(z["W_in"][k].astype(np.float64), n_low, n_rand, 91_000 + k)
                    for k in range(z["W_in"].shape[0])]
            got = map_trials(one_model, jobs, workers=workers, threads_per_worker=2)
            got = [g for g in got if g]
            over = np.array([g["subset_overestimate"] for g in got])
            found = np.array([g["subset_found_true_argmin"] for g in got])
            ranks = np.array([g["leverage_rank_of_true_argmin"] for g in got])
            rows.append({
                "loss": loss, "d": d, "n_models": len(got),
                "kappa_min_full_mean": float(np.mean([g["kappa_min_full"] for g in got])),
                "kappa_min_subset_mean": float(np.mean([g["kappa_min_subset"] for g in got])),
                "subset_overestimate_mean": float(over.mean()),
                "subset_overestimate_max": float(over.max()),
                "models_where_subset_found_the_argmin": int(found.sum()),
                "leverage_rank_of_true_argmin_median": float(np.median(ranks)),
                "leverage_rank_of_true_argmin_max": int(ranks.max()),
                "per_model": got})
            r = rows[-1]
            print(f"  {loss:6s} d={d:<4d} kappa_min full={r['kappa_min_full_mean']:.4f} "
                  f"subset={r['kappa_min_subset_mean']:.4f}  overestimate "
                  f"mean={r['subset_overestimate_mean']:+.4f} max={r['subset_overestimate_max']:+.4f}"
                  f"  subset found argmin in {r['models_where_subset_found_the_argmin']}/"
                  f"{r['n_models']}  true argmin leverage rank median="
                  f"{r['leverage_rank_of_true_argmin_median']:.0f}", flush=True)

    dest = ROOT / "results" / "e7" / "derived" / "all_feature_frontier.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"n_low": n_low, "n_random": n_rand, "cells": rows}, indent=2),
                    encoding="utf-8", newline="\n")
    print(f"\nwrote {dest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
