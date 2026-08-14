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

Outputs `results/<campaign>/derived/all_feature_frontier.json`. Usage:
`all_feature_frontier.py [workers] [campaign]`, defaulting to 8 workers and stage A.
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

CAMPAIGN = "e7"
CFG: Dict[str, Any] = {}


def configure(campaign: str) -> None:
    """Point the run at one campaign's weights and config.

    The exact frontier is what the trained-versus-untrained margin is now claimed on, and the subset
    shortcut's error is *asymmetric* between the arms: measured on stage A it inflates the trained
    arm by a growing amount (+0.0025, +0.0118, +0.0206 at d = 50, 100, 200) while the untrained arm's
    subset is exact at every width, because there the lowest-leverage feature really is the argmin.
    A margin computed from subset values is therefore biased in favour of training by an amount that
    grows with width, which is precisely the axis the claim is about. Hence: every stage gets the
    exact computation, not only stage A.
    """
    global CAMPAIGN, CFG
    CAMPAIGN = campaign
    cfg_path = ROOT / "configs" / ("e7.json" if campaign == "e7" else f"{campaign}.json")
    if not cfg_path.exists():
        raise SystemExit(f"no config for campaign {campaign!r}: expected {cfg_path}")
    CFG = json.loads(cfg_path.read_text(encoding="utf-8"))


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


def main(argv: List[str]) -> int:
    # `all_feature_frontier.py [workers] [campaign]`, both positional and both optional, so the
    # existing invocation in run_all.sh keeps working unchanged.
    workers = int(argv[0]) if argv else 8
    configure(argv[1] if len(argv) > 1 else "e7")
    n_low = CFG.get("frontier_n_low", 24)
    n_rand = CFG.get("frontier_n_random", 8)
    rows: List[Dict[str, Any]] = []
    # Discovered from the weights on disk: stage B varies the training sparsity and stage C has a
    # single width, so an enumerated (loss, d) grid would silently skip cells.
    for wf in sorted((ROOT / "results" / CAMPAIGN / "weights").glob("relu_*_d*.npz")):
        stem = wf.stem[len("relu_"):]
        loss, rest = stem.split("_p", 1)
        p_train, d = float(rest.split("_d")[0]), int(rest.split("_d")[1])
        z = np.load(wf, allow_pickle=True)
        jobs = [(z["W_in"][k].astype(np.float64), n_low, n_rand, 91_000 + k)
                for k in range(z["W_in"].shape[0])]
        got = map_trials(one_model, jobs, workers=workers, threads_per_worker=2)
        got = [g for g in got if g]
        over = np.array([g["subset_overestimate"] for g in got])
        found = np.array([g["subset_found_true_argmin"] for g in got])
        ranks = np.array([g["leverage_rank_of_true_argmin"] for g in got])
        rows.append({
            # p_train identifies the cell alongside (loss, d): stage B runs two training
            # sparsities at the same width, so the pair alone collides there.
            "loss": loss, "d": d, "p_train": p_train, "n_models": len(got),
            "kappa_min_full_mean": float(np.mean([g["kappa_min_full"] for g in got])),
            "kappa_min_subset_mean": float(np.mean([g["kappa_min_subset"] for g in got])),
            "subset_overestimate_mean": float(over.mean()),
            "subset_overestimate_max": float(over.max()),
            "models_where_subset_found_the_argmin": int(found.sum()),
            "leverage_rank_of_true_argmin_median": float(np.median(ranks)),
            "leverage_rank_of_true_argmin_max": int(ranks.max()),
            "per_model": got})
        r = rows[-1]
        print(f"  {loss:6s} p={p_train:<5g} d={d:<4d} kappa_min full={r['kappa_min_full_mean']:.4f} "
              f"subset={r['kappa_min_subset_mean']:.4f}  overestimate "
              f"mean={r['subset_overestimate_mean']:+.4f} max={r['subset_overestimate_max']:+.4f}"
              f"  subset found argmin in {r['models_where_subset_found_the_argmin']}/"
              f"{r['n_models']}  true argmin leverage rank median="
              f"{r['leverage_rank_of_true_argmin_median']:.0f}", flush=True)

    dest = ROOT / "results" / CAMPAIGN / "derived" / "all_feature_frontier.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"campaign": CAMPAIGN, "n_low": n_low, "n_random": n_rand,
                                "cells": rows}, indent=2),
                    encoding="utf-8", newline="\n")
    print(f"\nwrote {dest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
