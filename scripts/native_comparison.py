#!/usr/bin/env python3
"""Recompute the native-distribution comparison with KD1 fixed, over every saved model.

KD1: the detection threshold was selected by maximising accuracy at a base rate near 1%, which
places each decoder at whatever conservative operating point its own score distribution happens to
give. The fix selects on the metric being reported and adds a threshold-free ranking measure. This
script re-runs the corrected profile over all 180 Stage A models and reports paired differences.

Nothing is retrained: the profile draws its own states from the saved weights.

Outputs `results/e7/derived/native_comparison.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrtr.distributional import native_distribution_profile  # noqa: E402

CFG = json.loads((ROOT / "configs" / "e7.json").read_text(encoding="utf-8"))


def boot(d: np.ndarray, seed: int = 0, reps: int = 20000) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "ci_low": float(np.quantile(m, 0.025)),
            "ci_high": float(np.quantile(m, 0.975)),
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for d in (50, 100, 200):
        for loss in ("L4", "L2", "random"):
            wf = ROOT / "results" / "e7" / "weights" / f"relu_{loss}_p0.01_d{d}.npz"
            if not wf.exists():
                continue
            z = np.load(wf, allow_pickle=True)
            acc = {k: [] for k in ("f1_net", "f1_post", "f1_pre",
                                   "rank_net", "rank_post", "rank_pre")}
            for k in range(z["W_in"].shape[0]):
                prof = native_distribution_profile(
                    z["W_in"][k].astype(np.float64), z["W_out"][k].astype(np.float64),
                    p=0.01, n_train=CFG["native_n_train"], n_test=CFG["native_n_test"],
                    seed=90_000 + 13 * k)
                det = prof["detection"]
                acc["f1_net"].append(det["model"]["f1_selected"]["f1"])
                acc["f1_post"].append(det["probe_post"]["f1_selected"]["f1"])
                acc["f1_pre"].append(det["probe_pre"]["f1_selected"]["f1"])
                acc["rank_net"].append(det["model"]["ranking"]["topk_exact"])
                acc["rank_post"].append(det["probe_post"]["ranking"]["topk_exact"])
                acc["rank_pre"].append(det["probe_pre"]["ranking"]["topk_exact"])
            a = {k: np.array(v, float) for k, v in acc.items()}
            row = {"loss": loss, "d": d, "n_seeds": len(a["f1_net"]),
                   "f1": {k: float(a[f"f1_{k}"].mean()) for k in ("net", "post", "pre")},
                   "topk_exact": {k: float(a[f"rank_{k}"].mean()) for k in ("net", "post", "pre")},
                   "net_minus_post": {"f1": boot(a["f1_net"] - a["f1_post"]),
                                      "topk": boot(a["rank_net"] - a["rank_post"])},
                   "pre_minus_post": {"f1": boot(a["f1_pre"] - a["f1_post"]),
                                      "topk": boot(a["rank_pre"] - a["rank_post"])}}
            rows.append(row)
            n = row["net_minus_post"]["topk"]
            print(f"  {loss:6s} d={d:<4d} top-k exact: net {row['topk_exact']['net']:.3f} "
                  f"post {row['topk_exact']['post']:.3f} pre {row['topk_exact']['pre']:.3f}   "
                  f"net-post {n['mean']:+.4f} [{n['ci_low']:+.4f},{n['ci_high']:+.4f}]", flush=True)

    dest = ROOT / "results" / "e7" / "derived" / "native_comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"cells": rows}, indent=2), encoding="utf-8", newline="\n")

    print("\n| cell | F1 net | F1 post | F1 pre | net-post F1 (CI) | pre-post top-k (CI) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        f = r["net_minus_post"]["f1"]
        t = r["pre_minus_post"]["topk"]
        print(f"| {r['loss']} d={r['d']} | {r['f1']['net']:.3f} | {r['f1']['post']:.3f} | "
              f"{r['f1']['pre']:.3f} | {f['mean']:+.3f} [{f['ci_low']:+.3f}, {f['ci_high']:+.3f}] | "
              f"{t['mean']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] |")
    print(f"\nwrote {dest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
