#!/usr/bin/env python3
"""Is the frozen arm's `R_readout` ~ 1.92 an optimisation shortfall, or the price its own objective
asks for?

E8's frozen arm trains only `W_out`, for the full budget, and ends at nearly twice the cross-talk of
the best readout of the code it was handed. The tempting reading is that gradient descent fails at
the conditional problem. That reading needs a check it does not survive without.

The check that does *not* work, recorded here because it is the one we tried first: showing that some
linear readout of the same code attains `R_readout` close to 1. That is trivially true --- `pinv(Phi)`
attains the code-specific optimum by construction, for every code --- so it cannot distinguish a
failure from a trade-off. Neither can the `ls` readout of `lrtr.toymodel`, which is fit on Boolean
states and reads the *linear* representation `Phi b`: different objective, different input.

The check that works asks whether the network's own objective prefers `W_out` to that optimum. Same
loss the arm trained under (`L^4` against `relu(x)`), same representation (`relu(x W_in^T) W_out^T`),
same continuous amplitudes, on fresh draws from a seed no training used. If `W_out` beats `pinv(Phi)`
on that loss, then the cross-talk it gives up is bought, not lost.

Writes `results/e8/derived/frozen_readout_tradeoff.json`. Reads the saved weights of both E8 run
records; no retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from lrtr.toymodel import sample_task_batch, target_of  # noqa: E402

# The arms trained at p = 0.01; the evaluation must match the training distribution, and the seed
# must not. 777_001 is disjoint from every seed the campaigns use.
P, BATCH, SEED = 0.01, 20_000, 777_001
WEIGHTS = ("e8/weights", "e8_d200/weights")


def task_l4(Phi: np.ndarray, G: np.ndarray, x: torch.Tensor) -> float:
    """The E8 training objective, evaluated in float64 on a held-out batch."""
    pred = torch.relu(x @ torch.as_tensor(Phi).T) @ torch.as_tensor(G).T
    return float(((pred - target_of(x, "relu")) ** 4).mean())


def main() -> int:
    rows = []
    for rel in WEIGHTS:
        for wf in sorted((ROOT / "results" / rel).glob("*_d*.npz")):
            arm, dtag = wf.stem.rsplit("_d", 1)
            z = np.load(wf)
            for k in range(z["W_in"].shape[0]):
                Phi, W_out = z["W_in"][k], z["W_out"][k]
                gen = torch.Generator().manual_seed(SEED + k)
                x = sample_task_batch(Phi.shape[1], BATCH, P, gen, torch.float64)
                rows.append({"arm": arm, "d": int(dtag), "seed_index": k,
                             "l4_wout": task_l4(Phi, W_out, x),
                             "l4_pinv": task_l4(Phi, np.linalg.pinv(Phi), x)})

    cells = []
    for arm in ("trained", "frozen", "random"):
        for d in sorted({r["d"] for r in rows}):
            v = [r for r in rows if r["arm"] == arm and r["d"] == d]
            if not v:
                continue
            wo = float(np.median([r["l4_wout"] for r in v]))
            pi = float(np.median([r["l4_pinv"] for r in v]))
            cells.append({
                "arm": arm, "d": d, "n_models": len(v),
                "l4_wout_median": wo, "l4_pinv_median": pi,
                # > 1 means the trained readout beats the cross-talk optimum on the task.
                "task_gain_over_pinv": pi / wo,
                "models_where_wout_wins": sum(r["l4_wout"] < r["l4_pinv"] for r in v),
            })

    out = ROOT / "results" / "e8" / "derived" / "frozen_readout_tradeoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"p": P, "batch": BATCH, "seed": SEED,
                               "loss": "L4 against relu(x), post-ReLU representation",
                               "cells": cells, "per_model": rows}, indent=2),
                   encoding="utf-8", newline="\n")

    print(f"{'arm':<9}{'d':>5}{'L4(W_out)':>13}{'L4(pinv)':>13}{'gain':>8}{'W_out wins':>12}")
    for c in cells:
        print(f"{c['arm']:<9}{c['d']:>5}{c['l4_wout_median']:>13.3e}{c['l4_pinv_median']:>13.3e}"
              f"{c['task_gain_over_pinv']:>8.2f}{c['models_where_wout_wins']:>8}/"
              f"{c['n_models']}")
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
