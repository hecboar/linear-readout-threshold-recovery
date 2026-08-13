#!/usr/bin/env python3
"""What the ReLU costs, measured in sparsity levels on both sides of it.

Two earlier measurements disagreed about which code loses more to its own ReLU, because one was in
sparsity levels (Boolean `s95`) and the other in a probability (native top-`k`), and the codes sit
at different points on their curves. This measures both sides in the *same* unit -- a sparsity --
with no probe, no fitted threshold and no state distribution.

**Pre-ReLU.** The exact frontier is available: `kappa_i` from the linear programme gives
separability at every sparsity up to `ceil(kappa_i) - 1`. It is reported, but it is *not* what the
gap is computed from -- see below.

**Post-ReLU.** `ReLU(Phi b)` is not a linear image of the state polytope, so the linear programme
does not apply. Instead, at each sparsity we sample admissible Boolean states, apply the ReLU, and
ask a linear programme whether *those* states admit a strict affine separator for feature `i`.

The asymmetry of that test is what makes it usable. If a sample is **inseparable**, the full set
containing it is inseparable too, so the true frontier is below that sparsity. If a sample is
separable the test is inconclusive, so the sampled frontier **overestimates** the true one, and by a
lot: measured on a trained `L4` code whose exact `kappa` gives a frontier of 4, the sampler at 1500
states per side returns 6 or 7, because the rare colliding states are the ones a 1% sample misses.

That is why the quantity reported is

    gap_i  =  sampled_pre_frontier_i  -  sampled_post_frontier_i

with the **same sampler, the same budget and the same random stream on both sides**, so the
overestimation is common to both and cancels in the difference. Comparing the exact pre-ReLU
frontier against a sampled post-ReLU one would be comparing an exact quantity with a loose
estimate, which is the error a first version of this script made; the exact `kappa` is still
reported, as the calibration that shows how loose the sampler is.

Outputs `results/e7/derived/relu_frontier_gap.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrtr.affine_frontier import collision_radius_atmost  # noqa: E402
from lrtr.analog_optimum import leverage  # noqa: E402

N_SAMPLE = 1500          # per side, per sparsity
N_FEATURES = 6           # lowest-leverage features, which is where the frontier is smallest
N_SEEDS = 5              # models per cell
S_MAX = 12


def separable(X_pos: np.ndarray, X_neg: np.ndarray) -> bool:
    """Is there `(w, theta)` with `w.x > theta` on X_pos and `< theta` on X_neg?

    Asked with a unit margin, which is a normalisation rather than a restriction: the state sets are
    finite, so any strict separator can be scaled to achieve it.
    """
    d = X_pos.shape[1]
    A = np.vstack([np.hstack([-X_pos, np.ones((len(X_pos), 1))]),
                   np.hstack([X_neg, -np.ones((len(X_neg), 1))])])
    b = -np.ones(len(X_pos) + len(X_neg))
    res = linprog(np.zeros(d + 1), A_ub=A, b_ub=b,
                  bounds=[(None, None)] * (d + 1), method="highs")
    return bool(res.success)


def sample_states(Phi, i, s, rng, post_relu):
    F = Phi.shape[1]
    others = np.delete(np.arange(F), i)
    cap = len(others)                                  # cannot draw more partners than exist
    pos, neg = [], []
    for _ in range(N_SAMPLE):
        k = min(int(rng.integers(1, s + 1)), cap + 1)   # |S| <= s, and i is in S
        S = np.concatenate([[i], rng.choice(others, size=k - 1, replace=False)])
        pos.append(Phi[:, S].sum(axis=1))
        k2 = min(int(rng.integers(1, s + 1)), cap)      # |S| <= s, and i is not in S
        S2 = rng.choice(others, size=k2, replace=False)
        neg.append(Phi[:, S2].sum(axis=1))
    P, N = np.array(pos), np.array(neg)
    if post_relu:
        P, N = np.maximum(P, 0.0), np.maximum(N, 0.0)
    return P, N


def sampled_frontier(Phi, i, rng, post_relu) -> int:
    """Largest `s` at which the sampled states are still separable; `s_max` if never refuted."""
    for s in range(1, S_MAX + 1):
        P, N = sample_states(Phi, i, s, rng, post_relu)
        if not separable(P, N):
            return s - 1
    return S_MAX


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for d in (50, 100, 200):
        for loss in ("L4", "L2", "random"):
            wf = ROOT / "results" / "e7" / "weights" / f"relu_{loss}_p0.01_d{d}.npz"
            if not wf.exists():
                continue
            z = np.load(wf, allow_pickle=True)
            pre_exact, pre_s, post_s = [], [], []
            for k in range(min(N_SEEDS, z["W_in"].shape[0])):
                W = z["W_in"][k].astype(np.float64)
                for i in np.argsort(leverage(W))[:N_FEATURES]:
                    kap = collision_radius_atmost(W, int(i))["rho_hat"]
                    pre_exact.append(S_MAX if not np.isfinite(kap)
                                     else max(0, int(np.ceil(kap - 1e-9)) - 1))
                    seed = 7000 + 31 * k + int(i)
                    # Same seed on both sides: the sampled supports are identical, so the only
                    # difference between the two numbers is the ReLU.
                    pre_s.append(sampled_frontier(W, int(i), np.random.default_rng(seed), False))
                    post_s.append(sampled_frontier(W, int(i), np.random.default_rng(seed), True))
            pe, a, b = (np.array(pre_exact, float), np.array(pre_s, float),
                        np.array(post_s, float))
            gap = a - b
            rng = np.random.default_rng(0)
            m = gap[rng.integers(0, len(gap), size=(20000, len(gap)))].mean(axis=1)
            rows.append({
                "loss": loss, "d": d, "n_features": len(pe),
                "pre_frontier_exact_mean": float(pe.mean()),
                "pre_frontier_sampled_mean": float(a.mean()),
                "post_frontier_sampled_mean": float(b.mean()),
                "sampler_looseness_mean": float((a - pe).mean()),
                "gap_mean": float(gap.mean()),
                "gap_ci_low": float(np.quantile(m, 0.025)),
                "gap_ci_high": float(np.quantile(m, 0.975)),
                "n_positive": int((gap > 0).sum()), "n_zero": int((gap == 0).sum()),
                "n_negative": int((gap < 0).sum()),
            })
            r = rows[-1]
            print(f"  {loss:6s} d={d:<4d} pre(sampled)={r['pre_frontier_sampled_mean']:.2f} "
                  f"post(sampled)={r['post_frontier_sampled_mean']:.2f}  "
                  f"gap={r['gap_mean']:+.2f} [{r['gap_ci_low']:+.2f},{r['gap_ci_high']:+.2f}]  "
                  f"+/0/-: {r['n_positive']}/{r['n_zero']}/{r['n_negative']}  "
                  f"looseness vs exact={r['sampler_looseness_mean']:+.2f}", flush=True)

    dest = ROOT / "results" / "e7" / "derived" / "relu_frontier_gap.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"n_sample_per_side": N_SAMPLE, "s_max": S_MAX, "cells": rows},
                               indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {dest.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
