#!/usr/bin/env python3
"""E3 -- the average linear-readout energy floor under both sparse-state models.

Scientific question: does the per-coordinate average squared readout error follow the
predicted ``Omega(s/d)`` scale, and is the *uniform-support* floor -- the proposition that
lets both sides of the separation use one and the same distribution -- attained?

For every configuration the energy is computed in closed form (no Monte Carlo) under
(i) independent Bernoulli(``s/F``) states and (ii) uniformly random supports of size ``s``.
A Monte Carlo cross-check of the closed form is run on the smallest configuration so that the
closed form itself is validated rather than trusted.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from _common import RunRecord, base_parser, log, prepare, write_json

from lrtr.codes import random_unit_code
from lrtr.interface import (
    energy_floor_bernoulli,
    energy_floor_uniform,
    linear_energy_bernoulli,
    linear_energy_uniform,
    unit_diagonal,
)


def _monte_carlo_uniform(A: np.ndarray, s: int, n: int, rng: np.random.Generator) -> float:
    F = A.shape[1]
    acc = 0.0
    for _ in range(n):
        b = np.zeros(F)
        b[rng.choice(F, size=s, replace=False)] = 1.0
        r = A @ b
        acc += float(r @ r)
    return acc / n / F


def main() -> None:
    args = base_parser("e3").parse_args()
    cfg, out_dir, threads = prepare("e3", args)

    ds: List[int] = cfg["ds"]
    ratio: int = cfg["ratio"]
    sparsities: List[int] = cfg["sparsities"]
    trials: int = cfg["trials"]
    seed: int = cfg["seed"]
    mc_trials: int = cfg.get("mc_trials", 20000)

    with RunRecord("e3_linear_energy", out_dir, config={**cfg, "threads": threads},
                   smoke=args.smoke) as rec:
        rows: List[Dict[str, Any]] = []
        mc_checks: List[Dict[str, Any]] = []
        floor_violations = 0

        for d in ds:
            F = ratio * d
            rng = np.random.default_rng([seed, d])
            usable = [s for s in sparsities if 1 <= s <= F // 2]
            acc: Dict[int, Dict[str, List[float]]] = {
                s: {"bern": [], "unif": []} for s in usable}
            for t in range(trials):
                Phi = random_unit_code(d, F, rng)
                A = unit_diagonal(Phi.T @ Phi) - np.eye(F)
                for s in usable:
                    eb = linear_energy_bernoulli(A, s / F)
                    eu = linear_energy_uniform(A, s)
                    floor_violations += int(eb < energy_floor_bernoulli(F, d, s) * (1 - 1e-9))
                    floor_violations += int(eu < energy_floor_uniform(F, d, s) * (1 - 1e-9))
                    acc[s]["bern"].append(eb)
                    acc[s]["unif"].append(eu)
                if t == 0 and d == min(ds):
                    for s in usable[:3]:
                        mc = _monte_carlo_uniform(A, s, mc_trials, rng)
                        closed = linear_energy_uniform(A, s)
                        mc_checks.append({"d": d, "F": F, "s": s, "monte_carlo": mc,
                                          "closed_form": closed,
                                          "rel_diff": abs(mc - closed) / closed,
                                          "mc_trials": mc_trials})
            for s in usable:
                b = np.array(acc[s]["bern"]); u = np.array(acc[s]["unif"])
                rows.append({
                    "d": d, "F": F, "s": s, "s_over_d": s / d, "trials": trials,
                    "energy_bernoulli_mean": float(b.mean()),
                    "energy_bernoulli_std": float(b.std(ddof=1)) if b.size > 1 else 0.0,
                    "energy_uniform_mean": float(u.mean()),
                    "energy_uniform_std": float(u.std(ddof=1)) if u.size > 1 else 0.0,
                    "floor_bernoulli": energy_floor_bernoulli(F, d, s),
                    "floor_uniform": energy_floor_uniform(F, d, s),
                    "ratio_bernoulli_to_floor": float(b.mean()) / energy_floor_bernoulli(F, d, s),
                    "ratio_uniform_to_floor": float(u.mean()) / energy_floor_uniform(F, d, s),
                    "ratio_uniform_to_s_over_d": float(u.mean()) / (s / d),
                })
            log(f"  d={d:4d} F={F:5d}: {len(usable)} sparsities")

        ratios = [r["ratio_uniform_to_s_over_d"] for r in rows]
        summary = {
            "n_configurations": len(rows),
            "floor_violations": floor_violations,
            "ratio_uniform_to_s_over_d_min": float(min(ratios)),
            "ratio_uniform_to_s_over_d_max": float(max(ratios)),
        }
        rec.set("summary", summary)
        write_json(out_dir / "raw" / "e3_rows.json",
                   {"rows": rows, "monte_carlo_checks": mc_checks, "summary": summary})
        log(f"E3: {len(rows)} configurations, {floor_violations} floor violations, "
            f"E/(s/d) in [{summary['ratio_uniform_to_s_over_d_min']:.4f}, "
            f"{summary['ratio_uniform_to_s_over_d_max']:.4f}]")


if __name__ == "__main__":
    main()
