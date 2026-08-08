#!/usr/bin/env python3
"""E4 -- do gradient-optimised codes attain the Welch floor?

Scientific question: the tightness proposition says the floor is attained by unit-norm tight
frames, which are hand-constructed objects. Is it also attained by codes that are *learned*
under the unit-diagonal constraint?

Variants: ``free`` (independent ``G`` and ``Phi``), ``tied`` (``G = Phi^T``, which also tests
whether a tight frame emerges), ``softmax`` (targets the maximum form of the bound), and the
ablation ``uncalibrated`` (no diagonal constraint -- the optimiser is then free to shrink the
gains instead of reducing interference).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from _common import RunRecord, base_parser, log, map_trials, prepare, write_json

from lrtr.optimize import train_code


def _job(d: int, F: int, variant: str, seed: int, steps: int, lr: float,
         record_every: int) -> Dict[str, Any]:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    return train_code(d=d, F=F, variant=variant, steps=steps, lr=lr, seed=seed,
                      record_every=record_every)


def main() -> None:
    args = base_parser("e4").parse_args()
    cfg, out_dir, threads = prepare("e4", args)

    ds: List[int] = cfg["ds"]
    ratios: List[int] = cfg["ratios"]
    steps: int = cfg["steps"]
    lr: float = cfg["lr"]
    record_every: int = cfg["record_every"]
    seeds_by_variant: Dict[str, int] = cfg["seeds_by_variant"]
    workers = args.workers or cfg.get("workers", 4)

    jobs: List[Tuple[Any, ...]] = []
    for variant, n_seeds in seeds_by_variant.items():
        for d in ds:
            for r in ratios:
                for seed in range(int(n_seeds)):
                    jobs.append((d, int(r * d), variant, seed, steps, lr, record_every))

    with RunRecord("e4_optimized_codes", out_dir,
                   config={**cfg, "threads": threads, "workers": workers, "n_jobs": len(jobs)},
                   smoke=args.smoke) as rec:
        log(f"E4: {len(jobs)} optimisation runs on {workers} worker processes")
        # Longest configurations first: with 168 heterogeneous jobs, submitting them in
        # nesting order leaves the largest (d=64, F=1024) runs for the end and the pool
        # finishes on a long tail with most workers idle.
        jobs.sort(key=lambda j: -(j[0] * j[1] * j[1]))
        results = map_trials(_job, jobs, workers, threads_per_worker=1,
                             on_done=lambda k, n: log(f"  {k}/{n} runs finished")
                             if k % 10 == 0 or k == n else None)

        by_variant: Dict[str, List[Dict[str, Any]]] = {}
        for res in results:
            by_variant.setdefault(res["variant"], []).append(res)

        summary: Dict[str, Any] = {}
        for variant, runs in by_variant.items():
            ratios_ms = np.array([r["ratio_mean_sq"] for r in runs], dtype=float)
            entry: Dict[str, Any] = {
                "n_runs": len(runs),
                "ratio_mean_sq_mean": float(np.nanmean(ratios_ms)),
                "ratio_mean_sq_max": float(np.nanmax(ratios_ms)),
                "ratio_mean_sq_min": float(np.nanmin(ratios_ms)),
                "n_degenerate": int(sum(bool(r.get("interface_degenerate")) for r in runs)),
                "n_below_floor": int(np.nansum(ratios_ms < 1.0 - 1e-6)),
            }
            if variant == "tied":
                res_tf = np.array([r["tight_frame_residual"] for r in runs], dtype=float)
                entry["tight_frame_residual_mean"] = float(res_tf.mean())
                entry["tight_frame_residual_max"] = float(res_tf.max())
            if variant == "softmax":
                rmax = np.array([r["ratio_max"] for r in runs], dtype=float)
                entry["ratio_max_mean"] = float(np.nanmean(rmax))
                entry["ratio_max_max"] = float(np.nanmax(rmax))
            if variant == "uncalibrated":
                raw = np.array([r["ratio_mean_sq_raw"] for r in runs], dtype=float)
                gains = np.array([r["diag_gain_min_abs"] for r in runs], dtype=float)
                entry["ratio_mean_sq_raw_mean"] = float(raw.mean())
                entry["ratio_mean_sq_raw_min"] = float(raw.min())
                entry["diag_gain_min_abs_min"] = float(gains.min())
            summary[variant] = entry
            log(f"  {variant:13s}: n={entry['n_runs']:3d} "
                f"ratio to floor mean={entry['ratio_mean_sq_mean']:.4f} "
                f"max={entry['ratio_mean_sq_max']:.4f} "
                f"below-floor={entry['n_below_floor']}")

        rec.set("summary", summary)
        write_json(out_dir / "raw" / "e4_runs.json", {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
