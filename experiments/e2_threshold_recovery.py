#!/usr/bin/env python3
"""E2 -- threshold recovery at quadratic feature load, and the interface separation.

Scientific question: at ``F = d^2``, up to which sparsity does thresholding recover the exact
Boolean state, and what is the irreducible error of the *same* linear map read as a
real-valued readout at that sparsity?

This is the campaign that exhibits the separation on a single axis, which the earlier
manuscript never did. The threshold side is Monte Carlo; the linear side is exact.
"""
from __future__ import annotations

from typing import Any, Dict, List

from _common import RunRecord, base_parser, log, prepare, write_json

from lrtr.diagnostic import random_code_scaling_experiment


def main() -> None:
    parser = base_parser("e2")
    args = parser.parse_args()
    cfg, out_dir, threads = prepare("e2", args)

    ds: List[int] = cfg["ds"]
    trials: int = cfg["trials"]
    seed: int = cfg["seed"]
    thetas: List[float] = cfg.get("thetas", [0.5])
    s_max_by_d: Dict[str, int] = cfg["s_max"]

    with RunRecord("e2_threshold_recovery", out_dir, config={**cfg, "threads": threads},
                   smoke=args.smoke) as rec:
        profiles: List[Dict[str, Any]] = []
        for d in ds:
            F = d * d
            s_max = int(s_max_by_d[str(d)])
            for theta in thetas:
                prof = random_code_scaling_experiment(
                    d=d, F=F, sparsities=list(range(1, s_max + 1)), trials=trials,
                    master_seed=seed * 1000 + d, theta=theta)
                profiles.append(prof)
                cert = prof["separation_certificate"]
                log(f"  d={d:4d} F={F:7d} theta={theta}  s95={prof['s95']}  "
                    + (f"linear RMS at s95={cert['linear_rms_error']:.4f} "
                       f"({cert['linear_rms_over_dminushalf']:.2f} x d^-1/2)"
                       if cert else "no sparsity reached p_rec >= 0.95"))
        rec.set("n_profiles", len(profiles))
        write_json(out_dir / "raw" / "e2_profiles.json", {"profiles": profiles})


if __name__ == "__main__":
    main()
