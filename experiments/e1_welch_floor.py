#!/usr/bin/env python3
"""E1 -- the unit-diagonal Welch floor on random overcomplete codes.

Scientific question: for random unit-norm codes, how far above the floor does the calibrated
readout interface sit, and does the gap close as the feature load ``F/d`` grows?

Unlike the illustration shipped with the earlier manuscript, this campaign builds the code
``Phi`` and the interface ``M`` explicitly, and reports the mean-square and maximum
statistics together with a per-trial count of floor violations. Three readouts are compared:
the tied readout ``G = Phi^T`` (the Gram matrix), the pseudoinverse ``G = Phi^+``, and an
i.i.d. Gaussian readout, which is the least favourable of the three.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from _common import RunRecord, base_parser, log, prepare, write_json

from lrtr.codes import harmonic_tight_frame, random_unit_code, welch_floor, welch_floor_max
from lrtr.interface import crosstalk_stats, unit_diagonal

READOUTS = ("tied", "pinv", "gaussian")


def _interface(Phi: np.ndarray, readout: str, rng: np.random.Generator) -> np.ndarray:
    d, F = Phi.shape
    if readout == "tied":
        return Phi.T @ Phi
    if readout == "pinv":
        return unit_diagonal(np.linalg.pinv(Phi) @ Phi)
    if readout == "gaussian":
        return unit_diagonal(rng.standard_normal((F, d)) @ Phi)
    raise ValueError(f"unknown readout {readout!r}")


def main() -> None:
    args = base_parser("e1").parse_args()
    cfg, out_dir, threads = prepare("e1", args)

    ds: List[int] = cfg["ds"]
    ratios: List[int] = cfg["ratios"]
    trials: int = cfg["trials"]
    seed: int = cfg["seed"]

    with RunRecord("e1_welch_floor", out_dir, config={**cfg, "threads": threads},
                   smoke=args.smoke) as rec:
        rows: List[Dict[str, Any]] = []
        violations = 0
        for d in ds:
            for r in ratios:
                F = int(r * d)
                rng = np.random.default_rng([seed, d, F])
                per_readout: Dict[str, List[Dict[str, float]]] = {k: [] for k in READOUTS}
                for t in range(trials):
                    Phi = random_unit_code(d, F, rng)
                    for ro in READOUTS:
                        st = crosstalk_stats(_interface(Phi, ro, rng), d)
                        violations += int(st.violates_floor)
                        per_readout[ro].append(st.to_dict())
                row: Dict[str, Any] = {
                    "d": d, "F": F, "ratio": r, "trials": trials,
                    "welch_floor_mean_sq": welch_floor(F, d),
                    "welch_floor_max": welch_floor_max(F, d),
                }
                for ro in READOUTS:
                    vals = per_readout[ro]
                    for key in ("mean_sq_offdiag", "max_abs_offdiag",
                                "ratio_mean_sq", "ratio_max"):
                        arr = np.array([v[key] for v in vals], dtype=float)
                        row[f"{ro}_{key}_mean"] = float(arr.mean())
                        row[f"{ro}_{key}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                        row[f"{ro}_{key}_min"] = float(arr.min())
                        row[f"{ro}_{key}_max"] = float(arr.max())
                # Equality witness: a harmonic tight frame at the same (d, F).
                try:
                    st_tf = crosstalk_stats(
                        harmonic_tight_frame(d, F).T @ harmonic_tight_frame(d, F), d)
                    row["tight_frame_ratio_mean_sq"] = st_tf.ratio_mean_sq
                except ValueError as exc:
                    row["tight_frame_ratio_mean_sq"] = None
                    row["tight_frame_note"] = str(exc)
                rows.append(row)
                log(f"  d={d:4d} F={F:5d}  tied ratio={row['tied_ratio_mean_sq_mean']:.4f}  "
                    f"gaussian ratio={row['gaussian_ratio_mean_sq_mean']:.4f}")

        n_measurements = len(ds) * len(ratios) * trials * len(READOUTS)
        rec.set("n_measurements", n_measurements)
        rec.set("floor_violations", violations)
        write_json(out_dir / "raw" / "e1_rows.json",
                   {"rows": rows, "readouts": list(READOUTS),
                    "n_measurements": n_measurements, "floor_violations": violations})
        log(f"E1: {n_measurements} calibrated interfaces, {violations} floor violations")


if __name__ == "__main__":
    main()
