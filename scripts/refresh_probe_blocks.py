#!/usr/bin/env python3
"""Refit every affine probe with the threshold-selection defect (KD6) fixed.

`select_thresholds` searched a quantile grid and never scored `theta_fixed`, so for the `global` and
`per_feature` policies it could return -- and on this data did return -- a threshold that loses to
`theta_fixed` on the very validation objective it maximises. Everything downstream of a threshold is
therefore suspect: the probes' `s95` and recovery AUC under those two policies, and every comparison
built from them. The `fixed` policy, the network's own scores, the geometry (`R_geom`), and the
frontier (`kappa_min`) do not involve a selected threshold and are untouched.

This refits the probes from the saved `W_in` -- no retraining -- and rewrites only the blocks that
depend on a threshold. `network` and `fixed_readouts` are left alone: they are scored at the
registered `theta` and were never affected.

Both sides of the comparison move. That is the point: the defect handicapped the probes as well as
the network, so this is not a correction that picks a winner, it is one that makes the question
answerable. Run `scripts/primary_comparison.py <campaign>` afterwards to recompute the matched
comparison from the refreshed records.

Usage: `refresh_probe_blocks.py [workers] [campaign]`, defaulting to 8 workers and stage A.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _common import map_trials  # noqa: E402

from lrtr.probes import DEFAULT_RIDGE_GRID, probe_profile  # noqa: E402
from lrtr.splits import make_state_splits  # noqa: E402


def one_model(W_in: np.ndarray, cfg: Dict[str, Any], sparsities: List[int],
              seed: int) -> Dict[str, Any]:
    """The probe half of `e7_scaled_toy.diagnose`, with the same seeds and the same config."""
    F = W_in.shape[1]
    bundle = make_state_splits(F=F, sparsities=sparsities, n_train=cfg["probe_n_train"],
                              n_val=cfg["probe_n_val"], n_test=cfg["probe_n_test"],
                              seed=seed + 1)
    return probe_profile(W_in, bundle, grid=cfg.get("ridge_grid", None) or DEFAULT_RIDGE_GRID,
                         theta_fixed=cfg.get("theta", 0.5),
                         include_oracle=cfg.get("include_oracle", True),
                         steps=cfg.get("probe_steps", 300),
                         batch=cfg.get("probe_batch", 4096))


def main(argv: List[str]) -> int:
    workers = int(argv[0]) if argv else 8
    campaign = argv[1] if len(argv) > 1 else "e7"
    cfg_path = ROOT / "configs" / ("e7.json" if campaign == "e7" else f"{campaign}.json")
    if not cfg_path.exists():
        raise SystemExit(f"no config for campaign {campaign!r}: expected {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw = ROOT / "results" / campaign / "raw"

    t0 = time.perf_counter()
    for cf in sorted(raw.glob("cell_*.json")):
        name = cf.stem[len("cell_"):]
        wf = ROOT / "results" / campaign / "weights" / f"{name}.npz"
        if not wf.exists():
            print(f"  {name}: no weights, skipped", file=sys.stderr)
            continue
        rec = json.loads(cf.read_text(encoding="utf-8"))
        z = np.load(wf, allow_pickle=True)
        sparsities = z["sparsities"].tolist()
        jobs = [(z["W_in"][k].astype(np.float64), cfg, sparsities, 100_000 + 997 * k)
                for k in range(z["W_in"].shape[0])]
        t_cell = time.perf_counter()

        def _tick(done: int, total: int, _n=name, _t=t_cell) -> None:
            el = time.perf_counter() - _t
            eta = el / done * (total - done) if done else float("nan")
            print(f"    [{_n}] refit {done}/{total}  elapsed {el/60:.1f} min  eta {eta/60:.1f} min",
                  flush=True)

        got = map_trials(one_model, jobs, workers=workers,
                         threads_per_worker=2, on_done=_tick)

        before, after = [], []
        for k, probes in enumerate(got):
            if probes is None:
                print(f"  {name}[{k}]: refit failed, record left as it was", file=sys.stderr)
                continue
            diag = rec["diagnoses"][k]
            before.append(max(v["s95"] for v in diag["probes_global"].values()))
            diag["splits"] = probes["splits"]
            diag["probes_global"] = probes["global"]
            diag["probes_oracle"] = probes["oracle"]
            # `s95` and `s95_interpolated` mix the model, the fixed readouts and the probes in one
            # dict. Only the probe entries move; rebuilding the whole dict would silently restate
            # numbers this script did not recompute.
            for key, field in (("s95", "s95"), ("s95_interpolated", "s95_interp")):
                for pname, v in probes["global"].items():
                    diag[key][f"probe_{pname}"] = v[field]
            diag["kd6_fixed"] = True
            after.append(max(v["s95"] for v in probes["global"].values()))

        cf.write_text(json.dumps(rec, indent=2), encoding="utf-8", newline="\n")
        if before:
            print(f"  {name}: {len(after)} refit  best-probe s95 median "
                  f"{np.median(before):.1f} -> {np.median(after):.1f}", flush=True)

    print(f"\ndone in {(time.perf_counter() - t0)/60:.1f} min on {workers} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
