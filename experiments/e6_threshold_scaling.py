#!/usr/bin/env python3
"""E6 -- scaling threshold recovery to d = 1024 (F = d^2 ~ 1.05e6).

Scientific question: how does the empirical 95% recovery threshold ``s_95(d)`` grow with the
width? The theory predicts a ``d / log d`` scale; the earlier manuscript could only show
``d in {64, 128}``, where the transition sits at ``s = 1-2`` and two points cannot exhibit a
scale.

The code is never materialised. Each trial regenerates it in blocks from a counted seed
sequence and evaluates every sparsity in one pass using nested supports, so ``d = 1024``
(a 4 GB code in float32) runs in a few hundred MB.

Results are written per ``d`` and per chunk, so an interrupted campaign resumes without
recomputing completed work.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _common import RunRecord, base_parser, log, map_trials, prepare, write_json

from lrtr.codes import welch_floor
from lrtr.diagnostic import tied_energy_moments_streaming, tied_linear_energy
from lrtr.interface import energy_floor_uniform
from lrtr.runlog import read_json
from lrtr.stats import fit_c_over_log, wilson_interval
from lrtr.threshold import recovery_trial_streaming, s95_from_curve


def _job(master: int, trial: int, d: int, F: int, s_max: int, theta: float,
         block: int, with_energy: bool) -> Dict[str, Any]:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ok = recovery_trial_streaming(master, trial, d, F, s_max, theta=theta, block=block)
    out: Dict[str, Any] = {"trial": trial, "ok": [bool(v) for v in ok]}
    if with_energy:
        mom = tied_energy_moments_streaming(master, trial, d, F, block=block)
        out["energy"] = [tied_linear_energy(mom, s) for s in range(1, s_max + 1)]
        out["frob_sq_A"] = mom.frob_sq_A
    return out


def _load_partial(path: Path) -> tuple[Dict[int, Dict[str, Any]], float]:
    """Return the trials already on disk and the compute time they cost.

    The elapsed time is accumulated in the file so that a resumed campaign still reports the
    total cost of producing the results, not just the cost of the final session.
    """
    if not path.exists():
        return {}, 0.0
    try:
        payload = read_json(path)
        return ({int(r["trial"]): r for r in payload.get("trials", [])},
                float(payload.get("elapsed_seconds", 0.0)))
    except Exception:
        return {}, 0.0


def main() -> None:
    args = base_parser("e6").parse_args()
    cfg, out_dir, threads = prepare("e6", args)

    ds: List[int] = cfg["ds"]
    s_max_by_d: Dict[str, int] = cfg["s_max"]
    trials_by_d: Dict[str, int] = cfg["trials"]
    theta: float = cfg.get("theta", 0.5)
    block: int = cfg.get("block", 1 << 16)
    seed: int = cfg["seed"]
    chunk: int = cfg.get("chunk", 10)
    energy_every: int = cfg.get("energy_every", 5)
    workers = args.workers or cfg.get("workers", 6)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with RunRecord("e6_threshold_scaling", out_dir,
                   config={**cfg, "threads": threads, "workers": workers},
                   smoke=args.smoke) as rec:
        per_d: List[Dict[str, Any]] = []
        timings: Dict[str, float] = {}

        for d in ds:
            F = d * d
            s_max = int(s_max_by_d[str(d)])
            n_trials = int(trials_by_d[str(d)])
            master = seed * 1_000_003 + d
            path = raw_dir / f"e6_d{d}.json"
            done, prior_elapsed = _load_partial(path)
            if len(done) >= n_trials:
                log(f"  d={d}: {len(done)} trials already present "
                    f"({prior_elapsed:.1f}s of recorded compute), skipping")
            t0 = time.perf_counter()

            def _persist() -> None:
                write_json(path, {"d": d, "F": F, "s_max": s_max, "theta": theta,
                                  "master_seed": master, "block": block,
                                  "elapsed_seconds": round(prior_elapsed
                                                           + time.perf_counter() - t0, 2),
                                  "trials": [done[t] for t in sorted(done)]})

            while len(done) < n_trials:
                todo = [t for t in range(n_trials) if t not in done][:chunk]
                jobs = [(master, t, d, F, s_max, theta, block, (t % energy_every == 0))
                        for t in todo]
                for res in map_trials(_job, jobs, workers, threads_per_worker=1):
                    done[int(res["trial"])] = res
                _persist()
                log(f"  d={d}: {len(done)}/{n_trials} trials "
                    f"({prior_elapsed + time.perf_counter() - t0:.1f}s cumulative)")
            timings[str(d)] = round(prior_elapsed + time.perf_counter() - t0, 2)

            ok = np.array([done[t]["ok"] for t in sorted(done)][:n_trials], dtype=bool)
            energies = [done[t]["energy"] for t in sorted(done) if "energy" in done[t]]
            e_mean = np.mean(np.array(energies, dtype=float), axis=0) if energies else None

            rows: List[Dict[str, Any]] = []
            for i in range(s_max):
                k = int(ok[:, i].sum())
                lo, hi = wilson_interval(k, ok.shape[0])
                row: Dict[str, Any] = {
                    "s": i + 1, "successes": k, "trials": int(ok.shape[0]),
                    "p_rec": k / ok.shape[0], "ci_low": lo, "ci_high": hi,
                    "energy_floor_uniform": energy_floor_uniform(F, d, i + 1),
                }
                if e_mean is not None:
                    row["linear_energy_per_coord"] = float(e_mean[i])
                    row["linear_rms_error"] = float(np.sqrt(e_mean[i]))
                    row["rms_over_dminushalf"] = float(np.sqrt(e_mean[i]) * np.sqrt(d))
                rows.append(row)

            probs = [r["p_rec"] for r in rows]
            s95 = s95_from_curve([r["s"] for r in rows], probs)
            entry = {
                "d": d, "F": F, "s_max": s_max, "trials": int(ok.shape[0]),
                "theta": theta, "master_seed": master,
                "welch_floor_mean_sq": welch_floor(F, d),
                "rows": rows, "s95": s95,
                "n_energy_trials": len(energies),
                "reference_union_bound": d / (16 * np.log(d)),
                "duration_seconds": timings[str(d)],
            }
            per_d.append(entry)
            log(f"  d={d:5d} F={F:8d}: s95={s95}  "
                f"(union-bound reference {entry['reference_union_bound']:.2f})  "
                f"{timings[str(d)]:.1f}s")

        measured = [(e["d"], e["s95"]) for e in per_d if e["s95"] > 0]
        fit = fit_c_over_log([m[0] for m in measured], [m[1] for m in measured]) \
            if len(measured) >= 2 else None
        rec.set("timings_seconds", timings)
        rec.set("s95_by_d", {str(e["d"]): e["s95"] for e in per_d})
        rec.set("fit", fit)
        write_json(out_dir / "raw" / "e6_summary.json",
                   {"per_d": per_d, "fit": fit, "timings_seconds": timings})
        if fit:
            log(f"E6: s95 ~ c d/ln d with c = {fit['c']:.4f}, R^2 = {fit['r2']:.4f}")


if __name__ == "__main__":
    main()
