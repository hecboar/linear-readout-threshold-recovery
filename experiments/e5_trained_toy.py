#!/usr/bin/env python3
"""E5 -- interface separation inside a *trained* toy model of computation in superposition.

Scientific question: the theory says a linear readout interface can never be exact while a
threshold decoder can. Does that separation appear inside a network that was trained by
gradient descent on a real task, rather than in a hand-built or optimised code?

A width-``d`` network is trained on the compressed-computation task under an ``L2`` and an
``L4`` loss, and diagnosed against an i.i.d. random-code control with the same ``(d, F)``.
Three linear readouts of the *linear* representation ``Phi b`` are measured -- the algebraic
pseudoinverse, the model's own decoder, and a least-squares fit -- so that the Welch floor
genuinely applies to each. Which of the three is strongest is an empirical question and is
reported as measured, not assumed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from _common import RunRecord, base_parser, log, map_trials, prepare, write_json

from lrtr.stats import cliffs_delta, mann_whitney
from lrtr.toymodel import diagnose_model, random_code_baseline, train_toy_model


def _job(kind: str, d: int, F: int, p: float, steps: int, batch: int, lr: float, seed: int,
         sparsities: List[int], eval_trials: int, n_fit: int) -> Dict[str, Any]:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    if kind == "random":
        model = random_code_baseline(d, F, seed)
    else:
        model = train_toy_model(d=d, F=F, loss_kind=kind, p=p, steps=steps, batch=batch,
                                lr=lr, seed=seed, log_every=max(1, steps // 5))
    diag = diagnose_model(model, sparsities=sparsities, trials=eval_trials,
                          seed=10_000 + seed, n_fit=n_fit)
    diag["p_train"] = p
    diag["history"] = model.get("history", [])
    # The weights travel back with the diagnosis so that the campaign can be re-diagnosed
    # without retraining. They are stripped before the JSON is written and stored as .npz.
    diag["_weights"] = (np.asarray(model["W_in"], dtype=np.float32),
                        np.asarray(model["W_out"], dtype=np.float32))
    return diag


def main() -> None:
    args = base_parser("e5").parse_args()
    cfg, out_dir, threads = prepare("e5", args)

    d, F = cfg["d"], cfg["F"]
    ps: List[float] = cfg["train_sparsities"]
    n_seeds: int = cfg["n_seeds"]
    steps, batch, lr = cfg["steps"], cfg["batch"], cfg["lr"]
    sparsities: List[int] = cfg["eval_sparsities"]
    eval_trials: int = cfg["eval_trials"]
    n_fit: int = cfg["n_fit"]
    workers = args.workers or cfg.get("workers", 4)

    jobs: List[Tuple[Any, ...]] = []
    for p in ps:
        for kind in ("L2", "L4"):
            for seed in range(n_seeds):
                jobs.append((kind, d, F, p, steps, batch, lr, seed, sparsities,
                             eval_trials, n_fit))
    for seed in range(n_seeds):
        jobs.append(("random", d, F, ps[0], steps, batch, lr, seed, sparsities,
                     eval_trials, n_fit))

    with RunRecord("e5_trained_toy", out_dir,
                   config={**cfg, "threads": threads, "workers": workers, "n_jobs": len(jobs)},
                   smoke=args.smoke) as rec:
        log(f"E5: {len(jobs)} model runs on {workers} worker processes")
        results = map_trials(_job, jobs, workers, threads_per_worker=2,
                             on_done=lambda k, n: log(f"  {k}/{n} model runs finished"))

        # Persist the trained weights. Without them every number below would require a rerun
        # of the whole campaign to check, which is the opposite of what this repository is for.
        weight_dir = out_dir / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            W_in, W_out = r.pop("_weights")
            np.savez_compressed(
                weight_dir / f"{r['loss_kind']}_p{r['p_train']:g}_seed{r['seed']}.npz",
                W_in=W_in, W_out=W_out)
        log(f"  weights written to {weight_dir}")

        # ---- comparisons across model kinds at the primary training sparsity ----
        primary = ps[0]
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            if r["loss_kind"] == "random" or r["p_train"] == primary:
                groups.setdefault(r["loss_kind"], []).append(r)

        def _crosstalk(rs: List[Dict[str, Any]], readout: str) -> List[float]:
            out = []
            for r in rs:
                st = r["floor_stats"].get(readout, {})
                if "ratio_mean_sq" in st:
                    out.append(float(st["ratio_mean_sq"]))
            return out

        comparisons: Dict[str, Any] = {}
        for readout in ("pinv", "ls", "wout"):
            entry: Dict[str, Any] = {}
            for a, b in (("L4", "L2"), ("L4", "random"), ("L2", "random")):
                xa, xb = _crosstalk(groups.get(a, []), readout), _crosstalk(groups.get(b, []), readout)
                if len(xa) >= 2 and len(xb) >= 2:
                    entry[f"{a}_vs_{b}"] = {
                        **mann_whitney(xa, xb),
                        "cliffs_delta": cliffs_delta(xa, xb),
                        f"mean_{a}": float(np.mean(xa)),
                        f"mean_{b}": float(np.mean(xb)),
                    }
            comparisons[readout] = entry

        s95_summary: Dict[str, Any] = {}
        linearity_summary: Dict[str, Any] = {}
        for kind, rs in groups.items():
            keys = ["s95_model"] + [f"s95_linear_{ro}" for ro in ("pinv", "wout", "ls")]
            s95_summary[kind] = {
                key: {
                    "values": [int(r[key]) for r in rs],
                    "mean": float(np.mean([r[key] for r in rs])),
                    "max": int(np.max([r[key] for r in rs])),
                } for key in keys
            }
            linearity_summary[kind] = {
                k: {"mean": float(np.mean([r["linearity"][k] for r in rs])),
                    "std": float(np.std([r["linearity"][k] for r in rs], ddof=1))
                    if len(rs) > 1 else 0.0}
                for k in rs[0]["linearity"]
            }
            log(f"  {kind:7s}: s95(model)={s95_summary[kind]['s95_model']['values']} "
                f"s95(linear,ls)={s95_summary[kind]['s95_linear_ls']['values']}  "
                f"ReLU-clipped={linearity_summary[kind]['frac_relu_clipped']['mean']:.3f} "
                f"agree-with-linear={linearity_summary[kind]['decision_agreement_with_linear']['mean']:.3f}")

        # Separation certificate: sparsities where thresholding the network output is
        # (near) exact while every linear interface provably retains non-zero error.
        certificates: Dict[str, Any] = {}
        for kind, rs in groups.items():
            per_seed = []
            for r in rs:
                s95m = int(r["s95_model"])
                if s95m > 0:
                    row = next(x for x in r["rows"] if x["s"] == s95m)
                    avail = [ro for ro in ("pinv", "wout", "ls") if f"linear_rms_{ro}" in row]
                    per_seed.append({
                        "seed": r["seed"], "s": s95m,
                        "p_rec_model": row["p_rec_model"],
                        "readouts_available": avail,
                        "best_linear_rms": min(row[f"linear_rms_{ro}"] for ro in avail),
                        "rms_floor": row["rms_floor_uniform"],
                        "best_linear_p_rec": max(row[f"p_rec_linear_{ro}"] for ro in avail),
                    })
            certificates[kind] = per_seed

        rec.set("comparisons", comparisons)
        rec.set("s95_summary", s95_summary)
        rec.set("linearity_summary", linearity_summary)
        rec.set("certificates", certificates)
        write_json(out_dir / "raw" / "e5_runs.json",
                   {"runs": results, "comparisons": comparisons,
                    "s95_summary": s95_summary, "linearity_summary": linearity_summary,
                    "certificates": certificates, "primary_train_sparsity": primary})


if __name__ == "__main__":
    main()
