#!/usr/bin/env python3
"""E7 -- the trained-network evidence at scale, on an accelerator.

E5 answered "does the separation appear inside a trained network?" at a single width
(``d=50``, ``F=100``) with five seeds and one task. That is enough to show the phenomenon and
not enough to characterise it, and it is the part of the evidence a referee is most entitled to
push on. E7 widens every axis that was pinned there:

* **width** -- several ``d`` at fixed overcompleteness, so the trend is a trend and not a point;
* **seeds** -- twenty or more per cell, trained as one batched computation on the device;
* **task** -- the ReLU target of Braun et al. plus a second elementwise target, so a conclusion
  cannot be an artefact of one objective;
* **baseline** -- cross-validated affine probes on both the pre-ReLU and post-ReLU
  representations, fitted per sparsity and scored on held-out states, alongside the three fixed
  readouts of the theory;
* **distribution** -- the diagnosis is repeated on the *training* distribution, not only on the
  Boolean states of the theory.

Weights are persisted per cell, which E5 did not do: every number here can be recomputed from
``weights/*.npz`` without retraining. Cells are checkpointed, so ``--resume`` picks up an
interrupted campaign where it stopped.

Run on the accelerator with ``--device cuda``; the default stays ``cpu`` so that the smoke
configuration works anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _common import RunRecord, base_parser, log, prepare, write_json

from lrtr.diagnostic import (
    interface_energy_moments,
    interface_floor_diagnostic,
    interface_linear_energy,
)
from lrtr.distributional import native_distribution_profile
from lrtr.probes import (DEFAULT_RIDGE_GRID, evaluate_fixed_readout, evaluate_network,
                         probe_profile)
from lrtr.splits import make_state_splits, one_hot_targets
from lrtr.toymodel import (
    linear_readouts,
    linearity_report,
    random_code_baseline,
    train_toy_models_batched,
)

READOUTS = ("pinv", "wout", "ls")


def _fit_states(bundle, F: int, n_fit: int) -> np.ndarray:
    """`(F, n)` indicator matrix for fitting the least-squares readout of the theory.

    Drawn from the probe *training* split, so the fixed readouts and the fitted probes are
    estimated from the same data and evaluated on the same locked states. Every feature must
    appear at least once: a feature that never does gets an identically zero row in the fitted
    readout, and the unit-diagonal calibration is then undefined for that coordinate.
    """
    B = one_hot_targets(bundle.train).T[:, :max(n_fit, 1)]
    missing = np.nonzero(B.sum(axis=1) == 0)[0]
    if missing.size:
        raise RuntimeError(
            f"{missing.size} of {F} features never appear in the readout fitting set; "
            f"raise probe_n_train (currently giving {B.shape[1]} states)")
    return B


def diagnose(model: Dict[str, Any], cfg: Dict[str, Any], sparsities: List[int],
             seed: int) -> Dict[str, Any]:
    """The full diagnosis of one trained model, all of it out of sample.

    The floor statistics and the analog energies use the ``O(F d^2)`` sufficient-statistic
    route, so nothing here forms the ``F x F`` interface and the campaign stays feasible at the
    widest settings.
    """
    W_in = np.asarray(model["W_in"], dtype=np.float64)
    W_out = np.asarray(model["W_out"], dtype=np.float64)
    d, F = W_in.shape

    bundle = make_state_splits(F=F, sparsities=sparsities, n_train=cfg["probe_n_train"],
                              n_val=cfg["probe_n_val"], n_test=cfg["probe_n_test"],
                              seed=seed + 1)
    B_fit = _fit_states(bundle, F, cfg["n_fit"])
    readouts = linear_readouts(W_in, W_out, B_fit)

    floor_stats: Dict[str, Any] = {}
    analog: Dict[str, Any] = {}
    for name, G in readouts.items():
        G = np.ascontiguousarray(G, dtype=np.float64)
        try:
            floor_stats[name] = interface_floor_diagnostic(W_in, G=G, statistics="mean_sq")
            mom = interface_energy_moments(G, W_in)
        except ValueError as exc:                       # vanishing gain: interface undefined
            floor_stats[name] = {"error": str(exc)}
            continue
        analog[name] = {str(s): interface_linear_energy(mom, s) for s in sparsities}

    # Fitted affine probes: every family, representation and threshold policy, selected on
    # validation and scored on the locked test set. Plus the network and the fixed readouts of
    # the theory, on those same states, so the whole comparison is paired.
    probes = probe_profile(W_in, bundle, grid=cfg.get("ridge_grid", None) or DEFAULT_RIDGE_GRID,
                           theta_fixed=cfg.get("theta", 0.5),
                           include_oracle=cfg.get("include_oracle", True),
                           steps=cfg.get("probe_steps", 300),
                           batch=cfg.get("probe_batch", 4096))
    network = evaluate_network(W_in, W_out, bundle.test_by_s, theta=cfg.get("theta", 0.5))
    fixed = {name: evaluate_fixed_readout(W_in, G, bundle.test_by_s,
                                          theta=cfg.get("theta", 0.5))
             for name, G in readouts.items()}

    # The exact analog error sits alongside the measured recovery rates, per sparsity.
    for row in network["rows"]:
        for name, per_s in analog.items():
            e = per_s[str(row["s"])]
            row[f"analog_energy_{name}"] = e
            row[f"analog_rms_{name}"] = float(np.sqrt(e))

    return {
        "d": d, "F": F, "seed": model["seed"], "loss_kind": model["loss_kind"],
        "task": model.get("task", "relu"), "p_train": model.get("p"),
        "final_mse": model.get("final_mse"),
        "mse_ratio_to_zero_predictor": model.get("mse_ratio_to_zero_predictor"),
        "floor_stats": floor_stats,
        "linearity": linearity_report(W_in, W_out, B_fit, cfg.get("theta", 0.5)),
        "splits": probes["splits"],
        "probes_global": probes["global"],
        "probes_oracle": probes["oracle"],
        "network": network,
        "fixed_readouts": fixed,
        "s95": {"model": network["s95"],
                **{f"linear_{k}": v["s95"] for k, v in fixed.items()},
                **{f"probe_{k}": v["s95"] for k, v in probes["global"].items()}},
        "s95_interpolated": {"model": network["s95_interp"],
                             **{f"linear_{k}": v["s95_interp"] for k, v in fixed.items()},
                             **{f"probe_{k}": v["s95_interp"]
                                for k, v in probes["global"].items()}},
        "native": native_distribution_profile(
            W_in, W_out, p=model.get("p") or cfg["train_sparsities"][0],
            n_train=cfg["native_n_train"], n_test=cfg["native_n_test"], seed=seed + 2,
            readouts={n: np.asarray(readouts[n]) for n in READOUTS}),
    }


def cell_name(task: str, loss: str, p: float, d: int) -> str:
    return f"{task}_{loss}_p{p:g}_d{d}"


def run_cell(task: str, loss: str, p: float, d: int, F: int, sparsities: List[int],
             cfg: Dict[str, Any], device: str, out_dir: Path,
             resume: bool) -> List[Dict[str, Any]]:
    name = cell_name(task, loss, p, d)
    ckpt = out_dir / "raw" / f"cell_{name}.json"
    if resume and ckpt.exists():
        log(f"  [{name}] resumed from checkpoint")
        return json.loads(ckpt.read_text(encoding="utf-8"))["diagnoses"]

    seeds = list(range(cfg["n_seeds"]))
    if loss == "random":
        models = [random_code_baseline(d, F, s) for s in seeds]
        for m in models:
            m["task"], m["p"] = task, p
    else:
        models = train_toy_models_batched(
            d=d, F=F, loss_kind=loss, p=p, seeds=seeds, steps=cfg["steps"],
            batch=cfg["batch"], lr=cfg["lr"], task=task, device=device,
            log_every=max(1, cfg["steps"] // 5))

    diagnoses = [diagnose(m, cfg, sparsities, seed=100_000 + 997 * i) for i, m in enumerate(models)]

    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "weights" / f"{name}.npz",
        W_in=np.stack([np.asarray(m["W_in"], dtype=np.float32) for m in models]),
        W_out=np.stack([np.asarray(m["W_out"], dtype=np.float32) for m in models]),
        seeds=np.array(seeds), sparsities=np.array(sparsities),
        meta=np.array(json.dumps({"task": task, "loss": loss, "p": p, "d": d, "F": F,
                                  "steps": cfg["steps"], "batch": cfg["batch"],
                                  "lr": cfg["lr"], "device": device})))
    write_json(ckpt, {"cell": name, "task": task, "loss": loss, "p": p, "d": d, "F": F,
                      "diagnoses": diagnoses,
                      "history": [m.get("history", []) for m in models]})
    return diagnoses


def main() -> None:
    parser = base_parser("e7")
    args = parser.parse_args()
    cfg, out_dir, threads = prepare("e7", args)

    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested but torch.cuda.is_available() is False. "
            "Run scripts/check_env_gpu.py to see what this interpreter can reach.")

    widths: List[int] = cfg["widths"]
    ratio: int = cfg["overcompleteness"]
    tasks: List[str] = cfg["tasks"]
    losses: List[str] = cfg["losses"]
    ps: List[float] = cfg["train_sparsities"]
    s_max_by_d: Dict[str, int] = cfg["eval_s_max"]

    cells = [(t, l, p, d) for t in tasks for l in losses for p in ps for d in widths]
    cells += [(tasks[0], "random", ps[0], d) for d in widths]

    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    with RunRecord("e7_scaled_toy", out_dir,
                   config={**cfg, "threads": threads, "device": device,
                           "n_cells": len(cells), "resume": bool(args.resume)},
                   smoke=args.smoke) as rec:
        log(f"E7: {len(cells)} cells x {cfg['n_seeds']} seeds on device={device}")
        if device == "cpu":
            log("  note: device=cpu. This configuration is meant for an accelerator; "
                "on CPU expect the full grid to take many hours.")

        all_diagnoses: List[Dict[str, Any]] = []
        for task, loss, p, d in cells:
            F = ratio * d
            s_max = int(s_max_by_d[str(d)])
            sparsities = list(range(1, s_max + 1))
            diags = run_cell(task, loss, p, d, F, sparsities, cfg, device, out_dir,
                             bool(args.resume))
            all_diagnoses.extend(diags)
            s95m = [x["s95"]["model"] for x in diags]
            best_probe = max(probes_key for probes_key in diags[0]["s95"]
                             if probes_key.startswith("probe_"))
            s95p = [max(v for k, v in x["s95"].items() if k.startswith("probe_"))
                    for x in diags]
            ratios = [x["floor_stats"]["pinv"]["ratio_mean_sq"] for x in diags
                      if "ratio_mean_sq" in x["floor_stats"].get("pinv", {})]
            log(f"  [{cell_name(task, loss, p, d)}] "
                f"s95(model) median={np.median(s95m):.1f}  "
                f"s95(best probe) median={np.median(s95p):.1f}  "
                + (f"pinv ratio mean={np.mean(ratios):.4f}" if ratios else "pinv degenerate"))

        rec.set("n_diagnoses", len(all_diagnoses))
        write_json(out_dir / "raw" / "e7_runs.json",
                   {"runs": all_diagnoses, "cells": [cell_name(*c) for c in cells],
                    "widths": widths, "overcompleteness": ratio, "tasks": tasks,
                    "losses": losses, "train_sparsities": ps, "device": device})


if __name__ == "__main__":
    main()
