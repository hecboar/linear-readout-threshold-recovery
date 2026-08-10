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

from lrtr.affine_frontier import (
    affine_failure_threshold,
    collision_frontier,
    leverage_upper_bound_on_kappa,
)
from lrtr.analog_optimum import code_specific_floor, leverage, leverage_excess
from lrtr.codes import welch_floor
from lrtr.diagnostic import (
    interface_energy_moments,
    interface_floor_diagnostic,
    interface_linear_energy,
)
from lrtr.interface import crosstalk_mean_sq
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


def _frontier_features(W_in: np.ndarray, n_low: int, n_random: int,
                       seed: int) -> Dict[str, Any]:
    """Which features to compute the affine frontier for, when `F` LPs is too many.

    The frontier is `min_i kappa_i`, so only the argmin matters, and E3 proves that low leverage
    forces `kappa` down -- which makes ranking by leverage a theory-justified rule rather than a
    convenience. Measured on nine codes (four random, five trained), the argmin sat in the lowest
    16 leverage ranks in eight of them.

    It failed once, at rank 35 of 100, and that failure is expected: the E3 bound is *sufficient*
    for collapse, not necessary, so a feature of middling leverage can still have a small `kappa`
    for reasons leverage does not see -- a near-duplicate partner, for instance. So a fixed random
    sample is drawn alongside the low-leverage block, and the record reports the leverage rank of
    whichever feature turned out to be the argmin, so the heuristic audits itself run by run.

    Testing a subset can only *overestimate* the minimum, so the resulting frontier is an upper
    bound and is labelled as one.
    """
    F = W_in.shape[1]
    h = leverage(W_in)
    order = np.argsort(h)
    low = order[:min(n_low, F)]
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(F), low)
    extra = rng.choice(pool, size=min(n_random, pool.size), replace=False) if pool.size else []
    feats = np.unique(np.concatenate([low, np.asarray(extra, dtype=int)]))
    return {"features": feats.tolist(), "n_low": int(low.size), "n_random": int(len(extra)),
            "leverage_rank_of": {int(f): int(np.where(order == f)[0][0]) for f in feats}}


def _theory_block(W_in: np.ndarray, W_out: np.ndarray, readouts: Dict[str, np.ndarray],
                  cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """G1's decomposition and G2's frontier for one model — the quantities the theory predicts.

    Two things the published diagnostics do not contain, and which the campaign exists to
    replicate across widths:

    * the **geometry / readout split** of the attainment ratio (finding D11). The published number
      divides by the global rank-trace floor and is therefore pure geometry under the
      pseudoinverse; splitting it separates a property of the code from a property of the decoder,
      and on the five E5 seeds the two point opposite ways for `L2` and `L4`.
    * the **affine frontier** and the **E3 arrow**: whether `h_min` below
      `(s-1)^2/((F-1)+(s-1)^2)` really does force the frontier below `s`. That implication is
      proved and was verified on one width; whether it survives scaling is an empirical question
      this records the answer to.
    """
    d, F = W_in.shape
    w_glob = welch_floor(F, d)
    w_code = code_specific_floor(W_in)
    h = leverage(W_in)

    analog: Dict[str, Any] = {
        "welch_floor": w_glob,
        "code_specific_floor": w_code,
        "R_geom": w_code / w_glob,
        "leverage_min": float(h.min()), "leverage_max": float(h.max()),
        "leverage_cv": float(np.std(h) / np.mean(h)),
        **{k: v for k, v in leverage_excess(W_in).items() if k == "excess"},
        "R_readout": {}, "ratio_vs_welch": {},
    }
    for name, G in readouts.items():
        try:
            measured = crosstalk_mean_sq(np.ascontiguousarray(G, dtype=np.float64), W_in)
        except ValueError as exc:
            analog["R_readout"][name] = None
            analog["ratio_vs_welch"][name] = None
            analog[f"error_{name}"] = str(exc)
            continue
        analog["R_readout"][name] = measured / w_code
        analog["ratio_vs_welch"][name] = measured / w_glob

    sel = _frontier_features(W_in, cfg.get("frontier_n_low", 24),
                             cfg.get("frontier_n_random", 8), seed)
    fr = collision_frontier(W_in, feature_subset=sel["features"], model="atmost",
                            alpha=cfg.get("frontier_alpha", 1.0))
    kappa_min = float(min(fr["rho_hat"]))
    argmin = fr["argmin_feature"]

    # The E3 arrow: the smallest sparsity the leverage bound says must fail, against the measured
    # frontier. `predicted >= observed + 1` would be a violation of the proposition.
    s_pred = next((s for s in range(2, F)
                   if h.min() <= affine_failure_threshold(F, s)), None)
    affine = {
        "model": "atmost", "alpha": fr["alpha"],
        "features_tested": fr["features_tested"], "is_upper_bound": True,
        "s_aff_robust_upper": fr["s_aff_robust"],
        "kappa_min": kappa_min,
        "argmin_feature": argmin,
        "argmin_leverage_rank": sel["leverage_rank_of"].get(argmin),
        "subset": {k: sel[k] for k in ("n_low", "n_random")},
        "leverage_upper_bound_on_kappa_at_argmin": float(
            leverage_upper_bound_on_kappa(W_in)[argmin]),
        "e3_first_predicted_failure_s": s_pred,
        "e3_bound_respected": bool(s_pred is None or fr["s_aff_robust"] < s_pred),
        "runtime_s": fr["runtime_s"],
    }
    return {"analog": analog, "affine": affine}


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
        "theory": _theory_block(W_in, W_out, readouts, cfg, seed + 3),
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
            geom = [x["theory"]["analog"]["R_geom"] for x in diags]
            wout = [x["theory"]["analog"]["R_readout"]["wout"] for x in diags
                    if x["theory"]["analog"]["R_readout"].get("wout") is not None]
            kap = [x["theory"]["affine"]["kappa_min"] for x in diags]
            ok = all(x["theory"]["affine"]["e3_bound_respected"] for x in diags)
            log(f"  [{cell_name(task, loss, p, d)}] "
                f"s95(model) med={np.median(s95m):.1f}  probe med={np.median(s95p):.1f}  "
                f"R_geom={np.mean(geom):.3f}  R_readout(wout)="
                + (f"{np.mean(wout):.4f}" if wout else "n/a")
                + f"  kappa_min={np.mean(kap):.2f}  E3 held={ok}")

        rec.set("n_diagnoses", len(all_diagnoses))
        write_json(out_dir / "raw" / "e7_runs.json",
                   {"runs": all_diagnoses, "cells": [cell_name(*c) for c in cells],
                    "widths": widths, "overcompleteness": ratio, "tasks": tasks,
                    "losses": losses, "train_sparsities": ps, "device": device})


if __name__ == "__main__":
    main()
