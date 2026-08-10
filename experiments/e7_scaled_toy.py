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


def run_stage_c(stage: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Stage C: the distribution sweep, by re-analysing weights rather than retraining.

    No model is trained here. Every state family is evaluated on the models Stages A and B already
    produced, which is what the external review specified and what makes the sweep cheap: the
    expensive part was the training and it is done.

    The families share supports, so the comparison across distributions is paired and only the
    amplitude law varies. For every family with a positive minimum amplitude the affine frontier is
    recomputed at that ``alpha``, which is the theory's falsifiable prediction for it (E2). Families
    whose amplitudes reach zero carry no frontier statement, and their exact-recovery numbers are
    flagged as not comparable with the rest -- an active coordinate drawn near zero is undetectable
    in principle, so such a rate measures the draw as much as the code.
    """
    from lrtr.probes import distribution_profile
    from lrtr.splits import make_state_splits

    names: List[str] = stage["families"]
    weight_files = sorted((out_dir / "weights").glob("*.npz"))
    if not weight_files:
        raise SystemExit(
            f"Stage C found no weights in {out_dir / 'weights'}. It re-analyses what A and B "
            f"trained, so run at least stage A first.")

    log(f"Stage C: {len(names)} families over {len(weight_files)} weight files"
        f"  -- {stage.get('note', '')}")
    per_model: List[Dict[str, Any]] = []
    for wf in weight_files:
        z = np.load(wf, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        W_in_all, W_out_all = z["W_in"], z["W_out"]
        sparsities = z["sparsities"].tolist()
        for k, seed in enumerate(z["seeds"].tolist()):
            W_in = W_in_all[k].astype(np.float64)
            W_out = W_out_all[k].astype(np.float64)
            F = W_in.shape[1]
            bundle = make_state_splits(F=F, sparsities=sparsities,
                                       n_train=cfg["probe_n_train"], n_val=cfg["probe_n_val"],
                                       n_test=cfg["probe_n_test"], seed=90_000 + 13 * k)
            sel = _frontier_features(W_in, cfg.get("frontier_n_low", 24),
                                     cfg.get("frontier_n_random", 8), 91_000 + k)
            prof = distribution_profile(
                W_in, W_out, bundle, names, frontier_features=sel["features"],
                theta=cfg.get("theta", 0.5), steps=cfg.get("probe_steps", 300),
                batch=cfg.get("probe_batch", 4096), include_oracle=False)
            per_model.append({"cell": wf.stem, "seed": int(seed), "d": int(meta["d"]),
                              "F": F, "loss_kind": meta["loss"], "p_train": meta["p"],
                              **prof})
        log(f"  [{wf.stem}] {len(z['seeds'])} models swept")

    # Aggregate per family, keeping truncated and untruncated apart: their recovery numbers are
    # not on the same footing and averaging across them would hide that.
    agg: Dict[str, Any] = {}
    for name in names:
        rows = [m["families"][name] for m in per_model if name in m["families"]]
        if not rows:
            continue
        agg[name] = {
            "alpha": rows[0]["alpha"],
            "exact_recovery_meaningful": rows[0]["exact_recovery_meaningful"],
            "n_models": len(rows),
            "network_auc_median": float(np.median([r["network"]["recovery_auc"] for r in rows])),
            "best_probe_auc_median": float(np.median([r["best_probe_auc"] for r in rows])),
            "network_beats_probe_fraction": float(np.mean(
                [1.0 if r["network"]["recovery_auc"] > r["best_probe_auc"] else 0.0
                 for r in rows])),
            "kappa_min_median": (float(np.median(
                [r["frontier_at_alpha"]["kappa_min"] for r in rows]))
                if "frontier_at_alpha" in rows[0] else None),
        }
    return {"stage": "C", "families": agg, "n_models": len(per_model),
            "alpha_monotone_all_models": all(m["alpha_monotone"] for m in per_model),
            "per_model": per_model}


def write_stage_c_report(rep: Dict[str, Any], out_dir: Path) -> Path:
    L = ["# Stage C - state distributions", "",
         f"{rep['n_models']} models re-analysed; nothing trained. Supports are shared across "
         "families, so the comparison is paired and only the amplitude law varies.", "",
         "| family | alpha | exact recovery meaningful | network AUC | best probe AUC | "
         "network wins | kappa_min |",
         "|---|---|---|---|---|---|---|"]
    for name, v in rep["families"].items():
        a = "n/a" if v["alpha"] is None else f"{v['alpha']:.2f}"
        km = "n/a" if v["kappa_min_median"] is None else f"{v['kappa_min_median']:.2f}"
        L.append(f"| {name} | {a} | {v['exact_recovery_meaningful']} | "
                 f"{v['network_auc_median']:.3f} | {v['best_probe_auc_median']:.3f} | "
                 f"{v['network_beats_probe_fraction'] * 100:.0f}% | {km} |")
    L += ["", "## Gates", "",
          f"- the affine frontier is monotone in alpha on every model: "
          f"**{rep['alpha_monotone_all_models']}**. E2 requires it, so a failure here is a bug.",
          "",
          "Rows with `exact recovery meaningful = False` have amplitudes reaching zero, so an "
          "active coordinate can be undetectable in principle. Their recovery numbers measure the "
          "draw as much as the code and must not be set against the truncated families."]
    path = out_dir / "stage_C_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return path


def stage_cells(stage: Dict[str, Any], tasks: List[str]) -> List[Any]:
    """Expand a stage definition into `(task, loss, p, d, F, n_seeds)` tuples."""
    out: List[Any] = []
    for block in stage.get("cells", []):
        ratio = int(block["overcompleteness"])
        for t in tasks:
            for loss in block["losses"]:
                for p in block["train_sparsities"]:
                    for d in block["widths"]:
                        out.append((t, loss, float(p), int(d), ratio * int(d),
                                    int(block["n_seeds"])))
    ctrl = stage.get("controls")
    if ctrl:
        ratio = int(ctrl["overcompleteness"])
        for d in ctrl["widths"]:
            out.append((tasks[0], "random", 0.01, int(d), ratio * int(d), int(ctrl["n_seeds"])))
    return out


def stage_report(diags: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    """The GO/NO-GO summary a stage must produce before further compute is spent.

    Four questions, each answerable from this stage alone and each able to end the campaign:

    * **does the network beat the strongest affine probe?** If not, decision D1 already commits us
      to reporting that as the finding rather than hunting for a metric that reverses it.
    * **does the L2/L4 distinction replicate across width?** This is the empirical hook. Without
      it the paper has a method and a null result.
    * **does the geometry/readout split replicate?** The two terms pointed opposite ways for `L2`
      and `L4` on five seeds at one width (finding D11); whether that survives is the question.
    * **does the E3 arrow hold?** A single violation refutes a *proved* proposition, so it would
      mean a bug rather than a result, and the run stops.
    """
    def med(rows, f):
        vals = [f(x) for x in rows]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def best_probe(x):
        return max(v for k, v in x["s95"].items() if k.startswith("probe_"))

    widths = sorted({x["d"] for x in diags})
    per_width: Dict[str, Any] = {}
    for d in widths:
        at_d = [x for x in diags if x["d"] == d]
        row: Dict[str, Any] = {}
        for kind in ("L2", "L4", "random"):
            sub = [x for x in at_d if x["loss_kind"] == kind]
            if not sub:
                continue
            row[kind] = {
                "n": len(sub),
                "s95_model": med(sub, lambda x: x["s95"]["model"]),
                "s95_best_probe": med(sub, best_probe),
                "R_geom": med(sub, lambda x: x["theory"]["analog"]["R_geom"]),
                "R_readout_wout": med(
                    sub, lambda x: x["theory"]["analog"]["R_readout"].get("wout")),
                "kappa_min": med(sub, lambda x: x["theory"]["affine"]["kappa_min"]),
                "leverage_min": med(sub, lambda x: x["theory"]["analog"]["leverage_min"]),
            }
        per_width[str(d)] = row

    trained = [x for x in diags if x["loss_kind"] != "random"]
    beats = [1.0 if x["s95"]["model"] > best_probe(x) else 0.0 for x in trained]

    seps = []
    for d in widths:
        a = [x for x in diags if x["d"] == d and x["loss_kind"] == "L4"]
        b = [x for x in diags if x["d"] == d and x["loss_kind"] == "L2"]
        if a and b:
            ga = med(a, lambda x: x["theory"]["analog"]["R_geom"])
            gb = med(b, lambda x: x["theory"]["analog"]["R_geom"])
            seps.append({"d": d, "R_geom_L4": ga, "R_geom_L2": gb,
                         "separated": bool(ga is not None and gb is not None
                                           and gb > 2.0 * ga)})

    violations = [{"d": x["d"], "loss": x["loss_kind"], "seed": x["seed"]}
                  for x in diags if not x["theory"]["affine"]["e3_bound_respected"]]

    return {
        "stage": stage, "n_models": len(diags), "widths": widths, "per_width": per_width,
        "network_beats_best_probe_fraction": (float(np.mean(beats)) if beats else None),
        "l2_l4_geometry_separated_by_width": seps,
        "l2_l4_separates_everywhere": bool(seps) and all(x["separated"] for x in seps),
        "e3_violations": violations,
        "e3_arrow_held": not violations,
    }


def write_stage_report(rep: Dict[str, Any], out_dir: Path) -> Path:
    def f(x, n=4):
        return "n/a" if x is None else f"{x:.{n}f}"

    L = [f"# Stage {rep['stage']} — GO/NO-GO report", "",
         f"{rep['n_models']} models, widths {rep['widths']}.", "",
         "## Per width", "",
         "| d | loss | n | s95(model) | s95(best probe) | R_geom | R_readout(wout) | kappa_min |",
         "|---|---|---|---|---|---|---|---|"]
    for d, row in rep["per_width"].items():
        for kind, v in row.items():
            L.append(f"| {d} | {kind} | {v['n']} | {f(v['s95_model'], 1)} | "
                     f"{f(v['s95_best_probe'], 1)} | {f(v['R_geom'], 3)} | "
                     f"{f(v['R_readout_wout'])} | {f(v['kappa_min'], 2)} |")
    frac = (rep["network_beats_best_probe_fraction"] or 0.0) * 100.0
    L += ["", "## Gates", "",
          f"- the network beats the best affine probe on **{frac:.0f}%** of trained models",
          f"- L2/L4 geometry separated at every width: **{rep['l2_l4_separates_everywhere']}**",
          f"- E3 arrow held: **{rep['e3_arrow_held']}**"
          + ("" if rep["e3_arrow_held"] else f" — VIOLATIONS: {rep['e3_violations']}"), ""]
    if not rep["e3_arrow_held"]:
        L.append("**A violated E3 arrow is a bug, not a result.** The implication is proved, so a "
                 "counterexample means the frontier or the leverage computation is wrong. Find it "
                 "before reading anything else here.")
    elif rep["l2_l4_separates_everywhere"]:
        L.append("The empirical hook survives this stage, so spending the next one is justified.")
    else:
        L.append("The L2/L4 separation did **not** hold at every width. Under decision D1 that is "
                 "reported rather than rescued: re-read the plan's yellow outcomes before "
                 "committing more compute, because the paper's framing may have to change.")
    path = out_dir / f"stage_{rep['stage']}_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return path


def main() -> None:
    parser = base_parser("e7")
    parser.add_argument("--stage", type=str, default="A",
                        help="Stage to run: A, B, C or 'all'. Stages are gated: each writes a "
                             "GO/NO-GO report and stops unless --proceed is given.")
    parser.add_argument("--proceed", action="store_true",
                        help="Continue past a stage gate. Pass this only after reading the "
                             "previous stage's report.")
    args = parser.parse_args()
    cfg, out_dir, threads = prepare("e7", args)

    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested but torch.cuda.is_available() is False. "
            "Run scripts/check_env_gpu.py to see what this interpreter can reach.")

    stages_cfg: Dict[str, Any] = cfg["stages"]
    order = ([args.stage] if args.stage != "all"
             else [k for k in ("A", "B", "C") if k in stages_cfg])
    for st in order:
        if st not in stages_cfg:
            raise SystemExit(f"unknown stage {st!r}; the config defines {sorted(stages_cfg)}")

    tasks: List[str] = cfg["tasks"]
    s_max_by_d: Dict[str, int] = cfg["eval_s_max"]

    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    with RunRecord("e7_scaled_toy", out_dir,
                   config={**cfg, "threads": threads, "device": device,
                           "stages_run": order, "resume": bool(args.resume)},
                   smoke=args.smoke) as rec:
        for st in order:
            stage = stages_cfg[st]
            if stage.get("reanalysis"):
                rep = run_stage_c(stage, cfg, out_dir)
                write_json(out_dir / "raw" / f"e7_stage_{st}.json", rep)
                path = write_stage_c_report(rep, out_dir)
                rec.set(f"stage_{st}", {k: v for k, v in rep.items() if k != "per_model"})
                log("")
                log(f"  Stage {st} report -> {path.name}")
                log(f"    frontier monotone in alpha on every model: "
                    f"{rep['alpha_monotone_all_models']}")
                if not rep["alpha_monotone_all_models"]:
                    raise SystemExit(
                        f"Stage {st}: the frontier was not monotone in alpha. E2 requires it, so "
                        f"this is a bug rather than a result.")
                continue
            cells = stage_cells(stage, tasks)
            if not cells:
                log(f"Stage {st}: no cells defined, skipping")
                continue
            log("")
            log(f"Stage {st}: {len(cells)} cells, {sum(c[5] for c in cells)} models on "
                f"device={device}  — {stage.get('note', '')}")
            if device == "cpu":
                log("  note: device=cpu. This campaign is meant for an accelerator.")

            diags: List[Dict[str, Any]] = []
            for task, loss, p, d, F, n_seeds in cells:
                sparsities = list(range(1, int(s_max_by_d[str(d)]) + 1))
                got = run_cell(task, loss, p, d, F, sparsities,
                               {**cfg, "n_seeds": n_seeds}, device, out_dir, bool(args.resume))
                diags.extend(got)
                s95m = [x["s95"]["model"] for x in got]
                s95p = [max(v for k, v in x["s95"].items() if k.startswith("probe_"))
                        for x in got]
                geom = [x["theory"]["analog"]["R_geom"] for x in got]
                wout = [x["theory"]["analog"]["R_readout"]["wout"] for x in got
                        if x["theory"]["analog"]["R_readout"].get("wout") is not None]
                kap = [x["theory"]["affine"]["kappa_min"] for x in got]
                ok = all(x["theory"]["affine"]["e3_bound_respected"] for x in got)
                log(f"  [{cell_name(task, loss, p, d)}] "
                    f"s95(model) med={np.median(s95m):.1f}  probe med={np.median(s95p):.1f}  "
                    f"R_geom={np.mean(geom):.3f}  R_readout(wout)="
                    + (f"{np.mean(wout):.4f}" if wout else "n/a")
                    + f"  kappa_min={np.mean(kap):.2f}  E3 held={ok}")

            rep = stage_report(diags, st)
            write_json(out_dir / "raw" / f"e7_stage_{st}.json", {"runs": diags, "report": rep})
            path = write_stage_report(rep, out_dir)
            rec.set(f"stage_{st}", {k: v for k, v in rep.items() if k != "per_width"})
            frac = (rep["network_beats_best_probe_fraction"] or 0.0) * 100.0
            log("")
            log(f"  Stage {st} report -> {path.name}")
            log(f"    network beats the best probe on {frac:.0f}% of models")
            log(f"    L2/L4 separated at every width: {rep['l2_l4_separates_everywhere']}")
            log(f"    E3 arrow held: {rep['e3_arrow_held']}")

            if not rep["e3_arrow_held"]:
                raise SystemExit(
                    f"Stage {st}: the E3 arrow was violated. That implication is proved, so this "
                    f"is a bug and not a result. Stopping rather than interpreting it.")
            if st != order[-1] and not args.proceed:
                nxt = order[order.index(st) + 1]
                log("")
                log(f"  Gate: stopping after stage {st}. Read {path.name}, then rerun with "
                    f"--stage {nxt} --proceed")
                break


if __name__ == "__main__":
    main()
