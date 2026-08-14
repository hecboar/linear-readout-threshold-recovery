#!/usr/bin/env python3
"""Regenerate every manuscript figure from the raw result files.

No number in any figure is written by hand: each panel reads a JSON under ``results/``.
Run after the campaigns:  ``python scripts/make_figures.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import MARKERS, PALETTE, RESULTS, WIDTH_1COL, WIDTH_2COL, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def load(rel: str) -> Any:
    path = RESULTS / rel
    if not path.exists():
        raise SystemExit(f"missing results file {path}; run the corresponding campaign first")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Figure 1 -- the diagnostic on random codes (E1 + E2 + E3)
# --------------------------------------------------------------------------------------

def figure_diagnostic_overview() -> None:
    e1 = load("e1/raw/e1_rows.json")["rows"]
    e2 = load("e2/raw/e2_profiles.json")["profiles"]
    e3 = load("e3/raw/e3_rows.json")["rows"]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 2.25))

    # (a) attainment ratio versus feature load, per width
    ax = axes[0]
    ds = sorted({r["d"] for r in e1})
    for i, d in enumerate(ds):
        rows = sorted([r for r in e1 if r["d"] == d], key=lambda r: r["ratio"])
        x = [r["ratio"] for r in rows]
        y = [r["tied_ratio_mean_sq_mean"] for r in rows]
        lo = [r["tied_ratio_mean_sq_min"] for r in rows]
        hi = [r["tied_ratio_mean_sq_max"] for r in rows]
        ax.plot(x, y, marker=MARKERS[i], color=PALETTE[i], label=fr"$d={d}$")
        ax.fill_between(x, lo, hi, color=PALETTE[i], alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.text(0.97, 0.06, "Welch floor", transform=ax.transAxes, ha="right", fontsize=6.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"feature load $F/d$")
    ax.set_ylabel(r"$\widehat{W}/W(F,d)$")
    ax.set_title("(a) attainment of the floor")
    ax.legend(ncol=2, loc="upper right")

    # (b) threshold recovery at F = d^2, with Wilson intervals
    ax = axes[1]
    prof05 = [p for p in e2 if abs(p["theta"] - 0.5) < 1e-12]
    for i, p in enumerate(sorted(prof05, key=lambda p: p["d"])):
        s = [r["s"] for r in p["rows"]]
        y = [r["p_rec"] for r in p["rows"]]
        lo = [r["p_rec"] - r["ci_low"] for r in p["rows"]]
        hi = [r["ci_high"] - r["p_rec"] for r in p["rows"]]
        ax.errorbar(s, y, yerr=[lo, hi], marker=MARKERS[i], color=PALETTE[i], capsize=1.5,
                    label=fr"$d={p['d']}$")
    ax.axhline(0.95, color="k", ls=":", lw=0.9)
    ax.set_xlabel(r"sparsity $s$")
    ax.set_ylabel(r"$\Pr[\hat b = \mathbf{1}_S]$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(r"(b) threshold recovery, $F=d^2$")
    ax.legend(loc="lower left")

    # (c) linear energy versus the uniform-support floor
    ax = axes[2]
    ds3 = sorted({r["d"] for r in e3})
    for i, d in enumerate(ds3):
        rows = sorted([r for r in e3 if r["d"] == d], key=lambda r: r["s"])
        ax.plot([r["s_over_d"] for r in rows], [r["energy_uniform_mean"] for r in rows],
                marker=MARKERS[i], color=PALETTE[i], label=fr"$d={d}$")
    rows0 = sorted([r for r in e3 if r["d"] == ds3[0]], key=lambda r: r["s"])
    xs = np.array([r["s_over_d"] for r in rows0])
    ax.plot(xs, xs, color="k", ls="--", lw=0.9, label=r"$E=s/d$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$s/d$")
    ax.set_ylabel(r"per-coordinate energy $E$")
    ax.set_title("(c) linear-readout energy")
    ax.legend(loc="upper left")

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig1_diagnostic_overview")


# --------------------------------------------------------------------------------------
# Figure 2 -- optimised codes (E4)
# --------------------------------------------------------------------------------------

def figure_optimized_codes() -> None:
    runs: List[Dict[str, Any]] = load("e4/raw/e4_runs.json")["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.4))

    ax = axes[0]
    free = [r for r in runs if r["variant"] == "free"]
    ds = sorted({r["d"] for r in free})
    for i, d in enumerate(ds):
        sub = [r for r in free if r["d"] == d]
        by_step: Dict[int, List[float]] = {}
        for r in sub:
            for st, v in zip(r["trajectory_steps"], r["trajectory_ratio_mean_sq"]):
                by_step.setdefault(st, []).append(v)
        steps = sorted(by_step)[1:]  # drop step 0: the random initialisation is off-scale
        med = [float(np.median(by_step[s])) for s in steps]
        lo = [float(np.min(by_step[s])) for s in steps]
        hi = [float(np.max(by_step[s])) for s in steps]
        ax.plot(steps, med, color=PALETTE[i], label=fr"$d={d}$")
        ax.fill_between(steps, lo, hi, color=PALETTE[i], alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("optimisation step")
    ax.set_ylabel(r"$\widehat{W}/W(F,d)$")
    ax.set_title("(a) free variant, all feature loads")
    ax.legend()

    ax = axes[1]
    for j, variant in enumerate(["free", "tied", "softmax"]):
        sub = [r for r in runs if r["variant"] == variant]
        x = [r["welch_floor_mean_sq"] for r in sub]
        y = [r["final_mean_sq_offdiag_calibrated"] for r in sub]
        ax.scatter(x, y, s=9, color=PALETTE[j], marker=MARKERS[j], label=variant, alpha=0.8,
                   edgecolors="none")
    lim = [min(r["welch_floor_mean_sq"] for r in runs) * 0.7,
           max(r["final_mean_sq_offdiag_calibrated"] for r in runs
               if r["variant"] != "uncalibrated") * 1.4]
    ax.plot(lim, lim, color="k", ls="--", lw=0.9, label=r"$y=x$ (floor)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(r"Welch floor $W(F,d)$")
    ax.set_ylabel(r"final cross-talk $\widehat{W}$")
    ax.set_title("(b) final cross-talk vs the floor")
    ax.legend(loc="upper left")

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig2_optimized_codes")


# --------------------------------------------------------------------------------------
# Figure 3 -- trained toy model (E5)
# --------------------------------------------------------------------------------------

KIND_STYLE = {"L4": (PALETTE[0], MARKERS[0], "$L^4$-trained"),
              "L2": (PALETTE[1], MARKERS[1], "$L^2$-trained"),
              "random": (PALETTE[2], MARKERS[2], "random code")}


def figure_trained_model() -> None:
    payload = load("e5/raw/e5_runs.json")
    primary = payload["primary_train_sparsity"]
    runs = [r for r in payload["runs"]
            if r["loss_kind"] == "random" or r["p_train"] == primary]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 2.4))

    # (a) attainment of the floor by the learned linear interface
    ax = axes[0]
    kinds = [k for k in ("L4", "L2", "random") if any(r["loss_kind"] == k for r in runs)]
    for i, kind in enumerate(kinds):
        vals = [r["floor_stats"]["ls"]["ratio_mean_sq"] for r in runs
                if r["loss_kind"] == kind and "ratio_mean_sq" in r["floor_stats"]["ls"]]
        colour, marker, label = KIND_STYLE[kind]
        ax.scatter([i] * len(vals), vals, color=colour, marker=marker, s=16, label=label,
                   edgecolors="none")
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([KIND_STYLE[k][2] for k in kinds], rotation=15)
    ax.set_ylabel(r"$\widehat{W}/W(F,d)$")
    ax.set_title("(a) learned interface vs floor")

    # (b) threshold recovery of the network output vs the best linear interface
    ax = axes[1]
    for kind in kinds:
        sub = [r for r in runs if r["loss_kind"] == kind]
        ss = sub[0]["sparsities"]
        colour, marker, label = KIND_STYLE[kind]
        ym = [float(np.mean([next(x for x in r["rows"] if x["s"] == s)["p_rec_model"]
                             for r in sub])) for s in ss]
        yl = [float(np.mean([max(next(x for x in r["rows"] if x["s"] == s)[f"p_rec_linear_{ro}"]
                                 for ro in ("pinv", "wout", "ls")
                                 if f"p_rec_linear_{ro}" in next(x for x in r["rows"] if x["s"] == s))
                             for r in sub])) for s in ss]
        ax.plot(ss, ym, color=colour, marker=marker, label=f"{label}, network")
        ax.plot(ss, yl, color=colour, marker=marker, ls=":", alpha=0.65,
                label=f"{label}, best linear")
    ax.axhline(0.95, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"sparsity $s$")
    ax.set_ylabel(r"$\Pr[\hat b = \mathbf{1}_S]$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(b) exact recovery")
    ax.legend(fontsize=5.6, ncol=1, loc="upper right")

    # (c) irreducible linear error against the floor
    ax = axes[2]
    for kind in kinds:
        sub = [r for r in runs if r["loss_kind"] == kind]
        ss = sub[0]["sparsities"]
        colour, marker, label = KIND_STYLE[kind]
        y = [float(np.mean([min(next(x for x in r["rows"] if x["s"] == s)[f"linear_rms_{ro}"]
                                for ro in ("pinv", "wout", "ls")
                                if f"linear_rms_{ro}" in next(x for x in r["rows"] if x["s"] == s))
                            for r in sub])) for s in ss]
        ax.plot(ss, y, color=colour, marker=marker, label=label)
    floor = [next(x for x in runs[0]["rows"] if x["s"] == s)["rms_floor_uniform"]
             for s in runs[0]["sparsities"]]
    ax.plot(runs[0]["sparsities"], floor, color="k", ls="--", lw=0.9, label="energy floor")
    ax.set_xlabel(r"sparsity $s$")
    ax.set_ylabel("per-coordinate RMS error")
    ax.set_title("(c) best linear readout error")
    ax.legend(fontsize=6)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig3_trained_model")


# --------------------------------------------------------------------------------------
# Figure 4 -- scaling (E6)
# --------------------------------------------------------------------------------------

def figure_scaling() -> None:
    payload = load("e6/raw/e6_summary.json")
    per_d = payload["per_d"]
    fit = payload["fit"]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.4))

    ax = axes[0]
    for i, e in enumerate(sorted(per_d, key=lambda e: e["d"])):
        s = [r["s"] for r in e["rows"]]
        y = [r["p_rec"] for r in e["rows"]]
        lo = [r["p_rec"] - r["ci_low"] for r in e["rows"]]
        hi = [r["ci_high"] - r["p_rec"] for r in e["rows"]]
        ax.errorbar(s, y, yerr=[lo, hi], marker=MARKERS[i], color=PALETTE[i], capsize=1.5,
                    label=fr"$d={e['d']}$, $F={e['F']:,}$".replace(",", r"\,"))
    ax.axhline(0.95, color="k", ls=":", lw=0.9)
    ax.set_xlabel(r"sparsity $s$")
    ax.set_ylabel(r"$\Pr[\hat b = \mathbf{1}_S]$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(r"(a) recovery at $F=d^2$")
    ax.legend(loc="lower left", fontsize=6)

    ax = axes[1]
    ds = np.array([e["d"] for e in sorted(per_d, key=lambda e: e["d"])], dtype=float)
    s95 = np.array([e["s95"] for e in sorted(per_d, key=lambda e: e["d"])], dtype=float)
    ax.plot(ds, s95, marker="o", color=PALETTE[0], ls="none", label=r"measured $s_{95}(d)$")
    grid = np.linspace(ds.min() * 0.9, ds.max() * 1.1, 200)
    if fit:
        ax.plot(grid, fit["c"] * grid / np.log(grid), color=PALETTE[0], ls="-",
                label=fr"fit $c\,d/\ln d$, $c={fit['c']:.3f}$")
    # The d/(16 ln d) reference curve was withdrawn (decision D7): its constant is inherited
    # from the earlier literature and is derived nowhere in this work, and it sat above the
    # measured thresholds at every width, so it was not the conservative reference its form
    # suggests. Corollary 6.5 fixes only the shape, for an unspecified universal constant.
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"width $d$")
    ax.set_ylabel(r"$s_{95}(d)$")
    ax.set_title("(b) empirical recovery threshold")
    ax.legend(loc="upper left", fontsize=6.5)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig4_scaling")



# --------------------------------------------------------------------------------------
# Figure 5 -- the matched comparison, and the geometry the loss produces (E7 stage A)
# --------------------------------------------------------------------------------------

ARM_STYLE = {"trained": (PALETTE[0], MARKERS[0], "both trained"),
             "frozen": (PALETTE[3], MARKERS[3], "code frozen"),
             "random": (PALETTE[2], MARKERS[2], "neither trained")}
WIDTHS = (50, 100, 200)


def figure_matched_comparison() -> None:
    """Why the headline is a tie, and where the difference actually lives.

    Panels (a) and (b) are the same paired difference under the two pre-registered estimands, which
    is the point: they disagree. Panel (c) is the geometry, and it carries its own warning -- the
    untrained code sits as close to the floor as the trained one, so proximity to the floor is not
    evidence of learning.
    """
    prim = {(c["loss"], c["d"]): c for c in load("e7/derived/primary_comparison.json")["cells"]}
    per_width = load("e7/raw/e7_stage_A.json")["report"]["per_width"]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 2.5))

    for ax, estimand, ylab, title in (
            (axes[0], "s95", r"network $-$ probe, $s_{95}$", "(a) exact recovery"),
            (axes[1], "auc", r"network $-$ probe, recovery AUC", "(b) recovery AUC")):
        for loss in ("L4", "L2"):
            colour, marker, label = KIND_STYLE[loss]
            for stage, ls, alpha in (("post_relu", "-", 1.0), ("pre_relu", ":", 0.6)):
                xs, ys, lo, hi = [], [], [], []
                for d in WIDTHS:
                    v = prim[(loss, d)]["network_minus_probe"][stage][estimand]
                    xs.append(d); ys.append(v["mean"])
                    lo.append(v["mean"] - v["ci_low"]); hi.append(v["ci_high"] - v["mean"])
                ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker, ls=ls,
                            alpha=alpha, capsize=2, lw=1.0, ms=3.5,
                            label=f"{label}, {'post' if stage == 'post_relu' else 'pre'}-ReLU")
        ax.axhline(0.0, color="k", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(WIDTHS)); ax.set_xticklabels([str(d) for d in WIDTHS])
        ax.set_xlabel(r"width $d$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].legend(fontsize=5.4, loc="center left")
    # Both panels are signed the same way, so the annotation goes on the side that is actually
    # below zero. Putting it at the top would have labelled the tie as the probe winning.
    for ax in (axes[0], axes[1]):
        ax.text(0.97, 0.04, "probe better $\\downarrow$", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=5.6, style="italic")

    ax = axes[2]
    for kind in ("L2", "L4", "random"):
        colour, marker, label = KIND_STYLE[kind]
        ys = [per_width[str(d)][kind]["R_geom"] for d in WIDTHS]
        ax.plot(WIDTHS, ys, color=colour, marker=marker, ms=3.5, lw=1.0, label=label)
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(WIDTHS)); ax.set_xticklabels([str(d) for d in WIDTHS])
    ax.set_xlabel(r"width $d$")
    ax.set_ylabel(r"$R_{\mathrm{geom}}$")
    ax.set_title("(c) geometry, by loss")
    ax.legend(fontsize=6, loc="center right")

    fig.tight_layout(w_pad=1.3)
    save(fig, "fig5_matched_comparison")


# --------------------------------------------------------------------------------------
# Figure 6 -- which half of the training does the work (E8)
# --------------------------------------------------------------------------------------

def figure_frozen_encoder() -> None:
    """The frontier follows the code; the readout column is a trade-off, not a failure.

    Panel (b) exists because R_readout alone invites the wrong reading. The frozen arm sits at twice
    the code-specific cross-talk optimum, which looks like a failed optimisation until it is scored
    under the loss it actually trained on, where it beats that optimum outright.
    """
    runs = []
    for rel in ("e8/raw/e8_runs.json", "e8_d200/raw/e8_runs.json"):
        runs += load(rel)["runs"]
    trade = {(c["arm"], c["d"]): c
             for c in load("e8/derived/frozen_readout_tradeoff.json")["cells"]}

    def med(arm: str, d: int, fn) -> float:
        return float(np.median([fn(r) for r in runs if r["arm"] == arm and r["d"] == d]))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.5))

    ax = axes[0]
    for arm in ("trained", "frozen", "random"):
        colour, marker, label = ARM_STYLE[arm]
        ys = [med(arm, d, lambda r: r["theory"]["affine"]["kappa_min"]) for d in WIDTHS]
        ax.plot(WIDTHS, ys, color=colour, marker=marker, ms=3.5, lw=1.0, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(WIDTHS)); ax.set_xticklabels([str(d) for d in WIDTHS])
    ax.set_xlabel(r"width $d$")
    ax.set_ylabel(r"$\kappa_{\min}$")
    ax.set_title("(a) frontier: the code, not the readout")
    ax.legend(fontsize=6, loc="upper left")

    # A scatter, not two y-axes. The first version drew R_readout as bars against a twinned
    # task-gain line, and the two scales crossed where the eye reads a trade-off that is not there:
    # the jointly trained arm is better on *both* axes, and dominance is what a reader should see.
    ax = axes[1]
    for arm in ("trained", "frozen"):
        colour, marker, label = ARM_STYLE[arm]
        xs = [med(arm, d, lambda r: r["theory"]["analog"]["R_readout"]["wout"]) for d in WIDTHS]
        ys = [trade[(arm, d)]["task_gain_over_pinv"] for d in WIDTHS]
        ax.plot(xs, ys, color=colour, marker=marker, ms=4.5, lw=0.9, ls="--", label=label)
        if arm == "frozen":
            # Its three widths land on top of one another, and that is the finding rather than a
            # plotting nuisance: neither coordinate moves as the problem grows. One label for the
            # cluster says so; three overlapping labels would only have hidden it.
            ax.annotate(r"$d=50,100,200$", (float(np.mean(xs)), min(ys)),
                        textcoords="offset points", xytext=(-4, -13), ha="center",
                        fontsize=5.4, color=colour)
        else:
            for d, x, y in zip(WIDTHS, xs, ys):
                ax.annotate(f"{d}", (x, y), textcoords="offset points", xytext=(6, -2),
                            fontsize=5.4, color=colour)
    ax.axvline(1.0, color="k", ls=":", lw=0.9)
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xlim(0.85, 2.15)
    ax.set_ylim(0.0, 11.0)
    ax.set_xlabel(r"$R_{\mathrm{readout}}$ (1 = code-specific optimum)")
    ax.set_ylabel(r"task loss of $\Phi^{+}$ / of $W_{\mathrm{out}}$")
    ax.set_title("(b) the readout is a trade-off, not a failure")
    ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig6_frozen_encoder")


def main() -> None:
    setup()
    print("Generating figures from raw results:")
    figure_diagnostic_overview()
    figure_optimized_codes()
    figure_trained_model()
    figure_scaling()
    figure_matched_comparison()
    figure_frozen_encoder()
    print("done.")


if __name__ == "__main__":
    main()
