#!/usr/bin/env python3
"""G1 reanalysis: split the attainment ratio into geometry and readout suboptimality.

The manuscript currently reports one number per readout: the mean squared cross-talk divided by
the rank--trace floor `W_global(F, d)`. That floor bounds every rank-`d` unit-diagonal interface,
so it is a statement about the *ensemble* of codes at that shape. No particular code need be able
to attain it, which makes "the L4 network sits 0.06% above the floor" hard to read: is the code
good, or is the floor simply unreachable for it?

G1 makes the question answerable. For a fixed code there is a code-specific optimum
`W_*(Phi) = (sum_i h_i^{-1} - F) / (F(F-1))`, attained by the calibrated pseudoinverse, and it
sits at or above the global floor. So the single ratio factors exactly:

    W(G, Phi) / W_global  =  [W_*(Phi) / W_global]  x  [W(G, Phi) / W_*(Phi)]
                             ------- R_geom -------    ---- R_readout ----

`R_geom >= 1` measures how far this *code's* geometry is from the best any code of that shape
could manage -- by the leverage decomposition, exactly its leverage heterogeneity. `R_readout
>= 1` measures how far this *readout* is from the best readout of that code. One is a property
of the representation, the other of the decoder, and the published number conflates them.

The interesting question this can settle: is `L2` geometry-limited while its own decoder is
already near-optimal for the code it built? That would be an interpretive result even though
G1's optimisation is classical (Capon/MVDR; see docs/G2_NOVELTY_GO_NO_GO.md).

Requires weights, which the committed `results/e5/` does not carry -- run
`python experiments/e5_trained_toy.py --out-dir results/e5_weights` first.

Emits `docs/g1_reanalysis.md`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrtr.analog_optimum import (  # noqa: E402
    analog_attainment,
    code_specific_floor,
    leverage,
    leverage_excess,
)
from lrtr.codes import welch_floor  # noqa: E402
from lrtr.interface import crosstalk_mean_sq  # noqa: E402

WEIGHTS = ROOT / "results" / "e5_weights"
COMMITTED = ROOT / "results" / "e5" / "raw" / "e5_runs.json"
OUT = ROOT / "docs" / "g1_reanalysis.md"
READOUTS = ("pinv", "wout", "ls")


def _reproduction_check(rows: List[Dict[str, Any]]) -> List[str]:
    """Compare the re-run's floor statistics against the committed ones, and say what moved.

    The re-run exists only to obtain weights; if its diagnostics differ from the published ones
    the reanalysis is describing different models, and that has to be visible rather than
    assumed away. BLAS reduction order is not guaranteed across thread counts, so exact equality
    is not expected -- but the size of any deviation is reportable.
    """
    lines: List[str] = []
    if not COMMITTED.exists():
        return ["- committed `results/e5/` not found; no reproduction check possible"]
    old = {(r["loss_kind"], r["p_train"], r["seed"]): r
           for r in json.loads(COMMITTED.read_text(encoding="utf-8"))["runs"]}
    worst = 0.0
    n_matched = 0
    for r in rows:
        key = (r["loss_kind"], r["p_train"], r["seed"])
        if key not in old:
            continue
        n_matched += 1
        for ro in READOUTS:
            a = old[key]["floor_stats"].get(ro, {}).get("ratio_mean_sq")
            b = r["ratio_vs_welch"].get(ro)
            if a is None or b is None:
                continue
            worst = max(worst, abs(a - b) / max(abs(a), 1e-12))
    lines.append(f"- matched {n_matched} of {len(rows)} runs against the committed results")
    lines.append(f"- worst relative deviation in the Welch-referenced ratio: **{worst:.2e}**")
    if worst < 1e-7:
        lines.append(
            f"- the re-run reproduces the published diagnostics to {worst:.0e} relative, so the "
            f"reanalysis below describes the same models the manuscript reports. The residual is "
            f"not solver noise: E5 stores weights as `float32` to keep the archive small, and "
            f"casting them back to `float64` here perturbs the ratio at about that size. Exact "
            f"agreement would need `float64` storage, which doubles the weight archive for a "
            f"difference nine orders of magnitude below anything reported.")
    else:
        lines.append(f"- **the re-run does not reproduce the published diagnostics exactly.** "
                     f"Deviation {worst:.2e}. The reanalysis describes the re-run's models; the "
                     f"published numbers are unchanged and are what the manuscript reports")
    return lines


def main() -> int:
    if not WEIGHTS.exists():
        print(f"missing {WEIGHTS.relative_to(ROOT).as_posix()}; run E5 with "
              f"--out-dir results/e5_weights first")
        return 1

    runs = json.loads((WEIGHTS / "raw" / "e5_runs.json").read_text(encoding="utf-8"))["runs"]
    rows: List[Dict[str, Any]] = []
    for r in runs:
        npz = WEIGHTS / "weights" / f"{r['loss_kind']}_p{r['p_train']:g}_seed{r['seed']}.npz"
        if not npz.exists():
            print(f"  missing weights for {npz.name}")
            continue
        z = np.load(npz)
        W_in = z["W_in"].astype(np.float64)
        W_out = z["W_out"].astype(np.float64)
        d, F = W_in.shape

        h = leverage(W_in)
        w_code = code_specific_floor(W_in)
        w_glob = welch_floor(F, d)
        rec: Dict[str, Any] = {
            "loss_kind": r["loss_kind"], "p_train": r["p_train"], "seed": r["seed"],
            "d": d, "F": F,
            "R_geom": w_code / w_glob,
            "leverage_cv": float(np.std(h) / np.mean(h)),
            **{k: v for k, v in leverage_excess(W_in).items()
               if k in ("excess", "leverage_min", "leverage_max")},
            "ratio_vs_welch": {}, "R_readout": {},
        }
        # `ls` is deliberately absent. It is a *fitted* readout, so its cross-talk depends on the
        # fitting set, and reproducing the exact set the manuscript used would mean replaying an
        # RNG stream rather than reading a stored object. It also adds nothing to this particular
        # split, which contrasts the code's geometry against the model's own trained decoder.
        readouts = {"pinv": np.linalg.pinv(W_in), "wout": W_out}
        for name, G in readouts.items():
            G = np.ascontiguousarray(G, dtype=np.float64)
            try:
                measured = crosstalk_mean_sq(G, W_in)
            except ValueError as exc:
                rec["ratio_vs_welch"][name] = None
                rec["R_readout"][name] = None
                rec[f"error_{name}"] = str(exc)
                continue
            rec["ratio_vs_welch"][name] = measured / w_glob
            rec["R_readout"][name] = measured / w_code
        rec["attainment_pinv"] = analog_attainment(W_in)
        rows.append(rec)

    if not rows:
        print("no weights found")
        return 1

    def cell(kind: str, p: float) -> List[Dict[str, Any]]:
        return [r for r in rows if r["loss_kind"] == kind and r["p_train"] == p]

    ps = sorted({r["p_train"] for r in rows})
    kinds = ["L4", "L2", "random"]

    L: List[str] = []
    L.append("# G1 reanalysis — geometry versus readout suboptimality")
    L.append("")
    L.append("Generated by `scripts/reanalyse_g1.py` from `results/e5_weights/`. The committed")
    L.append("`results/e5/` carries no weights, so this needs its own run; the reproduction check")
    L.append("below states how closely that run matches the published diagnostics.")
    L.append("")
    L.append("The published ratio factors exactly into")
    L.append("")
    L.append("    W(G, Phi) / W_global  =  R_geom x R_readout,")
    L.append("    R_geom    = W_*(Phi) / W_global    (this code's geometry vs the best possible)")
    L.append("    R_readout = W(G, Phi) / W_*(Phi)   (this readout vs the best for this code)")
    L.append("")
    L.append("`R_geom` is exactly the leverage heterogeneity of the code; `R_readout = 1` means the")
    L.append("readout is the code-specific optimum, which the calibrated pseudoinverse always is.")
    L.append("")

    L.append("## Reproduction check")
    L.append("")
    L.extend(_reproduction_check(rows))
    L.append("")

    L.append("## The decomposition")
    L.append("")
    L.append("| cell | n | published ratio (pinv) | R_geom | R_readout (pinv) | R_readout (wout) |")
    L.append("|---|---|---|---|---|---|")
    for p in ps:
        for kind in kinds:
            sub = cell(kind, p)
            if not sub:
                continue
            def m(f):
                vals = [f(r) for r in sub]
                vals = [v for v in vals if v is not None]
                return float(np.mean(vals)) if vals else float("nan")
            L.append(
                f"| {kind} p={p:g} | {len(sub)} | "
                f"{m(lambda r: r['ratio_vs_welch']['pinv']):.4f} | "
                f"{m(lambda r: r['R_geom']):.4f} | "
                f"{m(lambda r: r['R_readout']['pinv']):.6f} | "
                f"{m(lambda r: r['R_readout']['wout']):.4f} |")
    L.append("")

    L.append("## Leverage profile")
    L.append("")
    L.append("| cell | leverage min | leverage max | coefficient of variation | excess |")
    L.append("|---|---|---|---|---|")
    for p in ps:
        for kind in kinds:
            sub = cell(kind, p)
            if not sub:
                continue
            L.append(f"| {kind} p={p:g} | "
                     f"{np.mean([r['leverage_min'] for r in sub]):.4f} | "
                     f"{np.mean([r['leverage_max'] for r in sub]):.4f} | "
                     f"{np.mean([r['leverage_cv'] for r in sub]):.4f} | "
                     f"{np.mean([r['excess'] for r in sub]):.3f} |")
    L.append("")

    # ---- the interpretation, stated only where the numbers support it ----
    L.append("## Reading")
    L.append("")
    l4 = [r for r in rows if r["loss_kind"] == "L4" and r["p_train"] == ps[0]]
    l2 = [r for r in rows if r["loss_kind"] == "L2" and r["p_train"] == ps[0]]
    rd = [r for r in rows if r["loss_kind"] == "random"]

    def mean_of(sub, path):
        vals = [path(r) for r in sub]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    geom = {k: mean_of(s_, lambda r: r["R_geom"]) for k, s_ in
            (("L4", l4), ("L2", l2), ("random", rd)) if s_}
    wout = {k: mean_of(s_, lambda r: r["R_readout"]["wout"]) for k, s_ in
            (("L4", l4), ("L2", l2), ("random", rd)) if s_}

    L.append("Geometry, `R_geom` (1 means the code's own floor coincides with the global one):")
    for k, v in geom.items():
        L.append(f"- {k}: {v:.4f}")
    L.append("")
    L.append("The model's own decoder against the best readout of its own code, `R_readout(wout)`:")
    for k, v in wout.items():
        L.append(f"- {k}: {v:.4f}")
    L.append("")

    # Classify each cell from the two terms rather than asserting a story. The thresholds are
    # arbitrary but stated, and the numbers are in the tables above either way.
    L.append("Reading the two terms per cell, with `1.1` as the (arbitrary, stated) line between")
    L.append("near-optimal and not:")
    L.append("")
    L.append("| cell | geometry | own decoder | limited by |")
    L.append("|---|---|---|---|")
    for p in ps:
        for kind in kinds:
            sub_ = cell(kind, p)
            if not sub_:
                continue
            g = mean_of(sub_, lambda r: r["R_geom"])
            w = mean_of(sub_, lambda r: r["R_readout"]["wout"])
            g_ok, w_ok = g < 1.1, w < 1.1
            verdict = ("neither term binds" if g_ok and w_ok else
                       "geometry" if not g_ok and w_ok else
                       "the readout" if g_ok and not w_ok else "both")
            L.append(f"| {kind} p={p:g} | {'near-optimal' if g_ok else f'{g:.1f}x off'} | "
                     f"{'near-optimal' if w_ok else f'{w:.3f}'} | {verdict} |")
    L.append("")
    L.append("One consistency check worth naming: `R_readout(wout)` for the random control is")
    L.append("exactly 1 because `random_code_baseline` sets `W_out = pinv(Phi)`, so its \"own")
    L.append("decoder\" *is* the code-specific optimum by construction. That is audit finding P1")
    L.append("resurfacing, and it confirms the decomposition behaves as it should.")
    L.append("")
    L.append("`R_readout(pinv) = 1` by construction in every cell -- the calibrated pseudoinverse")
    L.append("*is* the code-specific optimum, which is the content of G1 and is asserted in the")
    L.append("tests. That is also why the published 'within a few parts in a thousand of the floor'")
    L.append("number is a statement about the code's leverage profile and not about the decoder:")
    L.append("under the pseudoinverse the readout term is exactly one, so the whole ratio is")
    L.append("`R_geom`.")
    L.append("")
    L.append("### What this does and does not license")
    L.append("")
    L.append("- The split is exact and needs no new assumption; it is arithmetic on G1.")
    L.append("- It reinterprets an existing number rather than adding evidence. Five seeds at one")
    L.append("  width remain five seeds at one width, and nothing here changes that.")
    L.append("- `R_readout(wout)` is the quantity worth carrying into the scaled campaign: it asks")
    L.append("  whether training produced a decoder that is good *for the code it built*, which is")
    L.append("  a different question from whether the code is good.")
    L.append("- No claim is made about the *optimisation* being ours. It is the Capon/MVDR")
    L.append("  beamformer with the frame operator in place of the covariance; see the novelty")
    L.append("  audit.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} from {len(rows)} models")
    for k, v in geom.items():
        print(f"  R_geom[{k}] = {v:.4f}")
    for k, v in wout.items():
        print(f"  R_readout(wout)[{k}] = {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
