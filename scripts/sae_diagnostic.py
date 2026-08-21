"""The Interface Diagnostic on released SAE dictionaries, across labs, loads and depths.

Every number comes from a decoder matrix: no model weights, no activations, no training, no linear
programme. The functions are the repository's own -- welch_floor, code_specific_floor, leverage,
affine_failure_threshold -- so this is the same diagnostic Section 10.5 reports, applied to
third-party artefacts.

Three suites, three labs, three SAE training recipes:
  Qwen-Scope   TopK,      d = 2048, F = 32768  (load 16x), operating sparsity 50
  Gemma Scope  JumpReLU,  d = 2304, F = 16k-131k (load 7x-57x), operating sparsity from the filename

Each dictionary is compared against an i.i.d. Gaussian code of ITS OWN shape, evaluated at ITS OWN
published operating sparsity. That matters: by the trace identity sum(h) = d the mean leverage of any
code is d/F, so the failure corollary rules out the average feature of every code once s grows. Any
claim about a dictionary has to be a claim about its departure from that control, not about the
absolute fraction ruled out.
"""
import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "lrtr" / "repo"
sys.path.insert(0, str(ROOT / "src"))
from lrtr.affine_frontier import affine_failure_threshold  # noqa: E402
from lrtr.analog_optimum import code_specific_floor, leverage  # noqa: E402
from lrtr.codes import welch_floor  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_cols(M):
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)


def as_code(W):
    """Orient a decoder as (d, F) with d < F."""
    return W if W.shape[0] < W.shape[1] else W.T


def load_qwen(path):
    import torch
    return as_code(torch.load(path, map_location="cpu",
                              weights_only=True)["W_dec"].to(torch.float64).numpy())


def load_gemma(path):
    z = np.load(path)
    key = next(k for k in z.files if k.lower() in ("w_dec", "wdec"))
    return as_code(z[key].astype(np.float64))


def first_failure_s(h_min, F, s_max=16384):
    for s in range(2, s_max + 1):
        if h_min <= affine_failure_threshold(F, s):
            return s
    return None


def measure(Phi, label, s_op, kind):
    d, F = Phi.shape
    h = leverage(Phi)
    thr = affine_failure_threshold(F, s_op)
    row = {"label": label, "kind": kind, "d": d, "F": F, "load": F / d, "s_operating": s_op,
           "R_geom": code_specific_floor(Phi) / welch_floor(F, d),
           "h_min": float(h.min()), "h_mean": float(h.mean()), "h_max": float(h.max()),
           "h_cv": float(h.std() / h.mean()),
           "failure_threshold_at_s_op": float(thr),
           "fraction_ruled_out_at_s_op": float((h <= thr).mean()),
           "first_failure_s": first_failure_s(float(h.min()), F)}
    print(f"  {label:<30} {kind:<11} d={d:<5} F={F:<7} load={F/d:5.1f}x s_op={s_op:<4} "
          f"R_geom={row['R_geom']:.4f} h_cv={row['h_cv']:.3f} "
          f"1er_s={row['first_failure_s']:<5} descartadas={100*row['fraction_ruled_out_at_s_op']:5.1f}%",
          flush=True)
    return row


JOBS = []
for d, s_op in ((Path.home() / "lrtr" / "qwen_sae", 50),
                (Path.home() / "lrtr" / "qwen_sae_17b", 50)):
    fam = "Qwen3.5-2B" if d.name == "qwen_sae" else "Qwen3-1.7B"
    for p in sorted(glob.glob(str(d / "layer*.sae.pt")),
                    key=lambda s: int(re.search(r"layer(\d+)", s).group(1))):
        JOBS.append((f"{fam} {Path(p).stem}", Path(p), load_qwen, s_op))
for p in sorted(glob.glob(str(Path.home() / "lrtr" / "gemma_scope" / "*.npz"))):
    m = re.match(r"L(\d+)_w(\w+?)_l0(\d+)", Path(p).stem)
    JOBS.append((f"Gemma2-2B L{m.group(1)} w{m.group(2)}", Path(p), load_gemma, int(m.group(3))))

rows, prov, rng = [], {}, np.random.default_rng(0)
seen_controls = {}
for label, path, loader, s_op in JOBS:
    Phi = unit_cols(loader(path))
    rows.append(measure(Phi, label, s_op, "dictionary"))
    prov[label] = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    key = (Phi.shape, s_op)
    if key not in seen_controls:
        G = unit_cols(rng.standard_normal(Phi.shape))
        seen_controls[key] = measure(G, f"i.i.d. control {Phi.shape[0]}x{Phi.shape[1]}",
                                    s_op, "control")
        rows.append(seen_controls[key])

commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or None
out = Path.home() / "lrtr" / "sae_grid.json"
out.write_text(json.dumps({"lrtr_commit": commit,
                           "suites": {"Qwen-Scope": "Qwen/SAE-Res-Qwen3{.5}-*-Base-W32K-L0_50",
                                      "Gemma Scope": "google/gemma-scope-2b-pt-res"},
                           "provenance": prov, "rows": rows}, indent=2))
print(f"\n-> {out}  ({len(rows)} filas)")
