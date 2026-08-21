"""The Interface Diagnostic, unmodified, on released sparse-autoencoder dictionaries.

Two questions, both answered from the decoder matrix alone -- no model weights, no activations, no
training, and no linear programme.

1. Is near-floor geometry generic at a feature load four times anything the campaigns reached, and
   is a *trained* dictionary near the floor? R_geom against an i.i.d. code of identical shape.

2. Does the repaired failure corollary bite at the dictionary's own published operating sparsity?
   `cor:failthresh` states h_i <= min{1/2, (s-1)^2/((F-1)+(s-1)^2)} implies feature i is not
   affinely separable at sparsity s, in the support-level worst case over amplitudes. Note the
   trace identity sum(h) = d: the MEAN leverage of any code is d/F, so at high load the corollary
   rules out the average feature of every code once s grows -- which is a statement about the
   regime, not about any particular dictionary, and has to be reported as such.

Nothing is reimplemented: welch_floor, code_specific_floor, leverage and affine_failure_threshold
are the repository's own functions, the same ones Section 10.5 draws on.

Shapes are read from each checkpoint rather than assumed, so this runs unchanged on any Qwen-Scope
dictionary.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.home() / "lrtr" / "repo"
sys.path.insert(0, str(ROOT / "src"))
from lrtr.affine_frontier import affine_failure_threshold  # noqa: E402
from lrtr.analog_optimum import code_specific_floor, leverage  # noqa: E402
from lrtr.codes import welch_floor  # noqa: E402

OPERATING = (50, 100)
# Provenance: D8 requires every result to name what produced it, and a downloaded artefact needs the
# repository and revision as much as a run needs a commit. The dictionaries are third-party files, so
# the sha256 of each is recorded: it is the only thing that lets a reader confirm they analysed the
# same weights.
SOURCES = {
    "Qwen3.5-2B": {"repo": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
                   "dir": Path.home() / "lrtr" / "qwen_sae"},
    "Qwen3-1.7B": {"repo": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
                   "dir": Path.home() / "lrtr" / "qwen_sae_17b"},
}


def unit_cols(M):
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)


def first_failure_s(h_min, F, s_max=8192):
    for s in range(2, s_max + 1):
        if h_min <= affine_failure_threshold(F, s):
            return s
    return None


def measure(Phi, label, out):
    d, F = Phi.shape
    h = leverage(Phi)
    w_glob, w_code = welch_floor(F, d), code_specific_floor(Phi)
    row = {"label": label, "d": d, "F": F, "load": F / d,
           "R_geom": w_code / w_glob,
           "h_min": float(h.min()), "h_max": float(h.max()),
           "h_mean": float(h.mean()), "h_cv": float(h.std() / h.mean()),
           "first_failure_s": first_failure_s(float(h.min()), F),
           "ruled_out": {str(s): float((h <= affine_failure_threshold(F, s)).mean())
                         for s in OPERATING}}
    out.append(row)
    print(f"  {label:<28} d={d:<5} F={F:<6} load={F/d:.0f}x  R_geom={row['R_geom']:.4f}  "
          f"h: min={row['h_min']:.4f} mean={row['h_mean']:.4f} max={row['h_max']:.4f} "
          f"cv={row['h_cv']:.3f}  1er s descartado={row['first_failure_s']}  "
          + "  ".join(f"s={s}: {100*row['ruled_out'][str(s)]:.1f}%" for s in OPERATING), flush=True)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows, prov = [], {}
for model, src in SOURCES.items():
    d = src["dir"]
    paths = sorted(glob.glob(str(d / "layer*.sae.pt")),
                   key=lambda s: int(re.search(r"layer(\d+)", s).group(1)))
    if not paths:
        print(f"\n### {model}: sin checkpoints, omitido", flush=True)
        continue
    print(f"\n### {model}", flush=True)
    prov[model] = {"repo": src["repo"], "revision": "main",
                   "files": {Path(x).name: {"sha256": sha256(x),
                                            "bytes": Path(x).stat().st_size}
                             for x in paths}}
    shape = None
    for p in paths:
        W = torch.load(p, map_location="cpu", weights_only=True)["W_dec"].to(torch.float64).numpy()
        Phi = W if W.shape[0] < W.shape[1] else W.T          # orient as (d, F), d < F
        shape = Phi.shape
        measure(unit_cols(Phi), f"{model} {Path(p).stem}", rows)
    rng = np.random.default_rng(0)
    measure(unit_cols(rng.standard_normal(shape)), f"{model} i.i.d. control", rows)

out = Path.home() / "lrtr" / "sae_diagnostic.json"
import subprocess
commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or None
out.write_text(json.dumps({"lrtr_commit": commit, "sources": prov,
                           "operating_sparsities": list(OPERATING), "rows": rows}, indent=2))
print(f"\n-> {out}")
