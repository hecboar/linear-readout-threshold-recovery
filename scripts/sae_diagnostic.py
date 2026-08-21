"""The Interface Diagnostic on released SAE dictionaries, along four axes.

Every number comes from a decoder matrix. No model weights, no activations, no training, no linear
programme. welch_floor, code_specific_floor, leverage and affine_failure_threshold are the
repository's own functions -- the same ones Section 10.5 reports.

    scale         Qwen-Scope, d = 2048 -> 4096 at fixed load 16x and fixed recipe
    load          Pythia-160m k=64, widths 768..131k at layers.6.mlp: 1x to 171x, d and k fixed
    causal        SmolLM2-135M 64x, TRAINED vs RANDOMLY INITIALISED model, 10 matched layers.
                  This is the one axis that separates "the dictionary inherits the geometry of what
                  the model learned" from "the geometry is an artefact of training a TopK SAE".
    architecture  LFM2.5 MLP down-projection: a hybrid convolution/attention model, not a standard
                  transformer. Its features are neurons, not monosemantic units -- stated as such.

Each dictionary is compared against an i.i.d. Gaussian code of ITS OWN shape at ITS OWN operating
sparsity. That comparison is the only meaningful one: by the trace identity sum(h) = d the mean
leverage of every code is exactly d/F, so absolute fractions ruled out are a property of the shape.

safetensors is read directly rather than with the library: an 8-byte little-endian header length, a
JSON header, then raw tensor bytes. Fifteen lines, and it avoids adding a dependency to the
environment that produced the published results.
"""
import glob
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "lrtr" / "repo"
sys.path.insert(0, str(ROOT / "src"))
from lrtr.affine_frontier import affine_failure_threshold  # noqa: E402
from lrtr.analog_optimum import code_specific_floor, leverage  # noqa: E402
from lrtr.codes import welch_floor  # noqa: E402

DIR = Path.home() / "lrtr" / "sae3"
QWEN_OLD = {"Qwen3.5-2B": Path.home() / "lrtr" / "qwen_sae",
            "Qwen3-1.7B": Path.home() / "lrtr" / "qwen_sae_17b"}
_DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": "bf16"}


def read_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
        blob = fh.read()
    out = {}
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        dt = _DT[meta["dtype"]]
        if dt == "bf16":                     # numpy has no bfloat16: widen via the exponent bits
            u = np.frombuffer(blob[a:b], dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        else:
            arr = np.frombuffer(blob[a:b], dtype=dt)
        out[name] = arr.reshape(meta["shape"])
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def unit_cols(M):
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)


def as_code(W):
    return np.ascontiguousarray(W if W.shape[0] < W.shape[1] else W.T, dtype=np.float64)


def decoder_from_safetensors(path):
    t = read_safetensors(path)
    name = next((k for k in t if "dec" in k.lower() and t[k].ndim == 2), None)
    if name is None:
        raise SystemExit(f"{path.name}: no 2-D decoder among {list(t)}")
    return as_code(t[name])


def decoder_from_pt(path):
    import torch
    return as_code(torch.load(path, map_location="cpu",
                              weights_only=True)["W_dec"].to(torch.float64).numpy())


def lfm_codes(path):
    """Phi = the MLP down-projection of each LFM2.5 layer: (d_model, d_ff), an overcomplete code.

    Its columns are neurons, not the monosemantic units the superposition literature means, and that
    is stated wherever the row is used. What it buys is an architecture axis no SAE suite covers:
    LFM2.5 is a hybrid, most of its layers being convolution blocks rather than attention, so the
    block type is recorded per layer and can be compared within one model.
    """
    t = read_safetensors(path)
    attn = {int(m.group(1)) for k in t
            if (m := re.match(r"model\.layers\.(\d+)\.self_attn\.", k))}
    out = []
    for k in sorted(t, key=lambda x: (len(x), x)):
        m = re.match(r"model\.layers\.(\d+)\.feed_forward\.w2\.weight$", k)
        if m:
            L = int(m.group(1))
            out.append((L, "attn" if L in attn else "conv", as_code(t[k])))
    return sorted(out)


def first_failure_s(h_min, F, s_max=32768):
    for s in range(2, s_max + 1):
        if h_min <= affine_failure_threshold(F, s):
            return s
    return None


def validated(Phi, label, expected_d):
    """Reject a dictionary whose shape does not match the model it claims to come from.

    Checking only `F > d` is not enough, and this is not hypothetical: the repository
    Pythia-160m-SAE-k64-131k ships a 200 KB file whose decoder is (64, 768) rather than
    (131072, 768). Oriented so that the smaller axis is `d`, that reads as a perfectly ordinary
    12x overcomplete code and yields a plausible R_geom of 1.0114 -- a row for a dictionary that
    does not exist. The model's hidden size is known independently of the file, so it is the check
    that catches a broken artefact instead of measuring it.
    """
    d, F = Phi.shape
    if d != expected_d:
        print(f"  DESCARTADO {label}: d={d} pero el modelo tiene d={expected_d} "
              f"(forma leida {d}x{F}) -- artefacto roto o mal etiquetado", flush=True)
        return False
    if F <= d:
        print(f"  DESCARTADO {label}: F={F} <= d={d}, el codigo no es sobrecompleto", flush=True)
        return False
    return True


def measure(Phi, label, axis, s_op, kind, rows):
    d, F = Phi.shape
    h = leverage(Phi)
    thr = affine_failure_threshold(F, s_op)
    # R_geom = d(sum 1/h - F) / (F(F-d)). Since sum(h) = d, Cauchy-Schwarz gives
    # sum 1/h >= F^2/d with equality iff h is constant, and substituting the minimum returns exactly
    # 1. So R_geom - 1 IS a measure of leverage dispersion, not merely correlated with one, and it is
    # driven by sum 1/h -- dominated by the LOWER tail of h, not by the variance. `identity_gap`
    # checks that reading against the implementation to machine precision; the quantiles are there to
    # separate "the spread grew" from "the lower tail grew", which the cv alone cannot do.
    inv = float(np.sum(1.0 / h))
    identity = d * (inv - F) / (F * (F - d))
    r_geom = code_specific_floor(Phi) / welch_floor(F, d)
    row = {"label": label, "axis": axis, "kind": kind, "d": d, "F": F, "load": F / d,
           "s_operating": s_op, "R_geom": r_geom,
           "R_geom_from_identity": identity, "identity_gap": abs(identity - r_geom),
           "h_harmonic_mean": float(F / inv),
           "h_q001": float(np.quantile(h, 0.001)), "h_q01": float(np.quantile(h, 0.01)),
           "h_q05": float(np.quantile(h, 0.05)), "h_q50": float(np.quantile(h, 0.50)),
           "h_min": float(h.min()), "h_mean": float(h.mean()), "h_max": float(h.max()),
           "h_cv": float(h.std() / h.mean()),
           "fraction_ruled_out_at_s_op": float((h <= thr).mean()),
           "first_failure_s": first_failure_s(float(h.min()), F)}
    rows.append(row)
    print(f"  {label:<26} {axis:<12} {kind:<10} d={d:<5} F={F:<7} {F/d:6.1f}x s={s_op:<4} "
          f"R_geom={row['R_geom']:.4f} cv={row['h_cv']:.3f} 1er_s={row['first_failure_s']}",
          flush=True)
    return row


JOBS = []   # (label, axis, path, loader, s_operating, expected_d)
for fam, d in QWEN_OLD.items():
    for p in sorted(glob.glob(str(d / "layer*.sae.pt")),
                    key=lambda s: int(re.search(r"layer(\d+)", s).group(1))):
        JOBS.append((f"{fam} {Path(p).stem}", "scale", Path(p), decoder_from_pt, 50, 2048))
for p in sorted(glob.glob(str(DIR / "qwen3_8b_L*.pt")),
                key=lambda s: int(re.search(r"L(\d+)", s).group(1))):
    JOBS.append((f"Qwen3-8B {Path(p).stem[-3:]}", "scale", Path(p), decoder_from_pt, 50, 4096))
for p in sorted(glob.glob(str(DIR / "pythia160m_k64_*.safetensors"))):
    JOBS.append((f"Pythia-160m {Path(p).stem.split('_')[-1]}", "load", Path(p),
                 decoder_from_safetensors, 64, 768))
for arm in ("trained", "random"):
    for p in sorted(glob.glob(str(DIR / f"smollm2_{arm}_L*.safetensors")),
                    key=lambda s: int(re.search(r"L(\d+)", s).group(1))):
        JOBS.append((f"SmolLM2 {arm} {Path(p).stem.split('_')[-1]}", "causal", Path(p),
                     decoder_from_safetensors, 64, 576))

LFM = DIR / "lfm25_350m.safetensors"

rows, prov, controls, rng = [], {}, {}, np.random.default_rng(0)
for label, axis, path, loader, s_op, exp_d in JOBS:
    if not path.exists():
        print(f"  (falta {path.name})", flush=True)
        continue
    Phi = unit_cols(loader(path))
    if not validated(Phi, label, exp_d):
        continue
    measure(Phi, label, axis, s_op, "dictionary", rows)
    prov[label] = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    key = (Phi.shape, s_op)
    if key not in controls:
        controls[key] = measure(unit_cols(rng.standard_normal(Phi.shape)),
                                f"i.i.d. {Phi.shape[0]}x{Phi.shape[1]}", axis, s_op, "control", rows)

if LFM.exists():
    prov["LFM2.5-350M"] = {"file": LFM.name, "sha256": sha256(LFM),
                           "bytes": LFM.stat().st_size}
    for L, block, W in lfm_codes(LFM):
        Phi = unit_cols(W)
        if not validated(Phi, f"LFM2.5-350M L{L}", 1024):
            continue
        measure(Phi, f"LFM2.5-350M L{L} ({block})", "architecture", 64, "dictionary", rows)
        key = (Phi.shape, 64)
        if key not in controls:
            controls[key] = measure(unit_cols(rng.standard_normal(Phi.shape)),
                                    f"i.i.d. {Phi.shape[0]}x{Phi.shape[1]}", "architecture",
                                    64, "control", rows)
else:
    print(f"  (falta {LFM.name}: eje de arquitectura omitido)", flush=True)

commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or None
out = Path.home() / "lrtr" / "sae_axes.json"
out.write_text(json.dumps({"lrtr_commit": commit, "provenance": prov, "rows": rows}, indent=2))
print(f"\n-> {out}  ({len(rows)} filas)")
