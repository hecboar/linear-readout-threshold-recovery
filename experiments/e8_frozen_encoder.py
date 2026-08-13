#!/usr/bin/env python3
"""E8 -- what does training the encoder contribute? A frozen random code with a trained readout.

The audit's sharpest scientific objection: an untrained random code already sits at `R_geom` ~1.01
and supports affine support recovery to within one or two sparsity levels of the `L4`-trained code,
so "training finds a good interface" may be claiming credit for something generic. The existing
controls cannot settle it, because they differ from the trained models in *two* ways at once -- the
code is random **and** the readout is `pinv(Phi)` rather than learned.

This separates them. Three arms, identical task, data, steps and budget:

* `frozen`  -- `W_in` is a random unit-norm code, **frozen**; only `W_out` trains.
* `trained` -- both train. This is the `L4` arm of Stage A, retrained here so the comparison is
  within one script and one seed convention.
* `random`  -- neither trains; `W_out = pinv(W_in)`. The Stage A control, for continuity.

What each comparison isolates:

    trained vs frozen  ->  what training the CODE contributes, readout held learnable
    frozen  vs random  ->  what training the READOUT contributes, code held random

Only `W_out` carries gradient in the frozen arm, so it is far cheaper than a full run, and the
frozen code's geometry is fixed by construction -- `R_geom` cannot move, which is precisely what
makes the arm informative about the code.

Reported per model: the analog decomposition, the frontier over a feature subset, and the
diagnosis `e7_scaled_toy` computes, so the numbers sit beside Stage A's without re-derivation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _common import RunRecord, base_parser, log, map_trials, prepare, write_json

from lrtr.codes import random_unit_code

ARMS = ("trained", "frozen", "random")


def train_arm(d: int, F: int, arm: str, seed: int, steps: int, batch: int, lr: float,
              p: float, device: str) -> Dict[str, np.ndarray]:
    """One model. `frozen` puts no gradient on `W_in`; `random` trains nothing at all."""
    import torch

    from lrtr.toymodel import sample_task_batch, target_of

    dev = torch.device(device)
    gen = torch.Generator().manual_seed(seed)                 # CPU stream: see D13
    W_in = (torch.randn(d, F, generator=gen, dtype=torch.float32) / np.sqrt(F))
    if arm == "random":
        Phi = W_in.numpy().astype(np.float64)
        cols = np.linalg.norm(Phi, axis=0, keepdims=True)
        Phi = Phi / np.maximum(cols, 1e-12) * float(cols.mean())
        return {"W_in": Phi, "W_out": np.linalg.pinv(Phi)}

    W_out = (torch.randn(F, d, generator=gen, dtype=torch.float32) / np.sqrt(d))
    W_in, W_out = W_in.to(dev), W_out.to(dev)
    W_in.requires_grad_(arm == "trained")
    W_out.requires_grad_(True)
    params = ([W_in, W_out] if arm == "trained" else [W_out])
    opt = torch.optim.Adam(params, lr=lr)
    dgen = torch.Generator(device=dev).manual_seed(seed) if dev.type != "cpu" else gen
    for _ in range(steps):
        x = sample_task_batch(F, batch, p, dgen, torch.float32).to(dev)
        pred = torch.relu(x @ W_in.T) @ W_out.T
        loss = ((pred - target_of(x, "relu")) ** 4).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {"W_in": W_in.detach().cpu().numpy().astype(np.float64),
            "W_out": W_out.detach().cpu().numpy().astype(np.float64)}


def diagnose_arm(W: Dict[str, np.ndarray], cfg: Dict[str, Any], sparsities: List[int],
                 seed: int) -> Dict[str, Any]:
    import e7_scaled_toy as E

    model = {"W_in": W["W_in"], "W_out": W["W_out"], "loss_kind": "L4",
             "seed": seed, "p": cfg["train_sparsities"][0], "task": "relu"}
    return E.diagnose(model, cfg, sparsities, seed=200_000 + 997 * seed)


def main() -> None:
    parser = base_parser("e8")
    parser.add_argument("--widths", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    cfg, out_dir, threads = prepare("e8", args)

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    steps, batch, lr = cfg["steps"], cfg["batch"], cfg["lr"]
    p = cfg["train_sparsities"][0] if "train_sparsities" in cfg else 0.01
    rows: List[Dict[str, Any]] = []
    with RunRecord("e8_frozen_encoder", out_dir, config=cfg, smoke=bool(args.smoke)) as rec:
      for d in args.widths:
          F = 2 * d
          sparsities = list(range(1, int(cfg["eval_s_max"][str(d)]) + 1))
          for arm in ARMS:
              weights = [train_arm(d, F, arm, s, steps, batch, lr, p, args.device)
                         for s in range(args.seeds)]
              diags = map_trials(diagnose_arm,
                                 [(w, cfg, sparsities, s) for s, w in enumerate(weights)],
                                 workers=args.workers,
                                 threads_per_worker=max(1, threads // max(1, args.workers or 1)))
              diags = [x for x in diags if x]
              for x in diags:
                  x["arm"], x["d"], x["F"] = arm, d, F
              rows.extend(diags)
              geom = [x["theory"]["analog"]["R_geom"] for x in diags]
              kap = [x["theory"]["affine"]["kappa_min"] for x in diags]
              log(f"  [{arm:8s} d={d}] R_geom={np.mean(geom):.4f}  "
                  f"kappa_min={np.mean(kap):.3f}  n={len(diags)}")
              (out_dir / "weights").mkdir(parents=True, exist_ok=True)
              np.savez_compressed(out_dir / "weights" / f"{arm}_d{d}.npz",
                                  W_in=np.stack([w["W_in"] for w in weights]),
                                  W_out=np.stack([w["W_out"] for w in weights]))

      write_json(out_dir / "raw" / "e8_runs.json", {"runs": rows})
      rec.payload["n_models"] = len(rows)


if __name__ == "__main__":
    main()
