#!/usr/bin/env python3
"""J0 -- measure what the campaign actually costs on this machine, before committing to it.

Every wall-clock number in the plan is currently an estimate. This replaces all of them with
measurements, and it is the gate the full grid waits behind: the batching strategy, the seed
count and the ordering of the experiment matrix are all decided from its output rather than
guessed. The networks here are small, so the risk is not that the accelerator is too slow but
that it is idle, and a throughput sweep over the seed-batch dimension is the only way to find
the point where that stops being true.

Two cost centres are measured, because the campaign has two:

* **training**, which goes on the accelerator -- swept over seed-batch `B` and width;
* **the robust affine frontier**, which is convex-solver work and stays on the CPU. It is a
  budget line of its own (plan §8) and it is measured here rather than assumed, now that the
  solver exists and is validated.

Outputs
    results/benchmark/spark_benchmark.json
    docs/spark_capacity_report.md
    configs/spark/selected_batching.yaml

Run on the accelerator with ``--device cuda``. On CPU it still runs and still produces a valid
report; the numbers are simply CPU numbers, which is useful as a reference point but is not
what the campaign will be launched from.
"""
from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from _common import REPO_ROOT, RunRecord, base_parser, log, prepare, write_json

from lrtr.affine_frontier import robust_affine_frontier
from lrtr.codes import random_unit_code
from lrtr.runlog import git_provenance
from lrtr.toymodel import train_toy_models_batched

SHAPES = [(50, 100), (100, 200), (200, 400), (400, 800)]
BATCHES = [1, 5, 10, 20]


class DeviceSampler:
    """Poll `nvidia-smi` in the background for utilisation and memory.

    Absent or unusable `nvidia-smi` is not an error: the sampler simply reports nothing, and
    the benchmark still produces timings. Reporting "no samples" is better than reporting a
    fabricated utilisation.
    """

    def __init__(self, period: float = 0.25) -> None:
        self.period, self.util, self.mem = period, [], []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    u, m = r.stdout.strip().splitlines()[0].split(",")
                    self.util.append(float(u))
                    self.mem.append(float(m))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, Any]:
        if not self.util:
            return {"samples": 0, "gpu_util_mean": None, "gpu_util_max": None,
                    "gpu_mem_max_mib": None}
        return {"samples": len(self.util),
                "gpu_util_mean": float(np.mean(self.util)),
                "gpu_util_max": float(np.max(self.util)),
                "gpu_mem_max_mib": float(np.max(self.mem))}


def time_training(d: int, F: int, B: int, steps: int, batch: int, device: str,
                  lr: float) -> Dict[str, Any]:
    """One throughput point. Separates the first-step cost from the steady-state rate."""
    import torch

    seeds = list(range(B))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warm-up: allocator, autotuning and any PTX JIT land here, not in the measured window.
    t_warm = time.perf_counter()
    train_toy_models_batched(d=d, F=F, loss_kind="L4", p=0.01, seeds=seeds, steps=2,
                             batch=batch, lr=lr, device=device, log_every=10 ** 9)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_warm

    with DeviceSampler() as sampler:
        t0 = time.perf_counter()
        train_toy_models_batched(d=d, F=F, loss_kind="L4", p=0.01, seeds=seeds, steps=steps,
                                 batch=batch, lr=lr, device=device, log_every=10 ** 9)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    rec = {
        "d": d, "F": F, "seed_batch": B, "steps": steps, "batch": batch, "device": device,
        "warmup_s": warmup_s,
        "wall_s": elapsed,
        "steps_per_s": steps / elapsed,
        "models_steps_per_s": B * steps / elapsed,
        "samples_per_s": B * steps * batch / elapsed,
        "s_per_model_50k": elapsed / steps * 50_000 / B,
        **sampler.summary(),
    }
    if device.startswith("cuda"):
        rec["torch_peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return rec


def time_affine_frontier(d: int, F: int, n_features: int, sparsities: List[int],
                         seed: int) -> Dict[str, Any]:
    """Solver cost of the exact frontier, per feature. This is the CPU-side budget line."""
    Phi = random_unit_code(d, F, np.random.default_rng(seed))
    feats = list(range(min(n_features, F)))
    t0 = time.perf_counter()
    out = robust_affine_frontier(Phi, sparsities, feature_subset=feats)
    elapsed = time.perf_counter() - t0
    n_solves = len(feats) * len(sparsities)
    return {
        "d": d, "F": F, "features_timed": len(feats), "sparsities": sparsities,
        "wall_s": elapsed,
        "s_per_solve": elapsed / n_solves,
        "projected_all_features_s": elapsed / len(feats) * F,
        "s_aff_robust": out["s_aff_robust"],
        "rows": [{k: r[k] for k in ("s", "all_separable", "n_separable", "n_infeasible",
                                    "n_unresolved", "gamma_min")} for r in out["rows"]],
    }


def write_report(payload: Dict[str, Any], out_dir: Path) -> None:
    tr = payload["training"]
    L: List[str] = []
    L.append("# DGX Spark capacity report")
    L.append("")
    L.append("Generated by `experiments/e0_benchmark.py`. Every number here is measured on the")
    L.append("machine named below; nothing is extrapolated from peak FLOPs.")
    L.append("")
    env = payload["environment"]
    L.append(f"- device: `{payload['device']}`")
    L.append(f"- platform: {env['platform']}  ({env['machine']})")
    L.append(f"- torch {env['torch']}, CUDA {env['torch_cuda']}")
    if env.get("gpu_name"):
        L.append(f"- GPU: {env['gpu_name']} ({env['gpu_capability']}, "
                 f"{env['gpu_memory_gib']:.1f} GiB)")
    L.append(f"- commit `{env['git']['commit'][:12]}`, dirty: {env['git']['dirty']}")
    L.append("")

    L.append("## Training throughput")
    L.append("")
    L.append("`s/model @50k` is the quantity that matters: seconds of wall clock per trained")
    L.append("model at the campaign's 50 000-step budget. Lower is better, and the point of")
    L.append("batching seeds is to drive it down.")
    L.append("")
    L.append("| d | F | seed batch | steps/s | s/model @50k | GPU util mean | peak MiB |")
    L.append("|---|---|---|---|---|---|---|")
    for r in tr:
        util = f"{r['gpu_util_mean']:.0f}%" if r.get("gpu_util_mean") is not None else "n/a"
        mem = f"{r['torch_peak_mem_mib']:.0f}" if r.get("torch_peak_mem_mib") else "n/a"
        L.append(f"| {r['d']} | {r['F']} | {r['seed_batch']} | {r['steps_per_s']:.1f} | "
                 f"{r['s_per_model_50k']:.1f} | {util} | {mem} |")
    L.append("")

    best = payload["selected_batching"]
    L.append("## Selected batching")
    L.append("")
    L.append("| d | best seed batch | s/model @50k | speedup vs B=1 |")
    L.append("|---|---|---|---|")
    for d, sel in sorted(best.items(), key=lambda kv: int(kv[0])):
        L.append(f"| {d} | {sel['seed_batch']} | {sel['s_per_model_50k']:.1f} | "
                 f"{sel['speedup_vs_b1']:.2f}x |")
    L.append("")

    L.append("## Projected campaign wall clock")
    L.append("")
    proj = payload["projection"]
    L.append("| block | models | projected hours |")
    L.append("|---|---|---|")
    for row in proj["blocks"]:
        L.append(f"| {row['name']} | {row['models']} | {row['hours']:.2f} |")
    L.append(f"| **total training** | **{proj['total_models']}** | "
             f"**{proj['total_hours']:.2f}** |")
    L.append("")
    L.append(f"Decision rule: rent additional GPUs only if this total exceeds roughly five to")
    L.append(f"seven days. Measured here: **{proj['total_hours']:.1f} h** "
             f"({proj['total_hours'] / 24:.1f} days) of training.")
    L.append("")

    if payload.get("affine_frontier"):
        L.append("## Robust affine frontier (CPU, convex solver)")
        L.append("")
        L.append("This does not run on the accelerator and is a separate budget line. The")
        L.append("projection is per model, for all features at the listed sparsities.")
        L.append("")
        L.append("| d | F | s/solve | all features, projected s |")
        L.append("|---|---|---|---|")
        for r in payload["affine_frontier"]:
            L.append(f"| {r['d']} | {r['F']} | {r['s_per_solve']:.3f} | "
                     f"{r['projected_all_features_s']:.0f} |")
        L.append("")

    (REPO_ROOT / "docs").mkdir(exist_ok=True)
    (REPO_ROOT / "docs" / "spark_capacity_report.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8", newline="\n")

    cfg_dir = REPO_ROOT / "configs" / "spark"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Selected by experiments/e0_benchmark.py from measured throughput.",
             "# Consumed by the campaign to size its seed batches; do not hand-edit.",
             f"device: {payload['device']}",
             f"measured_on: {env['platform']}",
             f"commit: {env['git']['commit']}",
             "seed_batch_by_width:"]
    for d, sel in sorted(best.items(), key=lambda kv: int(kv[0])):
        lines.append(f"  {d}: {sel['seed_batch']}")
    lines.append("projected_training_hours: %.2f" % proj["total_hours"])
    (cfg_dir / "selected_batching.yaml").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = base_parser("e0")
    parser.add_argument("--full-runs", action="store_true",
                        help="Also time two complete 50 000-step runs, at the smallest and "
                             "largest width. Adds real hours; the sweep alone sizes the grid.")
    parser.add_argument("--skip-frontier", action="store_true",
                        help="Skip the CPU-side affine-frontier timing.")
    args = parser.parse_args()
    cfg, out_dir, threads = prepare("e0", args)

    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible; "
                         "run scripts/check_env_gpu.py first")

    shapes = [tuple(x) for x in cfg.get("shapes", SHAPES)]
    batches = cfg.get("seed_batches", BATCHES)
    steps = cfg.get("steps", 5000)
    batch = cfg.get("batch", 2048)
    lr = cfg.get("lr", 1e-3)

    env: Dict[str, Any] = {
        "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "git": git_provenance(),
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu_name": p.name, "gpu_capability": f"sm_{p.major}{p.minor}",
                    "gpu_memory_gib": p.total_memory / 2 ** 30,
                    "torch_arch_list": torch.cuda.get_arch_list()})

    with RunRecord("e0_benchmark", out_dir,
                   config={**cfg, "device": device, "threads": threads,
                           "shapes": shapes, "seed_batches": batches, "steps": steps},
                   smoke=args.smoke) as rec:
        log(f"J0 benchmark on device={device}: {len(shapes)} shapes x {len(batches)} batch sizes")

        training: List[Dict[str, Any]] = []
        for (d, F) in shapes:
            for B in batches:
                r = time_training(d, F, B, steps, batch, device, lr)
                training.append(r)
                log(f"  d={d:4d} F={F:4d} B={B:3d}  {r['steps_per_s']:8.2f} steps/s  "
                    f"{r['s_per_model_50k']:8.1f} s/model@50k")

        # Best batch per width, by seconds per trained model.
        selected: Dict[str, Any] = {}
        for (d, F) in shapes:
            rows = [r for r in training if r["d"] == d]
            b1 = next(r for r in rows if r["seed_batch"] == min(batches))
            best = min(rows, key=lambda r: r["s_per_model_50k"])
            selected[str(d)] = {
                "seed_batch": best["seed_batch"],
                "s_per_model_50k": best["s_per_model_50k"],
                "speedup_vs_b1": b1["s_per_model_50k"] / best["s_per_model_50k"],
            }

        # Projection over the planned matrix (plan §5: J1 160, J2 60, J3 40 models).
        per_model = {int(k): v["s_per_model_50k"] for k, v in selected.items()}

        def hours(models_by_width: Dict[int, int]) -> float:
            return sum(n * per_model.get(d, max(per_model.values())) for d, n in
                       models_by_width.items()) / 3600.0

        blocks = [
            {"name": "J1 width scaling (20 paired seeds)", "models": 160,
             "hours": hours({50: 40, 100: 40, 200: 40, 400: 40})},
            {"name": "J2 training-sparsity robustness", "models": 60,
             "hours": hours({50: 20, 200: 20, 400: 20})},
            {"name": "J3 feature-load robustness (F=4d)", "models": 40,
             "hours": hours({100: 20, 200: 20})},
        ]
        projection = {"blocks": blocks,
                      "total_models": sum(b["models"] for b in blocks),
                      "total_hours": sum(b["hours"] for b in blocks)}

        frontier: List[Dict[str, Any]] = []
        if not args.skip_frontier:
            for (d, F) in cfg.get("frontier_shapes", [(50, 100), (100, 200)]):
                fr = time_affine_frontier(int(d), int(F), cfg.get("frontier_features", 5),
                                          cfg.get("frontier_sparsities", [1, 2, 3]), seed=0)
                frontier.append(fr)
                log(f"  frontier d={d} F={F}: {fr['s_per_solve']:.3f} s/solve, "
                    f"s_aff_robust={fr['s_aff_robust']}")

        payload = {"device": device, "environment": env, "training": training,
                   "selected_batching": selected, "projection": projection,
                   "affine_frontier": frontier}
        write_json(out_dir / "raw" / "spark_benchmark.json", payload)
        write_report(payload, out_dir)
        rec.set("projected_training_hours", projection["total_hours"])
        log(f"\nprojected training wall clock for J1+J2+J3: "
            f"{projection['total_hours']:.1f} h ({projection['total_hours'] / 24:.1f} days)")
        log("wrote docs/spark_capacity_report.md and configs/spark/selected_batching.yaml")


if __name__ == "__main__":
    main()
