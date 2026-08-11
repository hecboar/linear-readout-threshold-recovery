#!/usr/bin/env python3
"""Pre-flight check for the accelerator campaign (E7). Run this before anything else.

Three things go wrong when a CPU-developed campaign meets a new machine, and all three are
silent: the CUDA build does not match the card's architecture and every kernel falls back or
fails; the GPU is visible to ``nvidia-smi`` but hidden from this process; or it all runs and
quietly produces different numbers. This script checks for each, ending with two tests of the
actual training routine: that a seed names the same initial code on both devices, and that the
update itself agrees in float64 when both devices are fed identical data.

Exit status is 0 only if a usable accelerator is present *and* reproduces the CPU result.

Run: ``python scripts/check_env_gpu.py``
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

# Must precede the torch import: a stale empty value hides the device permanently.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lrtr.toymodel import train_toy_models_batched  # noqa: E402


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> int:
    failures = []

    section("interpreter and platform")
    print(f"python      {sys.version.split()[0]}  ({platform.machine()}, {platform.system()})")
    print(f"numpy       {np.__version__}")
    print(f"torch       {torch.__version__}")
    print(f"torch cuda  {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    section("device")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False.")
        print("      If nvidia-smi sees the GPU, this is a torch build mismatch: the wheel")
        print("      installed is CPU-only, or is built for a different CUDA major version.")
        print("      On DGX Spark (GB10, aarch64) install the CUDA build for that platform")
        print("      rather than the default PyPI wheel.")
        return 1

    n = torch.cuda.device_count()
    print(f"devices     {n}")
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {prop.name}  sm_{prop.major}{prop.minor}  "
              f"{prop.total_memory / 2**30:.1f} GiB  {prop.multi_processor_count} SMs")
    arch_list = torch.cuda.get_arch_list()
    print(f"built for   {', '.join(arch_list)}")
    cap = torch.cuda.get_device_capability(0)
    tag = f"sm_{cap[0]}{cap[1]}"
    if not any(tag == a.replace("compute_", "sm_") for a in arch_list):
        print(f"WARNING: this torch was not built for {tag}. It may still run via PTX JIT,")
        print("         with a long first-kernel delay, or fail outright.")

    section("throughput")
    torch.cuda.synchronize()
    a = torch.randn(4096, 4096, device="cuda")
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f"fp32 4096^3 matmul: {dt * 1e3:.2f} ms  ->  {2 * 4096**3 / dt / 1e12:.1f} TFLOP/s")

    section("numerical equivalence of the training routine")
    # A CUDA generator does not reproduce a CPU generator's stream for the same seed, so
    # comparing two runs that each draw their own data would compare different random problems
    # and fail on every machine, telling us nothing about the arithmetic. The two properties
    # that do matter are checked separately.
    kw = dict(d=8, F=24, loss_kind="L4", p=0.05, seeds=[0, 1], steps=20, batch=64,
              lr=1e-3, dtype=torch.float64)

    # (1) A seed must name the same initial code on both devices: every theoretical quantity is
    #     computed from W_in, so a device-dependent initialisation would make the theory and the
    #     measurement describe different objects.
    init_kw = dict(kw, steps=0)
    ci = train_toy_models_batched(device="cpu", **init_kw)
    gi = train_toy_models_batched(device="cuda", **init_kw)
    init_diff = max(float(np.abs(c[k] - g[k]).max())
                    for c, g in zip(ci, gi) for k in ("W_in", "W_out"))
    print(f"initialisation, max absolute difference:          {init_diff:.2e}")
    if init_diff != 0.0:
        print("FAIL: the same seed gives a different code on the accelerator.")
        failures.append("initialisation")

    # (2) Given the same initialisation and the same per-step data, the update must agree. This
    #     is the arithmetic test: it is what catches a wrong architecture, silent reduced
    #     precision, or a miscompiled kernel.
    cpu = train_toy_models_batched(device="cpu", **kw)
    gpu = train_toy_models_batched(device="cuda", cpu_data=True, **kw)
    worst = 0.0
    for c, g in zip(cpu, gpu):
        for key in ("W_in", "W_out"):
            denom = max(np.abs(c[key]).max(), 1e-12)
            worst = max(worst, float(np.abs(c[key] - g[key]).max() / denom))
    print(f"identical inputs, max relative weight difference: {worst:.2e}")
    if worst > 1e-6:
        print("FAIL: the accelerator does not reproduce the CPU reference in float64.")
        print("      Do not run the campaign until this is understood.")
        failures.append("equivalence")
    elif init_diff == 0.0:
        print("OK: float64 training agrees with the CPU reference.")

    section("verdict")
    if failures:
        print("NOT READY: " + ", ".join(failures))
        return 1
    print("READY. Launch with:  bash scripts/run_spark.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
