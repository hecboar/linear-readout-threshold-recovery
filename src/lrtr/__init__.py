"""lrtr -- linear-readout floors and threshold recovery.

Reference implementation of the Interface Diagnostic and of the six experimental campaigns
reported in the manuscript. Everything runs on CPU.
"""
from __future__ import annotations

import os

__version__ = "1.0.0"


def configure_cpu(threads: int | None = None, reserve: int = 2) -> int:
    """Pin the process to a fixed CPU thread count and disable any GPU use.

    Call this at the top of every experiment so that runs are reproducible and the machine
    keeps ``reserve`` cores free for the system. Returns the thread count actually set.
    """
    if threads is None:
        total = os.cpu_count() or 1
        threads = max(1, total - reserve)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    return threads


from . import codes, diagnostic, interface, runlog, stats, threshold  # noqa: E402

__all__ = ["codes", "diagnostic", "interface", "runlog", "stats", "threshold",
           "configure_cpu", "__version__"]
