"""Shared plumbing for the experimental campaigns.

Every campaign follows the same contract: parse a JSON config, honour ``--smoke``, pin the
CPU thread count, write raw results plus a ``run_record.json`` under ``results/<exp>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List


def _pin_threads_before_numpy() -> int:
    """Set the BLAS thread limits *before* NumPy is imported.

    OpenMP/MKL/OpenBLAS read these variables when their runtime is loaded, which happens on
    the first NumPy import. Setting them afterwards is silently ignored, so the whole
    pinning has to happen here, at the very top of the process, from a hand-parsed argv.
    """
    if os.environ.get("LRTR_CHILD") == "1":
        # Spawned worker: the parent already exported the per-worker limits, and re-deriving
        # them from argv here would give every worker the full-machine count.
        return int(os.environ.get("OMP_NUM_THREADS", "1"))
    threads = None
    device = None
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--threads" and i + 1 < len(argv):
            threads = int(argv[i + 1])
        elif tok.startswith("--threads="):
            threads = int(tok.split("=", 1)[1])
        elif tok == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
        elif tok.startswith("--device="):
            device = tok.split("=", 1)[1]
    if threads is None:
        threads = max(1, (os.cpu_count() or 3) - 2)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    # The CPU campaigns hide the GPU so that a stray accelerator cannot make a published run
    # unreproducible on a CPU-only machine. A campaign that asks for one opts back in here,
    # before torch is imported, which is the only point at which the variable is read.
    if (device or os.environ.get("LRTR_DEVICE", "cpu")).startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return threads


_THREADS = _pin_threads_before_numpy()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lrtr import configure_cpu  # noqa: E402
from lrtr.runlog import RunRecord, write_json  # noqa: E402

__all__ = ["REPO_ROOT", "base_parser", "load_config", "prepare", "RunRecord", "write_json",
           "map_trials", "log"]


def base_parser(experiment: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Run experiment {experiment}.")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / f"{experiment}.json",
                   help="JSON configuration file.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default results/<experiment>).")
    p.add_argument("--smoke", action="store_true",
                   help="Run the reduced configuration for a fast end-to-end check.")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads to use (default: cpu_count - 2).")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes for trial-level parallelism.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device for campaigns that train ('cpu' or 'cuda'). Anything "
                        "other than cpu also un-hides the GPU from this process.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse per-cell checkpoints already present in the output directory.")
    return p


def load_config(path: Path, smoke: bool) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if smoke:
        overrides = cfg.get("smoke", {})
        cfg = {**cfg, **overrides}
    cfg.pop("smoke", None)
    return cfg


def prepare(experiment: str, args: argparse.Namespace) -> tuple[Dict[str, Any], Path, int]:
    cfg = load_config(args.config, args.smoke)
    out_dir = args.out_dir or (REPO_ROOT / "results" / (experiment + ("_smoke" if args.smoke else "")))
    # The BLAS limits were already applied by _pin_threads_before_numpy(); this call pins the
    # torch pool, which can be set at any time.
    threads = configure_cpu(args.threads if args.threads is not None else _THREADS)
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, out_dir, threads


def log(msg: str) -> None:
    print(msg, flush=True)


def map_trials(fn: Callable[..., Any], jobs: Iterable[tuple], workers: int | None,
               threads_per_worker: int = 1, on_done: Callable[[int, int], None] | None = None
               ) -> List[Any]:
    """Run ``fn(*job)`` for each job, in parallel when ``workers`` exceeds one.

    ``fn`` must be a module-level function so that it can be pickled on Windows. The
    per-worker BLAS thread limit is exported before the pool is created so that spawned
    children inherit it at import time, which is the only moment at which it takes effect.
    Longest jobs are submitted first so that the pool does not end on a long tail.
    """
    jobs = list(jobs)
    if not workers or workers <= 1:
        out = []
        for i, j in enumerate(jobs):
            out.append(fn(*j))
            if on_done:
                on_done(i + 1, len(jobs))
        return out

    saved = {k: os.environ.get(k) for k in
             ("LRTR_CHILD", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    os.environ["LRTR_CHILD"] = "1"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads_per_worker)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fn, *j): i for i, j in enumerate(jobs)}
            results: List[Any] = [None] * len(jobs)
            done = 0
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if on_done:
                    on_done(done, len(jobs))
            return results
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
