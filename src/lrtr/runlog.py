"""Run bookkeeping: every campaign writes a self-describing record next to its results.

A ``run_record.json`` captures the exact command, seeds, environment, wall-clock duration and
exit code, so that any number in the manuscript can be traced back to the run that produced
it.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["REPO_ROOT", "RESULTS_ROOT", "environment_info", "git_provenance", "RunRecord",
           "write_json", "read_json"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    return _git("rev-parse", "HEAD")


def git_provenance() -> Dict[str, Any]:
    """Commit, branch and dirty status of the tree the run was launched from.

    A result is only traceable if the code that produced it can be identified. Recording the
    commit alone is not enough: a dirty tree is a different program from the commit it claims
    to be, so the modified paths are listed rather than silently omitted. Runs launched from a
    dirty tree are legitimate during development and must simply say so.
    """
    status = _git("status", "--porcelain")
    # Split on the status field rather than slicing a fixed offset: `_git` strips its output, so a
    # leading unmodified-index space (" M path") is gone from the *first* line only, and `ln[3:]`
    # then ate the first character of that one path. A record that misnames the file it warns about
    # is worse than no warning, and only the first line being wrong is exactly what hides it.
    dirty_files = [ln.split(maxsplit=1)[1] for ln in status.splitlines() if ln.split(maxsplit=1)[1:]
                   ] if status else []
    return {
        "commit": _git_commit(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:200],
        "dirty_file_count": len(dirty_files),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
    }


def environment_info() -> Dict[str, Any]:
    """Versions and CPU configuration, recorded with every run."""
    import numpy
    import scipy

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "git_commit": _git_commit(),
        "git": git_provenance(),
        "env_omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except Exception:
        info["matplotlib"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_num_threads"] = int(torch.get_num_threads())
    except Exception:
        info["torch"] = None
    return info


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_default)
    return path


def read_json(path: Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj)!r} is not JSON serialisable")


@dataclass
class RunRecord:
    """Context manager that records one experimental campaign."""

    experiment: str
    out_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    smoke: bool = False
    _t0: float = field(default=0.0, init=False, repr=False)
    payload: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "RunRecord":
        self._t0 = time.perf_counter()
        self.payload = {
            "experiment": self.experiment,
            "smoke": self.smoke,
            "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "argv": sys.argv,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": self.config,
            "environment": environment_info(),
        }
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.payload["duration_seconds"] = round(time.perf_counter() - self._t0, 3)
        self.payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.payload["exit_status"] = "error" if exc_type is not None else "ok"
        if exc_type is not None:
            self.payload["error"] = f"{exc_type.__name__}: {exc}"
        write_json(self.out_dir / "run_record.json", self.payload)
        return False

    def set(self, key: str, value: Any) -> None:
        self.payload[key] = value
