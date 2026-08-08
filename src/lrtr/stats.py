"""Statistics used in the experimental campaigns."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats as sps

__all__ = [
    "wilson_interval",
    "mann_whitney",
    "cliffs_delta",
    "fit_c_over_log",
]


def wilson_interval(successes: int, trials: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because recovery probabilities sit at 0 and 1.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    z = float(sps.norm.ppf(0.5 + conf / 2.0))
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    lo, hi = centre - half, centre + half
    # At k = 0 and k = n the endpoints are exactly 0 and 1 in exact arithmetic; snap away the
    # rounding error so that the interval always brackets the point estimate.
    snap = 1e-12
    lo = 0.0 if lo < snap else min(lo, p)
    hi = 1.0 if hi > 1.0 - snap else max(hi, p)
    return float(lo), float(hi)


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided Mann-Whitney U test with the exact p-value for small samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")
    res = sps.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return {"U": float(res.statistic), "p_value": float(res.pvalue),
            "n_a": int(a.size), "n_b": int(b.size)}


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size in ``[-1, 1]``; ``+1`` means every ``a`` exceeds every ``b``."""
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    return float((np.sign(a - b)).mean())


def fit_c_over_log(ds: Sequence[int], s_values: Sequence[float]) -> Dict[str, float]:
    """Least-squares fit of ``s(d) = c * d / ln d`` through the origin, plus ``R^2``.

    A one-parameter fit is used deliberately: the theory predicts the ``d / ln d`` shape and
    only the constant is free. ``R^2`` is reported against the mean of ``s_values``.
    """
    d = np.asarray(ds, dtype=float)
    y = np.asarray(s_values, dtype=float)
    if d.size != y.size or d.size < 2:
        raise ValueError(f"need at least 2 paired points, got {d.size} and {y.size}")
    x = d / np.log(d)
    c = float((x @ y) / (x @ x))
    pred = c * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"c": c, "r2": r2, "n_points": int(d.size),
            "predicted": [float(v) for v in pred]}
