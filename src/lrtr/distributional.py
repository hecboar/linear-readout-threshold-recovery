"""G4 -- the analog criterion under an arbitrary state distribution.

The diagnostic as published evaluates the analog error under one sparse-state model: uniformly
random Boolean supports of fixed size. That is the model the theory is stated for, and it is
not the distribution the networks are trained on, which is the single most visible limitation
of the trained-network results.

The fix is elementary and exact. For any state law with second moment `C = E[b b^T]` and any
analog error matrix `A = M - I`,

    E||A b||^2 = tr(A^T A C),

so the whole distributional dependence enters through `C`. The exchangeable Boolean model is
one choice of `C` among many, and native inputs, continuous amplitudes, heterogeneous firing
rates and correlations are all just different `C`.

Nothing here is a new theorem -- these are second-moment identities. Their value is that they
turn a fixed evaluation distribution into a parameter of the method.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "second_moment",
    "distribution_weighted_error",
    "bernoulli_second_moment",
    "bernoulli_energy",
    "uniform_support_second_moment",
]


def second_moment(B: np.ndarray) -> np.ndarray:
    """Empirical `C = E[b b^T]` from states stored as columns of `(F, n)`."""
    F, n = B.shape
    if n < 2:
        raise ValueError(f"need at least two states to estimate a second moment, got {n}")
    return (B @ B.T) / n


def distribution_weighted_error(A: np.ndarray, C: np.ndarray,
                                per_coordinate: bool = True) -> float:
    """`E||A b||^2 = tr(A^T A C)`, optionally divided by `F`.

    Exact for the given `C`; no Monte Carlo. When `C` is estimated from data the estimate
    inherits that sampling error, which :func:`distribution_weighted_error_ci` reports.
    """
    if A.shape[1] != C.shape[0] or C.shape[0] != C.shape[1]:
        raise ValueError(f"shape mismatch: A is {A.shape}, C is {C.shape}")
    val = float(np.einsum("ij,ik,jk->", A, A, C))
    return val / A.shape[0] if per_coordinate else val


def distribution_weighted_error_ci(A: np.ndarray, B: np.ndarray, conf: float = 0.95,
                                   per_coordinate: bool = True) -> Dict[str, float]:
    """Point estimate and normal interval for `E||Ab||^2` from sampled states.

    Reported because an exact identity evaluated at an estimated `C` is not an exact number,
    and quoting it as one would be the same category of error the audit found elsewhere.
    """
    from scipy.stats import norm

    F, n = B.shape
    per_state = np.einsum("ij,jk->ik", A, B)
    vals = np.sum(per_state * per_state, axis=0)
    if per_coordinate:
        vals = vals / F
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    z = float(norm.ppf(0.5 + conf / 2.0))
    return {"mean": mean, "se": se, "ci_low": mean - z * se, "ci_high": mean + z * se, "n": n}


def bernoulli_second_moment(p: np.ndarray) -> np.ndarray:
    """`C` for independent `b_i ~ Bernoulli(p_i)`: `C = p p^T + diag(p(1-p))`."""
    p = np.asarray(p, dtype=np.float64).ravel()
    return np.outer(p, p) + np.diag(p * (1.0 - p))


def bernoulli_energy(A: np.ndarray, p: np.ndarray, per_coordinate: bool = True) -> float:
    """`E||Ab||^2` for independent Bernoulli features, without forming `C`.

    Decomposing `b = p + eps` with independent mean-zero `eps_i` of variance `p_i(1-p_i)`,

        E||A b||^2 = ||A p||^2 + sum_i p_i (1 - p_i) ||A e_i||^2,

    i.e. a mean term plus a column-norm term. Costs `O(F^2)` given `A`, or `O(F d)` when `A`
    is available in factored form; either way it never builds `C`, which would be `F x F`.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    F = A.shape[1]
    if p.size == 1:
        p = np.full(F, float(p))
    if p.size != F:
        raise ValueError(f"p must have {F} entries (or be scalar), got {p.size}")
    mean_term = float(np.sum((A @ p) ** 2))
    var_term = float(np.sum(p * (1.0 - p) * np.sum(A * A, axis=0)))
    total = mean_term + var_term
    return total / A.shape[0] if per_coordinate else total


def uniform_support_second_moment(F: int, s: int) -> Dict[str, float]:
    """`C` for a uniformly random support of size `s`, given by its two distinct entries.

    `C_ii = s/F` and `C_ij = s(s-1)/(F(F-1))` for `i != j`, i.e.
    `C = (p - q) I + q 11^T` with `p = s/F`, `q = s(s-1)/(F(F-1))`. Returned as the two scalars
    rather than the matrix, since substituting them into `tr(A^T A C)` reproduces the closed
    form already used for the Boolean model:
    `E||Ab||^2 = (p - q)||A||_F^2 + q||A 1||^2`.
    """
    if not 1 <= s <= F:
        raise ValueError(f"need 1 <= s <= F, got s={s}, F={F}")
    p = s / F
    q = s * (s - 1) / (F * (F - 1.0))
    return {"diagonal": p, "off_diagonal": q}
