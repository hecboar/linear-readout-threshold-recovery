"""Readout interfaces ``M = G Phi`` and the quantities the Interface Diagnostic reports.

The interface matrix is always brought to unit diagonal before any cross-talk statistic is
computed; this is the calibration that makes the Welch-type floor meaningful (see the
calibration-free corollary in the manuscript).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from .codes import welch_floor, welch_floor_max

__all__ = [
    "unit_diagonal",
    "CrosstalkStats",
    "crosstalk_stats",
    "interface_matrix",
    "linear_energy_bernoulli",
    "linear_energy_uniform",
    "energy_floor_bernoulli",
    "energy_floor_uniform",
    "least_squares_readout",
]


def unit_diagonal(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rescale each row of ``M`` so that the diagonal is all ones.

    This is ``D^{-1} M`` with ``D = diag(M_11, ..., M_FF)``. Raises if any diagonal entry is
    numerically zero, since the interface is then undefined (a feature read out with zero
    gain).
    """
    diag = np.diagonal(M)
    if np.any(np.abs(diag) < eps):
        bad = int(np.argmin(np.abs(diag)))
        raise ValueError(
            f"interface has a near-zero diagonal gain at index {bad} "
            f"(|M_ii| = {abs(diag[bad]):.3e} < {eps:.1e}); the unit-diagonal calibration "
            "is undefined for this readout"
        )
    return M / diag[:, None]


@dataclass(frozen=True)
class CrosstalkStats:
    """Cross-talk summary of a calibrated interface."""

    d: int
    F: int
    mean_sq_offdiag: float
    max_abs_offdiag: float
    floor_mean_sq: float
    floor_max: float
    ratio_mean_sq: float
    ratio_max: float
    violates_floor: bool

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def crosstalk_stats(M: np.ndarray, d: int, tol: float = 1e-9) -> CrosstalkStats:
    """Cross-talk statistics of an interface ``M`` of rank at most ``d``.

    ``M`` must already be square and unit-diagonal (use :func:`unit_diagonal`).
    ``violates_floor`` is ``True`` only if the mean squared off-diagonal falls below the
    theoretical floor by more than ``tol`` relative; by the theorem this must never happen,
    so it doubles as a correctness check on the implementation.
    """
    F = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"interface must be square, got shape {M.shape}")
    off = M - np.eye(F, dtype=M.dtype)
    sum_sq = float(np.sum(off * off))
    mean_sq = sum_sq / (F * (F - 1))
    max_abs = float(np.abs(off).max())
    fl = welch_floor(F, d)
    fl_max = welch_floor_max(F, d)
    return CrosstalkStats(
        d=int(d),
        F=int(F),
        mean_sq_offdiag=mean_sq,
        max_abs_offdiag=max_abs,
        floor_mean_sq=fl,
        floor_max=fl_max,
        ratio_mean_sq=mean_sq / fl,
        ratio_max=max_abs / fl_max,
        violates_floor=bool(mean_sq < fl * (1.0 - tol)),
    )


def interface_matrix(G: np.ndarray, Phi: np.ndarray, calibrate: bool = True) -> np.ndarray:
    """Build ``M = G Phi``, optionally calibrated to unit diagonal."""
    if G.shape[1] != Phi.shape[0]:
        raise ValueError(f"shape mismatch: G is {G.shape}, Phi is {Phi.shape}")
    M = G @ Phi
    return unit_diagonal(M) if calibrate else M


def gains(G: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Diagonal gains ``M_ii = <g_i, phi_i>`` of the interface, in ``O(Fd)``.

    Computed without forming ``M``: the diagonal is a row-wise inner product.
    """
    if G.shape[1] != Phi.shape[0] or G.shape[0] != Phi.shape[1]:
        raise ValueError(f"shape mismatch: G is {G.shape}, Phi is {Phi.shape}")
    return np.einsum("ij,ji->i", G, Phi)


def crosstalk_mean_sq(G: np.ndarray, Phi: np.ndarray, eps: float = 1e-12) -> float:
    """Exact mean squared off-diagonal cross-talk of the calibrated interface.

    Computed from ``d x d`` sufficient statistics instead of the ``F x F`` interface, so the
    cost is ``O(F d^2)`` time and ``O(d^2)`` memory rather than ``O(F^2 d)`` and ``O(F^2)``.

    Writing ``Gbar = D^{-1} G`` for the row-calibrated readout, trace cyclicity gives

        ||Gbar Phi||_F^2 = tr(Phi^T Gbar^T Gbar Phi) = tr(H S),
        H = Gbar^T Gbar  (d x d),   S = Phi Phi^T  (d x d),

    and since the calibrated diagonal is all ones and contributes exactly ``F``,

        sum_{i != j} M_ij^2 = tr(H S) - F.

    This is exact, not an approximation. The maximum statistic
    ``max_{i != j} |M_ij|`` has no such reduction and remains ``O(F^2 d)``; use
    :func:`crosstalk_stats` on an explicit interface when it is needed.
    """
    d, F = Phi.shape
    diag = gains(G, Phi)
    if np.any(np.abs(diag) < eps):
        bad = int(np.argmin(np.abs(diag)))
        raise ValueError(
            f"interface has a near-zero diagonal gain at index {bad} "
            f"(|M_ii| = {abs(diag[bad]):.3e} < {eps:.1e}); the unit-diagonal calibration "
            "is undefined for this readout"
        )
    Gbar = G / diag[:, None]
    H = Gbar.T @ Gbar
    S = Phi @ Phi.T
    total = float(np.sum(H * S))          # tr(H S), both d x d
    return (total - F) / (F * (F - 1.0))


# --------------------------------------------------------------------------------------
# Average linear-readout energy (closed forms; no Monte Carlo needed)
# --------------------------------------------------------------------------------------

def linear_energy_bernoulli(A: np.ndarray, p: float) -> float:
    """``(1/F) E_b ||A b||^2`` for ``b_i ~ Bernoulli(p)`` independent.

    Uses ``E[b b^T] = p(1-p) I + p^2 11^T``.
    """
    F = A.shape[1]
    frob_sq = float(np.sum(A * A))
    row_sums = A @ np.ones(F, dtype=A.dtype)
    return float(p * (1.0 - p) * frob_sq + p * p * float(row_sums @ row_sums)) / F


def linear_energy_uniform(A: np.ndarray, s: int) -> float:
    """``(1/F) E_S ||A 1_S||^2`` for ``S`` a uniformly random support of size ``s``.

    Uses ``E[b b^T] = (p - q) I + q 11^T`` with ``p = s/F`` and ``q = s(s-1)/(F(F-1))``.
    """
    F = A.shape[1]
    if not 1 <= s <= F:
        raise ValueError(f"need 1 <= s <= F, got s={s}, F={F}")
    p = s / F
    q = s * (s - 1) / (F * (F - 1)) if F > 1 else 0.0
    frob_sq = float(np.sum(A * A))
    row_sums = A @ np.ones(F, dtype=A.dtype)
    return float((p - q) * frob_sq + q * float(row_sums @ row_sums)) / F


def energy_floor_bernoulli(F: int, d: int, s: int) -> float:
    """Per-coordinate lower bound ``s (F - d) / (2 d F)`` under Bernoulli(``s/F``) states."""
    if not 0 < s <= F / 2:
        raise ValueError(f"need 0 < s <= F/2 for this form, got s={s}, F={F}")
    return s * (F - d) / (2.0 * d * F)


def energy_floor_uniform(F: int, d: int, s: int) -> float:
    """Per-coordinate lower bound under uniformly random supports of size ``s``.

    ``(1/F) E_S ||A 1_S||^2 >= s (F - s) (F - d) / (F d (F - 1))``.
    """
    if not 1 <= s <= F:
        raise ValueError(f"need 1 <= s <= F, got s={s}, F={F}")
    return s * (F - s) * (F - d) / (F * d * (F - 1.0))


def leverage_scores(Phi: np.ndarray) -> np.ndarray:
    """Leverage scores ``h_i = (Phi^+ Phi)_ii`` of the code's features.

    ``P = Phi^+ Phi`` is the orthogonal projector onto the row space of ``Phi``, so
    ``sum_i h_i = tr(P) = rank(Phi) <= d`` and ``h_i in [0, 1]``.
    """
    P = np.linalg.pinv(Phi) @ Phi
    return np.diag(P).copy()


def crosstalk_mean_sq_pinv(Phi: np.ndarray, eps: float = 1e-12) -> float:
    """Mean squared off-diagonal cross-talk under the pseudoinverse readout, from leverage.

    For ``G = Phi^+`` the interface is the projector ``P = Phi^+ Phi``, which is symmetric and
    idempotent, so row ``i`` satisfies ``sum_j P_ij^2 = P_ii = h_i``. After calibrating the row
    by ``1/h_i``,

        sum_{j != i} (P_ij / h_i)^2 = (h_i - h_i^2) / h_i^2 = 1/h_i - 1,

    hence the whole statistic collapses to ``(sum_i 1/h_i - F) / (F(F-1))`` -- an ``O(Fd^2)``
    quantity that depends on the code *only* through its leverage scores.

    Because ``sum_i h_i = d``, Cauchy-Schwarz gives ``sum_i 1/h_i >= F^2/d`` with equality iff
    ``h_i = d/F`` for every feature. The attainment ratio of the pseudoinverse readout therefore
    reaches one exactly when leverage is perfectly equalised across features: it measures how
    evenly the code spreads its features over the row space, which is a weaker statement than
    "this code computes well".
    """
    d, F = Phi.shape
    h = leverage_scores(Phi)
    if np.any(h < eps):
        bad = int(np.argmin(h))
        raise ValueError(
            f"feature {bad} has near-zero leverage (h = {h[bad]:.3e} < {eps:.1e}); it lies "
            "essentially outside the row space and the calibration is undefined"
        )
    return (float(np.sum(1.0 / h)) - F) / (F * (F - 1.0))


def least_squares_readout(H: np.ndarray, B: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Best linear map ``G`` with ``G H ~ B`` in least squares (columns are samples).

    ``H`` is ``(d, n)`` of representations, ``B`` is ``(F, n)`` of targets. Returns ``(F, d)``.
    A small ``ridge`` can be used for conditioning; ``ridge=0`` gives the plain pseudoinverse
    solution, which is the honest "best linear readout" baseline.
    """
    if H.shape[1] != B.shape[1]:
        raise ValueError(f"sample count mismatch: H has {H.shape[1]}, B has {B.shape[1]}")
    d = H.shape[0]
    HHt = H @ H.T
    if ridge > 0:
        HHt = HHt + ridge * np.eye(d, dtype=H.dtype)
    return np.linalg.solve(HHt.T, (B @ H.T).T).T
