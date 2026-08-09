"""G1 -- the code-specific optimum of the analog level of the interface hierarchy.

The rank-trace floor `W(F,d)` bounds every rank-`d` unit-diagonal interface, so it is a
statement about the *ensemble* of codes. No particular code need be able to attain it, which
makes "this code sits 0.06% above the floor" hard to interpret: is the code good, or is the
floor simply unreachable here?

This module answers that by computing the floor of the code itself. Fixing feature `i` and
requiring unit gain, the readout vector that minimises cross-talk is available in closed form,
and the resulting per-code optimum `W_*(Phi)` exceeds the rank-trace floor by an amount that
decomposes exactly into leverage heterogeneity.

Provenance, stated because it matters and because a referee will know it: the constrained
minimisation solved here is the minimum-variance distortionless-response (Capon) beamformer
with the frame operator in place of the covariance, and the minimiser is the canonical dual
frame vector rescaled to unit gain. The optimisation is classical. What is used here is the
reading of `h_i` as a leverage score, the resulting code-specific floor, and the exact
attribution of the gap. See `docs/novelty_matrix.md`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .codes import welch_floor

__all__ = [
    "leverage",
    "optimal_analog_readout",
    "optimal_crosstalk_per_feature",
    "code_specific_floor",
    "leverage_excess",
    "analog_attainment",
]


def _frame_operator(Phi: np.ndarray) -> np.ndarray:
    d, F = Phi.shape
    if F <= d:
        raise ValueError(f"an overcomplete code is required, got F={F} <= d={d}")
    Sigma = Phi @ Phi.T
    if np.linalg.matrix_rank(Sigma) < d:
        raise ValueError("the code does not have full row rank; the frame operator is singular")
    return Sigma


def leverage(Phi: np.ndarray) -> np.ndarray:
    """Leverage scores `h_i = phi_i^T (Phi Phi^T)^{-1} phi_i`, in `O(F d^2)`.

    Equal to the diagonal of the projector `Phi^+ Phi` when `Phi` has full row rank, but
    computed here by solving against the frame operator rather than forming the `F x F`
    projector, so the cost stays linear in `F`.
    """
    Sigma = _frame_operator(Phi)
    return np.einsum("ij,ij->j", Phi, np.linalg.solve(Sigma, Phi))


def optimal_analog_readout(Phi: np.ndarray) -> np.ndarray:
    """The `(F, d)` readout whose row `i` minimises cross-talk subject to unit gain on `i`.

    Row `i` is `g_i* = Sigma^{-1} phi_i / h_i`. Stacked, this is exactly the calibrated
    pseudoinverse: the row-wise optimum and the algebraic pseudoinverse coincide, which is the
    content of the theorem and is asserted in the tests.
    """
    Sigma = _frame_operator(Phi)
    G = np.linalg.solve(Sigma, Phi).T          # row i is (Sigma^{-1} phi_i)^T
    return G / leverage(Phi)[:, None]


def optimal_crosstalk_per_feature(Phi: np.ndarray) -> np.ndarray:
    """`min_g sum_{j != i} (g^T phi_j)^2` subject to `g^T phi_i = 1`, for every `i`.

    Equals `1/h_i - 1`. Note the objective is the Capon minimum `1/h_i` less the contribution
    of the constrained direction, which is exactly 1 by the unit-gain constraint.
    """
    return 1.0 / leverage(Phi) - 1.0


def code_specific_floor(Phi: np.ndarray) -> float:
    """`W_*(Phi)`: the smallest mean-square off-diagonal cross-talk this code admits."""
    d, F = Phi.shape
    return float(np.sum(optimal_crosstalk_per_feature(Phi)) / (F * (F - 1.0)))


def leverage_excess(Phi: np.ndarray) -> Dict[str, float]:
    """Decompose the gap between the code-specific floor and the rank-trace floor.

    With `c = d/F` the mean leverage (since `sum_i h_i = d`),

        sum_i h_i^{-1} - F^2/d = sum_i (h_i - c)^2 / (c^2 h_i),

    a weighted chi-square-like dispersion of the leverage scores. It is zero exactly when
    leverage is uniform, which is when the code-specific floor meets the rank-trace floor.
    """
    d, F = Phi.shape
    h = leverage(Phi)
    c = d / F
    return {
        "sum_inv_leverage": float(np.sum(1.0 / h)),
        "uniform_value": float(F ** 2 / d),
        "excess": float(np.sum(1.0 / h) - F ** 2 / d),
        "excess_identity": float(np.sum((h - c) ** 2 / (c ** 2 * h))),
        "leverage_mean": float(np.mean(h)),
        "leverage_std": float(np.std(h)),
        "leverage_min": float(np.min(h)),
        "leverage_max": float(np.max(h)),
    }


def analog_attainment(Phi: np.ndarray, G: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Locate a readout against *both* floors: the code's own, and the rank-trace bound.

    `ratio_vs_code` is the quantity that says whether a readout is good for this code;
    `ratio_vs_welch` is what the current manuscript reports, kept so the two can be compared.
    A readout can sit at `ratio_vs_welch = 1.0006` while `ratio_vs_code = 1.0000`, in which case
    the near-attainment is a property of the code's leverage profile and not of the readout.
    """
    from .interface import crosstalk_mean_sq

    d, F = Phi.shape
    if G is None:
        G = optimal_analog_readout(Phi)
        readout = "optimal"
    else:
        readout = "explicit"
    measured = crosstalk_mean_sq(np.ascontiguousarray(G, dtype=np.float64), Phi)
    w_code = code_specific_floor(Phi)
    w_welch = welch_floor(F, d)
    return {
        "d": int(d), "F": int(F), "readout": readout,
        "mean_sq_offdiag": measured,
        "floor_code_specific": w_code,
        "floor_welch": w_welch,
        "ratio_vs_code": measured / w_code,
        "ratio_vs_welch": measured / w_welch,
        "code_floor_over_welch": w_code / w_welch,
        **leverage_excess(Phi),
    }
