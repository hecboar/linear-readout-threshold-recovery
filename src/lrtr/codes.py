"""Overcomplete codes and the Welch-type floors they are measured against.

Notation follows the manuscript: a code is ``Phi`` of shape ``(d, F)`` whose columns
``phi_i`` are the feature directions, ``d`` is the activation width and ``F > d`` the number
of features.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "welch_floor",
    "welch_floor_max",
    "random_unit_code",
    "harmonic_tight_frame",
    "tight_frame_residual",
    "coherence",
    "duplicated_basis_code",
    "basis_hadamard_code",
]


def welch_floor(F: int, d: int) -> float:
    """Mean-square Welch-type floor ``W(F, d) = (F - d) / (d (F - 1))``.

    Lower bound on ``mean_{i != j} M_ij^2`` for any ``M`` of rank at most ``d`` with unit
    diagonal (Theorem: unit-diagonal cross-Gramian Welch floor).
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if F <= 1:
        raise ValueError(f"F must be at least 2, got {F}")
    if F <= d:
        raise ValueError(f"the floor is only informative for F > d; got F={F}, d={d}")
    return (F - d) / (d * (F - 1))


def welch_floor_max(F: int, d: int) -> float:
    """Floor on ``max_{i != j} |M_ij|``: the square root of :func:`welch_floor`."""
    return float(np.sqrt(welch_floor(F, d)))


def random_unit_code(d: int, F: int, rng: np.random.Generator, dtype=np.float64) -> np.ndarray:
    """``(d, F)`` code with i.i.d. columns uniform on the unit sphere ``S^{d-1}``."""
    if d < 1 or F < 1:
        raise ValueError(f"d and F must be positive, got d={d}, F={F}")
    Phi = rng.standard_normal((d, F)).astype(dtype, copy=False)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi


def harmonic_tight_frame(d: int, F: int) -> np.ndarray:
    """Real harmonic unit-norm tight frame of ``F`` vectors in ``R^d`` (``d`` even).

    Columns are unit norm and ``Phi @ Phi.T == (F / d) I_d`` exactly (up to rounding), so the
    frame attains the Welch floor with equality. Used as the equality witness in tests and as
    the reference optimum in E4.
    """
    if d % 2 != 0:
        raise ValueError(f"harmonic_tight_frame requires even d, got {d}")
    m = d // 2
    if F <= 2 * m:
        raise ValueError(f"need F > d for an overcomplete frame; got F={F}, d={d}")
    freqs = np.arange(1, m + 1)
    if np.any(freqs % F == 0) or np.any((2 * freqs) % F == 0):
        raise ValueError(f"degenerate frequency set for d={d}, F={F}; choose a larger F")
    k = np.arange(F)
    ang = 2.0 * np.pi * np.outer(freqs, k) / F  # (m, F)
    Phi = np.empty((d, F), dtype=np.float64)
    Phi[0::2] = np.cos(ang)
    Phi[1::2] = np.sin(ang)
    Phi *= np.sqrt(2.0 / d)
    return Phi


def tight_frame_residual(Phi: np.ndarray) -> float:
    """Relative tight-frame residual ``||Phi Phi^T - (F/d) I||_F / ||(F/d) I||_F``.

    Zero exactly when ``Phi`` is a tight frame. Reported in E4 variant (b).
    """
    d, F = Phi.shape
    S = Phi @ Phi.T
    target = (F / d) * np.eye(d, dtype=S.dtype)
    return float(np.linalg.norm(S - target) / np.linalg.norm(target))


def coherence(Phi: np.ndarray) -> float:
    """``mu = max_{i != j} |<phi_i, phi_j>|`` for a unit-norm code."""
    G = Phi.T @ Phi
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())


def duplicated_basis_code(d: int) -> np.ndarray:
    """``[I, I]``: an orthonormal basis repeated. Unit-norm columns, frame operator ``2 I``."""
    I = np.eye(d)
    return np.hstack([I, I])


def basis_hadamard_code(d: int) -> np.ndarray:
    """``[I, H/sqrt(d)]`` with ``H`` a Hadamard matrix. Requires ``d`` a power of two.

    Paired with :func:`duplicated_basis_code` this is the separation that shows the affine level
    of the hierarchy is not a function of the analog level. Both codes are ``d x 2d``, both have
    unit-norm columns, and both have frame operator ``2 I`` -- since ``(H/sqrt d)(H/sqrt d)^T =
    I`` -- so both have every leverage score equal to ``1/2``, the same code-specific analog
    optimum and the same Welch ratio. Nothing at the analog level can tell them apart.

    Their affine frontiers could not differ more. ``[I, I]`` repeats every column, so features
    ``j`` and ``j + d`` are indistinguishable and the frontier is ``0``: not even a single active
    feature can be recovered. ``[I, H/sqrt d]`` has coherence ``1/sqrt(d)``, so by the coherence
    bound on the collision radius its frontier grows like ``sqrt(d)``.
    """
    from scipy.linalg import hadamard

    if d < 1 or (d & (d - 1)) != 0:
        raise ValueError(f"basis_hadamard_code needs d a power of two, got {d}")
    return np.hstack([np.eye(d), hadamard(d) / np.sqrt(d)])
