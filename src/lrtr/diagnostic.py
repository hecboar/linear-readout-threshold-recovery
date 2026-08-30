"""The Interface Diagnostic (ID): the method proposed in the manuscript.

The diagnostic takes an overcomplete code -- a hand-built code, a gradient-optimised code, or
the effective code of a trained network -- and returns computable quantities that place its
*decoding interface* in one of two regimes:

``ALG-1`` :func:`interface_floor_diagnostic`
    How close the calibrated readout interface sits to the unit-diagonal Welch floor.

``ALG-2`` :func:`fixed_code_separation_profile`
    Applied to *one given* code and readout. For each sparsity ``s``: the exact-recovery
    probability of the support-recovery criterion, and the irreducible per-coordinate error of
    the *same* linear map read as an analog readout. The criterion-separation witness is the
    largest ``s`` at which support recovery is exact while the analog criterion provably is not.

:func:`random_code_scaling_experiment` is *not* the diagnostic. It redraws a fresh random code
on every trial and therefore measures an ensemble, not a code; it is the vehicle for the
width-scaling study and is kept separate so the two are not confused.

Cost. Neither routine materialises the ``F x F`` interface. The mean-square side reduces to
``d x d`` sufficient statistics (:func:`interface_energy_moments`), so both run in ``O(F d^2)``
time and ``O(d^2)`` memory; the random-code version streams the code as well and so reaches
``F = 2^20`` on a laptop. Only the *maximum* off-diagonal statistic resists this reduction and
costs ``O(F^2 d)``; ALG-1 computes it only when asked.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .codes import welch_floor, welch_floor_max
from .interface import (
    crosstalk_mean_sq,
    crosstalk_stats,
    energy_floor_uniform,
    gains,
    unit_diagonal,
)
from .stats import wilson_interval
from .threshold import (
    DEFAULT_BLOCK,
    _block_columns,
    recovery_trial_fixed,
    recovery_trial_streaming,
    s95_from_curve,
)

__all__ = [
    "TiedEnergyMoments",
    "tied_energy_moments",
    "tied_energy_moments_streaming",
    "tied_linear_energy",
    "interface_energy_moments",
    "interface_linear_energy",
    "interface_floor_diagnostic",
    "fixed_code_separation_profile",
    "random_code_scaling_experiment",
    "interface_separation_profile",
]


# --------------------------------------------------------------------------------------
# ALG-2 support: the linear-error side in closed form, without forming M
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TiedEnergyMoments:
    """Sufficient statistics of the tied interface ``M = Phi^T Phi`` (unit diagonal).

    ``frob_sq_A`` is ``||A||_F^2`` and ``sum_A1_sq`` is ``||A 1||^2`` for ``A = M - I``.
    Together with ``F`` they determine the average linear energy under any exchangeable
    sparse-state model.
    """

    d: int
    F: int
    frob_sq_A: float
    sum_A1_sq: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _moments_from_S_v(S: np.ndarray, v: np.ndarray, d: int, F: int) -> TiedEnergyMoments:
    """Build the moments from the frame operator ``S = Phi Phi^T`` and ``v = Phi 1``.

    ``||A||_F^2 = ||Phi^T Phi||_F^2 - F = ||S||_F^2 - F`` (trace cyclicity), and
    ``||A 1||^2 = v^T S v - 2 ||v||^2 + F``.
    """
    frob_sq_A = float(np.sum(S * S)) - F
    sum_A1_sq = float(v @ (S @ v)) - 2.0 * float(v @ v) + F
    return TiedEnergyMoments(d=int(d), F=int(F),
                             frob_sq_A=max(frob_sq_A, 0.0),
                             sum_A1_sq=max(sum_A1_sq, 0.0))


def tied_energy_moments(Phi: np.ndarray) -> TiedEnergyMoments:
    """Moments of the tied interface for a materialised unit-norm code."""
    d, F = Phi.shape
    S = (Phi @ Phi.T).astype(np.float64)
    v = Phi.sum(axis=1).astype(np.float64)
    return _moments_from_S_v(S, v, d, F)


def tied_energy_moments_streaming(master: int, trial: int, d: int, F: int,
                                  block: int = DEFAULT_BLOCK,
                                  dtype=np.float32) -> TiedEnergyMoments:
    """Same moments for a code that is regenerated block by block and never stored."""
    S = np.zeros((d, d), dtype=np.float64)
    v = np.zeros(d, dtype=np.float64)
    n_blocks = (F + block - 1) // block
    for blk in range(n_blocks):
        idx0 = blk * block
        width = min(block, F - idx0)
        Mb = _block_columns(master, trial, blk, d, block, dtype)[:width]
        Mb64 = Mb.astype(np.float64)
        S += Mb64.T @ Mb64
        v += Mb64.sum(axis=0)
    return _moments_from_S_v(S, v, d, F)


def tied_linear_energy(mom: TiedEnergyMoments, s: int) -> float:
    """``(1/F) E_S ||A 1_S||^2`` under a uniformly random support of size ``s``."""
    F = mom.F
    if not 1 <= s <= F:
        raise ValueError(f"need 1 <= s <= F, got s={s}, F={F}")
    p = s / F
    q = s * (s - 1) / (F * (F - 1.0))
    return float((p - q) * mom.frob_sq_A + q * mom.sum_A1_sq) / F


def interface_energy_moments(G: np.ndarray, Phi: np.ndarray,
                             eps: float = 1e-12) -> TiedEnergyMoments:
    """Same two moments for an *arbitrary* readout ``G``, still without forming ``M``.

    With ``Gbar = D^{-1} G`` the calibrated readout and ``A = Gbar Phi - I``,

        ||A||_F^2 = tr(H Sigma) - F,        H = Gbar^T Gbar,  Sigma = Phi Phi^T   (both d x d)
        ||A 1||^2 = ||Gbar (Phi 1)||^2 - 2 * 1^T Gbar (Phi 1) + F

    so the cost is ``O(F d^2)`` time and ``O(d^2 + F)`` memory. Setting ``G = Phi^T`` on a
    unit-norm code recovers :func:`tied_energy_moments` exactly; the two are cross-checked in
    ``tests/test_diagnostic.py``.
    """
    d, F = Phi.shape
    if G.shape != (F, d):
        raise ValueError(f"readout must be ({F}, {d}) for a ({d}, {F}) code; got {G.shape}")
    diag = gains(G, Phi)
    if np.any(np.abs(diag) < eps):
        bad = int(np.argmin(np.abs(diag)))
        raise ValueError(f"near-zero diagonal gain at index {bad} (|M_ii| = {abs(diag[bad]):.3e})")
    Gbar = (G / diag[:, None]).astype(np.float64)
    Phi64 = Phi.astype(np.float64)
    H = Gbar.T @ Gbar
    Sigma = Phi64 @ Phi64.T
    frob_sq_A = float(np.sum(H * Sigma)) - F

    row_sums = Gbar @ (Phi64.sum(axis=1))     # M 1, in O(Fd)
    sum_A1_sq = float(row_sums @ row_sums) - 2.0 * float(row_sums.sum()) + F
    return TiedEnergyMoments(d=int(d), F=int(F),
                             frob_sq_A=max(frob_sq_A, 0.0),
                             sum_A1_sq=max(sum_A1_sq, 0.0))


# ``tied_linear_energy`` depends only on the two moments, so it applies verbatim to the
# general case; the alias exists so call sites read correctly.
interface_linear_energy = tied_linear_energy


# --------------------------------------------------------------------------------------
# ALG-1
# --------------------------------------------------------------------------------------

def interface_floor_diagnostic(Phi: np.ndarray, G: Optional[np.ndarray] = None,
                               readout: str = "tied",
                               statistics: str = "full") -> Dict[str, Any]:
    """ALG-1. Calibrate the readout interface and locate it relative to the Welch floor.

    Parameters
    ----------
    Phi : ``(d, F)`` code.
    G : optional explicit ``(F, d)`` readout. Overrides ``readout``.
    readout : ``"tied"`` for ``G = Phi^T``, ``"pinv"`` for ``G = Phi^+``.
    statistics :
        ``"mean_sq"`` computes only the mean squared off-diagonal, from ``d x d`` sufficient
        statistics, in ``O(F d^2)`` time and ``O(d^2)`` memory -- the interface is never
        formed, and the maximum-statistic fields come back ``None``.
        ``"full"`` (default) additionally reports ``max_{i != j} |M_ij|``, which admits no such
        reduction and therefore forms ``M`` explicitly at ``O(F^2 d)`` time and ``O(F^2)``
        memory. Use ``"mean_sq"`` whenever ``F`` is large; the theorem being tested is a
        statement about the mean square, so nothing about the floor comparison is lost.

    Returns a dict with the calibrated cross-talk statistics, the floor, and the attainment
    ratios. ``ratio_mean_sq >= 1`` always holds (it is the theorem); a value below one
    signals a bug or a numerically singular calibration, and is flagged.
    """
    d, F = Phi.shape
    if F <= d:
        raise ValueError(f"the diagnostic requires an overcomplete code F > d; got F={F}, d={d}")
    if statistics not in ("full", "mean_sq"):
        raise ValueError(f"unknown statistics {statistics!r}; use 'full' or 'mean_sq'")
    if G is None:
        if readout == "tied":
            G = Phi.T
        elif readout == "pinv":
            G = np.linalg.pinv(Phi)
        else:
            raise ValueError(f"unknown readout {readout!r}; use 'tied', 'pinv' or pass G")
    else:
        readout = "explicit"  # an explicitly supplied G is not one of the named rules

    if statistics == "full":
        M = unit_diagonal(G @ Phi)
        out = crosstalk_stats(M, d).to_dict()
    else:
        mean_sq = crosstalk_mean_sq(G, Phi)
        fl = welch_floor(F, d)
        out = {
            "d": int(d), "F": int(F),
            "mean_sq_offdiag": mean_sq,
            "max_abs_offdiag": None,
            "floor_mean_sq": fl,
            "floor_max": welch_floor_max(F, d),
            "ratio_mean_sq": mean_sq / fl,
            "ratio_max": None,
            "violates_floor": bool(mean_sq < fl * (1.0 - 1e-9)),
        }
    out["readout"] = readout
    out["statistics"] = statistics
    return out


# --------------------------------------------------------------------------------------
# ALG-2: the diagnostic proper -- one fixed code, randomness only in the support
# --------------------------------------------------------------------------------------

def _profile_rows(sparsities: Sequence[int], successes: np.ndarray, energy_mean: np.ndarray,
                  trials: int, d: int, F: int, conf: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in sparsities:
        k = int(successes[s - 1])
        lo, hi = wilson_interval(k, trials, conf=conf)
        e = float(energy_mean[s - 1])
        rows.append({
            "s": s,
            "successes": k,
            "trials": trials,
            "p_rec": k / trials,
            "ci_low": lo,
            "ci_high": hi,
            "linear_energy_per_coord": e,
            "linear_rms_error": float(np.sqrt(e)),
            "energy_floor_uniform": energy_floor_uniform(F, d, s),
            "rms_floor_uniform": float(np.sqrt(energy_floor_uniform(F, d, s))),
            "rms_over_dminushalf": float(np.sqrt(e) * np.sqrt(d)),
        })
    return rows


def _separation_witness(rows: List[Dict[str, Any]], s95: int) -> Optional[Dict[str, Any]]:
    """The largest certified ``s``, packaged with both criteria's errors at that point."""
    if s95 <= 0:
        return None
    r = next(r for r in rows if r["s"] == s95)
    return {
        "s": s95,
        "support_recovery_p_rec": r["p_rec"],
        "support_recovery_error": 0.0,
        "analog_rms_error": r["linear_rms_error"],
        "analog_rms_floor": r["rms_floor_uniform"],
        "analog_rms_over_dminushalf": r["rms_over_dminushalf"],
        # legacy key names, kept so already-serialised results stay readable
        "threshold_p_rec": r["p_rec"],
        "threshold_error": 0.0,
        "linear_rms_error": r["linear_rms_error"],
        "linear_rms_floor": r["rms_floor_uniform"],
        "linear_rms_over_dminushalf": r["rms_over_dminushalf"],
    }


def fixed_code_separation_profile(Phi: np.ndarray, sparsities: Sequence[int], trials: int,
                                  seed: int, G: Optional[np.ndarray] = None,
                                  readout: str = "tied", theta: float = 0.5,
                                  conf: float = 0.95,
                                  calibrate: bool = True) -> Dict[str, Any]:
    """ALG-2. Separation profile of **one given** code and readout.

    This is the diagnostic as it applies to an object you already have: a designed code, an
    optimised code, or the effective code of a trained network. ``(Phi, G)`` are held fixed;
    the only randomness is the support, drawn uniformly at each trial. That is the difference
    from :func:`random_code_scaling_experiment`, which redraws the code every trial and so
    characterises an ensemble rather than a code.

    The support-recovery side is Monte Carlo over ``trials`` supports; the analog side is
    exact for this code, in closed form from the ``d x d`` moments, so it carries no Monte
    Carlo error at all.

    Returns the same record shape as the ensemble version, plus the ALG-1 floor statistics of
    the very same interface, so both halves of the diagnostic describe one object.
    """
    d, F = Phi.shape
    sparsities = [int(s) for s in sparsities]
    if not sparsities:
        raise ValueError("sparsities must be non-empty")
    s_max = max(sparsities)
    if s_max > F:
        raise ValueError(f"max sparsity {s_max} exceeds F={F}")

    if G is None:
        if readout == "tied":
            G = Phi.T
        elif readout == "pinv":
            G = np.linalg.pinv(Phi)
        else:
            raise ValueError(f"unknown readout {readout!r}; use 'tied', 'pinv' or pass G")
    else:
        readout = "explicit"
    G = np.ascontiguousarray(G, dtype=np.float64)
    if calibrate:
        G = G / gains(G, Phi)[:, None]

    mom = interface_energy_moments(G, Phi)
    energy = np.array([interface_linear_energy(mom, s) for s in range(1, s_max + 1)])

    rng = np.random.default_rng(seed)
    successes = np.zeros(s_max, dtype=np.int64)
    for _ in range(trials):
        successes += recovery_trial_fixed(Phi, G, s_max, rng, theta=theta).astype(np.int64)

    rows = _profile_rows(sparsities, successes, energy, trials, d, F, conf)
    s95 = s95_from_curve(sparsities, [r["p_rec"] for r in rows])
    floor_stats = interface_floor_diagnostic(Phi, G=G, statistics="mean_sq")

    return {
        "kind": "fixed_code",
        "d": d,
        "F": F,
        "theta": theta,
        "trials": trials,
        "seed": seed,
        "readout": readout,
        "calibrated": bool(calibrate),
        "welch_floor_mean_sq": welch_floor(F, d),
        "welch_floor_max": welch_floor_max(F, d),
        "frob_sq_A": mom.frob_sq_A,
        "mean_crosstalk": floor_stats["mean_sq_offdiag"],
        "ratio_mean_sq": floor_stats["ratio_mean_sq"],
        "rows": rows,
        "s95": s95,
        "separation_witness": _separation_witness(rows, s95),
    }


# --------------------------------------------------------------------------------------
# The random-code ensemble: a scaling *experiment*, not the diagnostic
# --------------------------------------------------------------------------------------

def random_code_scaling_experiment(d: int, F: int, sparsities: Sequence[int], trials: int,
                                   master_seed: int, theta: float = 0.5,
                                   block: int = DEFAULT_BLOCK, dtype=np.float32,
                                   conf: float = 0.95,
                                   progress: Optional[Any] = None) -> Dict[str, Any]:
    """Support-recovery curve and analog error of the tied readout, over a random ensemble.

    A fresh random unit-norm code is drawn for **every trial** (streamed, never stored), so
    what this measures is a property of the ``(d, F)`` random-code ensemble, not of any one
    code. It answers "how does the separation scale with width", which is what E2 and E6 are
    for; it is *not* the diagnostic, and must not be described as applying to a given code.
    Use :func:`fixed_code_separation_profile` for that.

    The support-recovery side is Monte Carlo; the analog side is exact per drawn code (closed
    form from the frame operator) and then averaged over the same trials.
    """
    sparsities = [int(s) for s in sparsities]
    if not sparsities:
        raise ValueError("sparsities must be non-empty")
    s_max = max(sparsities)
    if s_max > F:
        raise ValueError(f"max sparsity {s_max} exceeds F={F}")

    successes = np.zeros(s_max, dtype=np.int64)
    energy_sum = np.zeros(s_max, dtype=np.float64)
    frob_sq_sum = 0.0

    for t in range(trials):
        ok = recovery_trial_streaming(master_seed, t, d, F, s_max, theta=theta,
                                      block=block, dtype=dtype)
        successes += ok.astype(np.int64)
        mom = tied_energy_moments_streaming(master_seed, t, d, F, block=block, dtype=dtype)
        frob_sq_sum += mom.frob_sq_A
        for s in range(1, s_max + 1):
            energy_sum[s - 1] += tied_linear_energy(mom, s)
        if progress is not None:
            progress(t + 1, trials)

    rows = _profile_rows(sparsities, successes, energy_sum / trials, trials, d, F, conf)
    s95 = s95_from_curve(sparsities, [r["p_rec"] for r in rows])
    cert = _separation_witness(rows, s95)

    return {
        "kind": "random_code_ensemble",
        "d": d,
        "F": F,
        "theta": theta,
        "trials": trials,
        "master_seed": master_seed,
        "dtype": np.dtype(dtype).name,
        "welch_floor_mean_sq": welch_floor(F, d),
        "welch_floor_max": welch_floor_max(F, d),
        "mean_frob_sq_A": frob_sq_sum / trials,
        "mean_crosstalk_tied": (frob_sq_sum / trials) / (F * (F - 1.0)),
        "rows": rows,
        "s95": s95,
        "separation_witness": cert,
        "separation_certificate": cert,  # previous name; kept so E2/E6 records stay readable
    }


# Former name of the routine above. It was called ALG-2 in the first version of this work, which
# conflated the ensemble experiment with the per-code diagnostic; the alias keeps older scripts
# working while the two are named apart.
interface_separation_profile = random_code_scaling_experiment
