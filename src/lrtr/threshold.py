"""Thresholded Boolean recovery: the second decoding interface studied in the manuscript.

Two implementations of a recovery trial are provided and are checked against each other in
``tests/test_threshold.py``:

* :func:`recovery_trial_dense` materialises the code ``Phi`` and computes ``z = Phi.T @ x``;
* :func:`recovery_trial_streaming` never materialises ``Phi``. It regenerates the code in
  blocks from a counted seed sequence and evaluates every sparsity of one trial in a single
  pass using nested supports. This is what makes ``d = 1024`` (``F = 2^20``, a 4 GB code)
  feasible on a laptop CPU.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "threshold_decode",
    "recovery_trial_dense",
    "recovery_trial_streaming",
    "recovery_trial_fixed",
    "s95_from_curve",
    "s95_interpolated",
]

DEFAULT_BLOCK = 1 << 16


def threshold_decode(z: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Threshold decoder ``Q(z)_i = 1{z_i >= theta}``."""
    return (z >= theta).astype(np.int8)


def _block_columns(master: int, trial: int, blk: int, d: int, block: int,
                   dtype=np.float32) -> np.ndarray:
    """Rows of the returned ``(block, d)`` array are the code columns of block ``blk``.

    Reproducible for a given ``(master, trial, blk)`` without materialising other blocks.
    """
    ss = np.random.SeedSequence(entropy=master, spawn_key=(trial, blk))
    rng = np.random.default_rng(ss)
    Mb = rng.standard_normal((block, d)).astype(dtype, copy=False)
    Mb /= np.linalg.norm(Mb, axis=1, keepdims=True)
    return Mb


def _support_seed(master: int, trial: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=master, spawn_key=(trial, 2 ** 62)))


def recovery_trial_dense(d: int, F: int, s: int, rng: np.random.Generator,
                         theta: float = 0.5, dtype=np.float64) -> bool:
    """One exact-recovery trial with a freshly drawn dense code. Returns success."""
    Phi = rng.standard_normal((d, F)).astype(dtype, copy=False)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    S = rng.choice(F, size=s, replace=False)
    x = Phi[:, S].sum(axis=1)
    z = Phi.T @ x
    b_hat = threshold_decode(z, theta)
    b = np.zeros(F, dtype=np.int8)
    b[S] = 1
    return bool(np.array_equal(b_hat, b))


def recovery_trial_streaming(master: int, trial: int, d: int, F: int, s_max: int,
                             theta: float = 0.5, block: int = DEFAULT_BLOCK,
                             dtype=np.float32) -> np.ndarray:
    """Evaluate sparsities ``1..s_max`` of one trial in a single streaming pass.

    Uses nested supports ``S_1 subset ... subset S_{s_max}``: the marginal law of each
    ``S_s`` is exactly uniform over size-``s`` supports, so every point of the resulting
    ``p_rec(s)`` curve is unbiased. Points within a trial are correlated by construction;
    this is documented in the manuscript and affects only the joint, not the marginals.

    Returns a boolean array ``ok`` of length ``s_max`` with ``ok[s-1]`` the success at
    sparsity ``s``.
    """
    if s_max < 1:
        raise ValueError(f"s_max must be >= 1, got {s_max}")
    if F < s_max:
        raise ValueError(f"F must be at least s_max, got F={F}, s_max={s_max}")

    rng = _support_seed(master, trial)
    S = rng.choice(F, size=s_max, replace=False)

    n_blocks = (F + block - 1) // block

    # Regenerate only the blocks containing support indices, once each, to build the
    # partial-sum matrix X with column s-1 equal to x_s = sum_{j<=s} phi_{S_j}.
    order = np.argsort(S // block, kind="stable")
    cols = np.empty((s_max, d), dtype=dtype)
    cur_blk = -1
    Mb = None
    for pos in order:
        j = int(S[pos])
        blk = j // block
        if blk != cur_blk:
            Mb = _block_columns(master, trial, blk, d, block, dtype)
            cur_blk = blk
        cols[pos] = Mb[j % block]
    X = np.cumsum(cols, axis=0).T.copy()  # (d, s_max)

    ok = np.ones(s_max, dtype=bool)
    # Local positions of the active indices inside each block, per sparsity level.
    for blk in range(n_blocks):
        idx0 = blk * block
        idx1 = min(idx0 + block, F)
        width = idx1 - idx0
        Mb = _block_columns(master, trial, blk, d, block, dtype)
        if width < block:
            Mb = Mb[:width]
        Z = Mb @ X  # (width, s_max)
        in_block = (S >= idx0) & (S < idx1)
        local = S[in_block] - idx0
        # position of each in-block support element within the nested ordering
        levels = np.nonzero(in_block)[0]
        for s in range(s_max):
            if not ok[s]:
                continue
            zs = Z[:, s]
            pred_on = zs >= theta
            n_pred = int(pred_on.sum())
            sel = levels <= s
            act_local = local[sel]
            n_act = int(act_local.size)
            if n_act and not bool(pred_on[act_local].all()):
                ok[s] = False
                continue
            if n_pred != n_act:
                ok[s] = False
        if not ok.any():
            break
    return ok


def recovery_trial_fixed(Phi: np.ndarray, G: np.ndarray, s_max: int,
                         rng: np.random.Generator, theta: float = 0.5) -> np.ndarray:
    """Evaluate sparsities ``1..s_max`` of one trial on a *fixed* code and readout.

    The streaming and dense trials above redraw the code every trial, so they measure a random
    ensemble. This one holds ``(Phi, G)`` fixed and randomises only the support, which is what
    a diagnostic applied to one given code -- a trained network's effective code, say -- has to
    do. Uses the same nested supports as :func:`recovery_trial_streaming`, so ``ok[s-1]`` is an
    unbiased sample of success at sparsity ``s``.

    ``G`` is used as supplied; calibrate it beforehand if the decision rule should be applied
    to the unit-diagonal interface.
    """
    d, F = Phi.shape
    if G.shape != (F, d):
        raise ValueError(f"readout must be ({F}, {d}) for a ({d}, {F}) code; got {G.shape}")
    if not 1 <= s_max <= F:
        raise ValueError(f"need 1 <= s_max <= F, got s_max={s_max}, F={F}")

    S = rng.choice(F, size=s_max, replace=False)
    X = np.cumsum(Phi[:, S], axis=1)          # (d, s_max); column s-1 is x_s
    Z = G @ X                                 # (F, s_max)

    ok = np.empty(s_max, dtype=bool)
    for s in range(1, s_max + 1):
        b_hat = threshold_decode(Z[:, s - 1], theta)
        b = np.zeros(F, dtype=np.int8)
        b[S[:s]] = 1
        ok[s - 1] = bool(np.array_equal(b_hat, b))
    return ok


def s95_from_curve(sparsities: Sequence[int], probs: Sequence[float],
                   level: float = 0.95) -> int:
    """Largest ``s`` in the grid with ``p_rec(s) >= level``, requiring a contiguous prefix.

    Returns 0 if even the smallest tested sparsity is below ``level``. Requiring the prefix
    to be contiguous avoids reporting an isolated lucky point past the transition.

    Note this is a *floored grid* statistic: the true crossing usually lies between two grid
    points, so it systematically under-reports, and it does so worst where the grid is coarse
    relative to the transition (small ``d``). Use :func:`s95_interpolated` alongside it when
    comparing across ``d``.
    """
    best = 0
    for s, p in zip(sparsities, probs):
        if p >= level:
            best = int(s)
        else:
            break
    return best


def s95_interpolated(sparsities: Sequence[int], probs: Sequence[float],
                     level: float = 0.95) -> float:
    """Linearly interpolated sparsity at which ``p_rec`` crosses ``level``.

    Removes the flooring bias of :func:`s95_from_curve`, which is not uniform across widths:
    a coarse grid relative to the transition truncates more, so comparing floored values
    across ``d`` bends the apparent growth curve upward. Returns 0.0 if the first grid point
    is already below ``level`` (no crossing observed).
    """
    pairs = [(float(s), float(p)) for s, p in zip(sparsities, probs)]
    if not pairs or pairs[0][1] < level:
        return 0.0
    for (s0, p0), (s1, p1) in zip(pairs, pairs[1:]):
        if p1 < level:
            if p0 == p1:  # flat segment: crossing is not localisable, take the left point
                return s0
            return s0 + (p0 - level) / (p0 - p1) * (s1 - s0)
    return pairs[-1][0]  # never dropped below `level` on the tested grid
