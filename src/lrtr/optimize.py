"""E4: gradient-optimised codes under the unit-diagonal constraint.

Four variants, all CPU-only:

``free``
    ``G`` and ``Phi`` independent; loss is the mean squared off-diagonal of the calibrated
    interface ``M = D^{-1} G Phi``.
``tied``
    ``G = Phi^T`` with unit-norm columns, so the interface is the Gram matrix. The
    tight-frame residual is reported to test whether a tight frame emerges.
``softmax``
    Smooth-maximum loss with temperature annealing, targeting the ``max_{i != j} |M_ij|``
    form of the bound rather than the mean-square form.
``uncalibrated`` (ablation A1)
    Same as ``free`` but the loss is the raw off-diagonal energy of ``G Phi`` with **no**
    diagonal calibration. The optimiser is then free to shrink the gains, so the raw
    off-diagonal energy falls below the floor while the calibrated interface does not. This
    is the experimental form of the calibration-free corollary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .codes import welch_floor, welch_floor_max

__all__ = ["VARIANTS", "train_code"]

VARIANTS = ("free", "tied", "softmax", "uncalibrated")


def _calibrated(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable ``D^{-1} M`` with a sign-preserving floor on the diagonal gain."""
    diag = torch.diagonal(M)
    sign = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    safe = sign * diag.abs().clamp_min(eps)
    return M / safe.unsqueeze(1)


def train_code(d: int, F: int, variant: str = "free", steps: int = 20000, lr: float = 1e-2,
               seed: int = 0, tau0: float = 1e-2, tau1: float = 1e-4,
               record_every: int = 200, device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    """Optimise a code/readout pair and report its position relative to the Welch floor.

    Returns a dict with the loss trajectory (as ratio to the floor), the final calibrated
    cross-talk statistics, and -- for the tied variant -- the tight-frame residual.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if F <= d:
        raise ValueError(f"need F > d, got F={F}, d={d}")

    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    W = welch_floor(F, d)
    W_max = welch_floor_max(F, d)

    Phi = torch.randn(d, F, generator=g, device=device, dtype=dtype)
    Phi = Phi / Phi.norm(dim=0, keepdim=True)
    Phi.requires_grad_(True)
    params: List[torch.Tensor] = [Phi]
    G: Optional[torch.Tensor] = None
    if variant != "tied":
        G = torch.randn(F, d, generator=g, device=device, dtype=dtype) / np.sqrt(d)
        G.requires_grad_(True)
        params.append(G)

    opt = torch.optim.Adam(params, lr=lr)
    eye = torch.eye(F, device=device, dtype=dtype)
    n_off = F * (F - 1)

    traj_steps: List[int] = []
    traj_ratio: List[float] = []

    for t in range(steps):
        if variant == "tied":
            Pn = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
            M = Pn.T @ Pn
            Mc = M
        else:
            M = G @ Phi
            Mc = M if variant == "uncalibrated" else _calibrated(M)

        off = Mc - eye
        off_sq = off * off
        if variant == "softmax":
            frac = t / max(steps - 1, 1)
            tau = tau0 * (tau1 / tau0) ** frac
            vals = off_sq.flatten()[:-1].view(F - 1, F + 1)[:, 1:].reshape(-1)  # strict off-diag
            loss = tau * torch.logsumexp(vals / tau, dim=0)
        else:
            loss = (off_sq.sum() - torch.diagonal(off_sq).sum()) / n_off

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if t % record_every == 0 or t == steps - 1:
            with torch.no_grad():
                traj_steps.append(t)
                traj_ratio.append(float(_mean_sq_offdiag(Mc, eye, n_off) / W))

    with torch.no_grad():
        if variant == "tied":
            Pn = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
            M_raw = (Pn.T @ Pn).double()
            Phi_final = Pn
        else:
            M_raw = (G @ Phi).double()
            Phi_final = Phi
        eye64 = torch.eye(F, dtype=torch.float64)
        diag_raw = torch.diagonal(M_raw)
        # Exact calibration in float64, with no clamping: the ratio is only meaningful
        # while every diagonal gain is non-degenerate. The uncalibrated ablation drives the
        # gains to zero on purpose, so this is reported rather than silently patched.
        min_gain = float(diag_raw.abs().min())
        degenerate = min_gain < 1e-8
        if degenerate:
            mean_sq_cal = float("nan")
            max_cal = float("nan")
        else:
            M_cal = M_raw / diag_raw.unsqueeze(1)
            mean_sq_cal = _mean_sq_offdiag(M_cal, eye64, n_off)
            max_cal = _max_abs_offdiag(M_cal, eye64)
        mean_sq_raw = _mean_sq_offdiag(M_raw, eye64, n_off)
        out: Dict[str, Any] = {
            "d": d, "F": F, "variant": variant, "seed": seed, "steps": steps, "lr": lr,
            "welch_floor_mean_sq": W,
            "welch_floor_max": W_max,
            "final_mean_sq_offdiag_calibrated": mean_sq_cal,
            "final_max_abs_offdiag_calibrated": max_cal,
            "ratio_mean_sq": mean_sq_cal / W,
            "ratio_max": max_cal / W_max,
            "final_mean_sq_offdiag_raw": mean_sq_raw,
            "ratio_mean_sq_raw": mean_sq_raw / W,
            "diag_gain_mean": float(diag_raw.mean()),
            "diag_gain_min_abs": min_gain,
            "interface_degenerate": bool(degenerate),
            "trajectory_steps": traj_steps,
            "trajectory_ratio_mean_sq": traj_ratio,
        }
        if variant == "tied":
            S = Phi_final @ Phi_final.T
            target = (F / d) * torch.eye(d, device=device, dtype=dtype)
            out["tight_frame_residual"] = float(
                torch.linalg.norm(S - target) / torch.linalg.norm(target))
    return out


def _mean_sq_offdiag(M: torch.Tensor, eye: torch.Tensor, n_off: int) -> float:
    off = M - eye
    off_sq = off * off
    return float((off_sq.sum() - torch.diagonal(off_sq).sum()) / n_off)


def _max_abs_offdiag(M: torch.Tensor, eye: torch.Tensor) -> float:
    off = (M - eye).abs()
    off = off - torch.diag(torch.diagonal(off))
    return float(off.max())
