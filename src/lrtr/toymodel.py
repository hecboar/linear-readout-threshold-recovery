"""E5: the compressed-computation toy model and its interface diagnostic.

A width-``d`` network is trained to compute ``F`` sparse ReLU features,
``y* = ReLU(x)`` coordinatewise, from ``x in R^F`` with ``F > d``:

    y_hat = W_out @ ReLU(W_in @ x),    W_in in R^{d x F},  W_out in R^{F x d}.

This is the task introduced by Braun et al. (2025) as a testbed for compressed computation.
Two training losses are compared: ``L2`` (the standard objective, the model that Bhagat et
al. (2026) argue does not genuinely compute in superposition) and ``L4`` (the variant that
Ferreira da Silva and Heimersheim (2026) report elicits a solution that does).

The diagnostic then treats ``W_in`` as the effective code ``Phi`` and measures, on Boolean
sparse states, (i) the calibrated cross-talk of three linear readouts against the Welch
floor, and (ii) the two interface criteria -- exact threshold recovery versus per-coordinate
linear error -- on the *same* representation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .codes import welch_floor, welch_floor_max
from .interface import (
    crosstalk_stats,
    energy_floor_uniform,
    linear_energy_uniform,
    unit_diagonal,
)
from .stats import wilson_interval
from .threshold import s95_from_curve

__all__ = ["train_toy_model", "train_toy_models_batched", "sample_task_batch",
           "diagnose_model", "random_code_baseline", "TASKS", "target_of",
           "linear_readouts", "linearity_report"]

# The elementwise target the network computes through the bottleneck. Only the ReLU target of
# Braun et al. (2025) is here, deliberately. Earlier drafts added `abs` and `square` as a cheap
# "second task"; they were withdrawn (decision D6) because a second task invented by us is not
# independent evidence of generality -- it shares the architecture, the training loop and the
# input distribution, so it can only confirm. Generality is tested against a second system with
# a published, independently characterised mechanism, or not claimed.
TASKS = {
    "relu": lambda x: torch.relu(x),
}


def target_of(x: torch.Tensor, task: str = "relu") -> torch.Tensor:
    """The elementwise target for ``task``; raises on an unknown name rather than defaulting."""
    try:
        return TASKS[task](x)
    except KeyError:
        raise ValueError(f"unknown task {task!r}; known tasks are {sorted(TASKS)}") from None


def sample_task_batch(F: int, batch: int, p: float, gen: torch.Generator,
                      dtype: torch.dtype = torch.float32,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """Compressed-computation inputs: each coordinate active w.p. ``p``, value ``~U[-1, 1]``.

    ``device`` must match the device of ``gen``; it defaults to the generator's own device so
    that CPU and CUDA runs both draw from their own stream without an implicit transfer.
    """
    if device is None:
        device = gen.device
    mask = (torch.rand(batch, F, generator=gen, dtype=dtype, device=device) < p).to(dtype)
    vals = torch.rand(batch, F, generator=gen, dtype=dtype, device=device) * 2.0 - 1.0
    return mask * vals


def train_toy_model(d: int, F: int, loss_kind: str = "L2", p: float = 0.01,
                    steps: int = 50000, batch: int = 2048, lr: float = 1e-3,
                    seed: int = 0, log_every: int = 5000, task: str = "relu",
                    dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    """Train ``y_hat = W_out ReLU(W_in x)`` under an ``L2`` or ``L4`` loss on CPU."""
    if loss_kind not in ("L2", "L4"):
        raise ValueError(f"loss_kind must be 'L2' or 'L4', got {loss_kind!r}")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    W_in = (torch.randn(d, F, generator=gen, dtype=dtype) / np.sqrt(F)).requires_grad_(True)
    W_out = (torch.randn(F, d, generator=gen, dtype=dtype) / np.sqrt(d)).requires_grad_(True)
    opt = torch.optim.Adam([W_in, W_out], lr=lr)

    history: List[Dict[str, float]] = []
    exponent = 2 if loss_kind == "L2" else 4
    for t in range(steps):
        x = sample_task_batch(F, batch, p, gen, dtype)
        target = target_of(x, task)
        pred = torch.relu(x @ W_in.T) @ W_out.T
        loss = ((pred - target) ** exponent).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if t % log_every == 0 or t == steps - 1:
            with torch.no_grad():
                mse = float(((pred - target) ** 2).mean())
            history.append({"step": t, "loss": float(loss.detach()), "mse": mse})

    with torch.no_grad():
        x = sample_task_batch(F, 8192, p, gen, dtype)
        target = target_of(x, task)
        pred = torch.relu(x @ W_in.T) @ W_out.T
        final_mse = float(((pred - target) ** 2).mean())
        # Baseline for context: the trivial predictor y_hat = 0 on the same batch.
        zero_mse = float((target ** 2).mean())

    return {
        "d": d, "F": F, "loss_kind": loss_kind, "p": p, "steps": steps,
        "batch": batch, "lr": lr, "seed": seed, "task": task,
        "W_in": W_in.detach().numpy().astype(np.float64),
        "W_out": W_out.detach().numpy().astype(np.float64),
        "history": history,
        "final_mse": final_mse,
        "zero_predictor_mse": zero_mse,
        "mse_ratio_to_zero_predictor": final_mse / zero_mse,
    }


def train_toy_models_batched(d: int, F: int, loss_kind: str, p: float, seeds: Sequence[int],
                             steps: int = 50000, batch: int = 2048, lr: float = 1e-3,
                             task: str = "relu", device: str = "cpu",
                             dtype: torch.dtype = torch.float32, log_every: int = 5000,
                             progress: Optional[Any] = None,
                             cpu_data: bool = False) -> List[Dict[str, Any]]:
    """Train ``len(seeds)`` independent models simultaneously as one batched computation.

    On CPU each model is *exactly* the model :func:`train_toy_model` would produce for that
    seed -- same initialisation, same per-step data, same optimiser trajectory. The only change
    is that the ``S`` forward passes are issued as one batched matmul and the ``S`` losses are
    summed, which leaves every per-seed gradient untouched because the seeds share no
    parameters. This matters on an accelerator: at these widths a single model leaves the device
    almost idle, while ``S`` of them saturate it, so the seed count stops being the thing that
    limits the study. ``tests/test_batched_training.py`` asserts the equivalence in float64.

    Every seed keeps its own generator, so adding seeds never perturbs the existing ones and a
    campaign can be extended without invalidating what has already been run.

    **What a seed names on an accelerator.** CUDA generators use a different algorithm from CPU
    ones, so the same seed gives a different stream on each. The initialisation is therefore
    always drawn on the CPU stream and then moved: ``W_in`` is the object every theoretical
    quantity is computed from -- leverage, the collision frontier, the geometry ratio -- so
    "seed 0" must name the same code whether or not an accelerator was used. The per-step
    batches are drawn on the compute device, because at these widths drawing 32M uniforms per
    step on the host costs two orders of magnitude more than the training step it feeds. A run
    is thus reproducible from its seed on the same device class, and ``device`` is recorded in
    the returned record. Pass ``cpu_data=True`` to draw the batches on the CPU stream as well,
    which makes a GPU run bit-identical to the CPU one at a large throughput cost; that is how
    ``scripts/check_env_gpu.py`` isolates arithmetic disagreement from stream divergence.
    """
    if loss_kind not in ("L2", "L4"):
        raise ValueError(f"loss_kind must be 'L2' or 'L4', got {loss_kind!r}")
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known tasks are {sorted(TASKS)}")
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("seeds must be non-empty")
    dev = torch.device(device)
    S = len(seeds)

    # Always the CPU stream, so that a seed names the same initial code on every device.
    init_gens = [torch.Generator().manual_seed(s) for s in seeds]
    W_in = torch.stack([torch.randn(d, F, generator=g, dtype=dtype) / np.sqrt(F)
                        for g in init_gens]).to(dev).requires_grad_(True)
    W_out = torch.stack([torch.randn(F, d, generator=g, dtype=dtype) / np.sqrt(d)
                         for g in init_gens]).to(dev).requires_grad_(True)
    # On CPU the batches continue the very same stream, which is what makes a CPU run
    # byte-identical to train_toy_model and keeps every committed result reproducible.
    if dev.type == "cpu" or cpu_data:
        gens = init_gens
    else:
        gens = [torch.Generator(device=dev).manual_seed(s) for s in seeds]
    opt = torch.optim.Adam([W_in, W_out], lr=lr)

    exponent = 2 if loss_kind == "L2" else 4
    history: List[List[Dict[str, float]]] = [[] for _ in seeds]

    for t in range(steps):
        x = torch.stack([sample_task_batch(F, batch, p, g, dtype) for g in gens]).to(dev)
        target = target_of(x, task)
        h = torch.relu(torch.einsum("sbf,sdf->sbd", x, W_in))
        pred = torch.einsum("sbd,sfd->sbf", h, W_out)
        per_seed = ((pred - target) ** exponent).mean(dim=(1, 2))   # (S,)
        # Summing keeps each seed's gradient identical to training it alone; averaging would
        # divide every gradient by S, which Adam almost but not exactly absorbs.
        opt.zero_grad(set_to_none=True)
        per_seed.sum().backward()
        opt.step()
        if t % log_every == 0 or t == steps - 1:
            with torch.no_grad():
                mse = ((pred - target) ** 2).mean(dim=(1, 2))
            for i in range(S):
                history[i].append({"step": t, "loss": float(per_seed[i].detach()),
                                   "mse": float(mse[i])})
            if progress is not None:
                progress(t + 1, steps)

    with torch.no_grad():
        x = torch.stack([sample_task_batch(F, 8192, p, g, dtype) for g in gens]).to(dev)
        target = target_of(x, task)
        h = torch.relu(torch.einsum("sbf,sdf->sbd", x, W_in))
        pred = torch.einsum("sbd,sfd->sbf", h, W_out)
        final_mse = ((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy()
        zero_mse = (target ** 2).mean(dim=(1, 2)).cpu().numpy()

    Win = W_in.detach().cpu().numpy().astype(np.float64)
    Wout = W_out.detach().cpu().numpy().astype(np.float64)
    return [{
        "d": d, "F": F, "loss_kind": loss_kind, "p": p, "steps": steps,
        "batch": batch, "lr": lr, "seed": seeds[i], "task": task, "device": str(dev),
        "W_in": Win[i], "W_out": Wout[i],
        "history": history[i],
        "final_mse": float(final_mse[i]),
        "zero_predictor_mse": float(zero_mse[i]),
        "mse_ratio_to_zero_predictor": float(final_mse[i] / zero_mse[i]),
    } for i in range(S)]


def random_code_baseline(d: int, F: int, seed: int) -> Dict[str, Any]:
    """Control: an i.i.d. random unit-norm code with the same ``(d, F)``, scaled like W_in."""
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((d, F))
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    return {"d": d, "F": F, "loss_kind": "random", "seed": seed,
            "W_in": Phi, "W_out": np.linalg.pinv(Phi), "final_mse": None,
            "zero_predictor_mse": None, "mse_ratio_to_zero_predictor": None,
            "history": []}


# --------------------------------------------------------------------------------------
# Interface diagnostic on a trained (or control) model
# --------------------------------------------------------------------------------------

READOUT_NAMES = ("pinv", "wout", "ls")


def _readouts(W_in: np.ndarray, W_out: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    """The three linear readouts of the *linear* representation ``x = Phi b``.

    ``pinv``  algebraic pseudoinverse of the effective code;
    ``wout``  the model's own decoder, applied to the linear representation;
    ``ls``    least squares fit of ``G (Phi B) ~ B`` on Boolean sparse states -- the honest
              best-linear-readout baseline, deliberately adversarial to our own hypothesis.

    All three are read out from ``Phi b``, never from ``ReLU(Phi b)``, so that the Welch floor
    genuinely applies to each of them.
    """
    d = W_in.shape[0]
    X = W_in @ B
    XXt = X @ X.T + 1e-10 * np.eye(d)
    G_ls = np.linalg.solve(XXt.T, (B @ X.T).T).T
    return {"pinv": np.linalg.pinv(W_in), "wout": W_out, "ls": G_ls}


# Public name for the three readouts, so campaigns outside this module can score the same
# three interfaces without reaching into a private helper.
linear_readouts = _readouts


def _linearity_report(W_in: np.ndarray, W_out: np.ndarray, B: np.ndarray,
                      theta: float) -> Dict[str, float]:
    """How close the trained pipeline is to being a plain linear readout.

    The Welch floor constrains linear readout interfaces. A network whose ReLU *clips* almost
    nothing -- so the nonlinearity passes its input through essentially unchanged -- and whose
    decoder is close to ``Phi^+`` is, for these inputs, *maintaining* a linear readout
    interface and is therefore bound by the floor; one that clips heavily is not.

    ``frac_relu_clipped`` is the fraction of preactivations the ReLU sets to zero. A value
    near 0 means "behaves linearly"; an i.i.d. random code sits near 0.5.
    """
    Gp = np.linalg.pinv(W_in)
    pre = W_in @ B
    h = np.maximum(pre, 0.0)
    y_model = W_out @ h
    z_lin = Gp @ pre
    same = np.all((y_model >= theta) == (z_lin >= theta), axis=0)
    return {
        "frac_relu_clipped": float((pre < 0).mean()),
        "rel_distance_wout_to_pinv": float(np.linalg.norm(W_out - Gp) / np.linalg.norm(Gp)),
        "decision_agreement_with_linear": float(same.mean()),
        "median_abs_gap_model_vs_linear": float(np.median(np.abs(y_model - z_lin).max(axis=0))),
        "mean_code_column_norm": float(np.linalg.norm(W_in, axis=0).mean()),
    }


def diagnose_model(model: Dict[str, Any], sparsities: Sequence[int], trials: int,
                   seed: int, theta: float = 0.5, n_fit: int = 4096) -> Dict[str, Any]:
    """Run the Interface Diagnostic on a trained model. Fixed-code: only supports are random.

    Two criteria are compared at each sparsity, on the same Boolean sparse states:

    * the **analog criterion** -- scores ``z = M b`` from the calibrated linear interface
      ``M = D^{-1} G Phi``. Its per-coordinate reconstruction error is computed in closed form
      and is bounded below by the Welch-derived energy floor, which it must respect.
    * the **support-recovery criterion** -- the network's own nonlinear output
      ``y = W_out ReLU(W_in b)``, thresholded. The floor does not apply to it, because the map
      from ``b`` to ``y`` is not a rank-``d`` linear interface.

    The contrast is *analog reconstruction versus support recovery*, not "linear versus
    nonlinear": thresholding the analog scores would itself be a linear-threshold rule, and
    that rule is reported here too (``p_rec_linear_*``). What the floor forbids is small
    real-valued error, not correct sign patterns.

    ``W_in`` and ``W_out`` are held fixed throughout, so this is the fixed-code diagnostic of
    :func:`lrtr.diagnostic.fixed_code_separation_profile`, specialised to a network whose
    nonlinear path also has to be evaluated. Evaluation is on Boolean states ``x = 1_S`` (the
    model of the theory), deliberately outside the training distribution; this is stated as a
    limitation in the manuscript.
    """
    W_in = np.asarray(model["W_in"], dtype=np.float64)
    W_out = np.asarray(model["W_out"], dtype=np.float64)
    d, F = W_in.shape
    rng = np.random.default_rng(seed)
    sparsities = [int(s) for s in sparsities]

    # ---- fitting set for the least-squares readout, Boolean sparse states ----
    # Every feature must appear at least once, otherwise its row of the fitted readout is
    # identically zero and the unit-diagonal calibration is undefined for that coordinate.
    s_fit = max(1, int(np.median(sparsities)))
    n_fit = max(n_fit, int(np.ceil(2 * F / s_fit)))
    Bf = np.zeros((F, n_fit))
    for n in range(n_fit):
        Bf[rng.choice(F, size=s_fit, replace=False), n] = 1.0
    missing = np.nonzero(Bf.sum(axis=1) == 0)[0]
    for i, feat in enumerate(missing):
        Bf[feat, i % n_fit] = 1.0
    if np.any(Bf.sum(axis=1) == 0):
        raise RuntimeError("could not cover every feature in the readout fitting set")
    Gs = _readouts(W_in, W_out, Bf)
    linearity = _linearity_report(W_in, W_out, Bf, theta)

    # ---- (i) calibrated cross-talk of each linear readout against the Welch floor ----
    floor_stats: Dict[str, Any] = {}
    interfaces: Dict[str, np.ndarray] = {}
    for name, G in Gs.items():
        try:
            M = unit_diagonal(G @ W_in)
            interfaces[name] = M
            floor_stats[name] = crosstalk_stats(M, d).to_dict()
        except ValueError as exc:  # near-zero diagonal gain: interface undefined
            floor_stats[name] = {"error": str(exc)}

    # ---- (ii) the two criteria at each sparsity ----
    rows: List[Dict[str, Any]] = []
    for s in sparsities:
        succ_model = 0
        succ_lin = {k: 0 for k in interfaces}
        for _ in range(trials):
            S = rng.choice(F, size=s, replace=False)
            b = np.zeros(F)
            b[S] = 1.0
            bi = b.astype(np.int8)
            y = W_out @ np.maximum(W_in @ b, 0.0)
            succ_model += int(np.array_equal((y >= theta).astype(np.int8), bi))
            for name, M in interfaces.items():
                z = M @ b
                succ_lin[name] += int(np.array_equal((z >= theta).astype(np.int8), bi))
        lo, hi = wilson_interval(succ_model, trials)
        floor_e = energy_floor_uniform(F, d, s)
        row: Dict[str, Any] = {
            "s": s, "trials": trials,
            "p_rec_model": succ_model / trials,
            "ci_low_model": lo, "ci_high_model": hi,
            "energy_floor_uniform": floor_e,
            "rms_floor_uniform": float(np.sqrt(floor_e)),
        }
        for name, M in interfaces.items():
            A = M - np.eye(F)
            e = linear_energy_uniform(A, s)
            l2, h2 = wilson_interval(succ_lin[name], trials)
            row[f"p_rec_linear_{name}"] = succ_lin[name] / trials
            row[f"ci_low_linear_{name}"] = l2
            row[f"ci_high_linear_{name}"] = h2
            row[f"linear_energy_{name}"] = e
            row[f"linear_rms_{name}"] = float(np.sqrt(e))
            row[f"linear_energy_over_floor_{name}"] = e / floor_e
            row[f"linear_rms_over_dminushalf_{name}"] = float(np.sqrt(e) * np.sqrt(d))
        rows.append(row)

    out: Dict[str, Any] = {
        "d": d, "F": F, "loss_kind": model["loss_kind"], "seed": model.get("seed", seed),
        "theta": theta, "trials": trials, "sparsities": sparsities,
        "welch_floor_mean_sq": welch_floor(F, d),
        "welch_floor_max": welch_floor_max(F, d),
        "final_mse": model.get("final_mse"),
        "mse_ratio_to_zero_predictor": model.get("mse_ratio_to_zero_predictor"),
        "linearity": linearity,
        "floor_stats": floor_stats,
        "rows": rows,
    }
    out["s95_model"] = s95_from_curve(sparsities, [r["p_rec_model"] for r in rows])
    for name in READOUT_NAMES:
        # A readout whose interface was degenerate gets -1, which is distinguishable from a
        # genuine 0 (measured, but no sparsity reached the 95% level).
        out[f"s95_linear_{name}"] = (
            s95_from_curve(sparsities, [r[f"p_rec_linear_{name}"] for r in rows])
            if name in interfaces else -1)
    return out


# Public name for the linearity report, recorded per model by the scaled campaign.
linearity_report = _linearity_report
