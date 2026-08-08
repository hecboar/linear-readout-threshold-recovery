"""Cross-validated affine probes and native-distribution evaluation.

Two objections to the E5 protocol are answered here, and both are answered by measurement
rather than argument.

*The linear baseline may have been too weak.* E5 fits its least-squares readout and evaluates
it on freshly drawn supports, but the probe has no intercept and only ever sees the *linear*
representation ``Phi b``. A referee is entitled to ask what happens against the strongest
affine probe of the representation the network actually produces, ``ReLU(Phi b)``, fitted with
a held-out split. :func:`boolean_probe_profile` runs exactly that comparison, on the same
evaluation states as the network, per sparsity, so neither side gets a distributional edge.

*The evaluation distribution is not the training distribution.* The network is trained on
``x = mask * U[-1, 1]`` and E5 diagnoses it on Boolean states ``1_S``. That is deliberate --
Boolean states are the object of the theory -- but it leaves the trained-network claims open to
the charge of being off-distribution. :func:`native_profile` repeats the diagnosis on the
training distribution itself, where the analog error has an exact closed form and support
detection is measured at the best held-out threshold.

Neither probe here carries a floor claim. The affine probes are rank ``d+1`` maps of a
representation that is nonlinear in the post-ReLU case, so Theorem 4.1 does not apply to them;
they are controls on the *empirical* comparison, and the floor statements stay with the three
readouts of :func:`lrtr.toymodel.diagnose_model`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .interface import energy_floor_uniform, unit_diagonal
from .stats import wilson_interval
from .threshold import s95_from_curve

__all__ = [
    "fit_affine_probe",
    "apply_affine_probe",
    "boolean_states",
    "boolean_probe_profile",
    "native_profile",
]

REPRESENTATIONS = ("pre", "post")


def _represent(W_in: np.ndarray, X: np.ndarray, representation: str) -> np.ndarray:
    """``(n, d)`` representations of the ``(F, n)`` states ``X``.

    ``pre`` is the linear representation ``Phi x``; ``post`` is the network's own hidden
    activation ``ReLU(Phi x)``.
    """
    pre = (W_in @ X).T
    if representation == "pre":
        return pre
    if representation == "post":
        return np.maximum(pre, 0.0)
    raise ValueError(f"unknown representation {representation!r}; use 'pre' or 'post'")


def fit_affine_probe(R: np.ndarray, Y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Ridge-regularised affine fit ``Y ~ [R, 1] @ W``. Returns ``W`` of shape ``(d+1, F)``.

    The intercept column is not penalised, which is the usual convention and matters here
    because the targets are Boolean and therefore far from mean-zero.
    """
    n, d = R.shape
    X = np.hstack([R, np.ones((n, 1))])
    A = X.T @ X
    pen = ridge * np.eye(d + 1)
    pen[d, d] = 0.0
    return np.linalg.solve(A + pen, X.T @ Y)


def apply_affine_probe(W: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Scores ``(n, F)`` of the probe ``W`` on representations ``R``."""
    return np.hstack([R, np.ones((R.shape[0], 1))]) @ W


def boolean_states(F: int, s: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(F, n)`` matrix whose columns are indicators of uniform random size-``s`` supports.

    Drawn by partial-sorting one uniform key per (column, feature): the ``s`` smallest keys of
    a column are exchangeable, so the selected set is uniform over size-``s`` subsets. This is
    the same law as repeated ``rng.choice(..., replace=False)`` and is orders of magnitude
    faster at the sizes the scaled campaign uses, where support drawing would otherwise cost
    more than the linear algebra it feeds.
    """
    if not 1 <= s <= F:
        raise ValueError(f"need 1 <= s <= F, got s={s}, F={F}")
    keys = rng.random((n, F))
    idx = np.argpartition(keys, s - 1, axis=1)[:, :s]
    B = np.zeros((F, n))
    B[idx.ravel(), np.repeat(np.arange(n), s)] = 1.0
    return B


_boolean_states = boolean_states  # previous private name


def boolean_probe_profile(W_in: np.ndarray, W_out: np.ndarray, sparsities: Sequence[int],
                          n_train: int, n_test: int, seed: int, theta: float = 0.5,
                          ridge: float = 1e-8, conf: float = 0.95,
                          extra_readouts: Optional[Dict[str, np.ndarray]] = None
                          ) -> Dict[str, Any]:
    """Network output versus cross-validated affine probes, on identical held-out states.

    For each sparsity a probe is fitted on ``n_train`` states and scored on ``n_test`` disjoint
    ones drawn from the same law, separately for the ``pre`` and ``post`` representations. The
    network's own thresholded output is scored on those same ``n_test`` states, so the
    comparison at each sparsity is paired.

    Fitting one probe *per sparsity* deliberately favours the probe: it is told the sparsity it
    will be tested at, which the network is not. A network that still wins has won against a
    generous baseline.

    ``extra_readouts`` maps a name to an ``(F, d)`` readout of the *linear* representation --
    the pseudoinverse, the model's own decoder, a least-squares fit -- which is calibrated to
    unit diagonal here and scored on the same held-out states, so the fixed readouts of the
    theory and the fitted probes appear side by side on identical evidence.
    """
    d, F = W_in.shape
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    fixed: Dict[str, np.ndarray] = {}
    for name, G in (extra_readouts or {}).items():
        try:
            fixed[name] = unit_diagonal(np.asarray(G, dtype=np.float64) @ W_in)
        except ValueError:
            continue                                          # vanishing gain: interface undefined

    for s in [int(x) for x in sparsities]:
        B_tr = _boolean_states(F, s, n_train, rng)
        B_te = _boolean_states(F, s, n_test, rng)
        target_te = B_te.T                                   # (n_test, F)

        row: Dict[str, Any] = {"s": s, "n_train": n_train, "n_test": n_test}

        # ---- the network, on the held-out states ----
        Y = (W_out @ np.maximum(W_in @ B_te, 0.0)).T         # (n_test, F)
        ok = np.all((Y >= theta) == (target_te > 0.5), axis=1)
        lo, hi = wilson_interval(int(ok.sum()), n_test, conf=conf)
        row.update({"p_rec_model": float(ok.mean()), "ci_low_model": lo, "ci_high_model": hi,
                    "rms_model": float(np.sqrt(((Y - target_te) ** 2).mean()))})

        # ---- affine probes, fitted out of sample ----
        for rep in REPRESENTATIONS:
            W = fit_affine_probe(_represent(W_in, B_tr, rep), B_tr.T, ridge=ridge)
            Z = apply_affine_probe(W, _represent(W_in, B_te, rep))
            ok_p = np.all((Z >= theta) == (target_te > 0.5), axis=1)
            lo_p, hi_p = wilson_interval(int(ok_p.sum()), n_test, conf=conf)
            row.update({
                f"p_rec_probe_{rep}": float(ok_p.mean()),
                f"ci_low_probe_{rep}": lo_p,
                f"ci_high_probe_{rep}": hi_p,
                f"rms_probe_{rep}": float(np.sqrt(((Z - target_te) ** 2).mean())),
            })

        # ---- the fixed linear interfaces of the theory, on the same held-out states ----
        for name, M in fixed.items():
            Zf = (M @ B_te).T
            ok_f = np.all((Zf >= theta) == (target_te > 0.5), axis=1)
            lo_f, hi_f = wilson_interval(int(ok_f.sum()), n_test, conf=conf)
            row.update({
                f"p_rec_linear_{name}": float(ok_f.mean()),
                f"ci_low_linear_{name}": lo_f,
                f"ci_high_linear_{name}": hi_f,
                f"rms_linear_{name}": float(np.sqrt(((Zf - target_te) ** 2).mean())),
            })

        row["rms_floor_uniform"] = float(np.sqrt(energy_floor_uniform(F, d, s)))
        rows.append(row)

    probs = {"model": [r["p_rec_model"] for r in rows]}
    for rep in REPRESENTATIONS:
        probs[f"probe_{rep}"] = [r[f"p_rec_probe_{rep}"] for r in rows]
    for name in fixed:
        probs[f"linear_{name}"] = [r[f"p_rec_linear_{name}"] for r in rows]
    s95 = {k: s95_from_curve([r["s"] for r in rows], v) for k, v in probs.items()}

    return {"d": d, "F": F, "theta": theta, "ridge": ridge, "seed": seed,
            "rows": rows, "s95": s95}


def native_profile(W_in: np.ndarray, W_out: np.ndarray, p: float, n_train: int, n_test: int,
                   seed: int, ridge: float = 1e-8,
                   readouts: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """The same diagnosis on the *training* distribution, ``x = mask * U[-1, 1]``.

    Two quantities are reported.

    *Analog reconstruction.* For a calibrated interface ``M`` with ``A = M - I`` and inputs
    whose coordinates are independent with ``E[x_i^2] = p/3``, the expected per-coordinate
    squared error is exactly ``(p/3) ||A||_F^2 / F``. No Monte Carlo is involved, and the floor
    ``||A||_F^2 >= F(F-d)/d`` turns into the native-distribution floor ``(p/3)(F-d)/d``.

    *Support detection.* Exact recovery of ``1{x_i != 0}`` is the wrong metric here: a
    coordinate drawn near zero is undetectable in principle, so the exact-recovery rate
    collapses to zero for reasons that have nothing to do with the code. We report instead the
    per-coordinate detection accuracy at the threshold that maximises it on the *training*
    split, evaluated on the held-out split, for the network and for both affine probes.
    """
    d, F = W_in.shape
    rng = np.random.default_rng(seed)

    def draw(n):
        mask = (rng.random((F, n)) < p).astype(float)
        return mask * (rng.random((F, n)) * 2.0 - 1.0)

    X_tr, X_te = draw(n_train), draw(n_test)
    on_tr, on_te = (X_tr != 0.0).T, (X_te != 0.0).T          # (n, F)

    # ---- analog side: exact, from the interfaces of the linear representation ----
    if readouts is None:
        readouts = {"pinv": np.linalg.pinv(W_in), "wout": W_out}
    second_moment = p / 3.0
    analog: Dict[str, Any] = {"second_moment": second_moment,
                              "rms_floor_native": float(np.sqrt(second_moment * (F - d) / d))}
    for name, G in readouts.items():
        try:
            A = unit_diagonal(np.asarray(G, dtype=np.float64) @ W_in) - np.eye(F)
        except ValueError as exc:                            # vanishing gain: undefined
            analog[name] = {"error": str(exc)}
            continue
        frob_sq = float(np.sum(A * A))
        analog[name] = {
            "frob_sq_A": frob_sq,
            "energy_per_coord": second_moment * frob_sq / F,
            "rms": float(np.sqrt(second_moment * frob_sq / F)),
        }

    # ---- detection side: best threshold chosen on train, reported on test ----
    def detection(score_tr: np.ndarray, score_te: np.ndarray) -> Dict[str, float]:
        grid = np.quantile(np.abs(score_tr), np.linspace(0.5, 0.999, 60))
        acc_tr = [float(((np.abs(score_tr) >= t) == on_tr).mean()) for t in grid]
        t_best = float(grid[int(np.argmax(acc_tr))])
        pred = np.abs(score_te) >= t_best
        tp = float((pred & on_te).sum())
        fp = float((pred & ~on_te).sum())
        fn = float((~pred & on_te).sum())
        return {
            "threshold": t_best,
            "accuracy": float((pred == on_te).mean()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        }

    detect = {"model": detection((W_out @ np.maximum(W_in @ X_tr, 0.0)).T,
                                 (W_out @ np.maximum(W_in @ X_te, 0.0)).T)}
    for rep in REPRESENTATIONS:
        R_tr, R_te = _represent(W_in, X_tr, rep), _represent(W_in, X_te, rep)
        W = fit_affine_probe(R_tr, X_tr.T, ridge=ridge)      # regress the *values*, not the mask
        detect[f"probe_{rep}"] = detection(apply_affine_probe(W, R_tr),
                                           apply_affine_probe(W, R_te))

    return {"d": d, "F": F, "p": p, "n_train": n_train, "n_test": n_test, "seed": seed,
            "analog": analog, "detection": detect,
            "base_rate": float(on_te.mean())}
