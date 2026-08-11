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
    "native_distribution_profile",
    "distribution_weighted_error_ci",
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


# --------------------------------------------------------------------------------------
# The diagnosis repeated on a non-Boolean state distribution
# --------------------------------------------------------------------------------------

def native_distribution_profile(W_in: np.ndarray, W_out: np.ndarray, p: float, n_train: int,
                                n_test: int, seed: int, ridge: float = 1e-8,
                                readouts: Optional[Dict[str, np.ndarray]] = None,
                                ) -> Dict[str, Any]:
    """Diagnose a trained model on its *own* training distribution, `x = mask * U[-1, 1]`.

    The Boolean audit measures what the representation can support; this measures what the
    network sees. Two quantities, and the split between them matters:

    *Analog reconstruction.* Exact, via `E||Ab||^2 = tr(A^T A C)`. For these inputs the
    coordinates are independent with `E[x_i^2] = p/3`, so `C = (p/3) I` and the per-coordinate
    error is `(p/3) ||A||_F^2 / F`, with the rank-trace floor turning into `(p/3)(F-d)/d`. No
    Monte Carlo, and the general identity means a different input law is a different `C` rather
    than a different derivation.

    *Support detection.* Exact recovery of `1{x_i != 0}` is the wrong metric here: a coordinate
    drawn near zero is undetectable in principle, so the exact-recovery rate collapses for
    reasons that have nothing to do with the code. Reported instead is per-coordinate detection
    quality at the threshold that maximises accuracy on the *training* split, evaluated on the
    held-out split.

    **KNOWN DEFECT -- do not compare decoders on these numbers.** The threshold is selected by
    accuracy, and the base rate here is `p`, around 1%. At that base rate the accuracy-optimal
    threshold is close to "predict everything off", so each decoder is placed at whatever
    conservative operating point its own score distribution happens to give: measured on a
    trained `L4` model at `d=50`, the network lands at precision 1.00 with recall 0.10 (F1 0.19)
    while an affine probe on the same representation lands at precision 0.70 with recall 0.75
    (F1 0.73). The criterion is the same for both, so there is no asymmetry of method, but the
    resulting F1 gap measures where each score distribution puts its accuracy optimum rather
    than how well either decodes.

    This is the same trap `select_thresholds` in :mod:`lrtr.probes` documents and avoids -- there
    the threshold is chosen on the objective actually reported. The fix here is the same: select
    on the reported metric. It is cheap, because this profile is recomputed from saved weights
    without retraining. Until then, the Boolean audit is the comparison to quote.
    See `docs/known_defects.md`.
    """
    from .interface import unit_diagonal

    d, F = W_in.shape
    rng = np.random.default_rng(seed)

    def draw(n):
        return (rng.random((F, n)) < p) * (rng.random((F, n)) * 2.0 - 1.0)

    X_tr, X_te = draw(n_train), draw(n_test)
    on_tr, on_te = (X_tr != 0.0).T, (X_te != 0.0).T

    if readouts is None:
        readouts = {"pinv": np.linalg.pinv(W_in), "wout": W_out}
    second_moment = p / 3.0
    analog: Dict[str, Any] = {
        "second_moment": second_moment,
        "rms_floor_native": float(np.sqrt(second_moment * (F - d) / d)),
    }
    for name, G in readouts.items():
        try:
            A = unit_diagonal(np.asarray(G, dtype=np.float64) @ W_in) - np.eye(F)
        except ValueError as exc:
            analog[name] = {"error": str(exc)}
            continue
        C = second_moment * np.eye(F)
        energy = distribution_weighted_error(A, C)
        analog[name] = {"frob_sq_A": float(np.sum(A * A)), "energy_per_coord": energy,
                        "rms": float(np.sqrt(energy)),
                        "energy_empirical": distribution_weighted_error_ci(A, X_te)}

    def detection(score_tr: np.ndarray, score_te: np.ndarray) -> Dict[str, float]:
        grid = np.quantile(np.abs(score_tr), np.linspace(0.5, 0.999, 60))
        acc = [float(((np.abs(score_tr) >= t) == on_tr).mean()) for t in grid]
        t_best = float(grid[int(np.argmax(acc))])
        pred = np.abs(score_te) >= t_best
        tp = float((pred & on_te).sum())
        fp = float((pred & ~on_te).sum())
        fn = float((~pred & on_te).sum())
        return {"threshold": t_best, "accuracy": float((pred == on_te).mean()),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}

    detect = {"model": detection((W_out @ np.maximum(W_in @ X_tr, 0.0)).T,
                                 (W_out @ np.maximum(W_in @ X_te, 0.0)).T)}
    for rep, fn_rep in (("pre", lambda Xm: (W_in @ Xm).T),
                        ("post", lambda Xm: np.maximum(W_in @ Xm, 0.0).T)):
        R_tr, R_te = fn_rep(X_tr), fn_rep(X_te)
        Xd = np.hstack([R_tr, np.ones((R_tr.shape[0], 1))])
        A_ = Xd.T @ Xd + ridge * np.eye(Xd.shape[1])
        A_[-1, -1] -= ridge
        Wp = np.linalg.solve(A_, Xd.T @ X_tr.T)
        detect[f"probe_{rep}"] = detection(
            Xd @ Wp, np.hstack([R_te, np.ones((R_te.shape[0], 1))]) @ Wp)

    return {"d": int(d), "F": int(F), "p": p, "n_train": n_train, "n_test": n_test,
            "seed": seed, "analog": analog, "detection": detect,
            "base_rate": float(on_te.mean())}
