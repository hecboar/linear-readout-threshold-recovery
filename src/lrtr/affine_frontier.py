"""G2 -- the robust affine support frontier: where affine decoding stops working, exactly.

The floor of the analog level says a linear readout cannot reconstruct values accurately. It
says nothing about whether a *threshold* applied to a linear score recovers the support, and
conflating the two is the error the earlier drafts made. This module computes the affine limit
directly.

Fix a code `Phi`, a feature `i` and a sparsity `s`. A featurewise affine rule
`b_i_hat = 1{w^T x + c >= theta}` recovers feature `i` on every Boolean support of size exactly
`s` iff the worst active score exceeds the worst inactive score. Writing `a_j = w^T phi_j`,
those two extremes are available in closed form:

    worst active   = c + a_i + L_{s-1}(a_{-i})     (feature i on, the s-1 others adversarial)
    worst inactive = c + U_s(a_{-i})               (feature i off, s others adversarial)

with `L_k` the sum of the `k` smallest and `U_k` the sum of the `k` largest entries. The gap is
independent of `c`, and the optimal `c` simply centres the threshold in it.

Formulation. Separability over *all* supports is therefore

    exists w :  a_i(w) + L_{s-1}(a_{-i}(w)) - U_s(a_{-i}(w)) >= 1,

after a harmless rescaling: the gap is positively homogeneous of degree one in `w`, so a
positive gap can always be scaled to 1. This is why the programme is stated with a margin
normalisation rather than a norm constraint -- maximising the gap over `||w|| <= 1` is
degenerate, since `w = 0` is feasible and gives gap 0, making the optimum non-negative by
construction and destroying the certificate on the failure side (decision D2).

We therefore solve

    minimise  1/2 ||w||^2   subject to that gap being at least 1,

whose feasible set is convex (the constraint function is concave: `a_i` is linear, `L_{s-1}` is
a min of linear functions, `-U_s` is a min of linear functions). Expanding the two extremes,
the constraint is equivalent to the family of linear inequalities

    w^T ( phi_i + sum_{j in A} phi_j - sum_{j in B} phi_j ) >= 1

over all `A` of size `s-1` and `B` of size `s` in `[F] \\ {i}`. There are exponentially many;
we never enumerate them. Constraint generation adds only the most violated one at each step,
found by sorting the scores -- the `s-1` smallest for `A`, the `s` largest for `B`.

Two outcomes, reported as different objects rather than as one signed scalar:

* **feasible** -> a *robust affine margin* `gamma = 1 / (2 ||w*||)`, attained by an explicit
  `(w, c)`;
* **infeasible** -> an *affine infeasibility certificate*: weights `lambda >= 0` summing to one
  with `sum_k lambda_k v_k = 0`. By Gordan's theorem this is exactly the obstruction, and it is
  checkable without trusting the solver: recompute the combination and see that it vanishes.
  Geometrically it places the origin in the convex hull of the active-minus-inactive
  differences, i.e. the two convex hulls touch.

This is a statement about the *worst case over supports*, and must not be read as a statement
about typical supports; empirical recovery is a separate, average-case quantity.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linprog, minimize

__all__ = [
    "worst_case_scores",
    "cut_vector",
    "robust_affine_margin",
    "robust_affine_frontier",
    "verify_certificate",
]

TOL = 1e-9


def worst_case_scores(a: np.ndarray, i: int, s: int) -> Dict[str, Any]:
    """Worst active and worst inactive scores of feature `i` at sparsity `s`, with witnesses.

    Exact: equivalent to enumerating all `C(F-1, s-1)` active and `C(F-1, s)` inactive supports,
    which `tests/test_affine_frontier.py` verifies by brute force.
    """
    F = a.size
    if not 1 <= s <= F - 1:
        raise ValueError(f"need 1 <= s <= F-1, got s={s}, F={F}")
    others = np.delete(np.arange(F), i)
    order = others[np.argsort(a[others], kind="stable")]
    low = order[: s - 1]                      # s-1 smallest: worst companions when i is on
    high = order[-s:]                         # s largest: worst distractors when i is off
    return {
        "worst_active": float(a[i] + a[low].sum()),
        "worst_inactive": float(a[high].sum()),
        "gap": float(a[i] + a[low].sum() - a[high].sum()),
        "active_support": np.sort(np.append(low, i)),
        "inactive_support": np.sort(high),
    }


def cut_vector(Phi: np.ndarray, i: int, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """`v = phi_i + sum_{j in low} phi_j - sum_{j in high} phi_j`, one linear constraint."""
    v = Phi[:, i].copy()
    if low.size:
        v += Phi[:, low].sum(axis=1)
    if high.size:
        v -= Phi[:, high].sum(axis=1)
    return v


def _most_violated(Phi: np.ndarray, w: np.ndarray, i: int, s: int) -> Dict[str, Any]:
    a = w @ Phi
    ws = worst_case_scores(a, i, s)
    low = np.setdiff1d(ws["active_support"], [i], assume_unique=False)
    return {"v": cut_vector(Phi, i, low, ws["inactive_support"]), **ws}


def _feasible(V: np.ndarray) -> Dict[str, Any]:
    """Phase 1: is `{w : V^T w >= 1}` non-empty? Decided by an LP, with duals for the witness.

    Maximise `t` subject to `V^T w >= t` and `|w|_inf <= 1`. By homogeneity the box costs no
    generality: a feasible `w0` rescaled to unit sup-norm gives `t = 1/||w0||_inf > 0`, and any
    `t > 0` rescales to a feasible point.
    """
    d, K = V.shape
    # variables [w (d, free within the box), t]
    c = np.zeros(d + 1)
    c[-1] = -1.0                                     # maximise t
    A_ub = np.hstack([-V.T, np.ones((K, 1))])        # -V^T w + t <= 0
    b_ub = np.zeros(K)
    bounds = [(-1.0, 1.0)] * d + [(None, 1.0)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return {"feasible": False, "t": None, "status": res.message}
    t = float(res.x[-1])
    return {"feasible": t > TOL, "t": t, "status": "ok"}


def _min_norm(V: np.ndarray) -> Dict[str, Any]:
    """Solve `min 1/2 ||w||^2 s.t. V^T w >= 1` through its dual.

    Dual: `min_{lambda >= 0} 1/2 ||V lambda||^2 - 1^T lambda`, with `w = V lambda`. Bound
    constraints only, so it is well conditioned and needs no general-purpose QP solver.
    """
    d, K = V.shape
    G = V.T @ V

    def f(lam):
        return 0.5 * float(lam @ G @ lam) - float(lam.sum())

    def g(lam):
        return G @ lam - 1.0

    res = minimize(f, np.full(K, 1.0 / max(K, 1)), jac=g, method="L-BFGS-B",
                   bounds=[(0.0, None)] * K, options={"maxiter": 20000, "ftol": 1e-16,
                                                      "gtol": 1e-12})
    w = V @ res.x
    return {"w": w, "lambda": res.x, "obj": float(res.fun), "status": res.message}


def _farkas(V: np.ndarray) -> Dict[str, Any]:
    """Find `lambda >= 0`, `sum lambda = 1`, `V lambda = 0` — the infeasibility certificate.

    Gordan: exactly one of `{w : V^T w >= 1}` non-empty and this system solvable. Recovered by
    minimising `||V lambda||^2` over the simplex; a residual at numerical zero is the witness.
    """
    d, K = V.shape
    G = V.T @ V

    def f(lam):
        return float(lam @ G @ lam)

    def g(lam):
        return 2.0 * (G @ lam)

    cons = [{"type": "eq", "fun": lambda lam: lam.sum() - 1.0,
             "jac": lambda lam: np.ones_like(lam)}]
    res = minimize(f, np.full(K, 1.0 / K), jac=g, method="SLSQP", constraints=cons,
                   bounds=[(0.0, None)] * K, options={"maxiter": 5000, "ftol": 1e-16})
    lam = np.clip(res.x, 0.0, None)
    ssum = lam.sum()
    if ssum > 0:
        lam = lam / ssum
    return {
        "lambda": lam,
        "residual": float(np.linalg.norm(V @ lam)),
        "scale": float(np.max(np.linalg.norm(V, axis=0))),
        "sum_lambda": float(lam.sum()),
        "support": np.nonzero(lam > 1e-8)[0],
    }


def verify_certificate(Phi: np.ndarray, i: int, s: int, certificate: Dict[str, Any],
                       tol: float = 1e-6) -> Dict[str, Any]:
    """Re-derive an infeasibility certificate from the code alone and check it.

    Deliberately takes the *supports*, not the cut vectors: it rebuilds every `v_k` from `Phi`
    and recomputes the combination, so a certificate can be checked in a fresh process by code
    that never saw the solver. "The solver reported infeasible" is not evidence; this is.
    """
    lam = np.asarray(certificate["lambda"], dtype=np.float64)
    cuts = certificate["cuts"]
    if lam.size != len(cuts):
        return {"valid": False, "reason": "lambda and cut list have different lengths"}
    V = np.stack([cut_vector(Phi, i, np.asarray(c["low"], dtype=int),
                             np.asarray(c["high"], dtype=int)) for c in cuts], axis=1)
    combo = V @ lam
    scale = float(np.max(np.linalg.norm(V, axis=0))) or 1.0
    rel = float(np.linalg.norm(combo)) / scale
    checks = {
        "nonnegative": bool(np.all(lam >= -1e-12)),
        "sums_to_one": bool(abs(lam.sum() - 1.0) < 1e-8),
        "combination_vanishes": bool(rel < tol),
        "relative_residual": rel,
        "n_active": int((lam > 1e-8).sum()),
    }
    checks["valid"] = all(v for k, v in checks.items()
                          if k in ("nonnegative", "sums_to_one", "combination_vanishes"))
    return checks


def robust_affine_margin(Phi: np.ndarray, feature_index: int, sparsity: int,
                         max_iter: int = 500, theta: float = 0.5) -> Dict[str, Any]:
    """The affine limit for one feature at one sparsity. See the module docstring.

    Returns a record with `status` in `{"separable", "infeasible"}` and, respectively, the
    robust margin with its optimal `(w, bias)` and worst-case supports, or a Farkas certificate
    with the cut list needed to re-verify it independently.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i, s = int(feature_index), int(sparsity)
    if not 0 <= i < F:
        raise ValueError(f"feature index {i} out of range for F={F}")
    t0 = time.perf_counter()

    cuts: List[Dict[str, Any]] = []
    V = np.zeros((d, 0))
    w = Phi[:, i] / max(float(Phi[:, i] @ Phi[:, i]), 1e-12)   # unit gain on i, a sane start
    status, out = "unresolved", {}

    for it in range(max_iter):
        viol = _most_violated(Phi, w, i, s)
        if V.shape[1] and float(w @ viol["v"]) >= 1.0 - 1e-7:
            status = "separable"
            break
        low = np.setdiff1d(viol["active_support"], [i])
        cuts.append({"low": low.tolist(), "high": viol["inactive_support"].tolist()})
        V = np.hstack([V, viol["v"][:, None]])

        feas = _feasible(V)
        if not feas["feasible"]:
            status = "infeasible"
            cert = _farkas(V)
            cert["cuts"] = cuts
            out = {"certificate": cert,
                   "verification": verify_certificate(Phi, i, s, cert)}
            break

        sol = _min_norm(V)
        w = sol["w"]
    else:
        status = "max_iter"

    runtime = time.perf_counter() - t0
    rec: Dict[str, Any] = {
        "d": int(d), "F": int(F), "feature": i, "sparsity": s,
        "status": status, "n_cuts": len(cuts), "iterations": it + 1,
        "runtime_s": runtime,
    }

    if status == "separable":
        norm = float(np.linalg.norm(w))
        final = _most_violated(Phi, w, i, s)
        # At the optimum the binding constraint has gap 1, so the geometric margin of the
        # unit-norm direction is half of 1/||w||.
        rec.update({
            "margin": 1.0 / (2.0 * norm) if norm > 0 else np.inf,
            "w_norm": norm,
            "w": w.tolist(),
            "bias": float(theta - (final["worst_active"] + final["worst_inactive"]) / 2.0),
            "worst_active_support": final["active_support"].tolist(),
            "worst_inactive_support": final["inactive_support"].tolist(),
            "primal_residual": float(max(0.0, 1.0 - final["gap"])),
        })
    else:
        rec.update(out)
    return rec


def robust_affine_frontier(Phi: np.ndarray, sparsity_grid: Sequence[int],
                           feature_subset: Optional[Sequence[int]] = None,
                           max_iter: int = 500) -> Dict[str, Any]:
    """`gamma_aff*(s) = min_i gamma_i*(s)` over a sparsity grid, and `s_aff_robust`.

    `s_aff_robust` is the largest `s` such that every tested feature is separable at every
    `k <= s` on the grid. When `feature_subset` is given the frontier is an *upper* bound on the
    true one -- an untested feature can only make it smaller -- and the record says so.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    feats = list(range(F)) if feature_subset is None else [int(x) for x in feature_subset]
    grid = sorted(int(s) for s in sparsity_grid)

    per_s: List[Dict[str, Any]] = []
    for s in grid:
        recs = [robust_affine_margin(Phi, i, s, max_iter=max_iter) for i in feats]
        sep = [r for r in recs if r["status"] == "separable"]
        infeas = [r for r in recs if r["status"] == "infeasible"]
        unresolved = [r for r in recs if r["status"] not in ("separable", "infeasible")]
        per_s.append({
            "s": s,
            "all_separable": len(sep) == len(recs),
            "n_separable": len(sep), "n_infeasible": len(infeas),
            "n_unresolved": len(unresolved),
            "gamma_min": float(min(r["margin"] for r in sep)) if sep and not infeas else None,
            "gamma_median": float(np.median([r["margin"] for r in sep])) if sep else None,
            "first_infeasible_feature": infeas[0]["feature"] if infeas else None,
            "certificate": infeas[0].get("certificate") if infeas else None,
            "verification": infeas[0].get("verification") if infeas else None,
            "runtime_s": float(sum(r["runtime_s"] for r in recs)),
        })

    s_robust = 0
    for row in per_s:
        if row["all_separable"]:
            s_robust = row["s"]
        else:
            break

    return {
        "d": int(d), "F": int(F),
        "features_tested": len(feats),
        "is_upper_bound": feature_subset is not None,
        "grid": grid,
        "rows": per_s,
        "s_aff_robust": s_robust,
    }
