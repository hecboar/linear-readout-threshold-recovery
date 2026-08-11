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

  What it witnesses, stated precisely because an earlier draft got this wrong: each `v_k` is an
  (active state - inactive state) difference, so the certificate exhibits a **convex combination
  of active states equal to a convex combination of inactive states**. It reduces to a single
  pair of supports only when `lambda` is supported on one index, which is not the general case;
  `n_supports_involved` records how many are actually needed.

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
    "collision_radius",
    "collision_frontier",
    "separable_from_rho",
    "collision_radius_atmost",
    "separable_atmost",
    "collision_radius_boxless",
    "leverage_bound_on_rho",
    "min_l2_representation",
    "leverage_upper_bound_on_kappa",
    "affine_failure_threshold",
    "s_max_from_rho",
    "local_coherence",
    "coherence_bound_on_rho",
    "affine_circuit_size",
    "collision_distance",
    "cut_vector",
    "robust_affine_margin",
    "robust_affine_margin_compact",
    "robust_affine_frontier",
    "verify_certificate",
    "radon_witness",
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
        # A convex combination, not a single colliding pair: this is how many states it needs.
        "n_supports_involved": int((lam > 1e-8).sum()),
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

    For the *verdict* alone, :func:`collision_radius` is cheaper and exact: one linear programme
    per feature settles every sparsity at once. This routine is what supplies the margin, which
    the collision radius does not, and serves as the independent check on it.
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


def collision_radius(Phi: np.ndarray, feature_index: int) -> Dict[str, Any]:
    """`rho_i(Phi)`: one linear programme that determines the whole per-feature frontier.

    Affine separability of feature `i` at sparsity `s` fails exactly when the convex hulls of
    the active and inactive states touch,

        C_i^+(s) = phi_i + Phi_{-i} D(F-1, s-1),   C_i^-(s) = Phi_{-i} D(F-1, s),

    with `D(n, k) = {z in [0,1]^n : 1^T z = k}` the hypersimplex. Writing a point of the
    intersection as `phi_i + Phi_{-i} u = Phi_{-i} v` and setting `z = v - u` gives

        Phi_{-i} z = phi_i,    1^T z = 1,    ||z||_inf <= 1,

    and conversely any such `z` splits back into an admissible `(u, v)` iff the budget fits:
    `v_j` must lie in `[z_j^+, 1 - z_j^-]`, non-empty by `||z||_inf <= 1`, and `1^T v = s` is
    reachable iff `sum_j z_j^+ <= s <= (F-1) - sum_j z_j^-`. Since `1^T z = 1` forces
    `sum z^+ = (||z||_1 + 1)/2` and `sum z^- = (||z||_1 - 1)/2`, both reduce to a single bound:

        the hulls meet at sparsity s   <=>   ||z||_1 <= 2 min(s, F - s) - 1  for some such z.

    Therefore, defining

        rho_i(Phi) = min { ||z||_1 : Phi_{-i} z = phi_i, 1^T z = 1, ||z||_inf <= 1 },

    feature `i` is affinely separable at sparsity `s` **iff** `rho_i > 2 min(s, F-s) - 1`. The
    entire frontier of a feature -- every sparsity at once -- is decided by this one number, and
    `rho_i` is a linear programme. An infeasible programme means `rho_i = inf`: `phi_i` is not an
    affine combination of the other columns within the box, and the feature is separable at
    every sparsity.

    Returns `rho`, the minimiser `z` when finite, and the derived per-feature frontier.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i = int(feature_index)
    P = np.delete(Phi, i, axis=1)
    n = F - 1
    t0 = time.perf_counter()

    # Variables [z (n) | t (n)] with t >= |z|, minimising 1^T t.
    c = np.concatenate([np.zeros(n), np.ones(n)])
    A_ub = np.vstack([np.hstack([np.eye(n), -np.eye(n)]),
                      np.hstack([-np.eye(n), -np.eye(n)])])
    A_eq = np.vstack([np.hstack([P, np.zeros((d, n))]),
                      np.concatenate([np.ones(n), np.zeros(n)])[None, :]])
    b_eq = np.concatenate([Phi[:, i], [1.0]])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * n), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(-1.0, 1.0)] * n + [(0.0, None)] * n, method="highs")

    rec: Dict[str, Any] = {"feature": i, "d": int(d), "F": int(F),
                           "runtime_s": time.perf_counter() - t0}
    if not res.success:
        # Infeasible: no admissible z exists, so the hulls never meet.
        rec.update({"rho": float("inf"), "z": None, "s_max": (F - 1) // 2,
                    "status": "infeasible_lp_separable_everywhere"})
        return rec

    rho = float(res.fun)
    rec.update({"rho": rho, "z": res.x[:n].tolist(), "s_max": s_max_from_rho(rho, F),
                "status": "ok"})
    return rec


def s_max_from_rho(rho: float, F: int, tol: float = 1e-9) -> int:
    """Largest `s` in the monotone regime with `rho > 2s - 1`: the per-feature frontier.

    The frontier is the largest integer **strictly below** `(rho + 1) / 2`, which is
    `ceil(t) - 1` whether or not `t` is an integer, so no case split is needed. The `ceil` does
    need a tolerance: a `rho` that should land exactly on the tie `t = 2` arrives as
    `2 + 1e-16` and would return 2 instead of 1. Exact ties are not a corner case here -- a
    duplicated column gives `rho = 1` exactly, and its tie must resolve to `s_max = 0`.

    **Valid only for `s <= floor(F/2)`, and capped there.** The threshold `2 min(s, F-s) - 1`
    increases with `s` only in that regime; beyond `F/2` it decreases again, so separability can
    reappear at very large `s`. That is a symmetry artefact -- identifying which `s` features are
    on is the same problem as identifying which `F - s` are off -- and not a regime any
    sparse-coding claim inhabits. For sparsities past `F/2`, call
    :func:`separable_from_rho` directly instead of reading this number.
    """
    if not np.isfinite(rho):
        return F // 2
    s_max = int(np.ceil((rho + 1.0) / 2.0 - tol)) - 1
    return max(0, min(s_max, F // 2))


def collision_radius_atmost(Phi: np.ndarray, feature_index: int,
                            alpha: float = 1.0) -> Dict[str, Any]:
    """The frontier for supports of size **at most** `s`, with amplitudes in `[alpha, 1]`.

    This is the better-behaved statement and should be preferred to :func:`collision_radius`.
    Because the support sizes are inequalities, `1^T z` is no longer pinned, and rebuilding
    `(v, w)` from `z = w - v` leaves `||z||_inf <= 1`, `sum z^+ <= s` and `sum z^- <= s-1`. The
    last two collapse into one scalar, so

        rho_hat_i = min { max(sum z^+, sum z^- + 1) : Phi_{-i} z = phi_i, ||z||_inf <= 1 },

    and feature `i` is separable over every support of size at most `s` iff `rho_hat_i > s`. The
    threshold is `s` rather than `2 min(s, F-s) - 1`, monotonicity holds by construction, and the
    `s > F/2` artefact of the exactly-`s` formulation disappears.

    **Amplitudes.** With at most `s` active features and amplitudes in `[alpha, 1]`, the state
    law drops out of the convex hulls entirely: for `U_k = {u : u_j in {0} u [alpha,1],
    |supp(u)| <= k}` the hull is `{u in [0,1]^F : 1^T u <= k}` regardless of `alpha`, because the
    unit-weight knapsack polytope is integral and its 0/1 vertices lie in `U_k`. Only the target
    feature's own amplitude survives, and the worst case is the smallest one, so

        rho_hat_i(alpha) = min { max(alpha sum z^+, alpha sum z^- + 1) :
                                 Phi_{-i} z = phi_i, ||z||_inf <= 1/alpha }.

    So the affine level of the hierarchy depends on the state distribution through **one scalar**,
    the minimum detectable amplitude -- against the full second moment `C` that the analog level
    needs. A vanishing amplitude makes nothing separable, which is why `alpha > 0` is a modelling
    necessity rather than a convenience.

    Both statements are checked against brute-force separation over the hull vertices in
    ``tests/test_frontier_extensions.py``.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i = int(feature_index)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    P = np.delete(Phi, i, axis=1)
    n = F - 1
    t0 = time.perf_counter()

    # Variables [z (n) | p (n) >= 0 | m (n) >= 0 | T], with z = p - m and T bounding both budgets.
    nv = 3 * n + 1
    iz, ip, im, iT = 0, n, 2 * n, 3 * n
    c = np.zeros(nv)
    c[iT] = 1.0
    A_eq = np.zeros((d + n, nv))
    b_eq = np.zeros(d + n)
    A_eq[:d, iz:iz + n] = P
    b_eq[:d] = Phi[:, i]
    rows = np.arange(n)
    A_eq[d + rows, iz + rows] = 1.0
    A_eq[d + rows, ip + rows] = -1.0
    A_eq[d + rows, im + rows] = 1.0
    A_ub = np.zeros((2, nv))
    A_ub[0, ip:ip + n] = alpha
    A_ub[0, iT] = -1.0
    A_ub[1, im:im + n] = alpha
    A_ub[1, iT] = -1.0
    b_ub = np.array([0.0, -1.0])
    B = 1.0 / alpha
    bounds = [(-B, B)] * n + [(0.0, None)] * (2 * n) + [(0.0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    rec: Dict[str, Any] = {"feature": i, "d": int(d), "F": int(F), "alpha": float(alpha),
                           "runtime_s": time.perf_counter() - t0}
    if not res.success:
        rec.update({"rho_hat": float("inf"), "s_max": F, "z": None,
                    "status": "infeasible_lp_separable_everywhere"})
        return rec
    rho = float(res.fun)
    rec.update({"rho_hat": rho, "z": res.x[:n].tolist(), "status": "ok",
                "s_max": max(0, int(np.ceil(rho - 1e-9)) - 1)})
    return rec


def separable_atmost(rho_hat: float, s: int) -> bool:
    """The at-most-`s` rule: separable iff `rho_hat > s`. Monotone in `s` by construction."""
    return bool(rho_hat > s + 1e-9)


def collision_radius_boxless(Phi: np.ndarray, feature_index: int) -> float:
    """`min ||z||_1` subject to `Phi_{-i} z = phi_i` and `1^T z = 1`, with no box.

    Provided to bound a claim rather than to make one. Measured on random codes, the box
    `||z||_inf <= 1` binds on only about 4% of features, so for almost every feature the
    collision radius *is* this pure minimum-`ell_1`-affine-representation quantity. That is
    closer to standard null-space-property machinery than the box formulation suggests, and it is
    recorded here so the point is made by us rather than by a referee. It also means the
    `ell_1` phase-transition literature is available for the frontier's asymptotics.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i = int(feature_index)
    P = np.delete(Phi, i, axis=1)
    n = F - 1
    A_eq = np.zeros((d + 1, 2 * n))
    A_eq[:d, :n] = P
    A_eq[:d, n:] = -P
    A_eq[d, :n] = 1.0
    A_eq[d, n:] = -1.0
    b_eq = np.concatenate([Phi[:, i], [1.0]])
    res = linprog(np.ones(2 * n), A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * (2 * n),
                  method="highs")
    return float(res.fun) if res.success else float("inf")


def min_l2_representation(Phi: np.ndarray) -> np.ndarray:
    """`min ||z||_2^2` subject to `Phi_{-i} z = phi_i`, in closed form: `h_i / (1 - h_i)`.

    Sherman--Morrison, since `Phi_{-i} Phi_{-i}^T = Sigma - phi_i phi_i^T`:

        min ||z||_2^2 = phi_i^T (Sigma - phi_i phi_i^T)^{-1} phi_i
                      = h_i + h_i^2/(1 - h_i) = h_i/(1 - h_i).

    So leverage measures redundancy directly: a feature with low leverage is reproducible by the
    others with small coefficients, no detour through coherence required. This is the identity
    that makes the hierarchy an implication rather than an observation.
    """
    from .analog_optimum import leverage

    h = leverage(Phi)
    return h / np.maximum(1.0 - h, 1e-300)


def leverage_upper_bound_on_kappa(Phi: np.ndarray) -> np.ndarray:
    """`kappa_i <= 1 + sqrt((F-1) h_i / (1 - h_i))`: low leverage forces the frontier down.

    From :func:`min_l2_representation` there is an admissible `z` with
    `||z||_2 = sqrt(h_i/(1-h_i))`, hence `||z||_1 <= sqrt(F-1) ||z||_2`. Since
    `max(sum z^+, 1 + sum z^-) <= 1 + ||z||_1`, that `z` certifies

        kappa_i <= 1 + sqrt((F-1) h_i/(1-h_i)).

    The box is satisfied for free wherever the bound is informative: `h_i <= 1/2` gives
    `||z||_2 <= 1`, hence `||z||_inf <= 1`.

    **This is the direction the hierarchy needs.** The matching lower bound
    (:func:`leverage_bound_on_rho`) is nearly vacuous, and an argument through coherence cannot
    work at all -- `rho >= 1/mu` bounds `rho` from below, so a large `mu` places no ceiling on it.
    Leverage does, and the resulting implication is one-way: see
    :func:`affine_failure_threshold`, and `[I, I]` for the failure of the converse.

    The `sqrt(F-1)` step is lossy, so the bound is far from tight -- on the trained `L4` models it
    predicts failure at `s = 11` where the measured frontier is 4. It is a sufficient condition
    for collapse, not an estimate of the frontier.
    """
    d, F = Phi.shape
    return 1.0 + np.sqrt((F - 1) * min_l2_representation(Phi))


def affine_failure_threshold(F: int, s: int) -> float:
    """Leverage below which affine separability at sparsity `s` is impossible.

    Solving `1 + sqrt((F-1) h/(1-h)) <= s` for `h`:

        h_i <= (s-1)^2 / ((F-1) + (s-1)^2)   ==>   feature i is not separable at sparsity s.

    A statement about the code alone, with no decoder and no training in it. Verified on 967
    predicted failures with no counterexample, and it correctly predicts the collapse of the
    `L2`-trained models from their leverage alone: `h_min = 0.0091` at `F = 100` falls below the
    `s = 2` threshold of `0.01`, and the measured frontier is indeed 1.
    """
    if s < 2:
        return 0.0
    k = (s - 1) ** 2
    return float(k / ((F - 1) + k))


def leverage_bound_on_rho(Phi: np.ndarray) -> np.ndarray:
    """`rho_i >= sqrt(h_i / (1 - h_i))`: the only inequality linking the two levels.

    Applying the code-specific optimal readout `g_i* = Sigma^{-1} phi_i / h_i` to any admissible
    `z` gives `1 = sum_j z_j (g_i*^T phi_j) <= ||z||_1 sqrt(1/h_i - 1)`, since the optimal
    cross-talk is exactly `1/h_i - 1` (G1). So a feature that is *analog*-friendly cannot have a
    small collision radius, and the implication runs analog -> affine only: `[I, I]` has perfectly
    uniform leverage and the worst possible frontier.

    The bound is tight -- `[I, I]` attains it exactly -- and largely vacuous. Since `rho_i >= 1`
    always and `sum_i h_i = d`, it says anything only for features with `h_i > 1/2`, of which
    there can be at most `2d`. In the strongly overcomplete regime it is empty for nearly every
    feature, so it does *not* order the hierarchy; the `[I,I]` versus `[I,H]` separation remains
    the operative statement.
    """
    from .analog_optimum import leverage

    h = leverage(Phi)
    return np.sqrt(h / np.maximum(1.0 - h, 1e-300))


def local_coherence(Phi: np.ndarray) -> np.ndarray:
    """`mu_i = max_{j != i} |<phi_i, phi_j>|` for unit-norm columns: coherence, per feature."""
    G = np.abs(np.asarray(Phi, dtype=np.float64).T @ Phi)
    np.fill_diagonal(G, -np.inf)
    return G.max(axis=1)


def coherence_bound_on_rho(Phi: np.ndarray) -> np.ndarray:
    """The lower bound `rho_i >= 1 / mu_i`, and with it a refinement of the coherence condition.

    Taking the inner product of `Phi_{-i} z = phi_i` with `phi_i` gives
    `1 = sum_{j != i} z_j <phi_i, phi_j> <= ||z||_1 mu_i` for every admissible `z`, so
    `rho_i >= 1 / mu_i`.

    Consequence: `mu_i < 1/(2s - 1)` suffices for separability at sparsity `s`. The classical
    coherence condition for threshold recovery asks `mu < 1/(2s)`, so this refines it twice
    over -- the constant improves from `2s` to `2s - 1`, and the requirement is *local*, on
    `mu_i` rather than on the global coherence. It also shows how much is lost by stopping at
    coherence: `rho_i` can exceed `1/mu_i` by a wide margin, and it is `rho_i`, not `mu_i`, that
    decides.
    """
    return 1.0 / local_coherence(Phi)


def affine_circuit_size(Phi: np.ndarray, tol: float = 1e-9) -> Dict[str, Any]:
    """Smallest affine circuit `q` of the columns, and the dimensional cap it forces.

    An affine circuit is a minimally affinely dependent set: `sum_{j in C} l_j phi_j = 0` with
    `sum_j l_j = 0` and every `l_j != 0`. Any `d + 2` points in `R^d` are affinely dependent, so
    `q <= d + 2` whenever `F > d + 1`.

    A circuit bounds the collision radius. Given one, take `i` with `|l_i|` largest and set
    `z_j = -l_j / l_i` on `C \\ {i}`, zero elsewhere. Then `Phi_{-i} z = phi_i`; the affine
    constraint `1^T z = 1` holds *automatically*, because `sum_j l_j = 0` forces
    `sum_{j != i} l_j = -l_i`; `||z||_inf <= 1` by the choice of `i`; and `||z||_1 <= q - 1`.
    Hence

        rho_min <= q - 1 <= d + 1,      and so      s_aff_robust <= ceil(d / 2).

    This is a hard dimensional ceiling on the affine level: no code, however well designed,
    supports featurewise affine support recovery beyond `ceil(d/2)`. It is the affine counterpart
    of the analog floor -- one bounds error from below, the other bounds sparsity from above, and
    both follow from the dimension alone.

    Returns the dimensional bound always, and the exact `q` by ascending search when the code is
    small enough for that to be cheap.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    out: Dict[str, Any] = {
        "d": int(d), "F": int(F),
        "q_upper_bound": int(min(F, d + 2)),
        "rho_min_upper_bound": float(min(F, d + 2) - 1),
        "s_aff_robust_cap": int(np.ceil(d / 2.0)),
    }
    if F <= 40 and d <= 12:
        import itertools as _it
        for k in range(2, min(F, d + 3) + 1):
            found = False
            for C in _it.combinations(range(F), k):
                M = Phi[:, list(C)]
                if np.linalg.matrix_rank(M[:, 1:] - M[:, :1], tol=tol) < k - 1:
                    out["q_exact"], out["circuit"] = k, list(C)
                    found = True
                    break
            if found:
                break
    return out


def collision_distance(Phi: np.ndarray, feature_index: int, sparsity: int) -> Dict[str, Any]:
    """`delta_i(s)`: the Euclidean distance between the active and inactive convex hulls.

    The admissible set of `z = v - u` at sparsity `s` is exactly

        Z_s = { z : ||z||_inf <= 1,  1^T z = 1,  ||z||_1 <= 2 min(s, F-s) - 1 },

    by the same reconstruction argument behind the collision radius, so

        delta_i(s) = min_{z in Z_s} || Phi_{-i} z - phi_i ||_2 .

    `rho_i` is the exact-representation version of this programme; `delta_i(s)` is its residual
    once the budget is capped at what sparsity `s` permits. So `delta_i(s) = 0` exactly when
    `rho_i <= 2 min(s, F-s) - 1`, which is exactly when the feature is not separable -- the two
    quantities are two readings of one object.

    **The margin is half this distance.** The margin programme is `min ||w||` subject to
    `min_{u in D} w^T u >= 1` with `D = C_i^+ - C_i^-`. Support-function/distance duality gives
    `max_{||w|| <= 1} min_{u in D} w^T u = dist(0, D) = delta_i(s)`, hence `||w*|| = 1/delta_i(s)`
    and `gamma_i(s) = 1/(2||w*||) = delta_i(s)/2`.

    **Noise robustness, as a corollary.** With the optimal unit-norm `w` and its centred bias, a
    perturbed input `x + eta` still receives the correct label for feature `i` on every support of
    size `s` provided `||eta||_2 < delta_i(s)/2 = gamma_i(s)`, since the score moves by at most
    `||w|| ||eta||`. The robust margin is therefore a certified noise tolerance in the units of
    the representation, not merely a scale.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i, s = int(feature_index), int(sparsity)
    P = np.delete(Phi, i, axis=1)
    phi = Phi[:, i]
    n = F - 1
    budget = 2.0 * min(s, F - s) - 1.0
    t0 = time.perf_counter()

    # Variables [z (n) | t (n)] with t >= |z|, 1^T t <= budget, 1^T z = 1, |z| <= 1.
    A_ub = np.vstack([np.hstack([np.eye(n), -np.eye(n)]),
                      np.hstack([-np.eye(n), -np.eye(n)]),
                      np.concatenate([np.zeros(n), np.ones(n)])[None, :]])
    b_ub = np.concatenate([np.zeros(2 * n), [budget]])
    A_eq = np.concatenate([np.ones(n), np.zeros(n)])[None, :]

    def obj(x):
        r = P @ x[:n] - phi
        return float(r @ r)

    def jac(x):
        g = np.zeros(2 * n)
        g[:n] = 2.0 * (P.T @ (P @ x[:n] - phi))
        return g

    x0 = np.zeros(2 * n)
    x0[0] = x0[n] = 1.0
    res = minimize(obj, x0, jac=jac, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda x: b_ub - A_ub @ x,
                                 "jac": lambda x: -A_ub},
                                {"type": "eq", "fun": lambda x: A_eq @ x - 1.0,
                                 "jac": lambda x: A_eq}],
                   bounds=[(-1.0, 1.0)] * n + [(0.0, None)] * n,
                   options={"maxiter": 2000, "ftol": 1e-14})
    dist = float(np.sqrt(max(res.fun, 0.0))) if res.success else float("nan")
    return {"feature": i, "sparsity": s, "budget": budget, "delta": dist,
            "margin_from_delta": dist / 2.0, "success": bool(res.success),
            "runtime_s": time.perf_counter() - t0}


def separable_from_rho(rho: float, s: int, F: int) -> bool:
    """The frontier rule: separable at `s` iff `rho > 2 min(s, F-s) - 1`."""
    return bool(rho > 2.0 * min(s, F - s) - 1.0 + 1e-9)


def collision_frontier(Phi: np.ndarray, feature_subset: Optional[Sequence[int]] = None,
                       model: str = "atmost", alpha: float = 1.0) -> Dict[str, Any]:
    """The complete robust affine frontier of a code, from one linear programme per feature.

    ``model="atmost"`` (the default) is the primary statement: supports of size **at most** `s`,
    threshold `kappa_i > s`, monotone by construction. ``model="exact"`` is the earlier
    exactly-`s` formulation, kept because the published E2/E6 results are stated in it; its
    threshold is `2 min(s, F-s) - 1`, which is non-monotone past `F/2`.

    ``alpha`` is the minimum detectable amplitude and applies to the at-most model only. Since
    the state amplitudes wash out of the convex hulls, this single scalar is the whole
    dependence of the affine level on the state distribution.

    Either way the cost is `F` linear programmes for the entire frontier, against one convex
    programme per (feature, sparsity) pair for :func:`robust_affine_frontier`.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    if model not in ("atmost", "exact"):
        raise ValueError(f"unknown model {model!r}; use 'atmost' or 'exact'")
    if model == "exact" and alpha != 1.0:
        raise ValueError("alpha applies to the at-most model only; the exactly-s formulation "
                         "has no amplitude version")
    feats = list(range(F)) if feature_subset is None else [int(x) for x in feature_subset]

    if model == "atmost":
        recs = [collision_radius_atmost(Phi, i, alpha=alpha) for i in feats]
        key = "rho_hat"
    else:
        recs = [collision_radius(Phi, i) for i in feats]
        key = "rho"
    vals = [r[key] for r in recs]

    return {
        "d": int(d), "F": int(F), "model": model, "alpha": float(alpha),
        "features_tested": len(feats),
        "is_upper_bound": feature_subset is not None,
        key: vals,
        # `frontier_min` is the stable name: callers should not have to know whether the model in
        # force calls its scalar `rho` or `rho_hat`. The model-specific keys are kept alongside it
        # because the two quantities are not interchangeable and mixing them would be a real error.
        "frontier_min": float(min(vals)),
        f"{key}_min": float(min(vals)),
        "argmin_feature": feats[int(np.argmin(vals))],
        "s_aff_robust": int(min(r["s_max"] for r in recs)),
        "per_feature": recs,
        "runtime_s": float(sum(r["runtime_s"] for r in recs)),
    }


def robust_affine_margin_compact(Phi: np.ndarray, feature_index: int, sparsity: int,
                                 theta: float = 0.5) -> Dict[str, Any]:
    """Reference implementation via the compact robust-counterpart reformulation.

    The worst-case terms are support functions of a hypersimplex, so LP duality replaces each
    with `O(F)` auxiliary variables and constraints instead of exponentially many. This is the
    cardinality-constrained robust-optimisation reduction of Bertsimas and Sim, applied here to
    a classification constraint:

        U_s(a)   = min  { s r + sum_j t_j : r + t_j >= a_j,  t >= 0 }
        L_{s-1}(a) = max { (s-1) p - sum_j q_j : p - q_j <= a_j,  q >= 0 }

    so the separability constraint becomes a finite linear system in `(w, p, q, r, t)`.

    This exists to cross-check :func:`robust_affine_margin`, which reaches the same feasible set
    by constraint generation. Agreement between two independent routes is the evidence that the
    separation oracle is not silently missing constraints.

    Scope of what it returns. The lifted problem is solved here as an LP that maximises slack,
    so the verdict (separable or not) is exact, but the `w` it returns is merely *some* feasible
    point rather than the minimum-norm one. Its margin therefore **lower-bounds** the true
    robust margin, which is what :func:`robust_affine_margin` computes. The tests use it that
    way: verdicts must match exactly, margins must satisfy the inequality.

    Constraint generation stays the production path: its per-iteration cost is a sort, so it
    scales in `F`, whereas this formulation grows to `d + 2F` variables.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i, s = int(feature_index), int(sparsity)
    others = np.delete(np.arange(F), i)
    n = others.size
    P = Phi[:, others]                                   # d x n
    phi_i = Phi[:, i]
    t0 = time.perf_counter()

    # Variable layout: w (d) | p | q (n) | r | t (n) | tau
    nv = d + 1 + n + 1 + n + 1
    iw, ip, iq, ir, it, ita = 0, d, d + 1, d + 1 + n, d + 2 + n, d + 2 + 2 * n

    rows, rhs = [], []
    # -(margin expression) + tau <= 0
    row = np.zeros(nv)
    row[iw:iw + d] = -phi_i
    row[ip] = -(s - 1)
    row[iq:iq + n] = 1.0
    row[ir] = s
    row[it:it + n] = 1.0
    row[ita] = 1.0
    rows.append(row); rhs.append(0.0)
    # p - q_j - w^T phi_j <= 0
    block = np.zeros((n, nv))
    block[:, iw:iw + d] = -P.T
    block[:, ip] = 1.0
    block[np.arange(n), iq + np.arange(n)] = -1.0
    rows.append(block); rhs.extend([0.0] * n)
    # -(r + t_j) + w^T phi_j <= 0
    block2 = np.zeros((n, nv))
    block2[:, iw:iw + d] = P.T
    block2[:, ir] = -1.0
    block2[np.arange(n), it + np.arange(n)] = -1.0
    rows.append(block2); rhs.extend([0.0] * n)

    A_ub = np.vstack([r if r.ndim == 2 else r[None, :] for r in rows])
    b_ub = np.array(rhs)
    big = float(np.sqrt(d)) * 4.0 + 1.0
    bounds = ([(-1.0, 1.0)] * d + [(-big, big)] + [(0.0, big)] * n
              + [(-big, big)] + [(0.0, big)] * n + [(None, 1.0)])
    c = np.zeros(nv); c[ita] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    rec: Dict[str, Any] = {"d": int(d), "F": int(F), "feature": i, "sparsity": s,
                           "method": "compact", "runtime_s": time.perf_counter() - t0}
    if not res.success:
        rec.update({"status": "solver_failure", "message": res.message})
        return rec
    tau = float(res.x[ita])
    if tau <= TOL:
        rec["status"] = "infeasible"
        return rec

    # Feasible. This LP finds *a* feasible point, not the minimum-norm one, so the margin it
    # yields lower-bounds the exact one. That is an objective difference, not a formulation gap:
    # the lifted system describes the same feasible set. The exact margin comes from
    # `robust_affine_margin`, whose value is checked against the definition -- the minimum-norm
    # solution over *all* (A, B) constraints -- in `tests/test_theory_g124.py`.
    w = res.x[iw:iw + d] / tau
    gap = worst_case_scores(w @ Phi, i, s)["gap"]
    if gap <= TOL:
        rec.update({"status": "solver_failure", "message": f"lifted feasible but gap={gap:.2e}"})
        return rec
    w = w / gap
    final = worst_case_scores(w @ Phi, i, s)
    rec.update({
        "status": "separable",
        "margin": 1.0 / (2.0 * float(np.linalg.norm(w))),
        "w_norm": float(np.linalg.norm(w)),
        "w": w.tolist(),
        "bias": float(theta - (final["worst_active"] + final["worst_inactive"]) / 2.0),
        "worst_active_support": final["active_support"].tolist(),
        "worst_inactive_support": final["inactive_support"].tolist(),
    })
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


# =====================================================================================
# G3 -- the finite nonlinear witness. Consumes the frontier's certificate.
# =====================================================================================

def _decompose_capped_box(u: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Write `u` in `{u in [0,1]^n : 1^T u <= k}` as a convex combination of its 0/1 vertices.

    The vertices are the indicators of at most `k` coordinates, so the polytope is the
    independent-set polytope of the uniform matroid. The obvious level-set decomposition --
    sort, then take nested top-`j` indicators -- does **not** work here, because a point with
    `1^T u <= k` can have far more than `k` nonzeros and the wide indicators are not vertices.

    Peeling does. Repeatedly take the `k` largest positive coordinates, subtract their minimum
    from all of them, and record that indicator with that weight. Each step zeroes at least one
    coordinate, so it terminates in at most `n` steps, and it removes `t * |S|` from `1^T u`
    while spending weight `t`, so the total weight is about `1^T u / k <= 1`. Whatever slack is
    left goes on the zero vector, which is also a vertex.

    The weights are asserted to be convex and the reconstruction exact; nothing here is trusted
    on the strength of the argument above.
    """
    u = np.array(u, dtype=np.float64)
    n = u.size
    if k <= 0:
        if np.any(u > TOL):
            raise ValueError(f"cannot decompose a nonzero point with budget k={k}")
        return [{"support": [], "weight": 1.0}]

    parts: List[Dict[str, Any]] = []
    r = u.copy()
    for _ in range(n + 1):
        pos = np.flatnonzero(r > TOL)
        if pos.size == 0:
            break
        S = pos[np.argsort(-r[pos])[:k]]
        t = float(r[S].min())
        parts.append({"support": sorted(int(j) for j in S), "weight": t})
        r[S] -= t
        r[np.abs(r) <= TOL] = 0.0
    if np.any(np.abs(r) > 1e-8):
        raise ValueError(f"peeling did not terminate; residual {np.abs(r).max():.3e}")

    spent = float(sum(p["weight"] for p in parts))
    if spent > 1.0 + 1e-8:
        raise ValueError(f"decomposition weight {spent:.6f} exceeds one, so it is not convex")
    if spent < 1.0 - 1e-12:
        parts.append({"support": [], "weight": 1.0 - spent})

    got = np.zeros(n)
    for p in parts:
        if p["support"]:
            got[np.asarray(p["support"], dtype=int)] += p["weight"]
    if not np.allclose(got, u, atol=1e-8):
        raise ValueError(f"reconstruction error {np.abs(got - u).max():.3e}")
    return parts


def radon_witness(Phi: np.ndarray, feature_index: int, s: int,
                  alpha: float = 1.0) -> Dict[str, Any]:
    """Explicit states that **no** affine rule can label correctly for feature `i` at sparsity `s`.

    When `kappa_i <= s` the two hulls meet, and the meeting point is a convex combination of
    active states that equals a convex combination of inactive ones:

        sum_k lambda_k x_k^+  =  sum_l mu_l x_l^- .

    Any affine `f` would have to make the left side positive and the right side negative at the
    same point, so the listed states are a finite, self-contained refutation. This is what turns
    "a solver reports the hulls intersect somewhere among millions of states" into "these
    explicit states cannot all be labelled by any affine rule", and the trained network can then
    be evaluated on exactly those states.

    The certificate is built from the frontier's own optimal `z`, so nothing new is solved: with
    `u = z^-` and `v = z^+`, `Phi (e_i + u) = Phi v`, and the budgets `1 + sum u <= kappa_i <= s`
    and `sum v <= kappa_i <= s` are what make both sides admissible at sparsity `s`. Each side is
    then decomposed into Boolean states by :func:`_decompose_capped_box`.

    **Not implemented:** the Radon/Caratheodory compression to at most `d + 2` states. The
    witness returned is valid but may be larger than that bound, and `n_states` reports what it
    actually is rather than what the theory permits.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    d, F = Phi.shape
    i = int(feature_index)
    s = int(s)
    rec = collision_radius_atmost(Phi, i, alpha=alpha)
    kappa = rec["rho_hat"]
    if not np.isfinite(kappa) or kappa > s + TOL:
        return {"feature": i, "s": s, "kappa": kappa, "status": "separable",
                "reason": f"kappa={kappa} > s={s}, so no collision exists at this sparsity"}

    z = np.asarray(rec["z"], dtype=np.float64)
    v = np.clip(z, 0.0, None)                 # inactive side
    u = np.clip(-z, 0.0, None)                # extra active features, beyond i itself
    others = [j for j in range(F) if j != i]

    active = _decompose_capped_box(u, s - 1)
    inactive = _decompose_capped_box(v, s)

    def lift(support: List[int], with_i: bool) -> List[int]:
        out = [others[j] for j in support]
        if with_i:
            out.append(i)
        return sorted(out)

    act = [{"support": lift(p["support"], True), "weight": p["weight"]} for p in active]
    ina = [{"support": lift(p["support"], False), "weight": p["weight"]} for p in inactive]

    # Verify the identity from Phi alone, the way a third party would.
    def combine(states: List[Dict[str, Any]]) -> np.ndarray:
        acc = np.zeros(d)
        for st in states:
            if st["support"]:
                acc += st["weight"] * Phi[:, np.asarray(st["support"], dtype=int)].sum(axis=1)
        return acc

    residual = float(np.linalg.norm(combine(act) - combine(ina)))
    sizes_ok = (all(len(st["support"]) <= s and i in st["support"] for st in act)
                and all(len(st["support"]) <= s and i not in st["support"] for st in ina))
    return {
        "feature": i, "s": s, "kappa": kappa, "status": "witness",
        "active": act, "inactive": ina,
        "n_states": len(act) + len(ina),
        "radon_bound": int(d + 2),
        "residual": residual,
        "supports_admissible": bool(sizes_ok),
        "verified": bool(residual < 1e-6 and sizes_ok),
    }
