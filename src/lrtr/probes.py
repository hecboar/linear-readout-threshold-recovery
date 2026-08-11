"""Affine support-recovery probes: the strongest baseline the network has to beat.

The middle level of the interface hierarchy is *affine support recovery*. Two things measure it,
and they answer different questions:

* :mod:`lrtr.affine_frontier` computes the **exact worst-case frontier** of the code -- the
  largest sparsity at which *some* featurewise affine rule succeeds on *every* support. It is a
  property of the code and needs no data.
* this module fits the **best affine rule we can actually train**, and scores it on held-out
  states. It is an empirical, average-case quantity.

Both are needed. The frontier says what is possible; the probe says what is achievable by
fitting, which is the comparison a referee cares about when the claim is that a trained network
decodes better than a linear readout of its own representation.

What the audit demanded, and what is here. The published protocol fitted one least-squares
readout, without an intercept, at a single sparsity, on states drawn from the same stream as the
evaluation states, and scored it at a fixed threshold. Every one of those is a way to understate
the baseline. This module instead offers:

* two representations -- pre-ReLU `Phi b` and post-ReLU `ReLU(Phi b)`;
* three probe families -- ridge, logistic and squared-hinge (linear SVM) -- all with an intercept;
* a regularisation grid, selected on validation;
* three threshold policies -- the semantically fixed `theta`, a validation-selected global
  threshold, and validation-selected per-feature thresholds;
* three disjoint splits, with disjointness enforced rather than assumed (:mod:`lrtr.splits`);
* both a **global** interface, fitted once on a sparsity mixture, and a **per-sparsity oracle**,
  refitted and retuned at each sparsity. The oracle is an upper envelope, not an implementable
  interface, and is labelled as such everywhere it appears.

Nothing in the selection path ever sees the locked test set. `tests/test_probes.py` enforces that
structurally: it poisons the test set and asserts that not one selected hyperparameter moves.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .splits import SplitBundle, StateSplit, one_hot_targets, representations
from .stats import wilson_interval
from .threshold import s95_from_curve, s95_interpolated

__all__ = [
    "PROBE_FAMILIES",
    "REPRESENTATIONS",
    "THRESHOLD_POLICIES",
    "fit_probe",
    "score_probe",
    "evaluate_scores",
    "evaluate_network",
    "evaluate_fixed_readout",
    "select_probe",
    "SELECTION_OBJECTIVES",
    "select_thresholds",
    "evaluate_probe",
    "probe_profile",
    "distribution_profile",
]

PROBE_FAMILIES = ("ridge", "logistic", "svm")
REPRESENTATIONS = ("pre", "post")
THRESHOLD_POLICIES = ("fixed", "global", "per_feature")
DEFAULT_RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)


def _design(R: np.ndarray) -> np.ndarray:
    return np.hstack([R, np.ones((R.shape[0], 1))])


def _xty(X: np.ndarray, split: StateSplit) -> np.ndarray:
    """`X^T Y` for one-hot `Y`, by scatter-add: never forms the `(n, F)` target matrix."""
    out = np.zeros((X.shape[1], split.F))
    for j in range(split.supports.shape[1]):
        active = split.sparsity > j
        if not active.any():
            break
        np.add.at(out.T, split.supports[active, j], X[active])
    return out


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------

def _fit_ridge(X: np.ndarray, split: StateSplit, grid: Sequence[float]) -> Dict[float, np.ndarray]:
    """Ridge for every penalty on the grid, from one pass over the data.

    The intercept is never penalised. `X^T X` is `(d+1) x (d+1)` and `X^T Y` is `(d+1) x F`, both
    accumulated once; each penalty then costs one small solve, so the size of the grid is never a
    reason to shrink it.
    """
    XtX = X.T @ X
    XtY = _xty(X, split)
    p = XtX.shape[0]
    mask = np.ones(p)
    mask[-1] = 0.0                                     # intercept unpenalised
    out: Dict[float, np.ndarray] = {}
    for lam in grid:
        out[float(lam)] = np.linalg.solve(XtX + lam * np.diag(mask), XtY)
    return out


def _fit_margin_family(X: np.ndarray, split: StateSplit, grid: Sequence[float], kind: str,
                       steps: int, batch: int, lr: float, seed: int) -> Dict[float, np.ndarray]:
    """Logistic or squared-hinge fit, minibatched over states, vectorised over all `F` features.

    Both are multi-label problems sharing one design matrix, so the weights are a single
    `(d+1, F)` array and one pass costs one `(m, F)` product. The `(n, F)` logit matrix is never
    materialised -- that is what makes `F = 800` at `n = 100 000` affordable on a CPU.

    The label matrix is built **once**, before the loops. It used to be rebuilt every step, which
    meant constructing a `StateSplit` object and a dense `(batch, F)` one-hot array inside the
    hottest loop in the campaign -- of the order of `10^5` times per model. That was the cost
    centre: the arithmetic here is tiny, and what dominated was Python-level allocation. Indexing
    a precomputed `Y` gives exactly the same batches, because the random stream is untouched.
    """
    n, p = X.shape
    F = split.F
    Y = one_hot_targets(split)
    rng = np.random.default_rng(seed)
    out: Dict[float, np.ndarray] = {}
    for lam in grid:
        W = np.zeros((p, F))
        vel = np.zeros_like(W)
        for t in range(steps):
            idx = rng.integers(0, n, size=min(batch, n))
            Xb = X[idx]
            Yb = Y[idx]
            Zb = Xb @ W
            if kind == "logistic":
                # Stable sigmoid: exp(-|z|) cannot overflow, unlike exp(-z) for very negative z.
                e = np.exp(-np.abs(Zb))
                sig = np.where(Zb >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
                G = Xb.T @ (sig - Yb) / len(idx)
            else:                                       # squared hinge on +-1 labels
                S = 2.0 * Yb - 1.0
                slack = np.maximum(0.0, 1.0 - S * Zb)
                G = Xb.T @ (-2.0 * S * slack) / len(idx)
            G[:-1] += lam * W[:-1]                      # intercept unpenalised
            vel = 0.9 * vel - lr * G
            W += vel
        out[float(lam)] = W
    return out


def fit_probe(W_in: np.ndarray, split: StateSplit, family: str, representation: str,
              grid: Sequence[float] = DEFAULT_RIDGE_GRID, steps: int = 300, batch: int = 4096,
              lr: float = 0.5, seed: int = 0) -> Dict[float, np.ndarray]:
    """Fit one probe family over a regularisation grid. Returns `{penalty: W (d+1, F)}`."""
    if family not in PROBE_FAMILIES:
        raise ValueError(f"unknown family {family!r}; use one of {PROBE_FAMILIES}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; use 'pre' or 'post'")
    X = _design(representations(W_in, split, post_relu=(representation == "post")))
    if family == "ridge":
        return _fit_ridge(X, split, grid)
    return _fit_margin_family(X, split, grid, family, steps, batch, lr, seed)


def score_probe(W_in: np.ndarray, split: StateSplit, W: np.ndarray,
                representation: str) -> np.ndarray:
    """`(n, F)` scores of a fitted probe on a split."""
    X = _design(representations(W_in, split, post_relu=(representation == "post")))
    return X @ W


# --------------------------------------------------------------------------------------
# Selection -- validation only
# --------------------------------------------------------------------------------------

def _exact_recovery(Z: np.ndarray, split: StateSplit, theta: np.ndarray | float) -> np.ndarray:
    """Per-state indicator that *every* feature was labelled correctly."""
    Y = one_hot_targets(split) > 0.5
    return np.all((Z >= theta) == Y, axis=1)


def select_thresholds(Z_val: np.ndarray, val: StateSplit, policy: str, theta_fixed: float = 0.5,
                      n_grid: int = 64) -> np.ndarray | float:
    """Choose thresholds on validation scores. Never called with test data.

    ``fixed`` keeps the semantically meaningful `theta` -- for unit-gain Boolean targets the
    midpoint is not an arbitrary choice, and reporting it keeps the comparison with the
    theory honest. ``global`` maximises the exact-recovery rate over a quantile grid.
    ``per_feature`` gives every feature its own absolute threshold, but selects them jointly:
    `theta_i = quantile(Z_val[:, i], q)` for a single level `q` chosen to maximise exact recovery
    on validation. The per-feature adaptivity is real -- each feature gets a threshold matched to
    its own score distribution -- while the *objective* is the metric actually reported.

    Tuning each threshold independently, on any per-feature criterion, is the trap here and it
    was measured before this was written. Exact recovery needs all `F` coordinates right at once,
    so a per-coordinate error rate `eps` gives roughly `(1-eps)^F` exact recovery: independent
    per-feature optima each accept a few false positives, and those compound. Both raw and
    balanced per-feature accuracy came out *worse* than a single fixed threshold for that reason.
    Selecting one shared quantile level fixes it without giving up the per-feature scale.
    """
    if policy not in THRESHOLD_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; use one of {THRESHOLD_POLICIES}")
    if policy == "fixed":
        return float(theta_fixed)

    Y = one_hot_targets(val) > 0.5
    if policy == "global":
        grid = np.quantile(Z_val, np.linspace(0.01, 0.999, n_grid))
        best, best_theta = -1.0, float(theta_fixed)
        for t in grid:
            score = float(np.all((Z_val >= t) == Y, axis=1).mean())
            if score > best:
                best, best_theta = score, float(t)
        return best_theta

    # per_feature: one shared quantile level, per-feature absolute thresholds.
    qs = np.linspace(0.5, 0.9995, n_grid)
    best, best_thetas = -1.0, np.full(Z_val.shape[1], float(theta_fixed))
    for q in qs:
        thetas = np.quantile(Z_val, q, axis=0)
        score = float(np.all((Z_val >= thetas) == Y, axis=1).mean())
        if score > best:
            best, best_thetas = score, thetas
    return best_thetas


SELECTION_OBJECTIVES = ("curve_auc", "mixture_exact")


def _val_curve(W_in: np.ndarray, val_by_s: Dict[int, StateSplit], W: np.ndarray,
               representation: str, theta) -> List[Tuple[int, float]]:
    rows = []
    for s in sorted(val_by_s):
        sub_ = val_by_s[s]
        if len(sub_) == 0:
            continue
        Z = score_probe(W_in, sub_, W, representation)
        rows.append((s, float(_exact_recovery(Z, sub_, theta).mean())))
    return rows


def select_probe(W_in: np.ndarray, train: StateSplit, val: StateSplit, family: str,
                 representation: str, policy: str, grid: Sequence[float] = DEFAULT_RIDGE_GRID,
                 theta_fixed: float = 0.5, val_by_s: Optional[Dict[int, StateSplit]] = None,
                 objective: str = "curve_auc",
                 fits: Optional[Dict[float, np.ndarray]] = None, **fit_kw) -> Dict[str, Any]:
    """Fit over the grid, pick the penalty and thresholds on validation, return the winner.

    ``objective`` decides what "best on validation" means, and it must match what is reported or
    the selection quietly optimises the wrong thing. Measured on a small model, choosing by
    exact recovery over the sparsity *mixture* picked configurations whose recovery **AUC over
    the curve** was four times worse than the best -- the mixture is dominated by whichever
    sparsities happen to be easy. So the default is ``curve_auc``: the area under the
    per-sparsity validation recovery curve, which is the co-primary estimand of the analysis
    plan. ``mixture_exact`` is retained for comparison and needs no per-sparsity validation
    split.

    ``fits`` lets a caller supply the fitted weights instead of having them refitted here. The
    fit depends on ``(family, representation, train, penalty)`` and **not** on ``policy``, which
    only chooses thresholds from validation scores afterwards, so a caller sweeping policies can
    fit once and pass the result in. :func:`probe_profile` does exactly that.
    """
    if objective not in SELECTION_OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; use one of {SELECTION_OBJECTIVES}")
    if objective == "curve_auc" and not val_by_s:
        objective = "mixture_exact"          # no per-sparsity validation available
    if fits is None:
        fits = fit_probe(W_in, train, family, representation, grid=grid, **fit_kw)
    best: Optional[Dict[str, Any]] = None
    for lam, W in fits.items():
        Z = score_probe(W_in, val, W, representation)
        theta = select_thresholds(Z, val, policy, theta_fixed=theta_fixed)
        mixture_exact = float(_exact_recovery(Z, val, theta).mean())
        if objective == "curve_auc":
            curve = _val_curve(W_in, val_by_s, W, representation, theta)
            xs = [a for a, _ in curve]
            ys = [b for _, b in curve]
            crit = (float(np.trapezoid(ys, xs) / (max(xs) - min(xs))) if len(xs) > 1
                    else (ys[0] if ys else 0.0))
        else:
            crit = mixture_exact
        if best is None or crit > best["val_criterion"]:
            best = {"W": W, "penalty": lam, "theta": theta, "val_criterion": crit,
                    "val_objective": objective, "val_exact_recovery": mixture_exact,
                    "family": family, "representation": representation, "policy": policy}
    assert best is not None
    return best


# --------------------------------------------------------------------------------------
# Locked evaluation
# --------------------------------------------------------------------------------------

def _margins(Z: np.ndarray, split: StateSplit, theta: np.ndarray | float) -> Dict[str, float]:
    """Signed distance to the threshold, worst feature per state."""
    S = 2.0 * (one_hot_targets(split) > 0.5) - 1.0
    m = np.min(S * (Z - theta), axis=1)
    return {"margin_median": float(np.median(m)), "margin_p05": float(np.quantile(m, 0.05)),
            "margin_min": float(m.min()), "frac_positive_margin": float((m > 0).mean())}


def _calibration(Z: np.ndarray, split: StateSplit, n_bins: int = 10) -> Dict[str, float]:
    """Brier score and expected calibration error of the logistic link applied to the scores."""
    # Stable sigmoid, as in the fitter: exp(-|z|) cannot overflow where exp(-z) does. Calibration
    # is evaluated on raw scores that can be large, so this is not a theoretical concern.
    e = np.exp(-np.abs(Z))
    P = np.where(Z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    Y = (one_hot_targets(split) > 0.5).astype(np.float64)
    brier = float(np.mean((P - Y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    flat_p, flat_y = P.ravel(), Y.ravel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (flat_p >= lo) & (flat_p < hi)
        if m.any():
            ece += m.mean() * abs(flat_p[m].mean() - flat_y[m].mean())
    return {"brier": brier, "ece": float(ece)}


def evaluate_scores(score_fn, test_by_s: Dict[int, StateSplit], theta: np.ndarray | float,
                    conf: float = 0.95, calibrated: bool = False) -> Dict[str, Any]:
    """Score any decision rule on the locked test sets. No selection happens here.

    `score_fn(split) -> (n, F)` is the only thing that varies between a fitted probe, a fixed
    calibrated readout and the network's own output, so all three are evaluated by this one
    function on the *same* states. That is what keeps the comparison paired: every decoder sees
    an identical set of supports at every sparsity.
    """
    rows: List[Dict[str, Any]] = []
    for s in sorted(test_by_s):
        split = test_by_s[s]
        if len(split) == 0:
            continue
        Z = score_fn(split)
        ok = _exact_recovery(Z, split, theta)
        Y = one_hot_targets(split) > 0.5
        k, n = int(ok.sum()), len(split)
        lo, hi = wilson_interval(k, n)
        rows.append({"s": s, "n_test": n, "successes": k, "p_rec": k / n,
                     "ci_low": lo, "ci_high": hi,
                     "coord_accuracy": float(((Z >= theta) == Y).mean()),
                     **_margins(Z, split, theta),
                     **(_calibration(Z, split) if calibrated else {})})
    sp = [r["s"] for r in rows]
    pr = [r["p_rec"] for r in rows]
    return {"rows": rows, "sparsities": sp,
            "s95": s95_from_curve(sp, pr), "s95_interp": s95_interpolated(sp, pr),
            "s50": s95_from_curve(sp, pr, level=0.5),
            "s50_interp": s95_interpolated(sp, pr, level=0.5),
            "recovery_auc": float(np.trapezoid(pr, sp) / (max(sp) - min(sp))) if len(sp) > 1
            else float(pr[0])}


def evaluate_probe(W_in: np.ndarray, test_by_s: Dict[int, StateSplit], W: np.ndarray,
                   representation: str, theta: np.ndarray | float,
                   conf: float = 0.95, calibrated: bool = False) -> Dict[str, Any]:
    """Locked evaluation of a fitted probe.

    ``calibrated`` reports Brier score and expected calibration error, which mean something only
    when the scores are logits. Set it for the logistic family and leave it off elsewhere, rather
    than quoting a calibration number for a ridge fit where the logistic link is arbitrary.
    """
    return evaluate_scores(lambda sp: score_probe(W_in, sp, W, representation),
                           test_by_s, theta, conf, calibrated=calibrated)


def evaluate_network(W_in: np.ndarray, W_out: np.ndarray, test_by_s: Dict[int, StateSplit],
                     theta: float = 0.5, conf: float = 0.95) -> Dict[str, Any]:
    """Locked evaluation of the network's own thresholded output, `W_out ReLU(W_in b)`."""
    def scores(split: StateSplit) -> np.ndarray:
        return representations(W_in, split, post_relu=True) @ W_out.T
    return evaluate_scores(scores, test_by_s, theta, conf)


def evaluate_fixed_readout(W_in: np.ndarray, G: np.ndarray, test_by_s: Dict[int, StateSplit],
                           theta: float = 0.5, conf: float = 0.95) -> Dict[str, Any]:
    """Locked evaluation of a fixed readout, gain-normalised to unit diagonal first.

    These are the readouts of the theory -- the calibrated pseudoinverse, the model's own
    decoder, a least-squares fit. They carry the floor statements that the fitted probes do not,
    and they are scored here on the same states so the two families sit side by side.
    """
    from .interface import gains

    G = np.ascontiguousarray(G, dtype=np.float64)
    Gc = G / gains(G, W_in)[:, None]

    def scores(split: StateSplit) -> np.ndarray:
        return representations(W_in, split, post_relu=False) @ Gc.T
    return evaluate_scores(scores, test_by_s, theta, conf)


def _tail_auc(rows: List[Dict[str, Any]], s_from: float) -> Optional[float]:
    tail = [(r["s"], r["p_rec"]) for r in rows if r["s"] > s_from]
    if len(tail) < 2:
        return None
    xs, ys = zip(*tail)
    return float(np.trapezoid(ys, xs) / (max(xs) - min(xs)))


def probe_profile(W_in: np.ndarray, bundle: SplitBundle,
                  families: Sequence[str] = PROBE_FAMILIES,
                  reps: Sequence[str] = REPRESENTATIONS,
                  policies: Sequence[str] = THRESHOLD_POLICIES,
                  grid: Sequence[float] = DEFAULT_RIDGE_GRID, theta_fixed: float = 0.5,
                  include_oracle: bool = True, objective: str = "curve_auc",
                  **fit_kw) -> Dict[str, Any]:
    """Every (family, representation, policy) combination, as a global probe and as an oracle.

    The **global** probe is fitted once on the sparsity mixture and evaluated across the whole
    curve: one interface, as a real decoder would be. The **per-sparsity oracle** is refitted and
    retuned at every sparsity from that sparsity's own train/val split. The oracle is told the
    sparsity it will be tested at, which the network is not, so it is an *upper envelope* and is
    tagged `is_oracle` wherever it appears.

    The tail AUC is taken past the global probe's own validation-selected `s95`, so the interval
    is fixed by validation data before the locked test set is touched.
    """
    out: Dict[str, Any] = {"d": int(W_in.shape[0]), "F": int(W_in.shape[1]),
                           "seed": bundle.seed, "sparsities": bundle.sparsities,
                           "splits": {"train": len(bundle.train), "val": len(bundle.val),
                                      "test_by_s": {s: len(v) for s, v in
                                                    bundle.test_by_s.items()},
                                      "n_redrawn": bundle.n_redrawn,
                                      "exhausted_sparsities": bundle.exhausted},
                           "global": {}, "oracle": {}}

    for fam in families:
        for rep in reps:
            # The weights depend on (family, representation, split, penalty) and not on the
            # threshold policy, so they are fitted once here and reused across policies. Fitting
            # inside the policy loop repeated every fit three times identically, which on the
            # margin families is the dominant cost of the whole campaign.
            global_fits = fit_probe(W_in, bundle.train, fam, rep, grid=grid, **fit_kw)
            oracle_fits: Dict[int, Dict[float, np.ndarray]] = {}
            if include_oracle:
                for s in bundle.sparsities:
                    tr = bundle.train_by_s[s]
                    if len(tr) and len(bundle.val_by_s[s]) and len(bundle.test_by_s[s]):
                        oracle_fits[s] = fit_probe(W_in, tr, fam, rep, grid=grid, **fit_kw)
            for pol in policies:
                key = f"{fam}_{rep}_{pol}"
                sel = select_probe(W_in, bundle.train, bundle.val, fam, rep, pol,
                                   grid=grid, theta_fixed=theta_fixed,
                                   val_by_s=bundle.val_by_s, objective=objective,
                                   fits=global_fits, **fit_kw)
                ev = evaluate_probe(W_in, bundle.test_by_s, sel["W"], rep, sel["theta"],
                                    calibrated=(fam == "logistic"))
                # The tail interval is fixed from validation, before the test rows are read.
                Zv = score_probe(W_in, bundle.val, sel["W"], rep)
                val_rows = []
                for s in bundle.sparsities:
                    sub = bundle.val.of_sparsity(s)
                    if len(sub):
                        Zs = score_probe(W_in, sub, sel["W"], rep)
                        val_rows.append((s, float(_exact_recovery(Zs, sub, sel["theta"]).mean())))
                s95_val = s95_from_curve([a for a, _ in val_rows],
                                         [b for _, b in val_rows]) if val_rows else 0
                out["global"][key] = {
                    "penalty": sel["penalty"], "policy": pol, "family": fam,
                    "representation": rep, "is_oracle": False,
                    "theta": (sel["theta"] if np.isscalar(sel["theta"])
                              else {"per_feature": True,
                                    "median": float(np.median(sel["theta"]))}),
                    "val_exact_recovery": sel["val_exact_recovery"],
                    "val_criterion": sel["val_criterion"],
                    "val_objective": sel["val_objective"],
                    "s95_validation": s95_val,
                    "tail_auc": _tail_auc(ev["rows"], s95_val),
                    **{k: v for k, v in ev.items() if k != "sparsities"},
                }
                del Zv

                if not include_oracle:
                    continue
                rows: List[Dict[str, Any]] = []
                for s in bundle.sparsities:
                    tr, va, te = (bundle.train_by_s[s], bundle.val_by_s[s], bundle.test_by_s[s])
                    if min(len(tr), len(va), len(te)) == 0:
                        continue
                    o = select_probe(W_in, tr, va, fam, rep, pol, grid=grid,
                                     theta_fixed=theta_fixed, objective="mixture_exact",
                                     fits=oracle_fits[s], **fit_kw)
                    ev_s = evaluate_probe(W_in, {s: te}, o["W"], rep, o["theta"],
                                          calibrated=(fam == "logistic"))
                    r = ev_s["rows"][0]
                    r["penalty"] = o["penalty"]
                    rows.append(r)
                sp = [r["s"] for r in rows]
                pr = [r["p_rec"] for r in rows]
                out["oracle"][key] = {
                    "family": fam, "representation": rep, "policy": pol, "is_oracle": True,
                    "note": "upper envelope: refitted and retuned per sparsity, not an "
                            "implementable single interface",
                    "rows": rows,
                    "s95": s95_from_curve(sp, pr), "s95_interp": s95_interpolated(sp, pr),
                    "s50": s95_from_curve(sp, pr, level=0.5),
                    "recovery_auc": (float(np.trapezoid(pr, sp) / (max(sp) - min(sp)))
                                     if len(sp) > 1 else (float(pr[0]) if pr else None)),
                }
    return out


# --------------------------------------------------------------------------------------
# The same diagnosis under several state distributions
# --------------------------------------------------------------------------------------

def distribution_profile(W_in: np.ndarray, W_out: np.ndarray, bundle: SplitBundle,
                         family_names: Sequence[str], frontier_features: Optional[Sequence[int]]
                         = None, families_cfg: Optional[Dict[str, Any]] = None,
                         theta: float = 0.5, **profile_kw) -> Dict[str, Any]:
    """Refit and rescore everything under each state distribution, on identical supports.

    The probes are **refitted and reselected per family**, which is the whole point: a probe tuned
    on Boolean states and then shown continuous amplitudes would be a straw baseline, and the
    audit already found one of those. The supports are shared across families, so a change in
    recovery is attributable to the amplitude law and not to an easier draw.

    Two things this reports that a blind six-distribution sweep would not:

    * **the theory's prediction for each family.** By E2 the affine frontier depends on the state
      law through the minimum amplitude alone, so for every truncated family the frontier is
      recomputed at that family's `alpha`. That is a falsifiable prediction, not a description.
    * **which families exact recovery even means something for.** A law whose amplitudes reach
      zero has undetectable active coordinates by construction, so its exact-recovery rate
      measures the draw. Those families carry `exact_recovery_meaningful: False` and their
      recovery numbers must not be compared against the truncated ones.
    """
    from .affine_frontier import collision_frontier
    from .state_families import apply_family_to_bundle, family

    d = W_in.shape[0]
    out: Dict[str, Any] = {"d": int(d), "F": int(W_in.shape[1]), "families": {}}
    for name in family_names:
        fam = family(name)
        # zlib.crc32, not hash(): Python randomises string hashing per process, so hash(name)
        # would draw different amplitudes on every run and silently break reproducibility.
        bb = apply_family_to_bundle(bundle, fam, d=d,
                                    seed=int(zlib.crc32(name.encode())) % (2 ** 31))
        probes = probe_profile(W_in, bb, theta_fixed=theta, **profile_kw)
        net = evaluate_network(W_in, W_out, bb.test_by_s, theta=theta)

        rec: Dict[str, Any] = {
            "alpha": fam.alpha,
            "rep_noise_sigma": fam.rep_noise_sigma,
            "exact_recovery_meaningful": fam.exact_recovery_meaningful,
            "note": fam.note,
            "network": {k: v for k, v in net.items() if k != "rows"},
            "network_rows": net["rows"],
            "probes": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                       for k, v in probes["global"].items()},
            "best_probe_s95": max(v["s95"] for v in probes["global"].values()),
            "best_probe_auc": max(v["recovery_auc"] for v in probes["global"].values()),
        }
        if fam.alpha is not None and frontier_features is not None:
            fr = collision_frontier(W_in, feature_subset=list(frontier_features),
                                    model="atmost", alpha=fam.alpha)
            rec["frontier_at_alpha"] = {
                "s_aff_robust_upper": fr["s_aff_robust"], "kappa_min": fr["frontier_min"],
                "is_upper_bound": True,
                # The frontier is worst case over supports; s95 is the 95th percentile over a
                # random draw. So the frontier should sit at or below the s95 an *optimal* affine
                # decoder attains. Our probe is fitted rather than optimal, so a violation here is
                # weak evidence about the probe and none at all about the theory.
                "at_most_best_probe_s95": bool(fr["s_aff_robust"] <= rec["best_probe_s95"]),
            }
        out["families"][name] = rec

    # Does the frontier move with alpha the way E2 requires? A weaker activation cannot be
    # easier to detect, so kappa must not increase as alpha falls.
    trunc = sorted(((v["alpha"], v["frontier_at_alpha"]["kappa_min"])
                    for v in out["families"].values()
                    if v["alpha"] is not None and "frontier_at_alpha" in v),
                   key=lambda t: -t[0])
    out["alpha_monotone"] = all(a >= b - 1e-6 for (_, a), (_, b) in zip(trunc, trunc[1:]))
    out["alpha_vs_kappa"] = [{"alpha": a, "kappa_min": k} for a, k in trunc]
    return out
