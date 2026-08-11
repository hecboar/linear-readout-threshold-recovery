"""The witness has to be a refutation, not a report: no affine rule may label it correctly.

`radon_witness` claims to return explicit states that defeat *every* affine decoder. That claim
is checkable directly -- ask a linear programme for a separating `(w, theta)` on exactly those
states and require it to be infeasible -- so it is checked here rather than argued.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from lrtr.affine_frontier import (
    _decompose_capped_box,
    collision_radius_atmost,
    radon_witness,
)
from lrtr.codes import random_unit_code


def affinely_separable(X_pos: np.ndarray, X_neg: np.ndarray) -> bool:
    """Is there `(w, theta)` with `w^T x > theta` on X_pos and `< theta` on X_neg?

    Asked with a unit margin, which is a normalisation and not a restriction: any strict
    separator can be scaled to achieve it, since the state sets are finite.
    """
    d = X_pos.shape[1]
    nv = d + 1
    A_ub, b_ub = [], []
    for x in X_pos:                       # theta - w^T x <= -1
        row = np.zeros(nv); row[:d] = -x; row[d] = 1.0
        A_ub.append(row); b_ub.append(-1.0)
    for x in X_neg:                       # w^T x - theta <= -1
        row = np.zeros(nv); row[:d] = x; row[d] = -1.0
        A_ub.append(row); b_ub.append(-1.0)
    res = linprog(np.zeros(nv), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(None, None)] * nv, method="highs")
    return bool(res.success)


def states_of(Phi: np.ndarray, entries) -> np.ndarray:
    return np.array([Phi[:, np.asarray(st["support"], dtype=int)].sum(axis=1)
                     for st in entries])


# =====================================================================================
# The decomposition it rests on
# =====================================================================================

@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_decomposition_is_convex_exact_and_respects_the_cap(k):
    rng = np.random.default_rng(k)
    for _ in range(20):
        n = 12
        u = rng.random(n)
        u *= min(1.0, k / max(u.sum(), 1e-12)) * rng.uniform(0.3, 1.0)
        parts = _decompose_capped_box(u, k)
        assert sum(p["weight"] for p in parts) == pytest.approx(1.0, abs=1e-9)
        assert all(p["weight"] >= -1e-12 for p in parts)
        assert all(len(p["support"]) <= k for p in parts)
        got = np.zeros(n)
        for p in parts:
            if p["support"]:
                got[np.asarray(p["support"], dtype=int)] += p["weight"]
        assert got == pytest.approx(u, abs=1e-8)


def test_decomposition_rejects_a_point_it_cannot_represent():
    with pytest.raises(ValueError, match="budget k=0"):
        _decompose_capped_box(np.array([0.5, 0.0]), 0)


def test_level_set_shortcut_would_have_been_wrong():
    """A point can satisfy the cap while having many more nonzeros than the cap allows.

    This is why the decomposition peels instead of using nested top-j indicators: the wide
    indicators are not vertices of this polytope.
    """
    u = np.full(20, 0.1)                       # 1^T u = 2 <= k, but 20 nonzeros
    parts = _decompose_capped_box(u, 2)
    assert all(len(p["support"]) <= 2 for p in parts)


# =====================================================================================
# The witness itself
# =====================================================================================

def test_duplicate_columns_give_the_minimal_witness():
    """`[I, I]` fails at s = 2, and the reason is a pair of identical representations."""
    d = 4
    Phi = np.hstack([np.eye(d), np.eye(d)])
    w = radon_witness(Phi, 0, s=2)
    assert w["status"] == "witness" and w["verified"]
    assert w["kappa"] == pytest.approx(1.0, abs=1e-9)
    assert w["n_states"] == 2
    assert not affinely_separable(states_of(Phi, w["active"]),
                                  states_of(Phi, w["inactive"]))


def test_no_witness_is_offered_where_the_feature_is_separable():
    Phi = random_unit_code(10, 20, np.random.default_rng(3))
    rec = collision_radius_atmost(Phi, 0)
    s = int(np.floor(rec["rho_hat"]))          # strictly below the frontier
    w = radon_witness(Phi, 0, s=max(s, 1))
    if rec["rho_hat"] > s:
        assert w["status"] == "separable"
        assert "active" not in w


@pytest.mark.parametrize("d,F,seed", [(6, 18, 0), (8, 24, 1), (5, 20, 2)])
def test_every_witness_defeats_every_affine_rule(d, F, seed):
    """The whole point. Extracted states must be inseparable by any (w, theta)."""
    Phi = random_unit_code(d, F, np.random.default_rng(seed))
    checked = 0
    for i in range(min(F, 8)):
        rec = collision_radius_atmost(Phi, i)
        kappa = rec["rho_hat"]
        if not np.isfinite(kappa):
            continue
        s = int(np.ceil(kappa))                # the first sparsity at which it fails
        w = radon_witness(Phi, i, s=s)
        if w["status"] != "witness":
            continue
        assert w["verified"], w
        assert w["residual"] < 1e-6
        assert not affinely_separable(states_of(Phi, w["active"]),
                                      states_of(Phi, w["inactive"])), (i, s, kappa)
        checked += 1
    assert checked > 0, "no failing feature found, so nothing was tested"


def test_witness_states_are_admissible_and_labelled_consistently():
    Phi = random_unit_code(6, 18, np.random.default_rng(7))
    i = 0
    kappa = collision_radius_atmost(Phi, i)["rho_hat"]
    s = int(np.ceil(kappa))
    w = radon_witness(Phi, i, s=s)
    assert w["status"] == "witness"
    for st in w["active"]:
        assert i in st["support"] and len(st["support"]) <= s
    for st in w["inactive"]:
        assert i not in st["support"] and len(st["support"]) <= s
    for side in ("active", "inactive"):
        assert sum(st["weight"] for st in w[side]) == pytest.approx(1.0, abs=1e-9)


def test_brute_force_agrees_that_the_hulls_meet():
    """Independent confirmation: enumerate Boolean states and separate them by LP.

    If the witness says feature `i` fails at `s`, then the *full* state sets at that sparsity
    must be inseparable too -- the witness is a subset of them, so this is a weaker check, but it
    is the one that does not go through our own construction at all.
    """
    d, F, s = 5, 12, 2
    Phi = random_unit_code(d, F, np.random.default_rng(11))
    for i in range(F):
        kappa = collision_radius_atmost(Phi, i)["rho_hat"]
        pos, neg = [], []
        for k in range(1, s + 1):
            for S in itertools.combinations(range(F), k):
                x = Phi[:, list(S)].sum(axis=1)
                (pos if i in S else neg).append(x)
        brute = affinely_separable(np.array(pos), np.array(neg))
        assert brute == (kappa > s + 1e-9), (i, kappa, brute)
