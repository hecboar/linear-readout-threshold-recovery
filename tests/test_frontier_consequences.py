"""Consequences of the collision-radius characterisation, each checked against its definition.

Five results, in the order they are stated in the manuscript:

1. `rho_i >= 1/mu_i`, refining the classical coherence condition for threshold recovery;
2. the affine circuit bound `rho_min <= q - 1 <= d + 1`, hence `s_aff_robust <= ceil(d/2)`;
3. the `[I, I]` versus `[I, H]` separation: identical analog level, incomparable affine level;
4. `delta_i(s)` is the distance between the two hulls, and the margin is exactly `delta_i(s)/2`;
5. the strict-integer frontier formula, including exact ties.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from lrtr.affine_frontier import (
    affine_circuit_size,
    coherence_bound_on_rho,
    collision_distance,
    collision_frontier,
    collision_radius,
    local_coherence,
    robust_affine_margin,
    s_max_from_rho,
    separable_from_rho,
)
from lrtr.analog_optimum import code_specific_floor, leverage
from lrtr.codes import (
    basis_hadamard_code,
    coherence,
    duplicated_basis_code,
    random_unit_code,
)


def _code(d, F, seed):
    return random_unit_code(d, F, np.random.default_rng(seed))


# =====================================================================================
# 1. The coherence bound on the collision radius
# =====================================================================================

@pytest.mark.parametrize("d,F", [(3, 8), (4, 10), (5, 14), (6, 20)])
def test_rho_is_at_least_one_over_local_coherence(d, F):
    Phi = _code(d, F, 4001 + d + F)
    rho = np.array([collision_radius(Phi, i)["rho"] for i in range(F)])
    bound = coherence_bound_on_rho(Phi)
    assert np.all(rho >= bound - 1e-7), (rho - bound).min()


@pytest.mark.parametrize("d,F", [(4, 10), (6, 20)])
def test_coherence_condition_is_sufficient_for_separability(d, F):
    """mu_i < 1/(2s-1) must imply separability -- and it must be a strict refinement of 1/(2s)."""
    Phi = _code(d, F, 909 + F)
    mu = local_coherence(Phi)
    for i in range(F):
        rho = collision_radius(Phi, i)["rho"]
        for s in range(1, F // 2 + 1):
            if mu[i] < 1.0 / (2 * s - 1) - 1e-9:
                assert separable_from_rho(rho, s, F), (i, s, mu[i], rho)
    # The refined threshold is strictly weaker than the classical one, so it certifies more.
    for s in range(1, 6):
        assert 1.0 / (2 * s - 1) > 1.0 / (2 * s)


def test_coherence_bound_is_only_a_bound():
    """rho can exceed 1/mu substantially: it is rho, not mu, that decides the frontier."""
    Phi = _code(6, 24, 77)
    rho = np.array([collision_radius(Phi, i)["rho"] for i in range(24)])
    assert np.max(rho / coherence_bound_on_rho(Phi)) > 1.5


# =====================================================================================
# 2. Affine circuit size and the dimensional cap
# =====================================================================================

@pytest.mark.parametrize("d,F", [(3, 8), (4, 10), (5, 12)])
def test_circuit_bound_and_dimensional_cap(d, F):
    Phi = _code(d, F, 313 + d * F)
    info = affine_circuit_size(Phi)
    fr = collision_frontier(Phi, model="exact")   # the circuit bound is stated for exactly-s

    assert info["q_exact"] is not None and info["q_exact"] <= d + 2
    assert fr["rho_min"] <= info["q_exact"] - 1 + 1e-7
    assert fr["rho_min"] <= d + 1 + 1e-7
    assert fr["s_aff_robust"] <= info["s_aff_robust_cap"]
    assert info["s_aff_robust_cap"] == int(np.ceil(d / 2.0))


@pytest.mark.parametrize("d,F", [(4, 12), (6, 18), (8, 24)])
def test_cap_holds_on_many_codes_including_designed_ones(d, F):
    cap = int(np.ceil(d / 2.0))
    for seed in range(6):
        Phi = _code(d, F, 5000 + seed)
        assert collision_frontier(Phi, model="exact")["s_aff_robust"] <= cap


def test_circuit_construction_yields_an_admissible_z():
    """The circuit -> z construction must satisfy all three constraints, 1^T z = 1 for free."""
    d, F = 4, 9
    Phi = _code(d, F, 4242)
    info = affine_circuit_size(Phi)
    C = info["circuit"]
    M = Phi[:, C]
    # Affine dependence: solve for lambda with M lambda = 0 and 1^T lambda = 0, lambda != 0.
    A = np.vstack([M, np.ones(len(C))])
    _, _, Vt = np.linalg.svd(A)
    lam = Vt[-1]
    assert np.allclose(M @ lam, 0, atol=1e-8) and abs(lam.sum()) < 1e-8
    k = int(np.argmax(np.abs(lam)))
    i = C[k]
    z_full = np.zeros(F)
    for pos, j in enumerate(C):
        if j != i:
            z_full[j] = -lam[pos] / lam[k]
    z = np.delete(z_full, i)
    P = np.delete(Phi, i, axis=1)
    assert P @ z == pytest.approx(Phi[:, i], abs=1e-8)
    assert z.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.max(np.abs(z)) <= 1.0 + 1e-9
    assert np.abs(z).sum() <= len(C) - 1 + 1e-9
    assert collision_radius(Phi, i)["rho"] <= np.abs(z).sum() + 1e-7


# =====================================================================================
# 3. [I, I] versus [I, H]: same analog level, incomparable affine level
# =====================================================================================

@pytest.mark.parametrize("d", [8, 16, 32])
def test_analog_level_cannot_distinguish_the_two_codes(d):
    A, B = duplicated_basis_code(d), basis_hadamard_code(d)
    for P in (A, B):
        assert np.allclose(np.linalg.norm(P, axis=0), 1.0)
        assert np.allclose(P @ P.T, 2.0 * np.eye(d))
        assert np.allclose(leverage(P), 0.5)
    assert code_specific_floor(A) == pytest.approx(code_specific_floor(B), rel=1e-12)


@pytest.mark.parametrize("d", [8, 16, 32])
def test_affine_level_separates_them_completely(d):
    A, B = duplicated_basis_code(d), basis_hadamard_code(d)
    fa, fb = collision_frontier(A, model="exact"), collision_frontier(B, model="exact")

    # [I, I] repeats every column, so no feature is separable even at s = 1.
    assert fa["rho_min"] == pytest.approx(1.0, abs=1e-8)
    assert fa["s_aff_robust"] == 0
    assert robust_affine_margin(A, 0, 1)["status"] == "infeasible"

    # [I, H] has coherence 1/sqrt(d), so its frontier grows like sqrt(d).
    assert coherence(B) == pytest.approx(1.0 / np.sqrt(d), rel=1e-9)
    assert fb["rho_min"] >= np.sqrt(d) - 1e-6
    assert fb["s_aff_robust"] >= max(1, int(np.floor((np.sqrt(d) + 1) / 2)) - 1)
    assert fb["s_aff_robust"] > fa["s_aff_robust"]


def test_hadamard_frontier_grows_with_width():
    fronts = [collision_frontier(basis_hadamard_code(d), model="exact")["s_aff_robust"]
              for d in (8, 16, 32, 64)]
    assert fronts == sorted(fronts), fronts
    assert fronts[-1] > fronts[0]


def test_basis_hadamard_code_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        basis_hadamard_code(12)


# =====================================================================================
# 4. Collision distance, the margin identity, and noise robustness
# =====================================================================================

def _hull_distance_direct(Phi, i, s):
    """dist(C_i^+, C_i^-) by optimising over the hypersimplices directly, in (u, v)."""
    d, F = Phi.shape
    P = np.delete(Phi, i, axis=1)
    phi = Phi[:, i]
    n = F - 1

    def obj(x):
        r = phi + P @ x[:n] - P @ x[n:]
        return float(r @ r)

    def jac(x):
        r = phi + P @ x[:n] - P @ x[n:]
        g = np.zeros(2 * n)
        g[:n] = 2.0 * (P.T @ r)
        g[n:] = -2.0 * (P.T @ r)
        return g

    x0 = np.concatenate([np.full(n, (s - 1) / n), np.full(n, s / n)])
    cons = [{"type": "eq", "fun": lambda x: x[:n].sum() - (s - 1)},
            {"type": "eq", "fun": lambda x: x[n:].sum() - s}]
    res = minimize(obj, x0, jac=jac, method="SLSQP", constraints=cons,
                   bounds=[(0.0, 1.0)] * (2 * n), options={"maxiter": 3000, "ftol": 1e-14})
    return float(np.sqrt(max(res.fun, 0.0)))


@pytest.mark.parametrize("d,F", [(3, 8), (4, 9), (4, 10)])
def test_collision_distance_matches_the_direct_hull_optimisation(d, F):
    """Two independent routes to the same distance: via Z_s, and over (u, v) in the hulls."""
    for seed in range(2):
        Phi = _code(d, F, 616 + seed + F)
        for i in range(min(F, 4)):
            for s in range(1, F // 2 + 1):
                got = collision_distance(Phi, i, s)["delta"]
                ref = _hull_distance_direct(Phi, i, s)
                assert got == pytest.approx(ref, abs=2e-5), (d, F, i, s, got, ref)


@pytest.mark.parametrize("d,F", [(3, 8), (4, 10)])
def test_margin_is_half_the_collision_distance(d, F):
    """gamma_i(s) = delta_i(s)/2, by support-function/distance duality."""
    checked = 0
    for seed in range(3):
        Phi = _code(d, F, 2727 + seed + F)
        for i in range(min(F, 4)):
            for s in range(1, F // 2 + 1):
                rec = robust_affine_margin(Phi, i, s)
                dist = collision_distance(Phi, i, s)["delta"]
                if rec["status"] == "separable":
                    assert rec["margin"] == pytest.approx(dist / 2.0, rel=5e-4), (i, s)
                    checked += 1
                else:
                    assert dist == pytest.approx(0.0, abs=1e-5), (i, s, dist)
    assert checked > 0


@pytest.mark.parametrize("d,F", [(4, 10)])
def test_collision_distance_vanishes_exactly_when_not_separable(d, F):
    Phi = _code(d, F, 31)
    for i in range(F):
        rho = collision_radius(Phi, i)["rho"]
        for s in range(1, F // 2 + 1):
            zero = collision_distance(Phi, i, s)["delta"] < 1e-5
            assert zero == (not separable_from_rho(rho, s, F)), (i, s)


def test_margin_certifies_noise_tolerance():
    """Any perturbation of norm below the margin must leave every decision correct."""
    import itertools
    d, F, s = 4, 9, 2
    Phi = _code(d, F, 8080)
    rng = np.random.default_rng(0)
    theta = 0.5
    tested = 0
    for i in range(F):
        rec = robust_affine_margin(Phi, i, s, theta=theta)
        if rec["status"] != "separable":
            continue
        w = np.array(rec["w"]) / np.linalg.norm(rec["w"])       # unit norm, as the bound assumes
        gap = 2.0 * rec["margin"]
        a = w @ Phi
        # Re-centre the bias for the normalised w.
        lows = np.sort(np.delete(a, i))
        c = theta - (a[i] + lows[:s - 1].sum() + lows[-s:].sum()) / 2.0
        eps = 0.49 * gap                                        # strictly inside the tolerance
        for S in itertools.combinations(range(F), s):
            x = Phi[:, list(S)].sum(axis=1)
            for _ in range(5):
                eta = rng.standard_normal(d)
                eta *= eps / np.linalg.norm(eta)
                assert ((w @ (x + eta) + c) >= theta) == (i in S), (i, S)
                tested += 1
    assert tested > 0


# =====================================================================================
# 5. The strict-integer frontier formula
# =====================================================================================

@pytest.mark.parametrize("rho,F,expected", [
    (1.0, 8, 0),        # exact tie: 2*1 - 1 = 1 is not < 1
    (3.0, 8, 1),        # exact tie at s = 2
    (5.0, 12, 2),
    (3.5, 8, 2),
    (0.5, 8, 0),
    (float("inf"), 8, 4),
    (float("inf"), 9, 4),
])
def test_s_max_formula_including_exact_ties(rho, F, expected):
    assert s_max_from_rho(rho, F) == expected


@pytest.mark.parametrize("F", [8, 9, 12, 13])
def test_s_max_agrees_with_the_rule_on_the_monotone_regime(F):
    """s_max must be the largest s <= floor(F/2) with the rule satisfied at every k <= s."""
    for rho in [0.5, 1.0, 1.7, 3.0, 4.2, 7.0, 11.0]:
        s_max = s_max_from_rho(rho, F)
        for s in range(1, F // 2 + 1):
            assert separable_from_rho(rho, s, F) == (s <= s_max), (rho, F, s, s_max)


def test_monotonicity_is_restricted_to_the_lower_half():
    """Past F/2 the threshold decreases again, so separability can reappear. Documented, not a bug."""
    F, rho = 12, 3.0
    lower = [separable_from_rho(rho, s, F) for s in range(1, F // 2 + 1)]
    assert lower == sorted(lower, reverse=True)
    # At s = F - 1 the threshold is 2*1 - 1 = 1 < rho, so it is separable again.
    assert separable_from_rho(rho, F - 1, F)
    assert not separable_from_rho(rho, F // 2, F)
