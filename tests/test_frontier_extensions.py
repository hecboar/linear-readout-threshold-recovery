"""The at-most-s frontier, its amplitude version, and the leverage bound.

Each is checked against brute-force strict separation of the relevant convex hulls over their
full vertex sets — the definition, not another formulation. See `docs/theory_extensions.md`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from lrtr.affine_frontier import (
    collision_radius,
    collision_radius_atmost,
    collision_radius_boxless,
    leverage_bound_on_rho,
    local_coherence,
    separable_atmost,
)
from lrtr.analog_optimum import leverage
from lrtr.codes import basis_hadamard_code, duplicated_basis_code, random_unit_code


def _code(d, F, seed):
    return random_unit_code(d, F, np.random.default_rng(seed))


def _separable_by_hulls(Vp, Vm, d):
    """Strict separation of two finite point sets, by LP. The definition."""
    nv = d + 2
    rows = []
    for p in Vp:
        r = np.zeros(nv); r[:d] = -p; r[d] = 1.0; r[d + 1] = 1.0; rows.append(r)
    for q in Vm:
        r = np.zeros(nv); r[:d] = q; r[d] = -1.0; r[d + 1] = 1.0; rows.append(r)
    c = np.zeros(nv); c[d + 1] = -1.0
    res = linprog(c, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  bounds=[(-1, 1)] * d + [(None, None)] + [(None, 1.0)], method="highs")
    return bool(res.success and res.x[d + 1] > 1e-9)


def _brute_atmost(Phi, i, s, alpha=1.0):
    d, F = Phi.shape
    others = [j for j in range(F) if j != i]
    amps = (1.0,) if alpha == 1.0 else (alpha, 1.0)
    Vp = [t * Phi[:, i] + (Phi[:, list(A)].sum(1) if A else 0.0)
          for t in amps for k in range(0, s) for A in itertools.combinations(others, k)]
    Vm = [Phi[:, list(B)].sum(1) if B else np.zeros(d)
          for k in range(0, s + 1) for B in itertools.combinations(others, k)]
    return _separable_by_hulls(Vp, Vm, d)


# =====================================================================================
# E1 -- at most s
# =====================================================================================

@pytest.mark.parametrize("d,F", [(3, 7), (4, 8), (4, 9)])
def test_atmost_frontier_matches_brute_force(d, F):
    for seed in range(3):
        Phi = _code(d, F, 1000 + 11 * seed + F)
        for i in range(F):
            rho = collision_radius_atmost(Phi, i)["rho_hat"]
            for s in range(1, F):
                assert separable_atmost(rho, s) == _brute_atmost(Phi, i, s), \
                    f"d={d} F={F} seed={seed} i={i} s={s} rho_hat={rho}"


@pytest.mark.parametrize("d,F", [(4, 9), (5, 11)])
def test_atmost_frontier_is_monotone_by_construction(d, F):
    """No `min(s, F-s)`: separability, once lost, never returns."""
    for seed in range(3):
        Phi = _code(d, F, 55 + seed)
        for i in range(F):
            rho = collision_radius_atmost(Phi, i)["rho_hat"]
            flags = [separable_atmost(rho, s) for s in range(1, F)]
            assert flags == sorted(flags, reverse=True), (i, rho, flags)


def test_atmost_s_max_is_the_last_separable_sparsity():
    Phi = _code(4, 10, 7)
    for i in range(10):
        rec = collision_radius_atmost(Phi, i)
        s_max, rho = rec["s_max"], rec["rho_hat"]
        if s_max >= 1:
            assert separable_atmost(rho, s_max)
        assert not separable_atmost(rho, s_max + 1)


def test_duplicated_column_is_never_separable_under_at_most():
    """A duplicate makes even s=1 impossible, since the inactive hull contains that column."""
    Phi = _code(4, 8, 3)
    Phi[:, 1] = Phi[:, 0]
    assert collision_radius_atmost(Phi, 0)["s_max"] == 0


# =====================================================================================
# E2 -- amplitudes in [alpha, 1]
# =====================================================================================

@pytest.mark.parametrize("alpha", [1.0, 0.7, 0.4])
def test_amplitude_frontier_matches_brute_force(alpha):
    for d, F in [(3, 7), (4, 8)]:
        for seed in range(2):
            Phi = _code(d, F, 500 + seed + F)
            for i in range(min(F, 4)):
                rho = collision_radius_atmost(Phi, i, alpha=alpha)["rho_hat"]
                for s in range(1, F):
                    assert separable_atmost(rho, s) == _brute_atmost(Phi, i, s, alpha), \
                        f"alpha={alpha} d={d} F={F} i={i} s={s} rho={rho}"


def test_smaller_amplitude_never_helps():
    """The frontier is monotone in alpha: a weaker activation cannot be easier to detect."""
    Phi = _code(5, 12, 21)
    for i in range(12):
        rhos = [collision_radius_atmost(Phi, i, alpha=a)["rho_hat"]
                for a in (1.0, 0.8, 0.6, 0.4)]
        assert rhos == sorted(rhos, reverse=True), (i, rhos)


def test_alpha_one_reduces_to_the_boolean_case():
    Phi = _code(4, 10, 9)
    for i in range(10):
        assert (collision_radius_atmost(Phi, i, alpha=1.0)["rho_hat"]
                == pytest.approx(collision_radius_atmost(Phi, i)["rho_hat"], rel=1e-9))


def test_alpha_out_of_range_is_rejected():
    Phi = _code(4, 10, 1)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            collision_radius_atmost(Phi, 0, alpha=bad)


# =====================================================================================
# E3 -- the leverage bound, and its extremal code
# =====================================================================================

@pytest.mark.parametrize("d,F", [(8, 40), (16, 96), (6, 20)])
def test_leverage_bound_is_never_violated(d, F):
    Phi = _code(d, F, 300 + d)
    bound = leverage_bound_on_rho(Phi)
    for i in range(F):
        rho = collision_radius(Phi, i)["rho"]
        if np.isfinite(rho):
            assert rho >= bound[i] - 1e-7, (i, rho, bound[i])


@pytest.mark.parametrize("d", [8, 16])
def test_duplicated_basis_attains_the_leverage_bound(d):
    """`[I, I]` saturates it: uniform leverage 1/2, bound 1, and rho exactly 1."""
    Phi = duplicated_basis_code(d)
    assert leverage(Phi) == pytest.approx(np.full(2 * d, 0.5), rel=1e-12)
    bound = leverage_bound_on_rho(Phi)
    assert bound == pytest.approx(np.ones(2 * d), rel=1e-9)
    for i in range(2 * d):
        assert collision_radius(Phi, i)["rho"] == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("d", [8, 16])
def test_bound_is_loose_on_a_good_code(d):
    """`[I, H]` clears it by a wide margin, so tightness at `[I,I]` is not an artefact."""
    Phi = basis_hadamard_code(d)
    bound = leverage_bound_on_rho(Phi)
    rho = np.array([collision_radius(Phi, i)["rho"] for i in range(2 * d)])
    assert np.min(rho / bound) > 3.0


def test_bound_is_vacuous_for_most_features_when_strongly_overcomplete():
    """It bites only where h_i > 1/2, and sum h_i = d caps how many features that can be."""
    d, F = 8, 96
    Phi = _code(d, F, 4)
    informative = int((leverage_bound_on_rho(Phi) > 1.0).sum())
    assert informative <= 2 * d
    assert informative == 0                      # at F = 12d nothing clears h_i > 1/2


# =====================================================================================
# E4 -- the two negatives, kept honest by tests
# =====================================================================================

@pytest.mark.parametrize("d,F", [(4, 12), (6, 20), (8, 40)])
def test_box_rarely_binds(d, F):
    """The frontier is, for most features, the boxless ell-1 quantity. Recorded, not hidden."""
    Phi = _code(d, F, 77 + d)
    binds = 0
    for i in range(F):
        a = collision_radius_boxless(Phi, i)
        b = collision_radius(Phi, i)["rho"]
        assert b >= a - 1e-7                     # the box can only raise the minimum
        if np.isfinite(b) and b > a * (1 + 1e-6):
            binds += 1
    assert binds / F < 0.25, f"box bound on {binds}/{F}"


@pytest.mark.parametrize("d", [8, 16, 32])
def test_coherence_bound_is_approached_but_not_attained(d):
    """Even on the most symmetric family the bound is not tight, so rho beats coherence."""
    Phi = basis_hadamard_code(d)
    mu = local_coherence(Phi)
    ratio = np.array([collision_radius(Phi, i)["rho"] for i in range(2 * d)]) * mu
    assert ratio.min() > 1.0                     # never attained
    assert ratio.max() > 1.5                     # and loose for most features


# =====================================================================================
# E3b -- the leverage UPPER bound: the direction that makes the hierarchy an implication
# =====================================================================================

@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 96), (50, 100)])
def test_min_l2_representation_is_h_over_one_minus_h(d, F):
    """Sherman-Morrison, since Phi_{-i} Phi_{-i}^T = Sigma - phi_i phi_i^T."""
    from lrtr.affine_frontier import min_l2_representation
    Phi = _code(d, F, d)
    claimed = min_l2_representation(Phi)
    for i in (0, F // 3, F - 1):
        A = np.delete(Phi, i, axis=1)
        z = np.linalg.lstsq(A, Phi[:, i], rcond=None)[0]
        assert float(z @ z) == pytest.approx(claimed[i], rel=1e-9)


@pytest.mark.parametrize("d,F", [(4, 12), (6, 20), (8, 40)])
def test_leverage_upper_bounds_the_frontier(d, F):
    from lrtr.affine_frontier import leverage_upper_bound_on_kappa
    Phi = _code(d, F, 7 + d)
    ub = leverage_upper_bound_on_kappa(Phi)
    for i in range(F):
        k = collision_radius_atmost(Phi, i)["rho_hat"]
        if np.isfinite(k):
            assert k <= ub[i] + 1e-6, (i, k, ub[i])


@pytest.mark.parametrize("d,F", [(4, 12), (6, 20), (8, 40)])
def test_failure_threshold_never_admits_a_separable_feature(d, F):
    """h_i below the threshold must imply not separable at that sparsity. No exceptions allowed."""
    from lrtr.affine_frontier import affine_failure_threshold
    Phi = _code(d, F, 31 + d)
    h = leverage(Phi)
    checked = 0
    for i in range(F):
        k = collision_radius_atmost(Phi, i)["rho_hat"]
        for s in range(2, min(F, 10)):
            if h[i] <= affine_failure_threshold(F, s):
                checked += 1
                assert not separable_atmost(k, s), (i, s, h[i], k)
    assert checked > 0, "the test would be vacuous"


def test_box_is_free_where_the_bound_bites():
    """h_i <= 1/2 makes the ell-2 minimiser satisfy the box, so the certificate is admissible."""
    from lrtr.affine_frontier import min_l2_representation
    Phi = _code(8, 40, 5)
    m = min_l2_representation(Phi)
    h = leverage(Phi)
    assert np.all(h <= 0.5)                      # strongly overcomplete: every leverage is small
    assert np.all(m <= 1.0)                      # hence ||z||_2 <= 1, so ||z||_inf <= 1


def test_hierarchy_is_strictly_one_way():
    """Bad analog geometry forces a bad frontier; good geometry does not guarantee a good one."""
    from lrtr.affine_frontier import affine_failure_threshold
    # Forward: a code with a near-dead feature must collapse.
    Phi = _code(6, 40, 12).copy()
    Phi[:, 0] = 0.02 * Phi[:, 0] + 0.98 * Phi[:, 1]
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    h0 = leverage(Phi)[0]
    if h0 <= affine_failure_threshold(40, 2):
        assert collision_radius_atmost(Phi, 0)["s_max"] <= 1
    # Converse fails: [I, I] has perfectly uniform leverage and the worst possible frontier.
    A = duplicated_basis_code(8)
    assert leverage(A) == pytest.approx(np.full(16, 0.5), rel=1e-12)
    assert collision_radius_atmost(A, 0)["s_max"] == 0
