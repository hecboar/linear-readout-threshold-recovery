"""Exhaustive small-scale validation of G1, G2 and G4.

Each claim is checked against the definition it is supposed to replace, by brute force wherever
brute force is affordable: the analog optimum against a general-purpose constrained minimiser,
the worst-case support formulas against enumerating every support, and the affine frontier
against enumerating every constraint. Certificates are re-verified from the code alone.

These are the tests that have to pass before any number produced by these modules is quoted.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from lrtr.affine_frontier import (
    cut_vector,
    robust_affine_frontier,
    robust_affine_margin,
    verify_certificate,
    worst_case_scores,
)
from lrtr.analog_optimum import (
    analog_attainment,
    code_specific_floor,
    leverage,
    leverage_excess,
    optimal_analog_readout,
    optimal_crosstalk_per_feature,
)
from lrtr.codes import harmonic_tight_frame, random_unit_code, welch_floor
from lrtr.distributional import (
    bernoulli_energy,
    bernoulli_second_moment,
    distribution_weighted_error,
    second_moment,
    uniform_support_second_moment,
)
from lrtr.interface import crosstalk_mean_sq, unit_diagonal


def _code(d, F, seed):
    return random_unit_code(d, F, np.random.default_rng(seed))


# =====================================================================================
# G1 -- code-specific analog optimum
# =====================================================================================

@pytest.mark.parametrize("d,F", [(4, 12), (6, 25), (8, 40)])
def test_optimum_matches_a_general_purpose_minimiser(d, F):
    """The closed form must beat nothing and match everything a solver finds."""
    Phi = _code(d, F, 20260808 + d)
    claimed = optimal_crosstalk_per_feature(Phi)
    for i in (0, F // 2, F - 1):
        A = np.delete(Phi, i, axis=1)
        res = minimize(lambda w: float(np.sum((A.T @ w) ** 2)), np.zeros(d),
                       jac=lambda w: 2.0 * (A @ (A.T @ w)),
                       constraints=[{"type": "eq", "fun": lambda w: w @ Phi[:, i] - 1.0,
                                     "jac": lambda w: Phi[:, i]}],
                       method="SLSQP", options={"maxiter": 5000, "ftol": 1e-16})
        assert res.fun == pytest.approx(claimed[i], rel=1e-6)


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 96)])
def test_optimal_readout_is_the_calibrated_pseudoinverse(d, F):
    Phi = _code(d, F, 7 + d)
    Gp = np.linalg.pinv(Phi)
    calibrated = Gp / np.diag(Gp @ Phi)[:, None]
    assert optimal_analog_readout(Phi) == pytest.approx(calibrated, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 96)])
def test_leverage_matches_the_projector_diagonal(d, F):
    Phi = _code(d, F, 99 + d)
    assert leverage(Phi) == pytest.approx(np.diag(np.linalg.pinv(Phi) @ Phi), rel=1e-9)
    assert leverage(Phi).sum() == pytest.approx(d, rel=1e-9)


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 96)])
def test_no_readout_beats_the_code_specific_floor(d, F):
    """The point of the floor: sample readouts and check none goes below it."""
    Phi = _code(d, F, 313 + d)
    w_code = code_specific_floor(Phi)
    assert crosstalk_mean_sq(optimal_analog_readout(Phi), Phi) == pytest.approx(w_code, rel=1e-8)
    rng = np.random.default_rng(0)
    for _ in range(15):
        G = np.linalg.pinv(Phi) + 0.2 * rng.standard_normal((F, d))
        assert crosstalk_mean_sq(G, Phi) >= w_code * (1 - 1e-9)
    assert crosstalk_mean_sq(np.ascontiguousarray(Phi.T), Phi) >= w_code * (1 - 1e-9)


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 96)])
def test_code_floor_sits_above_the_welch_floor_and_excess_identity_holds(d, F):
    Phi = _code(d, F, 555 + d)
    assert code_specific_floor(Phi) >= welch_floor(F, d) * (1 - 1e-12)
    ex = leverage_excess(Phi)
    assert ex["excess"] == pytest.approx(ex["excess_identity"], rel=1e-8)


def test_tight_frame_meets_both_floors():
    """Uniform leverage collapses the two floors onto each other."""
    Phi = harmonic_tight_frame(8, 40)
    ex = leverage_excess(Phi)
    assert ex["excess"] == pytest.approx(0.0, abs=1e-8)
    assert code_specific_floor(Phi) == pytest.approx(welch_floor(40, 8), rel=1e-8)
    out = analog_attainment(Phi)
    assert out["ratio_vs_code"] == pytest.approx(1.0, rel=1e-8)
    assert out["ratio_vs_welch"] == pytest.approx(1.0, rel=1e-8)


def test_attainment_separates_the_two_floors_on_a_heterogeneous_code():
    """A code with uneven leverage is at its own floor but above the Welch floor."""
    rng = np.random.default_rng(3)
    Phi = harmonic_tight_frame(6, 30) + 0.35 * rng.standard_normal((6, 30))
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    out = analog_attainment(Phi)
    assert out["ratio_vs_code"] == pytest.approx(1.0, rel=1e-8)
    assert out["ratio_vs_welch"] > 1.0 + 1e-6
    assert out["code_floor_over_welch"] > 1.0


# =====================================================================================
# G2 -- robust affine frontier
# =====================================================================================

def _brute_worst(a, i, s):
    F = a.size
    wa, wi = np.inf, -np.inf
    for S in itertools.combinations(range(F), s):
        tot = a[list(S)].sum()
        if i in S:
            wa = min(wa, tot)
        else:
            wi = max(wi, tot)
    return wa, wi


@pytest.mark.parametrize("seed", range(30))
def test_worst_case_scores_match_exhaustive_enumeration(seed):
    rng = np.random.default_rng(seed)
    F = int(rng.integers(5, 10))
    d = 3
    s = int(rng.integers(1, min(4, F - 1) + 1))
    Phi = _code(d, F, 1000 + seed)
    w = rng.standard_normal(d)
    a = w @ Phi
    i = int(rng.integers(0, F))
    got = worst_case_scores(a, i, s)
    wa, wi = _brute_worst(a, i, s)
    assert got["worst_active"] == pytest.approx(wa)
    assert got["worst_inactive"] == pytest.approx(wi)
    assert set(got["active_support"].tolist()) <= set(range(F))
    assert i in got["active_support"] and i not in got["inactive_support"]
    assert a[got["active_support"]].sum() == pytest.approx(wa)
    assert a[got["inactive_support"]].sum() == pytest.approx(wi)


def _brute_separable(Phi, i, s):
    """Decide separability by building every constraint explicitly. Only for tiny F."""
    d, F = Phi.shape
    others = [j for j in range(F) if j != i]
    V = []
    for A in itertools.combinations(others, s - 1):
        for B in itertools.combinations(others, s):
            V.append(cut_vector(Phi, i, np.array(A, dtype=int), np.array(B, dtype=int)))
    V = np.stack(V, axis=1)
    from scipy.optimize import linprog
    K = V.shape[1]
    c = np.zeros(d + 1); c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([-V.T, np.ones((K, 1))]), b_ub=np.zeros(K),
                  bounds=[(-1.0, 1.0)] * d + [(None, 1.0)], method="highs")
    return bool(res.success and res.x[-1] > 1e-9)


@pytest.mark.parametrize("F,s", [(8, 1), (8, 2), (9, 2), (10, 2), (10, 3), (12, 2)])
def test_frontier_agrees_with_exhaustive_constraint_enumeration(F, s):
    """Constraint generation must reach the same verdict as enumerating all (A, B) pairs."""
    for seed in range(3):
        Phi = _code(4, F, 4242 + 17 * seed + F)
        for i in range(min(F, 5)):
            rec = robust_affine_margin(Phi, i, s)
            assert rec["status"] in ("separable", "infeasible"), rec["status"]
            assert (rec["status"] == "separable") == _brute_separable(Phi, i, s), \
                f"F={F} s={s} seed={seed} i={i}: {rec['status']}"


@pytest.mark.parametrize("F,s", [(9, 2), (10, 3)])
def test_separable_solution_really_separates_every_support(F, s):
    """The returned (w, bias) must classify every support of size s correctly, by enumeration."""
    theta = 0.5
    for seed in range(4):
        Phi = _code(4, F, 909 + seed)
        for i in range(F):
            rec = robust_affine_margin(Phi, i, s, theta=theta)
            if rec["status"] != "separable":
                continue
            w = np.array(rec["w"])
            c = rec["bias"]
            a = w @ Phi
            for S in itertools.combinations(range(F), s):
                score = c + a[list(S)].sum()
                assert (score >= theta) == (i in S), (rec["margin"], S, i)


@pytest.mark.parametrize("F,s", [(8, 2), (10, 3), (12, 3)])
def test_infeasibility_certificates_verify_independently(F, s):
    """A certificate must check out when rebuilt from the code, with no solver involved."""
    found = 0
    for seed in range(6):
        Phi = _code(4, F, 77 + seed)
        for i in range(F):
            rec = robust_affine_margin(Phi, i, s)
            if rec["status"] != "infeasible":
                continue
            found += 1
            chk = verify_certificate(Phi, i, s, rec["certificate"])
            assert chk["valid"], chk
            assert chk["nonnegative"] and chk["sums_to_one"]
            assert chk["relative_residual"] < 1e-6
            # And the certificate must be incompatible with separability, by brute force.
            assert not _brute_separable(Phi, i, s)
    assert found > 0, "no infeasible case arose; the test would be vacuous"


def test_duplicated_feature_is_never_separable():
    """Two identical columns are indistinguishable: the certificate must exist for s >= 2."""
    d = 5
    Phi = _code(d, 9, 11)
    Phi[:, 1] = Phi[:, 0]
    rec = robust_affine_margin(Phi, 0, 2)
    assert rec["status"] == "infeasible"
    assert verify_certificate(Phi, 0, 2, rec["certificate"])["valid"]


def test_margin_is_scale_equivariant():
    """Scaling the code scales the margin inversely; the verdict cannot change."""
    Phi = _code(4, 10, 5)
    a = robust_affine_margin(Phi, 0, 1)
    b = robust_affine_margin(3.0 * Phi, 0, 1)
    assert a["status"] == b["status"] == "separable"
    assert b["margin"] == pytest.approx(3.0 * a["margin"], rel=1e-5)


def test_frontier_is_monotone_and_reports_its_own_scope():
    Phi = _code(5, 14, 31)
    out = robust_affine_frontier(Phi, [1, 2, 3])
    seps = [r["all_separable"] for r in out["rows"]]
    assert seps == sorted(seps, reverse=True), "separability must not come back once lost"
    assert out["s_aff_robust"] == sum(1 for x in seps if x) * (1 if seps and seps[0] else 0) \
        or out["s_aff_robust"] in out["grid"] + [0]
    sub = robust_affine_frontier(Phi, [1, 2], feature_subset=[0, 1])
    assert sub["is_upper_bound"] and sub["features_tested"] == 2


# =====================================================================================
# G4 -- distribution-aware analog error
# =====================================================================================

@pytest.mark.parametrize("d,F", [(4, 12), (8, 40)])
def test_trace_identity_matches_monte_carlo(d, F):
    Phi = _code(d, F, 60 + d)
    A = unit_diagonal(np.linalg.pinv(Phi) @ Phi) - np.eye(F)
    rng = np.random.default_rng(2)
    B = (rng.random((F, 40000)) < 0.1) * (rng.random((F, 40000)) * 2 - 1)
    empirical = float(np.mean(np.sum((A @ B) ** 2, axis=0))) / F
    assert distribution_weighted_error(A, second_moment(B)) == pytest.approx(empirical, rel=1e-9)


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40)])
def test_bernoulli_decomposition_matches_the_trace_form(d, F):
    Phi = _code(d, F, 70 + d)
    A = unit_diagonal(np.linalg.pinv(Phi) @ Phi) - np.eye(F)
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.4, size=F)          # heterogeneous firing rates
    assert bernoulli_energy(A, p) == pytest.approx(
        distribution_weighted_error(A, bernoulli_second_moment(p)), rel=1e-9)


def test_uniform_support_moment_reproduces_the_boolean_closed_form():
    """The published Boolean model must fall out of the general identity as a special case."""
    d, F, s = 6, 24, 3
    Phi = _code(d, F, 80)
    A = unit_diagonal(np.linalg.pinv(Phi) @ Phi) - np.eye(F)
    m = uniform_support_second_moment(F, s)
    C = (m["diagonal"] - m["off_diagonal"]) * np.eye(F) + m["off_diagonal"] * np.ones((F, F))
    closed = ((m["diagonal"] - m["off_diagonal"]) * float(np.sum(A * A))
              + m["off_diagonal"] * float(A.sum(axis=1) @ A.sum(axis=1))) / F
    assert distribution_weighted_error(A, C) == pytest.approx(closed, rel=1e-9)


# =====================================================================================
# G2 -- the compact robust-counterpart route must reach the same feasible set
# =====================================================================================

@pytest.mark.parametrize("F,s", [(8, 1), (8, 2), (9, 2), (10, 2), (10, 3), (12, 2), (12, 3)])
def test_compact_reformulation_agrees_with_constraint_generation(F, s):
    """Bertsimas-Sim lifting vs. our separation oracle: verdicts must match exactly.

    This is the check that the oracle is not missing constraints. The compact route derives its
    constraints from LP duality rather than from a sort, so the two share no code path.
    """
    from lrtr.affine_frontier import robust_affine_margin_compact
    for seed in range(3):
        Phi = _code(4, F, 5150 + 13 * seed + F)
        for i in range(min(F, 4)):
            a = robust_affine_margin(Phi, i, s)
            b = robust_affine_margin_compact(Phi, i, s)
            assert b["status"] != "solver_failure", b
            assert a["status"] == b["status"], f"F={F} s={s} i={i}: {a['status']} vs {b['status']}"
            if a["status"] == "separable":
                # The compact route returns a feasible, not minimum-norm, w: it lower-bounds.
                assert b["margin"] <= a["margin"] * (1 + 1e-6)


@pytest.mark.parametrize("F,s", [(9, 2), (11, 3)])
def test_compact_solution_also_separates_every_support(F, s):
    from lrtr.affine_frontier import robust_affine_margin_compact
    theta = 0.5
    for seed in range(3):
        Phi = _code(4, F, 606 + seed)
        for i in range(F):
            rec = robust_affine_margin_compact(Phi, i, s, theta=theta)
            if rec["status"] != "separable":
                continue
            w, c = np.array(rec["w"]), rec["bias"]
            a = w @ Phi
            for S in itertools.combinations(range(F), s):
                assert ((c + a[list(S)].sum()) >= theta) == (i in S)


# =====================================================================================
# G2 -- the collision radius: one LP per feature decides every sparsity
# =====================================================================================

def _brute_hulls_disjoint(Phi, i, s):
    """Ground truth: strict separation of the two hulls, over their full vertex sets."""
    from scipy.optimize import linprog
    d, F = Phi.shape
    others = [j for j in range(F) if j != i]
    Vp = [Phi[:, i] + (Phi[:, list(A)].sum(1) if s > 1 else 0)
          for A in itertools.combinations(others, s - 1)]
    Vm = [Phi[:, list(B)].sum(1) for B in itertools.combinations(others, s)]
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


@pytest.mark.parametrize("d,F", [(3, 7), (4, 8), (4, 9), (5, 9)])
def test_collision_radius_decides_every_sparsity(d, F):
    """rho_i > 2 min(s, F-s) - 1  <=>  separable, for every feature and every sparsity."""
    from lrtr.affine_frontier import collision_radius, separable_from_rho
    for seed in range(4):
        Phi = _code(d, F, 31337 + 7 * seed + F)
        for i in range(F):
            rho = collision_radius(Phi, i)["rho"]
            for s in range(1, F):
                assert separable_from_rho(rho, s, F) == _brute_hulls_disjoint(Phi, i, s), \
                    f"d={d} F={F} seed={seed} i={i} s={s} rho={rho}"


def test_collision_frontier_matches_the_quadratic_route():
    """The LP frontier and the per-(feature, sparsity) QP frontier must agree."""
    from lrtr.affine_frontier import collision_frontier
    for d, F in [(3, 7), (4, 8)]:
        Phi = _code(d, F, 808 + F)
        fast = collision_frontier(Phi)
        slow = robust_affine_frontier(Phi, list(range(1, (F - 1) // 2 + 2)))
        assert fast["s_aff_robust"] == slow["s_aff_robust"], (fast["s_aff_robust"],
                                                              slow["s_aff_robust"])


def test_duplicated_column_has_collision_radius_one():
    """Two identical columns collide at s=1: rho must be exactly 1, so s_max = 0."""
    from lrtr.affine_frontier import collision_radius
    Phi = _code(4, 8, 11)
    Phi[:, 1] = Phi[:, 0]
    rec = collision_radius(Phi, 0)
    assert rec["rho"] == pytest.approx(1.0, abs=1e-8)
    assert rec["s_max"] == 0


def test_margin_equals_the_minimum_norm_over_all_constraints():
    """The reported margin must equal the definition: min-norm w over every (A, B) constraint."""
    from lrtr.affine_frontier import _min_norm
    for F, s in [(8, 1), (9, 2), (10, 2)]:
        for seed in range(2):
            Phi = _code(4, F, 2024 + 5 * seed + F)
            for i in range(min(F, 3)):
                rec = robust_affine_margin(Phi, i, s)
                if rec["status"] != "separable":
                    continue
                others = [j for j in range(F) if j != i]
                V = np.stack([cut_vector(Phi, i, np.array(A, dtype=int), np.array(B, dtype=int))
                              for A in itertools.combinations(others, s - 1)
                              for B in itertools.combinations(others, s)], axis=1)
                w_ref = _min_norm(V)["w"]
                ref = 1.0 / (2.0 * float(np.linalg.norm(w_ref)))
                assert rec["margin"] == pytest.approx(ref, rel=2e-4), (F, s, i)


# =====================================================================================
# G1 -- the geometry / readout factorisation used by the reanalysis
# =====================================================================================

@pytest.mark.parametrize("d,F", [(8, 40), (12, 60), (50, 100), (16, 96)])
def test_attainment_ratio_factors_into_geometry_and_readout(d, F):
    """W(G,Phi)/W_global = [W_*(Phi)/W_global] x [W(G,Phi)/W_*(Phi)], exactly.

    The published number is the left-hand side, which conflates a property of the code with a
    property of the decoder. The split is arithmetic, but it has to be exact for the reanalysis
    to be reporting the same quantity the manuscript does.
    """
    Phi = _code(d, F, d)
    w_code, w_glob = code_specific_floor(Phi), welch_floor(F, d)
    rng = np.random.default_rng(0)
    for G in (optimal_analog_readout(Phi), np.ascontiguousarray(Phi.T),
              np.linalg.pinv(Phi) + 0.1 * rng.standard_normal((F, d))):
        measured = crosstalk_mean_sq(np.ascontiguousarray(G), Phi)
        assert (w_code / w_glob) * (measured / w_code) == pytest.approx(measured / w_glob,
                                                                       rel=1e-12)


@pytest.mark.parametrize("d,F", [(8, 40), (50, 100)])
def test_readout_term_is_exactly_one_for_the_pseudoinverse(d, F):
    """Under the calibrated pseudoinverse the whole ratio is geometry: R_readout = 1.

    This is why "the trained code sits within a few parts in a thousand of the floor" under the
    pseudoinverse is a statement about the code's leverage profile, not about its decoder.
    """
    Phi = _code(d, F, 5 + d)
    measured = crosstalk_mean_sq(optimal_analog_readout(Phi), Phi)
    assert measured / code_specific_floor(Phi) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("d,F", [(8, 40), (16, 96)])
def test_readout_term_is_at_least_one_for_any_readout(d, F):
    Phi = _code(d, F, 700 + d)
    w_code = code_specific_floor(Phi)
    rng = np.random.default_rng(1)
    for _ in range(10):
        G = np.linalg.pinv(Phi) + 0.3 * rng.standard_normal((F, d))
        assert crosstalk_mean_sq(G, Phi) / w_code >= 1.0 - 1e-9
