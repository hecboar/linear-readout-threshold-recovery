"""The d x d sufficient-statistic paths must agree with the dense F x F computation.

These are the tests behind the complexity claim in the manuscript: the cheap route is exact,
not an approximation, so the reported ``O(F d^2)`` cost buys nothing at the expense of the
numbers. Everything here compares a reduced computation against the definition it replaces.
"""
from __future__ import annotations

import numpy as np
import pytest

from lrtr.codes import harmonic_tight_frame, random_unit_code, welch_floor
from lrtr.diagnostic import (
    fixed_code_separation_profile,
    interface_energy_moments,
    interface_floor_diagnostic,
    tied_energy_moments,
    tied_linear_energy,
)
from lrtr.interface import (
    crosstalk_mean_sq,
    crosstalk_mean_sq_pinv,
    crosstalk_stats,
    gains,
    leverage_scores,
    unit_diagonal,
)
from lrtr.threshold import recovery_trial_fixed, threshold_decode

SHAPES = [(8, 40), (16, 96), (24, 200)]


def _code_and_readout(d, F, readout, rng):
    Phi = random_unit_code(d, F, rng)
    G = Phi.T if readout == "tied" else np.linalg.pinv(Phi)
    return Phi, np.ascontiguousarray(G)


@pytest.mark.parametrize("d,F", SHAPES)
@pytest.mark.parametrize("readout", ["tied", "pinv"])
def test_crosstalk_mean_sq_matches_dense(d, F, readout):
    rng = np.random.default_rng(20260808 + d)
    Phi, G = _code_and_readout(d, F, readout, rng)
    dense = crosstalk_stats(unit_diagonal(G @ Phi), d).mean_sq_offdiag
    assert crosstalk_mean_sq(G, Phi) == pytest.approx(dense, rel=1e-10)


@pytest.mark.parametrize("d,F", SHAPES)
def test_gains_match_interface_diagonal(d, F):
    rng = np.random.default_rng(7 + d)
    Phi, G = _code_and_readout(d, F, "pinv", rng)
    assert gains(G, Phi) == pytest.approx(np.diag(G @ Phi), rel=1e-10)


@pytest.mark.parametrize("d,F", SHAPES)
def test_mean_sq_statistics_mode_matches_full(d, F):
    rng = np.random.default_rng(99 + d)
    Phi = random_unit_code(d, F, rng)
    full = interface_floor_diagnostic(Phi, readout="tied", statistics="full")
    cheap = interface_floor_diagnostic(Phi, readout="tied", statistics="mean_sq")
    assert cheap["ratio_mean_sq"] == pytest.approx(full["ratio_mean_sq"], rel=1e-10)
    assert cheap["max_abs_offdiag"] is None and full["max_abs_offdiag"] is not None
    assert not cheap["violates_floor"]


@pytest.mark.parametrize("d,F", SHAPES)
@pytest.mark.parametrize("readout", ["tied", "pinv"])
def test_interface_energy_moments_match_dense(d, F, readout):
    rng = np.random.default_rng(313 + d)
    Phi, G = _code_and_readout(d, F, readout, rng)
    A = unit_diagonal(G @ Phi) - np.eye(F)
    mom = interface_energy_moments(G, Phi)
    assert mom.frob_sq_A == pytest.approx(float(np.sum(A * A)), rel=1e-9)
    assert mom.sum_A1_sq == pytest.approx(float(A.sum(axis=1) @ A.sum(axis=1)), rel=1e-8)


@pytest.mark.parametrize("d,F", SHAPES)
def test_general_moments_reduce_to_tied(d, F):
    rng = np.random.default_rng(555 + d)
    Phi = random_unit_code(d, F, rng)
    general = interface_energy_moments(np.ascontiguousarray(Phi.T), Phi)
    tied = tied_energy_moments(Phi)
    assert general.frob_sq_A == pytest.approx(tied.frob_sq_A, rel=1e-9)
    assert general.sum_A1_sq == pytest.approx(tied.sum_A1_sq, rel=1e-8)


@pytest.mark.parametrize("d,F", SHAPES)
def test_pinv_crosstalk_from_leverage(d, F):
    """The pseudoinverse ratio depends on the code only through its leverage scores."""
    rng = np.random.default_rng(4242 + d)
    Phi = random_unit_code(d, F, rng)
    G = np.linalg.pinv(Phi)
    assert crosstalk_mean_sq_pinv(Phi) == pytest.approx(crosstalk_mean_sq(G, Phi), rel=1e-8)
    assert leverage_scores(Phi).sum() == pytest.approx(min(d, F), rel=1e-9)


def test_equal_leverage_attains_the_floor_under_pinv():
    """A tight frame with equal leverage sits exactly on the floor; a perturbed one does not."""
    d, F = 8, 40
    Phi = harmonic_tight_frame(d, F)
    fl = welch_floor(F, d)
    assert crosstalk_mean_sq_pinv(Phi) == pytest.approx(fl, rel=1e-8)

    rng = np.random.default_rng(0)
    Psi = Phi + 0.3 * rng.standard_normal(Phi.shape)
    Psi /= np.linalg.norm(Psi, axis=0, keepdims=True)
    assert crosstalk_mean_sq_pinv(Psi) > fl


# --------------------------------------------------------------------------------------
# ALG-2 on a fixed code
# --------------------------------------------------------------------------------------

def test_recovery_trial_fixed_matches_direct_evaluation():
    d, F, s_max = 12, 60, 4
    rng = np.random.default_rng(11)
    Phi = random_unit_code(d, F, rng)
    G = np.ascontiguousarray(Phi.T)

    trial_rng = np.random.default_rng(5)
    ok = recovery_trial_fixed(Phi, G, s_max, trial_rng, theta=0.5)

    check_rng = np.random.default_rng(5)
    S = check_rng.choice(F, size=s_max, replace=False)
    for s in range(1, s_max + 1):
        b = np.zeros(F, dtype=np.int8)
        b[S[:s]] = 1
        expected = np.array_equal(threshold_decode(G @ Phi[:, S[:s]].sum(axis=1), 0.5), b)
        assert bool(ok[s - 1]) == expected


def test_fixed_code_profile_is_deterministic_and_exact_on_the_analog_side():
    d, F = 16, 96
    rng = np.random.default_rng(2026)
    Phi = random_unit_code(d, F, rng)
    kw = dict(sparsities=[1, 2, 3], trials=40, seed=1, readout="tied")
    a = fixed_code_separation_profile(Phi, **kw)
    b = fixed_code_separation_profile(Phi, **kw)

    assert [r["p_rec"] for r in a["rows"]] == [r["p_rec"] for r in b["rows"]]
    assert a["kind"] == "fixed_code"
    assert a["ratio_mean_sq"] >= 1.0 - 1e-9

    # The analog side is a closed form, so it must equal the moment formula exactly.
    mom = tied_energy_moments(Phi)
    for r in a["rows"]:
        assert r["linear_energy_per_coord"] == pytest.approx(tied_linear_energy(mom, r["s"]),
                                                            rel=1e-8)


def test_fixed_code_profile_differs_from_the_random_ensemble():
    """Guard against the two being silently interchanged: they measure different objects."""
    d, F = 16, 96
    rng = np.random.default_rng(3)
    Phi_a = random_unit_code(d, F, rng)
    Phi_b = random_unit_code(d, F, rng)
    kw = dict(sparsities=[1, 2, 3, 4], trials=64, seed=9, readout="tied")
    a = fixed_code_separation_profile(Phi_a, **kw)
    b = fixed_code_separation_profile(Phi_b, **kw)
    assert a["frob_sq_A"] != b["frob_sq_A"]
