"""Numerical verification of the theorems the Interface Diagnostic relies on.

These tests are correctness checks on the implementation, not evidence for the theorems:
the statements are proved in the manuscript. A failure here means the code is wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from lrtr.codes import (
    coherence,
    harmonic_tight_frame,
    random_unit_code,
    tight_frame_residual,
    welch_floor,
    welch_floor_max,
)
from lrtr.interface import (
    crosstalk_stats,
    energy_floor_bernoulli,
    energy_floor_uniform,
    linear_energy_bernoulli,
    linear_energy_uniform,
    unit_diagonal,
)


def test_welch_floor_value():
    # W(100, 50) = 50 / (50 * 99)
    assert welch_floor(100, 50) == pytest.approx(50 / (50 * 99))
    assert welch_floor_max(100, 50) == pytest.approx(np.sqrt(50 / (50 * 99)))


def test_welch_floor_rejects_non_overcomplete():
    with pytest.raises(ValueError, match="F > d"):
        welch_floor(16, 16)


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 100), (32, 257)])
def test_harmonic_frame_is_a_unit_norm_tight_frame(d, F):
    Phi = harmonic_tight_frame(d, F)
    assert np.allclose(np.linalg.norm(Phi, axis=0), 1.0, atol=1e-12)
    assert tight_frame_residual(Phi) < 1e-12


@pytest.mark.parametrize("d,F", [(4, 12), (8, 40), (16, 100), (32, 257)])
def test_tight_frame_attains_the_floor_with_equality(d, F):
    """Equality case: for a unit-norm tight frame the mean-square floor is attained."""
    Phi = harmonic_tight_frame(d, F)
    M = Phi.T @ Phi
    st = crosstalk_stats(M, d)
    assert st.ratio_mean_sq == pytest.approx(1.0, rel=1e-10)
    assert not st.violates_floor


@pytest.mark.parametrize("seed", range(8))
def test_welch_floor_never_violated_by_random_interfaces(seed):
    """Random G and Phi, calibrated to unit diagonal: the floor must hold every time."""
    rng = np.random.default_rng(seed)
    d, F = 12, 60
    Phi = random_unit_code(d, F, rng)
    G = rng.standard_normal((F, d))
    st = crosstalk_stats(unit_diagonal(G @ Phi), d)
    assert not st.violates_floor
    assert st.ratio_mean_sq >= 1.0 - 1e-12
    assert st.ratio_max >= 1.0 - 1e-12


def test_calibration_is_not_cosmetic():
    """The remark accompanying the calibration-free corollary, verified exactly.

    Three unit vectors at 120 degrees in R^2 (an equiangular tight frame) read out by
    ``G = D Phi^T`` with ``D = diag(1/10, 1/20, 1/20)``. The raw off-diagonal entries of
    ``G Phi`` are ten times *below* the floor, yet after calibration the interface sits
    exactly *on* it.
    """
    ang = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    Phi = np.vstack([np.cos(ang), np.sin(ang)])  # (2, 3)
    D = np.diag([1 / 10, 1 / 20, 1 / 20])
    M_raw = D @ (Phi.T @ Phi)

    assert np.allclose(np.diagonal(M_raw), [1 / 10, 1 / 20, 1 / 20])
    off_raw = np.abs(M_raw - np.diag(np.diagonal(M_raw)))
    assert off_raw.max() == pytest.approx(1 / 20)
    floor_max = welch_floor_max(3, 2)
    assert floor_max == pytest.approx(0.5)
    assert off_raw.max() < floor_max  # apparent "violation" before calibration

    st = crosstalk_stats(unit_diagonal(M_raw), 2)
    assert st.max_abs_offdiag == pytest.approx(0.5)
    assert st.ratio_max == pytest.approx(1.0)
    assert not st.violates_floor


def test_shrinking_the_readout_cannot_evade_the_floor():
    """Scaling G down shrinks raw cross-talk arbitrarily but leaves the ratio invariant."""
    rng = np.random.default_rng(0)
    d, F = 8, 40
    Phi = random_unit_code(d, F, rng)
    G = rng.standard_normal((F, d))
    base = crosstalk_stats(unit_diagonal(G @ Phi), d)
    for scale in (1e-1, 1e-3, 1e-6):
        st = crosstalk_stats(unit_diagonal((scale * G) @ Phi), d)
        assert st.ratio_mean_sq == pytest.approx(base.ratio_mean_sq, rel=1e-8)


def test_unit_diagonal_rejects_zero_gain():
    M = np.array([[1.0, 0.2], [0.2, 0.0]])
    with pytest.raises(ValueError, match="near-zero diagonal gain"):
        unit_diagonal(M)


@pytest.mark.parametrize("s", [1, 3, 7])
def test_uniform_energy_closed_form_matches_monte_carlo(s):
    rng = np.random.default_rng(11)
    d, F = 10, 50
    Phi = random_unit_code(d, F, rng)
    A = Phi.T @ Phi - np.eye(F)
    closed = linear_energy_uniform(A, s)
    n = 40000
    acc = 0.0
    for _ in range(n):
        b = np.zeros(F)
        b[rng.choice(F, size=s, replace=False)] = 1.0
        r = A @ b
        acc += float(r @ r)
    mc = acc / n / F
    assert mc == pytest.approx(closed, rel=0.03)


@pytest.mark.parametrize("p", [0.02, 0.1])
def test_bernoulli_energy_closed_form_matches_monte_carlo(p):
    rng = np.random.default_rng(12)
    d, F = 10, 50
    Phi = random_unit_code(d, F, rng)
    A = Phi.T @ Phi - np.eye(F)
    closed = linear_energy_bernoulli(A, p)
    n = 40000
    B = (rng.random((n, F)) < p).astype(float)
    R = B @ A.T
    mc = float((R * R).sum()) / n / F
    assert mc == pytest.approx(closed, rel=0.03)


@pytest.mark.parametrize("seed", range(5))
def test_energy_floors_are_respected(seed):
    """Both energy floors must hold for every calibrated rank-d interface."""
    rng = np.random.default_rng(seed)
    d, F = 12, 80
    Phi = random_unit_code(d, F, rng)
    G = rng.standard_normal((F, d))
    M = unit_diagonal(G @ Phi)
    A = M - np.eye(F)
    for s in (1, 2, 5, 10):
        assert linear_energy_uniform(A, s) >= energy_floor_uniform(F, d, s) * (1 - 1e-9)
        assert linear_energy_bernoulli(A, s / F) >= energy_floor_bernoulli(F, d, s) * (1 - 1e-9)


def test_uniform_floor_is_nearly_tight_for_tight_frames():
    """For a unit-norm tight frame the uniform-support floor is attained up to the A1 term.

    The bound drops the non-negative ``q ||A 1||^2`` contribution, so equality is not exact;
    for a harmonic frame the excess is well under one percent.
    """
    d, F, s = 16, 100, 4
    Phi = harmonic_tight_frame(d, F)
    A = Phi.T @ Phi - np.eye(F)
    exact = linear_energy_uniform(A, s)
    floor = energy_floor_uniform(F, d, s)
    assert exact >= floor * (1 - 1e-9)
    assert exact / floor < 1.01


def test_random_code_coherence_scale():
    """Coherence of a random code at F = d^2 sits at the c sqrt(log d / d) scale."""
    rng = np.random.default_rng(3)
    d = 64
    Phi = random_unit_code(d, d * d, rng, dtype=np.float32)
    mu = coherence(Phi.astype(np.float64))
    ref = np.sqrt(np.log(d) / d)
    assert 1.0 < mu / ref < 6.0
