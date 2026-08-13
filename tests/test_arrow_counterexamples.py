"""The counterexamples that an external audit used to refute two statements of ours.

Both were stated for every `h_i < 1` while their shared proof only covers `h_i <= 1/2`. The
existing tests reported no violations because they sampled random overcomplete codes, whose
lowest-leverage features sit below one half -- so the tests were not wrong, they were blind. These
pin the boundary explicitly, in both directions: the bound must hold below one half and must not
be asserted above it.
"""
from __future__ import annotations

import numpy as np
import pytest

from lrtr.affine_frontier import (
    affine_failure_threshold,
    collision_radius_atmost,
    leverage_upper_bound_on_kappa,
    min_l2_representation,
    separable_atmost,
)
from lrtr.analog_optimum import leverage
from lrtr.codes import random_unit_code


def arrow_counterexample():
    """`h_1 = 8/9 > 1/2`, unique representation `z = (2, 2)`, so `kappa_1 = infinity`."""
    r = np.sqrt(15) / 4
    return np.array([[1.0, 0.25, 0.25], [0.0, r, -r]])


def threshold_counterexample():
    """`h_i = 5/9 > 1/2`, unique representation `z = (1/5, 11/10)`, so `kappa_i = infinity`."""
    a = np.array([1.0, 0.0])
    b = np.array([-25.0 / 44.0, np.sqrt(1311.0) / 44.0])
    phi = np.array([-17.0 / 40.0, np.sqrt(1311.0) / 40.0])
    return np.column_stack([phi, a, b])


@pytest.mark.parametrize("build", [arrow_counterexample, threshold_counterexample])
def test_counterexample_columns_are_unit_norm(build):
    """Otherwise the codes would be outside the setting the statements are about."""
    assert np.linalg.norm(build(), axis=0) == pytest.approx(np.ones(3), abs=1e-12)


def test_the_frontier_is_infinite_when_the_box_makes_the_programme_infeasible():
    for build in (arrow_counterexample, threshold_counterexample):
        rec = collision_radius_atmost(build(), 0)
        assert rec["rho_hat"] == float("inf")
        assert rec["status"] == "infeasible_lp_separable_everywhere"
        assert separable_atmost(rec["rho_hat"], 10 ** 6)


def test_the_arrow_bound_is_not_asserted_above_one_half():
    """The refuted case: h = 8/9 gives a finite expression of 5 against a frontier of infinity."""
    Phi = arrow_counterexample()
    h = leverage(Phi)
    assert h[0] == pytest.approx(8.0 / 9.0, rel=1e-9)
    naive = 1.0 + np.sqrt((Phi.shape[1] - 1) * h[0] / (1.0 - h[0]))
    assert naive == pytest.approx(5.0, rel=1e-9)          # what we used to claim
    assert collision_radius_atmost(Phi, 0)["rho_hat"] > naive
    assert leverage_upper_bound_on_kappa(Phi)[0] == float("inf")   # what we claim now


def test_the_failure_threshold_is_capped_at_one_half():
    """The refuted case: h = 5/9 <= 2/3 predicted failure at s = 3 for a separable feature."""
    Phi = threshold_counterexample()
    h = leverage(Phi)
    assert h[0] == pytest.approx(5.0 / 9.0, rel=1e-9)
    assert min_l2_representation(Phi)[0] == pytest.approx(1.25, rel=1e-9)
    assert 4.0 / 6.0 == pytest.approx(2.0 / 3.0)          # the uncapped s = 3 threshold
    assert h[0] <= 2.0 / 3.0                              # so the uncapped rule fired
    assert affine_failure_threshold(3, 3) == pytest.approx(0.5)
    assert h[0] > affine_failure_threshold(3, 3)          # the capped rule does not
    assert collision_radius_atmost(Phi, 0)["rho_hat"] == float("inf")


def test_the_cap_only_bites_past_the_square_root():
    """For `s - 1 <= sqrt(F-1)` the original expression is already below one half."""
    for F in (20, 100, 400):
        edge = int(np.floor(np.sqrt(F - 1))) + 1
        for s in range(2, edge + 1):
            k = (s - 1) ** 2
            assert affine_failure_threshold(F, s) == pytest.approx(k / ((F - 1) + k))
        big = edge + 40
        assert affine_failure_threshold(F, big) == pytest.approx(0.5)


@pytest.mark.parametrize("d,F,seed", [(8, 24, 0), (10, 40, 1), (6, 30, 2)])
def test_the_bound_still_holds_wherever_it_is_asserted(d, F, seed):
    """The corrected statement, checked on random codes: no violation where h <= 1/2."""
    Phi = random_unit_code(d, F, np.random.default_rng(seed))
    h = leverage(Phi)
    bound = leverage_upper_bound_on_kappa(Phi)
    for i in np.flatnonzero(h <= 0.5)[:12]:
        kappa = collision_radius_atmost(Phi, int(i))["rho_hat"]
        assert np.isfinite(bound[i])
        assert kappa <= bound[i] + 1e-6, (i, h[i], kappa, bound[i])


def test_duplicate_columns_are_separable_at_no_positive_sparsity():
    """`prop:oneway` claimed separability at s = 1; the rule needs kappa > s, and kappa = 1."""
    Phi = np.hstack([np.eye(4), np.eye(4)])
    rec = collision_radius_atmost(Phi, 0)
    assert rec["rho_hat"] == pytest.approx(1.0, abs=1e-9)
    assert not separable_atmost(rec["rho_hat"], 1)
    assert rec["s_max"] == 0
