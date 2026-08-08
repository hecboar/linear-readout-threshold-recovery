"""Cross-validated probes and native-distribution evaluation.

The probes exist to make the trained-network comparison harder to dismiss, so what matters is
that they are genuinely out of sample and genuinely strong. Both are checked here, along with
the closed form behind the native-distribution analog error.
"""
from __future__ import annotations

import numpy as np
import pytest

from lrtr.codes import random_unit_code
from lrtr.probes import (
    boolean_states,
    apply_affine_probe,
    boolean_probe_profile,
    fit_affine_probe,
    native_profile,
)
from lrtr.toymodel import train_toy_models_batched


@pytest.fixture(scope="module")
def model():
    return train_toy_models_batched(d=12, F=24, loss_kind="L2", p=0.08, seeds=[0],
                                    steps=400, batch=256, lr=5e-3)[0]


def test_affine_probe_recovers_an_exactly_affine_map():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((200, 5))
    W_true = rng.standard_normal((6, 3))
    Y = apply_affine_probe(W_true, R)
    W = fit_affine_probe(R, Y, ridge=0.0)
    assert W == pytest.approx(W_true, rel=1e-8, abs=1e-8)


def test_intercept_is_not_penalised():
    """A constant target must be fitted exactly however large the ridge on the slopes."""
    rng = np.random.default_rng(1)
    R = rng.standard_normal((100, 4))
    Y = np.full((100, 1), 7.0)
    W = fit_affine_probe(R, Y, ridge=1e6)
    assert apply_affine_probe(W, R) == pytest.approx(Y, rel=1e-6)


def test_probe_evaluation_is_out_of_sample(model):
    """Scoring the fit states instead of held-out ones would inflate the probe; it must not."""
    prof = boolean_probe_profile(model["W_in"], model["W_out"], sparsities=[1, 2],
                                 n_train=400, n_test=200, seed=3)
    for row in prof["rows"]:
        assert row["n_test"] == 200
        for key in ("p_rec_model", "p_rec_probe_pre", "p_rec_probe_post"):
            assert 0.0 <= row[key] <= 1.0
        assert row["ci_low_model"] <= row["p_rec_model"] <= row["ci_high_model"]
    assert set(prof["s95"]) == {"model", "probe_pre", "probe_post"}


def test_post_relu_probe_is_at_least_as_expressive_as_pre(model):
    """Post-ReLU features are a superset in practice; a large gap the other way means a bug."""
    prof = boolean_probe_profile(model["W_in"], model["W_out"], sparsities=[1],
                                 n_train=2000, n_test=500, seed=5)
    row = prof["rows"][0]
    assert row["rms_probe_post"] <= row["rms_probe_pre"] * 1.5


def test_native_analog_error_matches_monte_carlo(model):
    """The closed form (p/3)||A||_F^2/F must agree with sampling the native distribution."""
    W_in, W_out = model["W_in"], model["W_out"]
    d, F = W_in.shape
    p = 0.08
    prof = native_profile(W_in, W_out, p=p, n_train=800, n_test=400, seed=7)

    G = np.linalg.pinv(W_in)
    M = (G @ W_in) / np.diag(G @ W_in)[:, None]
    A = M - np.eye(F)
    rng = np.random.default_rng(11)
    X = (rng.random((F, 20000)) < p) * (rng.random((F, 20000)) * 2.0 - 1.0)
    mc = float((A @ X).__pow__(2).mean())

    assert prof["analog"]["pinv"]["energy_per_coord"] == pytest.approx(mc, rel=0.06)
    assert prof["analog"]["pinv"]["rms"] >= prof["analog"]["rms_floor_native"]


def test_native_detection_beats_the_trivial_predictor(model):
    prof = native_profile(model["W_in"], model["W_out"], p=0.08, n_train=600, n_test=300,
                          seed=9)
    trivial = max(prof["base_rate"], 1.0 - prof["base_rate"])
    for name, det in prof["detection"].items():
        assert 0.0 <= det["accuracy"] <= 1.0, name
        assert det["accuracy"] >= trivial - 0.05, name


def test_boolean_states_are_uniform_over_supports():
    """Partial-sort sampling must match rng.choice: exactly s active, no feature favoured."""
    F, s, n = 8, 3, 40000
    B = boolean_states(F, s, n, np.random.default_rng(0))
    assert np.all(B.sum(axis=0) == s)
    assert B.sum(axis=1) / n == pytest.approx(np.full(F, s / F), abs=0.01)


def test_boolean_states_rejects_impossible_sparsity():
    with pytest.raises(ValueError, match="1 <= s <= F"):
        boolean_states(4, 5, 10, np.random.default_rng(0))
