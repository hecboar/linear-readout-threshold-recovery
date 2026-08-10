"""State distributions: paired supports, honest metrics, and the E2 prediction.

The point of the sweep is not to enumerate distributions. It is that the affine level of the
hierarchy reads only one number off the state law -- the minimum detectable amplitude -- so the
sweep is over that scalar and the theory makes a falsifiable prediction for each family.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from lrtr.affine_frontier import collision_frontier
from lrtr.analog_optimum import leverage
from lrtr.codes import random_unit_code
from lrtr.probes import distribution_profile
from lrtr.splits import make_state_splits, one_hot_targets, representations
from lrtr.state_families import (
    FAMILIES,
    apply_family,
    apply_family_to_bundle,
    family,
    truncated_families,
)
from lrtr.toymodel import train_toy_models_batched


@pytest.fixture(scope="module")
def model():
    return train_toy_models_batched(d=16, F=40, loss_kind="L2", p=0.06, seeds=[0],
                                    steps=300, batch=256, lr=5e-3)[0]


@pytest.fixture(scope="module")
def bundle():
    return make_state_splits(F=40, sparsities=[2, 3], n_train=300, n_val=120, n_test=80, seed=0)


# =====================================================================================
# The families themselves
# =====================================================================================

@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_family_respects_its_own_alpha(name):
    """A declared minimum amplitude has to be a real floor on the draw."""
    fam = family(name)
    b = make_state_splits(F=30, sparsities=[2, 3], n_train=200, n_val=80, n_test=60, seed=1)
    sp = apply_family(b.train, fam, d=8, seed=3)
    if fam.amplitude is None:
        assert sp.amplitudes is None                      # Boolean: implicit ones
        return
    live = np.abs(sp.amplitudes[sp.sparsity[:, None] > np.arange(sp.supports.shape[1])])
    assert np.all(live <= 1.0 + 1e-12)
    if fam.alpha is not None:
        assert np.all(live >= fam.alpha - 1e-12), live.min()
    else:
        assert live.min() < 0.1        # untruncated: amplitudes really do approach zero


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_families_share_supports_and_labels(name):
    """Only the amplitudes may change, so the cross-distribution comparison stays paired."""
    b = make_state_splits(F=30, sparsities=[2, 3], n_train=200, n_val=80, n_test=60, seed=2)
    bb = apply_family_to_bundle(b, family(name), d=8, seed=5)
    for s in b.sparsities:
        assert bb.test_by_s[s].keys() == b.test_by_s[s].keys()
        assert np.array_equal(one_hot_targets(bb.test_by_s[s]),
                              one_hot_targets(b.test_by_s[s]))
    assert bb.exhausted == b.exhausted and bb.seed == b.seed


def test_amplitudes_scale_the_representation():
    W = random_unit_code(6, 30, np.random.default_rng(0))
    b = make_state_splits(F=30, sparsities=[2], n_train=80, n_val=40, n_test=40, seed=0)
    base = representations(W, b.val, post_relu=False)
    half = representations(W, b.val.with_amplitudes(
        np.full(b.val.supports.shape, 0.5)), post_relu=False)
    assert half == pytest.approx(0.5 * base, abs=1e-12)


def test_representation_noise_is_added_before_the_nonlinearity():
    W = random_unit_code(6, 30, np.random.default_rng(1))
    b = make_state_splits(F=30, sparsities=[2], n_train=80, n_val=40, n_test=40, seed=0)
    noisy = apply_family(b.val, family("boolean_noise_0.15"), d=6, seed=9)
    assert noisy.rep_noise is not None and noisy.rep_noise.shape == (len(b.val), 6)
    clean = representations(W, b.val, post_relu=False)
    got = representations(W, noisy, post_relu=False)
    assert got == pytest.approx(clean + noisy.rep_noise, abs=1e-12)
    # And the ReLU is applied after, so the two are not simply offset post-nonlinearity.
    assert not np.allclose(representations(W, noisy, post_relu=True),
                           np.maximum(clean, 0.0) + noisy.rep_noise)


def test_mismatched_noise_shape_is_rejected():
    from lrtr.splits import StateSplit
    W = random_unit_code(6, 30, np.random.default_rng(2))
    b = make_state_splits(F=30, sparsities=[2], n_train=40, n_val=20, n_test=20, seed=0)
    bad = StateSplit(supports=b.val.supports, sparsity=b.val.sparsity, F=30,
                     rep_noise=np.zeros((len(b.val), 99)))
    with pytest.raises(ValueError, match="rep_noise must be"):
        representations(W, bad, post_relu=False)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown state family"):
        family("gaussian_everything")


def test_truncated_families_are_exactly_those_with_a_floor():
    names = sorted(FAMILIES)
    trunc = set(truncated_families(names))
    assert trunc == {n for n in names if family(n).alpha is not None}
    assert "native" not in trunc          # its amplitudes reach zero, so exact recovery is not
    assert "boolean" in trunc             # a meaningful metric for it


def test_amplitudes_are_deterministic_across_processes():
    """hash() is per-process randomised, so the family seed must not come from it."""
    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from lrtr.splits import make_state_splits\n"
        "from lrtr.state_families import apply_family_to_bundle, family\n"
        "b = make_state_splits(F=20, sparsities=[2], n_train=40, n_val=20, n_test=20, seed=0)\n"
        "bb = apply_family_to_bundle(b, family('amp_0.5'), d=6, seed=1)\n"
        "print(round(float(bb.test_by_s[2].amplitudes.sum()), 8))\n")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True).stdout.strip() for _ in range(2)}
    assert len(outs) == 1 and outs != {""}, outs


# =====================================================================================
# The E2 prediction
# =====================================================================================

@pytest.mark.parametrize("d,F", [(8, 24), (10, 30)])
def test_frontier_is_monotone_in_alpha(d, F):
    """A weaker activation cannot be easier to detect, so kappa must not rise as alpha falls."""
    Phi = random_unit_code(d, F, np.random.default_rng(d))
    feats = np.argsort(leverage(Phi))[:6].tolist()
    prev = None
    for alpha in (1.0, 0.8, 0.5, 0.25):
        k = collision_frontier(Phi, feature_subset=feats, model="atmost",
                               alpha=alpha)["frontier_min"]
        if prev is not None:
            assert k <= prev + 1e-6, (alpha, k, prev)
        prev = k


def test_distribution_profile_reports_per_family(model, bundle):
    W_in, W_out = model["W_in"], model["W_out"]
    feats = np.argsort(leverage(W_in))[:6].tolist()
    names = ["boolean", "amp_0.5", "native"]
    out = distribution_profile(W_in, W_out, bundle, names, frontier_features=feats,
                               families=("ridge",), reps=("pre",), policies=("fixed",),
                               grid=(1e-4,), include_oracle=False)
    assert set(out["families"]) == set(names)
    for name in names:
        v = out["families"][name]
        assert v["exact_recovery_meaningful"] == (family(name).alpha is not None)
        assert 0.0 <= v["network"]["recovery_auc"] <= 1.0
        assert 0.0 <= v["best_probe_auc"] <= 1.0
        if v["alpha"] is not None:
            assert v["frontier_at_alpha"]["is_upper_bound"] is True
            assert v["frontier_at_alpha"]["kappa_min"] >= 1.0 - 1e-9
        else:
            assert "frontier_at_alpha" not in v      # no frontier statement without a floor
    assert out["alpha_monotone"] is True
    assert [r["alpha"] for r in out["alpha_vs_kappa"]] == sorted(
        [r["alpha"] for r in out["alpha_vs_kappa"]], reverse=True)


def test_probes_are_refitted_per_family(model, bundle):
    """A probe tuned on Boolean states and shown continuous ones would be a straw baseline."""
    W_in, W_out = model["W_in"], model["W_out"]
    out = distribution_profile(W_in, W_out, bundle, ["boolean", "amp_0.25"],
                               frontier_features=[0, 1], families=("ridge",), reps=("pre",),
                               policies=("fixed",), grid=(1e-6, 1e-4, 1e-2, 1.0),
                               include_oracle=False)
    a = out["families"]["boolean"]["probes"]["ridge_pre_fixed"]
    b = out["families"]["amp_0.25"]["probes"]["ridge_pre_fixed"]
    # Different data means the selected penalty or the achieved AUC has to differ somewhere;
    # identical records would mean the family never reached the fit.
    assert (a["penalty"], a["recovery_auc"]) != (b["penalty"], b["recovery_auc"])
