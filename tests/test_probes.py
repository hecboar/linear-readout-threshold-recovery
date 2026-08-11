"""Affine probes and the split machinery.

The probes exist to make the trained-network comparison hard to dismiss, so what is tested is
exactly that: the splits are genuinely disjoint, the fits are correct, and nothing in the
selection path can see the locked test set. The last one is checked structurally -- by poisoning
the test set and asserting that not one selected hyperparameter moves -- rather than by reading
the code and trusting it.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from lrtr.codes import random_unit_code
from lrtr.probes import (
    DEFAULT_RIDGE_GRID,
    PROBE_FAMILIES,
    THRESHOLD_POLICIES,
    evaluate_fixed_readout,
    evaluate_network,
    evaluate_probe,
    fit_probe,
    probe_profile,
    score_probe,
    select_probe,
    select_thresholds,
)
from lrtr.splits import (
    StateSplit,
    draw_supports,
    make_state_splits,
    one_hot_targets,
    representations,
)
from lrtr.toymodel import train_toy_models_batched


@pytest.fixture(scope="module")
def model():
    return train_toy_models_batched(d=16, F=40, loss_kind="L2", p=0.06, seeds=[0],
                                    steps=400, batch=256, lr=5e-3)[0]


@pytest.fixture(scope="module")
def bundle():
    return make_state_splits(F=40, sparsities=[2, 3, 4], n_train=400, n_val=160,
                             n_test=100, seed=0)


# =====================================================================================
# Splits
# =====================================================================================

def test_splits_are_pairwise_and_globally_disjoint(bundle):
    keys = bundle.all_keys()
    for a, b in itertools.combinations(keys, 2):
        assert not (keys[a] & keys[b]), f"{a} overlaps {b}"
    total = sum(len(v) for v in keys.values())
    assert total == len(set().union(*keys.values()))


def test_draw_supports_is_uniform_and_never_repeats():
    F, s, n = 12, 3, 150
    seen = set()
    sup, redrawn, ok = draw_supports(F, s, n, np.random.default_rng(0), seen)
    assert ok and len(sup) == n
    assert len(seen) == n                                # every draw distinct
    assert np.all(np.diff(sup, axis=1) > 0)              # sorted, no repeated index
    freq = np.bincount(sup.ravel(), minlength=F) / (n * s)
    assert freq == pytest.approx(np.full(F, 1.0 / F), abs=0.05)


def test_exhaustion_is_reported_not_hidden():
    b = make_state_splits(F=6, sparsities=[3], n_train=40, n_val=10, n_test=10, seed=0)
    assert b.exhausted == [3]                            # C(6,3) = 20 cannot fill 60 states
    assert len(b.test_by_s[3]) == 10                     # the locked set is drawn first, intact


def test_representations_match_the_explicit_indicator_route(model, bundle):
    W_in = model["W_in"]
    B = one_hot_targets(bundle.val).T
    for post in (False, True):
        ref = (W_in @ B).T
        if post:
            ref = np.maximum(ref, 0.0)
        assert representations(W_in, bundle.val, post_relu=post) == pytest.approx(ref, abs=1e-12)


def test_of_sparsity_selects_the_right_states(bundle):
    for s in bundle.sparsities:
        sub = bundle.val.of_sparsity(s)
        assert np.all(sub.sparsity == s)
        assert sub.supports.shape[1] == s


# =====================================================================================
# Fitting
# =====================================================================================

def test_ridge_matches_an_explicit_least_squares_solve(model, bundle):
    W_in = model["W_in"]
    X = np.hstack([representations(W_in, bundle.train, post_relu=False),
                   np.ones((len(bundle.train), 1))])
    Y = one_hot_targets(bundle.train)
    fits = fit_probe(W_in, bundle.train, "ridge", "pre", grid=(1e-3,))
    pen = np.eye(X.shape[1]) * 1e-3
    pen[-1, -1] = 0.0
    ref = np.linalg.solve(X.T @ X + pen, X.T @ Y)
    assert fits[1e-3] == pytest.approx(ref, rel=1e-8, abs=1e-10)


def test_ridge_leaves_the_intercept_unpenalised(model):
    """A constant target must be fitted exactly however large the penalty on the slopes."""
    W_in = model["W_in"]
    F = W_in.shape[1]
    sup = np.zeros((80, 1), dtype=np.int32)
    split = StateSplit(supports=sup, sparsity=np.ones(80, dtype=np.int32), F=F)
    X = np.hstack([representations(W_in, split, post_relu=False), np.ones((80, 1))])
    fits = fit_probe(W_in, split, "ridge", "pre", grid=(1e6,))
    # Feature 0 is on in every state, so its column of Y is all ones and must be reproduced.
    assert (X @ fits[1e6])[:, 0] == pytest.approx(np.ones(80), rel=1e-4)


@pytest.mark.parametrize("family", PROBE_FAMILIES)
def test_every_family_fits_and_scores(family, model, bundle):
    W_in = model["W_in"]
    fits = fit_probe(W_in, bundle.train, family, "post", grid=(1e-4, 1e-2),
                     steps=60, batch=128)
    assert set(fits) == {1e-4, 1e-2}
    for W in fits.values():
        assert W.shape == (W_in.shape[0] + 1, W_in.shape[1])
        assert np.all(np.isfinite(W))
        Z = score_probe(W_in, bundle.val, W, "post")
        assert Z.shape == (len(bundle.val), W_in.shape[1])
        assert np.all(np.isfinite(Z))


def test_logistic_gradient_is_stable_at_extreme_scores(model, bundle):
    """The stable sigmoid must not overflow even with a large learning rate."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fits = fit_probe(model["W_in"], bundle.train, "logistic", "pre", grid=(0.0,),
                         steps=200, batch=128, lr=20.0)
    assert np.all(np.isfinite(fits[0.0]))


def test_unknown_family_and_representation_are_rejected(model, bundle):
    with pytest.raises(ValueError, match="unknown family"):
        fit_probe(model["W_in"], bundle.train, "randomforest", "pre")
    with pytest.raises(ValueError, match="unknown representation"):
        fit_probe(model["W_in"], bundle.train, "ridge", "mid")


# =====================================================================================
# Thresholds
# =====================================================================================

@pytest.mark.parametrize("policy", THRESHOLD_POLICIES)
def test_threshold_policies_return_the_right_shape(policy, model, bundle):
    W_in = model["W_in"]
    W = fit_probe(W_in, bundle.train, "ridge", "pre", grid=(1e-4,))[1e-4]
    Z = score_probe(W_in, bundle.val, W, "pre")
    theta = select_thresholds(Z, bundle.val, policy)
    if policy == "per_feature":
        assert np.shape(theta) == (W_in.shape[1],)
    else:
        assert np.isscalar(theta) or np.shape(theta) == ()


def test_fixed_policy_returns_exactly_theta(model, bundle):
    W_in = model["W_in"]
    W = fit_probe(W_in, bundle.train, "ridge", "pre", grid=(1e-4,))[1e-4]
    Z = score_probe(W_in, bundle.val, W, "pre")
    assert select_thresholds(Z, bundle.val, "fixed", theta_fixed=0.37) == 0.37


def test_unknown_policy_is_rejected(model, bundle):
    W_in = model["W_in"]
    W = fit_probe(W_in, bundle.train, "ridge", "pre", grid=(1e-4,))[1e-4]
    Z = score_probe(W_in, bundle.val, W, "pre")
    with pytest.raises(ValueError, match="unknown policy"):
        select_thresholds(Z, bundle.val, "magic")


# =====================================================================================
# Leakage: the selection path must be blind to the locked test set
# =====================================================================================

def test_selection_is_structurally_blind_to_the_test_set(model, bundle):
    """Poison the locked test set. Nothing selected may move.

    This is the test that matters. Reading the code and concluding "selection only sees
    validation" is an argument; replacing the test set with garbage and observing that the chosen
    penalty and thresholds are bit-identical is evidence.
    """
    W_in = model["W_in"]
    clean = select_probe(W_in, bundle.train, bundle.val, "ridge", "pre", "per_feature",
                         val_by_s=bundle.val_by_s)

    poisoned = make_state_splits(F=40, sparsities=[2, 3, 4], n_train=400, n_val=160,
                                 n_test=100, seed=999)
    bundle.test_by_s.update(poisoned.test_by_s)          # replace the locked set entirely
    after = select_probe(W_in, bundle.train, bundle.val, "ridge", "pre", "per_feature",
                         val_by_s=bundle.val_by_s)

    assert after["penalty"] == clean["penalty"]
    assert np.asarray(after["theta"]) == pytest.approx(np.asarray(clean["theta"]), abs=0.0)
    assert after["val_criterion"] == clean["val_criterion"]


def test_probe_scores_are_worse_out_of_sample_than_in_sample(model, bundle):
    """A sanity check that evaluation really is held out: fitting states score better."""
    W_in = model["W_in"]
    sel = select_probe(W_in, bundle.train, bundle.val, "ridge", "post", "fixed",
                       val_by_s=bundle.val_by_s)
    Z_in = score_probe(W_in, bundle.train, sel["W"], "post")
    Y_in = one_hot_targets(bundle.train) > 0.5
    in_sample = float(((Z_in >= sel["theta"]) == Y_in).mean())
    ev = evaluate_probe(W_in, bundle.test_by_s, sel["W"], "post", sel["theta"])
    out_sample = float(np.mean([r["coord_accuracy"] for r in ev["rows"]]))
    assert in_sample >= out_sample - 0.02


def test_shuffled_labels_give_no_recovery(model, bundle):
    """Negative control: break the input-target correspondence and recovery must collapse.

    Note what this does *not* do. Permuting the supports would relabel the states while keeping
    every (representation, target) pair consistent, so the probe would learn the correct map and
    the control would be vacuous -- that was the first version of this test, and it passed for
    the wrong reason. Here the design matrix keeps its row order while the targets are permuted
    across rows, so no affine map can do better than chance.
    """
    W_in = model["W_in"]
    F = W_in.shape[1]
    X = np.hstack([representations(W_in, bundle.train, post_relu=False),
                   np.ones((len(bundle.train), 1))])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(bundle.train))
    mismatched = StateSplit(supports=bundle.train.supports[order],
                            sparsity=bundle.train.sparsity[order], F=F)
    Y = one_hot_targets(mismatched)
    pen = np.eye(X.shape[1]) * 1e-4
    pen[-1, -1] = 0.0
    W = np.linalg.solve(X.T @ X + pen, X.T @ Y)
    ev = evaluate_probe(W_in, bundle.test_by_s, W, "pre", 0.5)
    assert max(r["p_rec"] for r in ev["rows"]) < 0.02


# =====================================================================================
# Locked evaluation and the profile
# =====================================================================================

def test_evaluation_reports_intervals_that_contain_the_estimate(model, bundle):
    W_in = model["W_in"]
    sel = select_probe(W_in, bundle.train, bundle.val, "ridge", "pre", "fixed",
                       val_by_s=bundle.val_by_s)
    ev = evaluate_probe(W_in, bundle.test_by_s, sel["W"], "pre", sel["theta"])
    for r in ev["rows"]:
        assert r["ci_low"] <= r["p_rec"] <= r["ci_high"]
        assert 0.0 <= r["coord_accuracy"] <= 1.0
        assert r["margin_min"] <= r["margin_p05"] <= r["margin_median"]
    assert set(ev) >= {"s95", "s95_interp", "s50", "recovery_auc"}


def test_calibration_reported_only_when_asked(model, bundle):
    W_in = model["W_in"]
    W = fit_probe(W_in, bundle.train, "logistic", "pre", grid=(1e-4,), steps=60,
                  batch=128)[1e-4]
    plain = evaluate_probe(W_in, bundle.test_by_s, W, "pre", 0.0)
    cal = evaluate_probe(W_in, bundle.test_by_s, W, "pre", 0.0, calibrated=True)
    assert "brier" not in plain["rows"][0]
    assert 0.0 <= cal["rows"][0]["brier"] <= 1.0 and cal["rows"][0]["ece"] >= 0.0


def test_network_and_fixed_readouts_are_scored_on_the_same_states(model, bundle):
    W_in, W_out = model["W_in"], model["W_out"]
    net = evaluate_network(W_in, W_out, bundle.test_by_s)
    fix = evaluate_fixed_readout(W_in, np.linalg.pinv(W_in), bundle.test_by_s)
    assert [r["s"] for r in net["rows"]] == [r["s"] for r in fix["rows"]]
    assert [r["n_test"] for r in net["rows"]] == [r["n_test"] for r in fix["rows"]]


def test_profile_covers_the_grid_and_labels_the_oracle(model, bundle):
    prof = probe_profile(model["W_in"], bundle, families=("ridge",), reps=("pre", "post"),
                         policies=("fixed", "per_feature"), grid=(1e-4, 1e-2))
    assert set(prof["global"]) == {"ridge_pre_fixed", "ridge_pre_per_feature",
                                   "ridge_post_fixed", "ridge_post_per_feature"}
    assert set(prof["oracle"]) == set(prof["global"])
    for v in prof["global"].values():
        assert v["is_oracle"] is False and v["penalty"] in (1e-4, 1e-2)
        assert v["val_objective"] in ("curve_auc", "mixture_exact")
    for v in prof["oracle"].values():
        assert v["is_oracle"] is True and "upper envelope" in v["note"]
    assert prof["splits"]["exhausted_sparsities"] == []


def test_oracle_is_at_least_as_strong_as_the_global_probe(model, bundle):
    """It is told the sparsity it will be tested at, so it should not lose on average."""
    prof = probe_profile(model["W_in"], bundle, families=("ridge",), reps=("pre",),
                         policies=("fixed",), grid=DEFAULT_RIDGE_GRID)
    g = prof["global"]["ridge_pre_fixed"]["recovery_auc"]
    o = prof["oracle"]["ridge_pre_fixed"]["recovery_auc"]
    assert o >= g - 0.05, (o, g)


def test_selection_objective_is_recorded_and_switchable(model, bundle):
    a = select_probe(model["W_in"], bundle.train, bundle.val, "ridge", "pre", "fixed",
                     val_by_s=bundle.val_by_s, objective="curve_auc")
    b = select_probe(model["W_in"], bundle.train, bundle.val, "ridge", "pre", "fixed",
                     objective="mixture_exact")
    assert a["val_objective"] == "curve_auc" and b["val_objective"] == "mixture_exact"
    with pytest.raises(ValueError, match="unknown objective"):
        select_probe(model["W_in"], bundle.train, bundle.val, "ridge", "pre", "fixed",
                     objective="vibes")


def test_the_fit_does_not_depend_on_the_threshold_policy():
    """The invariant that lets probe_profile fit once and sweep policies.

    The weights depend on (family, representation, split, penalty); the policy only picks
    thresholds from validation scores afterwards. probe_profile relies on this to avoid refitting
    every probe once per policy, which was three times the necessary work on the margin families
    and dominated the campaign's wall clock. If this ever stops holding, the caching is wrong.
    """
    W_in = random_unit_code(10, 30, np.random.default_rng(4))
    b = make_state_splits(F=30, sparsities=[1, 2], n_train=400, n_val=200, n_test=150, seed=5)
    for fam in ("ridge", "logistic", "svm"):
        ref = None
        for pol in ("fixed", "global", "per_feature"):
            sel = select_probe(W_in, b.train, b.val, fam, "pre", pol, grid=(1e-2, 1.0),
                               steps=25, batch=64)
            fits = fit_probe(W_in, b.train, fam, "pre", grid=(1e-2, 1.0), steps=25, batch=64)
            # The selected penalty may differ by policy; the fitted weights for a given penalty
            # must not.
            assert sel["W"] == pytest.approx(fits[sel["penalty"]], rel=0, abs=0), (fam, pol)
            if ref is None:
                ref = {k: v.copy() for k, v in fits.items()}
            else:
                for k in ref:
                    assert fits[k] == pytest.approx(ref[k], rel=0, abs=0), (fam, pol, k)


def test_supplying_fits_changes_nothing():
    """Passing precomputed fits must reproduce the refitting path exactly."""
    W_in = random_unit_code(8, 24, np.random.default_rng(6))
    b = make_state_splits(F=24, sparsities=[1, 2], n_train=300, n_val=150, n_test=120, seed=8)
    kw = dict(grid=(1e-3, 1e-1), steps=20, batch=64)
    for fam in ("ridge", "svm"):
        a = select_probe(W_in, b.train, b.val, fam, "post", "global", **kw)
        fits = fit_probe(W_in, b.train, fam, "post", **kw)
        c = select_probe(W_in, b.train, b.val, fam, "post", "global", fits=fits, **kw)
        assert a["penalty"] == c["penalty"]
        assert a["W"] == pytest.approx(c["W"], rel=0, abs=0)
        assert a["val_objective"] == pytest.approx(c["val_objective"], rel=0, abs=0)
