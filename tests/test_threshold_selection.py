"""`select_thresholds` must never return a threshold worse than `theta_fixed`.

The defect this closes (KD6). The routine searched a quantile grid of the validation scores and
initialised `best = -1.0`, so `theta_fixed` was the initial value of `best_theta` but was never
scored -- the first grid point always displaced it, whatever it was worth. A quantile grid does not
have to contain a good threshold: when the scores are bimodal, with most coordinates inactive near
zero and a few active near one, the quantiles crowd into the two modes and sample the decision
region between them sparsely or not at all.

That is not hypothetical. On E7's trained networks the selected `global` threshold lost to
`theta = 0.5` on validation exact recovery in 100% of models at every width, and cost up to five
sparsity levels of `s95` at `d=400`. Because the same routine sets the probes' thresholds, it
silently handicapped both sides of the comparison the manuscript's headline rested on.

The post-condition is the fix: whatever the search does, the returned threshold is compared against
`theta_fixed` and can only be an improvement on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrtr.probes import select_thresholds  # noqa: E402
from lrtr.splits import StateSplit, one_hot_targets  # noqa: E402

THETA = 0.5


def _split(supports: np.ndarray, F: int) -> StateSplit:
    return StateSplit(supports=supports, sparsity=np.full(len(supports), supports.shape[1]), F=F)


def _objective(Z: np.ndarray, split: StateSplit, theta) -> float:
    Y = one_hot_targets(split) > 0.5
    return float(np.all((Z >= theta) == Y, axis=1).mean())


def _bimodal_case(n: int = 400, F: int = 40, s: int = 3, seed: int = 0):
    """Scores that separate perfectly at 0.5 while the quantile grid avoids that region.

    Inactive coordinates sit near 0.02, active ones near 0.98, and nothing lands in between. This
    is the shape a trained network's output takes on sparse Boolean states, and it is exactly the
    shape a quantile grid handles worst.
    """
    rng = np.random.default_rng(seed)
    supports = np.stack([rng.choice(F, size=s, replace=False) for _ in range(n)])
    split = _split(supports, F)
    Y = one_hot_targets(split) > 0.5
    Z = np.where(Y, 0.98, 0.02) + 0.005 * rng.standard_normal((n, F))
    return Z, split


@pytest.mark.parametrize("policy", ["global", "per_feature"])
def test_selected_threshold_never_loses_to_theta_fixed(policy: str) -> None:
    Z, split = _bimodal_case()
    chosen = select_thresholds(Z, split, policy, theta_fixed=THETA)
    assert _objective(Z, split, chosen) >= _objective(Z, split, THETA)


@pytest.mark.parametrize("policy", ["global", "per_feature"])
@pytest.mark.parametrize("seed", range(6))
def test_post_condition_holds_on_random_score_distributions(policy: str, seed: int) -> None:
    """The guarantee must not depend on the scores being well behaved."""
    rng = np.random.default_rng(1000 + seed)
    n, F, s = 300, 25, 4
    supports = np.stack([rng.choice(F, size=s, replace=False) for _ in range(n)])
    split = _split(supports, F)
    Y = one_hot_targets(split) > 0.5
    # Deliberately awkward: an arbitrary offset and scale, so theta_fixed is not on the natural
    # scale of the scores. This is the probes' situation -- ridge, logistic and hinge outputs do
    # not share a scale with each other or with the network.
    Z = (Y * rng.uniform(0.5, 3.0) + rng.standard_normal((n, F)) * 0.4
         + rng.uniform(-1.0, 1.0))
    chosen = select_thresholds(Z, split, policy, theta_fixed=THETA)
    assert _objective(Z, split, chosen) >= _objective(Z, split, THETA) - 1e-12


def test_the_old_quantile_only_search_would_have_failed_this() -> None:
    """Reproduce the defect, so the test proves the fix rather than merely passing beside it.

    This is the previous implementation of the `global` branch, verbatim in behaviour: quantiles
    only, `best` initialised below every attainable score. On the bimodal case it must do worse than
    `theta_fixed`, which is what the current implementation is required not to do.
    """
    Z, split = _bimodal_case()
    Y = one_hot_targets(split) > 0.5
    grid = np.quantile(Z, np.linspace(0.01, 0.999, 64))
    best, old_theta = -1.0, THETA
    for t in grid:
        score = float(np.all((Z >= t) == Y, axis=1).mean())
        if score > best:
            best, old_theta = score, float(t)

    assert _objective(Z, split, old_theta) < _objective(Z, split, THETA), (
        "the regression case no longer reproduces the defect; it is not testing the fix")
    assert _objective(Z, split, select_thresholds(Z, split, "global", theta_fixed=THETA)) \
        >= _objective(Z, split, THETA)


def test_ties_resolve_toward_theta_fixed() -> None:
    """When nothing beats it, the registered threshold is what comes back.

    Moving off a semantically meaningful threshold should take evidence. Here every candidate in the
    decision region is equally perfect, so the answer must be `theta_fixed` itself.
    """
    Z, split = _bimodal_case()
    assert select_thresholds(Z, split, "global", theta_fixed=THETA) == pytest.approx(THETA)


def test_fixed_policy_is_untouched() -> None:
    Z, split = _bimodal_case()
    assert select_thresholds(Z, split, "fixed", theta_fixed=0.37) == pytest.approx(0.37)
