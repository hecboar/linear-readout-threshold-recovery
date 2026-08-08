"""Seed-batched training must reproduce individually-trained models exactly.

The scaled campaign trains many seeds as one batched computation so that an accelerator is
actually busy. That is only legitimate if it changes nothing about each model, so the
equivalence is asserted here rather than assumed: same initialisation, same data, same
optimiser trajectory. Run in float64, where the only remaining discrepancy is the reduction
order inside the matmuls.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lrtr.toymodel import TASKS, train_toy_model, train_toy_models_batched

SEEDS = [0, 1, 2]


@pytest.mark.parametrize("loss_kind", ["L2", "L4"])
def test_batched_matches_individual(loss_kind):
    kw = dict(d=8, F=24, p=0.05, steps=25, batch=64, lr=1e-3, dtype=torch.float64)
    batched = train_toy_models_batched(loss_kind=loss_kind, seeds=SEEDS, **kw)
    for i, seed in enumerate(SEEDS):
        single = train_toy_model(loss_kind=loss_kind, seed=seed, log_every=10**9, **kw)
        assert batched[i]["W_in"] == pytest.approx(single["W_in"], rel=1e-8, abs=1e-12)
        assert batched[i]["W_out"] == pytest.approx(single["W_out"], rel=1e-8, abs=1e-12)
        assert batched[i]["final_mse"] == pytest.approx(single["final_mse"], rel=1e-8)
        assert batched[i]["seed"] == seed


def test_seeds_are_independent():
    """Adding a seed must not perturb the others: each keeps its own generator."""
    kw = dict(d=6, F=18, loss_kind="L2", p=0.05, steps=10, batch=32, lr=1e-3,
              dtype=torch.float64)
    a = train_toy_models_batched(seeds=[0, 1], **kw)
    b = train_toy_models_batched(seeds=[0, 1, 7], **kw)
    for i in range(2):
        assert a[i]["W_in"] == pytest.approx(b[i]["W_in"], rel=1e-10, abs=1e-14)
    assert not np.allclose(b[2]["W_in"], b[0]["W_in"])


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_trains_and_beats_the_zero_predictor(task):
    out = train_toy_models_batched(d=10, F=20, loss_kind="L2", p=0.05, seeds=[0],
                                   steps=300, batch=256, lr=5e-3, task=task)[0]
    assert out["task"] == task
    assert np.isfinite(out["final_mse"])
    assert out["mse_ratio_to_zero_predictor"] < 1.0


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown task"):
        train_toy_models_batched(d=4, F=8, loss_kind="L2", p=0.1, seeds=[0], steps=1,
                                 task="cube")
