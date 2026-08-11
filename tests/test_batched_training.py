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


def test_initialisation_comes_from_the_cpu_stream():
    """A seed must name the same initial code on every device.

    Every theoretical quantity in the study -- leverage, the collision frontier, the geometry
    ratio -- is a function of ``W_in``. A CUDA generator does not reproduce a CPU generator's
    stream for the same seed, so drawing the initialisation on the compute device would silently
    make "seed 0" mean a different code on the accelerator than in the committed results. This
    pins the initialisation to the CPU stream, which is checkable without a GPU present.
    """
    d, F, seeds = 8, 24, [0, 3]
    got = train_toy_models_batched(d=d, F=F, loss_kind="L2", p=0.05, seeds=seeds, steps=0,
                                   batch=32, dtype=torch.float64)
    for rec, seed in zip(got, seeds):
        g = torch.Generator().manual_seed(seed)
        want_in = (torch.randn(d, F, generator=g, dtype=torch.float64) / np.sqrt(F)).numpy()
        want_out = (torch.randn(F, d, generator=g, dtype=torch.float64) / np.sqrt(d)).numpy()
        assert rec["W_in"] == pytest.approx(want_in, rel=0, abs=0)
        assert rec["W_out"] == pytest.approx(want_out, rel=0, abs=0)


def test_cpu_data_is_a_no_op_on_cpu():
    """The flag only forces the batches onto the CPU stream, where they already are."""
    kw = dict(d=6, F=18, loss_kind="L4", p=0.05, seeds=[0, 1], steps=12, batch=32, lr=1e-3,
              dtype=torch.float64)
    a = train_toy_models_batched(**kw)
    b = train_toy_models_batched(cpu_data=True, **kw)
    for x, y in zip(a, b):
        assert x["W_in"] == pytest.approx(y["W_in"], rel=0, abs=0)
        assert x["W_out"] == pytest.approx(y["W_out"], rel=0, abs=0)


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
