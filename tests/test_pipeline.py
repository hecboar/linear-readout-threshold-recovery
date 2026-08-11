"""Smoke-level tests of the optimiser, the toy model and the diagnostic entry points."""
from __future__ import annotations

import os

import numpy as np
import pytest

from lrtr.codes import harmonic_tight_frame, random_unit_code
from lrtr.diagnostic import interface_floor_diagnostic, interface_separation_profile
from lrtr.optimize import train_code
from lrtr.toymodel import diagnose_model, random_code_baseline, train_toy_model


def test_floor_diagnostic_ratio_is_at_least_one():
    rng = np.random.default_rng(0)
    Phi = random_unit_code(16, 128, rng)
    for readout in ("tied", "pinv"):
        out = interface_floor_diagnostic(Phi, readout=readout)
        assert out["ratio_mean_sq"] >= 1.0 - 1e-9
        assert not out["violates_floor"]


def test_floor_diagnostic_is_one_for_a_tight_frame():
    Phi = harmonic_tight_frame(16, 128)
    out = interface_floor_diagnostic(Phi, readout="tied")
    assert out["ratio_mean_sq"] == pytest.approx(1.0, rel=1e-9)


def test_floor_diagnostic_rejects_undercomplete_codes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="overcomplete"):
        interface_floor_diagnostic(random_unit_code(32, 16, rng))


def test_separation_profile_shape_and_certificate():
    # d must be large enough that the coherence condition can hold at all; at d = 32 the
    # transition already sits below s = 1 (see tests/test_threshold.py).
    prof = interface_separation_profile(d=128, F=2048, sparsities=[1, 2, 3, 4],
                                        trials=25, master_seed=1, block=512)
    assert len(prof["rows"]) == 4
    for r in prof["rows"]:
        assert 0.0 <= r["p_rec"] <= 1.0
        assert r["ci_low"] <= r["p_rec"] <= r["ci_high"]
        # the measured linear energy must respect its own theoretical floor
        assert r["linear_energy_per_coord"] >= r["energy_floor_uniform"] * (1 - 1e-6)
    assert prof["s95"] >= 1
    assert prof["separation_certificate"]["threshold_error"] == 0.0


@pytest.mark.parametrize("variant", ["free", "tied", "softmax"])
def test_optimiser_moves_towards_the_floor_and_never_below(variant):
    out = train_code(d=8, F=32, variant=variant, steps=400, seed=0, record_every=100)
    traj = out["trajectory_ratio_mean_sq"]
    assert traj[-1] < traj[0]
    assert out["ratio_mean_sq"] >= 1.0 - 1e-4
    if variant == "tied":
        assert 0.0 <= out["tight_frame_residual"] <= 2.0


def test_uncalibrated_ablation_can_fall_below_the_raw_floor():
    """Without the diagonal constraint the raw off-diagonal energy is not bounded below.

    The optimiser shrinks the diagonal gains instead of reducing interference. Either the
    calibrated interface still obeys the floor, or the gains have collapsed far enough that
    the calibration is numerically undefined -- which is itself the point of the ablation.
    """
    out = train_code(d=8, F=32, variant="uncalibrated", steps=1500, seed=0, record_every=500)
    assert out["ratio_mean_sq_raw"] < 1.0
    assert out["diag_gain_min_abs"] < 1.0
    if not out["interface_degenerate"]:
        assert out["ratio_mean_sq"] >= 1.0 - 1e-6


def test_toy_model_trains_and_diagnoses():
    model = train_toy_model(d=10, F=20, loss_kind="L2", p=0.05, steps=1500, batch=512, seed=0,
                            log_every=500)
    assert model["final_mse"] < model["zero_predictor_mse"]
    diag = diagnose_model(model, sparsities=[1, 2], trials=20, seed=0, n_fit=256)
    assert set(diag["floor_stats"]) == {"pinv", "wout", "ls"}
    for name, st in diag["floor_stats"].items():
        if "error" not in st:
            assert st["ratio_mean_sq"] >= 1.0 - 1e-6, name
    # The linear side must respect its own energy floor at every sparsity and readout.
    for row in diag["rows"]:
        for name in ("pinv", "wout", "ls"):
            assert row[f"linear_energy_over_floor_{name}"] >= 1.0 - 1e-6, (name, row["s"])
    assert 0.0 <= diag["linearity"]["frac_relu_clipped"] <= 1.0
    assert 0.0 <= diag["linearity"]["decision_agreement_with_linear"] <= 1.0


def test_least_squares_readout_is_the_best_linear_one():
    """The fitted readout must not be beaten by pinv or W_out on the linear criterion."""
    model = train_toy_model(d=12, F=30, loss_kind="L2", p=0.05, steps=800, batch=256, seed=1,
                            log_every=400)
    diag = diagnose_model(model, sparsities=[2], trials=10, seed=0, n_fit=2048)
    row = diag["rows"][0]
    assert row["linear_energy_ls"] <= min(row["linear_energy_pinv"],
                                          row["linear_energy_wout"]) * 1.05


def test_random_code_baseline_is_diagnosable():
    model = random_code_baseline(d=10, F=20, seed=0)
    diag = diagnose_model(model, sparsities=[1, 2], trials=20, seed=0, n_fit=256)
    assert diag["loss_kind"] == "random"
    assert diag["s95_linear_ls"] >= 0
    assert diag["s95_model"] >= 0


def test_configure_cpu_hides_the_gpu_by_default(monkeypatch):
    """A published CPU run must not depend on whether the machine happens to have a GPU."""
    from lrtr import configure_cpu
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    configure_cpu(threads=1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_configure_cpu_can_be_told_to_leave_the_gpu_alone(monkeypatch):
    """The accelerator campaign opts in, and the opt-in must survive this call.

    configure_cpu imports torch, so hiding the device here would hide it for the rest of the
    process however the campaign was invoked -- which is exactly how the accelerator run came
    to report that no CUDA device was visible on a machine where one plainly was.
    """
    from lrtr import configure_cpu
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    configure_cpu(threads=1, allow_gpu=True)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    configure_cpu(threads=1, allow_gpu=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
