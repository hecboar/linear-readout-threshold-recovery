"""Threshold recovery: the streaming implementation must agree with the dense one."""
from __future__ import annotations

import numpy as np
import pytest

from lrtr.diagnostic import (
    tied_energy_moments,
    tied_energy_moments_streaming,
    tied_linear_energy,
)
from lrtr.interface import linear_energy_uniform
from lrtr.stats import fit_c_over_log, wilson_interval
from lrtr.threshold import (
    _block_columns,
    recovery_trial_dense,
    recovery_trial_streaming,
    s95_from_curve,
    threshold_decode,
)


def _dense_code_from_blocks(master: int, trial: int, d: int, F: int, block: int) -> np.ndarray:
    """Reconstruct the streamed code explicitly, for cross-checking."""
    cols = []
    for blk in range((F + block - 1) // block):
        width = min(block, F - blk * block)
        cols.append(_block_columns(master, trial, blk, d, block)[:width])
    return np.vstack(cols).T  # (d, F)


@pytest.mark.parametrize("trial", range(6))
def test_streaming_matches_explicit_evaluation(trial):
    """The block-streamed decision must equal a direct dense evaluation of the same code."""
    master, d, F, s_max, block = 20260807, 24, 600, 5, 128
    Phi = _dense_code_from_blocks(master, trial, d, F, block)
    assert np.allclose(np.linalg.norm(Phi, axis=0), 1.0, atol=1e-5)

    rng = np.random.default_rng(np.random.SeedSequence(entropy=master,
                                                       spawn_key=(trial, 2 ** 62)))
    S = rng.choice(F, size=s_max, replace=False)

    expected = []
    for s in range(1, s_max + 1):
        x = Phi[:, S[:s]].sum(axis=1)
        z = Phi.T @ x
        b = np.zeros(F, dtype=np.int8)
        b[S[:s]] = 1
        expected.append(bool(np.array_equal(threshold_decode(z, 0.5), b)))

    got = recovery_trial_streaming(master, trial, d, F, s_max, block=block)
    assert list(got) == expected


def test_streaming_handles_a_partial_final_block():
    """F not a multiple of the block size must not corrupt the trailing block."""
    master, d, F, s_max, block = 7, 16, 330, 3, 128
    Phi = _dense_code_from_blocks(master, 0, d, F, block)
    rng = np.random.default_rng(np.random.SeedSequence(entropy=master, spawn_key=(0, 2 ** 62)))
    S = rng.choice(F, size=s_max, replace=False)
    expected = []
    for s in range(1, s_max + 1):
        x = Phi[:, S[:s]].sum(axis=1)
        b = np.zeros(F, dtype=np.int8)
        b[S[:s]] = 1
        expected.append(bool(np.array_equal(threshold_decode(Phi.T @ x, 0.5), b)))
    assert list(recovery_trial_streaming(master, 0, d, F, s_max, block=block)) == expected


def test_streaming_is_deterministic():
    a = recovery_trial_streaming(99, 3, 20, 500, 4, block=256)
    b = recovery_trial_streaming(99, 3, 20, 500, 4, block=256)
    assert list(a) == list(b)


def test_recovery_probability_agrees_between_implementations():
    """Dense and streaming estimators must agree statistically at a fixed (d, F, s)."""
    d, F, s, n = 32, 1024, 2, 200
    rng = np.random.default_rng(5)
    dense = np.mean([recovery_trial_dense(d, F, s, rng) for _ in range(n)])
    stream = np.mean([recovery_trial_streaming(4242, t, d, F, s, block=512)[s - 1]
                      for t in range(n)])
    assert abs(dense - stream) < 0.12


def test_recovery_is_certain_when_the_coherence_condition_holds():
    """The sufficient condition ``s*mu < 1/2`` must imply success, trial by trial.

    Note this is *not* automatic at small ``d``: at ``d = 32`` and ``F = 2048`` the coherence
    of a random code exceeds ``1/2``, and recovery does fail at ``s = 1``. The test checks the
    implication, not the premise.
    """
    from lrtr.codes import coherence

    master, d, F, block = 1234, 128, 4096, 512
    checked = 0
    for t in range(10):
        Phi = _dense_code_from_blocks(master, t, d, F, block).astype(np.float64)
        mu = coherence(Phi)
        ok = recovery_trial_streaming(master, t, d, F, 1, block=block)[0]
        if mu < 0.5:
            assert bool(ok), f"coherence {mu:.3f} < 1/2 but recovery failed"
            checked += 1
    assert checked > 0, "no trial satisfied the coherence premise; test is vacuous"


def test_recovery_degrades_monotonically_in_s_on_average():
    d, F, s_max, n = 32, 1024, 6, 120
    ok = np.array([recovery_trial_streaming(777, t, d, F, s_max, block=512) for t in range(n)])
    probs = ok.mean(axis=0)
    assert np.all(np.diff(probs) <= 1e-9)


@pytest.mark.parametrize("d,F", [(16, 400), (24, 900)])
def test_tied_moments_streaming_matches_dense(d, F):
    block = 128
    Phi = _dense_code_from_blocks(11, 0, d, F, block)
    mom_s = tied_energy_moments_streaming(11, 0, d, F, block=block)
    mom_d = tied_energy_moments(Phi.astype(np.float64))
    assert mom_s.frob_sq_A == pytest.approx(mom_d.frob_sq_A, rel=1e-4)
    assert mom_s.sum_A1_sq == pytest.approx(mom_d.sum_A1_sq, rel=1e-3)


def test_tied_linear_energy_matches_explicit_matrix():
    """The closed form from (S, v) must equal the direct computation with the full A."""
    rng = np.random.default_rng(2)
    d, F = 12, 200
    Phi = rng.standard_normal((d, F))
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    A = Phi.T @ Phi - np.eye(F)
    mom = tied_energy_moments(Phi)
    for s in (1, 3, 9):
        assert tied_linear_energy(mom, s) == pytest.approx(linear_energy_uniform(A, s), rel=1e-9)


def test_s95_requires_a_contiguous_prefix():
    assert s95_from_curve([1, 2, 3, 4], [1.0, 0.98, 0.5, 0.99]) == 2
    assert s95_from_curve([1, 2, 3], [0.9, 1.0, 1.0]) == 0
    assert s95_from_curve([1, 2, 3], [1.0, 1.0, 1.0]) == 3


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo, hi = wilson_interval(100, 100)
    assert hi == 1.0 and lo > 0.9
    lo, hi = wilson_interval(0, 50)
    assert lo < 1e-12 and hi < 0.1


def test_fit_recovers_a_known_constant():
    ds = [128, 256, 512, 1024]
    c_true = 0.37
    ys = [c_true * d / np.log(d) for d in ds]
    fit = fit_c_over_log(ds, ys)
    assert fit["c"] == pytest.approx(c_true, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
