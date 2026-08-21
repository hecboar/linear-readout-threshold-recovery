"""The paper's central empirical contrast, guarded by a test.

The manuscript's largest and most-replicated finding is that the training loss decides whether a
superposed code supports linear readout: `L2` drives `R_geom` to 18-32 and collapses the affine
frontier to 1, while `L4` holds `R_geom` at 1.000. A reproduction audit found that claim was not
protected by anything. Redefining the loss exponent so that `L4` silently trains as `L2` --

    exponent = 2 if loss_kind == "L2" else 4     ->     exponent = 2

-- left all 350 tests passing. The mutation is one character per site and there are two sites,
`train_toy_model` and `train_toy_models_batched`, so a test that covers only one of them is not
enough.

Two tests, deliberately at different levels.

`test_exponent_is_honoured_*` is the one that catches the mutation: from an identical seed the two
losses must produce different weights, because if the exponent does not depend on `loss_kind` the
trajectories are identical. Milliseconds, exact, and it fails on the mutation in either code path.

`test_l2_excess_exceeds_l4` tests the scientific claim rather than the implementation. It cannot be
run at the manuscript's scale in a unit test, and the collapse itself does not appear at toy scale --
at `d=32` the `L2` excess over the rank-trace floor is about 0.013 against `L4`'s 0.003, not the
factor of twenty the campaigns measure. What survives at this scale is the *ordering*, and the
threshold is set from a calibration over four seeds per loss (`L2` 0.0116-0.0155, `L4`
0.0024-0.0030, no overlap, ratio 3.85) with a factor of two left as margin. So this test asserts the
direction of the finding, and says so, rather than pretending to reproduce its magnitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrtr.analog_optimum import code_specific_floor  # noqa: E402
from lrtr.codes import welch_floor  # noqa: E402
from lrtr.toymodel import train_toy_model, train_toy_models_batched  # noqa: E402

TINY = dict(d=8, F=32, p=0.1, steps=3, batch=32)
CLAIM = dict(d=32, F=128, p=0.05, steps=800, batch=256)


def _excess(W_in) -> float:
    """R_geom - 1: by the trace identity this is the leverage dispersion of the code."""
    Phi = np.asarray(W_in, dtype=np.float64)
    Phi = Phi / np.maximum(np.linalg.norm(Phi, axis=0, keepdims=True), 1e-12)
    d, F = Phi.shape
    return code_specific_floor(Phi) / welch_floor(F, d) - 1.0


def test_exponent_is_honoured_single() -> None:
    a = train_toy_model(loss_kind="L2", seed=0, log_every=10**9, **TINY)["W_in"]
    b = train_toy_model(loss_kind="L4", seed=0, log_every=10**9, **TINY)["W_in"]
    assert not np.allclose(np.asarray(a), np.asarray(b)), (
        "L2 and L4 produced identical weights from the same seed: the loss exponent is not "
        "being taken from loss_kind in train_toy_model")


def test_exponent_is_honoured_batched() -> None:
    kw = dict(seeds=[0], log_every=10**9, **TINY)
    a = train_toy_models_batched(loss_kind="L2", **kw)[0]["W_in"]
    b = train_toy_models_batched(loss_kind="L4", **kw)[0]["W_in"]
    assert not np.allclose(np.asarray(a), np.asarray(b)), (
        "L2 and L4 produced identical weights from the same seed: the loss exponent is not "
        "being taken from loss_kind in train_toy_models_batched")


def test_l2_excess_exceeds_l4() -> None:
    """The direction of the paper's central contrast, at a scale a unit test can afford."""
    ex = {k: [_excess(train_toy_model(loss_kind=k, seed=s, log_every=10**9, **CLAIM)["W_in"])
              for s in (0, 1)] for k in ("L2", "L4")}
    assert min(ex["L2"]) > 2.0 * max(ex["L4"]), (
        f"L2 should sit further from the rank-trace floor than L4; got L2={ex['L2']} "
        f"L4={ex['L4']}. Calibrated margin was 3.85x over four seeds per loss.")
