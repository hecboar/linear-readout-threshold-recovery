"""State distributions for the diagnosis, and the one parameter the affine level sees.

The published trained-network evidence evaluates Boolean states while the networks train on
`x = mask * U[-1,1]`. The manuscript discloses that, which does not make it defensible: it leaves
every trained-network claim open to being an artefact of the audit distribution.

What the theory says about this, and it is unusually clean. The two lower levels of the hierarchy
read *different summaries* of the same state law:

* the **analog** level needs the full second moment `C = E[a a^T]` (G4);
* the **affine** level needs only the **minimum detectable amplitude** `alpha`, because the
  amplitudes wash out of the convex hulls entirely (E2). Two laws with the same support and the
  same `alpha` have the same affine frontier whatever else differs between them.

So the six-distribution sweep the external review asked for collapses: for the affine level it is
a sweep over one scalar, and the families below are chosen to span it rather than to enumerate
distributions for their own sake.

**Truncated versus untruncated, and why the distinction is not pedantry.** A family with a
minimum amplitude `alpha > 0` admits exact recovery as a meaningful metric and the frontier at
that `alpha` applies to it. A family whose amplitudes reach zero -- the native training law is one
-- does not: a coordinate drawn at `1e-6` is active by definition and undetectable in principle,
so an exact-recovery rate for it measures the draw and not the code. Those families are scored on
detection quality instead, and this module records which is which so the campaign cannot quietly
mix them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .splits import SplitBundle, StateSplit

__all__ = ["StateFamily", "FAMILIES", "family", "apply_family", "apply_family_to_bundle"]


@dataclass(frozen=True)
class StateFamily:
    """One law for the amplitudes of the active coordinates.

    ``alpha`` is the smallest attainable `|amplitude|`, which is the whole dependence of the
    affine frontier on this family. ``None`` means the amplitudes reach zero, in which case
    ``exact_recovery_meaningful`` is ``False`` and no frontier statement applies.

    ``amplitude`` returns an array shaped like the support block; ``None`` means all ones.
    ``rep_noise_sigma`` adds isotropic Gaussian noise to the `d`-dimensional representation,
    which is the perturbation the robust affine margin already certifies a tolerance for, so the
    sweep over it is a *test of that bound* rather than another distribution.
    """

    name: str
    alpha: Optional[float]
    exact_recovery_meaningful: bool
    amplitude: Optional[Callable[[np.random.Generator, tuple], np.ndarray]] = None
    rep_noise_sigma: float = 0.0
    note: str = ""


def _uniform_in(lo: float, hi: float) -> Callable[[np.random.Generator, tuple], np.ndarray]:
    return lambda rng, shape: rng.uniform(lo, hi, size=shape)


def _signed_unit(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=shape)


def _signed_in(lo: float, hi: float) -> Callable[[np.random.Generator, tuple], np.ndarray]:
    def f(rng, shape):
        return rng.choice(np.array([-1.0, 1.0]), size=shape) * rng.uniform(lo, hi, size=shape)
    return f


FAMILIES: Dict[str, StateFamily] = {
    "boolean": StateFamily(
        name="boolean", alpha=1.0, exact_recovery_meaningful=True, amplitude=None,
        note="the model of the theory; every active coordinate is exactly 1"),
    "signed": StateFamily(
        name="signed", alpha=1.0, exact_recovery_meaningful=True, amplitude=_signed_unit,
        note="amplitudes in {-1, +1}: same alpha as boolean, so the same affine frontier, but a "
             "harder decision problem because interference can cancel"),
    "amp_0.5": StateFamily(
        name="amp_0.5", alpha=0.5, exact_recovery_meaningful=True,
        amplitude=_uniform_in(0.5, 1.0),
        note="positive continuous amplitudes with a floor at 0.5"),
    "amp_0.25": StateFamily(
        name="amp_0.25", alpha=0.25, exact_recovery_meaningful=True,
        amplitude=_uniform_in(0.25, 1.0),
        note="the same law with a lower floor: the frontier must not improve"),
    "signed_amp_0.5": StateFamily(
        name="signed_amp_0.5", alpha=0.5, exact_recovery_meaningful=True,
        amplitude=_signed_in(0.5, 1.0),
        note="signed continuous amplitudes, floor 0.5"),
    "native": StateFamily(
        name="native", alpha=None, exact_recovery_meaningful=False,
        amplitude=_signed_in(0.0, 1.0),
        note="the training law, U[-1,1] on the support. Amplitudes reach zero, so exact recovery "
             "is not a meaningful metric here and detection quality is reported instead"),
    "boolean_noise_0.05": StateFamily(
        name="boolean_noise_0.05", alpha=1.0, exact_recovery_meaningful=True, amplitude=None,
        rep_noise_sigma=0.05,
        note="Boolean states with isotropic noise on the representation: a test of the robust "
             "affine margin, which certifies tolerance to exactly this perturbation"),
    "boolean_noise_0.15": StateFamily(
        name="boolean_noise_0.15", alpha=1.0, exact_recovery_meaningful=True, amplitude=None,
        rep_noise_sigma=0.15,
        note="the same at a larger radius, to locate where the certified tolerance is exceeded"),
}


def family(name: str) -> StateFamily:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown state family {name!r}; known: {sorted(FAMILIES)}") from None


def apply_family(split: StateSplit, fam: StateFamily, d: int, seed: int) -> StateSplit:
    """Attach this family's amplitudes and representation noise to an existing split.

    The **supports are untouched**. Every family therefore evaluates the same states, differing
    only in their amplitudes, which makes the comparison across distributions paired: a change in
    recovery cannot be attributed to having drawn easier supports.
    """
    rng = np.random.default_rng(seed)
    amps = None
    if fam.amplitude is not None:
        amps = fam.amplitude(rng, split.supports.shape)
    noise = None
    if fam.rep_noise_sigma > 0.0:
        noise = fam.rep_noise_sigma * rng.standard_normal((len(split), d))
    return StateSplit(supports=split.supports, sparsity=split.sparsity, F=split.F,
                      amplitudes=amps, rep_noise=noise)


def apply_family_to_bundle(bundle: SplitBundle, fam: StateFamily, d: int,
                           seed: int) -> SplitBundle:
    """The same bundle under a different amplitude law, with disjointness inherited.

    Each split gets its own noise and amplitude stream, keyed off `seed`, so the draw is
    reproducible and the locked test set stays locked: nothing about the family changes which
    supports live in which split.
    """
    off = 0

    def go(sp: StateSplit) -> StateSplit:
        nonlocal off
        off += 1
        return apply_family(sp, fam, d, seed + 1000 * off)

    return SplitBundle(
        train=go(bundle.train), val=go(bundle.val),
        train_by_s={s: go(v) for s, v in bundle.train_by_s.items()},
        val_by_s={s: go(v) for s, v in bundle.val_by_s.items()},
        test_by_s={s: go(v) for s, v in bundle.test_by_s.items()},
        sparsities=bundle.sparsities, F=bundle.F, seed=bundle.seed,
        n_redrawn=bundle.n_redrawn, exhausted=bundle.exhausted)


def truncated_families(names: List[str]) -> List[str]:
    """Those with a positive minimum amplitude, i.e. the ones a frontier statement covers."""
    return [n for n in names if family(n).exact_recovery_meaningful]
