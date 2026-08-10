"""Disjoint state splits for probe fitting, selection and locked evaluation.

The Phase-0 audit found that the published protocol drew its probe-fitting states and its
evaluation states from the same generator, sequentially. There is no systematic leakage in that,
but at `s = 3` about one evaluation state in forty was also a fitting state, and "unlikely to
collide" is not a defensible design when the comparison it feeds is the paper's contested claim.

So disjointness here is **enforced, not probabilistic**: every support drawn is checked against
the set of supports already issued, and duplicates are redrawn. That makes the three splits
provably disjoint rather than probably disjoint, and `tests/test_probes.py` asserts it.

States are stored as **support index arrays**, never as the `(F, n)` indicator matrix. At
`F = 800` and `n = 100 000` the indicator matrix would be 320 MB in single precision, while the
`(n, d)` representation the probes actually consume is 160 MB and the `(d+1, F)` normal equations
are negligible. Every consumer here works from indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = ["StateSplit", "SplitBundle", "draw_supports", "make_state_splits", "representations",
           "one_hot_targets"]


@dataclass(frozen=True)
class StateSplit:
    """A set of Boolean states, as `(n, s)` support indices plus the sparsity of each."""

    supports: np.ndarray          # (n, s_max) int; row k uses the first sparsity[k] entries
    sparsity: np.ndarray          # (n,) int
    F: int

    def __len__(self) -> int:
        return int(self.supports.shape[0])

    def of_sparsity(self, s: int) -> "StateSplit":
        m = self.sparsity == s
        return StateSplit(supports=self.supports[m, :s], sparsity=self.sparsity[m], F=self.F)

    def keys(self) -> List[Tuple[int, ...]]:
        """Canonical hashable form of each state, for disjointness checks."""
        return [tuple(sorted(self.supports[k, :self.sparsity[k]].tolist()))
                for k in range(len(self))]


@dataclass
class SplitBundle:
    """Three disjoint splits: fit, select, and a locked test set held per sparsity.

    ``train`` and ``val`` are mixtures over the sparsity grid, which is what a *global* affine
    interface must be fitted and tuned on. ``train_by_s`` and ``val_by_s`` are per-sparsity and
    exist for the per-sparsity oracle. ``test_by_s`` is the locked set: nothing may be selected
    on it, ever.
    """

    train: StateSplit
    val: StateSplit
    train_by_s: Dict[int, StateSplit]
    val_by_s: Dict[int, StateSplit]
    test_by_s: Dict[int, StateSplit]
    sparsities: List[int]
    F: int
    seed: int
    n_redrawn: int = 0
    exhausted: List[int] = field(default_factory=list)

    def all_keys(self) -> Dict[str, Set[Tuple[int, ...]]]:
        out = {"train": set(self.train.keys()), "val": set(self.val.keys())}
        for name, d in (("train_by_s", self.train_by_s), ("val_by_s", self.val_by_s),
                        ("test_by_s", self.test_by_s)):
            acc: Set[Tuple[int, ...]] = set()
            for split in d.values():
                acc |= set(split.keys())
            out[name] = acc
        return out


def draw_supports(F: int, s: int, n: int, rng: np.random.Generator,
                  seen: Set[Tuple[int, ...]], max_tries: int = 50) -> Tuple[np.ndarray, int, bool]:
    """Draw `n` distinct size-`s` supports not already in `seen`, adding them to it.

    Returns the `(n, s)` array, how many draws were discarded as duplicates, and whether the
    request was met. Sampling is by partial-sorting uniform keys, which is uniform over size-`s`
    subsets and fast; only the duplicate filter is sequential.

    Exhaustion is reported rather than hidden: when `C(F, s)` is small relative to the total
    request the space genuinely runs out, and a silently short split would misstate every
    interval computed from it.
    """
    if not 1 <= s <= F:
        raise ValueError(f"need 1 <= s <= F, got s={s}, F={F}")
    out = np.empty((n, s), dtype=np.int32)
    filled = redrawn = 0
    for _ in range(max_tries):
        if filled >= n:
            break
        want = n - filled
        keys = rng.random((max(want * 2, 64), F))
        cand = np.sort(np.argpartition(keys, s - 1, axis=1)[:, :s], axis=1)
        for row in cand:
            key = tuple(row.tolist())
            if key in seen:
                redrawn += 1
                continue
            seen.add(key)
            out[filled] = row
            filled += 1
            if filled >= n:
                break
    return out[:filled], redrawn, filled >= n


def make_state_splits(F: int, sparsities: Sequence[int], n_train: int, n_val: int, n_test: int,
                      seed: int, n_train_per_s: Optional[int] = None,
                      n_val_per_s: Optional[int] = None) -> SplitBundle:
    """Build the three disjoint splits. Every state in the bundle is distinct from every other.

    The locked test set is served **first**, capped at half the available space so the other
    splits always have room. That priority is deliberate: a short evaluation set invalidates
    every confidence interval computed from it, whereas a short fitting set merely costs
    accuracy. Whenever the space cannot meet the full request the sparsity is listed in
    ``exhausted`` rather than the shortfall being passed on silently.
    """
    from math import comb

    sparsities = [int(s) for s in sparsities]
    rng = np.random.default_rng(seed)
    seen: Set[Tuple[int, ...]] = set()
    redrawn_total = 0
    exhausted: List[int] = []

    def grab(s: int, n: int) -> StateSplit:
        nonlocal redrawn_total
        sup, red, ok = draw_supports(F, s, max(n, 0), rng, seen)
        redrawn_total += red
        if not ok:
            exhausted.append(s)
        return StateSplit(supports=sup, sparsity=np.full(sup.shape[0], s, dtype=np.int32), F=F)

    per_s_train = n_train_per_s if n_train_per_s is not None else max(1, n_train // len(sparsities))
    per_s_val = n_val_per_s if n_val_per_s is not None else max(1, n_val // len(sparsities))
    mix_train = max(1, n_train // len(sparsities))
    mix_val = max(1, n_val // len(sparsities))

    # Budget the five consumers against the size of the state space. At `s = 1` only `F` states
    # exist, so any sizeable request exhausts it, and a naive first-come allocation drains the
    # space on the test set and hands a *zero-length* split to whatever comes last -- an empty
    # validation split then fails deep inside a quantile call with an unreadable IndexError.
    #
    # The policy: the locked test set is served first, because a short evaluation set invalidates
    # every interval computed from it while a short fitting set merely costs accuracy. It is
    # capped at half the space so the other four always have room, and the remainder is shared
    # proportionally with a floor of one state each. Any shortfall is recorded in `exhausted`.
    want = {"test": n_test, "train_s": per_s_train, "val_s": per_s_val,
            "mix_train": mix_train, "mix_val": mix_val}
    others = [k for k in want if k != "test"]
    quota: Dict[int, Dict[str, int]] = {}
    for s in sparsities:
        avail = comb(F, s) if s <= 40 else 2 ** 62
        if sum(want.values()) <= avail:
            quota[s] = dict(want)
            continue
        exhausted.append(s)
        q = {"test": min(want["test"], max(1, avail // 2))}
        left = avail - q["test"]
        denom = sum(want[k] for k in others) or 1
        for k in others:
            q[k] = max(1, int(want[k] * left / denom))
        while sum(q.values()) > avail:                      # trim the largest until it fits
            k = max(others, key=lambda kk: q[kk])
            if q[k] <= 1:
                break
            q[k] -= 1
        quota[s] = q

    test_by_s = {s: grab(s, quota[s]["test"]) for s in sparsities}
    train_by_s = {s: grab(s, quota[s]["train_s"]) for s in sparsities}
    val_by_s = {s: grab(s, quota[s]["val_s"]) for s in sparsities}

    def mixture(key: str) -> StateSplit:
        parts = [grab(s, quota[s][key]) for s in sparsities]
        s_max = max(sparsities)
        sup = np.zeros((sum(len(p) for p in parts), s_max), dtype=np.int32)
        spa = np.empty(sup.shape[0], dtype=np.int32)
        at = 0
        for p in parts:
            k, w = p.supports.shape
            sup[at:at + k, :w] = p.supports
            spa[at:at + k] = p.sparsity
            at += k
        order = rng.permutation(at)
        return StateSplit(supports=sup[order], sparsity=spa[order], F=F)

    train = mixture("mix_train")
    val = mixture("mix_val")

    empty = [name for name, split in (("train", train), ("val", val)) if len(split) == 0]
    empty += [f"{name}[s={s}]" for name, dd in (("train_by_s", train_by_s),
                                                ("val_by_s", val_by_s),
                                                ("test_by_s", test_by_s))
              for s, split in dd.items() if len(split) == 0]
    if empty:
        raise ValueError(
            f"the state space is too small for this split budget: {', '.join(empty)} came out "
            f"empty at F={F}, sparsities={sparsities}. The binding constraint is usually the "
            f"smallest sparsity, where only C(F, s) states exist. Reduce n_test / n_train / "
            f"n_val, drop the smallest sparsity, or raise F.")

    return SplitBundle(train=train, val=val, train_by_s=train_by_s, val_by_s=val_by_s,
                       test_by_s=test_by_s, sparsities=sparsities, F=F, seed=seed,
                       n_redrawn=redrawn_total, exhausted=sorted(set(exhausted)))


def representations(W_in: np.ndarray, split: StateSplit, post_relu: bool) -> np.ndarray:
    """`(n, d)` representations of the states, built from support indices.

    `pre` is the linear representation `Phi 1_S`; `post` is the hidden activation
    `ReLU(Phi 1_S)`. The `(F, n)` indicator matrix is never formed: each of the `s_max` support
    positions contributes one gather-and-add of shape `(n, d)`.
    """
    d = W_in.shape[0]
    n, s_max = split.supports.shape
    R = np.zeros((n, d), dtype=np.float64)
    for j in range(s_max):
        active = split.sparsity > j
        if not active.any():
            break
        R[active] += W_in[:, split.supports[active, j]].T
    return np.maximum(R, 0.0) if post_relu else R


def one_hot_targets(split: StateSplit) -> np.ndarray:
    """`(n, F)` Boolean targets. Only call this on a batch: the full matrix is large."""
    n = len(split)
    Y = np.zeros((n, split.F), dtype=np.float64)
    for j in range(split.supports.shape[1]):
        active = split.sparsity > j
        if not active.any():
            break
        Y[np.nonzero(active)[0], split.supports[active, j]] = 1.0
    return Y
