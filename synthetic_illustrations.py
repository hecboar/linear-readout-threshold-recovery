#!/usr/bin/env python3
"""Regenerate synthetic numerical illustrations for the manuscript.

The experiments are sanity checks for the mathematical toy model only.
They use random unit-vector codes and fixed random seeds for reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def random_unit_code(d: int, f: int, rng: np.random.Generator) -> np.ndarray:
    """Return a d x f matrix with iid random unit-norm columns."""
    x = rng.standard_normal((d, f))
    x /= np.linalg.norm(x, axis=0, keepdims=True)
    return x


def welch_floor(f: int, d: int) -> float:
    return float((f - d) / (d * (f - 1)))


def sample_sphere_dot(norm_x: float, d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample <u, x> for u uniform on S^{d-1} and fixed ||x||=norm_x.

    Uses the exact representation g / sqrt(g^2 + chi2_{d-1}).
    """
    g = rng.standard_normal(n)
    chi = rng.chisquare(d - 1, n)
    return norm_x * g / np.sqrt(g * g + chi)


def experiment_welch(out_dir: Path) -> Dict[str, Any]:
    """Experiment 1: Welch floor for random unit-norm codes.

    Instead of materializing all F x F Gram matrices, we estimate the empirical
    average squared off-diagonal cross-talk by sampling random off-diagonal pairs.
    """
    rng = np.random.default_rng(42)
    ds = np.array([16, 32, 64, 128])
    ratios = np.array([2, 4, 8, 16])
    trials = 15
    pairs_per_trial = 5000

    empirical = np.zeros((len(ds), len(ratios)))
    errors = np.zeros_like(empirical)
    floors = np.zeros_like(empirical)

    plt.figure(figsize=(8.5, 5.2))
    for a, d in enumerate(ds):
        for b, r in enumerate(ratios):
            f = int(r * d)
            vals = []
            for _ in range(trials):
                # For two independent random unit vectors, <u,v> has the same
                # distribution as the first coordinate of a random unit vector.
                dots = sample_sphere_dot(1.0, int(d), pairs_per_trial, rng)
                vals.append(float(np.mean(dots * dots)))
            empirical[a, b] = np.mean(vals)
            errors[a, b] = np.std(vals)
            floors[a, b] = welch_floor(f, int(d))
        plt.errorbar(ratios, empirical[a], yerr=errors[a], marker="o", linewidth=2, label=fr"Empirical, $d={d}$")
        plt.plot(ratios, floors[a], linestyle="--", linewidth=2, label=fr"Welch floor, $d={d}$")

    plt.xscale("log", base=2)
    plt.xlabel(r"$F/d$ (feature load)")
    plt.ylabel("Average squared off-diagonal cross-talk")
    plt.title("Experiment 1: Welch floor for unit-diagonal linear readouts")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "exp1_welch_floor.png", dpi=200)
    plt.close()

    return {"ds": ds, "ratios": ratios, "empirical": empirical, "errors": errors, "floors": floors}


def threshold_recovery_trial(d: int, f: int, s: int, rng: np.random.Generator) -> bool:
    """One random-code, random-support threshold recovery trial.

    Active code vectors are sampled explicitly. Inactive scores are sampled
    from their exact conditional distribution given the active superposition
    vector x, avoiding construction of the full d x F code matrix.
    """
    active = random_unit_code(d, s, rng)
    x = np.sum(active, axis=1)
    norm_x = float(np.linalg.norm(x))

    active_scores = active.T @ x
    if np.any(active_scores < 0.5):
        return False

    inactive_scores = sample_sphere_dot(norm_x, d, f - s, rng)
    return bool(np.all(inactive_scores < 0.5))


def experiment_threshold(out_dir: Path) -> Dict[str, Any]:
    rng = np.random.default_rng(43)
    ds = np.array([64, 128])
    sparsities = np.arange(1, 8)
    trials = 30

    probs = np.zeros((len(ds), len(sparsities)))
    refs = np.zeros(len(ds))

    plt.figure(figsize=(8.8, 5.2))
    for a, d in enumerate(ds):
        f = int(d * d)
        refs[a] = d / (16 * np.log(d))
        for b, s in enumerate(sparsities):
            successes = sum(threshold_recovery_trial(int(d), f, int(s), rng) for _ in range(trials))
            probs[a, b] = successes / trials
        plt.plot(sparsities, probs[a], marker="o", linewidth=2, label=fr"$d={d}, F={f}$")
        plt.axvline(refs[a], linestyle=":", alpha=0.6)
        plt.text(refs[a] + 0.02, 0.7, fr"$d/(16\log d)\approx {refs[a]:.2f}$", rotation=90, va="center")

    plt.xlabel(r"Sparsity $s$ (support size)")
    plt.ylabel(r"Empirical $\mathbb{P}$(exact recovery)")
    plt.title(r"Experiment 2: threshold recovery at quadratic load $F=d^2$")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exp2_threshold_recovery.png", dpi=200)
    plt.close()

    return {"ds": ds, "sparsities": sparsities, "probs": probs, "refs": refs, "trials": trials}


def experiment_linear_energy(out_dir: Path) -> Dict[str, Any]:
    rng = np.random.default_rng(44)
    ds = np.array([32, 64, 128])
    ratio = 8
    s_over_d = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2], dtype=float)
    trials = 5

    values = np.zeros((len(ds), len(s_over_d)))

    plt.figure(figsize=(8.8, 5.2))
    for a, d in enumerate(ds):
        f = int(ratio * d)
        for b, sd in enumerate(s_over_d):
            s = max(1, int(round(sd * d)))
            es = []
            for _ in range(trials):
                psi = random_unit_code(int(d), f, rng)
                gram = psi.T @ psi
                a_mat = gram - np.eye(f)
                p = s / f
                # E ||A b||^2 / F = [p(1-p)||A||_F^2 + p^2 ||A 1||^2] / F
                e = (p * (1 - p) * np.sum(a_mat * a_mat) + p * p * np.sum((a_mat @ np.ones(f)) ** 2)) / f
                es.append(e)
            values[a, b] = np.mean(es)
        plt.plot(s_over_d, values[a], marker="o", linewidth=2, label=fr"Empirical, $d={d}, F={f}$")

    xs = np.logspace(np.log10(s_over_d.min()), np.log10(s_over_d.max()), 200)
    plt.plot(xs, xs, "--", color="black", linewidth=2, label=r"Reference: $E=s/d$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$s/d$")
    plt.ylabel(r"Average per-coordinate squared error $E$")
    plt.title("Experiment 3: linear-readout energy floor")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exp3_linear_energy.png", dpi=200)
    plt.close()

    return {"ds": ds, "s_over_d": s_over_d, "values": values, "ratio": ratio, "trials": trials}


def combined_figure(out_dir: Path) -> None:
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    names = ["exp1_welch_floor.png", "exp2_threshold_recovery.png", "exp3_linear_energy.png"]
    titles = ["(a) Welch floor", r"(b) Threshold recovery at $F=d^2$", "(c) Linear readout energy"]
    for ax, name, title in zip(axes, names, titles):
        ax.imshow(mpimg.imread(out_dir / name))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "combined_capacity_diagram.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic numerical illustrations for the companion manuscript.")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"), help="Directory for PNG figures.")
    parser.add_argument("--data_out", type=Path, default=Path("data/synthetic_illustrations_data.npz"), help="Output .npz file with numerical arrays.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_out.parent.mkdir(parents=True, exist_ok=True)

    exp1 = experiment_welch(args.out_dir)
    exp2 = experiment_threshold(args.out_dir)
    exp3 = experiment_linear_energy(args.out_dir)
    combined_figure(args.out_dir)

    np.savez(
        args.data_out,
        exp1_ds=exp1["ds"], exp1_ratios=exp1["ratios"], exp1_empirical=exp1["empirical"], exp1_errors=exp1["errors"], exp1_floors=exp1["floors"],
        exp2_ds=exp2["ds"], exp2_sparsities=exp2["sparsities"], exp2_probs=exp2["probs"], exp2_refs=exp2["refs"], exp2_trials=exp2["trials"],
        exp3_ds=exp3["ds"], exp3_s_over_d=exp3["s_over_d"], exp3_values=exp3["values"], exp3_ratio=exp3["ratio"], exp3_trials=exp3["trials"],
    )

    print(f"Wrote figures to {args.out_dir}")
    print(f"Wrote numerical arrays to {args.data_out}")


if __name__ == "__main__":
    main()
