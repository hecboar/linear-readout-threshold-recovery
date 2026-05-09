# Linear Readout and Threshold Recovery in Overcomplete Neural Representations

Companion code for the manuscript:

> **Interface Limits for Overcomplete Neural Representations: Linear Readout, Threshold Recovery, and Computation in Superposition**  
> H. Borobia, E. Segu\'i-Mas, G. Tormo-Carb\'o.

This repository reproduces the **synthetic numerical illustrations** used in the manuscript.
The experiments illustrate three phenomena studied theoretically in the paper:

1. The Welch-type floor for unit-diagonal linear readouts.
2. Threshold recovery at quadratic feature load \(F=d^2\).
3. The linear-readout energy floor \(\Omega(s/d)\) on Bernoulli sparse states.

These experiments are sanity checks for the mathematical toy model. They are **not** empirical validation on trained neural networks, do **not** construct a recursive reset module, and do **not** estimate sharp finite-dimensional constants.

## Quickstart

```bash
git clone https://github.com/hectorborobia/linear-readout-threshold-recovery.git
cd linear-readout-threshold-recovery
python -m pip install -r requirements.txt
python synthetic_illustrations.py --out_dir figures --data_out data/synthetic_illustrations_data.npz
```

The command regenerates:

```text
figures/exp1_welch_floor.png
figures/exp2_threshold_recovery.png
figures/exp3_linear_energy.png
figures/combined_capacity_diagram.png
data/synthetic_illustrations_data.npz
```

## Repository layout

```text
linear-readout-threshold-recovery/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── synthetic_illustrations.py
├── figures/
│   ├── exp1_welch_floor.png
│   ├── exp2_threshold_recovery.png
│   ├── exp3_linear_energy.png
│   └── combined_capacity_diagram.png
└── data/
    └── synthetic_illustrations_data.npz
```

## Reproducibility

All experiments use `numpy.random.default_rng` with fixed seeds.
The figures should be numerically reproducible. The script uses a light Monte Carlo configuration intended for fast reproducibility checks; increasing the trial counts gives smoother curves. Minor byte-level differences in PNG files can occur across Matplotlib versions, fonts, or rendering backends.

| Experiment | Seed | Description |
|---|---:|---|
| 1 | 42 | Welch floor for random unit-norm codes |
| 2 | 43 | Threshold recovery at \(F=d^2\) |
| 3 | 44 | Average linear-readout energy |

## Notes on the experiments

- **Experiment 1** samples random unit-norm codes and compares the empirical average squared off-diagonal cross-talk with the Welch floor \((F-d)/(d(F-1))\).
- **Experiment 2** estimates exact threshold-recovery probability for random supports at quadratic load \(F=d^2\). The dotted vertical lines show the heuristic reference scale \(d/(16\log d)\).
- **Experiment 3** computes the average per-coordinate squared error for Bernoulli sparse states and compares it with the reference scale \(E=s/d\).

## Citation

If you use this code, please cite the companion paper. A machine-readable citation template is provided in `CITATION.cff`.

## License

Code is released under the MIT License. See `LICENSE`.
