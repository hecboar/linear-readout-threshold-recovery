# Stage A — GO/NO-GO report

180 models, widths [50, 100, 200].

## Per width

| d | loss | n | s95(model) | s95(best probe) | R_geom | R_readout(wout) | kappa_min |
|---|---|---|---|---|---|---|---|
| 50 | L2 | 20 | 0.0 | 0.0 | 18.564 | 1.0045 | 1.00 |
| 50 | L4 | 20 | 3.0 | 4.0 | 1.001 | 1.0263 | 4.49 |
| 50 | random | 20 | n/a | 3.0 | 1.019 | 1 by construction | 3.18 |
| 100 | L2 | 20 | 0.0 | 0.0 | 24.978 | 1.0122 | 1.00 |
| 100 | L4 | 20 | 4.0 | 5.0 | 1.000 | 1.0229 | 6.04 |
| 100 | random | 20 | n/a | 4.0 | 1.010 | 1 by construction | 4.41 |
| 200 | L2 | 20 | 0.0 | 0.0 | 29.873 | 1.0323 | 1.02 |
| 200 | L4 | 20 | 7.0 | 8.0 | 1.000 | 1.0185 | 8.06 |
| 200 | random | 20 | n/a | 6.0 | 1.005 | 1 by construction | 6.22 |

## Gates

- **do not read the next line as a decoder comparison.** It scores the network at a
  fixed `theta` against the best of eighteen probe configurations, two thirds of which
  tune their threshold on validation and two thirds of which read the *pre*-ReLU state
  the network's output layer never sees. Three asymmetries, all favouring the probe.
  The matched comparison -- same input, same threshold treatment, paired within seed --
  is `scripts/primary_comparison.py`, and it reports a tie. See docs/known_defects.md
  KD2 and KD4.
- unmatched, retained for continuity: the network beats the best affine probe on **0%** of trained models
- L2/L4 geometry separated at every width: **True**
- E3 arrow held: **True**

The empirical hook survives this stage, so spending the next one is justified.
