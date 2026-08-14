# Stage C — GO/NO-GO report

30 models, widths [400].

## Per width

| d | loss | n | s95(model) | s95(best probe) | R_geom | R_readout(wout) | kappa_min |
|---|---|---|---|---|---|---|---|
| 400 | L2 | 10 | 0.0 | 0.0 | 32.313 | 1.0835 | 1.07 |
| 400 | L4 | 10 | 13.0 | 11.0 | 1.000 | 1.0158 | 10.56 |
| 400 | random | 10 | n/a | 7.0 | 1.002 | 1 by construction | 8.92 |

## Gates

- **do not read the next line as a decoder comparison.** It scores the network at a
  fixed `theta` against the best of eighteen probe configurations, two thirds of which
  tune their threshold on validation and two thirds of which read the *pre*-ReLU state
  the network's output layer never sees. Three asymmetries, all favouring the probe.
  The matched comparison -- same input, same threshold treatment, paired within seed --
  is `scripts/primary_comparison.py`, and it reports a tie. See docs/known_defects.md
  KD2 and KD4.
- unmatched, retained for continuity: the network beats the best affine probe on **50%** of trained models
- L2/L4 geometry separated at every width: **True**
- E3 arrow held: **True**

The L2/L4 separation replicates at every width, so that hook survives. **That is not sufficient to justify the next stage**, and this report cannot tell you whether it is: the gate above compares an untuned decoder against the best of a tuned family, so a 0% there is uninformative about whether the network out-decodes an affine probe. Read the matched comparison first. When it was run for Stage A it returned a tie, which changed the paper's thesis and made most of Stage B's design -- robustness of a nonlinear advantage -- the wrong question.
