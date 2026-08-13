# External adversarial audit, 2026-08-12 — what verified and what did not

The audit was run on `internal/gpt_pro_audit_prompt.md`. Its recommendation was reject-and-resubmit
or major structural revision. Every checkable claim was verified independently before being
accepted; the reproductions are one-liners against the committed weights and are recorded below so
they can be rerun.

## Confirmed — the audit is right

### A1. `thm:arrow` is false as stated

Claimed: `kappa_i <= 1 + sqrt((F-1) h_i / (1 - h_i))` for every `h_i < 1`.

Counterexample, unit-norm columns, `d=2`, `F=3`:
`phi_1 = (1,0)`, `phi_2 = (1/4, sqrt(15)/4)`, `phi_3 = (1/4, -sqrt(15)/4)`.

Measured: `h_1 = 0.888889`, bound `= 5.0000`, and the solver returns `kappa_1 = inf` — the LP is
infeasible because the unique representation is `z = (2, 2)`, which violates the box. So
`inf <= 5` is asserted and false.

**Root cause, and it is in our own proof.** The proof takes the minimum-`l2` witness and needs
`||z||_inf <= 1` for it to be admissible. Our text says the box "is satisfied for free wherever the
bound is informative: `h_i <= 1/2` gives `||z||_2 <= 1`". That establishes the theorem *only for*
`h_i <= 1/2`, and we then stated it for all `h_i < 1`.

Valid corrected form: assume `h_i <= 1/2`. Or state the general bound for a box-free frontier and
keep them separate quantities.

### A2. `cor:failthresh` is false as stated

Counterexample, unit-norm, `d=2`, `F=3`: `a = (1,0)`,
`b = (-25/44, sqrt(1311)/44)`, `phi_i = (-17/40, sqrt(1311)/40)`.
Unique representation `z = (0.2, 1.1)`, so `||z||_inf = 1.1 > 1` and `kappa_i = inf`.

Measured `h_i = 0.555556`, `h/(1-h) = 1.25` (matching `lem:sm` exactly). For `F=3`, `s=3` the
threshold is `2/3`, and `0.5556 <= 0.6667`, so the corollary predicts non-separability at `s=3`
while the feature is separable at every sparsity.

Same root cause: `h_i = 0.5556 > 1/2`.

### A3. `prop:oneway` draws the wrong conclusion

We wrote that `[I, I]` has `kappa_i = 1` and is therefore "separable at `s=1` and at no larger
sparsity". `thm:frontier` requires `kappa_i > s`, and `1 > 1` is false. The largest separable
sparsity is `ceil(kappa)-1 = 0`. The audit is right, and this is visible directly from our own
theorem — a duplicated column means a singleton active state and a singleton inactive state have
identical representations, so nothing is separable at any positive sparsity.

This does not damage the proposition's purpose: `[I, I]` still refutes the converse of the
hierarchy, and refutes it more strongly than we claimed.

### A4. `thm:codefloor` contains a self-contradictory sentence

We write that stacking the minimisers "gives exactly the gain-calibrated pseudoinverse, so the
row-wise optimum and the algebraic pseudoinverse coincide". The second clause contradicts the
first. The optimum is `G_star = D_h^{-1} Phi^+` with `D_h = diag(h_1..h_F)`; it equals the
algebraic pseudoinverse only if every `h_i = 1`. The clause must go.

The audit also notes we should say `R_readout = 1` means optimal *for the calibrated quadratic
cross-talk objective*, not optimal for `s95` or for classification. That is correct and we do not
currently say it.

### A5. The headline is driven by pre-ReLU probes — verified per seed

This is the audit's most important empirical finding. The network reads `ReLU(Phi b)`; a pre-ReLU
probe reads `Phi b`, which the ReLU has not yet truncated. Restricting the probe to the same
post-ReLU state, under matched threshold policies, per seed rather than by medians:

| cell | policy | network | best post-ReLU probe | best pre-ReLU | winner |
|---|---|---|---|---|---|
| L4 d=50 | fixed | 3 | 2 | 2 | network (20/20) |
| L4 d=50 | global | 3 | 3 | 4 | tie (20/20) |
| L4 d=50 | per_feature | 3 | 3 | 3 | tie (20/20) |
| L4 d=100 | fixed | 4 | 4 | 5 | tie (20/20) |
| L4 d=100 | global | 5 | 5 | 5 | tie (20/20) |
| L4 d=100 | per_feature | 5 | 4 | 5 | network (18/20) |
| L4 d=200 | fixed | 7 | 7 | 8 | tie (20/20) |
| L4 d=200 | global | 6 | 6 | 7 | probe (1/20) |
| L4 d=200 | per_feature | 5 | 6 | 7 | probe (20/20) |

Tally: **network 2, tie 5, probe 2.** Against KD2's mixed-representation tally of 1/3/5 and the
gate's 0-for-120.

So "an affine probe beats the network" is not supported once the decoder input is held fixed. What
is supported is a different and more interesting claim: **the information is more affinely
accessible before the ReLU than after it.** That has to be named as such, not as a decoder
comparison.

### A6. `theta = 0.5` is not a common handicap across probe families

Our E7 called the fixed-threshold rows "the same handicap as the network". A ridge prediction, an
SVM margin and a logistic probability are not on one scale, so 0.5 does not mean the same thing.
The data show it plainly: `ridge_pre_fixed` and `ridge_post_fixed` reach `s95 = 0.0` at `d=50` and
`d=200`, which is a scale mismatch and not a decoding failure. The audit's recommendation — make
the validation-selected *global* threshold the matched policy, since it adds exactly one
calibration parameter to each decoder — is sound.

### A7. Our description of the disjoint splits was incomplete

At `s=1` only `F` supports exist — 100, 200, 400 — against thousands of requested states, so
literal disjointness is impossible. The implementation does handle it: it caps the smallest
sparsities and records `exhausted` and `n_redrawn` (measured: `test_by_s[1]` holds 50 states at
`F=100` with `exhausted=[1,2]`). But the packet asserted "enforced-disjoint supports" without
explaining the cap, which is exactly the kind of unexplained claim an audit should catch.

## Where the audit is wrong

### B1. The zeros of the random control do not enter the 0% gate

The audit worries that `s95 = 0` for the untrained control contaminates averages and the gate.
Checked: `stage_report` computes the gate over `trained = [x for x in diags if x["loss_kind"] !=
"random"]`, so the random cells are excluded by construction. The presentation problem is real —
those zeros appear as numbers in the per-width table where they should read `NA` — but the gate
itself is clean.

### B2. The empirical E3 claim is not invalidated by A1 and A2

The theorem statement is wrong. The *use* of it is not. `e3_bound_respected` evaluates the
corollary at `h.min()` only, and across all 180 models the largest `h_min` observed is **0.4885**,
so every application fell inside the `h <= 1/2` region where the proof is valid. The reported "E3
arrow held: True" therefore stands, and the `L2` prediction from leverage alone stands. What must
change is the theorem's hypothesis, not the measurement.

## Reproductions

```
# A1, A2, A3
python - <<'EOF'
import sys; sys.path.insert(0,'src')
import numpy as np
from lrtr.affine_frontier import collision_radius_atmost, affine_failure_threshold
from lrtr.analog_optimum import leverage
s15=np.sqrt(15); P=np.array([[1,.25,.25],[0,s15/4,-s15/4]])
print(leverage(P)[0], collision_radius_atmost(P,0)["rho_hat"])          # 0.8889, inf ; bound = 5
a=np.array([1,0.]); b=np.array([-25/44,np.sqrt(1311)/44]); p=np.array([-17/40,np.sqrt(1311)/40])
Q=np.column_stack([p,a,b])
print(leverage(Q)[0], collision_radius_atmost(Q,0)["rho_hat"], affine_failure_threshold(3,3))
I=np.eye(4); print(collision_radius_atmost(np.hstack([I,I]),0)["rho_hat"])   # 1.0 -> largest separable 0
EOF

# B2: largest h_min over all 180 models
# A5: post-ReLU restricted comparison, per seed
```

---

# The corrected primary comparison, and what it produced

Run by `scripts/primary_comparison.py`; output in `results/e7/derived/primary_comparison.json`.
Nothing retrained: the network is rescored from the saved weights and the probe numbers are read
from the per-seed records. Same decoder input (post-ReLU), same threshold treatment (one
validation-selected global threshold each), paired within seed, bootstrap over models.

## Result: the network and the best affine probe tie

`s95`, network minus best post-ReLU probe:

| cell | mean diff | 95% CI | net / tie / probe |
|---|---|---|---|
| L4 d=50 | +0.00 | [+0.00, +0.00] | 0 / **20** / 0 |
| L4 d=100 | +0.00 | [+0.00, +0.00] | 0 / **20** / 0 |
| L4 d=200 | −0.05 | [−0.15, +0.00] | 0 / **19** / 1 |
| L2 d=50 | +0.00 | [+0.00, +0.00] | 0 / **20** / 0 |
| L2 d=100 | +0.00 | [+0.00, +0.00] | 0 / **20** / 0 |
| L2 d=200 | +0.00 | [+0.00, +0.00] | 0 / **20** / 0 |

The AUC differences are resolved but tiny: `+0.0101` at `L4 d=50` (network ahead), `−0.0164` and
`−0.0198` at `d=100` and `d=200` (probe ahead), against absolute AUCs around 0.5–0.65.

**This is the outcome D1 registered, and it is the word D1 used: the affine decoder *matches* the
network.** Not "beats". The 0-for-120 gate was an artefact of three stacked asymmetries; with them
removed the comparison is a tie in 119 of 120 trained models.

## And a finding neither we nor the audit anticipated

Measuring the probe alone — the network is not involved — pre-ReLU `s95` minus post-ReLU `s95`,
paired, matched policy. This is what the ReLU discards:

| d | L4 | L2 | random |
|---|---|---|---|
| 50 | +1.00 [1.00, 1.00] | 0.00 | **+2.00** [2.00, 2.00] |
| 100 | 0.00 [0.00, 0.00] | 0.00 | **+2.00** [2.00, 2.00] |
| 200 | +0.95 [0.85, 1.00] | 0.00 | **+2.85** [2.70, 3.00] |

For an untrained random code the ReLU costs two to three sparsity levels of affine decodability.
For the `L4`-trained code it costs zero to one. `L2` loses nothing because it has nothing to lose —
its frontier is already collapsed at both stages.

So the answer to "what does training contribute", which the random control seemed to make
embarrassing, is not "very little". It is: **training arranges the code so that the nonlinearity it
will be read through destroys less of what a linear probe could have recovered.** That is a
mechanistic claim about the interaction between the code and the ReLU, it is invisible to any
analog-geometry statistic (`R_geom` is ~1.00 for both `L4` and random), and it is measured on 120
models with non-overlapping intervals.

It also reframes the audit's recommended thesis. The audit proposed that most affine decodability
is generic in random codes and `L4` adds a modest amount. That is right about the *pre-ReLU*
representation and wrong about the post-ReLU one, which is the state that actually gets composed
into the next layer.
