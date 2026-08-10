# Five theoretical extensions — worked through, corrected, and verified

Second pass. The first pass got E1 right, overclaimed E2, took the wrong route on E3, and
underrated E5. All four corrections came from the external review and all four are adopted here;
where its derivation is better than mine, the record says so.

Everything below is verified numerically against brute force. The headline is E3: it turns
"hierarchy" from a name into an implication, and it makes a correct sharp prediction about the
trained models.

---

## E1. The frontier for `|S| <= s` — solved, and it becomes the primary statement

With inequalities instead of equalities the size constraints no longer pin `1^T z`. Rebuilding
`(v, w)` from `z = w - v` leaves three conditions, and the last two collapse into one scalar:

> **Proposition.** `kappa_i = min { max(sum z^+, 1 + sum z^-) : Phi_{-i} z = phi_i,
> ||z||_inf <= 1 }`. Feature `i` is affinely separable over every support of size at most `s`
> iff `kappa_i > s`. The per-feature frontier is `ceil(kappa_i) - 1`.

The threshold is `s`, not `2 min(s, F-s) - 1`; the sets nest, so monotonicity holds **by
construction** and the `s > F/2` artefact disappears; and `kappa_i = 7.4` now reads directly as
"separable up to 7, lost at 8", which is what a capacity ought to mean.

**Verified:** 4754 (feature, sparsity) pairs against brute-force separation of the at-most-`s`
hulls. Zero mismatches.

This is no longer a tidy-up. Dropping `1^T z = 1` is precisely what makes E3 provable, so E1 and
E3 are one reform, not two.

---

## E2. The same state model at both levels — solved, but **not** "distribution-aware"

*The first pass claimed the affine level had become distribution-aware. That was too strong and
the review was right to reject it.*

With at most `s` active features and amplitudes in `[alpha, 1]`, the amplitudes wash out of the
convex hulls: for `U_k = {u : u_j in {0} u [alpha,1], |supp(u)| <= k}` the hull is
`{u in [0,1]^F : 1^T u <= k}` whatever `alpha` is, because the unit-weight knapsack polytope is
integral and its 0/1 vertices lie in `U_k`. Only the target feature's own amplitude survives, and
the worst case is the smallest:

> `kappa_i(alpha) = min { max(alpha sum z^+, 1 + alpha sum z^-) : Phi_{-i} z = phi_i,
> ||z||_inf <= 1/alpha }`, and separability at `s` iff `kappa_i(alpha) > s`.

A minimum amplitude is a necessity, not a convenience: as `a_i -> 0` an active state converges to
an inactive one, the hulls touch, and no uniform margin exists.

**Verified:** 2484 cases across `alpha in {1.0, 0.7, 0.4}`. Zero mismatches. Monotone in `alpha`.

**The correct framing.** Two state laws with the same *support* give the same `kappa`, so the
affine level is a **support-level worst-case property**, not a distribution-weighted one. What is
true is that both levels are now induced by one state law `P`:

| level | what it reads off `P` |
|---|---|
| analog reconstruction | the second moment `C = E_P[a a^T]` — distribution-weighted |
| affine support recovery | `supp(P) subset K_s` — support-level worst case |
| learned recovery | `f(x)`, `x ~ P` |

That closes the coherence objection without overclaiming. A genuinely probabilistic affine
frontier — a `1-delta` version — is possible but needs machinery we are not opening.

---

## E3. Leverage forces the frontier down — **the result, and my first route was wrong**

*The first pass derived `rho_i >= sqrt(h_i/(1-h_i))`, a lower bound, which is the wrong direction
for a hierarchy claim and nearly vacuous besides. My original conjecture went through coherence,
which cannot work at all: `rho >= 1/mu` bounds `rho` from below, so a large `mu` places no ceiling
on it. The review caught both. The route below is its derivation.*

Since `Phi_{-i} Phi_{-i}^T = Sigma - phi_i phi_i^T`, Sherman--Morrison gives the exact identity

> `min { ||z||_2^2 : Phi_{-i} z = phi_i } = h_i / (1 - h_i)`.

So leverage *is* redundancy, with no detour: low leverage means the other columns reproduce
`phi_i` with small coefficients. Then `||z||_1 <= sqrt(F-1) ||z||_2` and
`max(sum z^+, 1 + sum z^-) <= 1 + ||z||_1` give a certificate:

> **Proposition.** `kappa_i <= 1 + sqrt( (F-1) h_i / (1 - h_i) )`. The box comes free wherever the
> bound is informative: `h_i <= 1/2` implies `||z||_2 <= 1`, hence `||z||_inf <= 1`.
>
> **Corollary.** `h_i <= (s-1)^2 / ( (F-1) + (s-1)^2 )` implies feature `i` is **not** affinely
> separable at sparsity `s`.

**Verified:** the Sherman--Morrison identity to 4e-15; the upper bound with 0 violations; and the
failure threshold on **967 predicted failures with not one counterexample**.

### It predicts the trained models

Measured on the E5 weights (`d = 50`, `F = 100`, 5 seeds per cell):

| | `h_min` | `kappa` (worst feature) | affine frontier | bound predicts failure from |
|---|---|---|---|---|
| **L2** | 0.0091 | **1.001** | **1** | `s = 2` (8/10 seeds), `s = 3` (2/10) |
| **L4** | 0.4822 | **4.672** | 4 | `s = 11` |
| random | 0.3927 | 3.301 | 3 | `s = 9`–`10` |

At `F = 100` the `s = 2` threshold is `1/100 = 0.01`, and `L2` has `h_min = 0.0091 < 0.01`. So the
bound predicts, **from leverage alone and with no decoder in it**, that `L2` loses affine
separability at `s = 2`. Measured `kappa = 1.001`: separable at `s = 1`, lost at `s = 2`. Correct.

The bound is loose upward — the `sqrt(F-1)` step costs a lot, and on `L4` it predicts `s = 11`
against a measured 4. It is a sufficient condition for collapse, not an estimate of the frontier.

### And it is strictly one-way

`[I, I]` has perfectly uniform leverage (`h_i = 1/2` throughout, the ideal analog geometry) and
`kappa_i = 1` for every feature — the worst frontier possible, since each feature has a duplicate.
So:

> bad analog geometry **implies** bad affine frontier; good analog geometry **does not imply** a
> good one.

That is a strict one-way hierarchy, with the forward arrow proved and instantiated on a trained
network, and the converse refuted by an explicit code. It is what makes the word "hierarchy"
literally true rather than a label on three separate measurements.

---

## E4. Two cheap technical results — one negative, one explained

### E4a. The box is mostly inactive

Dropping `||z||_inf <= 1` from the exactly-`s` problem leaves
`min{ ||z||_1 : Phi_{-i} z = phi_i, 1^T z = 1 }`, which is basis pursuit with one extra affine
row. **Measured: the box binds on 3 of 72 features.** I expected the opposite, since `ell_1`
minimisation concentrates; it does not, here.

Reported rather than buried, because it cuts against novelty: for almost every feature the
collision radius *is* a minimum-`ell_1`-representation quantity, so the box is not what makes the
problem distinctive. It cuts the other way too — `ell_1` geometry, duality and descent cones
become available for the frontier's analysis.

Two cautions. An LP can have many optima, so a solver returning `||z||_inf > 1` does not prove
the box binds; the clean test is a second programme minimising `||z||_inf` subject to
`||z||_1 <= rho^{(0)} + eps`. And the identification is with *basis-pursuit geometry*, not with
the compressed-sensing phase transition, which asks a different question (recovery of a planted
sparse vector). We will write the former and not the latter.

### E4b. Why `[I, H]` approaches the coherence bound

Equality in `1 = sum_j z_j <phi_i, phi_j> <= mu_i ||z||_1` needs `z` supported only on
maximally-coherent partners with aligned signs. For `[I, H]`, `e_i = sum_j H_ij h_j` with every
coefficient of magnitude `1/sqrt(d)`, so `||z||_1 = d / sqrt(d) = sqrt(d) = 1/mu_i` exactly — the
code is built so that every atom used saturates the same coherence with the right sign. The dual
certificate `y = phi_i / mu_i` is feasible and optimal for the pure `ell_1` problem.

**Verified:** the pure `ell_1` minimum equals `sqrt(d)` exactly at `d = 8, 16, 32, 64`.

So the affine constraint `1^T z = 1` is precisely what stops our `rho` from attaining the bound,
which explains the measured `rho_min / sqrt(d) -> 1.018` instead of `1`. The numerical curiosity
has a geometric cause.

---

## E5. The nonlinear level does have an object — a witness, not a frontier

*The first pass said level 3 has no clean object and the reason is structural. Half right: there
is no **frontier**, because "all nonlinear decoders" has no capacity to compute without a
restricted class, and restricting it leads to sign-rank, which is out of scope. But there is a
clean **certificate**, it is small, and it falls out of G2 for free. The review is right that this
is the more interesting reading.*

When `kappa_i <= s` the hulls meet, so there are active states `x_k^+`, inactive states `x_l^-`
and convex weights with

    sum_k alpha_k x_k^+ = sum_l beta_l x_l^-.

No affine `f` can label them all correctly: affinity would give
`f(sum alpha_k x_k^+) > 0` and `f(sum beta_l x_l^-) < 0` for the same point. And by
Radon/Carathéodory the obstruction compresses to at most about `d + 2` states.

That changes what can be reported. Not "an LP says the hulls intersect somewhere among millions
of states", but: *these 11 explicit states cannot all be labelled correctly by any affine rule* —
and then, evaluating the trained network on those same 11 states, either it resolves them or it
does not, with a margin

    gamma_net = min( min_k N_i(x_k^+) - theta, min_l theta - N_i(x_l^-) ).

A line such as "feature 37, `s = 5`: 8-state Radon witness, affine decoding impossible, the
learned decoder resolves all eight with margin 0.21" is a categorically stronger claim than a
recovery curve. And it composes: G2's collision certificate already produces `u` and `v`, whose
vertex decompositions *are* the witness states, so G3 consumes G2's output rather than
duplicating it.

The chain becomes one theory instead of three metrics: **G1 analog optimum → G2 affine frontier →
G3 finite nonlinear witness**, each consuming the previous.

---

## Summary and experimental consequences

| # | result | status | experiments? |
|---|---|---|---|
| E1 | `kappa_i > s` for at-most-`s`; monotone by construction | **solved**, 4754/4754 | No — replaces the current statement |
| E2 | amplitudes wash out; support-level worst case at one level, `C` at the other | **solved**, 2484/2484 | **Yes, and cheaper**: R4's six distributions become a sweep over `alpha` |
| E3 | `kappa_i <= 1 + sqrt((F-1)h_i/(1-h_i))`; failure threshold; predicts `L2` | **solved**, 967/967, correct on trained models | **Yes**: replicate the arrow across widths in Stage A |
| E4a | box binds on 3/72 | **negative**, measured | One figure: `rho` vs `rho^(0)` against `F/d` |
| E4b | `[I,H]` saturates coherence for pure `ell_1`; the affine row breaks it | **solved**, exact at four widths | No |
| E5 | finite Radon witness, `<= d+2` states, derived from G2 | **derivation done**, not implemented | **Yes**: extract witnesses and test the network on them |

**What is now worth building, in order.** E3's arrow needs replication across widths — it is
currently one width — and that rides along with Stage A at no extra cost. E5 needs the witness
extractor, which is a day and would give the paper its strongest single sentence. E2 *reduces*
the campaign: one parameter instead of six distributions.

**What has not changed.** These are statements about codes. The `L2` prediction is the one place
theory touches a trained network, and it holds on five seeds at one width. That is a coincidence
until Stage A says otherwise.
