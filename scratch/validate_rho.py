"""Exhaustive validation of the collision-radius characterisation of the affine frontier."""
import sys, itertools, numpy as np
sys.path.insert(0, "src")
from scipy.optimize import linprog
from lrtr.affine_frontier import collision_radius, separable_from_rho, robust_affine_margin
from lrtr.codes import harmonic_tight_frame

def brute_separable(Phi, i, s):
    """Strict separation of the two hulls, by LP over all vertices. Ground truth."""
    d, F = Phi.shape
    others = [j for j in range(F) if j != i]
    Vp = [Phi[:, i] + (Phi[:, list(A)].sum(1) if s > 1 else 0)
          for A in itertools.combinations(others, s - 1)]
    Vm = [Phi[:, list(B)].sum(1) for B in itertools.combinations(others, s)]
    nv = d + 2
    rows = []
    for p in Vp:
        r = np.zeros(nv); r[:d] = -p; r[d] = 1.0; r[d+1] = 1.0; rows.append(r)
    for q in Vm:
        r = np.zeros(nv); r[:d] = q; r[d] = -1.0; r[d+1] = 1.0; rows.append(r)
    c = np.zeros(nv); c[d+1] = -1.0
    res = linprog(c, A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  bounds=[(-1, 1)]*d + [(None, None)] + [(None, 1.0)], method="highs")
    return bool(res.success and res.x[d+1] > 1e-9)

def check(Phi, also_g2=False):
    d, F = Phi.shape
    bad_rho = bad_g2 = n = 0
    for i in range(F):
        rho = collision_radius(Phi, i)["rho"]
        for s in range(1, F):
            truth = brute_separable(Phi, i, s); n += 1
            if separable_from_rho(rho, s, F) != truth:
                bad_rho += 1
            if also_g2 and s <= F - 1:
                if (robust_affine_margin(Phi, i, s)["status"] == "separable") != truth:
                    bad_g2 += 1
    return bad_rho, bad_g2, n

rng = np.random.default_rng(7)
tot_bad = tot_g2 = tot = 0
print("=== random codes ===")
for t in range(60):
    d = int(rng.integers(2, 6)); F = int(rng.integers(d + 2, 10))
    Phi = rng.standard_normal((d, F)); Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    b, g, n = check(Phi, also_g2=False)
    tot_bad += b; tot_g2 += g; tot += n
print(f"  {tot} (feature,s) pairs over 60 codes -> rho-rule mismatches: {tot_bad}")

print("\n=== constructed cases ===")
def norm(M): return M / np.linalg.norm(M, axis=0, keepdims=True)
B4 = np.linalg.qr(rng.standard_normal((4, 4)))[0]
B3 = norm(rng.standard_normal((3, 4)))
dup = norm(rng.standard_normal((4, 8))); dup[:, 1] = dup[:, 0]
cases = {
    "duplicated column":      dup,
    "harmonic tight frame":   harmonic_tight_frame(4, 9),
    "harmonic tight frame 2": harmonic_tight_frame(6, 11),
    "orthonormal basis tiled": np.hstack([B4, B4]),
    "near-collinear":         norm(np.repeat(rng.standard_normal((4, 1)), 8, axis=1)
                                   + 1e-2 * rng.standard_normal((4, 8))),
    "antipodal pairs":        np.hstack([B3, -B3]),
}
for name, Phi in cases.items():
    Phi = norm(np.asarray(Phi, float))
    b, _, n = check(Phi)
    rr = [collision_radius(Phi, i)["rho"] for i in range(Phi.shape[1])]
    fin = [x for x in rr if np.isfinite(x)]
    tot_bad += b; tot += n
    print(f"  {name:24s} d={Phi.shape[0]} F={Phi.shape[1]}  mismatches={b}/{n}  "
          f"rho_min={min(rr):.3f}  n_inf={sum(1 for x in rr if not np.isfinite(x))}")

print(f"\nTOTAL: {tot_bad} mismatches over {tot} (feature, sparsity) pairs")
