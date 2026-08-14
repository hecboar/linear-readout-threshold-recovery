"""Complete the 2x2: does the network also win when IT gets a tuned threshold?

The campaign scores the network at a fixed theta = 0.5 and gives probes three threshold policies,
two of them validation-selected. That asymmetry favours the probe. This recomputes the network's
s95 with the *same* threshold policies the probes get, selected on the same validation split, and
reports the resulting 2x2 so neither side has a tuning advantage the other lacks.

Nothing here is a new estimand: s95 and the recovery AUC are unchanged, and the thresholds are
selected on validation exactly as select_thresholds does for the probes.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrtr.probes import _exact_recovery, evaluate_scores, select_thresholds  # noqa: E402
from lrtr.splits import make_state_splits, representations  # noqa: E402
from lrtr.threshold import s95_from_curve  # noqa: E402

cfg = json.loads((ROOT / "configs" / "e7.json").read_text(encoding="utf-8"))
CELLS = [("L4", 50), ("L4", 100), ("L4", 200), ("L2", 50), ("L2", 200)]
rows = []

for loss, d in CELLS:
    name = f"relu_{loss}_p0.01_d{d}"
    z = np.load(ROOT / "results" / "e7" / "weights" / f"{name}.npz", allow_pickle=True)
    W_in_all, W_out_all = z["W_in"], z["W_out"]
    sparsities = z["sparsities"].tolist()
    seeds = z["seeds"].tolist()
    F = int(W_in_all.shape[2])
    per = {"fixed": [], "global": [], "per_feature": []}
    for k, seed in enumerate(seeds):
        W_in = W_in_all[k].astype(np.float64)
        W_out = W_out_all[k].astype(np.float64)
        bundle = make_state_splits(F=F, sparsities=sparsities, n_train=cfg["probe_n_train"],
                                  n_val=cfg["probe_n_val"], n_test=cfg["probe_n_test"],
                                  seed=(100_000 + 997 * k) + 1)

        def scores(split):
            return representations(W_in, split, post_relu=True) @ W_out.T

        Zv = scores(bundle.val)
        for pol in ("fixed", "global", "per_feature"):
            th = (cfg["theta"] if pol == "fixed"
                  else select_thresholds(Zv, bundle.val, pol, theta_fixed=cfg["theta"]))
            ss, pr = [], []
            for s in sparsities:
                sub = bundle.test_by_s[s]
                if not len(sub):
                    continue
                ss.append(s)
                pr.append(float(_exact_recovery(scores(sub), sub, th).mean()))
            per[pol].append(s95_from_curve(ss, pr))
    rows.append((loss, d, {p: float(np.median(v)) for p, v in per.items()}))
    print(f"{loss} d={d}: network s95 by its own threshold policy -> "
          + "  ".join(f"{p}={np.median(v):.1f}" for p, v in per.items()), flush=True)

# Written where docs/known_defects.md KD2 says it is. The first version wrote `net_tuned.json`
# into whatever directory it was launched from, and the file the documentation cites got there by
# hand -- so the table in KD2 was not reproducible by running the script it names.
dest = ROOT / "results" / "e7" / "derived" / "network_threshold_policies.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(
    [{"loss": l, "d": d, "network_s95_by_policy": v} for l, d, v in rows], indent=2),
    encoding="utf-8", newline="\n")
print(f"wrote {dest.relative_to(ROOT)}")
