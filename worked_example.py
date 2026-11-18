"""One field, worked by hand-sized numbers. Writes results/worked_example.json.

python3 worked_example.py

A field with K = 4 values. Production shares p (Zipf s = 1, ASSUMED) and one made-up eval
set of n = 200 records with counts 116, 38, 28, 18.
"""
import json

import numpy as np

from ra_ceiling.aggregate import field_ceiling, field_ceiling_mc, noisy_baseline, self_baseline
from ra_ceiling.ra import alignment
from ra_ceiling.world import zipf

rng = np.random.default_rng(20261118)
p = zipf(4, 1.0)
counts = np.array([116, 38, 28, 18])
n = int(counts.sum())
p_hat = counts / n
a, d, base = alignment(p, p_hat, 50_000, rng)
B = self_baseline(p, 50_000, rng)
c, s = field_ceiling(p, n, B)
mc_c, mc_s, _ = field_ceiling_mc(p, n, 2000, 2000, rng)
out = {"K": 4, "n": n, "p": p.tolist(), "eval_counts": counts.tolist(), "p_hat": p_hat.tolist(),
       "d_js_distance": d, "shuffle_baseline": base, "a_field": a,
       "B_self_baseline": B, "b_n": noisy_baseline(p, n, B),
       "ceiling_closed": c, "ceiling_sd_closed": s, "ceiling_sim": mc_c, "ceiling_sd_sim": mc_s,
       "adjusted_field": min(1.0, a / mc_c), "gap_field": (mc_c - a) / mc_s}
json.dump(out, open("results/worked_example.json", "w"), indent=1)
print(json.dumps(out, indent=1))
