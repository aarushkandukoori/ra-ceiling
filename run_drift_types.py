"""Does the kind of drift matter? Writes results/drift_types.json.

python3 run_drift_types.py     (a few minutes, numpy only, seeded)

Same setup as run_aggregate.py part D: 21 fields with the paper's value counts, Zipf s = 1
value shapes (ASSUMED), fields independent, 1,000,000 production records. An eval set of n
records is a perfect random sample of production in which each record is, with probability
lam, swapped for a drifted record. A drifted record draws each field from a drifted
distribution q_i instead of production's p_i. Six kinds of q (all ASSUMED):

  reverse_all      every field: value ranks reversed (rare values become common). This is
                   the drift used in the earlier notes.
  swap_top2_all    every field: the two most common values trade shares (mild).
  flatten_all      every field: all values equally likely.
  reverse_3        only intent_types, tone_emotion, output_type reversed; the other 18
                   fields come from production.
  reverse_1        only intent_types reversed.
  spike_1          only intent_types: every drifted record takes its rarest value (a burst
                   of one rare topic).

Reported per kind, lam and n: total RA, the noise-adjusted score min(1, RA / C(n)), the share
of sets the total gap test flags (gap > 1.645), and the share flagged by a per-field test
(the largest per-field gap above its own 95th percentile in perfect sets). Ceilings and both
thresholds come from multinomial draws on the production marginals (aggregates only): one
batch of draws sets each field's ceiling c_i and spread s_i, a second batch sets the cut-off
for the largest per-field gap, so a perfect set is flagged about 5% of the time.
"""
import json
import time

import numpy as np

from ra_ceiling.ra import hist, jsd, ra_score, redundancy_weights
from ra_ceiling.world import NAMES, World, perfect_sample

SEED = 20261118
T = 2000
N_PROD = 1_000_000
NS = [200, 1000]
LAMS = [0.0, 0.05, 0.1, 0.2]
R = 200           # eval sets per setting
R_CEIL = 400      # multinomial draws for the ceilings (two batches of this size)
Z_TOTAL = 1.645
rng = np.random.default_rng(SEED)
t0 = time.time()

w = World(s=1.0, seed=0)
prod = w.sample(N_PROD, rng)
P = [hist(prod[:, i], k) for i, k in enumerate(w.ks)]
eff = redundancy_weights(prod, w.ks)
I_INTENT = NAMES.index("intent_types")
THREE = [NAMES.index(x) for x in ("intent_types", "tone_emotion", "output_type")]


def q_for(kind):
    """Drifted distribution per field, or None where the field is not drifted."""
    qs = [None] * w.D
    for i, p in enumerate(P):
        if kind == "reverse_all":
            qs[i] = w.base[i][::-1].copy()
        elif kind == "swap_top2_all":
            q = w.base[i].copy()
            q[[0, 1]] = q[[1, 0]]
            qs[i] = q
        elif kind == "flatten_all":
            qs[i] = np.full(w.ks[i], 1.0 / w.ks[i])
    if kind == "reverse_3":
        for i in THREE:
            qs[i] = w.base[i][::-1].copy()
    elif kind == "reverse_1":
        qs[I_INTENT] = w.base[I_INTENT][::-1].copy()
    elif kind == "spike_1":
        q = np.zeros(w.ks[I_INTENT])
        q[-1] = 1.0
        qs[I_INTENT] = q
    return qs


KINDS = ["reverse_all", "swap_top2_all", "flatten_all", "reverse_3", "reverse_1", "spike_1"]
QS = {k: q_for(k) for k in KINDS}


def drifted_set(n, lam, qs):
    X = perfect_sample(prod, n, rng)
    mask = rng.random(n) < lam
    m = int(mask.sum())
    if m:
        rows = np.flatnonzero(mask)
        for i, q in enumerate(qs):
            if q is not None:
                X[rows, i] = rng.choice(w.ks[i], size=m, p=q)
    return X


def hists(X):
    return [hist(X[:, i], k) for i, k in enumerate(w.ks)]


def drift_size(qs, lam):
    """JS distance between production and the drifted mixture, per field; returns the
    weighted mean over fields and the largest single field."""
    d = np.array([float(jsd(p, (1 - lam) * p + lam * q)) if q is not None else 0.0
                  for p, q in zip(P, qs)])
    return float((eff * d).sum() / eff.sum()), float(d.max())


out = {"settings": {"seed": SEED, "T": T, "N_prod": N_PROD, "n": NS, "lams": LAMS, "repeats": R,
                    "ceiling_draws": R_CEIL, "z_total": Z_TOTAL,
                    "shape": "Zipf s=1, independent fields", "three_fields": [NAMES[i] for i in THREE]},
       "kinds": {}}

for n in NS:
    # ceilings from the aggregates: multinomial(n, p_i) per field, scored as RA does
    ra_c, a_c = [], []
    for _ in range(R_CEIL):
        ra, a, _, _ = ra_score(P, [rng.multinomial(n, p) / n for p in P], eff, T, rng)
        ra_c.append(ra)
        a_c.append(a)
    C, S = float(np.mean(ra_c)), float(np.std(ra_c, ddof=1))
    a_c = np.array(a_c)
    c_i, s_i = a_c.mean(axis=0), a_c.std(axis=0, ddof=1)
    s_safe = np.where(s_i > 0, s_i, 1.0)

    def max_gap(a):
        return float(np.max(np.where(s_i > 0, (c_i - a) / s_safe, 0.0)))

    # second batch: cut-off for the largest per-field gap in a perfect set
    mg = [max_gap(ra_score(P, [rng.multinomial(n, p) / n for p in P], eff, T, rng)[1]) for _ in range(R_CEIL)]
    z_field = float(np.quantile(mg, 0.95))
    out[f"ceiling_n{n}"] = {"C": C, "S": S, "z_field_cutoff": z_field}
    for kind in KINDS:
        rows = []
        for lam in LAMS:
            ra_l, adj_l, ft, ff = [], [], [], []
            for _ in range(R):
                ra, a, _, _ = ra_score(P, hists(drifted_set(n, lam, QS[kind])), eff, T, rng)
                ra_l.append(ra)
                adj_l.append(min(1.0, ra / C))
                ft.append((C - ra) / S > Z_TOTAL)
                ff.append(max_gap(a) > z_field)
            ds_mean, ds_max = drift_size(QS[kind], lam)
            rows.append({"lam": lam, "RA_mean": float(np.mean(ra_l)), "RA_sd": float(np.std(ra_l, ddof=1)),
                         "adjusted_mean": float(np.mean(adj_l)),
                         "flag_total": float(np.mean(ft)), "flag_any_field": float(np.mean(ff)),
                         "drift_jsd_weighted_mean": ds_mean, "drift_jsd_max_field": ds_max})
        out["kinds"].setdefault(kind, {})[str(n)] = rows
        print(f"n={n} {kind} done {time.time()-t0:.0f}s", flush=True)

out["runtime_s"] = time.time() - t0
json.dump(out, open("results/drift_types.json", "w"), indent=1)
print(f"all done {time.time()-t0:.0f}s")
