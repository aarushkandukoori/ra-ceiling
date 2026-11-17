"""Aggregates-only experiments. Writes results/aggregate.json.

python3 run_aggregate.py     (a few minutes, numpy only, seeded)

A  per field: closed-form ceiling vs simulation from the aggregate (multinomial draws)
B  all 21 fields: ceiling from marginals vs perfect samples of production records
C  linked fields: what the spread of total RA needs (marginals vs pairwise tables)
D  noise-adjusted score vs RA as defined, on the same simulated cases
E  sampling gap (1 - ceiling) next to a fixed 0.10 rehydration loss
"""
import json
import time

import numpy as np

from ra_ceiling.aggregate import (field_ceiling, field_ceiling_mc, noise_adjusted,
                                  ra_ceiling_from_marginals, ra_spread_pairwise, self_baseline)
from ra_ceiling.ra import hist, ra_score, redundancy_weights
from ra_ceiling.world import KS, World, perfect_sample, shifted_sample, zipf

SEED = 20261117
T = 2000          # shuffles per baseline when scoring an eval set
TB = 50_000       # shuffles for the self baseline B(p) (the paper's T)
N_PROD = 1_000_000
NS = [100, 200, 500, 1000, 2000, 5000, 10000]
rng = np.random.default_rng(SEED)
out = {"settings": {"seed": SEED, "T": T, "T_self_baseline": TB, "N_prod": N_PROD, "n_grid": NS}}
t0 = time.time()


def q(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}


def hists(X, ks):
    return [hist(X[:, i], k) for i, k in enumerate(ks)]


# ---- A: one field ----
A = {}
errs_mean, errs_sd = [], []
for k in (4, 6, 11, 25):
    for s in (0.25, 1.0, 2.0):
        p = zipf(k, s)
        B = self_baseline(p, TB, rng)
        rows = []
        for n in NS:
            cf_m, cf_s = field_ceiling(p, n, B)
            mc_m, mc_s, _ = field_ceiling_mc(p, n, 400, T, rng)
            rows.append({"n": n, "closed_mean": cf_m, "closed_sd": cf_s, "sim_mean": mc_m, "sim_sd": mc_s,
                         "n_times_min_share": float(n * p.min())})
            errs_mean.append((abs(cf_m - mc_m), k, s, n)); errs_sd.append((abs(cf_s - mc_s), k, s, n))
        A[f"K{k}_s{s}"] = {"B": B, "rows": rows}
out["A_field"] = A


def summary(errs, cond):
    e = [x[0] for x in errs if cond(x)]
    worst = max((x for x in errs if cond(x)), key=lambda x: x[0])
    return {"count": len(e), "max_abs_err": float(max(e)), "median_abs_err": float(np.median(e)),
            "worst_case": {"K": worst[1], "s": worst[2], "n": worst[3]}}


out["A_error"] = {
    "mean_all": summary(errs_mean, lambda x: True),
    "mean_n_ge_1000": summary(errs_mean, lambda x: x[3] >= 1000),
    "mean_n_ge_1000_not_flat": summary(errs_mean, lambda x: x[3] >= 1000 and x[2] >= 1.0),
    "sd_all": summary(errs_sd, lambda x: True),
    "sd_n_ge_1000": summary(errs_sd, lambda x: x[3] >= 1000),
}
print(f"A done {time.time()-t0:.0f}s", flush=True)

# ---- B: 21 fields, independent worlds ----
worlds = {}
out["B_total"] = {}
for s in (0.25, 1.0, 2.0):
    w = World(s=s, seed=0)
    prod = w.sample(N_PROD, rng)
    P = hists(prod, w.ks)
    eff = redundancy_weights(prod, w.ks)
    Bs = [self_baseline(p, TB, rng) for p in P]
    worlds[s] = (w, prod, P, eff, Bs)
    rows = []
    for n in NS:
        agg = ra_ceiling_from_marginals(P, eff, n, Bs)
        # simulation from aggregates: each field multinomial(n, p) on its own
        sim = [ra_score(P, [rng.multinomial(n, p) / n for p in P], eff, T, rng)[0] for _ in range(150)]
        # perfect samples of the production records themselves
        rec = [ra_score(P, hists(perfect_sample(prod, n, rng), w.ks), eff, T, rng)[0] for _ in range(150)]
        rows.append({"n": n, "closed_mean": agg["mean"], "closed_sd_independent": agg["sd_independent"],
                     "closed_sd_max": agg["sd_max"], "sim_from_marginals": q(sim), "records": q(rec)})
    out["B_total"][str(s)] = rows
    print(f"B s={s} done {time.time()-t0:.0f}s", flush=True)

# ---- C: linked fields (tree world), what the spread needs ----
out["C_linked"] = {}
for alpha in (0.6, 0.9):
    w = World(s=1.0, tree=True, alpha=alpha, seed=0)
    prod = w.sample(N_PROD, rng)
    P = hists(prod, w.ks)
    eff = redundancy_weights(prod, w.ks)
    Bs = [self_baseline(p, TB, rng) for p in P]
    pairs = {}
    for i in range(w.D):
        for j in range(i + 1, w.D):
            J = np.bincount(prod[:, i] * w.ks[j] + prod[:, j], minlength=w.ks[i] * w.ks[j])
            pairs[(i, j)] = (J / prod.shape[0]).reshape(w.ks[i], w.ks[j])
    ones = np.ones(w.D)
    rows = []
    for n in (100, 1000, 10000):
        agg = ra_ceiling_from_marginals(P, eff, n, Bs)
        agg_eq = ra_ceiling_from_marginals(P, ones, n, Bs)
        pm, ps = ra_spread_pairwise(P, pairs, eff, n, Bs, 4000, rng)
        rec = [ra_score(P, hists(perfect_sample(prod, n, rng), w.ks), eff, T, rng)[0] for _ in range(300)]
        rows.append({"n": n, "records": q(rec), "marginals_mean": agg["mean"],
                     "marginals_sd_independent": agg["sd_independent"], "marginals_sd_max": agg["sd_max"],
                     "pairwise_mean": pm, "pairwise_sd": ps,
                     "equal_weights_mean": agg_eq["mean"]})
    out["C_linked"][str(alpha)] = {"eff_weight_sum": float(eff.sum()), "rows": rows}
    print(f"C alpha={alpha} done {time.time()-t0:.0f}s", flush=True)

# ---- D: noise-adjusted score vs RA as defined ----
out["D_compare"] = []
for s in (0.25, 1.0, 2.0):
    w, prod, P, eff, Bs = worlds[s]
    for n in (100, 1000, 10000):
        agg = ra_ceiling_from_marginals(P, eff, n, Bs)
        # spread: use the simulated-from-marginals sd from B (closed sd is rough for flat fields at small n)
        sim_sd = next(r for r in out["B_total"][str(s)] if r["n"] == n)["sim_from_marginals"]["sd"]
        sim_mean = next(r for r in out["B_total"][str(s)] if r["n"] == n)["sim_from_marginals"]["mean"]
        for lam in (0.0, 0.1):
            ra_l, adj_l, z_l = [], [], []
            for _ in range(100):
                X = shifted_sample(prod, w, n, lam, rng)
                ra = ra_score(P, hists(X, w.ks), eff, T, rng)[0]
                sc, z = noise_adjusted(ra, sim_mean, sim_sd)
                ra_l.append(ra); adj_l.append(sc); z_l.append(z)
            out["D_compare"].append({"s": s, "n": n, "lambda": lam, "ceiling_mean": sim_mean,
                                     "ceiling_sd": sim_sd, "closed_ceiling_mean": agg["mean"],
                                     "RA": q(ra_l), "adjusted": q(adj_l),
                                     "flagged_share": float(np.mean(np.array(z_l) > 1.645))})
    print(f"D s={s} done {time.time()-t0:.0f}s", flush=True)

# ---- E: sampling gap vs rehydration loss ----
REHYD = 0.10   # illustrative rehydration loss of 0.10 on the RA scale
out["E_gap"] = {"rehydration_loss": REHYD, "rows": []}
for n in NS:
    row = {"n": n}
    for s in (0.25, 1.0, 2.0):
        m = next(r for r in out["B_total"][str(s)] if r["n"] == n)["sim_from_marginals"]["mean"]
        row[f"gap_s{s}"] = 1 - m
    out["E_gap"]["rows"].append(row)

out["runtime_s"] = time.time() - t0
json.dump(out, open("results/aggregate.json", "w"), indent=1)
print(f"all done {time.time()-t0:.0f}s")
