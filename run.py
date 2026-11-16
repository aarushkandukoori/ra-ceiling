"""Run every experiment, save numbers to results/results.json and figures to figures/.

python3 run.py          (about 5-10 minutes on a laptop, numpy only, seeded)
"""
import json
import time

import numpy as np

from ra_ceiling.ra import alignment, hist, jsd, perm_baseline, ra_score, redundancy_weights
from ra_ceiling.world import (KS, NAMES, World, chow_liu_sampler, chow_liu_tree,
                              dropped_sample, independent_sampler, perfect_sample,
                              shifted_sample, zipf)

SEED = 20261116
T = 2000            # shuffles per baseline in the sweeps (the paper uses 50,000; checked below)
N_PROD = 1_000_000  # production records
NS = [100, 200, 500, 1000, 2000, 5000, 10000]
R = 200             # repeats per setting
rng = np.random.default_rng(SEED)
out = {"settings": {"seed": SEED, "T": T, "N_prod": N_PROD, "repeats": R, "n_grid": NS}}


def q(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}


def hists(X, ks):
    return [hist(X[:, i], k) for i, k in enumerate(ks)]


def setup(world):
    prod = world.sample(N_PROD, rng)
    return prod, hists(prod, world.ks), redundancy_weights(prod, world.ks)


def ra_of(X, P, eff, ks, T=T):
    return ra_score(P, hists(X, ks), eff, T, rng)


t0 = time.time()

# ---- E1: ceiling = RA of a perfect random sample, vs n, for three skew levels ----
out["E1_ceiling_vs_n"] = {}
worlds = {}
for s in (0.25, 1.0, 2.0):
    w = World(s=s, seed=0)
    prod, P, eff = setup(w)
    worlds[s] = (w, prod, P, eff)
    res = {}
    for n in NS:
        vals = [ra_of(perfect_sample(prod, n, rng), P, eff, w.ks)[0] for _ in range(R)]
        res[str(n)] = q(vals)
    out["E1_ceiling_vs_n"][str(s)] = res
    print(f"E1 s={s} done {time.time()-t0:.0f}s", flush=True)

# single-field ceilings, K = 4 and 25, near-flat vs skewed
out["E1_single_field"] = {}
for k in (4, 25):
    for s in (0.25, 2.0):
        p = zipf(k, s)
        res = {}
        for n in NS:
            vals = [alignment(p, rng.multinomial(n, p) / n, T, rng)[0] for _ in range(R)]
            res[str(n)] = q(vals)
        out["E1_single_field"][f"K{k}_s{s}"] = res
print(f"E1 single done {time.time()-t0:.0f}s", flush=True)

# ---- E2: what skew does to the shuffle baseline (n = 1000) ----
SKEWS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
out["E2_skew"] = {"skews": SKEWS}
for k in (6, 25):
    base_l, ceil_l, scr_l, d_l = [], [], [], []
    for s in SKEWS:
        p = zipf(k, s)
        a_s, b_s, d_s = [], [], []
        for _ in range(100):
            a, d, b = alignment(p, rng.multinomial(1000, p) / 1000, T, rng)
            a_s.append(a); b_s.append(b); d_s.append(d)
        # "head right, tail scrambled": keep the top value's share, shuffle the rest
        sc = []
        for _ in range(100):
            o = p.copy()
            o[1:] = rng.permutation(o[1:])
            sc.append(alignment(p, o, T, rng)[0])
        base_l.append(q(b_s)); ceil_l.append(q(a_s)); scr_l.append(q(sc)); d_l.append(q(d_s))
    out["E2_skew"][f"K{k}"] = {"baseline": base_l, "ceiling_a": ceil_l,
                               "head_right_tail_scrambled_a": scr_l, "perfect_d": d_l,
                               "top_value_share": [float(zipf(k, s)[0]) for s in SKEWS]}
print(f"E2 done {time.time()-t0:.0f}s", flush=True)

# ---- E3: shifts of known size, s = 1 world ----
w, prod, P, eff = worlds[1.0]
LAMS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
N3 = [200, 1000, 5000]
R3 = 100

# expected squared perfect-sample distance per field, d0^2(n), from 200 perfect samples.
# Production owners can compute this inside the privacy wall (it needs only production).
def d0_of(n):
    ds = np.array([[float(jsd(P[i], h)) for i, h in enumerate(hists(perfect_sample(prod, n, rng), w.ks))]
                   for _ in range(200)])
    return (ds ** 2).mean(axis=0)

def calibrated(d, b, d0sq):
    """Noise-adjusted alignment: remove the sampling part of the distance in squared units
    (sampling noise and real drift add roughly in JS divergence = distance squared), then
    rescale as in eq. 3:  a* = 1 - sqrt(max(0, d^2 - d0^2)) / sqrt(base^2 - d0^2)."""
    a = 1 - np.sqrt(np.maximum(0, d ** 2 - d0sq)) / np.sqrt(np.maximum(1e-12, b ** 2 - d0sq))
    return float((eff * np.clip(a, 0, 1)).sum() / eff.sum())

out["E3_shift"] = {"lams": LAMS, "n": N3}
for n in N3:
    d0 = d0_of(n)
    raw, cal = [], []
    for lam in LAMS:
        r_l, c_l = [], []
        for _ in range(R3):
            ra, a, d, b = ra_of(shifted_sample(prod, w, n, lam, rng), P, eff, w.ks)
            r_l.append(ra); c_l.append(calibrated(d, b, d0))
        raw.append(q(r_l)); cal.append(q(c_l))
    out["E3_shift"][str(n)] = {"raw": raw, "calibrated": cal}
    print(f"E3 n={n} done {time.time()-t0:.0f}s", flush=True)

# smallest shift whose mean RA falls below the ceiling's 5th percentile, and whose 95th pct does
FINE = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
out["E3_detect"] = {}
for n in N3:
    ceil = out["E3_shift"][str(n)]["raw"][0]
    first_mean, first_p95 = None, None
    for lam in FINE[1:]:
        vals = [ra_of(shifted_sample(prod, w, n, lam, rng), P, eff, w.ks)[0] for _ in range(60)]
        st = q(vals)
        if first_mean is None and st["mean"] < ceil["p05"]:
            first_mean = lam
        if first_p95 is None and st["p95"] < ceil["p05"]:
            first_p95 = lam
            break
    out["E3_detect"][str(n)] = {"ceiling_p05": ceil["p05"], "lam_mean_below": first_mean,
                                "lam_p95_below": first_p95}
print(f"E3 detect done {time.time()-t0:.0f}s", flush=True)

# dropped value in intent_types (25 values), n = 1000
out["E3_dropped"] = []
for rank in (0, 2, 4, 9, 19):
    share = float(P[0][rank])
    ra_l, a_l = [], []
    for _ in range(100):
        ra, a, d, b = ra_of(dropped_sample(prod, 1000, 0, rank, rng), P, eff, w.ks)
        ra_l.append(ra); a_l.append(a[0])
    out["E3_dropped"].append({"value_rank": rank + 1, "prod_share": share,
                              "RA": q(ra_l), "intent_a": q(a_l)})
# same-field ceiling for intent_types at n = 1000, for comparison
out["E3_intent_ceiling_n1000"] = q([ra_of(perfect_sample(prod, 1000, rng), P, eff, w.ks)[1][0]
                                    for _ in range(100)])
print(f"E3 dropped done {time.time()-t0:.0f}s", flush=True)

# ---- E4: fields linked by a tree; independent vs Chow-Liu sampler, n = 1000 ----
wt = World(s=1.0, tree=True, alpha=0.6, seed=0)
prod_t, P_t, eff_t = setup(wt)
fit = chow_liu_tree(prod_t, wt.ks)
pairs = [(i, j) for i in range(wt.D) for j in range(i + 1, wt.D)]

PROD_JOINT = {(i, j): np.bincount(prod_t[:, i] * wt.ks[j] + prod_t[:, j],
                                  minlength=wt.ks[i] * wt.ks[j]) / prod_t.shape[0] for i, j in pairs}

def pair_jsd(X):
    """Mean JS distance between production and eval joint histograms, over all 210 field pairs."""
    vals = []
    for i, j in pairs:
        ki, kj = wt.ks[i], wt.ks[j]
        pj = PROD_JOINT[(i, j)]
        oj = np.bincount(X[:, i] * kj + X[:, j], minlength=ki * kj) / X.shape[0]
        vals.append(float(jsd(pj, oj)))
    return float(np.mean(vals))

R4 = 40
e4 = {"tree_edges_recovered": sorted(map(sorted, {frozenset((i, p)) for i, p in enumerate(fit[0]) if p >= 0}))
      == sorted(map(sorted, {frozenset((i, p)) for i, p in enumerate(wt.parent) if p >= 0})),
      "eff_weight_sum": float(eff_t.sum()), "eff_weight_sum_independent_world": float(worlds[1.0][3].sum())}
for name, make in (("perfect_sample", lambda: perfect_sample(prod_t, 1000, rng)),
                   ("independent_sampler", lambda: independent_sampler(prod_t, wt.ks, 1000, rng)),
                   ("chow_liu_sampler", lambda: chow_liu_sampler(prod_t, wt.ks, 1000, rng, fit))):
    ra_l, pj_l = [], []
    for _ in range(R4):
        X = make()
        ra_l.append(ra_of(X, P_t, eff_t, wt.ks)[0]); pj_l.append(pair_jsd(X))
    e4[name] = {"RA": q(ra_l), "pair_jsd": q(pj_l)}
out["E4_tree"] = e4
print(f"E4 done {time.time()-t0:.0f}s", flush=True)

# ---- Check: T = 2000 vs the paper's T = 50,000 ----
w, prod, P, eff = worlds[1.0]
chk = []
for n in (200, 1000, 5000):
    X = perfect_sample(prod, n, rng)
    r2 = ra_of(X, P, eff, w.ks, T=2000)[0]
    r50 = ra_of(X, P, eff, w.ks, T=50_000)[0]
    chk.append({"n": n, "RA_T2000": r2, "RA_T50000": r50, "abs_diff": abs(r2 - r50)})
out["check_T"] = chk
out["runtime_s"] = time.time() - t0
json.dump(out, open("results/results.json", "w"), indent=1)
print(f"all done {time.time()-t0:.0f}s")
