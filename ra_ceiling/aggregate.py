"""The RA ceiling from aggregates only.

Setting: production records cannot leave the compliance boundary. What can be seen is
aggregated: the per-field value shares of production (marginals) and, if the pipeline
exports them, pairwise co-occurrence tables. The eval set is eyes-on, so its histograms
are known too.

Question: what does a perfect random sample of n production records score, and how much
does that score vary, when we cannot draw such a sample?

Per field (marginal only). Write p for the production shares of one field, K for the
number of values with p > 0, and p_hat for the shares in a perfect n-record sample.
For large n, n * sum((p_hat - p)^2 / p) is chi-square with K - 1 degrees of freedom,
and a second-order expansion of the JS divergence gives

    JS(p, p_hat) in bits  ~  sum((p_hat - p)^2 / p) / (8 ln 2)  =  X / (8 n ln 2),

so the JS distance d = sqrt(JS) of a perfect sample has

    E[d]   ~  m(K - 1) / sqrt(8 n ln 2),   m(k) = sqrt(2) * Gamma((k + 1) / 2) / Gamma(k / 2)
    Var[d] ~  ((K - 1) - m(K - 1)^2) / (8 n ln 2).

The shuffle baseline of a perfect sample is close to B(p) = mean JS distance between p
and a random shuffle of p itself, which needs only p. A shuffled perfect sample also carries
the sample's own noise, and the two add roughly in squared distance, so we use
b(n) = sqrt(B(p)^2 + (K - 1) / (8 n ln 2)). (Without this term the formula is far too low for
near-flat fields at small n.) The per-field ceiling is then

    c(n) ~ 1 - E[d] / b(n),   with spread  s(n) ~ sd[d] / b(n).

The same two numbers can also be had by drawing multinomial(n, p) histograms, which is a
simulation that needs only the aggregate p (`field_ceiling_mc`).

Across fields. With the weights w~_i = g_i w_i fixed, the expected RA of a perfect sample
is the weighted mean of the c_i(n): that needs only marginals plus the weights. The
weights are another matter: g_i uses mutual information between fields, which needs
pairwise tables. The spread of total RA depends on how the fields' sampling errors move
together, which also needs pairwise tables. With marginals only, two bounds are available:
fields independent (sqrt of the weighted sum of variances) and fields perfectly in step
(weighted sum of the sds), which is the most the spread can be.

With pairwise tables, the covariance of every pair of field histograms is known exactly,
so `ra_spread_pairwise` draws the stacked histogram error from a Gaussian with that
covariance and pushes it through the same expansion. Nothing past pairs is used.
"""
import numpy as np
from math import lgamma, log, sqrt, exp

from .ra import jsd

LN2 = log(2.0)


def chi_mean_sqrt(k):
    """E[sqrt(X)] for X chi-square with k degrees of freedom."""
    if k <= 0:
        return 0.0
    return sqrt(2.0) * exp(lgamma((k + 1) / 2.0) - lgamma(k / 2.0))


def perfect_distance(p, n):
    """Closed form (large-n) mean and sd of the JS distance between p and a perfect n-sample."""
    p = np.asarray(p, dtype=float)
    k = int((p > 0).sum()) - 1
    scale = 1.0 / sqrt(8.0 * n * LN2)
    m = chi_mean_sqrt(k)
    return m * scale, sqrt(max(0.0, k - m * m)) * scale


def self_baseline(p, T, rng):
    """B(p): mean JS distance between p and T random shuffles of p. Needs only p."""
    p = np.asarray(p, dtype=float)
    perms = np.argsort(rng.random((T, p.size)), axis=1)
    return float(jsd(p, p[perms]).mean())


def field_ceiling(p, n, B):
    """Closed form per-field ceiling: (mean, sd) of the alignment of a perfect n-sample.
    B is self_baseline(p). Returns (0, 0) for a flat field (B = 0)."""
    if B <= 0:
        return 0.0, 0.0
    md, sd = perfect_distance(p, n)
    b = noisy_baseline(p, n, B)
    return max(0.0, 1.0 - md / b), sd / b


def noisy_baseline(p, n, B):
    """b(n) = sqrt(B^2 + (K - 1) / (8 n ln 2)): the shuffle baseline of a perfect n-sample."""
    k = int((np.asarray(p) > 0).sum()) - 1
    return sqrt(B * B + k / (8.0 * n * LN2))


def field_ceiling_mc(p, n, R, T, rng):
    """Simulated per-field ceiling from the aggregate alone: draw R multinomial(n, p)
    histograms and score each one exactly as RA does (shuffle baseline on the sample).
    Returns (mean, sd, list of alignments)."""
    from .ra import alignment
    vals = [alignment(p, rng.multinomial(n, p) / n, T, rng)[0] for _ in range(R)]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), vals


def ra_ceiling_from_marginals(P, eff_w, n, B):
    """Total-RA ceiling from marginals. P = list of production histograms, eff_w = the
    weights g_i * w_i, B = list of self baselines. Returns a dict with the mean, the spread
    if fields are independent, and the largest possible spread (fields in step)."""
    eff_w = np.asarray(eff_w, dtype=float)
    c = np.array([field_ceiling(p, n, b) for p, b in zip(P, B)])
    W = eff_w.sum()
    mean = float((eff_w * c[:, 0]).sum() / W)
    sd_ind = float(np.sqrt(((eff_w * c[:, 1]) ** 2).sum()) / W)
    sd_max = float((eff_w * c[:, 1]).sum() / W)
    return {"mean": mean, "sd_independent": sd_ind, "sd_max": sd_max,
            "field_mean": c[:, 0].tolist(), "field_sd": c[:, 1].tolist()}


def stacked_cov(P, pair_tables):
    """Covariance of the stacked one-hot vector of all fields for ONE record.
    pair_tables[(i, j)] (i < j) = joint shares of fields i and j, a K_i x K_j array.
    Only pairwise tables are used."""
    ks = [len(p) for p in P]
    off = np.concatenate([[0], np.cumsum(ks)])
    S = np.zeros((off[-1], off[-1]))
    for i, p in enumerate(P):
        S[off[i]:off[i + 1], off[i]:off[i + 1]] = np.diag(p) - np.outer(p, p)
    for (i, j), J in pair_tables.items():
        C = J - np.outer(P[i], P[j])
        S[off[i]:off[i + 1], off[j]:off[j + 1]] = C
        S[off[j]:off[j + 1], off[i]:off[i + 1]] = C.T
    return S, off


def ra_spread_pairwise(P, pair_tables, eff_w, n, B, R, rng):
    """Mean and sd of total RA for a perfect n-sample, using marginals + pairwise tables.
    Draws the stacked histogram error from N(0, Sigma / n), then per field
    d = sqrt(sum(delta^2 / p) / (8 ln 2)) and a = max(0, 1 - d / b(n)) (the same expansion
    as the closed form). Returns (mean, sd)."""
    S, off = stacked_cov(P, pair_tables)
    vals, vecs = np.linalg.eigh(S)
    L = vecs * np.sqrt(np.clip(vals, 0, None))
    Z = rng.standard_normal((R, L.shape[1])) @ L.T / sqrt(n)
    eff_w = np.asarray(eff_w, dtype=float)
    A = np.zeros((R, len(P)))
    for i, p in enumerate(P):
        dl = Z[:, off[i]:off[i + 1]]
        nz = p > 0
        q = (dl[:, nz] ** 2 / p[nz]).sum(axis=1) / (8 * LN2)
        A[:, i] = np.clip(1 - np.sqrt(q) / noisy_baseline(p, n, B[i]), 0, None) if B[i] > 0 else 0.0
    ra = (A * eff_w).sum(axis=1) / eff_w.sum()
    return float(ra.mean()), float(ra.std(ddof=1))


def noise_adjusted(ra, ceiling_mean, ceiling_sd):
    """Noise-adjusted score and gap, from the two ceiling numbers at the eval set's n.
    score = min(1, RA / ceiling): 1 = as close to production as a perfect sample of this size.
    gap_z = (ceiling - RA) / ceiling_sd: how many ceiling sds below the ceiling RA is
    (above 1.645 = below the one-sided 95% line of a perfect sample)."""
    score = min(1.0, ra / ceiling_mean) if ceiling_mean > 0 else 0.0
    gap = (ceiling_mean - ra) / ceiling_sd if ceiling_sd > 0 else float("inf")
    return score, gap
