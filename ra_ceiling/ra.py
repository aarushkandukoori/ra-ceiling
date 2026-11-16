"""The RA score, written to follow Sec. 4 of the ProxyDrift paper (arXiv 2608.08245).

Per field:
  d(P, O)   = Jensen-Shannon distance, base 2, square root (eq. 1), in [0, 1]
  base(P,O) = mean of d(P, pi(O)) over T random shuffles of O's value masses,
              with P held fixed (eqs. 2-3; the paper uses T = 50,000)
  a(P, O)   = max(0, 1 - d / base)

Across fields (eqs. 4-7):
  sort fields by w_i * H_i (largest first), H_i = entropy of production field i
  r_i = sum over earlier fields j of MI(i, j) / H_i
  g_i = max(0, 1 - 0.8 * r_i)
  RA  = sum(g_i w_i a_i) / sum(g_i w_i)

Details the paper leaves open, and what this code assumes:
  - the weights w_i: all equal to 1
  - MI and H: plug-in estimates (nats) on the production records
  - every field is single-valued (multi-valued fields are not modeled)
"""
import numpy as np


def jsd(p, q):
    """Jensen-Shannon distance (base 2, sqrt) between rows of p and q.

    p and q are arrays of probabilities with the values on the last axis.
    They broadcast, so p can be one histogram and q a stack of them.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0).sum(axis=-1)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0).sum(axis=-1)
    div = 0.5 * kl_pm + 0.5 * kl_qm
    return np.sqrt(np.clip(div, 0.0, 1.0))


def perm_baseline(p, o, T, rng):
    """Mean JSD between P and T random shuffles of O's value masses (P fixed)."""
    o = np.asarray(o, dtype=float)
    perms = np.argsort(rng.random((T, o.size)), axis=1)
    return float(jsd(p, o[perms]).mean())


def alignment(p, o, T, rng):
    """Per-field alignment a(P, O). Returns (a, d, base)."""
    d = float(jsd(p, o))
    base = perm_baseline(p, o, T, rng)
    a = max(0.0, 1.0 - d / base) if base > 0 else 0.0
    return a, d, base


def hist(col, k):
    """Value histogram (probabilities) of one integer column with k values."""
    c = np.bincount(col, minlength=k).astype(float)
    return c / c.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_info(x, y, kx, ky):
    """Plug-in mutual information (nats) between two integer columns."""
    joint = np.bincount(x.astype(np.int64) * ky + y, minlength=kx * ky).reshape(kx, ky)
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log(joint / (px * py)), 0.0)
    return float(terms.sum())


def redundancy_weights(prod, ks, w=None, lam=0.8):
    """Effective weights g_i * w_i from eqs. 4-6, computed on production records."""
    D = len(ks)
    w = np.ones(D) if w is None else np.asarray(w, dtype=float)
    H = np.array([entropy(hist(prod[:, i], ks[i])) for i in range(D)])
    order = np.argsort(-(w * H), kind="stable")
    eff = np.zeros(D)
    for pos, i in enumerate(order):
        r = 0.0
        for j in order[:pos]:
            if H[i] > 0:
                r += mutual_info(prod[:, i], prod[:, j], ks[i], ks[j]) / H[i]
        eff[i] = max(0.0, 1.0 - lam * r) * w[i]
    return eff


def ra_score(prod_hists, eval_hists, eff_w, T, rng):
    """RA over all fields. prod_hists / eval_hists are lists of histograms.

    Returns (RA, list of per-field a, list of per-field d, list of per-field base).
    """
    a_s, d_s, b_s = [], [], []
    for p, o in zip(prod_hists, eval_hists):
        a, d, b = alignment(p, o, T, rng)
        a_s.append(a)
        d_s.append(d)
        b_s.append(b)
    a_s = np.array(a_s)
    ra = float((eff_w * a_s).sum() / eff_w.sum())
    return ra, a_s, np.array(d_s), np.array(b_s)
