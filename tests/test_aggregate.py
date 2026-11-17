import numpy as np
from math import sqrt, log

from ra_ceiling.aggregate import (chi_mean_sqrt, field_ceiling, field_ceiling_mc, noise_adjusted,
                                  perfect_distance, ra_ceiling_from_marginals, ra_spread_pairwise,
                                  self_baseline, stacked_cov)
from ra_ceiling.ra import jsd
from ra_ceiling.world import zipf


def test_chi_mean_sqrt_known_values():
    # E[sqrt(chi2_1)] = sqrt(2/pi); E[sqrt(chi2_2)] = sqrt(pi/2)
    assert abs(chi_mean_sqrt(1) - sqrt(2 / np.pi)) < 1e-12
    assert abs(chi_mean_sqrt(2) - sqrt(np.pi / 2)) < 1e-12
    # and it approaches sqrt(k) for large k
    assert abs(chi_mean_sqrt(400) / sqrt(400) - 1) < 1e-3


def test_js_expansion_matches_exact_for_small_errors():
    p = zipf(6, 1.0)
    delta = np.array([0.004, -0.003, 0.002, -0.001, -0.001, -0.001])
    exact = jsd(p, p + delta) ** 2
    approx = (delta ** 2 / p).sum() / (8 * log(2))
    assert abs(exact - approx) / exact < 0.02


def test_perfect_distance_matches_simulation_large_n():
    rng = np.random.default_rng(3)
    p = zipf(11, 1.0)
    n = 5000
    d = np.array([jsd(p, rng.multinomial(n, p) / n) for _ in range(3000)])
    m, s = perfect_distance(p, n)
    assert abs(d.mean() - m) / m < 0.02
    assert abs(d.std() - s) / s < 0.08


def test_zero_share_values_do_not_count():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.5, 0.5])
    assert perfect_distance(p, 100) == perfect_distance(q, 100)


def test_flat_field_has_zero_ceiling():
    rng = np.random.default_rng(0)
    p = np.ones(8) / 8
    B = self_baseline(p, 1000, rng)
    assert B == 0.0
    assert field_ceiling(p, 1000, B) == (0.0, 0.0)


def test_closed_form_ceiling_close_to_simulation():
    rng = np.random.default_rng(5)
    for k, s in ((4, 1.0), (25, 2.0)):
        p = zipf(k, s)
        B = self_baseline(p, 20000, rng)
        cm, _ = field_ceiling(p, 1000, B)
        mm, _, _ = field_ceiling_mc(p, 1000, 200, 1000, rng)
        assert abs(cm - mm) < 0.015


def test_total_ceiling_is_weighted_mean_and_bounds_order():
    rng = np.random.default_rng(1)
    P = [zipf(4, 1.0), zipf(25, 1.0)]
    Bs = [self_baseline(p, 5000, rng) for p in P]
    w = np.array([1.0, 3.0])
    agg = ra_ceiling_from_marginals(P, w, 1000, Bs)
    fm = agg["field_mean"]
    assert abs(agg["mean"] - (fm[0] + 3 * fm[1]) / 4) < 1e-12
    assert agg["sd_independent"] <= agg["sd_max"] + 1e-15


def test_stacked_cov_rows_sum_to_zero_and_pairwise_independent_case():
    P = [zipf(3, 1.0), zipf(4, 0.5)]
    J = np.outer(P[0], P[1])                   # independent fields
    S, off = stacked_cov(P, {(0, 1): J})
    assert np.allclose(S[:, off[0]:off[1]].sum(axis=1), 0)
    assert np.allclose(S[off[0]:off[1], off[1]:off[2]], 0)


def test_pairwise_spread_independent_equals_marginal_spread():
    rng = np.random.default_rng(2)
    P = [zipf(5, 1.0), zipf(6, 1.0), zipf(4, 2.0)]
    Bs = [self_baseline(p, 5000, rng) for p in P]
    pairs = {(i, j): np.outer(P[i], P[j]) for i in range(3) for j in range(i + 1, 3)}
    w = np.ones(3)
    m, s = ra_spread_pairwise(P, pairs, w, 2000, Bs, 20000, rng)
    agg = ra_ceiling_from_marginals(P, w, 2000, Bs)
    assert abs(m - agg["mean"]) < 0.005
    assert abs(s - agg["sd_independent"]) / agg["sd_independent"] < 0.1


def test_noise_adjusted():
    assert noise_adjusted(0.9, 0.9, 0.01) == (1.0, 0.0)
    sc, z = noise_adjusted(0.8, 0.9, 0.02)
    assert abs(sc - 0.8 / 0.9) < 1e-12 and abs(z - 5.0) < 1e-9
    assert noise_adjusted(0.95, 0.9, 0.02)[0] == 1.0
