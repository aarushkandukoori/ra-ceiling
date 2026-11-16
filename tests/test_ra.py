import numpy as np

from ra_ceiling.ra import (alignment, jsd, mutual_info, perm_baseline, ra_score,
                           redundancy_weights, hist)
from ra_ceiling.world import (World, chow_liu_tree, dropped_sample, perfect_sample,
                              shifted_sample, zipf)


def test_jsd_bounds_and_symmetry():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    assert abs(jsd(p, p)) < 1e-12
    assert abs(jsd(p, q) - 1.0) < 1e-12          # disjoint support = 1 (base 2)
    r = np.array([0.2, 0.3, 0.5])
    assert abs(jsd(p, r) - jsd(r, p)) < 1e-12


def test_jsd_known_value():
    # JS divergence of (1,0) vs (0.5,0.5) in bits is 0.311278; distance is its sqrt
    p, q = np.array([1.0, 0.0]), np.array([0.5, 0.5])
    m = (p + q) / 2
    div = 0.5 * np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0])) + 0.5 * np.sum(q * np.log2(q / m))
    assert abs(jsd(p, q) - np.sqrt(div)) < 1e-12
    assert abs(div - 0.311278) < 1e-5


def test_baseline_two_values_by_hand():
    # With two values there are only two shuffles: keep or swap, each half the time.
    rng = np.random.default_rng(0)
    p, o = np.array([0.9, 0.1]), np.array([0.7, 0.3])
    exact = 0.5 * jsd(p, o) + 0.5 * jsd(p, o[::-1])
    assert abs(perm_baseline(p, o, 50_000, rng) - exact) < 0.01


def test_alignment_identical_is_one_and_flat_is_zero():
    rng = np.random.default_rng(1)
    p = zipf(10, 1.0)
    assert alignment(p, p, 2000, rng)[0] == 1.0
    flat = np.ones(10) / 10                      # every shuffle of a flat O is itself
    assert alignment(p, flat, 2000, rng)[0] == 0.0


def test_duplicate_field_is_discounted():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 5, size=50_000)
    b = rng.integers(0, 5, size=50_000)
    prod = np.stack([a, a.copy(), b], axis=1)     # field 1 copies field 0
    w = redundancy_weights(prod, [5, 5, 5])
    assert w.max() == 1.0
    assert sorted(np.round(w, 2)) == [0.2, 1.0, 1.0]   # the copy keeps 1 - 0.8 = 0.2


def test_mutual_info_of_independent_columns_is_near_zero():
    rng = np.random.default_rng(3)
    x, y = rng.integers(0, 4, 200_000), rng.integers(0, 4, 200_000)
    assert mutual_info(x, y, 4, 4) < 1e-3


def test_perfect_large_sample_scores_near_one_and_shift_scores_lower():
    rng = np.random.default_rng(4)
    w = World(seed=0)
    prod = w.sample(200_000, rng)
    eff = redundancy_weights(prod, w.ks)
    P = [hist(prod[:, i], k) for i, k in enumerate(w.ks)]
    big = perfect_sample(prod, 100_000, rng)
    ra_big = ra_score(P, [hist(big[:, i], k) for i, k in enumerate(w.ks)], eff, 500, rng)[0]
    assert ra_big > 0.97
    sh = shifted_sample(prod, w, 100_000, 0.3, rng)
    ra_sh = ra_score(P, [hist(sh[:, i], k) for i, k in enumerate(w.ks)], eff, 500, rng)[0]
    assert ra_sh < ra_big - 0.2


def test_dropped_value_is_absent():
    rng = np.random.default_rng(5)
    w = World(seed=0)
    prod = w.sample(50_000, rng)
    X = dropped_sample(prod, 2000, field=0, value=3, rng=rng)
    assert not (X[:, 0] == 3).any()


def test_chow_liu_recovers_true_tree():
    rng = np.random.default_rng(6)
    w = World(tree=True, alpha=0.6, seed=0)
    prod = w.sample(200_000, rng)
    parent, _ = chow_liu_tree(prod, w.ks)
    true_edges = {frozenset((i, p)) for i, p in enumerate(w.parent) if p >= 0}
    fit_edges = {frozenset((i, p)) for i, p in enumerate(parent) if p >= 0}
    assert fit_edges == true_edges
