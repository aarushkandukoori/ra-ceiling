"""A synthetic world of proxy records shaped like the published schema.

What comes from the paper (Table 2): the 21 field names and how many values each has.
What is ASSUMED (not in the paper): every value shape, the dependency tree between
fields, the shift used for drift, and that every field is single-valued.
"""
import numpy as np

# Field name -> number of values, from Table 2 of arXiv 2608.08245.
SCHEMA = [
    ("intent_types", 25), ("query_formality", 4), ("query_structure", 4),
    ("query_technical", 4), ("query_engagement", 5), ("query_constraints", 5),
    ("tone_sentiment", 6), ("tone_emotion", 9), ("tone_urgency", 5),
    ("output_type", 11), ("output_format", 6), ("output_length", 5),
    ("output_components", 6), ("output_creativity", 4), ("domain_names", 21),
    ("content_specificity", 4), ("content_embedded", 5), ("char_count_bucket", 6),
    ("word_count_bucket", 6), ("query_language", 11), ("grounding_explicit", 20),
]
NAMES = [n for n, _ in SCHEMA]
KS = [k for _, k in SCHEMA]


def zipf(k, s):
    """Zipf-shaped probabilities over k values. s = 0 is flat; bigger s = more skew. (ASSUMED shape)"""
    w = 1.0 / np.arange(1, k + 1) ** s
    return w / w.sum()


def random_tree(D, rng):
    """A random tree over D fields: field 0 is the root, every other field gets an earlier parent. (ASSUMED)"""
    parent = [-1]
    for i in range(1, D):
        parent.append(int(rng.integers(0, i)))
    return parent


class World:
    """Ground truth over the 21 fields.

    Independent world: each field is Zipf(k, s), drawn on its own.
    Tree world: field i given its parent's value v is
        (1 - alpha) * Zipf(k, s) + alpha * (Zipf(k, s) with its ranks rotated by v + 1).
    """

    def __init__(self, ks=KS, s=1.0, tree=False, alpha=0.6, seed=0):
        self.ks = list(ks)
        self.D = len(self.ks)
        self.s = s
        self.tree = tree
        self.alpha = alpha
        rng = np.random.default_rng(seed)
        self.parent = random_tree(self.D, rng) if tree else [-1] * self.D
        self.base = [zipf(k, s) for k in self.ks]

    def cond(self, i, v):
        """Distribution of field i given its parent has value v (tree world only)."""
        b = self.base[i]
        return (1 - self.alpha) * b + self.alpha * np.roll(b, v + 1)

    def sample(self, n, rng, shifted=False):
        """Draw n records (n x D int array). shifted=True draws from the drifted world:
        every field's ranks reversed, so rare values become common (ASSUMED shift)."""
        X = np.zeros((n, self.D), dtype=np.int64)
        for i in range(self.D):
            k = self.ks[i]
            if shifted:
                X[:, i] = rng.choice(k, size=n, p=self.base[i][::-1])
            elif self.parent[i] < 0:
                X[:, i] = rng.choice(k, size=n, p=self.base[i])
            else:
                pv = X[:, self.parent[i]]
                for v in np.unique(pv):
                    idx = np.flatnonzero(pv == v)
                    X[idx, i] = rng.choice(k, size=idx.size, p=self.cond(i, v))
        return X


def perfect_sample(prod, n, rng):
    """No-drift control: n records drawn at random from production (with replacement)."""
    return prod[rng.integers(0, prod.shape[0], size=n)]


def shifted_sample(prod, world, n, lam, rng):
    """Eval set with drift of known size: each record comes from the shifted world
    with probability lam, otherwise from production."""
    X = perfect_sample(prod, n, rng)
    mask = rng.random(n) < lam
    if mask.any():
        X[mask] = world.sample(int(mask.sum()), rng, shifted=True)
    return X


def dropped_sample(prod, n, field, value, rng):
    """Eval set that never contains one value of one field (a coverage gap)."""
    keep = prod[prod[:, field] != value]
    return keep[rng.integers(0, keep.shape[0], size=n)]


def independent_sampler(prod, ks, n, rng):
    """Draw each field from production's own marginal, independently."""
    X = np.zeros((n, len(ks)), dtype=np.int64)
    for i, k in enumerate(ks):
        p = np.bincount(prod[:, i], minlength=k) / prod.shape[0]
        X[:, i] = rng.choice(k, size=n, p=p)
    return X


def chow_liu_tree(prod, ks):
    """Fit a Chow-Liu tree on production: max-weight spanning tree on pairwise MI
    (MI damped by the paper's eq. 10 factor N / (N + 5 k_i k_j)), rooted at the
    highest-degree node. Returns (parent list, root)."""
    from .ra import mutual_info
    D = len(ks)
    N = prod.shape[0]
    edges = []
    for i in range(D):
        for j in range(i + 1, D):
            mi = mutual_info(prod[:, i], prod[:, j], ks[i], ks[j])
            damp = N / (N + 5 * ks[i] * ks[j])
            edges.append((mi * damp, i, j))
    edges.sort(reverse=True)
    root_of = list(range(D))

    def find(x):
        while root_of[x] != x:
            root_of[x] = root_of[root_of[x]]
            x = root_of[x]
        return x

    adj = {i: [] for i in range(D)}
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            root_of[ri] = rj
            adj[i].append(j)
            adj[j].append(i)
    root = max(range(D), key=lambda x: len(adj[x]))
    parent = [-1] * D
    order = [root]
    seen = {root}
    for x in order:
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                parent[y] = x
                order.append(y)
    return parent, order


def chow_liu_sampler(prod, ks, n, rng, fit=None):
    """Sample n records from a Chow-Liu tree fit on production, using the empirical
    conditionals P(child | parent). (The paper's shrinkage toward the marginal is
    skipped: with 10^6 production records it changes almost nothing.)"""
    parent, order = fit if fit is not None else chow_liu_tree(prod, ks)
    X = np.zeros((n, len(ks)), dtype=np.int64)
    for i in order:
        k = ks[i]
        if parent[i] < 0:
            p = np.bincount(prod[:, i], minlength=k) / prod.shape[0]
            X[:, i] = rng.choice(k, size=n, p=p)
            continue
        pj = parent[i]
        for v in np.unique(X[:, pj]):
            idx = np.flatnonzero(X[:, pj] == v)
            rows = prod[prod[:, pj] == v, i]
            p = np.bincount(rows, minlength=k) / rows.size
            X[idx, i] = rng.choice(k, size=idx.size, p=p)
    return X
