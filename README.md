# ra-ceiling

What RA score does a perfect eval set get?

The ProxyDrift paper (Levit et al., arXiv 2608.08245, Sec. 4) scores an eval set against production one field at a time with Jensen-Shannon distance, rescales so 1 = identical and 0 = a random shuffle of the eval set's own value shares, and averages over 21 fields with a redundancy discount (the RA score). The shuffle fixes where 0 is. This repo asks where the top is: an eval set of n records drawn perfectly at random from production still differs from production by chance, so it scores below 1. How far below depends on n, on how many values a field has, and on skew.

This is an independent simulation. It is not affiliated with the paper's authors or with Microsoft, and it uses no real user data.

Interactive page: https://aarushkandukoori.com/ra-ceiling/

## Main numbers

All from `results/results.json` (seed 20261116). Fields use the paper's 21 field names and value counts (Table 2); every value shape is ASSUMED (Zipf, exponent s).

- **The ceiling moves with n.** A perfect random sample scores RA 0.736 at n = 100, 0.916 at n = 1,000 (5-95%: 0.907 to 0.924) and 0.973 at n = 10,000 (Zipf fields, s = 1). With near-flat fields (s = 0.25) a perfect 1,000-record sample scores only 0.678.
- **Small drift hides below the ceiling at small n.** Mixing in records from a drifted world (ranks reversed): the average RA drops below the perfect-sample 5th percentile once 4% of records are drifted at n = 200, 3% at n = 1,000 and 1% at n = 5,000. For the drifted set's 95th percentile to clear that line takes 10%, 4% and 2%.
- **Skew makes the shuffle baseline generous.** For a 25-value field, an eval set that gets only the top value's share right (the rest shuffled) scores 0.22 when the field is near-flat (s = 0.25) and 0.62 when it is strongly skewed (s = 3, top value = 83% of records). A perfectly flat production field scores 0 for any eval set, because every shuffle is equally close.
- **A noise-adjusted score is steadier across n.** Subtracting the expected perfect-sample distance (in squared units) before rescaling: at 10% drift, plain RA is 0.741 / 0.803 / 0.815 at n = 200 / 1,000 / 5,000, the adjusted score is 0.860 / 0.828 / 0.820.

Also in the results: dropping a value that holds 2.6% of production from one field leaves total RA at 0.910 (inside the perfect-sample band) while that field's own score falls to 0.762 (its perfect-sample band is 0.848 to 0.908). With fields linked by a tree, RA (marginals only) scores an independent sampler and a Chow-Liu sampler about the same (0.801 vs 0.799, perfect sample 0.803); a pairwise score separates them (mean pairwise JS distance 0.110 vs 0.101, perfect 0.101). Hydration and re-tagging are not simulated.

## What is simulated and what is not

- Simulated: proxy records with the paper's 21 fields and value counts, a production set of 1,000,000 records, eval sets drawn from it, RA as in eqs. 1-7.
- Assumed (not in the paper): Zipf value shapes, equal field weights, plug-in mutual information on production, every field single-valued, the drift (ranks reversed), the dependency tree.
- Not simulated: LLM tagging, tagging errors, hydration into text, re-tagging, multi-valued fields, real traffic.
- The sweeps use 2,000 shuffles per baseline instead of the paper's 50,000; on three checks the RA difference was at most 0.0002 (`check_T` in the results).

## Files

- `ra_ceiling/ra.py` - JS distance, shuffle baseline, alignment, redundancy weights, RA
- `ra_ceiling/world.py` - the synthetic world, eval sets, independent and Chow-Liu samplers
- `run.py` - every experiment; writes `results/results.json`
- `make_figures.py` - the three figures in `figures/`
- `check_page.py` - checks the page's numbers against Python (`results/page_check.json`)
- `docs/index.html` - the interactive page (one field)
- `tests/` - unit tests

## Rerun

```
pip install numpy matplotlib pytest
python3 -m pytest -q
python3 run.py            # a few minutes, numpy only
python3 make_figures.py
python3 check_page.py     # needs node
```

Made by Aarush Kandukoori
