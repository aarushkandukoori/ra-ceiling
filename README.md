# ra-ceiling

What RA score does a perfect eval set get?

The ProxyDrift paper (Levit et al., arXiv 2608.08245, Sec. 4) scores an eval set against production one field at a time with Jensen-Shannon distance, rescales so 1 = identical and 0 = a random shuffle of the eval set's own value shares, and averages over 21 fields with a redundancy discount (the RA score). The shuffle fixes where 0 is. This repo asks where the top is: an eval set of n records drawn perfectly at random from production still differs from production by chance, so it scores below 1. How far below depends on n, on how many values a field has, and on skew.

This is an independent simulation. It is not affiliated with the paper's authors or with Microsoft, and it uses no real user data.

Interactive page: https://aarushkandukoori.github.io/ra-ceiling/

## Main numbers

All from `results/results.json` (seed 20261116). Fields use the paper's 21 field names and value counts (Table 2); every value shape is ASSUMED (Zipf, exponent s).

- **The ceiling moves with n.** A perfect random sample scores RA 0.736 at n = 100, 0.916 at n = 1,000 (5-95%: 0.907 to 0.924) and 0.973 at n = 10,000 (Zipf fields, s = 1). With near-flat fields (s = 0.25) a perfect 1,000-record sample scores only 0.678.
- **Small drift hides below the ceiling at small n.** Mixing in records from a drifted world (ranks reversed): the average RA drops below the perfect-sample 5th percentile once 4% of records are drifted at n = 200, 3% at n = 1,000 and 1% at n = 5,000. For the drifted set's 95th percentile to clear that line takes 10%, 4% and 2%.
- **Skew makes the shuffle baseline generous.** For a 25-value field, an eval set that gets only the top value's share right (the rest shuffled) scores 0.22 when the field is near-flat (s = 0.25) and 0.62 when it is strongly skewed (s = 3, top value = 83% of records). A perfectly flat production field scores 0 for any eval set, because every shuffle is equally close.
- **A noise-adjusted score is steadier across n.** Subtracting the expected perfect-sample distance (in squared units) before rescaling: at 10% drift, plain RA is 0.741 / 0.803 / 0.815 at n = 200 / 1,000 / 5,000, the adjusted score is 0.860 / 0.828 / 0.820.

Also in the results: dropping a value that holds 2.6% of production from one field leaves total RA at 0.910 (inside the perfect-sample band) while that field's own score falls to 0.762 (its perfect-sample band is 0.848 to 0.908). With fields linked by a tree, RA (marginals only) scores an independent sampler and a Chow-Liu sampler about the same (0.801 vs 0.799, perfect sample 0.803); a pairwise score separates them (mean pairwise JS distance 0.110 vs 0.101, perfect 0.101). Hydration and re-tagging are not simulated.

## Aggregates only (no record sampling)

Behind a compliance boundary you may only see aggregated distributions, so you cannot draw a perfect sample of production records. The ceiling can still be computed from those aggregates. For one field it depends only on that field's production value shares p and the eval-set size n:

- E[d] = m(K-1) / sqrt(8 n ln 2), sd[d] = sqrt((K-1) - m(K-1)^2) / sqrt(8 n ln 2), where d is the JS distance of a perfect sample, K counts values with a nonzero share, and m(k) is the mean of the square root of a chi-square with k degrees of freedom (second-order expansion of JS divergence).
- Ceiling c(n) = 1 - E[d] / b(n), spread sd[d] / b(n), with b(n) = sqrt(B^2 + (K-1)/(8 n ln 2)) and B = mean JS distance between p and a shuffle of p.
- Or simulate: draw multinomial(n, p) histograms and score them. This also needs only p.

Results (`results/aggregate.json`, seed 20261117, from `run_aggregate.py`):

- **Per field**, the formula is within 0.010 of simulation for n >= 1,000 (48 of 84 settings, K = 4 to 25; median error 0.0016 over all 84). Worst case 0.040, a 25-value field at n = 100, so simulate there.
- **All 21 fields** (Zipf s = 1): formula 0.916 at n = 1,000 and 0.742 at n = 100; perfect samples of the production records give 0.915 and 0.733.
- **Noise-adjusted score** = min(1, RA / ceiling at the same n), and gap = (ceiling - RA) / ceiling sd. RA fixes the bottom of the scale (a shuffle = 0); this also fixes the top (a perfect sample of the same size = 1). Where they disagree: at n = 1,000 a perfect sample of near-flat fields scores RA 0.679 ("average") and a set with 10% drifted records on skewed fields scores 0.766 ("good"); the adjusted scores are 0.988 and 0.807. Same flip with size: perfect n = 100 set RA 0.734 vs 10%-drifted n = 1,000 set 0.800 (adjusted 0.990 vs 0.874). It does not fix the skew generosity of the shuffle baseline.
- **What marginals cannot give.** The per-field score and ceiling need only marginals. The redundancy discount g_i needs pairwise mutual information, so pairwise tables (the paper's pipeline exports these, Sec. 6). The spread of total RA depends on how fields co-vary; `ra_spread_pairwise` gets it from pairwise tables only. In the tree-linked worlds here, assuming fields independent was within 25% of the spread from records at n = 1,000 ( sd 0.0149 vs 0.0144 from records at alpha = 0.6; 0.0375 vs 0.0300 at alpha = 0.9), and the largest possible spread from marginals alone is 4 to 5 times too wide at n = 1,000. Pairwise tables helped most at small n with strong links (n = 100, alpha = 0.9: 1.5 times the record spread vs 2.0 times assuming independence). Anything that depends on three or more fields at once (whether a combination of values is covered) is not recoverable from marginals or pairs.
- **Sampling gap vs rehydration.** Taking a rehydration loss of 0.10 as an illustrative value, the sampling gap 1 - c(n) for s = 1 fields is 0.265 at n = 100, 0.120 at 500, 0.085 at 1,000 and 0.027 at 10,000: sampling dominates below roughly 500 to 1,000 records, rehydration above. For near-flat fields sampling dominates up to 10,000 (0.106).

Page: the "Aggregates only mode" panel takes one field's production shares (and optionally eval counts) and returns both ceilings, the RA field score, the noise-adjusted score and the gap. `check_page_aggregate.py` checks it against Python (`results/page_check_aggregate.json`).

## Notation and one worked field

Everything is computed one field at a time, then combined. A field is one proxy dimension (for example `query_formality`); its values are its categories. One field's shares are the fraction of records that take each value of that field, so they sum to 1: production shares p_v, eval-set shares o_v. K is the number of values with p_v > 0, n is the eval-set size, and p_hat_v is the share of value v in a perfect sample (n records drawn at random from production), a hypothetical eval set used only to get the ceiling.

Worked example (`worked_example.py`, results in `results/worked_example.json`): a 4-value field with p = (0.48, 0.24, 0.16, 0.12) (ASSUMED Zipf s = 1) and an eval set of 200 records with counts 116, 38, 28, 18. JS distance 0.087, shuffle baseline 0.348, field alignment a = 0.750. A perfect 200-record sample scores 0.837 (sd 0.067 from 2,000 multinomial draws; 0.837 and 0.069 by the closed formula), so the noise-adjusted field score is 0.896 and the gap is 1.30, under the 1.645 line.

## How drift is added, and does its kind matter

Drift is a mixture. An eval set is a perfect random sample of production in which each record, with probability lam, is swapped for a drifted record. A drifted record draws each drifted field from a fixed distribution q_i (fields independent) and keeps production's distribution for the others, so a field's expected eval shares are (1 - lam) p_i + lam q_i. No noise is added to the generating distribution. The earlier results use q_i = p_i with value ranks reversed, in all 21 fields.

`run_drift_types.py` (results in `results/drift_types.json`, seed 20261118, about 2 minutes, figure 6) compares six ASSUMED kinds of q at n = 200 and 1,000, Zipf s = 1, 200 eval sets per setting: all fields reversed, all fields with the top two values swapped, all fields flattened, 3 fields reversed, 1 field (`intent_types`) reversed, and 1 field whose rarest value spikes. Two tests: the total gap test (gap > 1.645) and a per-field test (largest per-field gap above its 95th percentile in 400 perfect sets drawn from marginals). Perfect sets are flagged 2 to 7% of the time by either test.

- **The kind matters a lot.** At n = 1,000, 10% drifted records lower total RA by 0.116 when every field reverses and by 0.005 when only one field does.
- **Total RA averages concentrated drift away.** One reversed field at 10%: the total test flags 23% of sets, the per-field test 97%. Mild drift spread over all fields goes the other way (top-two swap at 10%: total 95%, per-field 56%). Reporting both is safer than either alone.
- **Small sets are harder.** At n = 200, one reversed field is flagged by the per-field test in 9% of sets at 10% drift and 72% at 20%.
- **Not tested.** Drift that changes how fields co-occur while every marginal stays the same scores exactly like no drift under RA and both tests, since all of them read marginals only. Also not tested: new values outside the taxonomy, real traffic shifts, multi-valued fields, tagging error, linked fields in this check.

## What is simulated and what is not

- Simulated: proxy records with the paper's 21 fields and value counts, a production set of 1,000,000 records, eval sets drawn from it, RA as in eqs. 1-7.
- Assumed (not in the paper): Zipf value shapes, equal field weights, plug-in mutual information on production, every field single-valued, the drift (ranks reversed; six kinds in `run_drift_types.py`), the dependency tree.
- Not simulated: LLM tagging, tagging errors, hydration into text, re-tagging, multi-valued fields, real traffic.
- The sweeps use 2,000 shuffles per baseline instead of the paper's 50,000; on three checks the RA difference was at most 0.0002 (`check_T` in the results).

## Files

- `ra_ceiling/ra.py` - JS distance, shuffle baseline, alignment, redundancy weights, RA
- `ra_ceiling/world.py` - the synthetic world, eval sets, independent and Chow-Liu samplers
- `run.py` - every experiment; writes `results/results.json`
- `make_figures.py` - the three figures in `figures/`
- `check_page.py` - checks the page's numbers against Python (`results/page_check.json`)
- `ra_ceiling/aggregate.py`, `run_aggregate.py`, `make_figures_aggregate.py`, `check_page_aggregate.py` - the aggregates-only analysis (results in `results/aggregate.json`, figures 4 and 5)
- `run_drift_types.py`, `make_figures_drift.py` - the drift-kind check (results in `results/drift_types.json`, figure 6)
- `worked_example.py` - one field worked through (results in `results/worked_example.json`)
- `docs/index.html` - the interactive page (one field)
- `tests/` - unit tests

## Rerun

```
pip install numpy matplotlib pytest
python3 -m pytest -q
python3 run.py            # a few minutes, numpy only
python3 make_figures.py
python3 check_page.py     # needs node
python3 run_aggregate.py  # about 2 minutes
python3 make_figures_aggregate.py
python3 check_page_aggregate.py
python3 run_drift_types.py  # about 2 minutes
python3 make_figures_drift.py
python3 worked_example.py
```

Made by Aarush Kandukoori
