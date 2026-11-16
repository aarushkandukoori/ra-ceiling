"""Draw the three figures from results/results.json.   python3 make_figures.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("results/results.json"))
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]   # Okabe-Ito, colorblind-safe
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def band(ax, x, stats, color, label, marker="o"):
    m = [s["mean"] for s in stats]
    lo = [s["p05"] for s in stats]
    hi = [s["p95"] for s in stats]
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
    ax.plot(x, m, marker=marker, ms=3.5, color=color, label=label)


# ---- Figure 1: the ceiling ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
ns = r["settings"]["n_grid"]
for c, (s, lab) in zip(C, [("0.25", "near-flat fields (s = 0.25)"), ("1.0", "Zipf fields (s = 1)"),
                           ("2.0", "skewed fields (s = 2)")]):
    band(axs[0], ns, [r["E1_ceiling_vs_n"][s][str(n)] for n in ns], c, lab)
axs[0].axhline(2 / 3, color="gray", ls=":", lw=1)
axs[0].text(110, 0.68, "good band starts (2/3)", fontsize=7, color="gray")
axs[0].set(xscale="log", xlabel="eval-set size n (records)", ylabel="RA of a perfect sample",
           title="A. All 21 fields", ylim=(0, 1))
axs[0].legend(fontsize=7, loc="lower right", frameon=False)
for c, (key, lab) in zip(C[2:] + C[:2], [("K4_s0.25", "4 values, near-flat"), ("K4_s2.0", "4 values, skewed"),
                                         ("K25_s0.25", "25 values, near-flat"), ("K25_s2.0", "25 values, skewed")]):
    band(axs[1], ns, [r["E1_single_field"][key][str(n)] for n in ns], c, lab)
axs[1].set(xscale="log", xlabel="eval-set size n (records)", ylabel="field alignment a of a perfect sample",
           title="B. One field", ylim=(0, 1))
axs[1].legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("figures/fig1_ceiling.png", dpi=200)

# ---- Figure 2: skew and the shuffle baseline (n = 1000) ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
sk = r["E2_skew"]["skews"]
for c, k in zip([C[0], C[3]], ["K6", "K25"]):
    e = r["E2_skew"][k]
    band(axs[0], sk, e["baseline"], c, f"{k[1:]} values: shuffle baseline")
    band(axs[0], sk, e["perfect_d"], c, f"{k[1:]} values: perfect-sample distance", marker="s")
    axs[0].lines[-1].set_linestyle("--")
axs[0].set(xlabel="skew s (Zipf exponent)", ylabel="JS distance to production",
           title="A. Baseline grows with skew", ylim=(0, 1))
axs[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
for c, k in zip([C[0], C[3]], ["K6", "K25"]):
    e = r["E2_skew"][k]
    band(axs[1], sk, e["ceiling_a"], c, f"{k[1:]} values: perfect sample")
    band(axs[1], sk, e["head_right_tail_scrambled_a"], c, f"{k[1:]} values: top value right, rest shuffled",
         marker="s")
    axs[1].lines[-1].set_linestyle("--")
axs[1].set(xlabel="skew s (Zipf exponent)", ylabel="field alignment a",
           title="B. What a field scores", ylim=(0, 1))
axs[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig2_skew.png", dpi=200)

# ---- Figure 3: shifts of known size ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
lams = r["E3_shift"]["lams"]
for c, n in zip(C, ["200", "1000", "5000"]):
    band(axs[0], lams, r["E3_shift"][n]["raw"], c, f"n = {n}")
    band(axs[1], lams, r["E3_shift"][n]["calibrated"], c, f"n = {n}")
axs[0].set(xlabel="share of drifted records (lambda)", ylabel="RA", title="A. RA as defined", ylim=(0, 1))
axs[1].set(xlabel="share of drifted records (lambda)", ylabel="noise-adjusted RA",
           title="B. Sampling noise removed", ylim=(0, 1))
for ax in axs:
    ax.legend(fontsize=7, loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("figures/fig3_shift.png", dpi=200)
print("saved 3 figures")
