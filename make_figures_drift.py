"""Figure for the drift-kind check (reads results/drift_types.json).

python3 make_figures_drift.py -> figures/fig6_drift_types.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

o = json.load(open("results/drift_types.json"))
N = "1000"
# Okabe-Ito colors
COL = {"reverse_all": "#D55E00", "swap_top2_all": "#0072B2", "flatten_all": "#009E73",
       "reverse_3": "#CC79A7", "reverse_1": "#E69F00", "spike_1": "#000000"}
LAB = {"reverse_all": "all fields, ranks reversed", "swap_top2_all": "all fields, top 2 swapped",
       "flatten_all": "all fields, flattened", "reverse_3": "3 fields, ranks reversed",
       "reverse_1": "1 field, ranks reversed", "spike_1": "1 field, rarest value spikes"}
MK = {"reverse_all": "o", "swap_top2_all": "s", "flatten_all": "^", "reverse_3": "D",
      "reverse_1": "v", "spike_1": "x"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
C, S = o[f"ceiling_n{N}"]["C"], o[f"ceiling_n{N}"]["S"]
ax[0].axhspan(C - 1.645 * S, 1.0, color="0.85", lw=0)
ax[0].axhline(C, color="0.4", lw=0.8, ls=":")
for k, by_n in o["kinds"].items():
    rows = by_n[N]
    lam = [100 * r["lam"] for r in rows]
    ax[0].plot(lam, [r["RA_mean"] for r in rows], "-", marker=MK[k], color=COL[k], ms=4, lw=1.3, label=LAB[k])
    ax[1].plot(lam, [r["flag_total"] for r in rows], "-", marker=MK[k], color=COL[k], ms=4, lw=1.3)
    ax[1].plot(lam, [r["flag_any_field"] for r in rows], "--", marker=MK[k], color=COL[k], ms=4, lw=1.0,
               mfc="none")
ax[0].set_ylim(0.6, 0.95)
ax[0].set_xlabel("drifted records in the eval set (%)"); ax[0].set_ylabel("total RA (mean of 200 sets)")
ax[0].set_title("A. Total RA by kind of drift, n = 1,000", loc="left", fontsize=9)
ax[0].legend(fontsize=6.3, frameon=False, loc="lower left")
ax[1].plot([], [], "k-", label="total gap test")
ax[1].plot([], [], "k--", mfc="none", label="largest per-field gap test")
ax[1].axhline(0.05, color="0.4", lw=0.8, ls=":")
ax[1].set_ylim(-0.02, 1.02)
ax[1].set_xlabel("drifted records in the eval set (%)"); ax[1].set_ylabel("share of eval sets flagged")
ax[1].set_title("B. Which test catches it", loc="left", fontsize=9)
ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/fig6_drift_types.png", dpi=200)
print("saved figures/fig6_drift_types.png")
