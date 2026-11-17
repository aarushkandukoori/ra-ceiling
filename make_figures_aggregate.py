"""Figures for the aggregates-only results (reads results/aggregate.json).

python3 make_figures_aggregate.py -> figures/fig4_aggregate_ceiling.png, figures/fig5_adjusted.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

o = json.load(open("results/aggregate.json"))
COL = {"0.25": "#0072B2", "1.0": "#D55E00", "2.0": "#009E73"}
LAB = {"0.25": "near-flat (s = 0.25)", "1.0": "Zipf s = 1", "2.0": "skewed (s = 2)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 4: ceiling from aggregates vs records; sampling gap vs rehydration ----
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for s, rows in o["B_total"].items():
    ns = [r["n"] for r in rows]
    ax[0].plot(ns, [r["closed_mean"] for r in rows], "-", color=COL[s], lw=1.6, label=LAB[s])
    ax[0].plot(ns, [r["sim_from_marginals"]["mean"] for r in rows], "o", color=COL[s], ms=3.5, mfc="none")
    ax[0].plot(ns, [r["records"]["mean"] for r in rows], "x", color=COL[s], ms=4)
    ax[0].fill_between(ns, [r["records"]["p05"] for r in rows], [r["records"]["p95"] for r in rows],
                       color=COL[s], alpha=0.15, lw=0)
    ax[1].plot(ns, [1 - r["sim_from_marginals"]["mean"] for r in rows], "-o", color=COL[s], ms=3, lw=1.4,
               label=LAB[s])
ax[0].plot([], [], "k-", label="closed formula (marginals)")
ax[0].plot([], [], "ko", mfc="none", ms=3.5, label="simulated from marginals")
ax[0].plot([], [], "kx", ms=4, label="perfect samples of records")
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1)
ax[0].set_xlabel("eval-set size n (log scale)"); ax[0].set_ylabel("RA of a perfect sample (21 fields)")
ax[0].set_title("A. Ceiling from aggregates only", loc="left", fontsize=9)
ax[0].legend(fontsize=6.5, frameon=False, loc="lower right")
reh = o["E_gap"]["rehydration_loss"]
ax[1].axhline(reh, color="black", ls="--", lw=1, label="rehydration loss, about 0.10")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 0.75)
ax[1].set_xlabel("eval-set size n (log scale)"); ax[1].set_ylabel("sampling gap = 1 - ceiling")
ax[1].set_title("B. Sampling gap vs rehydration loss", loc="left", fontsize=9)
ax[1].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("figures/fig4_aggregate_ceiling.png", dpi=200)

# ---- Fig 5: RA as defined vs noise-adjusted, same eval sets ----
fig, ax = plt.subplots(figsize=(7.2, 2.9))
MK = {100: "o", 1000: "s", 10000: "^"}
for r in o["D_compare"]:
    s = str(r["s"]); filled = r["lambda"] > 0
    ax.plot(r["RA"]["mean"], r["adjusted"]["mean"], MK[r["n"]], color=COL[s], ms=6,
            mfc=COL[s] if filled else "none", mew=1.3)
for s in COL:
    ax.plot([], [], "o", color=COL[s], label=LAB[s])
for n, m in MK.items():
    ax.plot([], [], m, color="gray", mfc="none", label=f"n = {n:,}")
ax.plot([], [], "o", color="gray", mfc="none", label="no drift (perfect sample)")
ax.plot([], [], "o", color="gray", label="10% drifted records")
for x in (1 / 3, 2 / 3):
    ax.axvline(x, color="gray", lw=0.6, ls=":")
ax.text(0.17, 0.72, "bad", fontsize=7, color="gray", ha="center")
ax.text(0.5, 0.72, "average", fontsize=7, color="gray", ha="center")
ax.text(0.83, 0.72, "good", fontsize=7, color="gray", ha="center")
ax.set_xlim(0, 1); ax.set_ylim(0.7, 1.02)
ax.set_xlabel("RA as defined (chance-calibrated, mean of 100 eval sets)")
ax.set_ylabel("noise-adjusted score")
ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig("figures/fig5_adjusted.png", dpi=200)
print("saved")
