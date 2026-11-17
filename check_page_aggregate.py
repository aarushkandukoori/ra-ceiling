"""Check the page's aggregates-only mode against ra_ceiling/aggregate.py.

python3 check_page_aggregate.py    (needs node) -> results/page_check_aggregate.json
"""
import json
import re
import subprocess

import numpy as np

from ra_ceiling.aggregate import field_ceiling, field_ceiling_mc, self_baseline
from ra_ceiling.world import zipf

SETTINGS = [(1000, 25, 1.0), (100, 6, 2.0), (5000, 4, 0.25), (200, 11, 1.5), (1000, 8, 0.5)]
R = 400

html = open("docs/index.html").read()
js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
open("results/_page_core.js", "w").write(js)
node = ("const m=require('./results/_page_core.js');"
        f"const S={json.dumps(SETTINGS)};"
        "console.log(JSON.stringify(S.map(([n,k,s],i)=>{const r=m.mulberry32(500+i);const p=m.zipf(k,s);"
        f"const B=m.selfBaseline(p,20000,r);return {{B, closed:m.fieldCeiling(p,n,B), sim:m.fieldCeilingSim(p,n,{R},500,r)}};}})));")
page = json.loads(subprocess.check_output(["node", "-e", node]))

rng = np.random.default_rng(7)
rows = []
for (n, k, s), pg in zip(SETTINGS, page):
    p = zipf(k, s)
    B = self_baseline(p, 50_000, rng)
    cm, cs = field_ceiling(p, n, B)
    mm, ms, _ = field_ceiling_mc(p, n, R, 2000, rng)
    se = ms / np.sqrt(R) * np.sqrt(2)
    row = {"n": n, "K": k, "s": s,
           "python": {"B": B, "closed_mean": cm, "closed_sd": cs, "sim_mean": mm, "sim_sd": ms},
           "page": {"B": pg["B"], "closed_mean": pg["closed"]["mean"], "closed_sd": pg["closed"]["sd"],
                    "sim_mean": pg["sim"]["mean"], "sim_sd": pg["sim"]["sd"]}}
    row["closed_mean_diff"] = abs(cm - pg["closed"]["mean"])
    row["sim_mean_diff"] = abs(mm - pg["sim"]["mean"])
    row["sim_within_3se"] = bool(row["sim_mean_diff"] <= 3 * se + 1e-9)
    rows.append(row)
    print({kk: row[kk] for kk in ("n", "K", "s", "closed_mean_diff", "sim_mean_diff", "sim_within_3se")})
json.dump(rows, open("results/page_check_aggregate.json", "w"), indent=1)
print("closed max diff:", max(r["closed_mean_diff"] for r in rows),
      "| sim all within 3 SE:", all(r["sim_within_3se"] for r in rows))
