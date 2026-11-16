"""Check that the page (docs/index.html) gives the same numbers as the Python code.

Runs the page's own simulate() with node, runs the same one-field setup in Python,
and saves both to results/page_check.json.   python3 check_page.py   (needs node)
"""
import json
import re
import subprocess

import numpy as np

from ra_ceiling.ra import alignment
from ra_ceiling.world import zipf

SETTINGS = [(1000, 25, 1.0, 0.0), (1000, 25, 1.0, 0.1), (200, 6, 2.0, 0.05),
            (5000, 4, 0.25, 0.2), (100, 11, 1.5, 0.3)]
R = 400

html = open("docs/index.html").read()
js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
open("results/_page_core.js", "w").write(js)
node_code = ("const m=require('./results/_page_core.js');"
             f"const S={json.dumps(SETTINGS)};"
             f"console.log(JSON.stringify(S.map(([n,k,s,l],i)=>m.simulate(n,k,s,l,{R},500,1000+i))));")
page = json.loads(subprocess.check_output(["node", "-e", node_code]))

rng = np.random.default_rng(99)
rows = []
for (n, k, s, lam), pg in zip(SETTINGS, page):
    p = zipf(k, s)
    mix = (1 - lam) * p + lam * p[::-1]
    ceil = [alignment(p, rng.multinomial(n, p) / n, 2000, rng)[0] for _ in range(R)]
    score = [alignment(p, rng.multinomial(n, mix) / n, 2000, rng)[0] for _ in range(R)]
    py = {"ceil_mean": float(np.mean(ceil)), "score_mean": float(np.mean(score)),
          "ceil_se": float(np.std(ceil) / np.sqrt(R)), "score_se": float(np.std(score) / np.sqrt(R))}
    row = {"n": n, "K": k, "s": s, "lambda": lam, "python": py,
           "page": {"ceil_mean": pg["ceil"]["mean"], "score_mean": pg["score"]["mean"]}}
    # combined standard error of the difference (both sides have about the same spread)
    for key in ("ceil", "score"):
        se = py[f"{key}_se"] * np.sqrt(2)
        diff = abs(py[f"{key}_mean"] - pg[key]["mean"])
        row[f"{key}_diff"] = diff
        row[f"{key}_within_3se"] = bool(diff <= 3 * se + 1e-9)
    rows.append(row)
    print(row)
json.dump(rows, open("results/page_check.json", "w"), indent=1)
print("all within 3 SE:", all(r["ceil_within_3se"] and r["score_within_3se"] for r in rows))
