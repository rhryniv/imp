"""F7 (fig-sweep): E2 feasibility over the (n,L) sweep, both specs, plus
the prop:sweep cap n < 2*(pi+V_J)/|J| from E4, checked against whether the
observed feasible L values respect it.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from fig_style import FIGSIZE_WIDE, save

OUT = "../figures"

with open("../data/e2_results.json") as f:
    e2 = json.load(f)
with open("../data/e4_results.json") as f:
    e4 = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_WIDE[0], 2.3), sharey=True)
colors = {1: "#0072B2", -1: "#D55E00"}
markers = {1: "o", -1: "^"}

for ax, name in zip(axes, ["A", "B"]):
    for n in range(1, 7):
        cells = e2[name][str(n)]
        for key, cell in cells.items():
            L = int(key.split("_")[0][1:])
            sigma = int(key.split("_s")[1])
            feas = cell.get("feasible", False)
            if feas:
                ax.scatter([n], [L], color=colors[sigma], marker=markers[sigma],
                           s=18, zorder=3, edgecolors="none")
            else:
                ax.scatter([n], [L], color="#CCCCCC", marker=".", s=6, zorder=1)
    # prop:sweep cap curve, n < 2*(pi+V_J)/|J|  -> plotted as the boundary L=cap(n)
    ns_cap = []
    caps = []
    for n in range(1, 10):
        r = e4[name].get(str(n))
        if r and "sweep_cap_n" in r:
            ns_cap.append(n)
            caps.append(r["sweep_cap_n"])
    ax.plot(ns_cap, caps, color="k", lw=0.9, ls="--", zorder=2, label="sweep cap")
    ax.set_xlabel(r"$n$")
    ax.set_title(f"Spec {name}", fontsize=8.5, y=1.0)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(1, 13)

axes[0].set_ylabel(r"$L$")
handles = [plt.Line2D([0], [0], marker="o", color=colors[1], lw=0, label=r"feasible, $\sigma=+1$"),
           plt.Line2D([0], [0], marker="^", color=colors[-1], lw=0, label=r"feasible, $\sigma=-1$"),
           plt.Line2D([0], [0], color="k", lw=0.9, ls="--", label="sweep cap")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=7, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
save(fig, f"{OUT}/fig-sweep")
print("saved fig-sweep")

# check: do observed feasible L respect the cap (L < cap(n))?
print("\nprop:sweep cap check (observed feasible L vs cap(n)):")
for name in ["A", "B"]:
    for n in range(1, 7):
        r = e4[name].get(str(n))
        if not r:
            continue
        cap = r["sweep_cap_n"]
        cells = e2[name][str(n)]
        feas_Ls = [int(k.split("_")[0][1:]) for k, c in cells.items() if c.get("feasible")]
        if feas_Ls:
            ok = all(L < cap for L in feas_Ls)
            print(f"  Spec {name} n={n}: feasible L={feas_Ls}, cap={cap:.2f}  respects_cap={ok}")
