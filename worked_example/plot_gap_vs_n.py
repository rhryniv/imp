"""Figure 2: delta*_n and delta_mag(n) vs n, both geometries, log scale.
Palette used by SERIES (not by N), per the brief's own instruction.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import setup, delta_n

# Okabe-Ito palette, first four colours, assigned by series (not by N)
SERIES_COLORS = {
    "G1 delta_mag": "#0072B2",
    "G1 delta*": "#D55E00",
    "G2 delta_mag": "#009E73",
    "G2 delta*": "#CC79A7",
}
SERIES_STYLES = {
    "G1 delta_mag": ("-", "o"),
    "G1 delta*": ("--", "s"),
    "G2 delta_mag": ("-", "o"),
    "G2 delta*": ("--", "s"),
}

with open("task_c_results.json") as f:
    task_c_results = json.load(f)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, ax = plt.subplots(figsize=(3.4, 3.0))

n_values = [1, 2, 3, 4, 5, 6]
for gname in ["G1", "G2"]:
    geo = setup(gname, 30)
    dmag = [float(delta_n(n, geo)) for n in n_values]
    ls, marker = SERIES_STYLES[f"{gname} delta_mag"]
    ax.semilogy(n_values, dmag, linestyle=ls, marker=marker, markersize=3, linewidth=1.1,
                color=SERIES_COLORS[f"{gname} delta_mag"], label=f"{gname} $\\delta_{{mag}}(n)$")

    dstar = []
    ns_star = []
    for n in n_values:
        entry = task_c_results.get(gname, {}).get(str(n), {})
        if entry.get("delta_star") is not None:
            dstar.append(entry["delta_star"])
            ns_star.append(n)
    ls, marker = SERIES_STYLES[f"{gname} delta*"]
    ax.semilogy(ns_star, dstar, linestyle=ls, marker=marker, markersize=3, linewidth=1.1,
                color=SERIES_COLORS[f"{gname} delta*"], label=f"{gname} $\\delta^*_n$")

ax.set_xlabel("$n$")
ax.set_ylabel(r"$\delta$")
ax.set_xticks(n_values)
ax.legend(frameon=False, fontsize=6.5, loc="best")

fig.tight_layout()
fig.savefig("gap_vs_n.pdf")
fig.savefig("gap_vs_n.png", dpi=200)
print("saved gap_vs_n.pdf/.png")
