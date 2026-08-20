"""Figure 1: example_kappa_TN.pdf -- the retained G1 design. Two stacked
panels: (a) kappa_B with band strip + I0/I1 markings; (b) T_N for
N=1,2,5, log scale, T_env dashed, spec tolerance lines.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import setup
from task_c_core import Q_and_kappa

N_COLORS = {1: "#0072B2", 2: "#D55E00", 5: "#009E73"}
N_STYLES = {1: "-", 2: "--", 5: ":"}
KAPPA_COLOR = "#000000"
BAND_FILL = "#E8E8E8"
I0_COLOR = "#666666"
I1_COLOR = "#BBBBBB"

EPS_0_SPEC = 1e-2
EPS_1_SPEC = 0.0826


def make_figure(gname, n, L, sign, alphas, out_stem):
    geo = setup(gname, 30)
    t, u = float(geo["t"]), float(geo["u"])

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    grid = np.linspace(0.0, np.pi, 4000)
    Q, kap = Q_and_kappa(np.array(alphas), L, grid)
    q2sq = Q - 1.0

    in_band = np.abs(kap) < 1.0
    ax1.axhspan(-1, 1, color=BAND_FILL, zorder=0)
    ax1.plot(grid, kap, color=KAPPA_COLOR, linewidth=1.2)
    ax1.set_ylabel(r"$\kappa_B(\vartheta)$")
    ymin, ymax = kap.min(), kap.max()
    bar_y = ymin - 0.12 * (ymax - ymin)
    ax1.plot([0, t], [bar_y, bar_y], color=I1_COLOR, linewidth=3, solid_capstyle="butt", clip_on=False)
    ax1.plot([u, np.pi], [bar_y, bar_y], color=I0_COLOR, linewidth=3, solid_capstyle="butt", clip_on=False)
    ax1.text(t / 2, bar_y, r"$I_1$", color=I1_COLOR, ha="center", va="top", fontsize=7)
    ax1.text((u + np.pi) / 2, bar_y, r"$I_0$", color=I0_COLOR, ha="center", va="top", fontsize=7)
    ax1.set_ylim(bar_y - 0.15 * (ymax - ymin), ymax + 0.15 * (ymax - ymin))

    def U_cheb2(m, x):
        if m == -1:
            return np.zeros_like(x)
        if m == 0:
            return np.ones_like(x)
        Um2, Um1 = np.ones_like(x), 2 * x
        for _ in range(2, m + 1):
            Um2, Um1 = Um1, 2 * x * Um1 - Um2
        return Um1

    for N in (1, 2, 5):
        TN = 1.0 / (1.0 + q2sq * U_cheb2(N - 1, kap) ** 2)
        ax2.semilogy(grid, TN, color=N_COLORS[N], linestyle=N_STYLES[N], linewidth=1.1, label=f"$N={N}$")

    phi = np.arccos(np.clip(kap[in_band], -1, 1))
    sin2phi = np.sin(phi) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        Tenv = 1.0 / (1.0 + q2sq[in_band] / sin2phi)
    ax2.semilogy(grid[in_band], Tenv, color="#000000", linestyle="--", linewidth=0.9, label=r"$T_{\rm env}$")

    ax2.axhline(1 - EPS_1_SPEC, color="#999999", linewidth=0.7)
    ax2.axhline(EPS_0_SPEC, color="#999999", linewidth=0.7)
    ax2.text(np.pi, 1 - EPS_1_SPEC, r"  $1-\epsilon_1$", va="bottom", ha="right", fontsize=6, color="#666666")
    ax2.text(np.pi, EPS_0_SPEC, r"  $\epsilon_0$", va="top", ha="right", fontsize=6, color="#666666")

    ax2.set_ylabel(r"$T_N(\vartheta)$")
    ax2.set_xlabel(r"$\vartheta$")
    ax2.set_xlim(0, np.pi)
    ax2.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax2.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    bar_y2 = ax2.get_ylim()[0]
    ax2.plot([0, t], [bar_y2, bar_y2], color=I1_COLOR, linewidth=3, solid_capstyle="butt", clip_on=False)
    ax2.plot([u, np.pi], [bar_y2, bar_y2], color=I0_COLOR, linewidth=3, solid_capstyle="butt", clip_on=False)
    ax2.legend(loc="lower left", frameon=False, fontsize=6.5, ncol=2)

    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    print(f"saved {out_stem}.pdf/.png")


if __name__ == "__main__":
    with open("task_c_results.json") as f:
        results = json.load(f)
    # pick the retained G1 design: least n with delta_star <= delta_target=1e-2
    DELTA_TARGET = 1e-2
    retained = None
    for n in range(1, 7):
        e = results.get("G1", {}).get(str(n), {})
        if e.get("delta_star") is not None and e["delta_star"] <= DELTA_TARGET:
            retained = (n, e)
            break
    if retained is None:
        # fall back to smallest delta_star found
        cand = [(int(k), v) for k, v in results["G1"].items() if v.get("delta_star") is not None]
        retained = min(cand, key=lambda kv: kv[1]["delta_star"])
    n, e = retained
    print(f"retained G1 design: n={n}  delta*={e['delta_star']:.4e}  L={e['L']}  sign={e['sign']}")
    make_figure("G1", n, e["L"], e["sign"], e["alphas"], "example_kappa_TN")
