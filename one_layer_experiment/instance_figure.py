"""Task 3 figure: kappa_B and T_N (N=2,8,16) for the s=1 instance.
Built from transfer_matrix.py's matrix machinery. Two stacked panels,
B&W-safe line styles, PDF (vector) + PNG output.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transfer_matrix import one_layer_alphas, kappa_B, T_N

T, U = np.pi / 4, 3 * np.pi / 4
L = 2
S = 1.0
N_VALUES = [2, 8, 16]
LINESTYLES = {2: "-", 8: "--", 16: ":"}
COLORS = {2: "#9EC9F0", 8: "#3A86D6", 16: "#0B3D80"}

PASS_COLOR = "#4a7a45"
STOP_COLOR = "#a06a2a"


def field_arrays(s, grid):
    alpha = np.arcsinh(np.sqrt(s))
    alphas = one_layer_alphas(alpha)
    kap = np.empty_like(grid)
    q2abs2 = np.empty_like(grid)
    for i, vartheta in enumerate(grid):
        kb, q1, q2 = kappa_B(alphas, vartheta / 2, L)
        kap[i] = kb.real
        q2abs2[i] = abs(q2) ** 2
    return kap, q2abs2


def mark_bands(ax, y_bar):
    ax.plot([0, T], [y_bar, y_bar], color=PASS_COLOR, linewidth=4, solid_capstyle="butt", clip_on=False)
    ax.plot([U, np.pi], [y_bar, y_bar], color=STOP_COLOR, linewidth=4, solid_capstyle="butt", clip_on=False)
    ax.text(T / 2, y_bar, r"$I_1$", color=PASS_COLOR, ha="center", va="bottom", fontsize=9)
    ax.text((U + np.pi) / 2, y_bar, r"$I_0$", color=STOP_COLOR, ha="center", va="bottom", fontsize=9)


def main():
    grid = np.linspace(0.0, np.pi, 4001)
    kap, q2abs2 = field_arrays(S, grid)

    # sanity checks before saving
    kap0 = kap[0]
    assert abs(kap0 - 1.0) < 1e-10, f"kappa_B(0) != 1: {kap0}"
    for N in N_VALUES:
        TN0 = T_N(np.array([q2abs2[0]]), np.array([kap0]), N)[0]
        assert abs(TN0 - 1.0) < 1e-10, f"T_{N}(0) != 1: {TN0}"

    in_band = np.abs(kap) < 1.0
    phi = np.arccos(np.clip(kap[in_band], -1, 1))
    sin2phi = np.sin(phi) ** 2
    Tenv_band = 1.0 / (1.0 + q2abs2[in_band] / sin2phi)

    TN_all = {}
    for N in N_VALUES:
        TN = T_N(q2abs2, kap, N)
        assert np.all(TN <= 1.0 + 1e-10), f"T_{N} exceeds 1"
        assert np.all(TN[in_band] >= Tenv_band - 1e-9), f"T_{N} < T_env inside band"
        TN_all[N] = TN

    # sized to sit at 0.9\linewidth in a two-column figure (~3.4in single-column
    # width) so nominal point sizes below are legible at final print size,
    # not scaled down further by the placement.
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    ax1.plot(grid, kap, color="#222222", linewidth=1.1)
    ax1.axhspan(-1, 1, color="#EDEDED", zorder=0)
    ax1.axhline(1.0, color="#999999", linewidth=0.6)
    ax1.axhline(-1.0, color="#999999", linewidth=0.6)
    ax1.set_ylabel(r"$\kappa_B(\vartheta)$")
    ymin = min(kap.min(), -1.2)
    mark_bands(ax1, ymin - 0.15 * abs(ymin))
    ax1.set_ylim(ymin - 0.3, max(kap.max(), 1.0) + 0.2)

    for N in N_VALUES:
        ax2.semilogy(grid, TN_all[N], color=COLORS[N], linestyle=LINESTYLES[N], linewidth=1.0, label=f"$N={N}$")
    ax2.semilogy(grid[in_band], Tenv_band, color="#555555", linestyle="--", linewidth=0.8, label=r"$T_{\rm env}$")
    ax2.set_ylabel(r"$T_N(\vartheta)$")
    ax2.set_xlabel(r"$\vartheta$")
    ax2.set_xlim(0, np.pi)
    ax2.set_xticks([0, T, np.pi / 2, U, np.pi])
    ax2.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    mark_bands(ax2, ax2.get_ylim()[0])
    ax2.legend(loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handlelength=1.8)

    fig.tight_layout()
    fig.savefig("onelayer_kappa_TN.pdf")
    fig.savefig("onelayer_kappa_TN.png", dpi=200)
    print("saved onelayer_kappa_TN.pdf, onelayer_kappa_TN.png")
    print("all sanity checks passed")


if __name__ == "__main__":
    main()
