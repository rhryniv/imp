"""Figure: kappa_B(vartheta) and T_N(vartheta) for N=2,8,16, at s=1,
baseline geometry (t=pi/4, u=3pi/4). Built from transfer_matrix.py's
matrix machinery (same code validated in Task 1/2/3), not the closed
forms.
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
N_COLORS = {2: "#9EC9F0", 8: "#3A86D6", 16: "#0B3D80"}  # sequential ramp, light->dark by N

STOP_TINT = "#FBE8D6"   # light amber
PASS_TINT = "#DCEEDB"   # light green


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


def main():
    grid = np.linspace(0.0, np.pi, 4001)
    kap, q2abs2 = field_arrays(S, grid)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for ax in (ax1, ax2):
        ax.axvspan(0, T, color=PASS_TINT, zorder=0)
        ax.axvspan(U, np.pi, color=STOP_TINT, zorder=0)

    ax1.plot(grid, kap, color="#222222", linewidth=1.6)
    ax1.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--")
    ax1.axhline(-1.0, color="#999999", linewidth=1.0, linestyle="--")
    ax1.set_ylabel(r"$\kappa_B(\vartheta)$")
    ax1.set_title(r"$\kappa_B$ and $T_N$ at $s=1$ ($t=\pi/4$, $u=3\pi/4$)")
    ax1.text(T / 2, ax1.get_ylim()[1], r"$I_1$ (pass)", ha="center", va="bottom", fontsize=9, color="#4a7a45")
    ax1.text((U + np.pi) / 2, ax1.get_ylim()[1], r"$I_0$ (stop)", ha="center", va="bottom", fontsize=9, color="#a06a2a")

    for N in N_VALUES:
        TN = T_N(q2abs2, kap, N)
        ax2.plot(grid, TN, color=N_COLORS[N], linewidth=1.6, label=f"$N={N}$")
    ax2.set_ylabel(r"$T_N(\vartheta)$")
    ax2.set_xlabel(r"$\vartheta$")
    ax2.set_xlim(0, np.pi)
    ax2.set_xticks([0, T, np.pi / 2, U, np.pi])
    ax2.set_xticklabels(["0", r"$t=\pi/4$", r"$\pi/2$", r"$u=3\pi/4$", r"$\pi$"])
    ax2.legend(loc="upper right", frameon=False, fontsize=10)
    ax2.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig("kappa_TN_s1.png", dpi=170)
    print("saved kappa_TN_s1.png")


if __name__ == "__main__":
    main()
