"""Task 3 figure: kappa_B and T_N (N=2,8,16) for the s=1 instance.
Built from transfer_matrix.py's matrix machinery. Same visual style as
plot_kappa_TN.py (the earlier figure for this project): full-height
axvspan shading for I0/I1, title, sequential-blue solid lines, same
sizing -- kept consistent rather than switching to a separate
paper-column layout. PDF (vector) + PNG output.
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

    # sanity checks before saving
    assert abs(kap[0] - 1.0) < 1e-10, f"kappa_B(0) != 1: {kap[0]}"
    for N in N_VALUES:
        TN0 = T_N(np.array([q2abs2[0]]), np.array([kap[0]]), N)[0]
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
        ax2.semilogy(grid, TN_all[N], color=N_COLORS[N], linewidth=1.6, label=f"$N={N}$")
    ax2.semilogy(grid[in_band], Tenv_band, color="#777777", linewidth=1.3, linestyle="--", label=r"$T_{\rm env}$")
    ax2.set_ylabel(r"$T_N(\vartheta)$")
    ax2.set_xlabel(r"$\vartheta$")
    ax2.set_xlim(0, np.pi)
    ax2.set_xticks([0, T, np.pi / 2, U, np.pi])
    ax2.set_xticklabels(["0", r"$t=\pi/4$", r"$\pi/2$", r"$u=3\pi/4$", r"$\pi$"])
    ax2.legend(loc="lower left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig("onelayer_kappa_TN.pdf")
    fig.savefig("onelayer_kappa_TN.png", dpi=170)
    print("saved onelayer_kappa_TN.pdf, onelayer_kappa_TN.png")
    print("all sanity checks passed")


if __name__ == "__main__":
    main()
