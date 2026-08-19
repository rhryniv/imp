"""Figure: T_N for N=1,2,5 at n=3, using the best configuration found
(L_best=16 from the realisability sweep). Linear scale, matching the
project's established plotting style.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import setup
from qhat import qhat_n_cheb_coeffs
from spectral import cosine_coeffs, spectral_factor, poly_eval

T, U = np.pi / 4, 3 * np.pi / 4
N_VALUES = [1, 2, 5]
COLORS = {1: "#9EC9F0", 2: "#3A86D6", 5: "#0B3D80"}
LINESTYLES = {1: "-", 2: "--", 5: ":"}
LINEWIDTHS = {1: 1.6, 2: 1.5, 5: 1.3}
ZORDER = {1: 3, 2: 2, 5: 1}
PASS_TINT = "#DCEEDB"
STOP_TINT = "#FBE8D6"

DPS = 60
N_LAYERS = 3
L_BEST = 16  # from run_all.py's sweep for n=3


def U_cheb2_mp(m, x):
    if m == -1:
        return mp.mpf(0)
    if m == 0:
        return mp.mpf(1)
    Um2, Um1 = mp.mpf(1), 2 * x
    for _ in range(2, m + 1):
        Um2, Um1 = Um1, 2 * x * Um1 - Um2
    return Um1


def main():
    geo = setup(DPS)
    q, dn = qhat_n_cheb_coeffs(N_LAYERS, geo)
    f = cosine_coeffs(q)
    p1, outside, rho, max_imag = spectral_factor(f, DPS)

    grid = [mp.mpf(0) + mp.pi * i / 6000 for i in range(6001)]

    kappa_vals = []
    q2abs2_vals = []
    for vartheta in grid:
        w = mp.e ** (-1j * vartheta)
        p1w = poly_eval(p1, w)
        Q = abs(p1w) ** 2
        kap = mp.re(mp.e ** (1j * L_BEST * vartheta / 2) * p1w)
        kappa_vals.append(kap)
        q2abs2_vals.append(Q - 1)

    # sanity checks
    assert abs(kappa_vals[0] - 1) < mp.mpf('1e-40'), "kappa_B(0) != 1"
    for N in N_VALUES:
        TN0 = 1 / (1 + q2abs2_vals[0] * U_cheb2_mp(N - 1, kappa_vals[0]) ** 2)
        assert abs(TN0 - 1) < mp.mpf('1e-40'), f"T_{N}(0) != 1"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(0, float(T), color=PASS_TINT, zorder=0)
    ax.axvspan(float(U), float(np.pi), color=STOP_TINT, zorder=0)

    for N in N_VALUES:
        TN = [float(1 / (1 + q2sq * U_cheb2_mp(N - 1, kap) ** 2))
              for kap, q2sq in zip(kappa_vals, q2abs2_vals)]
        assert all(v <= 1 + 1e-9 for v in TN), f"T_{N} exceeds 1"
        ax.plot([float(v) for v in grid], TN, color=COLORS[N], linestyle=LINESTYLES[N],
                 linewidth=LINEWIDTHS[N], zorder=ZORDER[N], label=f"$N={N}$")

    ax.set_xlim(0, np.pi)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, T, np.pi / 2, U, np.pi])
    ax.set_xticklabels(["0", r"$t=\pi/4$", r"$\pi/2$", r"$u=3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"$T_N(\vartheta)$")
    ax.set_title(rf"$T_N$ at $n={N_LAYERS}$, best configuration ($L={L_BEST}$), $N=1,2,5$")
    ax.text(T / 2, 1.0, r"$I_1$ (pass)", ha="center", va="bottom", fontsize=9, color="#4a7a45")
    ax.text((U + np.pi) / 2, 1.0, r"$I_0$ (stop)", ha="center", va="bottom", fontsize=9, color="#a06a2a")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig("TN_n3.pdf")
    fig.savefig("TN_n3.png", dpi=170)
    print("saved TN_n3.pdf/.png")
    print("all sanity checks passed")


if __name__ == "__main__":
    main()
