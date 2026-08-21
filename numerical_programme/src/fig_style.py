"""Shared style for figures F1-F8, per the brief's Section 5 style spec:
vector PDF+PNG, no titles, colorblind-safe Okabe-Ito palette with FIXED
role assignments, greyscale-legible, ticks at 0,pi/4,pi/2,3pi/4,pi
LaTeX-labeled, sized for 0.9\\linewidth single-column (~3.4in), 8-9pt fonts.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed Okabe-Ito role assignments (brief Section 5).
COLOR_N1 = "#0072B2"   # solid
COLOR_N2 = "#D55E00"   # dashed
COLOR_N5 = "#009E73"   # dotted
COLOR_N10 = "#CC79A7"  # dash-dot
COLOR_KAPPA_B = "#000000"  # solid 1.2pt
COLOR_TENV = "#000000"     # dashed 0.9pt
COLOR_BAND_FILL = "#E8E8E8"
COLOR_I0_BAR = "#666666"
COLOR_I1_BAR = "#BBBBBB"

N_STYLE = {
    1: dict(color=COLOR_N1, ls="-", lw=1.1),
    2: dict(color=COLOR_N2, ls="--", lw=1.1),
    5: dict(color=COLOR_N5, ls=":", lw=1.3),
    10: dict(color=COLOR_N10, ls="-.", lw=1.1),
}

FIGSIZE = (3.4, 2.4)
FIGSIZE_WIDE = (3.4, 3.0)

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
})

PI_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
PI_LABELS = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]


def set_theta_axis(ax):
    ax.set_xlim(0, np.pi)
    ax.set_xticks(PI_TICKS)
    ax.set_xticklabels(PI_LABELS)
    ax.set_xlabel(r"$\theta$")


def shade_interval(ax, lo, hi, color, alpha=1.0, y=None):
    ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=0)


def save(fig, path_base):
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    fig.savefig(path_base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
