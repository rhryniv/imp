"""Band/gap structure of the Floquet discriminant kappa_B(theta) on the
fundamental domain theta in [0, pi] (kappa_B is 2*pi-periodic and even in
theta = 2*k*h, so [0, pi] -- half the period -- determines everything, per
Section 5.2 of the paper: bands are {|kappa_B| <= 1}, gaps {|kappa_B| > 1}).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import kappa_B


@dataclass
class BandGapStructure:
    theta: np.ndarray
    kappa: np.ndarray
    bands: list           # list of (theta_lo, theta_hi)
    gaps: list             # list of (theta_lo, theta_hi, sign) with sign = sign(kappa_B) on that gap


def find_bands_gaps(a: np.ndarray, n_grid: int = 20000) -> BandGapStructure:
    theta = np.linspace(0.0, np.pi, n_grid)
    kap = kappa_B(a, theta)
    is_gap = np.abs(kap) > 1.0

    bands, gaps = [], []
    start = 0
    cur = is_gap[0]
    for i in range(1, n_grid):
        if is_gap[i] != cur:
            _add_run(bands, gaps, theta, kap, start, i - 1, cur)
            start = i
            cur = is_gap[i]
    _add_run(bands, gaps, theta, kap, start, n_grid - 1, cur)
    return BandGapStructure(theta=theta, kappa=kap, bands=bands, gaps=gaps)


def _add_run(bands, gaps, theta, kap, i0, i1, is_gap):
    if i1 <= i0:
        return
    lo, hi = theta[i0], theta[i1]
    if is_gap:
        sign = 1 if kap[(i0 + i1) // 2] > 0 else -1
        gaps.append((lo, hi, sign))
    else:
        bands.append((lo, hi))


def shrink_interval(interval: tuple, margin_frac: float = 0.15) -> tuple:
    """Move both endpoints inward by margin_frac of the interval width, to
    keep a guard band away from the (typically singular / hard-to-hit
    exactly) band edges."""
    lo, hi = interval[0], interval[1]
    w = hi - lo
    return (lo + margin_frac * w, hi - margin_frac * w)
