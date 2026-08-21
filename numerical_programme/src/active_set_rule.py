"""Extrapolated active-set rule, validated against the grid-LP-discovered
active sets for n=1..6 (see hp_refine.identify_active_points): within X1,
active points alternate ceiling(B)/floor(A) starting with a boundary B at
x=cos(t); odd n ends with an interior B, even n ends with the boundary A
at x=1. Constraint C is always active at the single boundary point
x=cos(u). Total active points = n+1 in every case (matches LP theory).
"""
from __future__ import annotations

import numpy as np

from variants import Geometry


def guess_active_set(n: int, geo: Geometry):
    """Returns a list of dicts in the same shape hp_refine.classify_points
    produces: {'x':..,'boundary_tag':..,'kind':..,'boundary':bool}, plus
    approximate (double-precision) seed locations for interior points via
    Chebyshev-extrema-like clustering across X1."""
    X1_A, X1_B = geo.X1
    m = n  # number of active points strictly within/spanning X1 (excludes the C point)
    if m == 1:
        xs = [X1_A]
    else:
        xs = [X1_A + (X1_B - X1_A) * (1 - np.cos(k * np.pi / (m - 1))) / 2 for k in range(m)]
    kinds = ["B" if k % 2 == 0 else "A" for k in range(m)]  # starts with B

    pts = []
    for k, (x, kind) in enumerate(zip(xs, kinds)):
        if k == 0:
            pts.append({"x": None, "boundary_tag": "X1_A", "kind": "B", "boundary": True})
        elif k == m - 1 and n % 2 == 0:
            pts.append({"x": None, "boundary_tag": "pos1", "kind": "A", "boundary": True})
        else:
            pts.append({"x": float(x), "boundary_tag": None, "kind": kind, "boundary": False})
    pts.append({"x": None, "boundary_tag": "X0_B", "kind": "C", "boundary": True})
    return pts
