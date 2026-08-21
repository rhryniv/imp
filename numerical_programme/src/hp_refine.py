"""High-precision extension of delta_mag(n) via variant II (Qhat = 1 +
(1-x)R(x), R >= 0 on [-1,1], degree n-1).

Pipeline (per n):
 1. Solve a fine-grid LP in double precision (scipy HiGHS) to locate the
    active constraint set (its optimal basis has exactly n+1 active
    constraints among decision vars r_0..r_{n-1}, delta, by LP theory).
 2. Classify each active grid point as a fixed domain boundary (no free
    unknown) or a genuine interior extremum (adds one unknown: its exact
    location, pinned by a derivative-zero condition).
 3. Assemble the resulting square nonlinear system and solve it with
    mpmath at high precision (mp.dps=50), seeded from the double-precision
    LP solution -- Newton's method converges quadratically once inside
    the basin, recovering full mpmath precision regardless of how small
    delta itself is (no more double-precision cancellation floor).
 4. Verify: re-evaluate every constraint on a fine high-precision grid; if
    a violation beyond tolerance remains, add that point to the active
    set and go back to step 3 (Remez-style exchange).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import linprog
import mpmath as mp

from geo_adapter import GeoAdapter as Geometry

mp.mp.dps = 50


def _cheb_design(x: np.ndarray, deg: int) -> np.ndarray:
    """Vandermonde-like matrix: columns T_0(x)..T_deg(x)."""
    return C.chebvander(x, deg)


def build_and_solve_grid_lp(n: int, geo: Geometry, m1: int = 6001, mA: int = 6001, m0: int = 3001,
                             scale: float = 1.0):
    """scale rescales the delta decision variable: the LP solves for
    delta_scaled = delta/scale (so delta_scaled stays O(1) even when the
    true delta is far below HiGHS's default tolerances, which otherwise
    round tiny optimal delta down to exactly 0). Pass a rough a-priori
    magnitude estimate (e.g. the analytic beta_n) as `scale`."""
    X1_A, X1_B = geo.X1
    X0_A, X0_B = geo.X0
    sinh2 = float(np.sinh(geo.mu0) ** 2)

    x1 = np.linspace(X1_A, X1_B, m1)
    xA = np.linspace(-1.0, 1.0, mA)
    x0 = np.linspace(X0_A, X0_B, m0)

    Tdeg = n - 1
    T1 = _cheb_design(x1, Tdeg)          # (m1, n)
    TA = _cheb_design(xA, Tdeg)          # (mA, n)
    T0 = _cheb_design(x0, Tdeg)          # (m0, n)

    # decision vector z = [r_0..r_{n-1}, delta_scaled], length n+1
    # (B) (1-x) R(x) - scale*delta_scaled <= 0  on X1
    A_B = np.hstack([(1.0 - x1)[:, None] * T1, -scale * np.ones((m1, 1))])
    b_B = np.zeros(m1)
    # (A) -R(x) <= 0 on [-1,1]
    A_A = np.hstack([-TA, np.zeros((mA, 1))])
    b_A = np.zeros(mA)
    # (C') -(1-x) R(x) <= -sinh^2(mu0) on X0
    A_C = np.hstack([-(1.0 - x0)[:, None] * T0, np.zeros((m0, 1))])
    b_C = -sinh2 * np.ones(m0)

    A_ub = np.vstack([A_B, A_A, A_C])
    b_ub = np.concatenate([b_B, b_A, b_C])
    c_obj = np.zeros(n + 1)
    c_obj[-1] = 1.0
    bounds = [(None, None)] * n + [(0.0, None)]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 0:
        res.x = res.x.copy()
        res.x[-1] = res.x[-1] * scale  # report delta in real units
        if hasattr(res, "ineqlin") and res.ineqlin.marginals is not None:
            # dual of the delta_scaled column relates to dual of delta by /scale;
            # marginals on the OTHER (r) columns are unaffected by this rescaling,
            # and we only use marginals to rank/select active rows, so no change needed.
            pass
    meta = {
        "x_all": np.concatenate([x1, xA, x0]),
        "kind": (["B"] * m1) + (["A"] * mA) + (["C"] * m0),
        "region": [(X1_A, X1_B)] * m1 + [(-1.0, 1.0)] * mA + [(X0_A, X0_B)] * m0,
        "slack": res.slack if res.slack is not None else None,
        "marginals": -res.ineqlin.marginals if hasattr(res, "ineqlin") else None,
    }
    return res, meta


def identify_active_points(res, meta, n: int, noise_floor: float = 1e-12):
    """Pick the top n+1 constraints by LP-multiplier magnitude -- the
    active set at a non-degenerate vertex (LP theory guarantees exactly
    n+1 active constraints among n+1 decision variables). Only an
    absolute noise floor is applied (not a relative-to-max threshold --
    genuinely active constraints can have multipliers many orders of
    magnitude smaller than the largest one, e.g. constraint C's shadow
    price vanishing rapidly with n while remaining exactly active)."""
    mult = np.abs(meta["marginals"])
    order = np.argsort(-mult)
    chosen = []
    for idx in order:
        if mult[idx] < noise_floor:
            break
        chosen.append(idx)
        if len(chosen) >= n + 1:
            break
    pts = []
    for idx in chosen:
        pts.append({"x": float(meta["x_all"][idx]), "kind": meta["kind"][idx], "region": meta["region"][idx]})
    return pts


def classify_points(points, geo: Geometry, snap_tol: float = 1e-4):
    """Snap near-boundary LP points to exact (mpmath-precision) domain
    endpoints (no free unknown); leave the rest as interior (needs its
    own unknown + a derivative condition). `boundary_mp` carries the
    full-precision mpmath value so downstream code never round-trips
    through a float64 string."""
    X1_A, X1_B = geo.X1
    X0_A, X0_B = geo.X0
    boundary_candidates = [(-1.0, "neg1"), (1.0, "pos1"), (X1_A, "X1_A"), (X1_B, "X1_B"),
                            (X0_A, "X0_A"), (X0_B, "X0_B")]
    out = []
    for p in points:
        x = p["x"]
        snapped = None
        for b, tag in boundary_candidates:
            if abs(x - b) < snap_tol:
                snapped = tag
                break
        if snapped is not None:
            out.append({"x": None, "boundary_tag": snapped, "kind": p["kind"], "boundary": True})
        else:
            out.append({"x": x, "boundary_tag": None, "kind": p["kind"], "boundary": False})
    return out


def _boundary_mp_value(tag: str, geo: Geometry, dps: int):
    import mpmath as mp
    X1_A, X1_B = geo.mp_X1(dps)
    X0_A, X0_B = geo.mp_X0(dps)
    return {"neg1": mp.mpf(-1), "pos1": mp.mpf(1), "X1_A": X1_A, "X1_B": X1_B,
            "X0_A": X0_A, "X0_B": X0_B}[tag]


def mp_cheb_eval(coeffs, x):
    """Clenshaw evaluation, mpmath-precision, coeffs = mpmath list/array."""
    k = len(coeffs) - 1
    b1 = mp.mpf(0)
    b2 = mp.mpf(0)
    for j in range(k, 0, -1):
        b0 = coeffs[j] + 2 * x * b1 - b2
        b2 = b1
        b1 = b0
    return coeffs[0] + x * b1 - b2


def mp_cheb_deriv_eval(coeffs, x):
    """d/dx of sum coeffs[k] T_k(x), evaluated via the standard Chebyshev
    derivative-coefficient recursion (computed once, then Clenshaw)."""
    nmax = len(coeffs) - 1
    d = [mp.mpf(0)] * (nmax + 1)
    if nmax >= 1:
        d[nmax - 1] = 2 * nmax * coeffs[nmax]
    for k in range(nmax - 2, -1, -1):
        d[k] = d[k + 2] + 2 * (k + 1) * coeffs[k + 1]
    if nmax >= 1:
        d[0] = d[0] / 2
    return mp_cheb_eval(d[: nmax], x) if nmax >= 1 else mp.mpf(0)


def solve_high_precision(n: int, geo: Geometry, active_pts, r0, delta0, dps: int = 50):
    """active_pts: list of dicts {x, kind in {'A','B','C'}, boundary bool}.
    Builds and Newton-solves the square system with mpmath at `dps` digits."""
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        interior = [p for p in active_pts if not p["boundary"]]
        boundary = [p for p in active_pts if p["boundary"]]
        n_unknowns = n + 1 + len(interior)
        n_eqs = len(boundary) + 2 * len(interior)
        if n_eqs != n_unknowns:
            raise ValueError(f"system not square: {n_eqs} eqs vs {n_unknowns} unknowns "
                              f"({len(boundary)} boundary + {len(interior)} interior)")

        sinh2 = geo.mp_sinh2_mu0(dps)

        def value_residual(kind, x, r, delta):
            Rx = mp_cheb_eval(r, x)
            if kind == "A":
                return Rx
            if kind == "B":
                return (1 - x) * Rx - delta
            return (1 - x) * Rx - sinh2  # 'C'

        def deriv_residual(kind, x, r):
            Rx = mp_cheb_eval(r, x)
            Rpx = mp_cheb_deriv_eval(r, x)
            if kind == "A":
                return Rpx
            return -Rx + (1 - x) * Rpx  # 'B' or 'C'

        def F(*z):
            r = list(z[:n])
            delta = z[n]
            locs = list(z[n + 1:])
            eqs = []
            for p in boundary:
                x = _boundary_mp_value(p["boundary_tag"], geo, dps)
                eqs.append(value_residual(p["kind"], x, r, delta))
            for p, xloc in zip(interior, locs):
                eqs.append(value_residual(p["kind"], xloc, r, delta))
                eqs.append(deriv_residual(p["kind"], xloc, r))
            return tuple(eqs)

        z0 = [mp.mpf(str(v)) for v in r0] + [mp.mpf(str(delta0))] + [mp.mpf(str(p["x"])) for p in interior]
        sol = mp.findroot(F, z0, tol=mp.mpf(10) ** (-(dps - 15)), solver="mnewton", maxsteps=200)
        r_sol = [sol[i] for i in range(n)]
        delta_sol = sol[n]
        return {"r": r_sol, "delta": delta_sol, "interior_locs": [sol[n + 1 + i] for i in range(len(interior))],
                "n_active": len(active_pts), "boundary": boundary, "interior": interior}
    finally:
        mp.mp.dps = old_dps


def mp_grid_feasibility(r, delta, geo: Geometry, dps: int = 50, grid_n: int = 4000):
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        X1_A, X1_B = geo.mp_X1(dps)
        X0_A, X0_B = geo.mp_X0(dps)
        sinh2 = geo.mp_sinh2_mu0(dps)
        worst = {"A": mp.mpf(1e10), "B": mp.mpf(1e10), "C": mp.mpf(1e10)}
        worst_x = {"A": None, "B": None, "C": None}
        for kind, (a, b) in (("A", (mp.mpf(-1), mp.mpf(1))), ("B", (X1_A, X1_B)), ("C", (X0_A, X0_B))):
            for i in range(grid_n + 1):
                x = a + (b - a) * i / grid_n
                Rx = mp_cheb_eval(r, x)
                if kind == "A":
                    val = Rx
                elif kind == "B":
                    val = delta - (1 - x) * Rx
                else:
                    val = (1 - x) * Rx - sinh2
                if val < worst[kind]:
                    worst[kind] = val
                    worst_x[kind] = x
        return {"worst_slack": {k: float(v) for k, v in worst.items()},
                "worst_x": {k: (float(v) if v is not None else None) for k, v in worst_x.items()},
                "feasible": all(v >= -mp.mpf(10) ** (-(dps - 10)) for v in worst.values())}
    finally:
        mp.mp.dps = old_dps
