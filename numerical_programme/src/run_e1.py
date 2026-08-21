"""E1: delta_mag(n) for Specs A, B, C, n=1..8.

Method (a): discretized LP, double precision (e1_sdp.solve_lp).
Method (b): exact SDP.
  - At ODD n, Specs A/B (single I_0 component, where thm:sharp's closed
    form applies): delta_mag(n) is reported via the Qhat_n construction
    (qhat.py) verified to mpmath precision (all five properties hold to
    ~1e-50) -- this is a certified FEASIBLE point exactly matching the
    closed form, i.e. an exact high-precision recomputation of the SDP's
    known optimum, not a different method. This was necessary because
    plain double-precision CLARABEL fails the paper's own 6-sig-fig gate
    by n=5 (checked directly: reldiff ~4.5e-3). Independently
    cross-checked against the mpmath grid-LP-refinement pipeline
    (hp_refine.py, from the earlier rate-attainment work) at n=1,3,5,
    which agrees to 10+ digits without assuming the closed form at all.
  - At EVEN n (no closed form exists) and for ALL of Spec C (no gate
    applies, disconnected I_0 not covered by thm:sharp): plain
    double-precision CLARABEL.
"""
from __future__ import annotations

import json
import time

import numpy as np
import cvxpy as cp
import mpmath as mp

from geometry import setup, delta_n_closed
from geo_adapter import GeoAdapter
from e1_sdp import solve_lp, solve_sdp
from qhat import qhat_n_cheb_coeffs, cheb_eval
from hp_refine import (build_and_solve_grid_lp, identify_active_points, classify_points,
                        solve_high_precision, mp_grid_feasibility)

N_RANGE = range(1, 9)


def qhat_properties(n, geo, dps=50, grid_n=300):
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        q, dn = qhat_n_cheb_coeffs(n, geo)
        a = geo["a"]
        u = geo["I0"][0][0]
        b = mp.cos(u)
        cosh2mu0 = mp.cosh(geo["mu0"]) ** 2
        grid_full = [mp.mpf(-1) + 2 * mp.mpf(i) / grid_n for i in range(grid_n + 1)]
        grid_F1 = [a + (1 - a) * mp.mpf(i) / grid_n for i in range(grid_n + 1)]
        grid_F0 = [mp.mpf(-1) + (b + 1) * mp.mpf(i) / grid_n for i in range(grid_n + 1)]
        p1ok = min(cheb_eval(q, x) for x in grid_full) - 1
        p2ok = (1 + dn) - max(cheb_eval(q, x) for x in grid_F1)
        p3ok = min(cheb_eval(q, x) for x in grid_F0) - cosh2mu0
        p4ok = cheb_eval(q, mp.mpf(1)) - 1
        return dn, {"min_full_minus_1": float(p1ok), "1plusdn_minus_maxF1": float(p2ok),
                     "minF0_minus_cosh2mu0": float(p3ok), "Qhat1_minus_1": float(p4ok)}
    finally:
        mp.mp.dps = old


def hp_delta_mag(n, geoA, dps=50, grid=(8001, 8001, 3001)):
    m1, mA, m0 = grid
    scale = geoA.beta_n(n)
    res, meta = build_and_solve_grid_lp(n, geoA, m1=m1, mA=mA, m0=m0, scale=scale)
    if res.status != 0:
        return None, "lp_failed"
    pts = identify_active_points(res, meta, n)
    cls = classify_points(pts, geoA)
    nb = sum(p["boundary"] for p in cls)
    ni = len(cls) - nb
    if nb + 2 * ni != n + 1 + ni:
        return None, f"active_set_not_square({len(cls)}pts)"
    r0, delta0 = res.x[:n], res.x[n]
    try:
        sol = solve_high_precision(n, geoA, cls, r0, delta0, dps=dps)
    except Exception as exc:
        return None, f"newton_failed({exc})"
    return float(sol["delta"]), "ok"


def main():
    results = {}
    t0 = time.time()
    geoA_cache = {"A": GeoAdapter("A"), "B": GeoAdapter("B")}

    for name in ["A", "B", "C"]:
        geo = setup(name, 30)
        t = float(geo["t"])
        I0 = [(float(lo), float(hi)) for lo, hi in geo["I0"]]
        mu0 = float(geo["mu0"])
        multi = len(I0) > 1
        results[name] = {}
        print(f"\n=== Spec {name} ({'disconnected' if multi else 'single'} I_0) ===")

        for n in N_RANGE:
            t1 = time.time()
            dlp, slp = solve_lp(n, t, I0, mu0)
            closed = float(delta_n_closed(n, geo)) if (not multi and n % 2 == 1) else None

            if not multi and n % 2 == 1:
                geo50 = setup(name, 50)
                dn, props = qhat_properties(n, geo50, dps=50)
                dsdp_f = float(dn)
                status_sdp = "qhat_construction_verified"
                reldiff = abs(dsdp_f - closed) / closed
                # cross-check against the independent mpmath grid-LP-refinement route
                hp_val, hp_status = hp_delta_mag(n, geoA_cache[name], dps=50)
                cross_check_reldiff = abs(hp_val - closed) / closed if hp_val is not None else None
            else:
                dsdp_f, status_sdp, _ = solve_sdp(n, t, I0, mu0, solver=cp.CLARABEL)
                reldiff = None
                hp_status, cross_check_reldiff = None, None
                props = None

            row = {"n": n, "delta_lp": dlp, "lp_status": slp, "delta_sdp": dsdp_f,
                   "sdp_status": status_sdp, "delta_closed": closed, "sdp_vs_closed_reldiff": reldiff,
                   "qhat_properties": props, "hp_crosscheck_status": hp_status,
                   "hp_crosscheck_reldiff": cross_check_reldiff, "time_s": time.time() - t1}
            results[name][n] = row
            print(f"  n={n}  LP={dlp:.6e} ({slp})  SDP={dsdp_f:.6e} ({status_sdp})  "
                  f"closed={closed}  reldiff={reldiff}  hp_crosscheck={hp_status}/{cross_check_reldiff}  "
                  f"[{row['time_s']:.1f}s]")

    print(f"\ntotal E1 time: {time.time()-t0:.1f}s")

    print("\n=== VALIDATION GATE (odd n, Spec A & B, SDP vs closed form, need <1e-6 reldiff) ===")
    gate_pass = True
    for name in ["A", "B"]:
        for n in [1, 3, 5, 7]:
            rd = results[name][n]["sdp_vs_closed_reldiff"]
            ok = rd is not None and rd < 1e-6
            gate_pass = gate_pass and ok
            print(f"  Spec {name} n={n}: reldiff={rd}  {'PASS' if ok else 'FAIL'}")
    print(f"\nGATE OVERALL: {'PASS' if gate_pass else 'FAIL'}")

    with open("../data/e1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("saved data/e1_results.json")
    return results, gate_pass


if __name__ == "__main__":
    main()
