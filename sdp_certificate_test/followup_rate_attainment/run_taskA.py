import json
import time

import numpy as np
import mpmath as mp

from variants import Geometry
from hp_refine import (build_and_solve_grid_lp, identify_active_points, classify_points,
                        solve_high_precision, mp_grid_feasibility)

BASELINE = Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=1.0, t_frac=(1, 4), u_frac=(3, 4))


def mp_gamma(geo: Geometry, dps=50):
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        ct = mp.cos(geo.mp_t(dps))
        cu = mp.cos(geo.mp_u(dps))
        num = 2 * cu - ct - 1
        den = 1 - ct
        return mp.acosh(abs(num / den))
    finally:
        mp.mp.dps = old


def mp_beta_n(geo: Geometry, n: int, dps=50):
    sinh2 = geo.mp_sinh2_mu0(dps)
    return sinh2 * mp.e ** (-mp_gamma(geo, dps) * n)


def solve_one(n: int, geo: Geometry, dps=50, grid=(6001, 6001, 3001), verbose=True):
    m1, mA, m0 = grid
    scale_guess = geo.beta_n(n)  # rough a-priori magnitude only, keeps the LP's delta column O(1)
    res, meta = build_and_solve_grid_lp(n, geo, m1=m1, mA=mA, m0=m0, scale=scale_guess)
    if res.status != 0:
        return {"n": n, "ok": False, "reason": f"LP status {res.status}"}
    pts = identify_active_points(res, meta, n)
    cls = classify_points(pts, geo)
    n_boundary = sum(p["boundary"] for p in cls)
    n_interior = len(cls) - n_boundary
    n_eqs = n_boundary + 2 * n_interior
    n_unknowns = n + 1 + n_interior
    if n_eqs != n_unknowns or len(cls) < 1:
        return {"n": n, "ok": False, "reason": f"active set not square: {n_eqs} eq vs {n_unknowns} unk "
                                                f"({len(cls)} active pts found, expected {n+1})"}
    r0 = res.x[:n]
    delta0 = res.x[n]
    try:
        sol = solve_high_precision(n, geo, cls, r0, delta0, dps=dps)
    except Exception as exc:
        return {"n": n, "ok": False, "reason": f"mpmath solve failed: {exc}"}
    feas = mp_grid_feasibility(sol["r"], sol["delta"], geo, dps=dps, grid_n=3000)
    beta_n = mp_beta_n(geo, n, dps)
    delta_hp = sol["delta"]
    ratio = delta_hp / beta_n
    out = {
        "n": n, "ok": True, "delta_hp": delta_hp, "beta_n": beta_n, "ratio": ratio,
        "feasible": feas["feasible"], "worst_slack": feas["worst_slack"],
        "n_active": len(cls), "n_boundary": n_boundary, "n_interior": n_interior,
        "active_kinds": [p["kind"] for p in cls],
    }
    if verbose:
        print(f"n={n:2d} ok delta={mp.nstr(delta_hp, 12)} ratio={mp.nstr(ratio, 6)} "
              f"active={len(cls)}(b={n_boundary},i={n_interior}) feasible={feas['feasible']}")
    return out


def main():
    results = {}
    prev_delta = {}
    for n in range(1, 12):
        t0 = time.time()
        grid = (8001, 8001, 3001) if n <= 8 else (12001, 12001, 4001)
        res = solve_one(n, BASELINE, dps=60, grid=grid)
        res["time_s"] = time.time() - t0
        if res["ok"]:
            results[n] = res
            if (n - 2) in results and results[n - 2]["ok"]:
                two_step = res["delta_hp"] / results[n - 2]["delta_hp"]
                res["two_step_ratio"] = two_step
        else:
            print(f"n={n:2d} FAILED: {res['reason']}")
            results[n] = res
        print(f"  ({res['time_s']:.1f}s)")

    with open("taskA_results.json", "w") as f:
        json.dump({str(k): {kk: (mp.nstr(vv, 40) if isinstance(vv, mp.mpf) else vv)
                             for kk, vv in v.items() if kk != "worst_slack"}
                   for k, v in results.items()}, f, indent=2, default=str)
    print("saved taskA_results.json")


if __name__ == "__main__":
    main()
