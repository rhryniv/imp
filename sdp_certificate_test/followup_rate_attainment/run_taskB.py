import json
import time

import numpy as np
import mpmath as mp

from variants import Geometry
from hp_refine import build_and_solve_grid_lp, identify_active_points, classify_points, solve_high_precision, mp_grid_feasibility
from run_taskA import mp_gamma, mp_beta_n

CASES = {
    "baseline": Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=1.0, t_frac=(1, 4), u_frac=(3, 4)),
    "narrower": Geometry(t=np.pi / 3, u=2 * np.pi / 3, mu0=1.0, t_frac=(1, 3), u_frac=(2, 3)),
    "wider": Geometry(t=np.pi / 6, u=5 * np.pi / 6, mu0=1.0, t_frac=(1, 6), u_frac=(5, 6)),
    "mu0=0.5": Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=0.5, t_frac=(1, 4), u_frac=(3, 4)),
    "mu0=2": Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=2.0, t_frac=(1, 4), u_frac=(3, 4)),
}

N_MAX = 6


def solve_range(geo, n_max=N_MAX, dps=60, grid=(8001, 8001, 3001)):
    out = {}
    for n in range(1, n_max + 1):
        m1, mA, m0 = grid
        scale_guess = geo.beta_n(n)
        res, meta = build_and_solve_grid_lp(n, geo, m1=m1, mA=mA, m0=m0, scale=scale_guess)
        if res.status != 0:
            out[n] = {"ok": False, "reason": f"LP status {res.status}"}
            continue
        pts = identify_active_points(res, meta, n)
        cls = classify_points(pts, geo)
        nb = sum(p["boundary"] for p in cls)
        ni = len(cls) - nb
        if nb + 2 * ni != n + 1 + ni:
            out[n] = {"ok": False, "reason": f"active set not square ({len(cls)} pts, need {n+1})"}
            continue
        r0, delta0 = res.x[:n], res.x[n]
        try:
            sol = solve_high_precision(n, geo, cls, r0, delta0, dps=dps)
        except Exception as exc:
            out[n] = {"ok": False, "reason": f"newton failed: {exc}"}
            continue
        feas = mp_grid_feasibility(sol["r"], sol["delta"], geo, dps=dps, grid_n=2000)
        out[n] = {"ok": True, "delta": sol["delta"], "feasible": feas["feasible"]}
    return out


def main():
    gamma_by_case = {}
    all_results = {}
    for name, geo in CASES.items():
        print(f"=== {name}: t={geo.t:.4f} u={geo.u:.4f} mu0={geo.mu0} gamma={geo.gamma:.4f} ===")
        t0 = time.time()
        res = solve_range(geo)
        gamma_by_case[name] = geo.gamma
        all_results[name] = res
        for n in sorted(res):
            r = res[n]
            if r["ok"]:
                lg = mp.log(r["delta"]) + mp_gamma(geo, 60) * n
                print(f"  n={n} delta={mp.nstr(r['delta'],12)} feasible={r['feasible']} "
                      f"log(delta)+gamma*n={mp.nstr(lg,10)}")
            else:
                print(f"  n={n} FAILED: {r['reason']}")
        print(f"  ({time.time()-t0:.1f}s)")

    with open("taskB_results.json", "w") as f:
        json.dump({name: {str(n): ({"ok": True, "delta": mp.nstr(r["delta"], 40), "feasible": r["feasible"]}
                                     if r["ok"] else r)
                           for n, r in res.items()}
                   for name, res in all_results.items()}, f, indent=2, default=str)
    print("saved taskB_results.json")


if __name__ == "__main__":
    main()
