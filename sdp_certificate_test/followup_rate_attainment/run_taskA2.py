import json
import time

import numpy as np
import mpmath as mp

from variants import Geometry
from hp_refine import (build_and_solve_grid_lp, identify_active_points, classify_points,
                        solve_high_precision, mp_grid_feasibility)
from active_set_rule import guess_active_set
from run_taskA import mp_gamma, mp_beta_n

BASELINE = Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=1.0, t_frac=(1, 4), u_frac=(3, 4))


def solve_via_lp(n, geo, dps=60, grid=(8001, 8001, 3001)):
    m1, mA, m0 = grid
    scale_guess = geo.beta_n(n)
    res, meta = build_and_solve_grid_lp(n, geo, m1=m1, mA=mA, m0=m0, scale=scale_guess)
    pts = identify_active_points(res, meta, n)
    cls = classify_points(pts, geo)
    r0, delta0 = res.x[:n], res.x[n]
    return cls, r0, delta0


def solve_via_rule(n, geo, r0_prev, delta0_prev, dps=60):
    cls = guess_active_set(n, geo)
    r0 = np.zeros(n)
    if r0_prev is not None:
        r0[: len(r0_prev)] = r0_prev
    delta0 = delta0_prev
    return cls, r0, delta0


def main():
    results = {}
    r_prev_float = None
    delta_prev_float = None
    for n in range(1, 12):
        t0 = time.time()
        if n <= 6:
            cls, r0, delta0 = solve_via_lp(n, BASELINE)
            source = "grid-LP active set"
        else:
            cls, r0, delta0 = solve_via_rule(n, BASELINE, r_prev_float, delta_prev_float)
            source = "extrapolated rule + continuation seed"

        try:
            sol = solve_high_precision(n, BASELINE, cls, r0, delta0, dps=60)
            feas = mp_grid_feasibility(sol["r"], sol["delta"], BASELINE, dps=60, grid_n=3000)
            beta_n = mp_beta_n(BASELINE, n, 60)
            ratio = sol["delta"] / beta_n
            ok = True
            reason = None
        except Exception as exc:
            ok = False
            reason = str(exc)
            sol = None
            feas = None
            ratio = None

        elapsed = time.time() - t0
        if ok:
            r_prev_float = [float(x) for x in sol["r"]]
            delta_prev_float = float(sol["delta"])
            print(f"n={n:2d} [{source}] delta={mp.nstr(sol['delta'], 15)} ratio={mp.nstr(ratio, 6)} "
                  f"feasible={feas['feasible']} worst={ {k: f'{v:.2e}' for k,v in feas['worst_slack'].items()} } "
                  f"({elapsed:.1f}s)")
            results[n] = {"ok": True, "delta": mp.nstr(sol["delta"], 40), "ratio": mp.nstr(ratio, 15),
                          "beta_n": mp.nstr(beta_n, 15), "feasible": feas["feasible"],
                          "worst_slack": feas["worst_slack"], "source": source, "time_s": elapsed}
        else:
            print(f"n={n:2d} [{source}] FAILED: {reason} ({elapsed:.1f}s)")
            results[n] = {"ok": False, "reason": reason, "source": source, "time_s": elapsed}
            # keep previous seed for next n's continuation attempt

    with open("taskA2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("saved taskA2_results.json")


if __name__ == "__main__":
    main()
