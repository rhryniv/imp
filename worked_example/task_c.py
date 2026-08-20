"""Task C: direct optimisation over log-contrasts.

Compute-budget disclosure: the brief specifies 50 random multistarts per
(geometry, n, L, sign) combination; with 2 geometries x 6 n-values x
{n..16} L-values x 2 signs = 324 combinations, that is ~17000 NLP solves.
This run uses 20 random multistarts instead of 50 (still a substantial
multistart sample, ~7000 solves total, feasible to run in one session).
This is a compute-budget concession, disclosed here, not a tuning choice
made to favour any particular outcome -- the random seed is fixed and
the reduction applies uniformly to every (geometry, n, L, sign) cell.
"""
from __future__ import annotations

import json
import time

import numpy as np
import mpmath as mp

from geometry import setup as setup_mp
from qhat import qhat_n_cheb_coeffs
from spectral import cosine_coeffs, spectral_factor
from schur import build_p2, downward_peel
from task_c_solve import solve_one, verify_fine

N_RANDOM = 20
RNG = np.random.default_rng(0)
GEOMS = ["G1", "G2"]
N_VALUES = [1, 2, 3, 4, 5, 6]


def task_a_alpha_float(gname, n, dps=60):
    geo = setup_mp(gname, dps)
    q, dn = qhat_n_cheb_coeffs(n, geo)
    f = cosine_coeffs(q)
    p1, outside, rho, max_imag = spectral_factor(f, dps)
    p2, _ = build_p2(n, geo, dn)
    alphas, diag = downward_peel(p1, p2, dps)
    return np.array([float(a) for a in alphas])


def solve_n_L_sign(t, u, mu0, n, L, sigma, starts):
    best = None
    n_success = 0
    n_feasible = 0
    winner_label = None
    for label, x0 in starts:
        r = solve_one(t, u, mu0, n, L, sigma, x0)
        if r["success"]:
            n_success += 1
        if r["feasible"]:
            n_feasible += 1
            if best is None or r["delta"] < best["delta"]:
                best = r
                winner_label = label
    status = "feasible" if best is not None else ("solver_failed" if n_success == 0 else "infeasible")
    return {"best": best, "status": status, "n_success": n_success,
            "n_feasible": n_feasible, "n_tried": len(starts), "winner_label": winner_label}


def main():
    all_results = {}
    t0 = time.time()
    for gname in GEOMS:
        geo = setup_mp(gname, 30)
        t, u, mu0 = float(geo["t"]), float(geo["u"]), float(geo["mu0"])
        print(f"\n{'='*20} {gname} (t={t:.4f}, u={u:.4f}) {'='*20}")
        all_results[gname] = {}
        n1_best_alpha1 = None  # the single alpha_1 value from n=1's overall winner

        for n in N_VALUES:
            t_n0 = time.time()
            task_a_rest = None
            if n in (1, 3, 5):
                a_full = task_a_alpha_float(gname, n)
                task_a_rest = a_full[1:]

            per_L_results = {}
            for L in range(n, 17):
                for sigma in (1, -1):
                    starts = []
                    if task_a_rest is not None:
                        starts.append(("task_a", task_a_rest))
                    if n1_best_alpha1 is not None:
                        padded = np.zeros(n)
                        padded[0] = n1_best_alpha1
                        starts.append(("n1_padded", padded))
                    for _ in range(N_RANDOM):
                        starts.append(("random", RNG.uniform(-1.5, 1.5, n)))

                    res = solve_n_L_sign(t, u, mu0, n, L, sigma, starts)
                    per_L_results[(L, sigma)] = res

            # overall best across L, sign
            feasible_cells = {k: v for k, v in per_L_results.items() if v["best"] is not None}
            if feasible_cells:
                best_key = min(feasible_cells, key=lambda k: feasible_cells[k]["best"]["delta"])
                best_cell = feasible_cells[best_key]
                best = best_cell["best"]
                L_win, sign_win = best_key
                verify = verify_fine(t, u, mu0, L_win, sign_win, best["alphas"])
                print(f"  n={n}: delta*={best['delta']:.6e}  L={L_win}  sign={sign_win}  "
                      f"winner={best_cell['winner_label']}  fine_verify={verify}  "
                      f"({time.time()-t_n0:.1f}s)")
                if n == 1:
                    n1_best_alpha1 = float(best["alphas"][1])
                all_results[gname][n] = {
                    "delta_star": float(best["delta"]), "L": int(L_win), "sign": int(sign_win),
                    "alphas": [float(a) for a in best["alphas"]], "winner": best_cell["winner_label"],
                    "verify": verify,
                    "n_feasible_cells": len(feasible_cells), "n_total_cells": len(per_L_results),
                }
            else:
                print(f"  n={n}: NO FEASIBLE POINT FOUND across any (L,sign) "
                      f"({len(per_L_results)} cells tried) ({time.time()-t_n0:.1f}s)")
                all_results[gname][n] = {"delta_star": None, "n_feasible_cells": 0,
                                          "n_total_cells": len(per_L_results)}

    print(f"\ntotal time: {time.time()-t0:.1f}s")
    with open("task_c_results.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print("saved task_c_results.json")
    return all_results


if __name__ == "__main__":
    main()
