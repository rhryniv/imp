"""Main sweep for Geometry C: n in {3,5,7}, d in {1,3,5,7,9} (all
sigma=(-1)^((n+d)/2), forced -- NOT enumerated, per spec Sec 3: even d
and the non-forced sign are excluded a priori as infeasible-as-theorems,
not searched and not counted against the budget).

Checkpointed to results_C_checkpoint.json after every case.

Continuation: from (n,d-2) and (n-2,d). Unlike Geometry A, sigma here is
not split into a free/forced pair -- it is a single value, entirely
forced by (n+d) parity, and (like Geometry A's forced component) FLIPS
between d and d-2 (since (n+(d-2))/2 = (n+d)/2 - 1). Continuation is
therefore keyed on (n,d) alone (no sigma dimension at all): the supplied
alpha_first is just a numerical starting point for SLSQP, re-solved
under whatever sigma is forced at the TARGET cell, not the source one.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from core import rho_from_alphas, A_minus1, full_alphas, block_poly
from optimize_C import solve_case, GEOMETRIES, Q_on_grid
from tnorm import eps0_direct, eps1_direct, eps0_formula

RNG_SEED = 20260828 + 100
CKPT_PATH = "results_C_checkpoint.json"
N_LIST = [3, 5, 7]
D_LIST = [1, 3, 5, 7, 9]
MU0 = 0.32303
NS_EPS = [1, 2, 5, 9, 20]


def load_ckpt():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}


def save_ckpt(results):
    with open(CKPT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def key(n, d):
    return f"n{n}_d{d}"


def eps_tables(alpha_first, n, L, sigma, betas):
    g = GEOMETRIES["C"]
    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)
    from core import kappa as kappa_fn

    eps0 = {}
    eps1 = {}
    eps0_form = {}
    G0 = np.linspace(g["I0"][0][0], g["I0"][0][1], 16001)
    Q0 = Q_on_grid(c, G0)
    kap0 = kappa_fn(c, L, G0)
    for N in NS_EPS:
        eps0[N] = eps0_direct(Q0, kap0, N)
        candidates = [eps0_formula(b, MU0, N) for b in betas] if betas else []
        eps0_form[N] = max(candidates) if candidates else None

    min_T1 = 2.0
    Q1_all, kap1_all = [], []
    for lo, hi in g["I1"]:
        seg = np.linspace(lo, hi, 16001)
        Q1_all.append(Q_on_grid(c, seg))
        kap1_all.append(kappa_fn(c, L, seg))
    Q1_all = np.concatenate(Q1_all)
    kap1_all = np.concatenate(kap1_all)
    for N in NS_EPS:
        eps1[N] = eps1_direct(Q1_all, kap1_all, N)

    return eps0, eps1, eps0_form


def main():
    results = load_ckpt()
    rng = np.random.default_rng(RNG_SEED)
    t_start = time.time()

    best_alpha_by = {}  # (n,d) -> alpha_first or None

    for n in N_LIST:
        for d in D_LIST:
            L = n + d
            sigma_val = (-1) ** (L // 2)
            sigma = (sigma_val,)
            k = key(n, d)

            if k in results:
                ba = results[k].get("alpha")
                best_alpha_by[(n, d)] = np.array(ba[:n]) if ba else None
                print(f"{k}  SKIP (checkpointed)")
                continue

            prev_same_n = best_alpha_by.get((n, d - 2))
            prev_nm2 = best_alpha_by.get((n - 2, d))

            t0 = time.time()
            best, nconv, ntot, wall = solve_case(
                "C", n, d, sigma, rng, prev_same_n=prev_same_n, prev_n_minus_2=prev_nm2,
                n_random=200, n_antisym=50)

            if best is None:
                row = {"n": n, "d": d, "L": L, "sigma": sigma_val, "feasible": False,
                       "delta": None, "delta_per_I1_component": None, "alpha": None, "rho": None,
                       "A_minus1": None, "antisym_residual": None, "sym_residual": None,
                       "p_per_component": None, "a": None, "e": None, "kappa_min_I0": None,
                       "beta": None, "max_abs_kappa_I1_excluding_zero": None,
                       "eps0": None, "eps1": None, "eps0_formula": None,
                       "n_starts_converged": nconv, "n_starts_total": ntot,
                       "best_start_type": None, "wall_time_s": time.time() - t0}
                best_alpha_by[(n, d)] = None
                print(f"{k}  sigma={sigma_val}  INFEASIBLE  [{nconv}/{ntot} converged, "
                      f"{row['wall_time_s']:.1f}s]")
            else:
                diag = best["diag"]
                alpha_first = best["alpha_first"]
                full = np.concatenate([alpha_first, [-np.sum(alpha_first)]])
                antisym_res = float(np.max(np.abs(full + full[::-1])))
                sym_res = float(np.max(np.abs(full - full[::-1])))
                eps0, eps1, eps0_form = eps_tables(alpha_first, n, L, sigma, diag["betas"])
                row = {"n": n, "d": d, "L": L, "sigma": sigma_val, "feasible": True,
                       "delta": diag["delta_fine"],
                       "delta_per_I1_component": diag["delta_per_component"],
                       "alpha": full.tolist(), "rho": rho_from_alphas(alpha_first).tolist(),
                       "A_minus1": A_minus1(alpha_first),
                       "antisym_residual": antisym_res, "sym_residual": sym_res,
                       "p_per_component": diag["p_per_component"], "a": diag["a_count"],
                       "e": diag["e_count"], "kappa_min_I0": diag["kappa_min_I0"],
                       "beta": diag["betas"],
                       "max_abs_kappa_I1_excluding_zero": diag["max_abs_kappa_I1_excl0"],
                       "eps0": eps0, "eps1": eps1, "eps0_formula": eps0_form,
                       "n_starts_converged": nconv, "n_starts_total": ntot,
                       "best_start_type": best["start_type"], "wall_time_s": time.time() - t0,
                       "cross_check_max_diff": diag["cross_check_max_diff"]}
                best_alpha_by[(n, d)] = alpha_first
                print(f"{k}  sigma={sigma_val}  delta={row['delta']:.6e}  "
                      f"[{nconv}/{ntot} converged, start={row['best_start_type']}, "
                      f"{row['wall_time_s']:.1f}s]")

            results[k] = row
            save_ckpt(results)

    print(f"\ntotal Geometry-C sweep time: {time.time()-t_start:.1f}s")
    save_ckpt(results)
    print(f"saved {CKPT_PATH}")
    return results


if __name__ == "__main__":
    import sys
    from gates_C import run_gate

    print("=== Gate (spec Sec 4) ===")
    ok, _ = run_gate()
    if not ok:
        print("\nGATE FAILED -- aborting per spec.")
        sys.exit(1)

    print("\n=== Full 15-case sweep ===")
    main()

    print("\n=== Building results_C.csv ===")
    import runpy
    runpy.run_path("make_outputs_C.py", run_name="__main__")
