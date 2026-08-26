"""Single entry point (spec deliverable (d)): gate check (abort on
failure) -> main sweep for both multi-component geometries -> clustered
active-point reanalysis -> results_multi.csv. Deterministic given the
RNG seeds in config_multi.json/gates_multi.py/this file.

    python3 run_multi.py

Checkpointed to
results_multi_checkpoint.json after every case (lesson learned: a
container reclaim during a long idle gap silently killed an earlier,
un-checkpointed background run in this session).

Geometry A (pi in I0, two stop components): d odd only, d in
{1,3,5,7,9,11,13} (cap 2*pi/min|I0_i| = 13.33); sigma_2 (the component
containing pi) forced to (-1)^(L/2), sigma_1 free in {+1,-1}.

Geometry B (pi in I1, two pass components, one stop): d in {1,...,7}
(cap 2*pi/|I0| = 8, both parities of L admissible); sigma_1 (the single
stop component) free in {+1,-1}. Geometry B additionally gets 50
alpha_j=alpha_{n-j} symmetric starts (not used for A, per spec Sec 4).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from core import rho_from_alphas, A_minus1
from optimize_multi import solve_case, GEOMETRIES

RNG_SEED = 20260827 + 100
CKPT_PATH = "results_multi_checkpoint.json"
N_LIST = [3, 5]

A_D_LIST = [1, 3, 5, 7, 9, 11, 13]
B_D_LIST = list(range(1, 8))


def load_ckpt():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}


def save_ckpt(results):
    with open(CKPT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def sigma_key(sigma):
    return "_".join(str(s) for s in sigma)


def key(geom, n, d, sigma):
    return f"{geom}_n{n}_d{d}_s{sigma_key(sigma)}"


def free_sigma(geom, sigma):
    """The FREE sign coordinate(s) of a case's sigma tuple, used to key
    continuation lookups. For Geometry A, sigma[1] (the pi-component) is
    forced by parity and FLIPS between d and d-2 (L/2 parity alternates
    every step of 2 in d), so a same-tuple lookup at d-2 would never
    match; continuation must therefore be keyed on the free coordinate
    sigma[0] alone, letting the forced coordinate re-derive itself at the
    target d. For Geometry B there is only one (free) coordinate, so this
    is the identity."""
    if geom == "A":
        return (sigma[0],)
    return sigma


def main():
    results = load_ckpt()
    rng = np.random.default_rng(RNG_SEED)
    t_start = time.time()

    best_alpha_by = {}  # (geom,n,d,sigma) -> alpha_first or None

    for geom in ["A", "B"]:
        d_list = A_D_LIST if geom == "A" else B_D_LIST
        for n in N_LIST:
            for d in d_list:
                L = n + d
                if geom == "A":
                    sigma2 = (-1) ** (L // 2)
                    sigma_list = [(1, sigma2), (-1, sigma2)]
                    n_sym = 0
                else:
                    sigma_list = [(1,), (-1,)]
                    n_sym = 50

                for sigma in sigma_list:
                    k = key(geom, n, d, sigma)
                    fs = free_sigma(geom, sigma)
                    if k in results:
                        ba = results[k].get("alpha")
                        alpha_first_ckpt = np.array(ba[:n]) if ba else None
                        best_alpha_by[(geom, n, d, fs)] = alpha_first_ckpt
                        print(f"{k}  SKIP (checkpointed)")
                        continue

                    prev_same_n = best_alpha_by.get((geom, n, d - 2, fs))
                    prev_nm2 = best_alpha_by.get((geom, n - 2, d, fs))

                    t0 = time.time()
                    best, nconv, ntot, wall = solve_case(
                        geom, n, d, sigma, rng, prev_same_n=prev_same_n,
                        prev_n_minus_2=prev_nm2, n_random=200, n_antisym=50, n_sym=n_sym)

                    if best is None:
                        row = {"geometry": geom, "n": n, "d": d, "L": L, "sigma": list(sigma),
                               "feasible": False, "delta": None, "alpha": None, "rho": None,
                               "A_minus1": None, "antisym_residual": None, "sym_residual": None,
                               "n_active_B_per_component": None, "n_active_C_per_component": None,
                               "kappa_min_per_C_component": None, "max_abs_kappa_I1": None,
                               "beta_per_active_C_point": None, "best_start_type": None,
                               "n_starts_converged": nconv, "n_starts_total": ntot,
                               "wall_time_s": time.time() - t0}
                        best_alpha_by[(geom, n, d, fs)] = None
                        print(f"{k}  INFEASIBLE  [{nconv}/{ntot} converged, {row['wall_time_s']:.1f}s]")
                    else:
                        diag = best["diag"]
                        alpha_first = best["alpha_first"]
                        full = np.concatenate([alpha_first, [-np.sum(alpha_first)]])
                        antisym_res = float(np.max(np.abs(full + full[::-1])))
                        sym_res = float(np.max(np.abs(full - full[::-1])))
                        row = {"geometry": geom, "n": n, "d": d, "L": L, "sigma": list(sigma),
                               "feasible": True, "delta": diag["delta_fine"],
                               "alpha": full.tolist(),
                               "rho": rho_from_alphas(alpha_first).tolist(),
                               "A_minus1": A_minus1(alpha_first),
                               "antisym_residual": antisym_res, "sym_residual": sym_res,
                               "n_active_B_per_component": diag["n_active_B_per_component"],
                               "n_active_C_per_component": diag["n_active_C_per_component"],
                               "kappa_min_per_C_component": diag["kappa_min_per_C"],
                               "max_abs_kappa_I1": diag["max_abs_kappa_I1"],
                               "beta_per_active_C_point": diag["beta_per_active_C"],
                               "best_start_type": best["start_type"],
                               "n_starts_converged": nconv, "n_starts_total": ntot,
                               "wall_time_s": time.time() - t0,
                               "cross_check_max_diff": diag["cross_check_max_diff"]}
                        best_alpha_by[(geom, n, d, fs)] = alpha_first
                        print(f"{k}  delta={row['delta']:.6e}  [{nconv}/{ntot} converged, "
                              f"start={row['best_start_type']}, {row['wall_time_s']:.1f}s]")

                    results[k] = row
                    save_ckpt(results)

    print(f"\ntotal multi-geometry sweep time: {time.time()-t_start:.1f}s")
    save_ckpt(results)
    print(f"saved {CKPT_PATH}")
    return results


if __name__ == "__main__":
    import sys
    from gates_multi import run_gate

    print("=== Gate (spec Sec 4) ===")
    ok, _ = run_gate()
    if not ok:
        print("\nGATE FAILED -- aborting per spec Sec 4.")
        sys.exit(1)

    print("\n=== Full 56-case sweep ===")
    main()

    print("\n=== Clustered active-point reanalysis ===")
    import runpy
    runpy.run_path("reanalyze.py", run_name="__main__")

    print("\n=== Building results_multi.csv ===")
    runpy.run_path("make_outputs_multi.py", run_name="__main__")
