"""Main 66-case sweep: n in {1,3,5}, d in 1..11, sigma in {+1,-1}.
Checkpointed to results_checkpoint.json after every case so a container
reclaim never loses more than the in-flight case (lesson learned from an
earlier, unrelated run in this session that lost work to an un-checkpointed
background process).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from core import rho_from_alphas
from optimize import solve_case, I0, COSH_MU0

RNG_SEED = 20260825
CKPT_PATH = "results_checkpoint.json"
N_LIST = [1, 3, 5]
D_LIST = list(range(1, 12))
SIGMAS = [1, -1]
ABS_I0 = I0[1] - I0[0]


def load_ckpt():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}


def save_ckpt(results):
    with open(CKPT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def key(n, d, sigma):
    return f"n{n}_d{d}_s{sigma}"


def main():
    results = load_ckpt()
    rng = np.random.default_rng(RNG_SEED + 100)  # offset from the gates' seed
    t_start = time.time()

    # per-(n,sigma) history of best alpha_first, keyed by d, for continuation
    best_alpha_by_nd = {}  # (n,d,sigma) -> alpha_first (python list) or None

    for n in N_LIST:
        for d in D_LIST:
            L = n + d
            for sigma in SIGMAS:
                k = key(n, d, sigma)
                if k in results:
                    ba = results[k].get("alpha_first")
                    best_alpha_by_nd[(n, d, sigma)] = np.array(ba) if ba else None
                    print(f"{k}  SKIP (checkpointed)")
                    continue

                prev_same_n = best_alpha_by_nd.get((n, d - 1, sigma))
                prev_nm2 = best_alpha_by_nd.get((n - 2, d, sigma))

                t0 = time.time()
                best, nconv, ntot, wall = solve_case(
                    n, d, sigma, rng, prev_same_n=prev_same_n, prev_n_minus_2=prev_nm2,
                    n_random=200, n_antisym=50)

                if best is None:
                    row = {"n": n, "d": d, "L": L, "sigma": sigma, "feasible": False,
                           "delta": None, "alpha_first": None, "alpha": None, "rho": None,
                           "kappa_min_I0": None, "kappa_max_I1": None, "Q_max_I0": None,
                           "beta_J": None, "d_cap_aposteriori": None, "min_Q": None,
                           "n_starts_converged": nconv, "n_starts_total": ntot,
                           "best_start_type": None, "solver_iters": None,
                           "wall_time_s": time.time() - t0}
                    best_alpha_by_nd[(n, d, sigma)] = None
                    print(f"{k}  INFEASIBLE  [{nconv}/{ntot} converged, {row['wall_time_s']:.1f}s]")
                else:
                    diag = best["diag"]
                    alpha_first = best["alpha_first"]
                    rho = rho_from_alphas(alpha_first)
                    Q_J = diag["Q_J"]
                    beta_J = float(np.arccos(min(COSH_MU0 / np.sqrt(Q_J), 1.0))) if Q_J > 0 else None
                    d_cap = 4 * beta_J / ABS_I0 if beta_J is not None else None
                    row = {"n": n, "d": d, "L": L, "sigma": sigma, "feasible": True,
                           "delta": diag["delta_fine"],
                           "alpha_first": alpha_first.tolist(),
                           "alpha": (alpha_first.tolist() + [float(-np.sum(alpha_first))]),
                           "rho": rho.tolist(),
                           "kappa_min_I0": diag["kappa_min"], "kappa_max_I1": diag["kappa_max"],
                           "Q_max_I0": Q_J, "beta_J": beta_J, "d_cap_aposteriori": d_cap,
                           "min_Q": diag["min_Q"], "cross_check_max_diff": diag["cross_check_max_diff"],
                           "n_starts_converged": nconv, "n_starts_total": ntot,
                           "best_start_type": best["start_type"], "solver_iters": best["iters"],
                           "wall_time_s": time.time() - t0}
                    best_alpha_by_nd[(n, d, sigma)] = alpha_first
                    print(f"{k}  delta={row['delta']:.6e}  [{nconv}/{ntot} converged, "
                          f"start={row['best_start_type']}, {row['wall_time_s']:.1f}s]")

                results[k] = row
                save_ckpt(results)

    print(f"\ntotal sweep time: {time.time()-t_start:.1f}s")
    save_ckpt(results)
    print(f"saved {CKPT_PATH}")
    return results


if __name__ == "__main__":
    main()
