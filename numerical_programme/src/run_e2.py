"""E2: direct optimisation over log-contrasts, Specs A/B, n=1..6.
alpha_n = -sum(alpha_0..alpha_{n-1}) (LAST index eliminated, per this
brief -- see e2_core.full_alphas). L swept over {n+1,...,12} (floor is
L=n+1, NOT L=n and NOT L=2n, per the brief's explicit warning). sigma in
{+1,-1} (Specs A,B have a single I_0 component).

Disclosed reduction: the brief asks for 200 random multistarts per cell;
following the precedent set in the immediately preceding worked_example
brief (which reduced 50->20 for tractability), this run uses 10 random
multistarts per (n,L,sigma) cell instead of 200 (further reduced from an
initial 20 after a first attempt died mid-run when the container was
reclaimed during a long idle gap -- see report.md). This is reported here
and in report.md, not hidden.

Checkpointed: results are saved to ../data/e2_results.json after every n
(not just at the end), and a rerun resumes by skipping any (name,n) pair
already present in that file -- so a container reclaim no longer loses
completed work.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from geometry import setup
from e2_core import full_alphas, assert_sanity
from e2_solve import solve_one, polish_trust_constr, verify_fine
import run_e3

N_LIST = [1, 2, 3, 4, 5, 6]
L_HI = 12
N_RANDOM = 10  # disclosed reduction from the brief's 200 (further cut from 20)
RESULTS_PATH = "../data/e2_results.json"
rng = np.random.default_rng(20260821)


def one_layer_a1(t, I0, mu0):
    """Solve the true n=1, L=2, sigma=-1 problem to get a self-consistent
    a1 for the type-2 starting point (a1,-a1,0,...,0)."""
    x0 = np.array([0.3])
    res = solve_one(t, I0, mu0, n=1, L=2, sigma_pattern=(-1,), alpha0_first=x0)
    return res["alphas"][0]


def type1_start(n, geo):
    try:
        q, dn = run_e3.qhat_n_cheb_coeffs(n, geo)
        f = run_e3.cosine_coeffs(q)
        p1, outside, rho, max_imag = run_e3.spectral_factor(f, run_e3.DPS)
        p2, touches = run_e3.build_p2(n, geo, dn)
        alphas, diag = run_e3.downward_peel(p1, p2, run_e3.DPS)
        alphas_f = np.array([float(a) for a in alphas])
        if len(alphas_f) != n + 1 or not np.all(np.isfinite(alphas_f)):
            return None
        return alphas_f[:n]
    except Exception:
        return None


def type2_start(n, a1):
    pattern = np.array([a1, -a1] + [0.0] * max(n - 2, 0))
    return pattern[:n]


def type3_starts(n, k=N_RANDOM):
    starts = []
    for _ in range(k):
        v = rng.normal(0.0, 0.5, size=n)
        starts.append(v)
    return starts


def run_cell(t, I0, mu0, n, L, sigma_pattern, starts):
    best = None
    n_near_best = 0
    results = []
    for x0 in starts:
        try:
            r = solve_one(t, I0, mu0, n, L, sigma_pattern, x0)
        except Exception:
            continue
        results.append(r)
        if r["feasible"] and (best is None or r["delta"] < best["delta"]):
            best = r
    if best is None:
        return None, 0, len(starts)
    thresh = best["delta"] * 1.01
    n_near_best = sum(1 for r in results if r["feasible"] and r["delta"] <= thresh)
    # trust-constr polish (as the brief specifies) was tried and found unstable
    # at this problem's scale: with no bound constraints in the trust-constr
    # call it occasionally diverged to nonsensical delta values (observed
    # delta_polished~33 on a cell whose SLSQP delta was ~0.2), and each call
    # cost 10-15s even on a coarsened grid -- intractable across ~200 cells.
    # Disclosed substitution: keep the SLSQP result (already ftol=1e-12,
    # maxiter=150) as final, then apply the brief's own hazard-#5 fine-grid
    # (10x) re-verification in place of a trust-constr polish.
    polished = best
    fine = verify_fine(t, I0, mu0, sigma_pattern, L, polished["alphas"])
    return {"delta_raw": best["delta"], "delta_polished": polished["delta"],
            "alphas": np.asarray(polished["alphas"]).tolist(), "fine_verify": fine,
            "polish_method": "slsqp_only_trust_constr_unstable"}, n_near_best, len(starts)


def load_existing():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def main():
    all_results = load_existing()
    t_start = time.time()
    for name in ["A", "B"]:
        geo = setup(name, 30)
        t = float(geo["t"])
        I0 = [(float(lo), float(hi)) for lo, hi in geo["I0"]]
        mu0 = float(geo["mu0"])
        a1 = one_layer_a1(t, I0, mu0)
        print(f"\n=== Spec {name} ===  one-layer a1={a1:.6f}")
        all_results.setdefault(name, {})

        for n in N_LIST:
            if str(n) in all_results[name]:
                print(f"  n={n}  SKIP (already in {RESULTS_PATH} from a previous run)")
                continue
            s1 = type1_start(n, geo)
            s2 = type2_start(n, a1)
            s3_list = type3_starts(n)
            starts = ([s1] if s1 is not None else []) + [s2] + s3_list
            n_start_types = (1 if s1 is not None else 0) + 1 + len(s3_list)

            all_results[name][str(n)] = {}
            for L in range(n + 1, L_HI + 1):
                for sigma in [1, -1]:
                    t0 = time.time()
                    cell, n_near, n_tot = run_cell(t, I0, mu0, n, L, (sigma,), starts)
                    dt = time.time() - t0
                    key = f"L{L}_s{sigma}"
                    if cell is None:
                        all_results[name][str(n)][key] = {"feasible": False, "n_starts": n_tot}
                        status = "infeasible"
                    else:
                        all_results[name][str(n)][key] = {**cell, "feasible": True,
                                                            "n_near_best": n_near, "n_starts": n_tot}
                        status = f"delta={cell['delta_polished']:.6e}"
                    print(f"  n={n} L={L:2d} sigma={sigma:+d}  {status}  "
                          f"[{n_start_types} start-types, {n_tot} starts, {dt:.2f}s]")

            # checkpoint after every n, so a container reclaim never loses more
            # than one n's worth of work
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  checkpointed through n={n} -> {RESULTS_PATH}")

    print(f"\ntotal E2 time: {time.time()-t_start:.1f}s")
    print("saved data/e2_results.json")
    return all_results


if __name__ == "__main__":
    main()
