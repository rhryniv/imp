"""Regression gates (spec Sec 4). Run first; abort the main sweep if any
fails.
"""
from __future__ import annotations

import numpy as np

from core import full_alphas
from optimize import solve_case, MU0, COSH_MU0

RNG_SEED = 20260825


def run_gates():
    results = {}
    overall_pass = True

    # ---- G1: n=1, d=1 (L=2), sigma=-1 -> alpha=(0.31118,-0.31118), delta=0.064437
    rng = np.random.default_rng(RNG_SEED)
    best, nconv, ntot, wall = solve_case(1, 1, -1, rng, n_random=200, n_antisym=50)
    a0 = best["alpha_first"][0] if best else None
    delta = best["delta_fine"] if best else None
    g1_alpha_ok = best is not None and np.isclose(a0, 0.31118, atol=5e-6)
    g1_delta_ok = best is not None and np.isclose(delta, 0.064437, atol=5e-6)
    g1_pass = g1_alpha_ok and g1_delta_ok
    overall_pass &= g1_pass
    results["G1"] = {"pass": bool(g1_pass), "alpha0": a0, "delta": delta,
                      "target_alpha0": 0.31118, "target_delta": 0.064437,
                      "nconv": nconv, "ntot": ntot, "wall_s": wall}
    print(f"G1 (n=1,d=1,sigma=-1): alpha0={a0} (target 0.31118), delta={delta} "
          f"(target 0.064437)  -> {'PASS' if g1_pass else 'FAIL'}")

    # ---- G2: n=3, d=1 (L=4), sigma=+1 -> delta=4.6722e-4, alpha matches seed
    #      to 5 dp, from RANDOM+ANTISYM starts only (exclude seed_ex, continuation)
    rng = np.random.default_rng(RNG_SEED + 1)
    best2, nconv2, ntot2, wall2 = solve_case(3, 1, 1, rng, n_random=200, n_antisym=50,
                                              include_seed_ex=False)
    delta2 = best2["delta_fine"] if best2 else None
    g2_delta_ok = best2 is not None and np.isclose(delta2, 4.6722e-4, rtol=0, atol=5e-8)
    # alpha match to 5 d.p. against the seed, allowing for the exact global
    # sign-flip degeneracy of the problem (negating all alphas leaves c,
    # hence Q and kappa, exactly invariant -- verified algebraically: the
    # forward recursion's p1 branch is invariant under alpha -> -alpha).
    seed_full = np.array([0.16228, -0.39891, 0.39891, -0.16228])
    g2_alpha_ok = False
    seed_first = None
    if best2 is not None:
        found_full = full_alphas(best2["alpha_first"])
        seed_first = seed_full
        g2_alpha_ok = (np.allclose(found_full, seed_full, atol=5e-5) or
                       np.allclose(found_full, -seed_full, atol=5e-5))
    g2_pass = g2_delta_ok and g2_alpha_ok
    overall_pass &= g2_pass
    results["G2"] = {"pass": bool(g2_pass), "delta": delta2, "target_delta": 4.6722e-4,
                      "alpha_first": best2["alpha_first"].tolist() if best2 else None,
                      "target_alpha_full": seed_full.tolist(),
                      "start_type": best2["start_type"] if best2 else None,
                      "nconv": nconv2, "ntot": ntot2, "wall_s": wall2}
    print(f"G2 (n=3,d=1,sigma=+1, random+antisym only): delta={delta2} (target 4.6722e-4), "
          f"alpha_full={full_alphas(best2['alpha_first']) if best2 else None} "
          f"(target +-{seed_full})  -> {'PASS' if g2_pass else 'FAIL'}")

    # ---- G3: matching condition on ANY converged case (reuse G1's and G2's best)
    g3_checks = []
    for label, b in [("G1", best), ("G2", best2)]:
        if b is None:
            continue
        alphas = full_alphas(b["alpha_first"])
        sum_alpha = float(np.sum(alphas))
        from core import block_poly
        c = block_poly(alphas)
        sum_c = float(np.sum(c))
        ok = abs(sum_alpha) < 1e-12 and abs(sum_c - 1.0) < 1e-12
        g3_checks.append({"case": label, "sum_alpha": sum_alpha, "sum_c": sum_c, "pass": ok})
        print(f"G3 ({label}): sum(alpha)={sum_alpha:.3e} (tol 1e-12), "
              f"sum(c)-1={sum_c-1:.3e} (tol 1e-12)  -> {'PASS' if ok else 'FAIL'}")
    g3_pass = all(chk["pass"] for chk in g3_checks) and len(g3_checks) > 0
    overall_pass &= g3_pass
    results["G3"] = {"pass": bool(g3_pass), "checks": g3_checks}

    print(f"\nGATES OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return overall_pass, results


if __name__ == "__main__":
    import json
    ok, res = run_gates()
    with open("gates_result.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    if not ok:
        raise SystemExit("GATES FAILED -- see gates_result.json; aborting per spec Sec 4.")
