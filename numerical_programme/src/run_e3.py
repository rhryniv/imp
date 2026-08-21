"""E3: extremal realisability. Spec A (odd n=1,3,5,7) and Spec B (odd
n=1,3,5,7,9 -- extended to include n=9 so the brief's own cited
validation target, "0 hits out of 51-91 for n=5,7,9", can be checked).
"""
from __future__ import annotations

import json

import numpy as np
import mpmath as mp

from geometry import setup
from qhat import qhat_n_cheb_coeffs
from spectral import cosine_coeffs, spectral_factor, poly_eval
from schur import build_p2, downward_peel
from e2_core import forward_recursion_np, poly_eval_np

DPS = 60


def kappa_and_Q_mp(p1, L, theta):
    w = mp.e ** (-1j * theta)
    p1w = poly_eval(p1, w)
    Q = abs(p1w) ** 2
    kap = mp.re(mp.e ** (1j * L * theta / 2) * p1w)
    return Q, kap


def run_n(n, geo, L_hi=14, grid_n=800):
    q, dn = qhat_n_cheb_coeffs(n, geo)
    f = cosine_coeffs(q)
    p1, outside, rho, max_imag = spectral_factor(f, DPS)
    p2, touches = build_p2(n, geo, dn)
    alphas, diag = downward_peel(p1, p2, DPS)

    a = geo["a"]
    t = geo["t"]
    I0 = geo["I0"]
    mu0 = geo["mu0"]
    cosh_mu0 = mp.cosh(mu0)

    grid_I1 = [t * i / grid_n for i in range(grid_n + 1)]
    grid_I0 = [[lo + (hi - lo) * i / grid_n for i in range(grid_n + 1)] for lo, hi in I0]

    hits = []
    kappa_min_profile = {}
    for L in range(n + 1, L_hi + 1):
        for sigma in [1, -1]:
            E_ok = True
            for theta in grid_I1:
                _, kap = kappa_and_Q_mp(p1, L, theta)
                if abs(kap) > 1 + mp.mpf('1e-9'):
                    E_ok = False
                    break
            kmins = []
            C_ok = True
            for comp in grid_I0:
                vals = [kappa_and_Q_mp(p1, L, theta)[1] for theta in comp]
                kmin_signed = min(sigma * v for v in vals)
                kmins.append(kmin_signed)
                if kmin_signed < cosh_mu0:
                    C_ok = False
            kappa_min_profile[(L, sigma)] = float(min(kmins))
            if E_ok and C_ok:
                hits.append((L, sigma))

    # mu_eff at best L (max over L,sigma of min |kappa| on I_0, restricted to hits if any,
    # else just report the best kappa_min achieved regardless of feasibility)
    best_key = max(kappa_min_profile, key=lambda k: kappa_min_profile[k])
    best_kmin = kappa_min_profile[best_key]
    mu_eff = float(mp.acosh(best_kmin)) if best_kmin >= 1 else None

    return {"n": n, "delta_n": float(dn), "rho_cond": float(rho), "alphas": [float(a) for a in alphas],
            "n_hits": len(hits), "n_total": 2 * (L_hi - n), "hits": hits,
            "best_L_sigma": best_key, "best_kappa_min": best_kmin, "mu_eff": mu_eff,
            "p1": [float(c) for c in p1]}


def main():
    results = {}
    for name, n_list in [("A", [1, 3, 5, 7]), ("B", [1, 3, 5, 7, 9])]:
        geo = setup(name, DPS)
        results[name] = {}
        print(f"=== Spec {name} ===")
        for n in n_list:
            r = run_n(n, geo)
            results[name][n] = r
            print(f"  n={n}  delta_n={r['delta_n']:.4e}  rho={r['rho_cond']:.4f}  "
                  f"hits={r['n_hits']}/{r['n_total']}  best(L,sigma)={r['best_L_sigma']}  "
                  f"kappa_min={r['best_kappa_min']:.6f}  mu_eff={r['mu_eff']}")

    with open("../data/e3_results.json", "w") as f:
        json.dump({name: {str(n): {k: v for k, v in r.items() if k != "hits"} | {"hits": list(r["hits"])}
                           for n, r in d.items()} for name, d in results.items()}, f, indent=2, default=str)
    print("saved data/e3_results.json")
    return results


if __name__ == "__main__":
    main()
