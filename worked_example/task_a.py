"""Task A: explicit Chebyshev design, both geometries, n in {1,3,5}."""
from __future__ import annotations

import json
import mpmath as mp

from geometry import setup, delta_n, KNOWN_DELTA_MAG
from qhat import qhat_n_cheb_coeffs, cheb_eval
from spectral import cosine_coeffs, spectral_factor, Q_of_vartheta, poly_eval
from schur import build_p2, downward_peel
from layers import forward_recursion

DPS = 60
N_VALUES = [1, 3, 5]
GEOMS = ["G1", "G2"]


def kappa_B_via_p1(p1, L, vartheta):
    w = mp.e ** (-1j * vartheta)
    return mp.re(mp.e ** (1j * L * vartheta / 2) * poly_eval(p1, w))


def run_n(geo, n):
    q, dn = qhat_n_cheb_coeffs(n, geo)
    known = KNOWN_DELTA_MAG.get((geo["name"], n))
    delta_check = abs(dn - known) / known if known else None

    a, b = geo["a"], geo["b"]
    cosh2mu0 = mp.cosh(geo["mu0"]) ** 2
    grid_n = 400
    xs_full = [mp.mpf(-1) + 2 * mp.mpf(i) / grid_n for i in range(grid_n + 1)]
    xs_F1 = [a + (1 - a) * mp.mpf(i) / grid_n for i in range(grid_n + 1)]
    xs_F0 = [mp.mpf(-1) + (b + 1) * mp.mpf(i) / grid_n for i in range(grid_n + 1)]
    prop = {
        "min_full_minus_1": min(cheb_eval(q, x) for x in xs_full) - 1,
        "1plusdn_minus_maxF1": (1 + dn) - max(cheb_eval(q, x) for x in xs_F1),
        "minF0_minus_cosh2mu0": min(cheb_eval(q, x) for x in xs_F0) - cosh2mu0,
        "Qhat_at_1_minus_1": cheb_eval(q, mp.mpf(1)) - 1,
        "leading_nonzero": q[-1] != 0,
    }

    f = cosine_coeffs(q)
    p1, outside, rho, max_imag = spectral_factor(f, DPS)

    max_Qcheck = mp.mpf(0)
    for i in range(0, 201):
        vartheta = mp.pi * i / 200
        w = mp.e ** (-1j * vartheta)
        lhs = abs(poly_eval(p1, w)) ** 2
        rhs = Q_of_vartheta(f, vartheta)
        max_Qcheck = max(max_Qcheck, abs(lhs - rhs))

    p2v, touch_varthetas = build_p2(n, geo, dn)
    alphas, diag = downward_peel(p1, p2v, DPS)
    p1_check, p2_check = forward_recursion(alphas)
    residual = max(abs(p1_check[k] - p1[k]) for k in range(n + 1))

    rho_j = []
    cum = mp.mpf(0)
    for j in range(n):
        cum += alphas[j]
        rho_j.append(mp.e ** cum)

    return {
        "n": n, "dn": dn, "delta_check_reldiff": delta_check, "properties": prop,
        "p1": p1, "rho_cond": rho, "max_imag": max_imag, "max_Qcheck": max_Qcheck,
        "alphas": alphas, "layer_residual": residual, "rho_j": rho_j,
    }


def sweep_L(p1, geo, L_lo, L_hi, grid_n=1200):
    t, u = geo["t"], geo["u"]
    I1 = (mp.mpf(0), t)
    I0 = (u, mp.pi)
    results = {}
    grid_I0 = [I0[0] + (I0[1] - I0[0]) * i / grid_n for i in range(grid_n + 1)]
    grid_I1 = [I1[0] + (I1[1] - I1[0]) * i / grid_n for i in range(grid_n + 1)]
    for L in range(L_lo, L_hi + 1):
        kap0 = [kappa_B_via_p1(p1, L, v) for v in grid_I0]
        kap1 = [kappa_B_via_p1(p1, L, v) for v in grid_I1]
        kappa_min = min(abs(k) for k in kap0)
        E_holds = all(abs(k) <= 1 + mp.mpf('1e-9') for k in kap1)
        signs = set(1 if k > 0 else -1 for k in kap0)
        const_sign = len(signs) == 1
        sign = signs.pop() if const_sign else None
        results[L] = {"kappa_min": kappa_min, "E_holds": E_holds,
                      "const_sign": const_sign, "sign": sign}
    return results


def main():
    all_results = {}
    for gname in GEOMS:
        geo = setup(gname, DPS)
        all_results[gname] = {}
        print(f"\n{'='*20} {gname} (t={mp.nstr(geo['t'],6)}, u={mp.nstr(geo['u'],6)}) {'='*20}")
        for n in N_VALUES:
            r = run_n(geo, n)
            print(f"\n--- n={n} ---")
            print(f"  delta_n = {mp.nstr(r['dn'],10)}  reldiff vs brief = {mp.nstr(r['delta_check_reldiff'],3)}")
            p = r["properties"]
            print(f"  properties: {[mp.nstr(v,3) if not isinstance(v,bool) else v for v in p.values()]}")
            print(f"  rho(min|z_l|)={mp.nstr(r['rho_cond'],8)}  max_Qcheck={mp.nstr(r['max_Qcheck'],3)}  "
                  f"schur residual={mp.nstr(r['layer_residual'],3)}")
            print(f"  alphas (4sf): {[mp.nstr(x,4) for x in r['alphas']]}")
            print(f"  rho_j (4sf):  {[mp.nstr(x,4) for x in r['rho_j']]}")

            sweep = sweep_L(r["p1"], geo, n, 16)
            L_best = max(sweep, key=lambda L: sweep[L]["kappa_min"])
            kmin_best = sweep[L_best]["kappa_min"]
            mu_eff = mp.acosh(kmin_best) if kmin_best >= 1 else None
            print(f"  L sweep {n}..16: L_best={L_best}  kappa_min={mp.nstr(kmin_best,6)}  "
                  f"mu_eff={'undefined' if mu_eff is None else mp.nstr(mu_eff,6)}  "
                  f"mu_eff>=mu0: {False if mu_eff is None else mu_eff>=geo['mu0']}")
            print(f"  (E) at L_best: {sweep[L_best]['E_holds']}  const_sign: {sweep[L_best]['const_sign']}  "
                  f"sign: {sweep[L_best]['sign']}")
            print(f"  profile: {[(L, mp.nstr(sweep[L]['kappa_min'],4)) for L in sorted(sweep)]}")

            r["sweep"] = sweep
            r["L_best"] = L_best
            r["mu_eff"] = mu_eff
            all_results[gname][n] = r

    return all_results


if __name__ == "__main__":
    main()
