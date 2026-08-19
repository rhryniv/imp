"""Full pipeline: Qhat_n -> Q -> p1 (Fejer-Riesz) -> p2 (explicit, from
the equioscillation touch points) -> alpha_j (explicit Schur-type
downward recursion) -> kappa_B(vartheta; L) -> kappa_min/mu_eff sweep
over L. No Newton iteration or seed-guessing anywhere.
"""
from __future__ import annotations

import json
import mpmath as mp

from geometry import setup, KNOWN_DELTA
from qhat import qhat_n_cheb_coeffs, cheb_eval
from spectral import cosine_coeffs, spectral_factor, Q_of_vartheta, poly_eval
from schur import build_p2, downward_peel
from layers import forward_recursion

DPS = 60
N_VALUES = [1, 3, 5, 7, 9]


def kappa_B_via_p1(p1, L, vartheta):
    w = mp.e ** (-1j * vartheta)
    return mp.re(mp.e ** (1j * L * vartheta / 2) * poly_eval(p1, w))


def kappa_B_via_sum(p1, L, vartheta):
    n = len(p1) - 1
    return sum(p1[j] * mp.cos((L - 2 * j) * vartheta / 2) for j in range(n + 1))


def run_n(n, geo):
    q, dn = qhat_n_cheb_coeffs(n, geo)
    known = KNOWN_DELTA[n]
    delta_check = abs(dn - known) / known

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

    p2, touch_varthetas = build_p2(n, geo, dn)
    alphas, diag = downward_peel(p1, p2, DPS)

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
        "schur_diag": diag,
    }


def sweep_L(p1, L_lo, L_hi, grid_n=1500, cross_check=True):
    I1 = (mp.mpf(0), mp.pi / 4)
    I0 = (3 * mp.pi / 4, mp.pi)
    results = {}
    grid_I0 = [I0[0] + (I0[1] - I0[0]) * i / grid_n for i in range(grid_n + 1)]
    grid_I1 = [I1[0] + (I1[1] - I1[0]) * i / grid_n for i in range(grid_n + 1)]
    for L in range(L_lo, L_hi + 1):
        kap0 = [kappa_B_via_p1(p1, L, v) for v in grid_I0]
        kap1 = [kappa_B_via_p1(p1, L, v) for v in grid_I1]
        if cross_check:
            # cross-check the two kappa_B formulas at a few points
            for v in grid_I0[::grid_n // 5]:
                d = abs(kappa_B_via_p1(p1, L, v) - kappa_B_via_sum(p1, L, v))
                assert d < mp.mpf('1e-40'), f"kappa_B formula mismatch at L={L}: {d}"
        kappa_min = min(abs(k) for k in kap0)
        E_holds = all(abs(k) <= 1 + mp.mpf('1e-9') for k in kap1)
        signs = set(1 if k > 0 else -1 for k in kap0)
        const_sign = len(signs) == 1
        sign = signs.pop() if const_sign else None
        results[L] = {"kappa_min": kappa_min, "E_holds": E_holds,
                      "const_sign": const_sign, "sign": sign}
    return results


def main():
    geo = setup(DPS)
    all_results = {}
    for n in N_VALUES:
        r = run_n(n, geo)
        all_results[n] = r

        print(f"\n=== n={n} ===")
        print(f"  delta_n = {mp.nstr(r['dn'],16)}  reldiff vs brief's value = {mp.nstr(r['delta_check_reldiff'],4)}")
        p = r["properties"]
        print(f"  property checks: min_full-1={mp.nstr(p['min_full_minus_1'],4)}  "
              f"(1+dn)-maxF1={mp.nstr(p['1plusdn_minus_maxF1'],4)}  "
              f"minF0-cosh2mu0={mp.nstr(p['minF0_minus_cosh2mu0'],4)}  "
              f"Qhat(1)-1={mp.nstr(p['Qhat_at_1_minus_1'],4)}  leading_nonzero={p['leading_nonzero']}")
        print(f"  Fejer-Riesz: rho(min|z_l|)={mp.nstr(r['rho_cond'],8)}  max_imag_residual={mp.nstr(r['max_imag'],4)}  "
              f"max|Qcheck diff|={mp.nstr(r['max_Qcheck'],4)}  p1(1)-1={mp.nstr(poly_eval(r['p1'],mp.mpc(1))-1,4)}")
        d = r["schur_diag"]
        print(f"  Schur peeling: max_lead_residual={mp.nstr(max(d['lead_residuals']) if d['lead_residuals'] else 0,4)}  "
              f"max_div_residual={mp.nstr(max(d['div_residuals']) if d['div_residuals'] else 0,4)}  "
              f"cosh2-sinh2-1={mp.nstr(d['cosh2_minus_sinh2_minus_1'],4)}  "
              f"forward-map check residual={mp.nstr(r['layer_residual'],4)}  sum(alphas)={mp.nstr(sum(r['alphas']),4)}")
        print(f"  alphas (4sf): {[mp.nstr(a,4) for a in r['alphas']]}")
        print(f"  rho_j (4sf):  {[mp.nstr(x,4) for x in r['rho_j']]}")

        L_lo, L_hi = 2 * n, 6 * n
        sweep = sweep_L(r["p1"], L_lo, L_hi)
        Ls = sorted(sweep)
        ext_note = ""
        while sweep[Ls[-1]]["kappa_min"] > sweep[Ls[-2]]["kappa_min"] and Ls[-1] - L_hi < 6 * n:
            new_lo, new_hi = Ls[-1] + 1, L_hi + n
            more = sweep_L(r["p1"], new_lo, new_hi)
            sweep.update(more)
            Ls = sorted(sweep)
            ext_note = f" (extended to L={Ls[-1]}; kappa_min was still increasing)"
        # supplementary: confirm the const_sign/kappa_min picture over a much
        # wider range (2n..12n), reported but not used to pick L_best (which
        # stays within the brief's own 2n..6n range unless extended above).
        wide = sweep_L(r["p1"], 2 * n, 12 * n, grid_n=500, cross_check=False)
        wide_const_sign = [(L, wide[L]["kappa_min"]) for L in sorted(wide) if wide[L]["const_sign"]]
        r_wide_summary = {"range": (2 * n, 12 * n), "const_sign_hits": wide_const_sign}

        L_best = max(sweep, key=lambda L: sweep[L]["kappa_min"])
        kmin_best = sweep[L_best]["kappa_min"]
        mu_eff = mp.acosh(kmin_best) if kmin_best >= 1 else None
        print(f"  L range swept: {min(sweep)}..{max(sweep)}{ext_note}")
        print(f"  L_best={L_best}  kappa_min={mp.nstr(kmin_best,8)}  "
              f"mu_eff={'undefined (kappa_min<1)' if mu_eff is None else mp.nstr(mu_eff,8)}  "
              f"mu_eff>=mu_0=1: {False if mu_eff is None else mu_eff >= 1}")
        print(f"  (E) holds at L_best: {sweep[L_best]['E_holds']}")
        print(f"  const sign on I_0 at L_best: {sweep[L_best]['const_sign']}  sign={sweep[L_best]['sign']}")
        print(f"  kappa_min vs L profile: {[(L, float(sweep[L]['kappa_min'])) for L in sorted(sweep)]}")

        print(f"  wide check (L=2n..12n): const_sign=True at {len(wide_const_sign)} of "
              f"{len(wide)} L values: {[(L, mp.nstr(km,4)) for L, km in wide_const_sign]}")

        all_results[n]["sweep"] = sweep
        all_results[n]["L_best"] = L_best
        all_results[n]["mu_eff"] = mu_eff
        all_results[n]["wide_check"] = r_wide_summary

    with open("run_all_results.json", "w") as fh:
        def conv(o):
            if isinstance(o, mp.mpf):
                return mp.nstr(o, 40)
            if isinstance(o, mp.mpc):
                return mp.nstr(o, 40)
            return str(o)
        json.dump({str(n): {k: v for k, v in r.items() if k not in ("p1", "sweep", "wide_check")}
                   for n, r in all_results.items()}, fh, indent=2, default=conv)
    print("\nsaved run_all_results.json")

    return all_results


if __name__ == "__main__":
    main()
