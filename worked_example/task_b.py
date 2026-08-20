"""Task B: the sweep bound L*|J|/2 < pi + V_J(psi), tested against Task A's
actual const_sign outcomes on I_0."""
from __future__ import annotations

import mpmath as mp

from geometry import setup
from qhat import qhat_n_cheb_coeffs
from spectral import cosine_coeffs, spectral_factor, poly_eval
from task_a import sweep_L

DPS = 60
N_VALUES = [1, 3, 5]
GEOMS = ["G1", "G2"]


def total_variation_psi(p1, a, b, grid_n=4000):
    """V_[a,b](psi), psi a continuous branch of arg(p1(e^{-i*vartheta}))."""
    grid = [a + (b - a) * i / grid_n for i in range(grid_n + 1)]
    psis = [mp.arg(poly_eval(p1, mp.e ** (-1j * v))) for v in grid]
    unwrapped = [psis[0]]
    for i in range(1, len(psis)):
        d = psis[i] - psis[i - 1]
        while d > mp.pi:
            d -= 2 * mp.pi
        while d < -mp.pi:
            d += 2 * mp.pi
        unwrapped.append(unwrapped[-1] + d)
    # total variation = sum of |consecutive differences| of the unwrapped branch
    tv = sum(abs(unwrapped[i] - unwrapped[i - 1]) for i in range(1, len(unwrapped)))
    return tv


def main():
    all_tallies = {}
    for gname in GEOMS:
        geo = setup(gname, DPS)
        t, u = geo["t"], geo["u"]
        I0_width = mp.pi - u
        tally = {"predicted_allowed_and_const": 0, "predicted_allowed_not_const": 0,
                 "predicted_disallowed_and_const": 0, "predicted_disallowed_not_const": 0}
        print(f"\n{'='*20} {gname} {'='*20}")
        for n in N_VALUES:
            q, dn = qhat_n_cheb_coeffs(n, geo)
            f = cosine_coeffs(q)
            p1, outside, rho, max_imag = spectral_factor(f, DPS)
            V = total_variation_psi(p1, u, mp.pi)
            L_cap = 2 * (mp.pi + V) / I0_width
            print(f"\n  n={n}: V_I0(psi)={mp.nstr(V,6)}  L_cap={mp.nstr(L_cap,6)}")

            sweep = sweep_L(p1, geo, n, 16)
            rows = []
            for L in sorted(sweep):
                predicted_allowed = L <= L_cap
                actually_const = sweep[L]["const_sign"]
                rows.append((L, predicted_allowed, actually_const))
                if predicted_allowed and actually_const:
                    tally["predicted_allowed_and_const"] += 1
                elif predicted_allowed and not actually_const:
                    tally["predicted_allowed_not_const"] += 1
                elif not predicted_allowed and actually_const:
                    tally["predicted_disallowed_and_const"] += 1
                else:
                    tally["predicted_disallowed_not_const"] += 1
            print(f"    L, predicted<=L_cap, actually_const: {rows}")
            violations = [L for L, pa, ac in rows if not pa and ac]
            if violations:
                print(f"    ** VIOLATION: constant sign found beyond L_cap at L={violations} "
                      f"(necessary condition should never be violated -- report this plainly) **")

        print(f"\n  {gname} tally: {tally}")
        all_tallies[gname] = tally

    return all_tallies


if __name__ == "__main__":
    main()
