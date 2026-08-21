"""E5: Green's function bound for Spec C's disconnected stop set.

g(x) = arccosh(|(2x-1-a)/(1-a)|) on F_1=[a,1] (a=cos(t)); this is the
brief's formula with its typo resolved as the standard Green's function
of C \\ F_1 evaluated on the real line outside F_1 (the l(x)=(2x-1-a)/(1-a)
affine map is the same one used throughout for F_1=[a,1]<->[-1,1]).

gamma = min over F_0 of g, where F_0 is the image under x=cos(theta) of
Spec C's two stop components I_0 = [pi/2,2pi/3] union [5pi/6,pi].
Compared against E1's Spec C delta_mag(n) via the thm:rate bound
S*exp(-gamma*n).
"""
from __future__ import annotations

import json

import mpmath as mp

from geometry import setup


def g(x, a):
    l = (2 * x - 1 - a) / (1 - a)
    return mp.acosh(abs(l))


def gamma_over_F0(geo, grid_n=4000):
    a = geo["a"]
    best = None
    best_x = None
    for lo_theta, hi_theta in geo["I0"]:
        # x = cos(theta); theta in [lo_theta,hi_theta] -> x in [cos(hi_theta),cos(lo_theta)]
        xlo, xhi = mp.cos(hi_theta), mp.cos(lo_theta)
        # check endpoints (g is monotone in |x-something|, min is always at
        # an endpoint of a component since g o cos is monotone within the
        # component -- no interior extremum for an affine-then-acosh map)
        for x in (xlo, xhi):
            val = g(x, a)
            if best is None or val < best:
                best = val
                best_x = x
        # also scan a fine grid as a direct check that no interior point beats
        # the endpoints (guards against a mistaken monotonicity assumption)
        for i in range(grid_n + 1):
            x = xlo + (xhi - xlo) * i / grid_n
            val = g(x, a)
            if val < best:
                best = val
                best_x = x
    return best, best_x


def main():
    geo = setup("C", 50)
    gamma, x_star = gamma_over_F0(geo)
    S = geo["S"]
    print(f"Spec C: a={mp.nstr(geo['a'],10)}  gamma_E5={mp.nstr(gamma,12)} at x*={mp.nstr(x_star,10)}")
    print(f"S = sinh^2(mu0) = {mp.nstr(S,10)}")

    with open("../data/e1_results.json") as f:
        e1 = json.load(f)

    rows = []
    print(f"\n{'n':>3} {'delta_mag(n) [LP]':>20} {'S*exp(-gamma n)':>20} {'bound holds?':>14} {'ratio':>10}")
    for n in range(1, 9):
        row = e1["C"][str(n)]
        dmag = row["delta_lp"]
        bound = float(S * mp.e ** (-gamma * n))
        holds = dmag <= bound
        ratio = dmag / bound
        rows.append({"n": n, "delta_mag_lp": dmag, "bound_S_exp": bound,
                     "holds": holds, "ratio": ratio})
        print(f"{n:>3} {dmag:>20.6e} {bound:>20.6e} {str(holds):>14} {ratio:>10.4f}")

    out = {"gamma": float(gamma), "x_star": float(x_star), "S": float(S), "rows": rows}
    with open("../data/e5_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved data/e5_results.json")
    return out


if __name__ == "__main__":
    main()
