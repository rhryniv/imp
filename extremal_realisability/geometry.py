"""Geometry and constants for the extremal-realisability brief."""
from __future__ import annotations

import mpmath as mp


def setup(dps=50):
    mp.mp.dps = dps
    a = mp.cos(mp.pi / 4)
    b = mp.cos(3 * mp.pi / 4)
    mu0 = mp.mpf(1)
    S = mp.sinh(mu0) ** 2
    gamma = 2 * mp.acosh(mp.sqrt((1 - b) / (1 - a)))
    return {"a": a, "b": b, "mu0": mu0, "S": S, "gamma": gamma}


def delta_n(n, geo):
    return geo["S"] / mp.cosh(n * geo["gamma"] / 2) ** 2


KNOWN_DELTA = {
    1: mp.mpf("2.369589283664516e-01"),
    3: mp.mpf("5.742415920014411e-04"),
    5: mp.mpf("1.269945501896186e-06"),
    7: mp.mpf("2.807925761e-09"),
    9: mp.mpf("6.208489623e-12"),
}

if __name__ == "__main__":
    geo = setup(50)
    print("a =", geo["a"])
    print("b =", geo["b"])
    print("gamma =", geo["gamma"])
    print("S =", geo["S"])
    for n in [1, 3, 5, 7, 9]:
        dn = delta_n(n, geo)
        known = KNOWN_DELTA[n]
        print(f"n={n}  delta_n = {mp.nstr(dn, 16)}  known = {mp.nstr(known, 16)}  "
              f"reldiff = {mp.nstr(abs(dn-known)/known, 4)}")
