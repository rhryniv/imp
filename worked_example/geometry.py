"""Geometry and constants, generalized to any (t, u) pair (pass edge t,
stop edge u), for the worked-example brief's G1/G2 geometries.
"""
from __future__ import annotations

import mpmath as mp

GEOMETRIES = {
    "G1": {"t_frac": (1, 2), "u_frac": (5, 6)},   # I1=[0,pi/2], I0=[5pi/6,pi]
    "G2": {"t_frac": (1, 4), "u_frac": (3, 4)},   # I1=[0,pi/4], I0=[3pi/4,pi]
}


def setup(name, dps=50, mu0=1):
    mp.mp.dps = dps
    tf, uf = GEOMETRIES[name]["t_frac"], GEOMETRIES[name]["u_frac"]
    t = mp.mpf(tf[0]) / tf[1] * mp.pi
    u = mp.mpf(uf[0]) / uf[1] * mp.pi
    a = mp.cos(t)
    b = mp.cos(u)
    mu0 = mp.mpf(mu0)
    S = mp.sinh(mu0) ** 2
    gamma = 2 * mp.acosh(mp.sqrt((1 - b) / (1 - a)))
    return {"name": name, "t": t, "u": u, "a": a, "b": b, "mu0": mu0, "S": S, "gamma": gamma}


def delta_n(n, geo):
    return geo["S"] / mp.cosh(n * geo["gamma"] / 2) ** 2


KNOWN_DELTA_MAG = {
    ("G1", 1): mp.mpf("7.40e-01"), ("G1", 3): mp.mpf("3.71e-02"),
    ("G1", 5): mp.mpf("1.35e-03"), ("G1", 7): mp.mpf("4.85e-05"),
    ("G2", 1): mp.mpf("2.37e-01"), ("G2", 3): mp.mpf("5.74e-04"),
    ("G2", 5): mp.mpf("1.27e-06"), ("G2", 7): mp.mpf("2.81e-09"),
}

if __name__ == "__main__":
    for name in ["G1", "G2"]:
        geo = setup(name, 50)
        print(f"=== {name} ===  t={mp.nstr(geo['t'],8)}  u={mp.nstr(geo['u'],8)}  "
              f"a={mp.nstr(geo['a'],8)}  b={mp.nstr(geo['b'],8)}  gamma={mp.nstr(geo['gamma'],8)}")
        for n in [1, 3, 5, 7]:
            dn = delta_n(n, geo)
            known = KNOWN_DELTA_MAG[(name, n)]
            print(f"  n={n}  delta_mag(n) = {mp.nstr(dn, 6)}  brief's value = {known}  "
                  f"reldiff = {mp.nstr(abs(dn-known)/known, 3)}")
