"""Geometry and specification constants for Specs A, B, C. I_0 is a LIST
of (start,end) fractions-of-pi tuples, supporting Spec C's disconnected
stop set; I_1 is always a single interval [0, t].
"""
from __future__ import annotations

import mpmath as mp

SPECS = {
    "A": {"t_frac": (1, 4), "I0_fracs": [((5, 6), (1, 1))],
          "mu0": 1, "N_max": 8, "eps_1": 0.1},
    "B": {"t_frac": (1, 4), "I0_fracs": [((3, 4), (1, 1))],
          "mu0": 1, "N_max": 8, "eps_1": 0.1},
    "C": {"t_frac": (1, 4), "I0_fracs": [((1, 2), (2, 3)), ((5, 6), (1, 1))],
          "mu0": 1, "N_max": 8, "eps_1": 0.1},
}


def _frac_to_mp(frac):
    return mp.mpf(frac[0]) / frac[1] * mp.pi


def setup(name, dps=50):
    mp.mp.dps = dps
    spec = SPECS[name]
    t = _frac_to_mp(spec["t_frac"])
    a = mp.cos(t)
    I0 = [(_frac_to_mp(lo), _frac_to_mp(hi)) for lo, hi in spec["I0_fracs"]]
    # "b" and "gamma" (via cosh^2(gamma/2)=(1-b)/(1-a)) are only defined for
    # a single-component I_0 (Specs A, B); Spec C uses the E5 Green's-function
    # gamma instead (see green.py).
    b = mp.cos(I0[0][0]) if len(I0) == 1 else None  # cos is decreasing: I0's LOWER theta-bound -> F_0's UPPER x-bound
    gamma = 2 * mp.acosh(mp.sqrt((1 - b) / (1 - a))) if b is not None else None

    mu0 = mp.mpf(spec["mu0"])
    S = mp.sinh(mu0) ** 2
    N_max = spec["N_max"]
    eps_0 = mp.mpf(4) * mp.e ** (-2 * N_max)  # eps_0 = 4*exp(-2*N_max) = 4*exp(-16) for N_max=8
    eps_1 = mp.mpf(spec["eps_1"])
    delta_target = eps_1 / (N_max ** 2 * (1 - eps_1))

    return {"name": name, "t": t, "a": a, "I0": I0, "b": b, "gamma": gamma,
            "mu0": mu0, "S": S, "N_max": N_max, "eps_0": eps_0, "eps_1": eps_1,
            "delta_target": delta_target}


def delta_n_closed(n, geo):
    """S/cosh^2(n*gamma/2) -- only meaningful when geo['gamma'] is defined
    (Specs A, B; single-component I_0)."""
    return geo["S"] / mp.cosh(n * geo["gamma"] / 2) ** 2


def cor_resource_bound(geo):
    """n >= log((1-eps1)*log^2(4/eps0)/(4*eps1)) / gamma"""
    eps0, eps1, gamma = geo["eps_0"], geo["eps_1"], geo["gamma"]
    val = mp.log((1 - eps1) * mp.log(4 / eps0) ** 2 / (4 * eps1)) / gamma
    return val


if __name__ == "__main__":
    for name in ["A", "B", "C"]:
        geo = setup(name, 50)
        print(f"=== Spec {name} ===")
        print(f"  t={mp.nstr(geo['t'],10)}  a={mp.nstr(geo['a'],10)}")
        print(f"  I0={[(mp.nstr(lo,8), mp.nstr(hi,8)) for lo,hi in geo['I0']]}")
        if geo["b"] is not None:
            print(f"  b={mp.nstr(geo['b'],10)}  gamma={mp.nstr(geo['gamma'],10)}")
        print(f"  mu0={geo['mu0']}  S={mp.nstr(geo['S'],10)}  eps_0={mp.nstr(geo['eps_0'],10)}  "
              f"eps_1={geo['eps_1']}  delta_target={mp.nstr(geo['delta_target'],10)}")
        if geo["gamma"] is not None:
            print(f"  cor:resource bound n >= {mp.nstr(cor_resource_bound(geo),10)}")
            for n in range(1, 8):
                print(f"    n={n}  delta_n_closed = {mp.nstr(delta_n_closed(n, geo), 8)}")
