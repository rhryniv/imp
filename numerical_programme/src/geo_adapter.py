"""Adapter exposing this brief's geometry.py dicts (Specs A/B, single
I_0 component only) through the interface hp_refine.py expects (ported
from the earlier rate-attainment work's variants.Geometry).
"""
from __future__ import annotations

import numpy as np
import mpmath as mp

from geometry import SPECS, _frac_to_mp


class GeoAdapter:
    def __init__(self, name):
        spec = SPECS[name]
        if len(spec["I0_fracs"]) != 1:
            raise ValueError(f"GeoAdapter only supports single-component I_0 (spec {name} has "
                              f"{len(spec['I0_fracs'])})")
        self.name = name
        self.t_frac = spec["t_frac"]
        self.u_frac = spec["I0_fracs"][0][0]  # the LOWER theta-bound of I_0 (maps to x=b)
        self.mu0 = spec["mu0"]
        self.t = float(self.t_frac[0]) / self.t_frac[1] * np.pi
        self.u = float(self.u_frac[0]) / self.u_frac[1] * np.pi

    def mp_t(self, dps=50):
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.mpf(self.t_frac[0]) / self.t_frac[1] * mp.pi
        finally:
            mp.mp.dps = old

    def mp_u(self, dps=50):
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.mpf(self.u_frac[0]) / self.u_frac[1] * mp.pi
        finally:
            mp.mp.dps = old

    def mp_X1(self, dps=50):
        return (mp.cos(self.mp_t(dps)), mp.mpf(1))

    def mp_X0(self, dps=50):
        return (mp.mpf(-1), mp.cos(self.mp_u(dps)))

    def mp_sinh2_mu0(self, dps=50):
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.sinh(mp.mpf(self.mu0)) ** 2
        finally:
            mp.mp.dps = old

    def mp_cosh2_mu0(self, dps=50):
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.cosh(mp.mpf(self.mu0)) ** 2
        finally:
            mp.mp.dps = old

    @property
    def X1(self):
        return (float(np.cos(self.t)), 1.0)

    @property
    def X0(self):
        return (-1.0, float(np.cos(self.u)))

    @property
    def cosh2_mu0(self):
        return float(np.cosh(self.mu0) ** 2)

    @property
    def gamma(self):
        num = 2 * np.cos(self.u) - np.cos(self.t) - 1
        den = 1 - np.cos(self.t)
        return float(np.arccosh(abs(num / den)))

    @property
    def beta1(self):
        return float(np.sinh(self.mu0) ** 2)

    def beta_n(self, n):
        return self.beta1 * np.exp(-self.gamma * n)
