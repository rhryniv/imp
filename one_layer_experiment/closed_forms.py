"""Claim 1-3 and Task-6 closed forms from the brief's Section 2 --
these are the formulas UNDER TEST, kept in a separate module so Task 1's
cross-check can honestly compare transfer_matrix.py's output against
them without any shared code path.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp


# ---- double precision (Claim 1, tables) ----

def kappa_B_closed(vartheta, s):
    return (1 + s) * np.cos(vartheta) - s


def Q_closed(vartheta, s):
    return 1.0 + 4.0 * s * (1.0 + s) * np.sin(vartheta / 2.0) ** 2


def q2_abs2_closed(vartheta, s):
    return Q_closed(vartheta, s) - 1.0


def s_minus(mu0, u):
    return (np.cosh(mu0) + np.cos(u)) / (1.0 - np.cos(u))


def s_plus(delta, t):
    return (np.sqrt(1.0 + delta / np.sin(t / 2.0) ** 2) - 1.0) / 2.0


def cot2_half(t):
    return (np.cos(t / 2.0) / np.sin(t / 2.0)) ** 2


def delta_min(mu0, t, u):
    sm = s_minus(mu0, u)
    return 4.0 * sm * (1.0 + sm) * np.sin(t / 2.0) ** 2


def delta_mag1(mu0, t, u):
    return np.sinh(mu0) ** 2 * (1.0 - np.cos(t)) / (1.0 - np.cos(u))


# ---- mpmath (high precision, Tasks 1/4/6) ----

def kappa_B_closed_mp(vartheta, s):
    return (1 + s) * mp.cos(vartheta) - s


def Q_closed_mp(vartheta, s):
    return 1 + 4 * s * (1 + s) * mp.sin(vartheta / 2) ** 2


def s_minus_mp(mu0, u):
    return (mp.cosh(mu0) + mp.cos(u)) / (1 - mp.cos(u))


def s_plus_mp(delta, t):
    return (mp.sqrt(1 + delta / mp.sin(t / 2) ** 2) - 1) / 2


def cot2_half_mp(t):
    return (mp.cos(t / 2) / mp.sin(t / 2)) ** 2


def delta_min_mp(mu0, t, u):
    sm = s_minus_mp(mu0, u)
    return 4 * sm * (1 + sm) * mp.sin(t / 2) ** 2


def delta_mag1_mp(mu0, t, u):
    return mp.sinh(mu0) ** 2 * (1 - mp.cos(t)) / (1 - mp.cos(u))
