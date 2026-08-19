"""Vectorized (numpy, elementwise over an array of k / vartheta)
re-expression of the SAME n=1 transfer-matrix product as
transfer_matrix.py -- still built directly from Section 1's matrix
definitions (M(alpha,kx)=E(-kx)C(alpha)E(kx), block product, kappa_B via
trace), just unrolled algebraically for speed on large grids (needed for
Task 4's 3-parameter sweep). Cross-checked against the (slow, honest
per-point matrix-multiply) transfer_matrix.py in run_task_vec_check.py
before being trusted anywhere.
"""
from __future__ import annotations

import numpy as np


def block_M11_M21(alpha, k, h=1.0):
    """n=1 block, alpha_0=alpha, alpha_1=-alpha, x_0=0, x_1=h.
    M_B = M(alpha_1, k h) @ M(alpha_0, 0) = M(-alpha, k h) @ C(alpha)
    (since M(alpha_0, k*0)=E(0)C(alpha)E(0)=C(alpha) exactly).
    M(-alpha, kh) = E(-kh) C(-alpha) E(kh).
    Returns (q1, q2) = (M_B[0,0], M_B[1,0]), elementwise for array k.
    """
    ca, sa = np.cosh(alpha), np.sinh(alpha)
    kh = k * h
    e_p = np.exp(1j * kh)
    e_m = np.exp(-1j * kh)
    # M(-alpha, kh) = [[e^{-ikh}, 0],[0, e^{ikh}]] @ [[ca,-sa],[-sa,ca]] @ [[e^{ikh},0],[0,e^{-ikh}]]
    #              = [[ca, -sa*e^{-2ikh}], [-sa*e^{2ikh}, ca]]
    Ma11 = ca * np.ones_like(k, dtype=complex)
    Ma12 = -sa * e_m ** 2
    Ma21 = -sa * e_p ** 2
    Ma22 = ca * np.ones_like(k, dtype=complex)
    # C(alpha) = [[ca, sa],[sa, ca]]
    # M_B = Ma @ C(alpha)
    q1 = Ma11 * ca + Ma12 * sa
    q2 = Ma21 * ca + Ma22 * sa
    return q1, q2


def kappa_B_vec(alpha, k, L, h=1.0):
    q1, q2 = block_M11_M21(alpha, k, h)
    kap = np.real(np.exp(1j * k * L * h) * q1)
    return kap, q1, q2
