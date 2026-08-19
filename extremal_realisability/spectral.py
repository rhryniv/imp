"""Step 1-2: cosine-series coefficients and Fejer-Riesz spectral
factorization of Q(vartheta) = Qhat_n(cos vartheta), giving the outer
polynomial p_1(w) = sum_j c_j w^j with |p_1(e^{-i vartheta})|^2 = Q(vartheta),
no zeros in the closed unit disc, c_0 = p_1(0) > 0.
"""
from __future__ import annotations

import mpmath as mp

from qhat import cheb_eval


def cosine_coeffs(q):
    """f_0 = q_0, f_m = q_m/2 for m=1..n (cheb-T coeffs q -> cosine series f)."""
    f = [q[0]] + [qm / 2 for qm in q[1:]]
    return f


def Q_of_vartheta(f, vartheta):
    n = len(f) - 1
    return f[0] + 2 * sum(f[m] * mp.cos(m * vartheta) for m in range(1, n + 1))


def spectral_factor(f, dps):
    """Returns (p1_coeffs, roots_outside, rho) where p1_coeffs = [c_0,...,c_n]
    (LOW degree first), roots_outside = the n roots z_l with |z_l|>1 used to
    build p1, rho = min |z_l| (conditioning indicator)."""
    n = len(f) - 1
    # associated real polynomial R(w) = sum_{k=0}^{2n} f_{|k-n|} w^k
    # polyroots wants HIGH degree first.
    coeffs_low_to_high = [f[abs(k - n)] for k in range(2 * n + 1)]
    coeffs_high_to_low = list(reversed(coeffs_low_to_high))
    roots = mp.polyroots(coeffs_high_to_low, maxsteps=200, extraprec=4 * mp.mp.dps)

    outside = [z for z in roots if abs(z) > 1]
    inside = [z for z in roots if abs(z) <= 1]
    if len(outside) != n:
        raise RuntimeError(f"expected {n} roots with |z|>1, got {len(outside)} "
                            f"(all moduli: {[float(abs(z)) for z in roots]})")

    rho = min(abs(z) for z in outside)

    # p1_raw(w) = prod_l (1 - w/z_l), so p1_raw(0)=1
    def p1_raw_coeffs(zs):
        c = [mp.mpf(1)]
        for z in zs:
            new_c = [mp.mpf(0)] * (len(c) + 1)
            for k, ck in enumerate(c):
                new_c[k] += ck
                new_c[k + 1] += -ck / z
            c = new_c
        return c

    p1_raw = p1_raw_coeffs(outside)
    p1_raw_at_1 = sum(p1_raw)  # sum of coeffs = value at w=1
    K = 1 / p1_raw_at_1  # forces p1(1) = 1 exactly, per the brief
    p1_complex = [K * c for c in p1_raw]

    # p1 should be real (Q has real cosine coefficients); report the residual
    # imaginary part as a conditioning check, then take the real part.
    max_imag = max(abs(mp.im(c)) for c in p1_complex)
    p1 = [mp.re(c) for c in p1_complex]

    return p1, outside, rho, max_imag


def poly_eval(coeffs, w):
    """coeffs low-degree first."""
    res = mp.mpc(0)
    for c in reversed(coeffs):
        res = res * w + c
    return res


if __name__ == "__main__":
    from geometry import setup
    from qhat import qhat_n_cheb_coeffs

    geo = setup(50)
    for n in [1, 3, 5, 7, 9]:
        q, dn = qhat_n_cheb_coeffs(n, geo)
        f = cosine_coeffs(q)
        p1, outside, rho, max_imag = spectral_factor(f, 50)
        c0 = p1[0]
        p1_at_1 = poly_eval(p1, mp.mpc(1))

        # verify |p1(e^{-i vartheta})|^2 = Q(vartheta) on a grid
        max_diff = mp.mpf(0)
        for i in range(0, 201):
            vartheta = mp.pi * i / 200
            w = mp.e ** (-1j * vartheta)
            lhs = abs(poly_eval(p1, w)) ** 2
            rhs = Q_of_vartheta(f, vartheta)
            max_diff = max(max_diff, abs(lhs - rhs))

        print(f"n={n}  rho(min|z_l|)={mp.nstr(rho,10)}  c0={mp.nstr(c0,8)} (>0: {c0>0})  "
              f"p1(1)={mp.nstr(p1_at_1,10)}  max|Q_check_diff|={mp.nstr(max_diff,4)}  "
              f"max_imag_residual={mp.nstr(max_imag,4)}")
