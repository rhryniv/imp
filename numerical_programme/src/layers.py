"""Step 3: recover contrasts alpha_0..alpha_n from p_1 by forward-simulating
the brief's recursion (Section 3) as a function of alpha_1..alpha_n (with
alpha_0 = -sum(alpha_1..n), enforcing the matching condition and, as a
built-in identity of the recursion, p_1(1)=1 automatically -- verified
below), and solving the resulting square nonlinear system with mpmath.
"""
from __future__ import annotations

import mpmath as mp


def poly_add(u, v):
    n = max(len(u), len(v))
    u = list(u) + [mp.mpf(0)] * (n - len(u))
    v = list(v) + [mp.mpf(0)] * (n - len(v))
    return [a + b for a, b in zip(u, v)]


def poly_scal(c, v):
    return [c * vi for vi in v]


def poly_mul_by_w(v):
    return [mp.mpf(0)] + list(v)


def forward_recursion(alphas):
    """alphas = [alpha_0,...,alpha_n]. Returns (p1_coeffs, p2_coeffs), low
    degree first, length n+1 each."""
    a0 = alphas[0]
    p1 = [mp.cosh(a0)]
    p2 = [mp.sinh(a0)]
    for aj in alphas[1:]:
        ca, sa = mp.cosh(aj), mp.sinh(aj)
        new_p1 = poly_add(poly_scal(ca, p1), poly_scal(sa, poly_mul_by_w(p2)))
        new_p2 = poly_add(poly_scal(sa, p1), poly_scal(ca, poly_mul_by_w(p2)))
        p1, p2 = new_p1, new_p2
    return p1, p2


def solve_alphas(target_p1, seed_alphas, dps, drop_index=0, tol_digits=None):
    """target_p1: length n+1, low degree first. seed_alphas: initial guess
    for alpha_1..alpha_n (length n). Returns (alphas0..n, residual info)."""
    n = len(target_p1) - 1
    if tol_digits is None:
        tol_digits = dps - 12

    def F(*rest):
        a_rest = list(rest)
        a0 = -sum(a_rest)
        alphas = [a0] + a_rest
        p1, p2 = forward_recursion(alphas)
        eqs = [p1[k] - target_p1[k] for k in range(n + 1) if k != drop_index]
        return tuple(eqs)

    sol = mp.findroot(F, [mp.mpf(str(s)) for s in seed_alphas],
                       tol=mp.mpf(10) ** (-tol_digits), solver="mnewton", maxsteps=200)
    a_rest = [sol[i] for i in range(n)]
    a0 = -sum(a_rest)
    alphas = [a0] + a_rest

    p1_check, p2_check = forward_recursion(alphas)
    residual = max(abs(p1_check[k] - target_p1[k]) for k in range(n + 1))
    return alphas, residual, p2_check


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from geometry import setup
    from qhat import qhat_n_cheb_coeffs
    from spectral import cosine_coeffs, spectral_factor

    geo = setup("G2", 50)
    for n in [1, 3]:
        q, dn = qhat_n_cheb_coeffs(n, geo)
        f = cosine_coeffs(q)
        p1, outside, rho, max_imag = spectral_factor(f, 50)

        seed = [mp.mpf('0.1')] * n
        alphas, residual, p2 = solve_alphas(p1, seed, dps=50)
        print(f"n={n}  alphas={[float(a) for a in alphas]}  sum={float(sum(alphas)):.2e}  "
              f"residual={mp.nstr(residual,4)}")
