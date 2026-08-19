"""Build Qhat_n(x) as a Chebyshev-T coefficient vector (in x), via the
affine substitution l(x) = A*x + B and the 3-term Chebyshev recurrence
T_{k+1}(l) = 2 l T_k(l) - T_{k-1}(l), carried out entirely in the
Chebyshev-in-x coefficient representation (never the monomial basis).
"""
from __future__ import annotations

import mpmath as mp

from geometry import setup, delta_n


def mult_by_x(v):
    """Chebyshev coeffs of x*p(x), given p's Chebyshev coeffs v (length k+1).
    x*T_0 = T_1;  x*T_m = 0.5*(T_{m+1}+T_{m-1})  for m>=1."""
    k = len(v) - 1
    out = [mp.mpf(0)] * (k + 2)
    if k >= 0:
        out[1] += v[0]  # x*T_0 = T_1
    for m in range(1, k + 1):
        out[m + 1] += v[m] * mp.mpf('0.5')
        out[m - 1] += v[m] * mp.mpf('0.5')
    return out


def add(u, v):
    n = max(len(u), len(v))
    u = u + [mp.mpf(0)] * (n - len(u))
    v = v + [mp.mpf(0)] * (n - len(v))
    return [ui + vi for ui, vi in zip(u, v)]


def scal(c, v):
    return [c * vi for vi in v]


def Tn_of_l_cheb_coeffs(n, A, B):
    """Chebyshev-in-x coefficients of T_n(l(x)), l(x)=A*x+B, via the
    recurrence carried out in Cheb-x coefficient space."""
    T_prev = [mp.mpf(1)]          # T_0(l) = 1
    T_curr = [B, A]               # T_1(l) = l = A*x + B (cheb coeffs [B, A])
    if n == 0:
        return T_prev
    if n == 1:
        return T_curr
    for k in range(1, n):
        # T_{k+1}(l) = 2*l*T_k(l) - T_{k-1}(l) = 2*(A*x+B)*T_k(l) - T_{k-1}(l)
        lx_Tk = add(scal(A, mult_by_x(T_curr)), scal(B, T_curr))
        T_next = add(scal(2, lx_Tk), scal(-1, T_prev))
        T_prev, T_curr = T_curr, T_next
    return T_curr


def qhat_n_cheb_coeffs(n, geo):
    """Chebyshev-in-x coefficients of Qhat_n(x) = 1 + (delta_n/2)*(1 - T_n(l(x))),
    l(x) = (2x-1-a)/(1-a) = A*x+B."""
    a = geo["a"]
    A = 2 / (1 - a)
    B = -(1 + a) / (1 - a)
    dn = delta_n(n, geo)
    Tn_l = Tn_of_l_cheb_coeffs(n, A, B)
    q = scal(-dn / 2, Tn_l)
    q[0] += 1 + dn / 2
    return q, dn


def cheb_eval(coeffs, x):
    """Clenshaw evaluation of sum coeffs[k] T_k(x)."""
    k = len(coeffs) - 1
    b1 = mp.mpf(0)
    b2 = mp.mpf(0)
    for j in range(k, 0, -1):
        b0 = coeffs[j] + 2 * x * b1 - b2
        b2 = b1
        b1 = b0
    return coeffs[0] + x * b1 - b2


if __name__ == "__main__":
    geo = setup(50)
    for n in [1, 3, 5, 7, 9]:
        q, dn = qhat_n_cheb_coeffs(n, geo)
        print(f"n={n}  deg={len(q)-1}  delta_n={mp.nstr(dn,8)}  leading coeff={mp.nstr(q[-1],6)}")
