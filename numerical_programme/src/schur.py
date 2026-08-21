"""Explicit (non-iterative) layer stripping: p_2's n roots are exactly
the equioscillation touch points of Qhat_n=1 within F_1 (closed form
from T_n's own structure -- no separate root-finding needed), then a
Schur-type downward recursion peels alpha_n, alpha_{n-1}, ..., alpha_0
one at a time, each step an exact polynomial division verified to have
zero remainder.
"""
from __future__ import annotations

import mpmath as mp


def touch_varthetas(n, geo):
    """l_k = cos(2 pi k/n), k=0..(n-1)/2 (T_n(l_k)=1); map back to
    vartheta via x=((1-a)l+1+a)/2, vartheta=arccos(x)."""
    a = geo["a"]
    out = []
    for k in range((n - 1) // 2 + 1):
        l = mp.cos(2 * mp.pi * k / n)
        x = ((1 - a) * l + 1 + a) / 2
        out.append(mp.acos(x))
    return out


def poly_mul(u, v):
    out = [mp.mpf(0)] * (len(u) + len(v) - 1)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            out[i + j] += ui * vj
    return out


def p2_raw_coeffs(n, geo):
    """(w-1) * prod_{k=1}^{(n-1)/2} (w^2 - 2cos(vartheta_k) w + 1), low-degree first."""
    varthetas = touch_varthetas(n, geo)
    coeffs = [mp.mpf(-1), mp.mpf(1)]  # (w - 1)
    for vartheta_k in varthetas[1:]:
        quad = [mp.mpf(1), -2 * mp.cos(vartheta_k), mp.mpf(1)]  # 1 - 2cos*w + w^2, low-to-high
        coeffs = poly_mul(coeffs, quad)
    return coeffs, varthetas


def poly_eval(coeffs, w):
    res = mp.mpc(0)
    for c in reversed(coeffs):
        res = res * w + c
    return res


def build_p2(n, geo, dn):
    """Normalize p2_raw so |p2(e^{-i*t})|^2 = dn exactly (t = geo['t'], the
    F_1 boundary where Q(t)-1=dn identically, by construction)."""
    raw, varthetas = p2_raw_coeffs(n, geo)
    t = geo["t"]
    val_raw = poly_eval(raw, mp.e ** (-1j * t))
    K2 = mp.sqrt(dn) / abs(val_raw)
    p2 = [K2 * c for c in raw]
    return p2, varthetas


def downward_peel(p1, p2, dps):
    """Schur-type downward recursion: returns (alphas[0..n], residuals)."""
    n = len(p1) - 1
    cur1, cur2 = list(p1), list(p2)
    alphas_rev = []
    div_residuals = []
    lead_residuals = []
    for j in range(n, 0, -1):
        Aj, Bj = cur1[j], cur2[j]
        ratio = Aj / Bj
        alpha_j = mp.atanh(ratio)
        ca, sa = mp.cosh(alpha_j), mp.sinh(alpha_j)

        new1_full = [ca * cur1[k] - sa * cur2[k] for k in range(j + 1)]
        lead_residuals.append(abs(new1_full[j]))
        new1 = new1_full[:j]

        num_full = [ca * cur2[k] - sa * cur1[k] for k in range(j + 1)]
        div_residuals.append(abs(num_full[0]))
        new2 = num_full[1:]  # divide by w: drop the (should-be-zero) constant term

        alphas_rev.append(alpha_j)
        cur1, cur2 = new1, new2

    # j=0: cur1=[cosh(a0)], cur2=[sinh(a0)]
    a0 = mp.atanh(cur2[0] / cur1[0])
    consistency = abs(cur1[0] ** 2 - cur2[0] ** 2 - 1)
    alphas = [a0] + list(reversed(alphas_rev))
    return alphas, {"lead_residuals": lead_residuals, "div_residuals": div_residuals,
                     "cosh2_minus_sinh2_minus_1": consistency}


if __name__ == "__main__":
    from geometry import setup
    from qhat import qhat_n_cheb_coeffs
    from spectral import cosine_coeffs, spectral_factor
    from layers import forward_recursion

    dps = 60
    geo = setup("G2", dps)
    for n in [1, 3, 5, 7, 9]:
        q, dn = qhat_n_cheb_coeffs(n, geo)
        f = cosine_coeffs(q)
        p1, outside, rho, max_imag = spectral_factor(f, dps)
        p2, varthetas = build_p2(n, geo, dn)

        alphas, diag = downward_peel(p1, p2, dps)
        p1_check, p2_check = forward_recursion(alphas)
        residual = max(abs(p1_check[k] - p1[k]) for k in range(n + 1))
        print(f"n={n}  sum(alphas)={mp.nstr(sum(alphas),4)}  forward-check residual={mp.nstr(residual,4)}  "
              f"max_lead_res={mp.nstr(max(diag['lead_residuals']) if diag['lead_residuals'] else 0,4)}  "
              f"max_div_res={mp.nstr(max(diag['div_residuals']) if diag['div_residuals'] else 0,4)}  "
              f"consistency={mp.nstr(diag['cosh2_minus_sinh2_minus_1'],4)}")
        print(f"   alphas: {[mp.nstr(a,6) for a in alphas]}")
