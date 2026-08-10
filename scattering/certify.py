"""Exact certification via the x=cos(theta) / Chebyshev-colleague-matrix
approach (spec Sec. 3.3, "Certification"; manuscript Sec. 6.2). Q and
kappa_B are ALREADY Chebyshev series in x=cos(theta) -- Q's own
coefficients are the doubled autocorrelation (Stage 1's Q_from_autocorr),
kappa_B's are the coefficient vector `a` itself, since T_m(x)=cos(m*theta)
-- so their extrema on any x-subinterval are found directly via numpy's
own Chebyshev root-finding: real roots only, half the degree of a
z-domain companion matrix on z=exp(i*theta), and no on-circle filtering
tolerance is needed (spec Sec. 8's own recommended choice, replacing the
earlier z-domain implementation).

Rule 3 (spec Sec. 5): arccosh is taken ONLY when kappa_min>1; otherwise
mu_min is None. Never clamped, never a fallback to mu_0 or 0.

Rule 6 (spec Sec. 5): s_0 is certified via its OWN degree-2n polynomial
1-kappa_B(x)^2 (not derived from kappa_B's own extremum) -- computed
here as C.chebmul(a,a) subtracted from the constant series [1]. If
s_0==0 the instance touches a band edge and no design is admissible;
REJECTING on that is the caller's responsibility (an algorithm-level,
not a certification-level, decision) -- certify_exact only reports the
exact value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.polynomial import chebyshev as C

from .forward import a_from_alphas, autocorr

Interval = tuple[float, float]


def _theta_interval_to_x(lo: float, hi: float) -> tuple[float, float]:
    """theta in [lo,hi] (0<=lo<hi<=pi) <-> x=cos(theta) in [cos(hi),cos(lo)]
    -- cos is decreasing on [0,pi], so the interval endpoints swap."""
    return float(np.cos(hi)), float(np.cos(lo))


def _extremize_chebyshev(cheb_coeffs: np.ndarray, x_lo: float, x_hi: float,
                          eval_fn, mode: str, root_tol: float = 1e-9) -> float:
    """max/min of eval_fn(x) over [x_lo, x_hi], exactly: eval_fn evaluated
    only at the two endpoints and the real roots of the derivative of the
    Chebyshev series `cheb_coeffs`, strictly inside the open interval.
    Roots come from numpy's own Chebyshev.deriv().roots() (colleague-matrix
    eigenvalues); only near-real roots are kept (spurious complex roots of
    the colleague matrix are discarded, not unit-circle artifacts as in
    the z-domain version -- there are none here, since x is already real).
    """
    deriv = C.Chebyshev(cheb_coeffs).deriv()
    if not np.any(np.abs(deriv.coef) > 0):
        roots = np.array([])
    else:
        r = deriv.roots()
        real = r[np.abs(r.imag) < root_tol].real
        roots = real[(real > x_lo + 1e-12) & (real < x_hi - 1e-12)]
    candidates = np.concatenate([roots, [x_lo, x_hi]])
    vals = eval_fn(candidates)
    return float(np.max(vals) if mode == "max" else np.min(vals))


def _q_cheb_coeffs(f: np.ndarray) -> np.ndarray:
    """Q(x) = f_0 + 2*sum_{m>=1} f_m T_m(x) -- Q's own Chebyshev-series
    coefficients, i.e. f doubled on every harmonic beyond the constant."""
    if len(f) == 1:
        return f.copy()
    return np.concatenate([[f[0]], 2.0 * f[1:]])


@dataclass
class CertifyResult:
    delta: float                        # exact max_{I1}(Q-1) (spec Sec. 1)
    kappa_min: float | None             # exact min_i min_{[u_i,v_i]} sigma_i*kappa_B; None if J0 empty
    mu_min: float | None                # arccosh(kappa_min) ONLY if kappa_min>1 (rule 3); else None, never clamped
    s_0: float | None                   # exact min_{I1}(1-kappa_B^2), its own degree-2n certification; None if J1 empty
    TN_stop_bound: dict = field(default_factory=dict)  # {N: upper bound on max_{I0} T_N}, Prop. asymmetry(a)
    TN_pass_bound: dict = field(default_factory=dict)  # {N: lower bound on min_{I1} T_N}, Prop. asymmetry(b)
    accept: dict = field(default_factory=dict)          # {N: bool}, only if eps0/eps1 given


def certify_exact(alphas: np.ndarray, J0: Sequence[Interval], J1: Sequence[Interval],
                   sigma: Sequence[int], N_values: Sequence[int] = (),
                   eps0: float | None = None, eps1: float | None = None) -> CertifyResult:
    """Exact certification (spec Sec. 3.3 / manuscript Sec. 6.2
    "Certification", Step 2 of the algorithm): report delta, kappa_min,
    mu_min and s_0 to machine precision, independently of whatever grid
    Phase 2's search used, then apply Proposition asymmetry's closed-form
    T_N bounds -- without ever evaluating T_N on a grid.
    """
    a = a_from_alphas(np.asarray(alphas, dtype=float))
    f = autocorr(a)
    Q_cheb = _q_cheb_coeffs(f)

    delta = 0.0
    if J1:
        vals = []
        for lo, hi in J1:
            x_lo, x_hi = _theta_interval_to_x(lo, hi)
            v = _extremize_chebyshev(Q_cheb, x_lo, x_hi,
                                      lambda x: C.chebval(x, Q_cheb) - 1.0, "max")
            vals.append(v)
        delta = max(vals)

    kappa_min = None
    if J0:
        vals = []
        for (lo, hi), s in zip(J0, sigma):
            x_lo, x_hi = _theta_interval_to_x(lo, hi)
            v = _extremize_chebyshev(a, x_lo, x_hi,
                                      lambda x, s=s: s * C.chebval(x, a), "min")
            vals.append(v)
        kappa_min = min(vals)
    mu_min = float(np.arccosh(kappa_min)) if (kappa_min is not None and kappa_min > 1.0) else None

    s_0 = None
    if J1:
        kappa_sq_cheb = C.chebmul(a, a)               # kappa_B(x)^2, degree 2n, its OWN series (rule 6)
        s0_cheb = -kappa_sq_cheb
        s0_cheb[0] += 1.0                              # 1 - kappa_B(x)^2
        vals = []
        for lo, hi in J1:
            x_lo, x_hi = _theta_interval_to_x(lo, hi)
            v = _extremize_chebyshev(s0_cheb, x_lo, x_hi,
                                      lambda x: C.chebval(x, s0_cheb), "min")
            vals.append(v)
        s_0 = min(vals)

    TN_stop_bound = {N: 1.0 / np.cosh(N * mu_min) ** 2 for N in N_values} if (J0 and mu_min is not None) else {}
    TN_pass_bound = ({N: 1.0 / (1.0 + delta / s_0) for N in N_values}
                      if (J1 and s_0 is not None and s_0 > 0) else {})

    accept = {}
    if eps0 is not None and eps1 is not None:
        for N in N_values:
            ok_stop = mu_min is not None and TN_stop_bound.get(N, 1.0) <= eps0
            ok_pass = s_0 is not None and s_0 > 0 and TN_pass_bound.get(N, 0.0) >= 1.0 - eps1
            accept[N] = bool(ok_stop and ok_pass)

    return CertifyResult(delta=delta, kappa_min=kappa_min, mu_min=mu_min, s_0=s_0,
                          TN_stop_bound=TN_stop_bound, TN_pass_bound=TN_pass_bound, accept=accept)


def grid_vs_exact(grid_delta: float | None, grid_kappa_min: float | None,
                   exact: CertifyResult) -> dict:
    """Direct emission source for `grid_vs_exact` (spec Sec. 6): how far a
    grid-based search's own reported delta/kappa_min are from the exact
    certified values -- a diagnostic on the search, not on certify_exact
    itself (which never touches a grid)."""
    return {
        "delta_discrepancy": None if grid_delta is None else abs(grid_delta - exact.delta),
        "kappa_min_discrepancy": (None if grid_kappa_min is None or exact.kappa_min is None
                                   else abs(grid_kappa_min - exact.kappa_min)),
    }
