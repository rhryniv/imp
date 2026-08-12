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

MANUSCRIPT REVISION (numerical-optimisation instructions, see
scattering/driver.py's own module docstring for the full account): the
old pass-band admissibility test `delta <= s_0*eps_1/(1-eps_1)` degenerates
whenever kappa_B touches +-1 inside I_1, which good designs routinely do
-- confirmed directly on this project's own Instance 1/2 scans (s_0
negative at essentially every degree tried, rejecting designs whose
actual T_N was excellent). It is REPLACED as a gate by constraint (E),
-1<=kappa_B<=1 on I_1 (enforced in direct.py's Phase 2, certified here as
max_kappa_B), together with the N-uniform bound T_N>=(1+N^2|q_2|^2)^-1
(Prop. asymmetry(c)), which needs no s_0 at all. s_0 (rule 6's own
degree-2n polynomial 1-kappa_B(x)^2, computed exactly, unchanged) and the
new quantity Lambda=max_{I1}(Q-1)/(1-kappa_B^2) are KEPT as reported
diagnostics only, for comparison against Prop. asymmetry(b)'s envelope --
never as a gate or a hypothesis anywhere downstream.
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


def _bisect_lambda(a: np.ndarray, Q_cheb: np.ndarray, J1: Sequence[Interval],
                    rho_hi_init: float = 1e8, tol: float = 1e-9,
                    max_iter: int = 200) -> float | None:
    """Lambda = max_{I1} (Q-1)/(1-kappa_B^2), found by bisection on the
    smallest rho such that rho*(1-kappa_B(x)^2) - (Q(x)-1) >= 0 for all x
    in I1 (spec Sec. 3.3 / manuscript Task 9c) -- a polynomial
    NONnegativity check via the same exact extremization used throughout
    this module, never an indeterminate 0/0 evaluation of the ratio
    itself at a point where kappa_B touches +-1 (a closed gap, q_2 also
    vanishing there, has a finite limiting ratio that direct division
    cannot see).

    Well-posed (min_x g(rho,x) monotone nondecreasing in rho, since
    g(rho,x)=rho*(1-kappa_B(x)^2)-(Q(x)-1) is affine in rho with slope
    1-kappa_B(x)^2) precisely when constraint (E), -1<=kappa_B<=1, holds
    on I1 -- i.e. for the designs this quantity is meant to be reported
    on. Returns None (Lambda = +infinity) if no finite rho up to
    rho_hi_init satisfies the inequality: kappa_B touches +-1 somewhere
    in I1 while q_2 does not vanish there (rem:edge's "open" case, no
    N-independent bound holds).
    """
    kappa_sq_cheb = C.chebmul(a, a)
    s0_cheb = -kappa_sq_cheb
    s0_cheb[0] += 1.0                                  # 1 - kappa_B(x)^2, degree 2n
    Qm1_cheb = np.zeros_like(s0_cheb)
    Qm1_cheb[: len(Q_cheb)] = Q_cheb
    Qm1_cheb[0] -= 1.0                                 # Q(x) - 1, padded to degree 2n

    def min_g(rho: float) -> float:
        g_cheb = rho * s0_cheb - Qm1_cheb
        vals = []
        for lo, hi in J1:
            x_lo, x_hi = _theta_interval_to_x(lo, hi)
            vals.append(_extremize_chebyshev(g_cheb, x_lo, x_hi,
                                              lambda x: C.chebval(x, g_cheb), "min"))
        return min(vals)

    if min_g(rho_hi_init) < -tol:
        return None

    lo, hi = 0.0, rho_hi_init
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if min_g(mid) >= -tol:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return hi


@dataclass
class CertifyResult:
    delta: float                        # exact max_{I1}(Q-1) (spec Sec. 1)
    kappa_min: float | None             # exact min_i min_{[u_i,v_i]} sigma_i*kappa_B; None if J0 empty
    mu_min: float | None                # arccosh(kappa_min) ONLY if kappa_min>1 (rule 3); else None, never clamped
    max_kappa_B: float | None           # exact max_{I1}|kappa_B|, certifies constraint (E); None if J1 empty
    s_0: float | None                   # exact min_{I1}(1-kappa_B^2); DIAGNOSTIC ONLY (never a gate); None if J1 empty
    Lambda: float | None                # exact max_{I1}(Q-1)/(1-kappa_B^2), via bisection; None means +infinity
    TN_stop_bound: dict = field(default_factory=dict)  # {N: upper bound on max_{I0} T_N}, Prop. asymmetry(a)
    TN_pass_bound: dict = field(default_factory=dict)  # {N: lower bound on min_{I1} T_N}, Prop. asymmetry(b) envelope
    accept: dict = field(default_factory=dict)          # {N: bool}, only if eps0/eps1 given -- uses Prop. asymmetry(c)


def certify_exact(alphas: np.ndarray, J0: Sequence[Interval], J1: Sequence[Interval],
                   sigma: Sequence[int], N_values: Sequence[int] = (),
                   eps0: float | None = None, eps1: float | None = None) -> CertifyResult:
    """Exact certification (spec Sec. 3.3 / manuscript Sec. 6.2
    "Certification", Step 2 of the algorithm): report delta, kappa_min,
    mu_min, max_kappa_B (certifies constraint (E)) to machine precision,
    independently of whatever grid Phase 2's search used; s_0 and Lambda
    are reported alongside as diagnostics only (see module docstring),
    never used to accept or reject anything here. `accept[N]` uses
    Prop. asymmetry(c)'s N-uniform bound, the one the algorithm actually
    relies on -- not the old s_0-based envelope of part (b).
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

    max_kappa_B = None
    s_0 = None
    Lambda = None
    if J1:
        vals_max = []
        for lo, hi in J1:
            x_lo, x_hi = _theta_interval_to_x(lo, hi)
            v_hi = _extremize_chebyshev(a, x_lo, x_hi, lambda x: C.chebval(x, a), "max")
            v_lo = _extremize_chebyshev(a, x_lo, x_hi, lambda x: C.chebval(x, a), "min")
            vals_max.append(max(abs(v_hi), abs(v_lo)))
        max_kappa_B = max(vals_max)

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

        Lambda = _bisect_lambda(a, Q_cheb, J1)

    TN_stop_bound = {N: 1.0 / np.cosh(N * mu_min) ** 2 for N in N_values} if (J0 and mu_min is not None) else {}
    TN_pass_bound = ({N: 1.0 / (1.0 + delta / s_0) for N in N_values}
                      if (J1 and s_0 is not None and s_0 > 0) else {})

    accept = {}
    if eps0 is not None and eps1 is not None:
        e_ok = max_kappa_B is not None and max_kappa_B <= 1.0 + 1e-9
        for N in N_values:
            ok_stop = mu_min is not None and TN_stop_bound.get(N, 1.0) <= eps0
            # Prop. asymmetry(c): T_N >= (1+N^2*delta)^-1, needs (E) to hold.
            ok_pass = e_ok and J1 and (1.0 / (1.0 + N ** 2 * delta)) >= 1.0 - eps1
            accept[N] = bool(ok_stop and ok_pass)

    return CertifyResult(delta=delta, kappa_min=kappa_min, mu_min=mu_min, max_kappa_B=max_kappa_B,
                          s_0=s_0, Lambda=Lambda,
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
