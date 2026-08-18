"""Parametrized (t, u, mu0) Variant I / Variant II magnitude-relaxation
builders, with an explicit rescaling of the delta objective by a known
scale factor so that the SDP's own optimization variable stays O(1) even
when the true delta is far below double-precision resolution (e.g.
delta ~ 1e-13 at n=11). See run_taskA.py / run_taskB.py for usage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp

from cheb_mk import markov_lukacs_cheb, cheb_poly_mul_fixed, dual_matrices_cheb

X_A, X_B = -1.0, 1.0


@dataclass(frozen=True)
class Geometry:
    t: float
    u: float
    mu0: float
    # exact (numerator, denominator) so that t = t_frac[0]/t_frac[1] * pi,
    # u = u_frac[0]/u_frac[1] * pi -- lets mpmath-precision code recompute
    # cos(t), cos(u) etc. from mpmath's own arbitrary-precision pi instead
    # of round-tripping through a float64 value (which caps accuracy at
    # ~1e-16 regardless of the mpmath working precision requested).
    t_frac: tuple = (1, 4)
    u_frac: tuple = (3, 4)

    def mp_t(self, dps=50):
        import mpmath as mp
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.mpf(self.t_frac[0]) / self.t_frac[1] * mp.pi
        finally:
            mp.mp.dps = old

    def mp_u(self, dps=50):
        import mpmath as mp
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.mpf(self.u_frac[0]) / self.u_frac[1] * mp.pi
        finally:
            mp.mp.dps = old

    def mp_X1(self, dps=50):
        import mpmath as mp
        return (mp.cos(self.mp_t(dps)), mp.mpf(1))

    def mp_X0(self, dps=50):
        import mpmath as mp
        return (mp.mpf(-1), mp.cos(self.mp_u(dps)))

    def mp_sinh2_mu0(self, dps=50):
        import mpmath as mp
        old = mp.mp.dps
        mp.mp.dps = dps
        try:
            return mp.sinh(mp.mpf(self.mu0)) ** 2
        finally:
            mp.mp.dps = old

    def mp_cosh2_mu0(self, dps=50):
        import mpmath as mp
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
    def cosh2_mu0(self) -> float:
        return float(np.cosh(self.mu0) ** 2)

    @property
    def gamma(self) -> float:
        num = 2 * np.cos(self.u) - np.cos(self.t) - 1
        den = 1 - np.cos(self.t)
        return float(np.arccosh(abs(num / den)))

    @property
    def beta1(self) -> float:
        """sinh^2(mu0), the analytic prefactor: beta_n = beta1 * exp(-gamma*n)."""
        return float(np.sinh(self.mu0) ** 2)

    def beta_n(self, n: int) -> float:
        return self.beta1 * np.exp(-self.gamma * n)

    def delta1_closed_form(self) -> float:
        return self.beta1 * (1 - np.cos(self.t)) / (1 - np.cos(self.u))


@dataclass
class BuiltProblem:
    problem: cp.Problem
    delta_scaled: cp.Variable
    scale: float
    equalities: list
    gram_vars: list
    n: int
    variant: str
    geo: Geometry

    @property
    def delta_value(self):
        return None if self.delta_scaled.value is None else self.scale * float(self.delta_scaled.value)


def _const_part(expr: cp.Expression, variables: list) -> np.ndarray:
    saved = [v.value for v in variables]
    for v in variables:
        v.value = np.zeros(v.shape) if v.shape else 0.0
    const = np.array(expr.value, dtype=float).reshape(-1)
    for v, s in zip(variables, saved):
        v.value = s
    return const


def build_variant_I(n: int, geo: Geometry, scale: float = 1.0) -> BuiltProblem:
    X1_A, X1_B = geo.X1
    X0_A, X0_B = geo.X0
    coshmu0_sq = geo.cosh2_mu0

    c = cp.Variable(n + 1)
    delta_scaled = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    equalities = []
    gram_vars = []
    constraints = [delta_scaled >= 0.0]

    consA, gvA, rawA = markov_lukacs_cheb(c - e0, n, X_A, X_B)
    constraints += consA
    gram_vars += gvA

    consB, gvB, rawB = markov_lukacs_cheb((1.0 + scale * delta_scaled) * e0 - c, n, X1_A, X1_B)
    constraints += consB
    gram_vars += gvB

    consC, gvC, rawC = markov_lukacs_cheb(c - coshmu0_sq * e0, n, X0_A, X0_B)
    constraints += consC
    gram_vars += gvC

    rawD = cp.sum(c) - 1.0
    eqD = (rawD == 0)
    constraints.append(eqD)

    all_vars = [c, delta_scaled] + gram_vars
    equalities.append(("A", -_const_part(rawA, all_vars), rawA, consA[-1]))
    equalities.append(("B", -_const_part(rawB, all_vars), rawB, consB[-1]))
    equalities.append(("C", -_const_part(rawC, all_vars), rawC, consC[-1]))
    equalities.append(("D", -_const_part(rawD, all_vars), rawD, eqD))

    problem = cp.Problem(cp.Minimize(delta_scaled), constraints)
    return BuiltProblem(problem=problem, delta_scaled=delta_scaled, scale=scale,
                         equalities=equalities, gram_vars=gram_vars, n=n, variant="I", geo=geo)


def build_variant_II(n: int, geo: Geometry, scale: float = 1.0) -> BuiltProblem:
    X1_A, X1_B = geo.X1
    X0_A, X0_B = geo.X0
    coshmu0_sq = geo.cosh2_mu0

    r = cp.Variable(n)  # R's Chebyshev coeffs, degree n-1
    delta_scaled = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    one_minus_x = np.array([1.0, -1.0])

    equalities = []
    gram_vars = []
    constraints = [delta_scaled >= 0.0]

    consA, gvA, rawA = markov_lukacs_cheb(r, n - 1, X_A, X_B)
    constraints += consA
    gram_vars += gvA

    Qhat_var_part = cheb_poly_mul_fixed(one_minus_x, r, n - 1)

    consB, gvB, rawB = markov_lukacs_cheb((1.0 + scale * delta_scaled) * e0 - e0 - Qhat_var_part, n, X1_A, X1_B)
    constraints += consB
    gram_vars += gvB

    consC, gvC, rawC = markov_lukacs_cheb(e0 + Qhat_var_part - coshmu0_sq * e0, n, X0_A, X0_B)
    constraints += consC
    gram_vars += gvC

    all_vars = [r, delta_scaled] + gram_vars
    equalities.append(("A", -_const_part(rawA, all_vars), rawA, consA[-1]))
    equalities.append(("B", -_const_part(rawB, all_vars), rawB, consB[-1]))
    equalities.append(("C", -_const_part(rawC, all_vars), rawC, consC[-1]))

    problem = cp.Problem(cp.Minimize(delta_scaled), constraints)
    return BuiltProblem(problem=problem, delta_scaled=delta_scaled, scale=scale,
                         equalities=equalities, gram_vars=gram_vars, n=n, variant="II", geo=geo)


def qhat_coeffs(built: BuiltProblem) -> np.ndarray | None:
    """Reconstruct Qhat's Chebyshev coeffs (length n+1) from the solved problem."""
    if built.problem.status not in ("optimal", "optimal_inaccurate"):
        return None
    if built.variant == "I":
        c_var = [v for v in built.problem.variables() if v.shape == (built.n + 1,)][0]
        return np.array(c_var.value, dtype=float)
    r_var = [v for v in built.problem.variables() if v.shape == (built.n,)][0]
    r = np.array(r_var.value, dtype=float)
    one_minus_x = np.array([1.0, -1.0])
    var_part = cheb_poly_mul_fixed(one_minus_x, cp.Constant(r), built.n - 1)
    c = np.zeros(built.n + 1)
    c[0] = 1.0
    return c + np.array(var_part.value, dtype=float)


def block_domain(name: str, deg_A: int, n: int, geo: Geometry):
    if name == "A":
        return deg_A, X_A, X_B
    if name == "B":
        return (n,) + geo.X1
    if name == "C":
        return (n,) + geo.X0
    return None


def extract_dual(built: BuiltProblem, deg_A: int) -> dict:
    """underline_delta (real units, after undoing the `scale` rescaling)
    plus a PSD-verified flag per equality block."""
    total = 0.0
    all_verified = True
    per_block = {}
    for name, b_vec, raw_expr, eq in built.equalities:
        dv = eq.dual_value
        if dv is None:
            return {"underline_delta": None, "verified": False, "per_block": {}}
        y = -np.asarray(dv, dtype=float).flatten()
        contrib = float(b_vec @ y)
        total += contrib
        dom = block_domain(name, deg_A, built.n, built.geo)
        if dom is not None:
            deg, a, b = dom
            mats = dual_matrices_cheb(y, deg, a, b)
            min_eig = min(float(np.linalg.eigvalsh(M).min()) for M in mats)
            verified = min_eig >= -1e-6
            all_verified = all_verified and verified
            per_block[name] = {"contrib": contrib, "min_eig": min_eig, "verified": verified}
        else:
            per_block[name] = {"contrib": contrib, "min_eig": None, "verified": True}
    underline_tilde = total
    underline_delta = built.scale * underline_tilde
    return {"underline_delta": underline_delta, "underline_tilde": underline_tilde,
            "verified": all_verified, "per_block": per_block}


def cheb_eval(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(c) - 1
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    for k in range(n, 0, -1):
        b0 = c[k] + 2 * x * b1 - b2
        b2 = b1
        b1 = b0
    return c[0] + x * b1 - b2


def grid_feasibility(c: np.ndarray, delta: float, geo: Geometry, grid_n: int = 4000, tol: float = 1e-6) -> dict:
    X1_A, X1_B = geo.X1
    X0_A, X0_B = geo.X0
    xA = np.linspace(-1.0, 1.0, grid_n)
    x1 = np.linspace(X1_A, X1_B, grid_n)
    x0 = np.linspace(X0_A, X0_B, grid_n)
    qA = cheb_eval(c, xA)
    q1 = cheb_eval(c, x1)
    q0 = cheb_eval(c, x0)
    q_at_1 = cheb_eval(c, np.array([1.0]))[0]
    viol_A = float(np.min(qA - 1.0))
    viol_B = float(np.min((1.0 + delta) - q1))
    viol_C = float(np.min(q0 - geo.cosh2_mu0))
    viol_D = float(abs(q_at_1 - 1.0))
    ok = (viol_A >= -tol) and (viol_B >= -tol) and (viol_C >= -tol) and (viol_D <= 1e-4)
    return {"min_slack_A": viol_A, "min_slack_B": viol_B, "min_slack_C": viol_C,
            "abs_err_D": viol_D, "feasible": ok}
