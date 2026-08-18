"""Variant I (as posed, with explicit equality (D')) and Variant II
(equality eliminated via Qhat = 1 + (1-x)*R(x)) of the magnitude
relaxation, built directly from cheb_mk's Chebyshev-basis Markov-Lukacs
encoding. Both expose their equality constraints' raw (unwrapped)
expressions so the explicit dual bound can be extracted without cvxpy
introspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cvxpy as cp

from cheb_mk import markov_lukacs_cheb, cheb_poly_mul_fixed, dual_matrices_cheb

T, U = np.pi / 4, 3 * np.pi / 4
X_A, X_B = -1.0, 1.0
X1_A, X1_B = np.cos(T), 1.0
X0_A, X0_B = -1.0, np.cos(U)


@dataclass
class BuiltProblem:
    problem: cp.Problem
    delta: cp.Variable
    equalities: list  # (name, b_vec, raw_expr, eq_constraint)
    gram_vars: list


def _const_part(expr: cp.Expression, variables: list) -> np.ndarray:
    saved = [v.value for v in variables]
    for v in variables:
        v.value = np.zeros(v.shape) if v.shape else 0.0
    const = np.array(expr.value, dtype=float).reshape(-1)
    for v, s in zip(variables, saved):
        v.value = s
    return const


def build_variant_I(n: int, mu0: float) -> BuiltProblem:
    c = cp.Variable(n + 1)
    delta = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    coshmu0_sq = float(np.cosh(mu0) ** 2)

    equalities = []
    gram_vars = []
    constraints = [delta >= 0.0]

    consA, gvA, rawA = markov_lukacs_cheb(c - e0, n, X_A, X_B)
    constraints += consA
    gram_vars += gvA

    consB, gvB, rawB = markov_lukacs_cheb((1.0 + delta) * e0 - c, n, X1_A, X1_B)
    constraints += consB
    gram_vars += gvB

    consC, gvC, rawC = markov_lukacs_cheb(c - coshmu0_sq * e0, n, X0_A, X0_B)
    constraints += consC
    gram_vars += gvC

    rawD = cp.sum(c) - 1.0
    eqD = (rawD == 0)
    constraints.append(eqD)

    all_vars = [c, delta] + gram_vars
    equalities.append(("A", -_const_part(rawA, all_vars), rawA, consA[-1]))
    equalities.append(("B", -_const_part(rawB, all_vars), rawB, consB[-1]))
    equalities.append(("C", -_const_part(rawC, all_vars), rawC, consC[-1]))
    equalities.append(("D", -_const_part(rawD, all_vars), rawD, eqD))

    problem = cp.Problem(cp.Minimize(delta), constraints)
    return BuiltProblem(problem=problem, delta=delta, equalities=equalities, gram_vars=gram_vars)


def build_variant_II(n: int, mu0: float) -> BuiltProblem:
    r = cp.Variable(n)  # R's Chebyshev coeffs, degree n-1
    delta = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    coshmu0_sq = float(np.cosh(mu0) ** 2)
    one_minus_x = np.array([1.0, -1.0])  # Cheb coeffs of (1-x) = T0 - T1

    equalities = []
    gram_vars = []
    constraints = [delta >= 0.0]

    # (A): R >= 0 on [-1,1], degree n-1, directly on r's own coeffs.
    consA, gvA, rawA = markov_lukacs_cheb(r, n - 1, X_A, X_B)
    constraints += consA
    gram_vars += gvA

    Qhat_var_part = cheb_poly_mul_fixed(one_minus_x, r, n - 1)  # degree n, affine in r, NO constant

    consB, gvB, rawB = markov_lukacs_cheb((1.0 + delta) * e0 - e0 - Qhat_var_part, n, X1_A, X1_B)
    constraints += consB
    gram_vars += gvB

    consC, gvC, rawC = markov_lukacs_cheb(e0 + Qhat_var_part - coshmu0_sq * e0, n, X0_A, X0_B)
    constraints += consC
    gram_vars += gvC

    all_vars = [r, delta] + gram_vars
    equalities.append(("A", -_const_part(rawA, all_vars), rawA, consA[-1]))
    equalities.append(("B", -_const_part(rawB, all_vars), rawB, consB[-1]))
    equalities.append(("C", -_const_part(rawC, all_vars), rawC, consC[-1]))

    problem = cp.Problem(cp.Minimize(delta), constraints)
    return BuiltProblem(problem=problem, delta=delta, equalities=equalities, gram_vars=gram_vars)


def block_domain(name: str, deg_A: int, n: int):
    """(deg, a, b) for the block named 'A'/'B'/'C'/'D' -- D has none."""
    if name == "A":
        return deg_A, X_A, X_B
    if name == "B":
        return n, X1_A, X1_B
    if name == "C":
        return n, X0_A, X0_B
    return None


def extract_dual(built: BuiltProblem, n: int, deg_A: int) -> dict:
    """underline_delta (raw b^Ty sum) plus a PSD-verified flag per block
    (D, if present, has no PSD block -- trivially 'verified')."""
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
        dom = block_domain(name, deg_A, n)
        if dom is not None:
            deg, a, b = dom
            mats = dual_matrices_cheb(y, deg, a, b)
            min_eig = min(float(np.linalg.eigvalsh(M).min()) for M in mats)
            verified = min_eig >= -1e-6
            all_verified = all_verified and verified
            per_block[name] = {"contrib": contrib, "min_eig": min_eig, "verified": verified}
        else:
            per_block[name] = {"contrib": contrib, "min_eig": None, "verified": True}
    return {"underline_delta": total, "verified": all_verified, "per_block": per_block}
