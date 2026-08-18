"""SDP certificate test for the impedance-filter lower bound.

Compares Variant I (explicit equality Qhat(1)=1) vs Variant II (equality
eliminated via Qhat=1+(1-x)R(x)) magnitude-relaxation dual bounds, n=1..8,
across CLARABEL and SCS. See report.md for the write-up.
"""
from __future__ import annotations

import json
import time

import numpy as np
import cvxpy as cp

from variants import build_variant_I, build_variant_II, extract_dual, T, U, X1_A, X1_B, X0_A, X0_B

np.random.seed(0)

MU0 = 1.0
N_RANGE = range(1, 9)
SOLVERS = [cp.CLARABEL, cp.SCS]
GRID_N = 4000

sinh2_mu0 = float(np.sinh(MU0) ** 2)
cosh2_mu0 = float(np.cosh(MU0) ** 2)
gamma = float(np.arccosh(abs((2 * np.cos(U) - np.cos(T) - 1) / (1 - np.cos(T)))))
print(f"gamma = {gamma:.6f} (expected ~3.06)")
print(f"analytic bound prefactor sinh^2(mu0) = {sinh2_mu0:.6f}")
analytic_prefactor = 1.3811  # per brief; close to sinh^2(1)=1.38109...


def cheb_eval(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate sum c_k T_k(x) via Clenshaw's recurrence."""
    n = len(c) - 1
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    for k in range(n, 0, -1):
        b0 = c[k] + 2 * x * b1 - b2
        b2 = b1
        b1 = b0
    return c[0] + x * b1 - b2


def grid_feasibility(c: np.ndarray, delta: float, tol: float = 1e-6) -> dict:
    xA = np.linspace(-1.0, 1.0, GRID_N)
    x1 = np.linspace(X1_A, X1_B, GRID_N)
    x0 = np.linspace(X0_A, X0_B, GRID_N)
    qA = cheb_eval(c, xA)
    q1 = cheb_eval(c, x1)
    q0 = cheb_eval(c, x0)
    q_at_1 = cheb_eval(c, np.array([1.0]))[0]
    viol_A = float(np.min(qA - 1.0))          # want >= 0
    viol_B = float(np.min((1.0 + delta) - q1))  # want >= 0
    viol_C = float(np.min(q0 - cosh2_mu0))      # want >= 0
    viol_D = float(abs(q_at_1 - 1.0))           # want ~ 0
    ok = (viol_A >= -tol) and (viol_B >= -tol) and (viol_C >= -tol) and (viol_D <= 1e-4)
    return {"min_slack_A": viol_A, "min_slack_B": viol_B, "min_slack_C": viol_C,
            "abs_err_D": viol_D, "feasible": ok}


results = []
for n in N_RANGE:
    for variant_name, builder, deg_A in (("I", build_variant_I, n), ("II", build_variant_II, n - 1)):
        for solver in SOLVERS:
            bp = builder(n, MU0)
            t0 = time.time()
            try:
                bp.problem.solve(solver=solver)
            except Exception as exc:  # solver-specific failure
                status = f"error:{exc}"
                delta_p = None
            else:
                status = bp.problem.status
                delta_p = bp.delta.value
            elapsed = time.time() - t0

            underline_delta = None
            verified = False
            feas = None
            if status in ("optimal", "optimal_inaccurate") and delta_p is not None:
                dualres = extract_dual(bp, n=n, deg_A=deg_A)
                underline_delta = dualres["underline_delta"]
                verified = dualres["verified"]

                if variant_name == "I":
                    c_var = [v for v in bp.problem.variables() if v.shape == (n + 1,)][0]
                    c = np.array(c_var.value, dtype=float)
                else:
                    r_var = [v for v in bp.problem.variables() if v.shape == (n,)][0]
                    r = np.array(r_var.value, dtype=float)
                    one_minus_x = np.array([1.0, -1.0])
                    from cheb_mk import cheb_poly_mul_fixed
                    import cvxpy as _cp
                    # Evaluate numerically: (1-x)*R(x) coeffs via same routine, using a Constant wrapping.
                    var_part = cheb_poly_mul_fixed(one_minus_x, _cp.Constant(r), n - 1)
                    c = np.zeros(n + 1)
                    c[0] = 1.0
                    c = c + np.array(var_part.value, dtype=float)
                feas = grid_feasibility(c, delta_p)

            gap = (delta_p - underline_delta) if (delta_p is not None and underline_delta is not None) else None
            analytic_val = analytic_prefactor * np.exp(-gamma * n)
            ratio = (underline_delta / analytic_val) if underline_delta is not None else None

            rec = {
                "n": n, "variant": variant_name, "solver": str(solver), "status": status,
                "delta_primal": delta_p, "underline_delta": underline_delta,
                "gap": gap, "dual_verified": verified, "ratio_to_analytic": ratio,
                "analytic_bound": analytic_val, "time_s": elapsed, "feas": feas,
            }
            results.append(rec)
            print(f"n={n} variant={variant_name} solver={str(solver):10s} status={status:20s} "
                  f"delta_p={delta_p} underline={underline_delta} gap={gap} verified={verified} "
                  f"ratio={ratio} feas_ok={feas['feasible'] if feas else None} t={elapsed:.2f}s")

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=_json_default)

print("\nSaved results.json")
