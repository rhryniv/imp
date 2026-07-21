"""SDP filter synthesis (Section 6 of the paper). Optimization only -- all
transfer-matrix / factorization physics is delegated to forward.py and
inverse.py.

Two SDPs are implemented:

  design_filter       -- the autocorrelation-domain relaxation (Remark
                          5.10): variables are the autocorrelation
                          coefficients c_l, constraints (A) G>=1 and
                          (B) G<=u on J1, objective min u. Cheap, no rank
                          relaxation, always solves a well-posed convex
                          problem -- but constraint (C) (stop-band depth)
                          is not part of it, only checked post hoc via
                          inverse.alphas_from_a + forward.kappa_B.

  design_filter_full   -- the full lifted SDP (Appendix C): constraints
                          (A)+(B)+(C)+(D) jointly, via the rank relaxation
                          A ~ a a^T. More expensive, and the relaxation is
                          essentially *never* tight in practice (see next
                          paragraph) -- its raw output is only a warm
                          start, refined by a local polish before being
                          returned.

IMPORTANT, found empirically (see commit history / prior analysis): for
*any* J0, J1, the unconstrained (A)+(B)+(D) problem's global optimum is
always the trivial filter a=(1,0,...,0) (u=1, delta1=0, zero stop-band
attenuation), because nothing in that problem penalizes doing nothing. So
design_filter's post-hoc check of (C) essentially always fails in
practice, and degree_scan/full_pipeline fall back to design_filter_full.
design_filter is still useful as a cheap admissibility/flatness bound and
is kept as the first, cheaper attempt per the "prefer it as primary path"
recipe -- just don't expect it to satisfy (C) on its own.

ALSO IMPORTANT: the rank relaxation A ~ a a^T in design_filter_full is a
genuine relaxation, and empirically (checked across the n=3..9, mu0=0.5..1.5
grid) its raw solution violates constraint (A) 100% of the time -- e.g.
G dipping to 0.01 against the required >=1, well outside solver tolerance.
The relaxed 'a' is therefore only a warm start; _polish() refines it with
a local, non-convex, grid-constrained solve (SLSQP, multi-restart) against
the *true* pointwise constraints, and design_filter_full returns the
polished vector, keeping the raw one as .a_raw for inspection. This is not
mentioned in the original module split (there is no separate polish.py
here) because it is still pure optimization -- it belongs next to the SDP
that needs it, not in forward.py/inverse.py which are physics-only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

from .poly_sdp import cheb_to_mono_matrix, interval_nonneg_constraints
from .forward import kappa_B, transmission_TN, q1_abs_sq
from .inverse import fejer_riesz, alphas_from_a, ensure_min_phase, check_min_phase

Interval = tuple[float, float]


def theta_interval_to_x(gamma: float, delta: float) -> Interval:
    if not (0.0 <= gamma < delta <= np.pi + 1e-12):
        raise ValueError(f"expected 0 <= gamma < delta <= pi, got ({gamma}, {delta})")
    return float(np.cos(delta)), float(np.cos(gamma))


# --------------------------------------------------------------------------
# design_filter: autocorrelation-domain SDP, (A)+(B)+(D) only
# --------------------------------------------------------------------------

@dataclass
class DesignResult:
    n: int
    status: str
    c: np.ndarray | None = None            # autocorrelation coeffs of G, length n+1
    a: np.ndarray | None = None            # a Fejer-Riesz factor of c (min-phase)
    u: float | None = None                 # = (1+delta1)^2
    delta1: float | None = None
    C_satisfied: bool = False              # post-hoc check of constraint (C), any sign pattern
    achieved_mu: float = 0.0               # min_{J0} arccosh(|kappa_B|), 0 if C fails somewhere
    problem: cp.Problem | None = None
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def design_filter(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                   solver: str = "CLARABEL", **solver_kwargs) -> DesignResult:
    """Autocorrelation-domain SDP: minimize u s.t. G>=1 on [-1,1], G<=u on J1,
    sum(c)=1 (constraint D, since G(1)=sum(c_l) and D requires q~_1(0)=1).
    Constraint (C) is checked post hoc on the Fejer-Riesz factor of c."""
    if n < 1:
        raise ValueError("n must be >= 1")
    T = cheb_to_mono_matrix(n)
    c = cp.Variable(n + 1)
    u = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    constraints = [u >= 1.0, cp.sum(c) == 1.0]
    consA, _ = interval_nonneg_constraints(T @ (c - e0), n, -1.0, 1.0)
    constraints += consA
    for gamma, delta in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta)
        consB, _ = interval_nonneg_constraints(T @ (u * e0 - c), n, xlo, xhi)
        constraints += consB

    problem = cp.Problem(cp.Minimize(u), constraints)
    problem.solve(solver=solver, **solver_kwargs)

    result = DesignResult(n=n, status=problem.status, problem=problem, J0=list(J0), J1=list(J1))
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return result

    result.c = c.value
    result.u = float(u.value)
    result.delta1 = float(np.sqrt(max(result.u, 0.0)) - 1.0)
    result.a = fejer_riesz(result.c)
    result.a *= np.sign(np.sum(result.a)) or 1.0  # constraint D wants sum(a)=+1

    mu_min, any_ok = np.inf, False
    for alpha, beta in J0:
        th = np.linspace(alpha, beta, 4000)
        kap = kappa_B(result.a, th)
        for sign in (1, -1):
            if np.all(sign * kap >= 1.0):
                any_ok = True
                mu_min = min(mu_min, np.arccosh(np.min(sign * kap)))
                break
        else:
            mu_min = 0.0
    result.achieved_mu = mu_min if any_ok else 0.0
    result.C_satisfied = any_ok and mu_min >= mu0 - 1e-9
    return result


# --------------------------------------------------------------------------
# design_filter_full: the full lifted SDP, (A)+(B)+(C)+(D) jointly
# --------------------------------------------------------------------------

def _autocorr_from_A(A: cp.Expression, n: int) -> cp.Expression:
    terms = [cp.trace(A)]
    for l in range(1, n + 1):
        diag_terms = [A[l + i, i] for i in range(n + 1 - l)]
        terms.append(cp.sum(cp.hstack(diag_terms)) if len(diag_terms) > 1 else diag_terms[0])
    return cp.hstack(terms)


@dataclass
class FullDesignResult:
    n: int
    mu0: float
    status: str
    sigma: tuple | None = None
    a: np.ndarray | None = None            # polished (see _polish): satisfies (A)/(B)/(C) pointwise
    a_raw: np.ndarray | None = None        # raw relaxed SDP solution, before polishing
    A: np.ndarray | None = None
    u: float | None = None
    delta1: float | None = None
    tightness_ratio: float | None = None
    polish_verified: bool | None = None    # did the polish actually reach a feasible point?
    min_phase: bool | None = None          # is .a already realizable (Thm 4.6), no reflection needed?
    problem: cp.Problem | None = None
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def _polish(a0: np.ndarray, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float, sigma,
            n_grid_A: int = 600, n_grid_local: int = 300, maxiter: int = 300, n_restarts: int = 5,
            seed: int = 0):
    """Local refinement of a relaxed-SDP solution a0 against the *true*
    pointwise constraints (A)/(B)/(C)/(D), via multi-start SLSQP. Needed
    because the lift in design_filter_full is not tight in practice (see
    module docstring). Returns (a, u, verified)."""
    n = len(a0) - 1
    m = np.arange(n + 1)
    theta_A = np.linspace(0.0, np.pi, n_grid_A)
    theta_B = [np.linspace(g, d, n_grid_local) for g, d in J1]
    theta_C = [np.linspace(al, be, n_grid_local) for al, be in J0]

    def G_and_grad(a, theta):
        cos_mt = np.cos(np.outer(theta, m))
        sin_mt = np.sin(np.outer(theta, m))
        re, im = cos_mt @ a, sin_mt @ a
        G = re ** 2 + im ** 2
        dG = 2 * re[:, None] * cos_mt + 2 * im[:, None] * sin_mt
        return G, dG

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    constraints = []

    def conA(x):
        G, _ = G_and_grad(x[:-1], theta_A)
        return G - 1.0

    def conA_jac(x):
        _, dG = G_and_grad(x[:-1], theta_A)
        J = np.zeros((len(theta_A), len(x)))
        J[:, :-1] = dG
        return J

    constraints.append({"type": "ineq", "fun": conA, "jac": conA_jac})

    for th in theta_B:
        def conB(x, th=th):
            G, _ = G_and_grad(x[:-1], th)
            return x[-1] - G

        def conB_jac(x, th=th):
            _, dG = G_and_grad(x[:-1], th)
            J = np.zeros((len(th), len(x)))
            J[:, :-1], J[:, -1] = -dG, 1.0
            return J

        constraints.append({"type": "ineq", "fun": conB, "jac": conB_jac})

    coshmu0 = np.cosh(mu0)
    for th, sig in zip(theta_C, sigma):
        cos_mt = np.cos(np.outer(th, m))

        def conC(x, sig=sig, cos_mt=cos_mt):
            return sig * (cos_mt @ x[:-1]) - coshmu0

        def conC_jac(x, sig=sig, cos_mt=cos_mt):
            J = np.zeros((cos_mt.shape[0], len(x)))
            J[:, :-1] = sig * cos_mt
            return J

        constraints.append({"type": "ineq", "fun": conC, "jac": conC_jac})

    def conD(x):
        return np.array([np.sum(x[:-1]) - 1.0])

    def conD_jac(x):
        J = np.zeros((1, len(x)))
        J[0, :-1] = 1.0
        return J

    constraints.append({"type": "eq", "fun": conD, "jac": conD_jac})

    def verify(a, u, tol=1e-6):
        if np.min(q1_abs_sq(a, np.linspace(0, np.pi, 8000))) < 1.0 - tol:
            return False
        for th in theta_B:
            if np.max(q1_abs_sq(a, th)) > u + tol:
                return False
        for th, sig in zip(theta_C, sigma):
            if np.min(sig * kappa_B(a, th)) < coshmu0 - tol:
                return False
        return True

    Gmax_pass = max((np.max(G_and_grad(a0, th)[0]) for th in theta_B), default=1.0)
    u0 = max(1.0 + 1e-9, Gmax_pass)
    rng = np.random.default_rng(seed)

    def run_attempts(seeds):
        """seeds: list of warm-start a-vectors. Returns (verified_mp, verified_any, any_)
        each a list of (a, u) sorted by u ascending, or empty."""
        verified_mp, verified_any, any_ = [], [], []
        for x0_a in seeds:
            x0 = np.concatenate([x0_a, [u0]])
            res = minimize(objective, x0, jac=objective_grad, constraints=constraints,
                            method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
            if not np.all(np.isfinite(res.x)):
                continue
            a_try, u_try = res.x[:-1], float(res.x[-1])
            any_.append((a_try, u_try))
            if verify(a_try, u_try):
                verified_any.append((a_try, u_try))
                if check_min_phase(a_try)[0]:
                    verified_mp.append((a_try, u_try))
        verified_mp.sort(key=lambda t: t[1])
        verified_any.sort(key=lambda t: t[1])
        any_.sort(key=lambda t: t[1])
        return verified_mp, verified_any, any_

    seeds = [a0] + [np.clip(a0 + 0.02 * rng.standard_normal(n + 1), -10, 10) for _ in range(n_restarts - 1)]
    vmp, vany, anyc = run_attempts(seeds)

    if not vmp and vany:
        # Found constraint-feasible points, but none already realizable
        # (minimum-phase). Reflecting onto the realizable branch generally
        # breaks (C) (kappa_B changes under reflection -- see
        # inverse.ensure_min_phase), so instead re-polish *from* the
        # reflection, hoping SLSQP finds a nearby min-phase-preserving
        # local optimum rather than drifting back off it.
        reflect_seeds = [ensure_min_phase(a)[0] for a, _ in vany[: min(3, len(vany))]]
        vmp2, vany2, anyc2 = run_attempts(reflect_seeds)
        vmp += vmp2
        vmp.sort(key=lambda t: t[1])
        vany += vany2
        vany.sort(key=lambda t: t[1])
        anyc += anyc2
        anyc.sort(key=lambda t: t[1])

    if vmp:
        return vmp[0][0], vmp[0][1], True
    if vany:
        return vany[0][0], vany[0][1], True
    if anyc:
        return anyc[0][0], anyc[0][1], False
    return a0, u0, False


def _solve_full_for_sigma(n, J0, J1, mu0, sigma, T, solver, solver_kwargs):
    a = cp.Variable(n + 1)
    A = cp.Variable((n + 1, n + 1), symmetric=True)
    u = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    lift = cp.bmat([[A, cp.reshape(a, (n + 1, 1), order="C")],
                    [cp.reshape(a, (1, n + 1), order="C"), np.array([[1.0]])]])
    constraints = [lift >> 0, u >= 1.0, cp.sum(a) == 1.0]

    c = _autocorr_from_A(A, n)
    consA, _ = interval_nonneg_constraints(T @ (c - e0), n, -1.0, 1.0)
    constraints += consA
    for gamma, delta in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta)
        consB, _ = interval_nonneg_constraints(T @ (u * e0 - c), n, xlo, xhi)
        constraints += consB

    coshmu0 = float(np.cosh(mu0))
    for (alpha, beta), sig in zip(J0, sigma):
        xlo, xhi = theta_interval_to_x(alpha, beta)
        stop_cheb = sig * a - coshmu0 * e0
        consC, _ = interval_nonneg_constraints(T @ stop_cheb, n, xlo, xhi)
        constraints += consC

    problem = cp.Problem(cp.Minimize(u), constraints)
    problem.solve(solver=solver, **solver_kwargs)

    result = FullDesignResult(n=n, mu0=mu0, status=problem.status, sigma=sigma,
                               problem=problem, J0=list(J0), J1=list(J1))
    if problem.status in ("optimal", "optimal_inaccurate"):
        result.a_raw = a.value
        result.A = A.value
        eigvals = np.sort(np.linalg.eigvalsh(result.A))[::-1]
        result.tightness_ratio = float(eigvals[1] / eigvals[0]) if eigvals[0] > 1e-12 else None

        a_pol, u_pol, verified = _polish(result.a_raw, J0, J1, mu0, sigma)
        result.a = a_pol
        result.u = u_pol
        result.delta1 = float(np.sqrt(max(u_pol, 0.0)) - 1.0)
        result.polish_verified = verified
        result.min_phase = check_min_phase(a_pol)[0]
    return result


def design_filter_full(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                        sigma: tuple | None = None, solver: str = "CLARABEL",
                        **solver_kwargs) -> FullDesignResult:
    """Full lifted SDP. If sigma is None, tries all 2^len(J0) sign patterns
    and returns the feasible one with smallest u."""
    T = cheb_to_mono_matrix(n)
    if sigma is not None:
        return _solve_full_for_sigma(n, J0, J1, mu0, sigma, T, solver, solver_kwargs)
    # Prefer verified + already-realizable (min-phase) results over merely
    # verified ones: a smaller delta1 that turns out unrealizable is not
    # actually better once you account for what reflecting onto the
    # realizable branch does to constraint (C) (see inverse.ensure_min_phase
    # / sdp_design._polish's re-polish-from-reflection step).
    best_verified_mp, best_verified_any, best_any = None, None, None
    for sig in itertools.product((1, -1), repeat=len(J0)):
        res = _solve_full_for_sigma(n, J0, J1, mu0, sig, T, solver, solver_kwargs)
        if res.status in ("optimal", "optimal_inaccurate"):
            if res.polish_verified and res.min_phase and (best_verified_mp is None or res.u < best_verified_mp.u):
                best_verified_mp = res
            if res.polish_verified and (best_verified_any is None or res.u < best_verified_any.u):
                best_verified_any = res
            if best_any is None or res.u < best_any.u:
                best_any = res
        elif best_any is None:
            best_any = res
    return best_verified_mp or best_verified_any or best_any


# --------------------------------------------------------------------------
# realize_design / degree_scan / full_pipeline
# --------------------------------------------------------------------------

@dataclass
class RealizationResult:
    a: np.ndarray               # the REALIZABLE vector actually used below (= info.a_used)
    a_as_designed: np.ndarray   # the raw input to realize_design, before any reflection
    alphas: np.ndarray
    impedances: np.ndarray
    admissible: bool
    G_min: float
    was_reflected: bool
    reconstruction_error: float
    C_satisfied: bool
    achieved_mu: float
    a_reconstructed: np.ndarray
    round_trip_error: float
    TN_stop_max: dict
    TN_pass_min: dict
    reliable: bool = True   # False => alphas/impedances (hence TN_*) should not be trusted, see .failure
    failure: str | None = None


def realize_design(a_or_c: np.ndarray, J0: Sequence[Interval], J1: Sequence[Interval],
                    mu0: float, sigma: Sequence[int] | None = None,
                    N_values: Sequence[int] = (5, 10, 20), from_autocorrelation: bool = False
                    ) -> RealizationResult:
    """Given a (or c, with from_autocorrelation=True): factor if needed ->
    inverse.alphas_from_a -> forward.a_from_alphas (round-trip check) ->
    evaluate T_N on I0/I1 for each N in N_values.

    IMPORTANT: if the input a is not already minimum-phase,
    inverse.alphas_from_a reflects it onto the realizable representative
    (info.a_used) before stripping -- reflection preserves |q~_1| but
    generally changes kappa_B = Re(q~_1). So (C), T_N, etc. are all
    evaluated on info.a_used (the vector the returned alphas/impedances
    actually correspond to, i.e. what could physically be built), *not*
    on the possibly-unrealizable input -- otherwise the reported
    performance would describe a filter that can't be built. The input is
    kept as .a_as_designed for comparison/diagnostics.
    """
    from .forward import a_from_alphas

    a_designed = fejer_riesz(np.asarray(a_or_c, dtype=float)) if from_autocorrelation else np.asarray(a_or_c, dtype=float)

    alphas, info = alphas_from_a(a_designed)
    a = info.a_used  # the realizable vector: everything below is evaluated on THIS
    a_reconstructed = a_from_alphas(alphas)
    round_trip_error = float(np.max(np.abs(a_reconstructed - a)))

    # kappa_B can change sign under reflection, so always redetect sigma on
    # the realizable a rather than trusting a sigma chosen for a_designed.
    sigma = [1 if np.mean(kappa_B(a, np.linspace(al, be, 4000))) > 0 else -1 for al, be in J0]

    all_ok, mu_min = True, np.inf
    for (alpha, beta), sig in zip(J0, sigma):
        kap = kappa_B(a, np.linspace(alpha, beta, 4000))
        if np.all(sig * kap >= 1.0):
            mu_min = min(mu_min, np.arccosh(np.min(sig * kap)))
        else:
            all_ok = False
    achieved_mu = mu_min if (all_ok and J0) else 0.0
    C_satisfied = all_ok and achieved_mu >= mu0 - 1e-9

    TN_stop_max, TN_pass_min = {}, {}
    for N in N_values:
        TN_stop_max[N] = max((float(np.max(transmission_TN(a, np.linspace(al, be, 2000), N)))
                               for al, be in J0), default=float("nan"))
        TN_pass_min[N] = min((float(np.min(transmission_TN(a, np.linspace(g, d, 2000), N)))
                               for g, d in J1), default=float("nan"))

    return RealizationResult(
        a=a, a_as_designed=a_designed, alphas=alphas, impedances=info.impedances,
        admissible=info.admissible, G_min=info.G_min, was_reflected=info.was_reflected,
        reconstruction_error=info.reconstruction_error,
        C_satisfied=C_satisfied, achieved_mu=achieved_mu,
        a_reconstructed=a_reconstructed,
        round_trip_error=round_trip_error,
        TN_stop_max=TN_stop_max, TN_pass_min=TN_pass_min,
        reliable=info.reliable, failure=info.failure,
    )


def degree_scan(n_range: Sequence[int], J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                 **kwargs) -> dict:
    """Sweep n; for each, try design_filter first (cheap), fall back to
    design_filter_full if (C) isn't satisfied post hoc (see module
    docstring -- in practice this fallback triggers almost always)."""
    results = {}
    for n in n_range:
        res = design_filter(n, J0, J1, mu0, **kwargs)
        if not res.C_satisfied:
            res = design_filter_full(n, J0, J1, mu0, **kwargs)
        results[n] = res
    return results


@dataclass
class PipelineResult:
    n: int
    design: object
    realization: RealizationResult
    figure: object = None


def full_pipeline(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                   N_values: Sequence[int] = (5, 10, 20), plot: bool = True, **kwargs) -> PipelineResult:
    """SDP -> realize -> forward T_N -> plot. Single entry point."""
    design = design_filter(n, J0, J1, mu0, **kwargs)
    sigma = None
    if not design.C_satisfied:
        design = design_filter_full(n, J0, J1, mu0, **kwargs)
        sigma = design.sigma
    realization = realize_design(design.a, J0, J1, mu0, sigma=sigma, N_values=N_values)

    fig = None
    if plot:
        from .forward import plot_filter
        fig = plot_filter(realization.a, N_values, I0=list(J0), I1=list(J1))

    return PipelineResult(n=n, design=design, realization=realization, figure=fig)
