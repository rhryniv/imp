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


def widen_intervals(intervals: Sequence[Interval], margin: float,
                     domain: Interval = (0.0, np.pi)) -> list[Interval]:
    """Expand each (lo, hi) interval by `margin` on both sides (clipped to
    domain). Intended as a guard band: impose constraints (B)/(C) on a
    region slightly wider than the true target, then evaluate/report
    performance only on the original, narrower interval. The worst-case
    point of a constrained trig polynomial over a prescribed interval tends
    to sit at or very near that interval's own edge (confirmed empirically
    -- see examples/multiband_demo.py), since nothing outside the
    prescribed interval controls the transition into it; solving on a wider
    region pushes that edge roughness away from the region actually being
    reported on."""
    lo_bound, hi_bound = domain
    widened = []
    for lo, hi in intervals:
        new_lo = max(lo_bound, lo - margin)
        new_hi = min(hi_bound, hi + margin)
        if new_hi <= new_lo:
            raise ValueError(f"margin {margin} too large for interval ({lo}, {hi})")
        widened.append((new_lo, new_hi))
    return widened


def assert_disjoint_intervals(intervals: Sequence[Interval]) -> None:
    """Raise if any two intervals in the combined list overlap -- meant to
    guard against widen_intervals' margin eating into a neighbouring band
    (e.g. I0's guard band creeping into I1's)."""
    ordered = sorted(intervals)
    for (lo1, hi1), (lo2, hi2) in zip(ordered, ordered[1:]):
        if hi1 > lo2:
            raise ValueError(f"intervals overlap: ({lo1}, {hi1}) and ({lo2}, {hi2}) -- reduce margin")


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


def sdp_lower_bound(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                     solver: str = "CLARABEL", **solver_kwargs) -> float | None:
    """Tightest valid lower bound on the true achievable delta at this n,
    from the *raw* (pre-polish) SDP relaxation. For a fixed sign pattern,
    relaxing the rank-1 constraint A ~ a a^T to A >= a a^T (Schur
    complement) can only enlarge the feasible set, so the relaxed optimum
    is <= the true optimum for that sign pattern; minimizing the relaxed
    optimum over all 2^len(J0) sign patterns therefore lower-bounds the
    true overall optimum (also a min over sign patterns).

    This is deliberately *not* the same as design_filter_full(...).problem:
    that reflects whichever single sign pattern gave the best *polished*
    result, which need not be the pattern with the tightest raw relaxed
    value -- using it as a "lower bound" can be unsound (confirmed: it
    produced an achieved delta below the reported bound on a test case).
    Returns None if no sign pattern's relaxation solved."""
    T = cheb_to_mono_matrix(n)
    best_u = None
    for sig in itertools.product((1, -1), repeat=len(J0)):
        res = _solve_full_for_sigma(n, J0, J1, mu0, sig, T, solver, solver_kwargs)
        if res.problem is not None and res.problem.value is not None:
            u_lb = float(res.problem.value)
            if best_u is None or u_lb < best_u:
                best_u = u_lb
    if best_u is None:
        return None
    return float(np.sqrt(max(best_u, 0.0)) - 1.0)


# --------------------------------------------------------------------------
# design_via_layers: optimize directly over the physical Schur parameters
# --------------------------------------------------------------------------
#
# design_filter_full optimizes over a directly, then _polish tries to reach
# a feasible point, then (in realize_design) that point gets reflected onto
# the realizable branch if it wasn't already one -- and reflection changes
# kappa_B, generally breaking constraint (C). Every remaining failure mode
# found in this project traces back to that same gap between "satisfies the
# design constraints" and "is a physically buildable filter."
#
# forward.forward_reconstruct closes that gap by construction: for *any*
# real alpha_0,...,alpha_n, it produces a q~_1 that is automatically
# admissible (constraint A, the SU(1,1) identity |p1|^2-|p2|^2=1 holds
# identically) and automatically realizable (Prop. 4.4's zero-free property
# falls out of the recursion itself). So optimizing directly over
# gamma_j = tanh(alpha_j) in (-1,1) -- rather than over a -- never needs
# reflection, never loses constraint (C) to it, and constraint (A) doesn't
# need to be imposed at all (it's automatic). Only (B) and (C) remain as
# actual constraints; (D) (sum(alphas)=0) is eliminated by solving for
# alpha_n as a function of the other n alphas (_free_to_alphas), which also
# removes an equality constraint that made early attempts at this approach
# converge badly.
#
# The tradeoff: this landscape is genuinely harder for a local solver than
# optimizing a directly (design_filter_full has an SDP relaxation to warm
# start from; this doesn't, and a naive random start is usually infeasible
# everywhere). design_via_layers compensates by warm-starting from
# design_filter_full's own (possibly unrealizable) result -- reflected onto
# the realizable branch first -- in addition to a user-supplied warm start
# and the trivial all-zero one, then multi-restarts with small perturbations
# around whichever seeds are available.
#
# A related, separate axis: alpha_j=0 is an ordinary *interior* value of
# each gamma_j, not a missing layer -- that lattice cell still applies its
# own transit phase (see forward.py), it just has no impedance jump. So an
# already-verified n_small-layer solution is not literally a special case
# of an n-layer one with trailing zeros appended (the extra spacer cells
# add phase, which generally changes kappa_B), but *embedding* it that way
# is still a genuinely different, often better, n-layer design: the same
# cluster of real impedance jumps, now with tunable spacing -- trailing
# zeros widen the gap before the next repeated block, interior zeros widen
# a layer inside the structure. Nothing prevents the local optimizer from
# finding such a point on its own, but a naive random restart essentially
# never lands near an exact- or near-zero gamma_j by chance. The
# smaller_solutions argument below seeds a handful of restarts with exactly
# these padded embeddings (trailing, leading, split-down-the-middle, and a
# few random insertion points), so the search actually explores that part
# of the space instead of relying on chance.

def _free_to_alphas(gamma_free: np.ndarray) -> np.ndarray:
    """gamma_free = (gamma_0,...,gamma_{n-1}) are free in (-1,1); alpha_n is
    *solved for* so that sum(alphas)=0 (constraint D) holds automatically,
    removing it as an explicit equality constraint."""
    alphas_free = np.arctanh(gamma_free)
    alpha_n = -np.sum(alphas_free)
    return np.concatenate([alphas_free, [alpha_n]])


def _padded_alpha_seeds(alphas_small: np.ndarray, n_target: int, rng: np.random.Generator,
                         max_variants: int = 4) -> list[np.ndarray]:
    """Embed a smaller, already-verified n_small-layer solution into an
    n_target-slot layer sequence by inserting (n_target - n_small) zero-alpha
    spacer layers. Trailing insertion widens the spacing before the next
    repeated block; interior insertion widens a layer inside the structure
    (see the section docstring above). These are only *seeds* for the local
    optimizer -- padding does not preserve kappa_B exactly -- not
    guaranteed-feasible points on their own."""
    n_small = len(alphas_small) - 1
    m = n_target - n_small
    if m <= 0:
        return [alphas_small[: n_target + 1]]
    mid = (n_small + 1) // 2
    variants = [
        np.concatenate([alphas_small, np.zeros(m)]),                              # trailing
        np.concatenate([np.zeros(m), alphas_small]),                              # leading
        np.concatenate([alphas_small[:mid], np.zeros(m), alphas_small[mid:]]),    # split down the middle
    ]
    for _ in range(max(0, max_variants - len(variants))):
        pos = int(rng.integers(0, n_small + 2))
        variants.append(np.concatenate([alphas_small[:pos], np.zeros(m), alphas_small[pos:]]))
    return variants


@dataclass
class LayerDesignResult:
    n: int
    mu0: float
    status: str
    sigma: tuple | None = None
    alphas: np.ndarray | None = None
    a: np.ndarray | None = None            # = forward.a_from_alphas(alphas): realizable by construction
    impedances: np.ndarray | None = None
    u: float | None = None
    delta1: float | None = None
    verified: bool | None = None           # True iff (B)/(C) hold pointwise on a fine grid
    achieved_mu: float = 0.0


def _sign_from_a(a, theta_C):
    """Auto-detect each J0 component's sign from the current iterate: the
    mean of kappa_B over that component's grid. Used instead of a squared,
    sign-free constraint (kappa_B**2 >= cosh(mu0)**2) -- mathematically
    equivalent to |kappa_B| >= cosh(mu0), but that form is the complement
    of a convex interval (a genuine disjunction), which SLSQP's local QP
    linearization handles badly in practice (confirmed empirically: it
    turned a previously-solved case into a reported infeasibility). Reading
    the sign off the current point and building the same simple *linear*
    constraint sign*kappa_B >= cosh(mu0) as before keeps every individual
    solve exactly as well-behaved as the original fixed-sigma version,
    while still needing no sign fixed or enumerated in advance -- each
    solve (each restart, each continuation step) just looks at its own
    current point instead of guessing upfront."""
    return tuple(1 if np.mean(kappa_B(a, th)) > 0 else -1 for th in theta_C)


def _solve_layers_for_sigma(n, J0, J1, mu0, sigma, gamma_bound, n_grid_B, n_grid_C, seeds, maxiter=400):
    """One sign pattern's worth of the search: plain linear
    sign*kappa_B>=cosh(mu0) constraints (well-behaved for SLSQP -- see
    _sign_from_a's docstring for why the squared/disjunctive alternative
    was tried and reverted). design_via_layers calls this once per sign
    pattern with the *full* seed/restart budget each time: pooling all
    patterns into one run (auto-detecting sign per seed instead) was also
    tried, but under-samples whichever pattern isn't naturally favoured by
    the seeds -- confirmed empirically to regress a known-hard case (n=9 on
    the multiband config) from delta1=0.065 to 0.36. Per-pattern enumeration
    is still fully automatic from the caller's side (see design_via_layers),
    just not collapsed into a single pooled solve."""
    from .forward import a_from_alphas

    theta_B = [np.linspace(g, d, n_grid_B) for g, d in J1]
    theta_C = [np.linspace(al, be, n_grid_C) for al, be in J0]
    coshmu0 = np.cosh(mu0)

    def a_of(gamma_free):
        return a_from_alphas(_free_to_alphas(gamma_free))

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    constraints = []
    for th in theta_B:
        def fun(x, th=th):
            return x[-1] - q1_abs_sq(a_of(x[:-1]), th)
        constraints.append({"type": "ineq", "fun": fun})
    for th, sig in zip(theta_C, sigma):
        def fun(x, th=th, sig=sig):
            return sig * kappa_B(a_of(x[:-1]), th) - coshmu0
        constraints.append({"type": "ineq", "fun": fun})

    bounds = [(-gamma_bound, gamma_bound)] * n + [(1.0, None)]

    def verify(gamma_free, u, tol=1e-6):
        a = a_of(gamma_free)
        for th in theta_B:
            if np.max(q1_abs_sq(a, th)) > u + tol:
                return False
        for th, sig in zip(theta_C, sigma):
            if np.min(sig * kappa_B(a, th)) < coshmu0 - tol:
                return False
        return True

    best = None
    for gamma0 in seeds:
        gamma0 = np.clip(gamma0, -gamma_bound, gamma_bound)
        Gmax = max((np.max(q1_abs_sq(a_of(gamma0), th)) for th in theta_B), default=1.0)
        u0 = max(1.0 + 1e-9, Gmax)
        x0 = np.concatenate([gamma0, [u0]])
        res = minimize(objective, x0, jac=objective_grad, constraints=constraints, bounds=bounds,
                        method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            continue
        gamma_try, u_try = res.x[:-1], float(res.x[-1])
        # SLSQP's own exit status is not trustworthy here either (same
        # lesson as _polish): verify directly instead of trusting res.status.
        if verify(gamma_try, u_try) and (best is None or u_try < best[1]):
            best = (gamma_try, u_try)

    if best is None:
        return None
    gamma_free, u = best
    return _free_to_alphas(gamma_free), a_of(gamma_free), u


def design_via_layers(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                       warm_start_alphas: np.ndarray | None = None,
                       use_sdp_warm_start: bool = True,
                       smaller_solutions: dict[int, np.ndarray] | None = None,
                       n_pad_variants: int = 4, use_continuation_seed: bool = True,
                       gamma_bound: float = 0.9995,
                       n_grid_B: int = 400, n_grid_C: int = 400, n_restarts: int = 8,
                       seed: int = 0, solver: str = "CLARABEL") -> LayerDesignResult:
    """Directly optimize over realizable layer sequences (see the section
    docstring above). No caller ever has to supply a sign pattern for
    constraint (C): all 2^len(J0) sign patterns are tried automatically
    internally (each with its own full seed/restart budget -- pooling them
    into one auto-detected-sign solve was tried and found to under-sample
    hard cases like the multiband n=9 dead zone, see
    _solve_layers_for_sigma's docstring), and the best result across all
    patterns is returned, with the winning sign reported on .sigma. Unlike
    design_filter_full's SDP -- which must fix a sign per component to keep
    (C) affine, and so has no alternative to this enumeration -- SLSQP does
    not require it; the enumeration here is purely a coverage choice, not
    a mathematical necessity.

    smaller_solutions: optional {n_small: alphas} of already-verified
    solutions at smaller n_small < n. Each is embedded into n slots by
    zero-padding (trailing/leading/interior, see _padded_alpha_seeds) and
    added as extra warm-start seeds -- lets the search actually reach the
    "same core structure, wider spacing" designs that a naive random
    restart essentially never finds on its own.

    use_continuation_seed: also run design_via_continuation (see below) and
    add its result as one more warm-start seed. Cheap (one homotopy
    trajectory, not a restart batch) and often lands very close to the
    best basin directly.

    Every returned design is realizable by construction; .verified means
    (B) and (C) additionally hold on a fine grid (there is no
    admissible-but-infeasible middle case to report, unlike design_filter_full).
    """
    rng = np.random.default_rng(seed)

    base_seeds = []
    if warm_start_alphas is not None:
        base_seeds.append(np.tanh(np.asarray(warm_start_alphas, dtype=float)[:n]))
    if use_sdp_warm_start:
        try:
            full_res = design_filter_full(n, J0, J1, mu0, solver=solver)
            if full_res is not None and full_res.a is not None:
                a_mp, _ = ensure_min_phase(full_res.a)
                alphas_ws, info = alphas_from_a(a_mp)
                if info.reliable and not np.any(np.isnan(alphas_ws)):
                    base_seeds.append(np.tanh(alphas_ws[:n]))
        except Exception:
            pass
    if smaller_solutions:
        for n_small, alphas_small in smaller_solutions.items():
            if n_small >= n:
                continue
            alphas_small = np.asarray(alphas_small, dtype=float)
            for padded in _padded_alpha_seeds(alphas_small, n, rng, max_variants=n_pad_variants):
                base_seeds.append(np.tanh(padded[:n]))
    if use_continuation_seed:
        try:
            cont_res = design_via_continuation(n, J0, J1, mu0, gamma_bound=gamma_bound,
                                                n_grid_B=n_grid_B, n_grid_C=n_grid_C)
            if cont_res.alphas is not None:
                base_seeds.append(np.tanh(cont_res.alphas[:n]))
        except Exception:
            pass
    if not base_seeds:
        base_seeds.append(np.zeros(n))

    best_overall = None
    for sig in itertools.product((1, -1), repeat=len(J0)):
        seeds = list(base_seeds)
        n_per_seed = max(1, n_restarts // len(base_seeds))
        for bs in base_seeds:
            seeds += [np.clip(bs + 0.05 * rng.standard_normal(n), -gamma_bound, gamma_bound)
                      for _ in range(n_per_seed)]
        result = _solve_layers_for_sigma(n, J0, J1, mu0, sig, gamma_bound, n_grid_B, n_grid_C, seeds)
        if result is not None:
            alphas, a, u = result
            delta1 = float(np.sqrt(max(u, 0.0)) - 1.0)
            if best_overall is None or delta1 < best_overall[0]:
                best_overall = (delta1, sig, alphas, a, u)

    if best_overall is None:
        return LayerDesignResult(n=n, mu0=mu0, status="infeasible")

    delta1, sig, alphas, a, u = best_overall
    impedances = np.empty(n + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(alphas):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    mu_min = np.inf
    for (al, be), s in zip(J0, sig):
        kap = s * kappa_B(a, np.linspace(al, be, 4000))
        mu_min = min(mu_min, np.arccosh(max(float(np.min(kap)), 1.0)))

    return LayerDesignResult(n=n, mu0=mu0, status="optimal", sigma=sig, alphas=alphas, a=a,
                              impedances=impedances, u=u, delta1=delta1, verified=True,
                              achieved_mu=mu_min if J0 else float("inf"))


# --------------------------------------------------------------------------
# design_via_continuation: homotopy from the trivial alpha=0 structure
# --------------------------------------------------------------------------
#
# alpha=0 identically means q2=0 identically (no back-scattering at all,
# every layer is transparent): T_N=1 exactly for *every* theta and N, so
# constraint (B) holds on I1 with the best possible u=1, trivially. But
# kappa_B(theta) = cos(n*theta) there (a=e_n, pure Chebyshev T_n), which
# only *touches* +-1 at isolated points, not over an interval, so no real
# stop band exists yet at this point -- constraint (C) requires genuinely
# moving away from it.
#
# design_via_layers's random-restart search treats mu0 as a fixed target
# from the start, so every restart independently has to find a basin that
# satisfies the *full* depth requirement; nothing ties the restarts
# together, and (per the multiband_demo.py investigation) they can all fall
# into the same mediocre generic attractor regardless of warm start.
# design_via_continuation instead ramps the required depth mu up from 0 to
# mu0 over a sequence of small steps, re-solving at each step from the
# *previous* step's converged alphas. That keeps the whole trajectory
# inside a single, continuously-deformed basin anchored at the one point
# (alpha=0) known to be exactly optimal for (B) -- rather than gambling on
# a disconnected random start landing somewhere good.

def _continuation_constraints(a_of, theta_B, theta_C, sig, coshmu0):
    """Plain linear constraint, with sig auto-detected fresh at each stage
    from the current iterate (see _sign_from_a) rather than fixed once at
    alpha=0 or left sign-free via a squared/disjunctive constraint -- the
    latter is mathematically equivalent but numerically much harder for
    SLSQP's local QP step (confirmed empirically: it turned a previously-
    solved case, n=13 on the widened multiband I1, into a reported
    infeasibility). Since the schedule only takes small steps in mu, the
    sign detected at the start of a stage essentially always still matches
    by the stage's end; nothing here fixes it in advance for the whole
    trajectory the way the original hard-coded default did."""
    constraints = []
    for th in theta_B:
        def fun(x, th=th):
            return x[-1] - q1_abs_sq(a_of(x[:-1]), th)
        constraints.append({"type": "ineq", "fun": fun})
    for th, s in zip(theta_C, sig):
        def fun(x, th=th, s=s):
            return s * kappa_B(a_of(x[:-1]), th) - coshmu0
        constraints.append({"type": "ineq", "fun": fun})
    return constraints


def design_via_continuation(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                             n_steps: int = 12, max_bisections: int = 5,
                             gamma_bound: float = 0.9995, n_grid_B: int = 400, n_grid_C: int = 400,
                             maxiter: int = 400) -> LayerDesignResult:
    """Homotopy solver: start at alphas=0 and ramp the required stop-band
    depth mu up from 0 to mu0 in n_steps stages, warm-starting each stage
    from the previous stage's alphas (see section docstring above). If a
    stage fails to reach a verified point, the step is halved (up to
    max_bisections times) before giving up -- makes the schedule robust to
    a coarse n_steps without needing it finer everywhere.

    No sign is fixed for any J0 component in advance: at the start of each
    stage, the sign is auto-detected from the *current* iterate (see
    _sign_from_a) and used to build a plain linear constraint for that
    stage's solve, rather than either hard-coding a sign upfront from
    alpha=0's cos(n*theta) or using a sign-free squared constraint (the
    latter tried and reverted -- see _continuation_constraints).

    Returns a LayerDesignResult; status is "optimal" if mu0 was fully
    reached, "partial" if the schedule stalled at some smaller achieved
    mu (still a valid, realizable design -- Prop. 5.4a: any mu0>0 is fine,
    N does the rest), or "infeasible" if not even the first tiny step
    succeeded.
    """
    from .forward import a_from_alphas

    theta_B = [np.linspace(g, d, n_grid_B) for g, d in J1]
    theta_C = [np.linspace(al, be, n_grid_C) for al, be in J0]

    def a_of(gamma_free):
        return a_from_alphas(_free_to_alphas(gamma_free))

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    bounds = [(-gamma_bound, gamma_bound)] * n + [(1.0, None)]

    def verify(gamma_free, u, sig, coshmu0_k, tol=1e-6):
        a = a_of(gamma_free)
        for th in theta_B:
            if np.max(q1_abs_sq(a, th)) > u + tol:
                return False
        for th, s in zip(theta_C, sig):
            if np.min(s * kappa_B(a, th)) < coshmu0_k - tol:
                return False
        return True

    def try_stage(gamma_start, mu_k):
        coshmu0_k = np.cosh(mu_k)
        sig = _sign_from_a(a_of(gamma_start), theta_C)
        constraints = _continuation_constraints(a_of, theta_B, theta_C, sig, coshmu0_k)
        Gmax = max((np.max(q1_abs_sq(a_of(gamma_start), th)) for th in theta_B), default=1.0)
        u0 = max(1.0 + 1e-9, Gmax)
        x0 = np.concatenate([gamma_start, [u0]])
        res = minimize(objective, x0, jac=objective_grad, constraints=constraints, bounds=bounds,
                        method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            return None
        gamma_try, u_try = res.x[:-1], float(res.x[-1])
        if verify(gamma_try, u_try, sig, coshmu0_k):
            return gamma_try, u_try
        return None

    gamma_free = np.zeros(n)
    mu_prev, u_prev = 0.0, 1.0
    schedule = list(np.linspace(0.0, mu0, n_steps + 1)[1:]) if n_steps > 0 else [mu0]
    idx = 0
    while idx < len(schedule):
        mu_target = schedule[idx]
        lo, hi = mu_prev, mu_target
        outcome = try_stage(gamma_free, hi)
        bisections = 0
        while outcome is None and bisections < max_bisections:
            hi = (lo + hi) / 2.0
            outcome = try_stage(gamma_free, hi)
            bisections += 1
        if outcome is None:
            break
        gamma_free, u_prev = outcome
        mu_prev = hi
        if hi >= mu_target - 1e-12:
            idx += 1
        # else: only got partway via bisection -- retry from here toward
        # the same (still unmet) schedule target on the next iteration.

    if mu_prev <= 0.0:
        return LayerDesignResult(n=n, mu0=mu0, status="infeasible")

    alphas = _free_to_alphas(gamma_free)
    a = a_of(gamma_free)
    delta1 = float(np.sqrt(max(u_prev, 0.0)) - 1.0)
    status = "optimal" if mu_prev >= mu0 - 1e-9 else "partial"

    impedances = np.empty(n + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(alphas):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    # sign is not imposed -- read it off the solution, per J0 component
    sig = tuple(1 if np.mean(kappa_B(a, np.linspace(al, be, 4000))) > 0 else -1 for al, be in J0)
    mu_min = np.inf
    for (al, be), s in zip(J0, sig):
        kap = s * kappa_B(a, np.linspace(al, be, 4000))
        mu_min = min(mu_min, np.arccosh(max(float(np.min(kap)), 1.0)))

    return LayerDesignResult(n=n, mu0=mu0, status=status, sigma=sig, alphas=alphas, a=a,
                              impedances=impedances, u=u_prev, delta1=delta1,
                              verified=(status == "optimal"),
                              achieved_mu=mu_min if J0 else float("inf"))


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

    # (C) is typically *tight* at a genuine constrained optimum (kappa_B
    # touches cosh(mu0) exactly, same as (B) touches u) -- an exact
    # >=1.0 comparison then just measures which side of floating-point
    # noise the run happened to land on. A small tolerance avoids treating
    # a ~1e-10 shortfall as total failure; achieved_mu uses
    # arccosh(clip(., 1, None)) so a true (not just noise-level) shortfall
    # still reports 0 rather than raising on an out-of-domain arccosh.
    kappa_tol = 1e-6
    all_ok, mu_min = True, np.inf
    for (alpha, beta), sig in zip(J0, sigma):
        kap_min = float(np.min(sig * kappa_B(a, np.linspace(alpha, beta, 4000))))
        mu_min = min(mu_min, np.arccosh(max(kap_min, 1.0)))
        if kap_min < 1.0 - kappa_tol:
            all_ok = False
    achieved_mu = mu_min if J0 else float("inf")
    C_satisfied = all_ok and achieved_mu >= mu0 - 1e-6

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
