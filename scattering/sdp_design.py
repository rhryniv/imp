"""SDP filter synthesis (Section 6 of the paper). Optimization only -- all
transfer-matrix / factorization physics is delegated to forward.py and
inverse.py.

CONVENTION, pinned down against the manuscript (eq. 6.5): for a coefficient
vector f=(f_0,...,f_n), Q(theta) = f_0 + 2*sum_{l>=1} f_l cos(l*theta) --
note the factor 2 on every harmonic beyond the constant term. poly_sdp.py's
own Gram/Markov-Lukacs machinery works in the *undoubled* Chebyshev-basis
convention (p(x) = sum_l c_l T_l(x), no factor of 2 anywhere), so any
f/autocorrelation-style vector must be rescaled via
_cheb_from_cosine_series (c_0=f_0, c_l=2*f_l for l>=1) before being handed
to interval_nonneg_constraints. Skipping this rescaling is a confirmed,
numerically verified bug (see below) -- not a stylistic choice.

Three SDP-adjacent tools are implemented:

  design_sdp_magnitude -- THE PRIMARY bound-computation method (manuscript
                          Sec. 6.2, eq. 6.6). Variables are f in R^{n+1}
                          and delta only -- no matrix variable, no vector
                          a, no rank relaxation, and no sign pattern to
                          enumerate (constraint (C) is replaced by its
                          sign-free consequence (C') Q>=cosh^2(mu0), which
                          follows from kappa^2<=Q and is solved once per
                          degree). Every feasible f is realisable by
                          Fejer-Riesz factorization of Q itself (not a
                          lift), giving delta_mag <= delta_n* rigorously
                          and a genuine warm-start block -- see
                          _factorize_magnitude_f.

  design_filter        -- the autocorrelation-domain relaxation (Remark
                          5.10): variables are the autocorrelation
                          coefficients c_l, constraints (A) G>=1 and
                          (B) G<=u on J1, objective min u. Constraint (C)
                          is not part of it, only checked post hoc via
                          inverse.alphas_from_a + forward.kappa_B.

  design_filter_full   -- the full lifted SDP (manuscript Remark 6.3 /
                          Appendix B.4: kept only as an OPTIONAL
                          comparison against design_sdp_magnitude, not the
                          primary path): constraints (A)+(B)+(C)+(D)
                          jointly, via the rank relaxation A ~ a a^T. The
                          relaxation is essentially *never* tight in
                          practice -- its raw output is only a warm start,
                          refined by a local polish before being returned.

BUG FOUND AND FIXED (this refactor, in two stages): design_filter and
design_filter_full's _autocorr_from_A fed the *undoubled* autocorrelation
f_l straight into interval_nonneg_constraints, silently treating Q's
"f_0 + 2*sum f_l cos" expansion as if it were the unscaled Chebyshev
expansion "sum c_l T_l(x)". Verified numerically on a concrete test
vector: the resulting constraint differs from the true Q by 0.35 (not a
rounding artifact). This plausibly root-causes the "(A) is violated 100%
of the time" empirical finding below -- the SDP was optimizing a
different, incorrect problem the whole time. forward.py itself was never
affected (q1_abs_sq evaluates |sum a_m e^{im theta}|^2 directly, no
coefficient-doubling step exists to get wrong), so every *achieved*
design from design_via_layers/design_via_continuation reported prior to
this fix remains valid; only the SDP relaxations' own bound/warm-start
quality was compromised. Fixed via _cheb_from_cosine_series, first
applied only in design_sdp_magnitude -- a follow-up audit (prompted by
building the lift/magnitude "conjunction" comparison, Sec. 7 of the
spec, which needs design_filter_full's *raw* relaxed value to be
trustworthy) found the fix had NOT actually been carried over to
design_filter/design_filter_full despite this docstring's earlier claim
that it had; _cheb_from_cosine_series is now applied there too. This
means sdp_lower_bound's returned value was not a valid lower bound
before this second fix (it read off the same mis-scaled raw SDP
objective) -- design_filter_full's *polished* .a/.delta were never
affected, since _polish always re-verifies against the true pointwise
Q/kappa_B formulas regardless of the raw relaxation's own internal
scaling.

IMPORTANT, found empirically (see commit history / prior analysis): for
*any* J0, J1, the unconstrained (A)+(B)+(D) problem's global optimum is
always the trivial filter a=(1,0,...,0) (u=1, delta=0, zero stop-band
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
from .forward import kappa_B, transmission_TN, q1_abs_sq, forward_with_grad, grad_free_vars, a_from_alphas
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
# design_sdp_magnitude: the primary bound-computation SDP (manuscript 6.2)
# --------------------------------------------------------------------------

def _cheb_from_cosine_series(g, n: int):
    """g=(g_0,...,g_n) represents G(theta) = g_0 + 2*sum_{l>=1} g_l cos(l
    theta) (manuscript eq. 6.5's own convention). Returns the Chebyshev-
    basis coefficients (c_l with G = sum_l c_l T_l(x), x=cos theta) that
    poly_sdp.cheb_to_mono_matrix/interval_nonneg_constraints expect:
    c_0=g_0, c_l=2*g_l for l>=1. Accepts either a numpy array or a cvxpy
    Expression. See this module's docstring for why this rescaling is a
    bug fix, not a style choice."""
    scale = np.concatenate([[1.0], 2.0 * np.ones(n)])
    if isinstance(g, cp.Expression):
        return cp.multiply(scale, g)
    return scale * np.asarray(g, dtype=float)


@dataclass
class MagnitudeSDPResult:
    n: int
    mu0: float
    status: str                            # "optimal" / "infeasible" / "solver_failure" -- kept distinct
    f: np.ndarray | None = None            # feasible Q-autocorrelation, Q = f_0 + 2*sum_{l>=1} f_l cos(l*theta)
    delta_mag: float | None = None         # valid lower bound on the true delta_n*
    problem: cp.Problem | None = None
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def design_sdp_magnitude(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                          solver: str = "CLARABEL", **solver_kwargs) -> MagnitudeSDPResult:
    """Magnitude SDP (manuscript Sec. 6.2, eq. 6.6): THE PRIMARY bound
    method, replacing the lifted SDP as the default (design_filter_full is
    kept only as an optional comparison, see its own docstring).

    Variables: f in R^{n+1} (Q's own coefficients) and delta in R. No
    matrix variable, no vector a, no rank relaxation. Constraint (C) is
    replaced by its sign-free consequence (C') Q>=cosh^2(mu0) on I0, which
    follows from kappa^2<=Q and is strictly weaker -- a large modulus does
    not force a large real part -- so (unlike design_filter_full) this
    problem needs no sign pattern sigma and is solved once per degree, not
    once per 2^len(J0) sign combination. All four constraints are linear
    in (f, delta), so the whole problem is fully SDP-representable via the
    Gram/Markov-Lukacs machinery of poly_sdp.py.

    status is kept as a distinct three-way outcome ("optimal" /
    "infeasible" / "solver_failure"): infeasible is mathematically
    meaningful here (see lower_bounds' docstring -- it proves no n-layer
    block meets the specification, since the true feasible set is
    contained in this relaxed one), solver_failure is not.

    Returns delta_mag <= delta_n* (a genuine lower bound: every f arising
    from a true feasible design also satisfies (A),(B),(C'),(D)), and the
    feasible f itself. Every feasible f is realisable -- see
    _factorize_magnitude_f -- unlike the lifted SDP's raw output, which
    generally is not.
    """
    T = cheb_to_mono_matrix(n)
    f = cp.Variable(n + 1)
    delta = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    scale = np.concatenate([[1.0], 2.0 * np.ones(n)])

    constraints = [delta >= 0.0, cp.sum(cp.multiply(scale, f)) == 1.0]  # (D'): Q(0) = 1

    # (A): Q - 1 >= 0 on all of R (x in [-1, 1])
    consA, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(f - e0, n), n, -1.0, 1.0)
    constraints += consA

    # (B): delta - (Q - 1) >= 0 on each I1 component
    for gamma, delta_hi in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta_hi)
        g_B = (delta + 1.0) * e0 - f
        consB, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(g_B, n), n, xlo, xhi)
        constraints += consB

    # (C'): Q - cosh^2(mu0) >= 0 on each I0 component (sign-free, no sigma)
    coshmu0_sq = float(np.cosh(mu0) ** 2)
    for u, v in J0:
        xlo, xhi = theta_interval_to_x(u, v)
        g_C = f - coshmu0_sq * e0
        consC, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(g_C, n), n, xlo, xhi)
        constraints += consC

    problem = cp.Problem(cp.Minimize(delta), constraints)
    try:
        problem.solve(solver=solver, **solver_kwargs)
    except cp.error.SolverError:
        return MagnitudeSDPResult(n=n, mu0=mu0, status="solver_failure", J0=list(J0), J1=list(J1))

    if problem.status in ("infeasible", "infeasible_inaccurate"):
        return MagnitudeSDPResult(n=n, mu0=mu0, status="infeasible", problem=problem,
                                   J0=list(J0), J1=list(J1))
    if problem.status not in ("optimal", "optimal_inaccurate") or f.value is None:
        return MagnitudeSDPResult(n=n, mu0=mu0, status="solver_failure", problem=problem,
                                   J0=list(J0), J1=list(J1))

    return MagnitudeSDPResult(n=n, mu0=mu0, status="optimal", f=np.asarray(f.value, dtype=float),
                               delta_mag=float(delta.value), problem=problem, J0=list(J0), J1=list(J1))


def _factorize_magnitude_f(f: np.ndarray, n_grid: int = 20000) -> tuple[np.ndarray, np.ndarray, object]:
    """Recover an actual n-layer block from a feasible f of the magnitude
    SDP (spec Sec. 3 / manuscript Cor. 4.1): every feasible f IS
    realisable -- factor Q = f_0 + 2*sum f_l cos(l*theta) directly via
    Fejer-Riesz (inverse.fejer_riesz already expects exactly this "G =
    c_0 + 2*sum c_l cos(l*theta)" convention, so f is passed unchanged, no
    adaptation needed), sign-fix so sum(a)=+1 (constraint D, i.e.
    p_1(1)=+1), then run the existing layer-stripping routine.

    Floors any negative Q-1 dip from solver imprecision before factoring
    (same reasoning as inverse.p2_from_a): Fejer-Riesz needs Q-1 >= 0
    *exactly*, and even a tiny negative dip makes root-finding locally
    inconsistent right there.

    Returns (alphas, a, info) with info the RealizabilityInfo from
    inverse.alphas_from_a. Use _seed_gap_quality to check whether the
    resulting block is a *useful* warm start, not just a valid one.
    """
    f = np.asarray(f, dtype=float)
    n = len(f) - 1
    theta = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    if n > 0:
        Q = f[0] + 2.0 * (np.cos(np.outer(theta, np.arange(1, n + 1))) @ f[1:])
    else:
        Q = np.full(n_grid, f[0])
    Q_min = float(np.min(Q))
    c = f.copy()
    if Q_min < 1.0:
        c[0] += (1.0 - Q_min) + 1e-12
    a = fejer_riesz(c)
    if np.sum(a) < 0:
        a = -a
    alphas, info = alphas_from_a(a)
    return alphas, a, info


def _seed_gap_quality(a: np.ndarray, J0: Sequence[Interval]) -> float:
    """min_{I0} |kappa_B(theta)| for a factorised magnitude-SDP seed --
    decides whether it is a *useful* warm start, not just a realisable
    one (spec Sec. 3: 'It may be near 1 (no gap), in which case the seed
    is realisable but useless -- that is a legitimate finding.')."""
    if not J0:
        return float("inf")
    vals = [np.min(np.abs(kappa_B(a, np.linspace(u, v, 4000)))) for u, v in J0]
    return float(min(vals))


# --------------------------------------------------------------------------
# design_filter: autocorrelation-domain SDP, (A)+(B)+(D) only
# --------------------------------------------------------------------------

@dataclass
class DesignResult:
    n: int
    status: str
    c: np.ndarray | None = None            # autocorrelation coeffs of G, length n+1
    a: np.ndarray | None = None            # a Fejer-Riesz factor of c (min-phase)
    u: float | None = None                 # = 1 + delta (constraint B: Q <= u on I1)
    delta: float | None = None             # = max_{I1}(Q-1) directly (manuscript eq. 6.6), NOT sqrt(u)-1
    C_satisfied: bool = False              # post-hoc check of constraint (C), any sign pattern
    achieved_mu: float = 0.0               # min_{J0} arccosh(|kappa_B|), 0 if C fails somewhere
    problem: cp.Problem | None = None
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def design_filter(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                   solver: str = "CLARABEL", **solver_kwargs) -> DesignResult:
    """Autocorrelation-domain SDP: minimize u s.t. G>=1 on [-1,1], G<=u on J1,
    G(0)=1 (constraint D, since G(0)=q~_1(0)^2 and D requires q~_1(0)=1).
    Constraint (C) is checked post hoc on the Fejer-Riesz factor of c.

    c is the RAW (undoubled) autocorrelation, G(theta) = c_0 + 2*sum_{l>=1}
    c_l cos(l*theta) -- same convention as design_sdp_magnitude's own f, and
    what inverse.fejer_riesz itself expects (see this module's docstring's
    "BUG FOUND AND FIXED" note: c must be rescaled via
    _cheb_from_cosine_series before poly_sdp's Chebyshev-basis machinery,
    which has no notion of the factor of 2, ever sees it)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    T = cheb_to_mono_matrix(n)
    c = cp.Variable(n + 1)
    u = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    scale = np.concatenate([[1.0], 2.0 * np.ones(n)])

    constraints = [u >= 1.0, cp.sum(cp.multiply(scale, c)) == 1.0]  # G(0) = 1
    consA, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(c - e0, n), n, -1.0, 1.0)
    constraints += consA
    for gamma, delta in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta)
        consB, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(u * e0 - c, n), n, xlo, xhi)
        constraints += consB

    problem = cp.Problem(cp.Minimize(u), constraints)
    problem.solve(solver=solver, **solver_kwargs)

    result = DesignResult(n=n, status=problem.status, problem=problem, J0=list(J0), J1=list(J1))
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return result

    result.c = c.value
    result.u = float(u.value)
    result.delta = float(max(result.u - 1.0, 0.0))  # u IS max_{I1}(Q) at optimality (B tight)
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
    delta: float | None = None             # = max_{I1}(Q-1) directly (manuscript eq. 6.6), NOT sqrt(u)-1
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


def _solve_full_for_sigma(n, J0, J1, mu0, sigma, T, solver, solver_kwargs, add_magnitude_C_prime=False):
    a = cp.Variable(n + 1)
    A = cp.Variable((n + 1, n + 1), symmetric=True)
    u = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    lift = cp.bmat([[A, cp.reshape(a, (n + 1, 1), order="C")],
                    [cp.reshape(a, (1, n + 1), order="C"), np.array([[1.0]])]])
    constraints = [lift >> 0, u >= 1.0, cp.sum(a) == 1.0]

    # c = _autocorr_from_A(A, n) is the RAW (undoubled) autocorrelation of A
    # (trace(A)=f_0, l-th diagonal band sum=f_l -- verified directly: for a
    # rank-1 A=a a^T this reproduces forward.autocorr(a) exactly), so it
    # needs the same _cheb_from_cosine_series rescaling as design_filter's
    # own c and design_sdp_magnitude's f before poly_sdp's Chebyshev-basis
    # machinery (which has no notion of the factor of 2) sees it -- this
    # was the confirmed bug described in the module docstring, still
    # present here despite that docstring's claim it was fixed everywhere.
    c = _autocorr_from_A(A, n)
    consA, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(c - e0, n), n, -1.0, 1.0)
    constraints += consA
    for gamma, delta in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta)
        consB, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(u * e0 - c, n), n, xlo, xhi)
        constraints += consB

    coshmu0 = float(np.cosh(mu0))
    for (alpha, beta), sig in zip(J0, sigma):
        xlo, xhi = theta_interval_to_x(alpha, beta)
        stop_cheb = sig * a - coshmu0 * e0
        consC, _ = interval_nonneg_constraints(T @ stop_cheb, n, xlo, xhi)
        constraints += consC

    if add_magnitude_C_prime:
        # The "conjunction" (spec Sec. 7): also impose the magnitude SDP's
        # own sign-free (C') Q_A >= cosh^2(mu0) on each J0 component, using
        # the SAME c as constraints (A)/(B) above. PROVABLY redundant given
        # the lift is already here, for any n/J0/J1/sigma/mu0, not just
        # empirically: the Schur complement lift>>0 is equivalent to
        # A - a a^T being PSD, so for the complex vector w(theta) =
        # (1, e^{i theta}, ..., e^{i n theta}), w^H A w >= w^H (a a^T) w =
        # |a^T w|^2 = |sum_m a_m e^{im theta}|^2 -- and w^H A w is exactly
        # Q_A(theta) (the identity _autocorr_from_A's diagonal-band-sum
        # construction encodes). Combined with the trivial |z|^2 >= Re(z)^2,
        # this gives Q_A(theta) >= kappa_a(theta)^2 for every theta, for
        # ANY PSD-feasible (not just rank-1) A -- so on J0, exact (C)
        # (sigma*kappa_a >= cosh(mu0) > 0) already forces
        # Q_A >= kappa_a^2 >= cosh^2(mu0), i.e. (C'), automatically. Kept
        # anyway (rather than skipped) so design_conjunction's own
        # numerical value is an independent check of this argument, not an
        # assumption baked into the code.
        coshmu0_sq = coshmu0 ** 2
        for lo, hi in J0:
            xlo, xhi = theta_interval_to_x(lo, hi)
            g_Cprime = c - coshmu0_sq * e0
            consCprime, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(g_Cprime, n), n, xlo, xhi)
            constraints += consCprime

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
        result.delta = float(max(u_pol - 1.0, 0.0))
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
    # verified ones: a smaller delta that turns out unrealizable is not
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
    return float(max(best_u - 1.0, 0.0))


def sdp_lower_bound_conjunction(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                                 solver: str = "CLARABEL", **solver_kwargs) -> float | None:
    """The "conjunction" bound (spec Sec. 7, optional comparison): same as
    sdp_lower_bound, but each sign pattern's lifted SDP also carries the
    magnitude SDP's own sign-free (C') constraint alongside the lift's
    exact (C) -- see _solve_full_for_sigma's add_magnitude_C_prime
    docstring for the argument that this is provably redundant (Q_A >=
    kappa_a^2 already follows from the lift's PSD structure alone, for any
    relaxed A, not just rank-1), so this is expected to equal
    sdp_lower_bound's own value up to solver noise, not improve on it.
    Kept as a genuine, separately-solved SDP rather than skipped, so that
    expectation is checked numerically, not assumed."""
    T = cheb_to_mono_matrix(n)
    best_u = None
    for sig in itertools.product((1, -1), repeat=len(J0)):
        res = _solve_full_for_sigma(n, J0, J1, mu0, sig, T, solver, solver_kwargs, add_magnitude_C_prime=True)
        if res.problem is not None and res.problem.value is not None:
            u_lb = float(res.problem.value)
            if best_u is None or u_lb < best_u:
                best_u = u_lb
    if best_u is None:
        return None
    return float(max(best_u - 1.0, 0.0))


def compare_relaxations(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                         solver: str = "CLARABEL", **solver_kwargs) -> dict:
    """One n's worth of the spec Sec. 7 "optional comparison run": delta_mag
    (design_sdp_magnitude's own bound), delta_lift (sdp_lower_bound, the
    plain lifted SDP's raw relaxed bound), delta_both (sdp_lower_bound_
    conjunction, lift + magnitude C'). Neither delta_mag nor delta_lift
    dominates the other a priori (spec's own wording, and confirmed by
    Experiment 1/2: delta_mag was sometimes far below delta_achieved,
    sometimes closer). delta_both, though, has a provable relationship to
    delta_lift specifically -- see sdp_lower_bound_conjunction's docstring
    for why it should equal delta_lift, not improve on it -- which this
    function's numbers should confirm rather than assume."""
    mag_res = design_sdp_magnitude(n, J0, J1, mu0, solver=solver, **solver_kwargs)
    delta_mag = mag_res.delta_mag if mag_res.status == "optimal" else None
    delta_lift = sdp_lower_bound(n, J0, J1, mu0, solver=solver, **solver_kwargs)
    delta_both = sdp_lower_bound_conjunction(n, J0, J1, mu0, solver=solver, **solver_kwargs)
    return {"n": n, "delta_mag": delta_mag, "delta_lift": delta_lift, "delta_both": delta_both}


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


# --------------------------------------------------------------------------
# design_direct: the two-phase synthesis algorithm (manuscript Sec. 6.3)
# --------------------------------------------------------------------------
#
# Phase 1 (open the gap) and Phase 2 (flatten the pass band) replace the
# single-stage search of design_via_layers/design_via_continuation with the
# manuscript's explicit two-phase structure, using EXACT gradients
# (forward.forward_with_grad) throughout instead of SLSQP's own internal
# finite-difference approximation of the constraint Jacobians. delta here
# IS the epigraph variable t = max(Q-1) directly (manuscript eq. 6.6) --
# not the old sqrt(u)-1 / (1+delta)^2 bookkeeping (see module docstring).

def _phase1_open_gap(n: int, J0: Sequence[Interval], sigma: tuple, mu0: float,
                      gamma_bound: float = 0.9995, n_grid_C: int = 200,
                      n_steps: int = 20, max_bisections: int = 4, maxiter: int = 400):
    """Phase 1: starting from alpha=0 (Q=1 identically, kappa=cos(n*theta),
    no gap at all), ramp the REQUIRED depth target from 0 to mu0 in
    stages, for a GIVEN, fixed sign pattern, with the pass-band condition
    entirely inactive.

    Read the manuscript's "maximise the gap depth... until d>=cosh(mu0)"
    as an ascent that STOPS at the threshold, not a genuine unconstrained
    maximisation: a plain "maximise depth" objective is unbounded above
    (larger alpha gives larger kappa excursions, with no ceiling except
    the box bound) -- confirmed empirically, an earlier epigraph-maximise
    version drove alpha to its bounds and produced delta~1e26, physically
    meaningless. Each stage is instead a feasibility problem at the
    current target, regularised by a *small* quadratic pull toward
    gamma_free=0 (weight tiny relative to the constraint) -- two things
    were needed together, found in sequence: (1) starting exactly at
    alpha=0 leaves the constraint gradient exactly zero there (Q-1=|q2|^2
    has its global minimum, value 0, exactly at alpha=0, so by basic
    calculus its gradient vanishes there -- and the same degeneracy
    propagates into d(kappa)/dalpha_j; verified directly: at alpha=0
    every A_j is diagonal, so the L_j K p^{(j)} gradient formula
    evaluates to exactly zero). The old design_via_continuation never hit
    this because it relied on SLSQP's own internal finite-difference
    Jacobian, which numerically smears that exact zero into a small
    nonzero estimate and escapes by luck; exact gradients remove that
    luck, so a tiny fixed perturbation is used as the actual starting
    point instead. But (2) a perturbed start with a genuinely *flat*
    objective (tried first) then wanders unboundedly once past the
    degeneracy, since any feasible point is equally "optimal" with no
    objective pressure at all -- confirmed empirically (depth ran off to
    ~1.75e9 against a target of ~1.001). The small regulariser (much
    weaker than the outright-fighting one from the exact-zero-start
    attempt, since the starting point is now already displaced) supplies
    just enough pressure to stop growing once feasible, without being
    strong enough to prevent reaching feasibility in the first place.
    Warm-started from the previous stage; the step is halved (up to
    max_bisections times) if a stage fails to verify -- the same robust
    ramping pattern already validated in (the now-superseded)
    design_via_continuation.

    Returns (alphas, achieved_depth) at the last stage successfully
    reached; achieved_depth may fall short of cosh(mu0) if the schedule
    stalled completely (feasibility against the target and cross-sigma
    comparison are design_direct's job).
    """
    theta_C = [np.linspace(u, v, n_grid_C) for u, v in J0]
    bounds = [(-gamma_bound, gamma_bound)] * n
    reg = 1e-4

    def objective(gamma_free):
        return reg * float(np.sum(gamma_free ** 2))

    def objective_grad(gamma_free):
        return 2.0 * reg * gamma_free

    def make_constraints(coshmu_k):
        constraints = []
        for th, sig in zip(theta_C, sigma):
            def fun(gamma_free, th=th, sig=sig):
                alphas = _free_to_alphas(gamma_free)
                kap = forward_with_grad(alphas, th)["kappa"]
                return sig * kap - coshmu_k

            def jac(gamma_free, th=th, sig=sig):
                alphas = _free_to_alphas(gamma_free)
                res = forward_with_grad(alphas, th)
                return grad_free_vars(sig * res["dkappa"]).T

            constraints.append({"type": "ineq", "fun": fun, "jac": jac})
        return constraints

    def verify(gamma_free, coshmu_k, tol=1e-4):
        alphas = _free_to_alphas(gamma_free)
        for th, sig in zip(theta_C, sigma):
            if np.min(sig * forward_with_grad(alphas, th)["kappa"]) < coshmu_k - tol:
                return False
        return True

    def try_stage(gamma_start, mu_k):
        coshmu_k = np.cosh(mu_k)
        res = minimize(objective, gamma_start, jac=objective_grad, constraints=make_constraints(coshmu_k),
                        bounds=bounds, method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            return None
        return res.x if verify(res.x, coshmu_k) else None

    def depth_of(gamma_free):
        alphas = _free_to_alphas(gamma_free)
        return min(np.min(sig * forward_with_grad(alphas, th)["kappa"]) for th, sig in zip(theta_C, sigma))

    def one_attempt(gamma0):
        # Try the FULL target directly first, in one SLSQP solve --
        # confirmed empirically to work reliably from a small perturbation.
        # Counter-intuitively, ramping in *small* steps starting right next
        # to the degenerate point is *harder* to converge precisely (SLSQP
        # repeatedly hit its iteration limit chasing tiny early targets like
        # cosh(mu0/n_steps), converging to within 2e-5 of feasible and no
        # closer even at 2000 iterations, while the full target converged
        # cleanly in under 400) -- conditioning right next to alpha=0 is
        # worse for a small ask than a moderate one, so ramping is used
        # only as a fallback, not the default path.
        direct = try_stage(gamma0, mu0)
        if direct is not None:
            return direct, depth_of(direct)

        gamma_free = gamma0
        mu_prev = 0.0
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
            gamma_free, mu_prev = outcome, hi
            if hi >= mu_target - 1e-12:
                idx += 1
        return gamma_free, depth_of(gamma_free)

    # Q-1=|q2|^2 >= 0 has its global minimum (value 0) exactly at alpha=0,
    # so d(kappa)/dalpha_j is *exactly* zero there too (verified: at
    # alpha=0 every A_j is diagonal, so the L_j K p^{(j)} gradient formula
    # evaluates to exactly zero) -- a genuine degeneracy, not a bug. The
    # earlier design_via_continuation never hit this because it relied on
    # SLSQP's own internal finite-difference Jacobian, which numerically
    # smears that exact zero into a small nonzero estimate and escapes by
    # luck; exact analytic gradients remove that luck, so SLSQP gets stuck
    # exactly at the (infeasible) starting point when seeded there. Start
    # from a small perturbation instead -- any nonzero point escapes the
    # degeneracy, since it is confined to the single point alpha=0.
    #
    # A single fixed perturbation isn't always enough, though (confirmed:
    # the manuscript's own worked example, n=5/mu0=1.0 -- a substantially
    # larger depth target than the mu0=0.05 cases this was first tuned on
    # -- stalled partway for its one reachable sign pattern, short of the
    # target even with a long ramp schedule). Try several perturbation
    # scales/seeds and keep whichever reaches the largest depth.
    rng = np.random.default_rng(0)
    best = None
    for scale in (1e-3, 1e-2, 0.05, 0.1, 0.2):
        gamma0 = rng.normal(0.0, scale, n)
        gamma_result, depth = one_attempt(gamma0)
        if best is None or depth > best[1]:
            best = (gamma_result, depth)
        if depth >= np.cosh(mu0) - 1e-6:
            break  # already reached the target, no need to try further scales

    gamma_free, depth = best
    return _free_to_alphas(gamma_free), depth


def _phase2_flatten(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float, sigma: tuple,
                     seed_pool: dict[str, np.ndarray], gamma_bound: float = 0.9995,
                     n_grid_B: int = 200, n_grid_C: int = 200, maxiter: int = 400):
    """Phase 2 (manuscript eq. direct): from the Phase-1 point and the rest
    of the seed pool, solve
        min_{alpha,t} t   s.t.  Q(theta)-1 <= t          (theta in I1),
                              sigma_j*kappa(theta) >= cosh(mu0)  (theta in I0)
    by SQP with exact gradients. t IS delta directly.

    Grid discretisation is harmless here (unlike the SDP): (A) and (D)
    hold identically in the alpha-chart (forward_reconstruct/
    forward_with_grad), so sampling can only cost margin in (B)/(C), never
    realizability -- see certify_exact() for the grid-independent final numbers.

    Run once per named seed in seed_pool ({name: gamma0}); returns
    (alphas, delta, per_start) with per_start = {name: achieved delta or
    None}, computed from this single pass -- NOT a separate diagnostic
    re-run (an earlier version re-solved every seed a second time purely
    to populate per_start, roughly doubling total runtime for no benefit).
    """
    theta_B = [np.linspace(g, d, n_grid_B) for g, d in J1]
    theta_C = [np.linspace(u, v, n_grid_C) for u, v in J0]
    coshmu0 = np.cosh(mu0)

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    constraints = []
    for th in theta_B:
        def fun(x, th=th):
            alphas = _free_to_alphas(x[:-1])
            Q = forward_with_grad(alphas, th)["Q"]
            return x[-1] - (Q - 1.0)

        def jac(x, th=th):
            alphas = _free_to_alphas(x[:-1])
            res = forward_with_grad(alphas, th)
            dQ_free = grad_free_vars(res["dQ"])
            Jm = np.zeros((len(th), len(x)))
            Jm[:, :-1] = -dQ_free.T
            Jm[:, -1] = 1.0
            return Jm

        constraints.append({"type": "ineq", "fun": fun, "jac": jac})

    for th, sig in zip(theta_C, sigma):
        def fun(x, th=th, sig=sig):
            alphas = _free_to_alphas(x[:-1])
            kap = forward_with_grad(alphas, th)["kappa"]
            return sig * kap - coshmu0

        def jac(x, th=th, sig=sig):
            alphas = _free_to_alphas(x[:-1])
            res = forward_with_grad(alphas, th)
            dkap_free = grad_free_vars(sig * res["dkappa"])
            Jm = np.zeros((len(th), len(x)))
            Jm[:, :-1] = dkap_free.T
            return Jm

        constraints.append({"type": "ineq", "fun": fun, "jac": jac})

    bounds = [(-gamma_bound, gamma_bound)] * n + [(0.0, None)]

    def verify(gamma_free, t, tol=1e-6):
        # A genuinely useful filter has delta bounded by a modest constant
        # (the manuscript's own worked examples: 1e-5 to a handful of
        # units, never remotely close to this). SLSQP occasionally
        # converges a badly-scaled seed to a "locally stationary" point
        # that technically satisfies the pointwise inequalities on the
        # grid with an enormous t -- confirmed empirically (delta~4.7e4
        # from a seed pool where every other member failed outright).
        # Reject rather than report as a successful design.
        if t > 100.0:
            return False
        alphas = _free_to_alphas(gamma_free)
        for th in theta_B:
            Q = forward_with_grad(alphas, th)["Q"]
            if np.max(Q - 1.0) > t + tol:
                return False
        for th, sig in zip(theta_C, sigma):
            kap = forward_with_grad(alphas, th)["kappa"]
            if np.min(sig * kap) < coshmu0 - tol:
                return False
        return True

    best = None
    per_start: dict[str, float | None] = {}
    for name, gamma0 in seed_pool.items():
        gamma0 = np.clip(np.asarray(gamma0, dtype=float), -gamma_bound, gamma_bound)
        alphas0 = _free_to_alphas(gamma0)
        Qmax0 = max((np.max(forward_with_grad(alphas0, th)["Q"]) for th in theta_B), default=1.0)
        t0 = max(0.0, Qmax0 - 1.0)
        x0 = np.concatenate([gamma0, [t0]])
        res = minimize(objective, x0, jac=objective_grad, constraints=constraints, bounds=bounds,
                        method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            per_start[name] = None
            continue
        gamma_try, t_try = res.x[:-1], float(res.x[-1])
        if verify(gamma_try, t_try):
            per_start[name] = t_try
            if best is None or t_try < best[1]:
                best = (gamma_try, t_try, name)
        else:
            per_start[name] = None

    if best is None:
        return None, per_start
    gamma_free, delta, winning_name = best
    return (_free_to_alphas(gamma_free), delta, winning_name), per_start


@dataclass
class DirectDesignResult:
    n: int
    mu0: float
    status: str                            # "optimal" / "phase1_infeasible" / "phase2_infeasible"
    sigma: tuple | None = None
    alphas: np.ndarray | None = None
    a: np.ndarray | None = None
    impedances: np.ndarray | None = None
    delta: float | None = None             # = max_{I1}(Q-1), directly (manuscript eq. 6.6)
    achieved_mu: float = 0.0
    winning_start: str | None = None       # which seed in the Phase-2 pool won, for the driver's record
    per_start: dict | None = None          # {seed_name: achieved delta or None}, for the driver's record
    time_phase1_s: float = 0.0             # summed over every sigma pattern tried, for the driver's record
    time_phase2_s: float = 0.0             # summed over every sigma pattern tried, for the driver's record


def design_direct(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                   smaller_solutions: dict[int, np.ndarray] | None = None,
                   magnitude_sdp_result: "MagnitudeSDPResult | None" = None,
                   n_pad_variants: int = 4, n_random_draws: int = 3,
                   gamma_bound: float = 0.9995, n_grid_B: int = 200, n_grid_C: int = 200,
                   phase1_n_steps: int = 20, seed: int = 0, maxiter: int = 400) -> DirectDesignResult:
    """The two-phase synthesis algorithm (manuscript Sec. 6.3): Phase 1
    opens the gap (all 2^len(J0) sign patterns tried; every pattern that
    clears the target depth goes on to Phase 2, not just the single
    largest-margin one -- see below), Phase 2 flattens the pass band from
    a multi-seed pool. Every returned design is realisable by construction
    (forward_reconstruct's SU(1,1) identity / Prop. 4.4), so -- unlike
    design_filter_full -- there is no admissible-but-unrealisable middle
    case to report.

    Known limitation, inherited from the manuscript's own algorithm, not
    an implementation bug: Phase 1 is a continuation/ramp from alpha=0,
    so it can only ever reach whichever sign pattern kappa=cos(n*theta)
    naturally has at each J0 component's location at that starting point
    -- confirmed directly (n=9, a two-stop-band/two-pass-band case tested
    extensively elsewhere in this project): sigma=(1,1) is known
    achievable (found previously by pure random-restart search, unrelated
    to alpha=0), but Phase 1 here reaches depth=-0.87 for it regardless of
    ramp schedule length, because the *first*, tiniest ramp target already
    fails from the (tiny, near-zero) starting perturbation and bisection
    cannot recover -- a genuinely unreachable *basin*, not a search-effort
    problem. This is the same limitation the now-superseded
    design_via_continuation had; the manuscript's Phase 1 does not solve
    it, since it is explicitly continuation-based ("starting from the
    trivial structure alpha=0").

    Seed pool for Phase 2 (manuscript's own prescription -- do NOT seed
    from the lifted SDP, it violates (A) and is not a block):
      1. the winning Phase-1 solution;
      2. the magnitude-SDP factorisation, if magnitude_sdp_result is given
         and its seed gap quality is not hopeless (see _seed_gap_quality);
      3. zero-padded embeddings of designs already found at smaller n'
         (smaller_solutions = {n_small: alphas});
      4. a small number of random draws around the Phase-1 point.
    """
    # Try Phase 2 for EVERY sigma pattern that cleared Phase 1's threshold,
    # not just the single largest-margin one. The manuscript's own wording
    # ("the one giving the largest margin is fixed") suggests a single
    # greedy pick, but that heuristic proved unreliable in practice: it is
    # sensitive to Phase 1's own tuning (grid size, ramp schedule), and a
    # margin-maximising sigma is not guaranteed to be the one Phase 2 can
    # actually flatten a good pass band from -- confirmed empirically,
    # tightening Phase 1's grid changed which sigma "won" for a case with
    # a previously-known-good answer, and the new winner failed Phase 2
    # entirely while the old winner (a smaller-margin pattern) was known
    # to work well. Cost is modest in practice: len(J0) is small (<=3 per
    # the manuscript), and only patterns that are Phase-1-feasible at all
    # reach Phase 2.
    import time as _time
    time_phase1_s = 0.0
    if not J0:
        feasible_sigmas = [((), None, float("inf"))]
    else:
        sigmas = list(itertools.product((1, -1), repeat=len(J0)))
        feasible_sigmas = []
        for sig in sigmas:
            _t0 = _time.time()
            alphas_p1, depth = _phase1_open_gap(n, J0, sig, mu0, gamma_bound=gamma_bound,
                                                 n_grid_C=n_grid_C, n_steps=phase1_n_steps)
            time_phase1_s += _time.time() - _t0
            if depth >= np.cosh(mu0) - 1e-4:  # match _phase1_open_gap's own verify() tolerance
                feasible_sigmas.append((sig, alphas_p1, depth))
        if not feasible_sigmas:
            return DirectDesignResult(n=n, mu0=mu0, status="phase1_infeasible", time_phase1_s=time_phase1_s)

    rng = np.random.default_rng(seed)

    # Seeds that don't depend on sigma, shared across every Phase-2 attempt.
    shared_seeds: dict[str, np.ndarray] = {}
    if magnitude_sdp_result is not None and magnitude_sdp_result.f is not None:
        try:
            alphas_sdp, a_sdp, info = _factorize_magnitude_f(magnitude_sdp_result.f)
            gap_q = _seed_gap_quality(a_sdp, J0)
            if info.reliable and gap_q > 1.0 + 1e-6:
                shared_seeds["sdp_seed"] = np.tanh(alphas_sdp[:n])
        except Exception:
            pass
    if smaller_solutions:
        for n_small, alphas_small in smaller_solutions.items():
            if n_small >= n:
                continue
            for i, padded in enumerate(_padded_alpha_seeds(np.asarray(alphas_small, dtype=float),
                                                             n, rng, max_variants=n_pad_variants)):
                shared_seeds[f"zero_pad_from_n{n_small}_{i}"] = np.tanh(padded[:n])
    # Small random draws near the *trivial* point (alpha=0), not around
    # phase1's own solution: phase1's gamma can already sit close to the
    # +-1 boundary (arctanh blows up there), and perturbing it further
    # with the naive scale tried first (0.1) occasionally pushed a
    # component past the boundary before clipping, producing a huge
    # arctanh(alpha) and a wildly infeasible-looking seed (confirmed
    # empirically: one such draw reported delta~2.3e6 after "succeeding").
    for i in range(n_random_draws):
        shared_seeds[f"random_{i}"] = np.clip(0.02 * rng.standard_normal(n), -gamma_bound, gamma_bound)

    best_overall = None  # (delta, alphas, sigma, winning_start, per_start)
    time_phase2_s = 0.0
    for sig, alphas_p1, depth in feasible_sigmas:
        seed_pool = dict(shared_seeds)
        if alphas_p1 is not None:
            seed_pool["phase1"] = np.tanh(alphas_p1[:n])
        if not seed_pool:
            seed_pool["zero"] = np.zeros(n)

        _t0 = _time.time()
        result, per_start = _phase2_flatten(n, J0, J1, mu0, sig, seed_pool,
                                             gamma_bound=gamma_bound, n_grid_B=n_grid_B, n_grid_C=n_grid_C,
                                             maxiter=maxiter)
        time_phase2_s += _time.time() - _t0
        if result is None:
            continue
        alphas, delta, winning_start = result
        if best_overall is None or delta < best_overall[0]:
            best_overall = (delta, alphas, sig, winning_start, per_start)

    if best_overall is None:
        return DirectDesignResult(n=n, mu0=mu0, status="phase2_infeasible",
                                   time_phase1_s=time_phase1_s, time_phase2_s=time_phase2_s)

    delta, alphas, sigma_best, winning_start, per_start = best_overall
    a = a_from_alphas(alphas)
    impedances = np.empty(n + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(alphas):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    mu_min = np.inf
    for (u, v), s in zip(J0, sigma_best):
        kap = s * kappa_B(a, np.linspace(u, v, 4000))
        mu_min = min(mu_min, np.arccosh(max(float(np.min(kap)), 1.0)))

    return DirectDesignResult(n=n, mu0=mu0, status="optimal", sigma=sigma_best, alphas=alphas, a=a,
                               impedances=impedances, delta=delta, achieved_mu=mu_min if J0 else float("inf"),
                               winning_start=winning_start, per_start=per_start,
                               time_phase1_s=time_phase1_s, time_phase2_s=time_phase2_s)


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
    delta: float | None = None             # = max_{I1}(Q-1) directly (manuscript eq. 6.6), NOT sqrt(u)-1
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
    the multiband config) from delta=0.065 to 0.36. Per-pattern enumeration
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
            delta = float(max(u - 1.0, 0.0))
            if best_overall is None or delta < best_overall[0]:
                best_overall = (delta, sig, alphas, a, u)

    if best_overall is None:
        return LayerDesignResult(n=n, mu0=mu0, status="infeasible")

    delta, sig, alphas, a, u = best_overall
    impedances = np.empty(n + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(alphas):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    mu_min = np.inf
    for (al, be), s in zip(J0, sig):
        kap = s * kappa_B(a, np.linspace(al, be, 4000))
        mu_min = min(mu_min, np.arccosh(max(float(np.min(kap)), 1.0)))

    return LayerDesignResult(n=n, mu0=mu0, status="optimal", sigma=sig, alphas=alphas, a=a,
                              impedances=impedances, u=u, delta=delta, verified=True,
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
    delta = float(max(u_prev - 1.0, 0.0))
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
                              impedances=impedances, u=u_prev, delta=delta,
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
