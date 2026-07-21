"""Inverse problem: a -> (q_1, q_2) -> alpha's.

a = (a_0, ..., a_n) is a candidate q~_1 coefficient vector (sum(a) = 1).
This module turns it into a physical layer sequence, or explains why it
can't: admissibility (constraint A), minimum-phase / realizability
(Theorem 4.6), spectral factorization (Fejer-Riesz), and layer stripping
(the Schur recursion of Theorem 4.6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .forward import q1_abs_sq


def check_admissibility(a: np.ndarray, n_grid: int = 10000) -> tuple[bool, float]:
    """G(theta) = |q~_1(theta)|^2 >= 1 for all theta (constraint A)?"""
    theta = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    G = q1_abs_sq(a, theta)
    G_min = float(np.min(G))
    return G_min >= 1.0 - 1e-8, G_min


def check_min_phase(a: np.ndarray, tol: float = 1e-6) -> tuple[bool, np.ndarray]:
    """Do all roots of q~_1(z) = sum a_m z^m lie strictly inside |z| < 1?

    This is exactly Theorem 4.6's zero-free-in-the-closed-disc hypothesis
    on p1(w) = sum a_{n-m} w^m, translated to q~_1: p1(w) = w^n q~_1(1/w),
    so p1's roots are the reciprocals of q~_1's roots, and "p1 zero-free in
    |w|<=1" <=> "all roots of q~_1 satisfy |z|<1".
    """
    a = np.asarray(a, dtype=float)
    n = len(a) - 1
    if n == 0:
        return True, np.array([])
    roots = np.roots(a[::-1])
    return bool(np.all(np.abs(roots) < 1.0 + tol)), roots


def ensure_min_phase(a: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, bool]:
    """Project a onto the minimum-phase representative of the same
    G = |q~_1|^2, if it isn't one already.

    Constraint (A) alone (|q~_1| >= 1 on the circle) does *not* imply
    zero-freeness inside the disc: e.g. a=[3,2] and a=[2,3] give the same
    |q~_1| on the circle, but only one has its root inside. So whatever
    produced `a` (an SDP relaxation, a local optimizer, ...) must be
    checked, not assumed. All branches share the same G; reflecting an
    outside root z0 to its conjugate reciprocal 1/conj(z0) leaves G
    unchanged. NOTE: this generally changes kappa_B = Re(q~_1) (it is a
    *different* function with the same magnitude on the circle -- see
    check_admissibility vs. constraint (C)), so re-verify (C) afterwards.

    Returns (a_projected, was_reflected).
    """
    ok, _ = check_min_phase(a, tol=tol)
    if ok:
        return np.asarray(a, dtype=float).copy(), False
    a = np.asarray(a, dtype=float)
    n = len(a) - 1
    c = np.array([np.sum(a[: n + 1 - l] * a[l:]) for l in range(n + 1)])
    a_mp = fejer_riesz(c)
    a_mp = a_mp * (np.sum(a) / np.sum(a_mp))  # keep the same overall sign/normalization
    return a_mp, True


def _polish_roots(coeffs_increasing: np.ndarray, roots: np.ndarray, iters: int = 6) -> np.ndarray:
    """A few Newton iterations to refine roots found via companion-matrix
    eigenvalues (numpy.roots), which lose accuracy at higher degree,
    especially for clustered/near-multiple roots."""
    coeffs_dec = coeffs_increasing[::-1]           # numpy.polyval wants highest power first
    d = len(coeffs_increasing) - 1
    deriv_dec = coeffs_dec[:-1] * np.arange(d, 0, -1)
    z = roots.copy()
    for _ in range(iters):
        p_val = np.polyval(coeffs_dec, z)
        dp_val = np.polyval(deriv_dec, z)
        step = np.where(np.abs(dp_val) > 1e-14, p_val / np.where(dp_val == 0, 1, dp_val), 0.0)
        z = z - step
    return z


def _autocorr_to_full_poly(c: np.ndarray) -> np.ndarray:
    """c (len n+1) -> coeffs (len 2n+1, increasing powers) of z^n * G(theta)|_{z=e^{i theta}}."""
    n = len(c) - 1
    coeffs = np.zeros(2 * n + 1)
    coeffs[n] = c[0]
    for l in range(1, n + 1):
        coeffs[n - l] += c[l]
        coeffs[n + l] += c[l]
    return coeffs


def _deflate_double_root(coeffs: np.ndarray, z0: float, tol: float = 1e-9):
    """If z0 (+-1) is a double root of coeffs (increasing powers), divide it
    out exactly (twice) via polynomial division and return the deflated
    coefficients; else return coeffs unchanged. z0=+-1 is checked exactly
    (no root-finding needed): coeffs evaluated at +-1 via alternating/plain
    sum, which is what makes this deflation possible and numerically exact
    in the first place."""
    from numpy.polynomial import polynomial as Pp
    factor = np.array([-z0, 1.0])
    cur = coeffs
    for _ in range(2):
        if abs(np.polyval(cur[::-1], z0)) > tol * max(1.0, np.max(np.abs(cur))):
            return coeffs, 0
        cur, rem = Pp.polydiv(cur, factor)
        if np.max(np.abs(rem)) > tol * max(1.0, np.max(np.abs(coeffs))):
            return coeffs, 0
    return cur, 1


def _factorization_error(a: np.ndarray, c: np.ndarray, n_grid: int = 4000) -> float:
    """max|G_target(theta) - |sum a_m e^{i m theta}||^2, the ultimate
    correctness criterion for a spectral factor (independent of *how*
    it was computed)."""
    n = len(c) - 1
    theta = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    l = np.arange(1, n + 1)
    G_target = c[0] + 2 * (np.cos(np.outer(theta, l)) @ c[1:]) if n > 0 else np.full(n_grid, c[0])
    m = np.arange(len(a))
    G_actual = np.abs(np.exp(1j * np.outer(theta, m)) @ a) ** 2
    return float(np.max(np.abs(G_target - G_actual)))


def _fejer_riesz_sdp(c: np.ndarray, n_iters: int = 4, reg: float = 1e-6, solver: str = "CLARABEL"):
    """Alternative spectral factorization via the Gram/SDP route (Nesterov's
    theorem -- the same trace-constrained PSD-matrix parametrization as
    poly_sdp.py's Theorem 4.1 machinery, applied here to *find* a factor
    rather than just to certify nonnegativity): solve for X >> 0 with
    sum_{i-j=k} X_ij = c_k, then read off a spectral factor from X.

    The trace constraints alone don't pin down a rank-1 X -- v^*Xv = G(theta)
    holds for *any* X satisfying them, rank irrelevant; PSD-ness is what
    proves G>=0, not what selects a single factor. Fejer-Riesz guarantees a
    rank-1 point exists in that affine+PSD slice, but a plain SDP solve
    generally lands on the analytic center instead (near-full-rank). Using
    iteratively reweighted trace minimization (log-det heuristic: minimize
    trace(W X) with W <- (X+reg*I)^{-1} each round) pushes toward that
    extremal rank-1 point. Returns None if it doesn't converge to
    (near-)rank-1, so the caller can fall back to something else.
    """
    import cvxpy as cp

    n = len(c) - 1
    if n == 0:
        return np.array([np.sqrt(max(c[0], 0.0))])
    W = np.eye(n + 1)
    X_val = None
    for _ in range(n_iters):
        X = cp.Variable((n + 1, n + 1), symmetric=True)
        constraints = [X >> 0, cp.trace(X) == c[0]]
        for k in range(1, n + 1):
            diag_terms = [X[k + i, i] for i in range(n + 1 - k)]
            constraints.append(cp.sum(cp.hstack(diag_terms)) == c[k])
        prob = cp.Problem(cp.Minimize(cp.trace(W @ X)), constraints)
        try:
            prob.solve(solver=solver)
        except cp.error.SolverError:
            break
        if prob.status not in ("optimal", "optimal_inaccurate") or X.value is None:
            break
        X_val = X.value
        W = np.linalg.inv(X_val + reg * np.eye(n + 1))

    if X_val is None:
        return None
    eigvals, eigvecs = np.linalg.eigh(X_val)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    return eigvecs[:, 0] * np.sqrt(max(eigvals[0], 0.0))


def fejer_riesz(c: np.ndarray, unit_circle_tol: float = 1e-6, polish_iters: int = 6,
                 verify_tol: float = 1e-4) -> np.ndarray:
    """Fejer-Riesz spectral factorization: given autocorrelation
    coefficients c_l (G(theta) = c_0 + 2*sum_l c_l cos(l*theta) >= 0),
    return a such that |sum_m a_m e^{i m theta}|^2 = G(theta), with a
    minimum-phase (all roots of sum a_m z^m strictly inside |z|<1).

    Primary path is root-based (_fejer_riesz_root below): fast (~ms), and
    extensively validated on well-conditioned cases. It becomes unreliable
    when G has multiple near-degenerate near-unit-circle zeros close
    together (confirmed via 50-digit mpmath: a root*selection* issue, not
    root-finding precision -- see _fejer_riesz_root's docstring). Rather
    than trust it blindly, the result is checked against G directly
    (_factorization_error); if that check fails, an SDP-based factorization
    (_fejer_riesz_sdp) is tried as a fallback, since it sidesteps root
    finding entirely and was confirmed to succeed on a case where the
    root-based method gave a ~50% error. The SDP route isn't used as the
    default because it is itself not universally robust (the reweighting
    heuristic can fail to converge to rank-1 on well-conditioned generic
    cases where the root-based method has no trouble at all) and is
    noticeably slower (SDP solves vs. one polynomial root-find).
    """
    a_root = _fejer_riesz_root(c, unit_circle_tol=unit_circle_tol, polish_iters=polish_iters)
    err_root = _factorization_error(a_root, c)
    if err_root <= verify_tol:
        return a_root

    a_sdp = _fejer_riesz_sdp(c)
    if a_sdp is not None:
        err_sdp = _factorization_error(a_sdp, c)
        if err_sdp < err_root:
            return a_sdp
    return a_root  # neither route is perfect; root-based is at least deterministic


def _fejer_riesz_root(c: np.ndarray, unit_circle_tol: float = 1e-6, polish_iters: int = 6) -> np.ndarray:
    """Root-based Fejer-Riesz factorization (the primary path -- see
    fejer_riesz's docstring for when this is and isn't reliable).

    Root-finding via numpy.roots on the degree-2n autocorrelation
    polynomial loses accuracy as n grows (companion-matrix eigenvalues);
    each root is refined with a few Newton steps before the inside/outside
    split, which matters once n gtrsim 8. This matters most exactly at
    z=+-1: G(0) = (sum a_m)^2 is often forced to a fixed value by
    normalization (e.g. F = G-1 in p2_from_a has F(0) = 0 *identically*,
    since q~_1(0)=1 is required of every valid design), so a guaranteed
    *double* root sits right at z=1 -- and double roots are exactly what
    companion-matrix eigenvalues resolve worst (perturbations of order
    sqrt(machine epsilon), not epsilon). Whenever G(+-1) essentially
    vanishes, that double root is deflated out exactly (via polynomial
    division, no root-finding needed) before touching the rest with
    numpy.roots, which removes the worst-conditioned part of the problem.
    """
    c = np.asarray(c, dtype=float)
    n = len(c) - 1
    if n == 0:
        return np.array([np.sqrt(max(c[0], 0.0))])

    coeffs = _autocorr_to_full_poly(c)
    forced = []
    for z0 in (1.0, -1.0):
        coeffs, got = _deflate_double_root(coeffs, z0)
        if got:
            forced.append(z0)

    roots = np.roots(coeffs[::-1]) if len(coeffs) > 1 else np.array([])
    roots = _polish_roots(coeffs, roots, iters=polish_iters) if len(roots) else roots
    mags = np.abs(roots)

    on_circle = np.abs(mags - 1.0) <= unit_circle_tol
    inside = (~on_circle) & (mags < 1.0)
    outside = (~on_circle) & (mags >= 1.0)

    chosen = list(roots[inside]) + forced
    n_needed = n - len(chosen)

    if np.any(on_circle):
        chosen += _half_of_unit_circle_roots(roots[on_circle], n_needed)
    elif n_needed > 0:
        rest = roots[outside]
        order = np.argsort(np.abs(rest))
        chosen += list(rest[order[:n_needed]])

    chosen = np.array(chosen[:n])
    poly = np.poly(chosen)[::-1].real   # increasing powers, monic; roots come in conjugate pairs -> real

    scale_sq = c[0] / np.sum(poly ** 2)
    if scale_sq < 0:
        raise RuntimeError("negative scale in spectral factorization; G may not be nonnegative")
    return np.sqrt(scale_sq) * poly


def _half_of_unit_circle_roots(unit_roots: np.ndarray, n_needed: int, cluster_tol: float = 1e-4):
    """G real & >=0 forces roots on |z|=1 to occur with even multiplicity
    (complex-conjugate pairs, or a real double root at z=+-1). Cluster by
    angle and keep half the multiplicity of each cluster."""
    angles = np.angle(unit_roots)
    order = np.argsort(angles)
    angles_sorted, roots_sorted = angles[order], unit_roots[order]

    clusters, cur = [], [roots_sorted[0]]
    for r, ang in zip(roots_sorted[1:], angles_sorted[1:]):
        if abs(ang - np.angle(cur[-1])) <= cluster_tol:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    clusters.append(cur)
    if len(clusters) > 1 and abs((angles_sorted[0] + 2 * np.pi) - np.angle(clusters[-1][-1])) <= cluster_tol:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    chosen = []
    for cl in clusters:
        chosen += list(cl[: len(cl) // 2])
    if len(chosen) != n_needed:
        chosen = [r for cl in clusters for r in cl][:n_needed]
    return chosen


def p2_from_a(a: np.ndarray, n_grid: int = 20000) -> np.ndarray:
    """A spectral factor of F(theta) = |q~_1(theta)|^2 - 1 = |q_2(theta)|^2
    (Remark 4.9: any spectral factor is admissible here, unlike q~_1
    itself -- this is the genuine, physically-meaningful freedom: it
    changes the realized alpha_j/impedances but not kappa_B or T_N).

    Fejer-Riesz needs F >= 0 *exactly*; an upstream a that only satisfies
    constraint (A) to some optimizer tolerance (e.g. from sdp_design's
    local polish) can leave F slightly negative at isolated points (found
    empirically: dips as small as -1e-7 are enough to make root-finding
    badly inconsistent right there, since there is no genuine real
    factorization to find in a neighborhood of a true sign violation).
    F is floored by shifting c_0 up by just enough to cover the worst
    dip found on a fine grid (plus a small safety margin), which is the
    minimal correction that restores a well-posed problem.
    """
    a = np.asarray(a, dtype=float)
    n = len(a) - 1
    c = np.array([np.sum(a[: n + 1 - l] * a[l:]) for l in range(n + 1)])
    c[0] -= 1.0
    theta = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    F_min = float(np.min(q1_abs_sq(a, theta) - 1.0))
    if F_min < 0:
        c[0] += -F_min + 1e-12
    return fejer_riesz(c)


def schur_strip(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Theorem 4.6's Schur/layer-peeling recursion. p1, p2: real coeff
    vectors (increasing powers of w), length n+1, |p1|^2-|p2|^2=1 on
    |w|=1, p1 zero-free in the closed unit disc. Returns
    alphas = (alpha_0,...,alpha_n)."""
    P1 = np.asarray(p1, dtype=float).copy()
    P2 = np.asarray(p2, dtype=float).copy()
    n = len(P1) - 1
    if len(P2) != n + 1:
        raise ValueError("p1 and p2 must have the same length")
    alphas = np.zeros(n + 1)
    for j in range(n, 0, -1):
        ratio = P2[0] / P1[0]
        if abs(ratio) >= 1.0:
            raise RuntimeError(f"Schur parameter out of (-1,1) at step j={j}: {ratio}")
        alpha_j = np.arctanh(ratio)
        alphas[j] = alpha_j
        ch, sh = np.cosh(alpha_j), np.sinh(alpha_j)
        newP1 = ch * P1 - sh * P2       # degree <= j-1: newP1[j] ~ 0
        newP2 = ch * P2 - sh * P1       # newP2[0] ~ 0 by construction of alpha_j
        P1 = newP1[:j]
        P2 = newP2[1 : j + 1]
    alphas[0] = np.arctanh(P2[0] / P1[0])
    return alphas


@dataclass
class RealizabilityInfo:
    admissible: bool
    G_min: float
    was_reflected: bool
    a_used: np.ndarray = field(default_factory=lambda: np.array([]))  # min-phase a actually stripped
    impedances: np.ndarray = field(default_factory=lambda: np.array([]))
    reconstruction_error: float = float("nan")
    su11_error: float = float("nan")   # max|p1|^2-|p2|^2-1| on the circle; large => p2 is untrustworthy
    reliable: bool = True              # False if p2_from_a's factorization looks broken (see su11_error)
    failure: str | None = None         # human-readable reason, set only when reliable=False


def _su11_error(p1: np.ndarray, p2: np.ndarray, n_grid: int = 4000) -> float:
    theta = np.linspace(0, 2 * np.pi, n_grid)
    z = np.exp(1j * theta)
    m1, m2 = np.arange(len(p1)), np.arange(len(p2))
    p1v = (z[:, None] ** m1) @ p1
    p2v = (z[:, None] ** m2) @ p2
    return float(np.max(np.abs(np.abs(p1v) ** 2 - np.abs(p2v) ** 2 - 1.0)))


def alphas_from_a(a: np.ndarray, su11_tol: float = 1e-4) -> tuple[np.ndarray, RealizabilityInfo]:
    """Full pipeline: a -> check admissibility -> ensure min-phase ->
    (p1, p2) -> Schur-strip -> alphas, physical impedances, diagnostics.

    KNOWN OPEN ISSUE: p2_from_a's root-based spectral factorization of
    F=G-1 can become unreliable for highly-optimized/near-degenerate
    designs, where F touches down close to zero at several points (or, if
    an upstream optimizer's tolerance let F dip slightly negative, two
    nearby simple real crossings appear instead of one clean double root)
    -- the root-selection logic can then pick a non-conjugate-symmetric
    set, producing a wrong (non-real-consistent) p2. This is *not* a
    root-finding precision issue (checked against 50-digit mpmath: same
    roots to 13+ digits) -- it's a combinatorial selection issue that
    remains open. Rather than silently return wrong alphas/impedances,
    the SU(1,1) identity |p1|^2-|p2|^2=1 is checked directly on the
    returned p2; if it's violated beyond su11_tol, `reliable=False` and
    `failure` explains why (alphas/impedances are still populated
    best-effort, from whatever schur_strip could do, but should not be
    trusted).
    """
    from .forward import forward_reconstruct  # local import: forward.py doesn't import inverse.py

    admissible, G_min = check_admissibility(a)
    a_mp, was_reflected = ensure_min_phase(a)

    p1 = a_mp[::-1].copy()
    p2 = p2_from_a(a_mp)
    su11_err = _su11_error(p1, p2)
    reliable, failure = True, None
    if su11_err > su11_tol:
        reliable = False
        failure = (f"|p1|^2-|p2|^2 deviates from 1 by up to {su11_err:.2e} "
                   "(> su11_tol); p2's spectral factorization is unreliable here "
                   "(see alphas_from_a docstring) -- alphas/impedances below are not trustworthy.")

    try:
        alphas = schur_strip(p1, p2)
    except RuntimeError as e:
        reliable = False
        failure = (failure + " " if failure else "") + f"schur_strip also failed: {e}"
        n = len(p1) - 1
        alphas = np.full(n + 1, np.nan)

    # alphas has n+1 entries (alpha_0..alpha_n); physical impedances are
    # p_0 (background), p_1..p_n (the n layers), p_{n+1} (background) --
    # n+2 = len(alphas)+1 values total, not +2 (that extra slot silently
    # stayed NaN before this fix).
    impedances = np.full(len(alphas) + 1, np.nan)
    if reliable or not np.any(np.isnan(alphas)):
        impedances[0] = 1.0
        for j, alpha_j in enumerate(alphas):
            impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    if not np.any(np.isnan(alphas)):
        p1_chk, p2_chk = forward_reconstruct(alphas)
        err = max(np.max(np.abs(p1_chk - p1)), np.max(np.abs(p2_chk - p2)))
    else:
        err = float("nan")

    info = RealizabilityInfo(
        admissible=admissible, G_min=G_min, was_reflected=was_reflected,
        a_used=a_mp, impedances=impedances, reconstruction_error=float(err),
        su11_error=su11_err, reliable=reliable, failure=failure,
    )
    return alphas, info
