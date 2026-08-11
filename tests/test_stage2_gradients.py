"""Stage 2, spec Sec. 3.1 ("Gradients"): forward_with_grad's exact
analytic dQ/dalpha_j, dkappa_B/dalpha_j against finite differences, at
the spec's own precision bar -- RELATIVE error < 1e-7. This is a
different, tighter bar than sanity.py's own battery (which checks
something else: that the two independently-coded evaluation paths agree
in *value*, plus a looser absolute-error gradient sanity check, not this
stage's own pass/fail requirement).

A plain central (2-point) difference at h=1e-5 -- the sweet spot for
*that* method, established elsewhere in this project -- cannot actually
deliver 1e-7 *relative* precision here: confirmed directly (n=12 case),
its own ~1e-8 absolute truncation/rounding noise floor, divided by a
merely-moderate (not tiny) gradient component (~5e-3) in an array whose
overall dynamic range spans several orders of magnitude (alpha compounds
multiplicatively through the layer recursion, same mechanism as Stage
1's coefficient-magnitude spread), gives ~2e-6 relative error on its
own -- exceeding the bar before the analytic gradient is even in
question. Using a 4th-order (5-point) stencil instead reduces truncation
error from O(h^2) to O(h^4); confirmed directly across a small h-sweep,
h=1e-3 reaches ~2e-11 to ~2e-8 max relative error across n=1..12 (worst
case still an order of magnitude inside the 1e-7 bar), so that is used
throughout below.

Run: python3 -m pytest tests/test_stage2_gradients.py -v   (from repo root)
"""
import numpy as np
import pytest

from scattering.forward import a_from_alphas, forward_with_grad, grad_free_vars, kappa_B, q1_abs_sq


def _random_matched_alphas(rng: np.random.Generator, n: int) -> np.ndarray:
    alphas = rng.uniform(-1.2, 1.2, n + 1)
    alphas[-1] -= np.sum(alphas)  # force sum(alphas) == 0 (matched)
    return alphas


def _relative_error(analytic: np.ndarray, fd: np.ndarray, floor: float = 1e-6) -> float:
    """max_j,l |analytic - fd| / max(|analytic|, floor). The floor exists
    for one reason: dQ/dalpha_j and dkappa/dalpha_j are EXACTLY zero at
    alpha=0 (Q-1=|q2|^2 has its global minimum there, and by the identical
    L_j K p^(j) structure so does d(kappa)/d(alpha_j) -- every A_j is
    diagonal at alpha=0), so a naive relative error is undefined (0/0)
    for any test case at or near that point. floor=1e-6 sits well above
    the 5-point stencil's own ~1e-9 noise floor (see module docstring),
    so it only forgives components that are genuinely near the alpha=0
    degeneracy, not the moderate-magnitude components a 2-point
    difference was unable to resolve."""
    denom = np.maximum(np.abs(analytic), floor)
    return float(np.max(np.abs(analytic - fd) / denom))


def _fd_gradients_5pt(alphas: np.ndarray, theta: np.ndarray, h: float = 1e-3):
    """4th-order (5-point) central difference: [-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/(12h)."""
    n = len(alphas) - 1
    dQ_fd = np.empty((n + 1, len(theta)))
    dkappa_fd = np.empty((n + 1, len(theta)))
    for j in range(n + 1):
        a2p, ap, am, a2m = (alphas.copy() for _ in range(4))
        a2p[j] += 2 * h
        ap[j] += h
        am[j] -= h
        a2m[j] -= 2 * h
        a_2p, a_p, a_m, a_2m = (a_from_alphas(x) for x in (a2p, ap, am, a2m))
        Q2p, Qp, Qm, Q2m = (q1_abs_sq(x, theta) for x in (a_2p, a_p, a_m, a_2m))
        k2p, kp, km, k2m = (kappa_B(x, theta) for x in (a_2p, a_p, a_m, a_2m))
        dQ_fd[j] = (-Q2p + 8 * Qp - 8 * Qm + Q2m) / (12 * h)
        dkappa_fd[j] = (-k2p + 8 * kp - 8 * km + k2m) / (12 * h)
    return dQ_fd, dkappa_fd


@pytest.mark.parametrize("n", [1, 2, 4, 7, 12])
def test_gradients_relative_error(n, tol=1e-7):
    """Spec Sec. 3.1's own test: compare forward_with_grad's analytic
    dQ/dalpha_j, dkappa_B/dalpha_j against finite differences at random
    matched alpha, random theta, relative error < 1e-7."""
    rng = np.random.default_rng(3000 + n)
    alphas = _random_matched_alphas(rng, n)
    theta = rng.uniform(0.05, np.pi - 0.05, 11)

    res = forward_with_grad(alphas, theta)
    dQ_fd, dkappa_fd = _fd_gradients_5pt(alphas, theta)

    dQ_rel_err = _relative_error(res["dQ"], dQ_fd)
    dkappa_rel_err = _relative_error(res["dkappa"], dkappa_fd)

    assert dQ_rel_err < tol, f"dQ relative error {dQ_rel_err:.3e} >= {tol:.0e}"
    assert dkappa_rel_err < tol, f"dkappa relative error {dkappa_rel_err:.3e} >= {tol:.0e}"


def test_gradients_relative_error_near_degenerate_point():
    """The alpha=0 degeneracy itself (see _relative_error's docstring):
    analytic and finite-difference gradients should BOTH be at or near
    zero there, not disagree -- this is what makes the floor in
    _relative_error meaningful rather than a loophole."""
    n = 5
    alphas = np.zeros(n + 1)
    theta = np.array([0.3, 1.0, 2.0, np.pi - 0.3])

    res = forward_with_grad(alphas, theta)
    assert np.max(np.abs(res["dQ"])) < 1e-12
    assert np.max(np.abs(res["dkappa"])) < 1e-12

    dQ_fd, dkappa_fd = _fd_gradients_5pt(alphas, theta)
    assert np.max(np.abs(dQ_fd)) < 1e-9
    assert np.max(np.abs(dkappa_fd)) < 1e-9


def test_grad_free_vars_matches_chain_rule():
    """grad_free_vars' d/dalpha_j(free) = d/dalpha_j - d/dalpha_n (matched
    elimination, spec Sec. 3.1) against a direct finite-difference check
    on the free-variable-eliminated (n free alphas, alpha_n solved for)
    parametrization, at the same 1e-7 relative bar."""
    n = 6
    rng = np.random.default_rng(4001)
    alphas_free = rng.uniform(-1.0, 1.0, n)
    theta = rng.uniform(0.1, np.pi - 0.1, 7)

    def full_alphas(free):
        return np.concatenate([free, [-np.sum(free)]])

    res = forward_with_grad(full_alphas(alphas_free), theta)
    dQ_free_analytic = grad_free_vars(res["dQ"])

    h = 1e-3
    dQ_free_fd = np.empty((n, len(theta)))
    for j in range(n):
        f2p, fp, fm, f2m = (alphas_free.copy() for _ in range(4))
        f2p[j] += 2 * h
        fp[j] += h
        fm[j] -= h
        f2m[j] -= 2 * h
        Q2p, Qp, Qm, Q2m = (q1_abs_sq(a_from_alphas(full_alphas(x)), theta) for x in (f2p, fp, fm, f2m))
        dQ_free_fd[j] = (-Q2p + 8 * Qp - 8 * Qm + Q2m) / (12 * h)

    rel_err = _relative_error(dQ_free_analytic, dQ_free_fd)
    assert rel_err < 1e-7
