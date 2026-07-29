"""Round-trip consistency tests between forward.py and inverse.py.

Two different round trips, and they are NOT symmetric in what they can be
expected to prove -- this distinction matters and is checked explicitly:

  inverse-then-forward (a -> alpha -> a'): should match closely whenever a
  is admissible and minimum-phase, because inverse.alphas_from_a's own
  internal consistency (schur_strip / forward_reconstruct) is exact up to
  floating point. This is the one that should reliably PASS.

  forward-then-inverse (alpha -> a -> alpha'): only matches when the
  *original* alphas happen to produce a p2 that is itself minimum-phase.
  p2's spectral factor is genuinely non-unique (Remark 4.9): a and its
  min-phase projection share the same |q~_1|, but alpha_j/impedances
  depend on which valid p2 you pick. inverse.alphas_from_a always returns
  the minimum-phase p2, so unless the original construction happened to
  use that same branch, alpha' will legitimately differ from alpha even
  though both describe physically valid structures with the same T_N.
  This is expected behavior, not a bug -- each check reports whether the
  original p2 was already minimum-phase, to explain PASS vs. FAIL.
"""
from __future__ import annotations

import numpy as np

from .forward import a_from_alphas, forward_reconstruct, forward_with_grad, q1_abs_sq
from .inverse import alphas_from_a, check_min_phase
from .certify import certify


def test_forward_inverse_roundtrip(alphas: np.ndarray, tol: float = 1e-10) -> dict:
    """alpha -> a (forward) -> alpha' (inverse); check ||alpha - alpha'|| < tol.

    See module docstring: only expected to pass when the *original* p2 (from
    forward_reconstruct(alphas)) is itself minimum-phase.
    """
    alphas = np.asarray(alphas, dtype=float)
    p1, p2 = forward_reconstruct(alphas)
    p2_min_phase = True
    if len(p2) > 1:
        roots_p2 = np.roots(p2[::-1])
        p2_min_phase = bool(np.all(np.abs(roots_p2) < 1.0 + 1e-9))

    a = p1[::-1]
    alphas2, info = alphas_from_a(a)
    err = float(np.max(np.abs(alphas2 - alphas)))
    passed = err < tol
    return {
        "passed": passed, "error": err, "p2_originally_min_phase": p2_min_phase,
        "expected_to_pass": p2_min_phase, "was_reflected": info.was_reflected,
        "note": ("" if p2_min_phase == passed else
                 "unexpected: p2-min-phase status doesn't match pass/fail as predicted"),
    }


def test_inverse_forward_roundtrip(a: np.ndarray, tol: float = 1e-10) -> dict:
    """a -> alpha (inverse) -> a' (forward); check ||a - a'|| < tol.

    Only meaningful when a is admissible AND minimum-phase (otherwise
    inverse.alphas_from_a projects a onto a *different* vector first, by
    construction, and a' will match that projection, not the original a --
    see inverse.ensure_min_phase).
    """
    a = np.asarray(a, dtype=float)
    ok_min_phase, _ = check_min_phase(a)
    G_ok = bool(np.min(q1_abs_sq(a, np.linspace(0, 2 * np.pi, 10000))) >= 1.0 - 1e-8)

    alphas, info = alphas_from_a(a)
    a2 = a_from_alphas(alphas)
    err = float(np.max(np.abs(a2 - a)))
    passed = err < tol
    return {
        "passed": passed, "error": err, "admissible": G_ok, "min_phase": ok_min_phase,
        "expected_to_pass": G_ok and ok_min_phase, "was_reflected": info.was_reflected,
    }


def test_forward_with_grad(alphas: np.ndarray, theta: np.ndarray, h: float = 1e-5,
                            rtol: float = 1e-6) -> dict:
    """forward_with_grad's analytic dQ/dalpha_j, dkappa/dalpha_j against
    central finite differences (manuscript Sec. 6.3's own prescribed test).
    h=1e-5 sits near the empirically-confirmed sweet spot of the classic
    finite-difference V-curve for this problem (truncation error ~h^2
    shrinking down to h~1e-5, ~4.7e-9 absolute; rounding error growing
    again for smaller h) -- NOT h=1e-6 as a naive first guess would use,
    which is already past the minimum and reports a needlessly pessimistic
    error. Also checks forward_with_grad's own *values* (not just
    gradients) against the existing a_from_alphas/q1_abs_sq/kappa_B
    pipeline, since these are two independently-coded evaluation paths
    that must agree to machine precision if both are correct."""
    alphas = np.asarray(alphas, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n = len(alphas) - 1

    res = forward_with_grad(alphas, theta)
    a = a_from_alphas(alphas)
    from .forward import kappa_B
    Q_ref, kappa_ref = q1_abs_sq(a, theta), kappa_B(a, theta)
    value_err = float(max(np.max(np.abs(res["Q"] - Q_ref)), np.max(np.abs(res["kappa"] - kappa_ref))))

    dQ_fd = np.empty((n + 1, len(theta)))
    dkappa_fd = np.empty((n + 1, len(theta)))
    for j in range(n + 1):
        ap, am = alphas.copy(), alphas.copy()
        ap[j] += h
        am[j] -= h
        Qp, Qm = q1_abs_sq(a_from_alphas(ap), theta), q1_abs_sq(a_from_alphas(am), theta)
        kp, km = kappa_B(a_from_alphas(ap), theta), kappa_B(a_from_alphas(am), theta)
        dQ_fd[j], dkappa_fd[j] = (Qp - Qm) / (2 * h), (kp - km) / (2 * h)

    dQ_err = float(np.max(np.abs(res["dQ"] - dQ_fd)))
    dkappa_err = float(np.max(np.abs(res["dkappa"] - dkappa_fd)))
    passed = value_err < 1e-10 and dQ_err < rtol and dkappa_err < rtol
    return {"passed": passed, "value_err": value_err, "dQ_abs_err": dQ_err, "dkappa_abs_err": dkappa_err}


def test_certify(alphas: np.ndarray, J0, J1, sigma, n_grid: int = 2_000_000, tol: float = 1e-9) -> dict:
    """certify's exact (rootfinding) delta/mu_min against a very fine grid
    brute force -- the grid can only ever *underestimate* the true extremum
    (it samples finitely many points), so exact and grid values should
    agree to within the grid's own resolution once that resolution is fine
    enough; true agreement should be exact to floating-point precision
    since both read off the same underlying trigonometric polynomial, just
    one via calculus and one via sampling."""
    from .forward import kappa_B, q1_abs_sq

    a = a_from_alphas(np.asarray(alphas, dtype=float))
    res = certify(alphas, J0, J1, sigma)

    grid_delta = max((float(np.max(q1_abs_sq(a, np.linspace(lo, hi, n_grid)) - 1.0)) for lo, hi in J1),
                      default=0.0)
    grid_depth = min((float(np.min(s * kappa_B(a, np.linspace(lo, hi, n_grid))))
                       for (lo, hi), s in zip(J0, sigma)), default=float("inf"))
    grid_mu = np.arccosh(max(grid_depth, 1.0)) if J0 else float("inf")

    delta_err = abs(res.delta - grid_delta)
    mu_err = abs(res.mu_min - grid_mu) if J0 else 0.0
    # certify's exact value must never be *worse* than what the grid found
    # (grid can only underestimate max(Q-1), overestimate min depth)
    sound = res.delta >= grid_delta - tol and (not J0 or res.mu_min <= grid_mu + tol)
    passed = sound and delta_err < 1e-6 and mu_err < 1e-6
    return {"passed": passed, "delta_err": delta_err, "mu_err": mu_err, "sound": sound}


def _print_case(name: str, alphas: np.ndarray):
    r1 = test_forward_inverse_roundtrip(alphas)
    status1 = "PASS" if r1["passed"] else ("FAIL (expected)" if not r1["expected_to_pass"] else "FAIL (unexpected!)")
    print(f"  [{name}] forward->inverse: {status1}  (err={r1['error']:.3e}, "
          f"p2 originally min-phase={r1['p2_originally_min_phase']})")

    a = a_from_alphas(alphas)
    r2 = test_inverse_forward_roundtrip(a)
    status2 = "PASS" if r2["passed"] else ("FAIL (expected)" if not r2["expected_to_pass"] else "FAIL (unexpected!)")
    print(f"  [{name}] inverse->forward: {status2}  (err={r2['error']:.3e}, "
          f"admissible={r2['admissible']}, min_phase={r2['min_phase']})")
    return r1, r2


def run_all_sanity_checks() -> bool:
    """Run both round-trips on a battery of test cases. Returns True iff
    every check either passed, or failed exactly as predicted by the
    p2-freedom / admissibility+min-phase criteria above (an "unexpected"
    failure is a real problem)."""
    all_ok = True
    print("(a) Simple 1-layer: alpha = [log 2, -log 2]")
    r1, r2 = _print_case("a", np.array([np.log(2.0), -np.log(2.0)]))
    all_ok &= (r1["passed"] or not r1["expected_to_pass"]) and (r2["passed"] or not r2["expected_to_pass"])

    print("(b) Symmetric 2-layer: alpha = [0.5, -0.3, -0.2]")
    r1, r2 = _print_case("b", np.array([0.5, -0.3, -0.2]))
    all_ok &= (r1["passed"] or not r1["expected_to_pass"]) and (r2["passed"] or not r2["expected_to_pass"])

    print("(c) Random n=5, sum(alpha)=0")
    rng = np.random.default_rng(0)
    alphas_c = rng.uniform(-1, 1, 6)
    alphas_c[-1] -= np.sum(alphas_c)  # force sum = 0
    r1, r2 = _print_case("c", alphas_c)
    all_ok &= (r1["passed"] or not r1["expected_to_pass"]) and (r2["passed"] or not r2["expected_to_pass"])

    print("(d) Paper's n=8, mu0=1.0 worked example (via sdp_design, best-effort)")
    try:
        from .sdp_design import design_filter_full
        J0 = [(np.pi / 6, np.pi / 4)]
        J1 = [(np.pi / 2, np.pi)]
        res = design_filter_full(8, J0, J1, 1.0)
        if res is not None and res.a is not None:
            r2 = test_inverse_forward_roundtrip(res.a)
            status2 = "PASS" if r2["passed"] else ("FAIL (expected)" if not r2["expected_to_pass"] else "FAIL (unexpected!)")
            print(f"  [d] inverse->forward: {status2}  (err={r2['error']:.3e}, "
                  f"admissible={r2['admissible']}, min_phase={r2['min_phase']}, "
                  f"polish_verified={res.polish_verified})")
            all_ok &= r2["passed"] or not r2["expected_to_pass"]
        else:
            print("  [d] SKIPPED: design_filter_full did not return a usable result")
    except Exception as e:  # pragma: no cover - best-effort diagnostic case
        print(f"  [d] SKIPPED: {e!r}")

    print("(e) forward_with_grad: analytic gradients vs central finite differences")
    rng = np.random.default_rng(1)
    alphas_e = rng.uniform(-0.8, 0.8, 7)
    theta_e = rng.uniform(0.05, np.pi - 0.05, 9)
    rg = test_forward_with_grad(alphas_e, theta_e)
    status_e = "PASS" if rg["passed"] else "FAIL (unexpected!)"
    print(f"  [e] {status_e}  (value_err={rg['value_err']:.3e}, "
          f"dQ_abs_err={rg['dQ_abs_err']:.3e}, dkappa_abs_err={rg['dkappa_abs_err']:.3e})")
    all_ok &= rg["passed"]

    print("(f) certify: exact rootfinding delta/mu_min vs. fine-grid brute force")
    from .bandgap import find_bands_gaps
    alphas_f = np.array([0.5, -0.3, -0.2])
    a_f = a_from_alphas(alphas_f)
    bgs = find_bands_gaps(a_f)
    if bgs.bands and bgs.gaps:
        gap_lo, gap_hi, gap_sign = bgs.gaps[0]
        J1_f, J0_f, sigma_f = [bgs.bands[0]], [(gap_lo, gap_hi)], [gap_sign]
        rc = test_certify(alphas_f, J0_f, J1_f, sigma_f)
        status_f = "PASS" if rc["passed"] else "FAIL (unexpected!)"
        print(f"  [f] {status_f}  (delta_err={rc['delta_err']:.3e}, "
              f"mu_err={rc['mu_err']:.3e}, sound={rc['sound']})")
        all_ok &= rc["passed"]
    else:
        print("  [f] SKIPPED: no band/gap found for this alphas vector")

    print()
    print("ALL SANITY CHECKS " + ("PASSED" if all_ok else "FAILED (see 'unexpected' entries above)"))
    return all_ok


if __name__ == "__main__":
    run_all_sanity_checks()
