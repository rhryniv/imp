"""Stage 3, spec Sec. 5 rule 2 ("report dual quantities"):
dual_certify.solve_magnitude_sdp_with_duals -- explicit dual-feasible
points / dual improving rays for the magnitude SDP, per manuscript
Remark rem:certificate (Slater fails for this problem structurally, so
the raw primal objective is never a certified bound).

Run: python3 -m pytest tests/test_stage3_dual_certify.py -v   (from repo root)
"""
import numpy as np
import cvxpy as cp
import pytest

from scattering.dual_certify import (
    _hankel, _localize, _block_dual_matrices, solve_magnitude_sdp_with_duals,
)
from scattering.poly_sdp import interval_nonneg_constraints
from scattering.sdp_design import design_sdp_magnitude


def test_hankel_adjoint_matches_sos_coeffs_inner_product():
    """<sigma(Y), y> == <Y, Hankel(y)> for random Y, y (defining property
    of an adjoint map, checked directly rather than trusting the
    derivation by inspection)."""
    from scattering.poly_sdp import _sos_coeffs
    rng = np.random.default_rng(0)
    m = 3
    Yv = cp.Variable((m + 1, m + 1), symmetric=True)
    Y_val = rng.standard_normal((m + 1, m + 1))
    Y_val = (Y_val + Y_val.T) / 2
    Yv.value = Y_val
    sigma_val = _sos_coeffs(Yv, m).value
    y = rng.standard_normal(2 * m + 1)
    lhs = float(sigma_val @ y)
    rhs = float(np.sum(Y_val * _hankel(y, m + 1)))
    assert abs(lhs - rhs) < 1e-9


def test_localize_adjoint_matches_poly_mul_fixed():
    from scattering.poly_sdp import _poly_mul_fixed
    rng = np.random.default_rng(1)
    weight = np.array([-0.3, 0.7, -1.0])
    q_deg = 4
    var = cp.Variable(q_deg + 1)
    var_val = rng.standard_normal(q_deg + 1)
    var.value = var_val
    out_val = _poly_mul_fixed(weight, var, q_deg).value
    y = rng.standard_normal(len(out_val))
    lhs = float(out_val @ y)
    rhs = float(var_val @ _localize(y, weight))
    assert abs(lhs - rhs) < 1e-9


@pytest.mark.parametrize("deg,a,b,target", [
    (2, -1.0, 1.0, [-0.3, 0.0, 1.0]),
    (4, -1.0, 1.0, [0.1, 0.0, -2.0, 0.0, 1.0]),
    (1, -1.0, 1.0, [-0.5, 2.0]),
    (3, -1.0, 1.0, [0.0, -0.2, 0.0, 1.0]),
    (5, -2.0, 1.0, [0.0, 0.1, 0.0, -0.5, 0.0, 1.0]),
])
def test_explicit_dual_matches_true_optimum_and_is_psd(deg, a, b, target):
    """min t s.t. target + t*e0 nonneg on [a,b]: true optimum is
    -min(target on [a,b]) in closed form (grid-verified); the explicit
    dual point constructed here (sign convention: y = -constraint's own
    CVXPY dual) must reproduce it exactly (weak duality is tight for
    this toy problem) AND its own Hankel/localizing matrices must be
    PSD -- both checked, not just the bound value."""
    target = np.asarray(target, dtype=float)
    t = cp.Variable()
    e0 = np.zeros(deg + 1)
    e0[0] = 1.0
    coeff_expr = target + t * e0
    cons, _ = interval_nonneg_constraints(coeff_expr, deg, a, b)
    eq = cons[-1]
    prob = cp.Problem(cp.Minimize(t), cons)
    prob.solve(solver="CLARABEL")

    xs = np.linspace(a, b, 400_000)
    t_true = -float(np.polyval(target[::-1], xs).min())

    y = -np.asarray(eq.dual_value, dtype=float).flatten()
    b_vec = -target
    underline = float(b_vec @ y)

    mats = _block_dual_matrices(y, deg, a, b)
    min_eig = min(float(np.linalg.eigvalsh(M).min()) for M in mats)

    assert abs(underline - t_true) < 1e-5
    assert min_eig >= -1e-6


def test_explicit_dual_ray_certifies_a_genuinely_negative_polynomial():
    """A polynomial that is negative everywhere on the interval (x^2-4 on
    [-1,1], max value -3): the SDP asking for it to be nonneg is
    infeasible, and the explicit dual ray constructed the same way as
    the optimal-case dual point must (a) have all its Hankel/localizing
    matrices PSD and (b) score a strictly positive certificate value --
    the Farkas-type infeasibility witness, independent of any objective."""
    deg, a, b = 2, -1.0, 1.0
    coeff = np.array([-4.0, 0.0, 1.0])
    coeff_var = cp.Parameter(deg + 1)
    coeff_var.value = coeff
    cons, _ = interval_nonneg_constraints(coeff_var, deg, a, b)
    eq = cons[-1]
    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver="CLARABEL")
    assert prob.status in ("infeasible", "infeasible_inaccurate")

    y = -np.asarray(eq.dual_value, dtype=float).flatten()
    b_vec = -coeff
    certificate_value = float(b_vec @ y)
    mats = _block_dual_matrices(y, deg, a, b)
    min_eig = min(float(np.linalg.eigvalsh(M).min()) for M in mats)

    assert certificate_value > 1e-6
    assert min_eig >= -1e-6


I0 = [(np.pi / 6, np.pi / 4)]
I1 = [(np.pi / 2, 2.5)]
MU0 = 0.05


@pytest.mark.parametrize("n", [3, 5, 8])
def test_dual_certified_bound_never_exceeds_raw_primal_delta(n):
    """Rule 2's own point: underline_delta is a RIGOROUS lower bound on
    delta_mag(n) by weak duality (regardless of whether the raw solver
    value equals the true optimum) -- checked here against
    design_sdp_magnitude's own raw (uncertified) primal value as a
    sanity/consistency cross-check, on a real instance drawn from the
    spec's own Sec. 7 parameters."""
    dual = solve_magnitude_sdp_with_duals(n, I0, I1, MU0)
    raw = design_sdp_magnitude(n, I0, I1, MU0)

    assert dual.status == "dual_certified"
    assert raw.status == "optimal"
    assert dual.underline_delta is not None
    for bc in dual.block_checks:
        assert bc.feasible, f"block {bc.name} min_eig={bc.min_eig}"
    assert dual.underline_delta <= raw.delta_mag + 1e-6


def test_dual_ray_found_on_a_genuinely_infeasible_instance():
    """A degree/parameter combination for which design_sdp_magnitude
    itself reports 'infeasible' (I0 hugging theta=0, demanding mu0, low
    degree -- confirmed empirically to be genuinely infeasible, not a
    solver artifact, precisely because the dual ray construction below
    independently certifies it): solve_magnitude_sdp_with_duals must
    report a genuine dual_ray_found, with every block's explicit witness
    matrix PSD."""
    I0_tight = [(0.02, 0.04)]
    I1_narrow = [(1.5, 1.8)]
    n, mu0 = 2, 20.0

    raw = design_sdp_magnitude(n, I0_tight, I1_narrow, mu0)
    assert raw.status == "infeasible"

    dual = solve_magnitude_sdp_with_duals(n, I0_tight, I1_narrow, mu0)
    assert dual.status == "dual_ray_found"
    assert dual.dual_ray_found is True
    for bc in dual.block_checks:
        assert bc.feasible, f"block {bc.name} min_eig={bc.min_eig}"


def test_never_reports_raw_primal_as_the_bound():
    """Structural check on the dataclass/return contract itself: a
    dual_certified result's underline_delta must be a DIFFERENT
    quantity from raw_primal_delta (not merely copied through) -- rule 2
    forbids reporting the primal objective as if it were the bound."""
    dual = solve_magnitude_sdp_with_duals(5, I0, I1, MU0)
    assert dual.status == "dual_certified"
    assert dual.underline_delta != dual.raw_primal_delta
    assert dual.underline_delta < dual.raw_primal_delta
