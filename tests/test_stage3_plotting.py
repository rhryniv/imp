"""Stage 3 deliverable 3 (spec Sec. 9): plot_design / plot_design_from_record
/ plot_bound_vs_achieved. Runs with the Agg backend (no display needed);
checked structurally (axes/lines/files exist with the right shape), not
pixel-compared, since there is no ground-truth image to compare against.

Run: python3 -m pytest tests/test_stage3_plotting.py -v   (from repo root)
"""
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pytest

from scattering.plotting import plot_design, plot_design_from_record, plot_bound_vs_achieved
from scattering.driver import DegreeRecord


ALPHA = np.array([0.5, -0.3, -0.2])
I0 = [(1.0, 1.5)]
I1 = [(2.0, 2.5)]


def test_plot_design_returns_two_panels_with_shaded_bands():
    fig = plot_design(ALPHA, I0, I1, N_values=(5, 10), mu0=0.05)
    assert len(fig.axes) == 2
    kappa_ax, TN_ax = fig.axes
    assert len(kappa_ax.lines) >= 1          # kappa_B curve (+ reference lines)
    assert len(TN_ax.lines) == 2             # one per N in N_values
    # I0/I1 shading: axvspan adds a PolyCollection-like patch per axis per interval
    assert len(kappa_ax.patches) == len(I0) + len(I1)
    assert len(TN_ax.patches) == len(I0) + len(I1)


def test_plot_design_legend_present_iff_N_values_given():
    fig_with = plot_design(ALPHA, I0, I1, N_values=(5,))
    fig_without = plot_design(ALPHA, I0, I1, N_values=())
    assert fig_with.axes[1].get_legend() is not None
    assert fig_without.axes[1].get_legend() is None


def test_plot_design_saves_to_file(tmp_path):
    path = str(tmp_path / "design.png")
    plot_design(ALPHA, I0, I1, N_values=(5,), savepath=path)
    assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_design_from_record_matches_plot_design():
    rec = DegreeRecord(
        n=2, status="optimal", underline_delta=1e-3, dual_ray_found=False,
        dual_status="dual_certified", dual_blocks_all_feasible=True, delta_achieved=0.1,
        kappa_min=1.5, mu_min=0.9, s_0=0.5, sigma_star=(-1,), kappa_min_by_sigma={},
        admissible=False, N_required=10.0, alpha=list(ALPHA), rho=[1.0, 1.1, 1.0, 0.9],
        start_origin="phase1", grid_vs_exact={}, gamma=1.2, n_min_green=3.0,
        time_direct_s=0.1, time_sdp_s=0.1,
    )
    fig = plot_design_from_record(rec, I0, I1, N_values=(5,))
    assert len(fig.axes) == 2


def test_plot_design_from_record_raises_without_alpha():
    rec = DegreeRecord(
        n=2, status="phase1_infeasible", underline_delta=None, dual_ray_found=True,
        dual_status="dual_ray_found", dual_blocks_all_feasible=True, delta_achieved=None,
        kappa_min=None, mu_min=None, s_0=None, sigma_star=None, kappa_min_by_sigma={},
        admissible=False, N_required=None, alpha=None, rho=None,
        start_origin=None, grid_vs_exact={}, gamma=None, n_min_green=None,
        time_direct_s=0.1, time_sdp_s=0.1,
    )
    with pytest.raises(ValueError):
        plot_design_from_record(rec, I0, I1)


def _fake_record(n, ud, da, ng, admissible):
    return DegreeRecord(
        n=n, status="optimal", underline_delta=ud, dual_ray_found=False,
        dual_status="dual_certified", dual_blocks_all_feasible=True, delta_achieved=da,
        kappa_min=1.5, mu_min=0.9, s_0=0.5, sigma_star=(-1,), kappa_min_by_sigma={},
        admissible=admissible, N_required=10.0, alpha=[0.1, -0.1], rho=[1.0, 1.1, 1.0],
        start_origin="phase1", grid_vs_exact={}, gamma=1.2, n_min_green=ng,
        time_direct_s=0.1, time_sdp_s=0.1,
    )


def test_plot_bound_vs_achieved_plots_both_curves_and_reference_lines():
    records = [_fake_record(3, 1e-3, 10.0, 2.1, False),
               _fake_record(4, 1e-4, 0.05, 3.4, True),
               _fake_record(5, 1e-5, 0.001, 4.9, True)]
    fig = plot_bound_vs_achieved(records, retained=records[1])
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    # underline_delta line, delta_achieved line, n_min_green(n) line,
    # retained n_min_green vline, retained n vline = 5 Line2D artists
    assert len(ax.lines) == 5
    labels = [line.get_label() for line in ax.lines]
    assert any("underline" in l or r"\underline" in l for l in labels)


def test_plot_bound_vs_achieved_handles_missing_retained():
    records = [_fake_record(3, 1e-3, 10.0, 2.1, False),
               _fake_record(4, 1e-4, 0.05, 3.4, True)]
    fig = plot_bound_vs_achieved(records, retained=None)  # must not raise
    assert len(fig.axes[0].lines) == 3   # no vlines without a retained record


def test_plot_bound_vs_achieved_skips_nonpositive_and_none_values():
    records = [_fake_record(3, None, None, None, False),
               _fake_record(4, 1e-4, 0.05, 3.4, True)]
    fig = plot_bound_vs_achieved(records)  # must not raise on the n=3 gaps
    ax = fig.axes[0]
    # only n=4 contributes to underline/achieved lines
    for line in ax.lines:
        xdata = list(line.get_xdata())
        if xdata:
            assert 3 not in xdata or line.get_label().startswith("_")


def test_plot_bound_vs_achieved_saves_to_file(tmp_path):
    records = [_fake_record(3, 1e-3, 10.0, 2.1, False),
               _fake_record(4, 1e-4, 0.05, 3.4, True)]
    path = str(tmp_path / "scan.png")
    plot_bound_vs_achieved(records, retained=records[1], savepath=path)
    assert os.path.exists(path) and os.path.getsize(path) > 0
