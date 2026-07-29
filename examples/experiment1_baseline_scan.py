"""Experiment 1 for the paper (spec Sec. 8, "Two experiments for the
paper"): start from the hand-picked p=(2,sqrt(2),2) baseline (n=3), then
three levels of improvement using the NEW pipeline throughout
(design_sdp_magnitude / design_direct / certify) -- NOT the old
design_via_layers/design_filter_full pipeline that three_level_demo.py
used, and NOT validated against the manuscript's own worked-example table
values (those were computed under the now-purged sqrt(u)-1 convention and
are not treated as ground truth here; see scattering/experiment.py's
module docstring).

  (a) unoptimised: p_1=p_3=2, p_2=sqrt(2), read I0/I1 off this baseline's
      own band/gap geometry (bandgap.find_bands_gaps), a moderate
      sub-window of each (same windowing rule as the earlier
      band_informed_demo.py/three_level_demo.py: a degree-n design has
      ~n natural oscillations across [0,pi], so demanding good behaviour
      across an *entire* gap/band is atypical/often infeasible, more so
      at small n).
  (b) optimise at fixed n=3 (design_direct).
  (c) full scan over n (scattering.experiment.degree_scan_direct), best
      degree kept (not first feasible).

Emits kappa and T_N (N=1,3,5) for (a) and (c) on a common theta-grid, with
I0/I1 marked, as a 2x2 figure -- and the full per-degree JSON record for
the Level-3 scan (spec Sec. 8's own schema).

Run from the repository root: python3 examples/experiment1_baseline_scan.py
"""
import numpy as np
import matplotlib.pyplot as plt

from scattering.forward import a_from_alphas, kappa_B, transmission_TN
from scattering.bandgap import find_bands_gaps
from scattering.sdp_design import design_direct
from scattering.certify import certify
from scattering.experiment import degree_scan_direct, best_degree_record, save_records, check_delta_mag_monotone


def main():
    # --- (a) unoptimised baseline: n=3, p_1=p_3=2, p_2=sqrt(2) ---
    p1, p2, p3 = 2.0, np.sqrt(2.0), 2.0
    alphas_a = np.array([np.log(p1), np.log(p2 / p1), np.log(p3 / p2), np.log(1.0 / p3)])
    print(f"(a) baseline impedances p_0..p_4 = [1, {p1}, {p2:.4f}, {p3}, 1]")
    print(f"    alphas = {np.round(alphas_a, 4)}  (sum={np.sum(alphas_a):.2e})")
    a_a = a_from_alphas(alphas_a)

    bg = find_bands_gaps(a_a)
    band_lo, band_hi = bg.bands[0]
    gap_lo, gap_hi, gap_sign = bg.gaps[0]
    print(f"    full gap  = [{gap_lo:.4f}, {gap_hi:.4f}]  full band = [{band_lo:.4f}, {band_hi:.4f}]")

    I0 = [(gap_lo + 0.70 * (gap_hi - gap_lo), gap_lo + 0.85 * (gap_hi - gap_lo))]
    I1 = [(band_lo + 0.40 * (band_hi - band_lo), band_lo + 0.60 * (band_hi - band_lo))]
    print(f"    I0 (stop) = {I0}")
    print(f"    I1 (pass) = {I1}")

    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_values = (1, 3, 5)

    # --- (b) optimise at fixed n=3 ---
    print("\n(b) design_direct at fixed n=3")
    res_b = design_direct(3, I0, I1, mu0, seed=0)
    print(f"    status={res_b.status} delta={res_b.delta} sigma={res_b.sigma} "
          f"achieved_mu={res_b.achieved_mu:.4f}")

    # --- (c) full scan over n ---
    print("\n(c) degree scan, n=3..9")
    records = degree_scan_direct(range(3, 10), I0, I1, mu0, seed=0, N_values=N_values)
    for r in records:
        print(f"    n={r.n}: status={r.status}  delta_mag={r.delta_mag}  "
              f"delta_achieved={r.delta_achieved}  delta_certified={r.delta_certified}")
    assert check_delta_mag_monotone(records), "delta_mag(n) should be monotone non-increasing -- internal consistency failure"
    print("    delta_mag(n) monotonicity check: OK")

    best = best_degree_record(records)
    if best is None:
        raise RuntimeError("no degree in the scan reached status=optimal")
    print(f"\n    best: n={best.n}, delta_achieved={best.delta_achieved:.4e}, delta_certified={best.delta_certified:.4e}")
    alphas_c = np.asarray(best.alpha, dtype=float)
    a_c = a_from_alphas(alphas_c)
    sigma_c = tuple(best.sigma)

    save_records(records, "experiment1_level3_scan.json")
    print("    saved experiment1_level3_scan.json")

    # certify (a) has no target sigma of its own -- report kappa/T_N only,
    # not a certified delta/mu_min (I0/I1 were read off *its* geometry,
    # they were never a design target for it).
    cert_c = certify(alphas_c, I0, I1, sigma_c, N_values=N_values)
    print(f"\n    certify(c): delta={cert_c.delta:.6e}  mu_min={cert_c.mu_min:.6f}  "
          f"TN_stop_bound={cert_c.TN_stop_bound}  TN_pass_bound={cert_c.TN_pass_bound}")

    # --- 2x2 figure: kappa/T_N for (a) and (c), common theta-grid ---
    theta = np.linspace(1e-3, np.pi - 1e-3, 4000)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for col, (label, a_vec) in enumerate([("(a) unoptimised, n=3", a_a), (f"(c) best scan, n={best.n}", a_c)]):
        kap = kappa_B(a_vec, theta)
        ax_k = axes[0, col]
        ax_k.plot(theta, kap, color="black")
        ax_k.axhline(1.0, color="gray", lw=0.7, ls="--")
        ax_k.axhline(-1.0, color="gray", lw=0.7, ls="--")
        ax_k.set_title(label)
        ax_k.set_ylabel(r"$\kappa_B(\theta)$")

        ax_T = axes[1, col]
        for N in N_values:
            ax_T.plot(theta, transmission_TN(a_vec, theta, N), label=f"N={N}")
        ax_T.set_ylabel(r"$T_N(\theta)$")
        ax_T.set_xlabel(r"$\theta = 2kh$")
        ax_T.legend(fontsize=8)

        for ax in (ax_k, ax_T):
            for lo, hi in I0:
                ax.axvspan(lo, hi, color="red", alpha=0.15)
            for lo, hi in I1:
                ax.axvspan(lo, hi, color="green", alpha=0.15)

    fig.tight_layout()
    fig.savefig("experiment1_2x2.png", dpi=150)
    print("\n    saved experiment1_2x2.png")


if __name__ == "__main__":
    main()
