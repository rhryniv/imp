"""Experiment 2 for the paper (spec Sec. 8): prescribed multi-band spec
(m_0=2), no starting structure -- scan, emit best design and one T_N plot.
Uses the NEW pipeline throughout (design_sdp_magnitude / design_direct /
certify via scattering.experiment), not design_via_layers/
design_filter_full.

I0/I1 reused from the earlier multiband_demo.py/multiband_continuation_demo.py
investigation (two disjoint stop bands, two disjoint pass bands, m_0=2)
since that is a genuine "prescribed multi-band spec" independent of any
single baseline's own geometry -- and it is a known-informative test case
for design_direct: n=9 was independently confirmed (three separate times
across this project, including this refactor's own design_direct) to be a
genuine dead zone for it, so the scan is expected to show delta_achieved
NOT monotonic in n even though delta_mag(n) (the SDP relaxation bound) is.

Run from the repository root: python3 examples/experiment2_multiband.py
"""
import numpy as np
import matplotlib.pyplot as plt

from scattering.forward import a_from_alphas, transmission_TN
from scattering.certify import certify
from scattering.experiment import degree_scan_direct, best_degree_record, save_records, check_delta_mag_monotone


def main():
    I0 = [(0.50, 0.65), (2.50, 2.65)]
    I1 = [(1.20, 1.40), (1.80, 2.00)]
    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_values = (1, 3, 5)
    print(f"I0 (stop, m_0={len(I0)}) = {I0}")
    print(f"I1 (pass) = {I1}")
    print(f"mu0 = {mu0}")

    n_range = (3, 5, 7, 9, 11)
    print(f"\nScanning n = {n_range} ...")
    records = degree_scan_direct(n_range, I0, I1, mu0, seed=0, N_values=N_values)
    for r in records:
        print(f"  n={r.n}: status={r.status}  delta_mag={r.delta_mag}  "
              f"delta_achieved={r.delta_achieved}  delta_certified={r.delta_certified}  "
              f"mu_min_certified={r.mu_min_certified}  sigma={r.sigma}")

    mono_ok = check_delta_mag_monotone(records)
    print(f"\ndelta_mag(n) monotonicity / delta_mag<=delta_achieved check: {'OK' if mono_ok else 'VIOLATED (unexpected!)'}")

    best = best_degree_record(records)
    if best is None:
        raise RuntimeError("no degree in the scan reached status=optimal")
    print(f"\nbest: n={best.n}, delta_achieved={best.delta_achieved:.4e}, "
          f"delta_certified={best.delta_certified:.4e}, mu_min_certified={best.mu_min_certified:.4f}")

    save_records(records, "experiment2_scan.json")
    print("saved experiment2_scan.json")

    alphas_best = np.asarray(best.alpha, dtype=float)
    a_best = a_from_alphas(alphas_best)
    sigma_best = tuple(best.sigma)
    cert = certify(alphas_best, I0, I1, sigma_best, N_values=N_values)
    print(f"\ncertify(best): delta={cert.delta:.6e}  mu_min={cert.mu_min:.6f}  phi_min={cert.phi_min:.6f}")
    print(f"  TN_stop_bound={cert.TN_stop_bound}")
    print(f"  TN_pass_bound={cert.TN_pass_bound}")

    # --- T_N plot for the best design ---
    # Full range [-0.05, 2*pi+0.05]: kappa_B/Q/T_N are all even and
    # 2*pi-periodic in theta, so forward.py's own formulas are already
    # valid there with no special-casing -- this just displays more of one
    # already-defined periodic function, not a new domain.
    theta = np.linspace(-0.05, 2 * np.pi + 0.05, 8000)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for N in N_values:
        ax.plot(theta, transmission_TN(a_best, theta, N), label=f"N={N}")
    for lo, hi in I0:
        ax.axvspan(lo, hi, color="red", alpha=0.15)
    for lo, hi in I1:
        ax.axvspan(lo, hi, color="green", alpha=0.15)
    ax.set_xlim(-0.05, 2 * np.pi + 0.05)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"$\theta = 2kh$")
    ax.set_ylabel(r"$T_N(\theta)$")
    ax.set_title(f"Experiment 2: best design n={best.n}, delta={best.delta_achieved:.3e}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("experiment2_TN.png", dpi=150)
    print("saved experiment2_TN.png")


if __name__ == "__main__":
    main()
