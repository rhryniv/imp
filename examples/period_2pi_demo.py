"""Same experiment as band_informed_demo.py, but exploiting the true period:
kappa_B, |q~_1|^2 and T_N are 2*pi-periodic in theta (not pi-periodic --
that would require every odd-indexed a_m to vanish, which is not the
generic case), and additionally even about theta=0 *and* mirror-symmetric
about theta=pi (kappa_B(2*pi-theta) = kappa_B(theta)). So this run targets:

  I1 (pass) centered at theta=0:  [-theta1, theta1]
  I0 (stop) centered at theta=pi: [pi-theta0, pi+theta0]

I0 centered at theta=0 is impossible instead (not just hard): constraint
(D) forces q~_1(0)=1 exactly, so kappa_B(0)=1 and T_N(0)=1 for *every*
valid design -- the paper's own design problem requires 0 not in I0.  This
run puts the pass band where the physics requires it (0) and the stop
band where our baseline structure's actual gap sits (near pi).

Since kappa_B/G/T_N are genuinely functions of x=cos(theta) alone, and
cos is even and 2*pi-periodic, a theta-interval straddling 0 or pi maps,
in x-space, to *exactly* the same set as its one-sided half:
  [-theta1, theta1]   <->   [0, theta1]        (same x-range, cos even)
  [pi-theta0, pi+theta0] <-> [pi-theta0, pi]    (cos mirror-symmetric about pi)
so the optimizer (design_via_layers) is called with the one-sided,
[0,pi]-contained form -- no changes needed there -- while the *plots*
use the full [-pi, pi] range and the genuinely straddling intervals for
shading, to actually show the periodic structure the theta in [0,pi]
view hides.
"""
import numpy as np

from scattering.forward import a_from_alphas, transmission_TN, plot_filter
from scattering.bandgap import find_bands_gaps
from scattering.sdp_design import design_via_layers


def main():
    p1_contrast = 2.0
    alpha0 = np.log(p1_contrast)
    alphas_simple = np.array([alpha0, -alpha0])
    a_simple = a_from_alphas(alphas_simple)
    print(f"Baseline: 1-layer block, p_1={p1_contrast} (alphas={np.round(alphas_simple, 4)})")

    bg = find_bands_gaps(a_simple)
    band_lo, band_hi = bg.bands[0]
    gap_lo, gap_hi, gap_sign = bg.gaps[0]
    print(f"  band (near 0)  = [{band_lo:.4f}, {band_hi:.4f}]")
    print(f"  gap  (near pi) = [{gap_lo:.4f}, {gap_hi:.4f}]")

    theta1 = 0.2    # I1 half-width, centered at theta=0
    theta0 = 0.15   # I0 half-width, centered at theta=pi
    I1_opt = [(0.0, theta1)]              # optimizer form (x-equivalent to [-theta1, theta1])
    I0_opt = [(np.pi - theta0, np.pi)]    # optimizer form (x-equivalent to [pi-theta0, pi+theta0])
    I1_plot = [(-theta1, theta1)]         # true straddling form, for shading only
    I0_plot = [(np.pi - theta0, np.pi + theta0)]
    print(f"  I1 (pass) = [{-theta1:.3f}, {theta1:.3f}]  (centered at 0)")
    print(f"  I0 (stop) = [{np.pi - theta0:.3f}, {np.pi + theta0:.3f}]  (centered at pi)")

    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_values = (1, 3, 5)

    th1 = np.linspace(*I1_plot[0], 4000)
    th0 = np.linspace(*I0_plot[0], 4000)
    print("\n--- Baseline (n=1) T_N ---")
    baseline_stop = {N: float(transmission_TN(a_simple, th0, N).max()) for N in N_values}
    baseline_pass = {N: float(transmission_TN(a_simple, th1, N).min()) for N in N_values}
    for N in N_values:
        print(f"  N={N:3d}:  max T_N on I0 = {baseline_stop[N]:.3e}   "
              f"min T_N on I1 = {baseline_pass[N]:.5f}")

    print("\n--- Optimized designs (design_via_layers: realizable by construction) ---")
    results = {}
    smaller_solutions = {}
    for n in (3, 5, 7, 9):
        res = design_via_layers(n, I0_opt, I1_opt, mu0, use_sdp_warm_start=False, n_restarts=15,
                                 smaller_solutions=smaller_solutions)
        if res.status != "optimal":
            print(f"\nn={n}: {res.status}")
            continue
        results[n] = res
        smaller_solutions[n] = res.alphas

        print(f"\nn={n}: delta1={res.delta1:.4e}  sigma={res.sigma}  "
              f"achieved_mu={res.achieved_mu:.4f} (target {mu0})")
        print(f"  alphas = {np.round(res.alphas, 4)}  (sum={np.sum(res.alphas):.2e})")
        print(f"  impedances p_0..p_{n + 1} = {np.round(res.impedances, 4)}")
        for N in N_values:
            tn_stop = float(transmission_TN(res.a, th0, N).max())
            tn_pass = float(transmission_TN(res.a, th1, N).min())
            print(f"  N={N:3d}:  max T_N on I0 = {tn_stop:.3e}  (baseline {baseline_stop[N]:.3e})   "
                  f"min T_N on I1 = {tn_pass:.5f}  (baseline {baseline_pass[N]:.5f})")

    if results:
        n_best = max(results)
        res_best = results[n_best]
        print(f"\nPlotting baseline (n=1) vs optimized (n={n_best}), theta in [-pi, pi] ...")
        theta_range = (-np.pi + 1e-3, np.pi - 1e-3)
        plot_filter(a_simple, N_values, theta_range=theta_range, I0=I0_plot, I1=I1_plot,
                    savepath="baseline_n1_2pi.png")
        plot_filter(res_best.a, N_values, theta_range=theta_range, I0=I0_plot, I1=I1_plot,
                    savepath=f"optimized_n{n_best}_2pi.png")
        print(f"  saved baseline_n1_2pi.png, optimized_n{n_best}_2pi.png")
    else:
        print("\nNo design succeeded across the n values tried; skipping plots.")


if __name__ == "__main__":
    main()
