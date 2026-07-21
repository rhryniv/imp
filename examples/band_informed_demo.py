"""Band-informed design demo (v2 -- corrected framing).

  1. Simple 1-layer structure, contrast p_1 > 1: alphas = [log p_1, -log p_1].
     Its Floquet discriminant kappa_B has exactly one gap and one band over
     the fundamental domain theta in [0, pi] (Fig. 2 of the paper).
  2. Target a *sub-window* of that gap/band, not the whole thing. A degree-n
     design has on the order of n natural oscillations of kappa_B across
     [0, pi] (confirmed empirically: random n=7 structures have a *median*
     of 14 bands+gaps total); demanding good behavior across almost the
     entire gap/band forces an atypical, near-global single-band/single-gap
     topology that becomes vanishingly rare as n grows -- pure random search
     found 0/20000 feasible n=7 points against the full-width targets used
     in the first version of this demo. A moderate sub-window sidesteps
     that entirely, and is also the physically sensible way to spend extra
     layers: sharpen performance over the region you actually care about,
     rather than fight the polynomial's natural oscillation count.
  3. For several n > 1, solve directly for a realizable layer sequence via
     sdp_design.design_via_layers (the gamma_j = tanh(alpha_j)
     reparametrization -- every returned design is physically buildable by
     construction, no reflection step, no risk of losing constraint (C) to
     it), and compare T_N against the baseline at matched N.

Run from the repository root: python3 examples/band_informed_demo.py
"""
import numpy as np

from scattering.forward import a_from_alphas, transmission_TN, plot_filter
from scattering.bandgap import find_bands_gaps
from scattering.sdp_design import design_via_layers


def main():
    # --- Step 1: simple 1-layer baseline structure ---
    p1_contrast = 2.0
    alpha0 = np.log(p1_contrast)
    alphas_simple = np.array([alpha0, -alpha0])
    a_simple = a_from_alphas(alphas_simple)
    print(f"Baseline: 1-layer block, p_1={p1_contrast} (alphas={np.round(alphas_simple, 4)})")

    bg = find_bands_gaps(a_simple)
    gap_lo, gap_hi, gap_sign = bg.gaps[0]
    band_lo, band_hi = bg.bands[0]
    print(f"  full gap  = [{gap_lo:.4f}, {gap_hi:.4f}]  (width {gap_hi - gap_lo:.4f})")
    print(f"  full band = [{band_lo:.4f}, {band_hi:.4f}]  (width {band_hi - band_lo:.4f})")

    # --- Step 2: a moderate sub-window of each, not the whole gap/band ---
    I0 = [(gap_lo + 0.70 * (gap_hi - gap_lo), gap_lo + 0.85 * (gap_hi - gap_lo))]
    I1 = [(band_lo + 0.40 * (band_hi - band_lo), band_lo + 0.60 * (band_hi - band_lo))]
    print(f"  I0 (stop) = {I0}  (width {I0[0][1] - I0[0][0]:.4f})")
    print(f"  I1 (pass) = {I1}  (width {I1[0][1] - I1[0][0]:.4f})")

    # mu0 barely matters (Prop. 5.4a: any mu0>0 is reached for large enough
    # N), so a small, easy target is used; N does the work of reaching a
    # given attenuation level afterwards (Step 4 of the synthesis algorithm).
    mu0 = 0.05
    N_values = (1, 3, 5)

    th0 = np.linspace(*I0[0], 4000)
    th1 = np.linspace(*I1[0], 4000)
    print("\n--- Baseline (n=1) T_N ---")
    baseline_stop = {N: float(transmission_TN(a_simple, th0, N).max()) for N in N_values}
    baseline_pass = {N: float(transmission_TN(a_simple, th1, N).min()) for N in N_values}
    for N in N_values:
        print(f"  N={N:3d}:  max T_N on I0 = {baseline_stop[N]:.3e}   "
              f"min T_N on I1 = {baseline_pass[N]:.5f}")

    # --- Step 3: design_via_layers for several n, compare ---
    print("\n--- Optimized designs (design_via_layers: realizable by construction) ---")
    results = {}
    for n in (3, 5, 7, 9):
        res = design_via_layers(n, I0, I1, mu0, use_sdp_warm_start=False, n_restarts=15)
        if res.status != "optimal":
            print(f"\nn={n}: {res.status}")
            continue
        results[n] = res

        print(f"\nn={n}: delta1={res.delta1:.4e}  sigma={res.sigma}  "
              f"achieved_mu={res.achieved_mu:.4f} (target {mu0})")
        print(f"  alphas = {np.round(res.alphas, 4)}  (sum={np.sum(res.alphas):.2e})")
        print(f"  impedances p_0..p_{n + 1} = {np.round(res.impedances, 4)}")
        for N in N_values:
            tn_stop = float(transmission_TN(res.a, th0, N).max())
            tn_pass = float(transmission_TN(res.a, th1, N).min())
            print(f"  N={N:3d}:  max T_N on I0 = {tn_stop:.3e}  (baseline {baseline_stop[N]:.3e})   "
                  f"min T_N on I1 = {tn_pass:.5f}  (baseline {baseline_pass[N]:.5f})")

    # --- Plot the largest-n design against the baseline ---
    if results:
        n_best = max(results)
        res_best = results[n_best]
        print(f"\nPlotting baseline (n=1) vs optimized (n={n_best}) ...")
        plot_filter(a_simple, N_values, I0=I0, I1=I1, savepath="baseline_n1.png")
        plot_filter(res_best.a, N_values, I0=I0, I1=I1, savepath=f"optimized_n{n_best}.png")
        print(f"  saved baseline_n1.png, optimized_n{n_best}.png")
    else:
        print("\nNo design succeeded across the n values tried; skipping plots.")


if __name__ == "__main__":
    main()
