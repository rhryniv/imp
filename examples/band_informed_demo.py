"""Band-informed design demo (per discussion): rather than picking I0/I1
arbitrarily, derive them from a simple structure's own band/gap structure,
so they are guaranteed well-posed, then show how much a richer optimized
block improves on the simple structure at matched N.

  1. Simple 1-layer structure, contrast p_1 > 1: alphas = [log p_1, -log p_1].
  2. Its Floquet discriminant kappa_B has exactly one gap and one band over
     the fundamental domain theta in [0, pi] (Fig. 2 of the paper).
  3. Take I0 strictly inside the gap, I1 strictly inside the band (with a
     margin, since band edges are singular for the admissibility bound).
  4. For several n > 1, run the SDP synthesis targeting I0/I1, realize the
     resulting layer sequence, and compare T_N against the simple
     structure's T_N at the same N.

Run from the repository root: python3 examples/band_informed_demo.py
"""
import numpy as np

from scattering.forward import a_from_alphas, kappa_B, transmission_TN, plot_filter
from scattering.bandgap import find_bands_gaps, shrink_interval
from scattering.sdp_design import design_filter_full, realize_design


def main():
    # --- Step 1: simple 1-layer baseline structure ---
    p1_contrast = 2.0
    alpha0 = np.log(p1_contrast)
    alphas_simple = np.array([alpha0, -alpha0])
    a_simple = a_from_alphas(alphas_simple)
    print(f"Baseline: 1-layer block, p_1={p1_contrast} (alphas={np.round(alphas_simple, 4)})")

    # --- Steps 2-3: band/gap structure, choose I0/I1 with a margin ---
    bg = find_bands_gaps(a_simple)
    print(f"  bands = {[tuple(round(x, 4) for x in b) for b in bg.bands]}")
    print(f"  gaps  = {[tuple(round(x, 4) for x in g) for g in bg.gaps]}")

    gap_lo, gap_hi, gap_sign = bg.gaps[0]
    band = bg.bands[0]
    margin = 0.15
    I0 = [shrink_interval((gap_lo, gap_hi), margin)]
    I1 = [shrink_interval(band, margin)]
    print(f"  I0 (stop, margin={margin}) = {[tuple(round(x, 4) for x in iv) for iv in I0]}")
    print(f"  I1 (pass, margin={margin}) = {[tuple(round(x, 4) for x in iv) for iv in I1]}")

    # Target stop-band depth: comfortably inside what the baseline itself
    # already achieves on I0, so the target is meaningful but not
    # gratuitously hard for small n.
    th0 = np.linspace(*I0[0], 4000)
    th1 = np.linspace(*I1[0], 4000)
    mu0_available = np.arccosh(np.min(gap_sign * kappa_B(a_simple, th0)))
    mu0 = 0.35 * mu0_available
    print(f"  baseline's own gap depth on I0: mu={mu0_available:.4f}; "
          f"target mu0={mu0:.4f} (35% of that)\n")

    N_values = (5, 10, 20)

    print("--- Baseline (n=1) T_N ---")
    baseline_stop = {N: float(transmission_TN(a_simple, th0, N).max()) for N in N_values}
    baseline_pass = {N: float(transmission_TN(a_simple, th1, N).min()) for N in N_values}
    for N in N_values:
        print(f"  N={N:3d}:  max T_N on I0 = {baseline_stop[N]:.3e}   "
              f"min T_N on I1 = {baseline_pass[N]:.5f}")

    # --- Step 4/5: optimize for several n > 1, compare ---
    print("\n--- Optimized designs ---")
    results = {}
    for n in (3, 5, 7, 9):
        res = design_filter_full(n, I0, I1, mu0)
        if res is None or res.a is None or res.status not in ("optimal", "optimal_inaccurate"):
            print(f"n={n}: SDP did not return a usable result (status={getattr(res, 'status', None)})")
            continue
        real = realize_design(res.a, I0, I1, mu0, sigma=res.sigma, N_values=N_values)
        results[n] = (res, real)

        print(f"\nn={n}: delta1={res.delta1:.4e}  tightness_ratio={res.tightness_ratio:.2e}  "
              f"polish_verified={res.polish_verified}")
        if not real.reliable:
            print(f"  *** layer realization UNRELIABLE: {real.failure}")
        print(f"  realized: admissible={real.admissible} (G_min={real.G_min:.6f})  "
              f"C_satisfied={real.C_satisfied} (achieved_mu={real.achieved_mu:.4f}, target={mu0:.4f})  "
              f"was_reflected={real.was_reflected}  round_trip_err={real.round_trip_error:.2e}")
        print(f"  alphas = {np.round(real.alphas, 4)}  (sum={np.sum(real.alphas):.2e})")
        print(f"  impedances p_0..p_{n+1} = {np.round(real.impedances, 4)}")
        for N in N_values:
            print(f"  N={N:3d}:  max T_N on I0 = {real.TN_stop_max[N]:.3e}  "
                  f"(baseline {baseline_stop[N]:.3e})   "
                  f"min T_N on I1 = {real.TN_pass_min[N]:.5f}  "
                  f"(baseline {baseline_pass[N]:.5f})")

    # --- Plot the best verified, largest-n design against the baseline ---
    verified_ns = [n for n, (res, real) in results.items()
                   if res.polish_verified and real.C_satisfied and real.reliable]
    if verified_ns:
        n_best = max(verified_ns)
        _, real_best = results[n_best]
        print(f"\nPlotting baseline (n=1) vs optimized (n={n_best}) ...")
        plot_filter(a_simple, N_values, I0=I0, I1=I1, savepath="baseline_n1.png")
        plot_filter(real_best.a, N_values, I0=I0, I1=I1, savepath=f"optimized_n{n_best}.png")
        print("  saved baseline_n1.png, optimized_n{}.png".format(n_best))
    else:
        print("\nNo verified+C-satisfied design found across the n values tried; skipping plots.")


if __name__ == "__main__":
    main()
