"""Multi-band design demo: I0 and I1 as several disjoint intervals each
(a genuine multi-stop-band / multi-pass-band filter spec), independent of
band_informed_demo.py's single-band-derived-from-a-baseline approach.

design_via_layers (and design_filter_full/design_filter underneath it)
already accept J0/J1 as arbitrary lists of (lo, hi) pairs -- every stop
component gets its own sign sigma_j, tried over all 2^len(J0) combinations
-- so no new machinery is needed, just a spec with more than one interval
per band and a few n to test.

I0, I1 here are picked directly (not derived from a single baseline's own
band/gap structure), each narrow with generous guard bands between
neighbours -- per the earlier finding that a degree-n design has on the
order of n natural oscillations across [0, pi], demanding good behaviour
over *wide* target regions becomes a topologically atypical, often
infeasible ask as n grows; that effect only gets stronger with more
target components, so intervals are kept modest here.

Run from the repository root: python3 examples/multiband_demo.py
"""
import numpy as np

from scattering.forward import transmission_TN, plot_filter
from scattering.sdp_design import design_via_layers


def _extreme_TN(a, intervals, N, mode, n_grid=2000):
    """max (mode='max') or min (mode='min') of T_N over the union of intervals."""
    vals = [transmission_TN(a, np.linspace(lo, hi, n_grid), N) for lo, hi in intervals]
    allvals = np.concatenate(vals)
    return float(allvals.max() if mode == "max" else allvals.min())


def main():
    # Two stop bands, two pass bands, interleaved with guard bands, all
    # strictly inside (0, pi) (0 itself can never be in a stop band --
    # constraint D forces kappa_B(0)=1, T_N(0)=1 for every valid design).
    I0 = [(0.50, 0.65), (2.50, 2.65)]
    I1 = [(1.20, 1.40), (1.80, 2.00)]
    print(f"I0 (stop) = {I0}")
    print(f"I1 (pass) = {I1}")

    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_values = (1, 3, 5)

    print("\n--- Optimized designs (design_via_layers: realizable by construction) ---")
    results = {}
    smaller_solutions = {}
    for n in (1, 3, 5, 7, 9):
        res = design_via_layers(n, I0, I1, mu0, use_sdp_warm_start=False, n_restarts=15,
                                 smaller_solutions=smaller_solutions)
        if res.status != "optimal":
            print(f"\nn={n}: {res.status} (too few layers for this many disjoint bands, most likely)")
            continue
        results[n] = res
        smaller_solutions[n] = res.alphas

        print(f"\nn={n}: delta1={res.delta1:.4e}  sigma={res.sigma}  "
              f"achieved_mu={res.achieved_mu:.4f} (target {mu0})")
        print(f"  alphas = {np.round(res.alphas, 4)}  (sum={np.sum(res.alphas):.2e})")
        print(f"  impedances p_0..p_{n + 1} = {np.round(res.impedances, 4)}")
        for N in N_values:
            tn_stop = _extreme_TN(res.a, I0, N, "max")
            tn_pass = _extreme_TN(res.a, I1, N, "min")
            print(f"  N={N:3d}:  max T_N over I0 = {tn_stop:.3e}   min T_N over I1 = {tn_pass:.5f}")

    # --- Plot the best-performing design (smallest delta1), not just the
    # largest n that happened to converge: local search quality is not
    # guaranteed monotonic in n (confirmed in this exact run -- n=9 landed
    # in a worse basin than n=7), so "largest n" and "best design" can
    # differ, and silently plotting the former would be misleading.
    if results:
        n_best = min(results, key=lambda n: results[n].delta1)
        res_best = results[n_best]
        print(f"\nPlotting best-performing design (n={n_best}, delta1={res_best.delta1:.4e}) ...")
        plot_filter(res_best.a, N_values, I0=I0, I1=I1, savepath=f"multiband_n{n_best}.png")
        print(f"  saved multiband_n{n_best}.png")
    else:
        print("\nNo design succeeded across the n values tried; skipping plot.")


if __name__ == "__main__":
    main()
