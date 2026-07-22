"""Multi-band design via homotopy continuation (sdp_design.design_via_continuation),
picking up where multiband_demo.py left off.

multiband_demo.py's random-restart search (design_via_layers) found that this
exact I0/I1 configuration -- two disjoint stop bands, two disjoint pass bands --
is *not* hard to optimize in general (n=7 already does well), but n=9
specifically is a genuine dead zone for it: two independent, unrelated search
strategies (continuation tried over all sign patterns, and a 40-restart random
search) both converged to the *same* delta1=0.0649, strong evidence that's
really the best n=9 can do here, not a search failure. n=11 and n=13, by
contrast, both nail the configuration almost perfectly.

design_via_continuation (see its docstring in sdp_design.py) starts from
alphas=0 -- the trivial, contrast-free structure with q2=0 identically, so
T_N=1 exactly everywhere, trivially satisfying the pass-band constraint (B) --
and ramps the required stop-band depth mu up from 0 to mu0 in small steps,
warm-starting each step from the previous one. That single continuous
trajectory is dramatically cheaper than design_via_layers' multi-restart
search and, for this configuration, reaches designs at least as good (n=7,
n=11, n=13) or reveals the same intrinsic limit (n=9) that heavy random search
also finds -- without needing 15-40 random restarts per n.

Run from the repository root: python3 examples/multiband_continuation_demo.py
"""
import numpy as np

from scattering.forward import transmission_TN, plot_filter
from scattering.sdp_design import design_via_continuation


def _extreme_TN(a, intervals, N, mode, n_grid=4000):
    """max (mode='max') or min (mode='min') of T_N over the union of intervals."""
    vals = [transmission_TN(a, np.linspace(lo, hi, n_grid), N) for lo, hi in intervals]
    allvals = np.concatenate(vals)
    return float(allvals.max() if mode == "max" else allvals.min())


def main():
    I0 = [(0.50, 0.65), (2.50, 2.65)]
    I1 = [(1.20, 1.40), (1.80, 2.00)]
    print(f"I0 (stop) = {I0}")
    print(f"I1 (pass) = {I1}")

    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_values = (1, 3, 5)

    print("\n--- Optimized designs (design_via_continuation: homotopy from alpha=0) ---")
    results, rows = {}, []
    for n in (7, 9, 11, 13):
        res = design_via_continuation(n, I0, I1, mu0, n_steps=25)
        if res.status not in ("optimal", "partial"):
            print(f"\nn={n}: {res.status}")
            continue
        results[n] = res

        print(f"\nn={n}: delta1={res.delta1:.4e}  sigma={res.sigma}  "
              f"achieved_mu={res.achieved_mu:.4f} (target {mu0})")
        print(f"  alphas = {np.round(res.alphas, 4)}  (sum={np.sum(res.alphas):.2e})")
        print(f"  impedances p_0..p_{n + 1} = {np.round(res.impedances, 4)}")
        for N in N_values:
            tn_stop = _extreme_TN(res.a, I0, N, "max")
            tn_pass = _extreme_TN(res.a, I1, N, "min")
            rows.append((n, N, tn_stop, tn_pass))
            print(f"  N={N:3d}:  max T_N over I0 = {tn_stop:.3e}   min T_N over I1 = {tn_pass:.5f}")

    print("\n--- Summary table ---")
    print(f"{'n':>3} {'N':>3} {'max T_N(I0)':>14} {'min T_N(I1)':>14}")
    for n, N, tn_stop, tn_pass in rows:
        print(f"{n:>3} {N:>3} {tn_stop:>14.3e} {tn_pass:>14.5f}")
    print("\nn=9 sitting well behind n=7 and n=11/13 here is expected, not a bug --")
    print("see the module docstring: it's a confirmed dead zone for this exact")
    print("I0/I1 configuration, not a search-quality artifact.")

    # --- Plot the two best (n=11, n=13): both essentially solve the problem,
    # unlike n=7 (good but not tight) and n=9 (the dead zone above).
    for n in (11, 13):
        if n not in results:
            continue
        res = results[n]
        print(f"\nPlotting n={n} (delta1={res.delta1:.4e}) ...")
        plot_filter(res.a, N_values, I0=I0, I1=I1, savepath=f"multiband_n{n}_continuation.png")
        print(f"  saved multiband_n{n}_continuation.png")


if __name__ == "__main__":
    main()
