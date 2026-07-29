"""Multi-band design via homotopy continuation (sdp_design.design_via_continuation),
picking up where multiband_demo.py left off, then exploiting a structural
observation to widen the pass band essentially for free.

multiband_demo.py's random-restart search (design_via_layers) found that this
I0/I1 configuration -- two disjoint stop bands, two disjoint pass bands -- is
*not* hard to optimize in general (n=7 already does well), but n=9
specifically is a genuine dead zone for it: two independent, unrelated search
strategies (continuation tried over all sign patterns, and a 40-restart random
search) both converged to the *same* delta=0.0649, strong evidence that's
really the best n=9 can do here, not a search failure. n=11 and n=13, by
contrast, both nail the original two-piece I1 = [(1.20,1.40), (1.80,2.00)]
almost perfectly.

Inspecting *those* n=11/n=13 solutions (found with only the two narrow I1
pieces constrained) turned up something worth exploiting: kappa_B has 3-4
extra local extrema in the *unconstrained* gap between the two I1 pieces
(1.40 to 1.80), but at every one |kappa_B| only pokes past 1 by 1e-3 to 1e-4
-- near-closed gaps, reflecting almost nothing, so T_N stays high across that
whole connecting region even though nothing asked for that. The natural
explanation: design_via_continuation starts at alphas=0 (kappa_B=cos(n theta),
G=1 identically, no gaps anywhere) and only deforms as far as required to open
the *requested* stop bands and flatten the *requested* pass pieces -- nothing
rewards opening extra gaps elsewhere, so the connecting region tends to stay
close to its flat, gap-free starting shape "for free".

That suggested explicitly merging I1 into the single wide interval
[1.20, 2.00] -- asking for exactly the flatness that was already showing up
by accident -- rather than the disjoint two-piece version. Unlike the earlier
guard-band experiment (widening I0/I1 by an unmotivated margin, which
consistently made things *worse* -- see the module docstring history in
sdp_design.py / the multiband_demo.py investigation), this widening costs
essentially nothing at the degrees that were already good: n=13 with the wide
I1 is *better* than with the split I1 (delta 1.8e-5 vs 2.5e-5), and n=17
reaches delta=4.0e-6, essentially perfect, with min T_N over the *entire*
wide pass band at 0.9999. The difference from the guard-band case is that
here there was direct empirical evidence the wider region was already nearly
free, rather than an unmotivated ask.

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
    I1 = [(1.20, 2.00)]  # wide: the two original pieces plus the connecting gap, merged
    print(f"I0 (stop) = {I0}")
    print(f"I1 (pass, widened) = {I1}")

    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_values = (1, 3, 5)

    print("\n--- Optimized designs (design_via_continuation: homotopy from alpha=0) ---")
    results, rows = {}, []
    for n in (7, 9, 11, 13, 15, 17):
        res = design_via_continuation(n, I0, I1, mu0, n_steps=25)
        if res.status not in ("optimal", "partial"):
            print(f"\nn={n}: {res.status}")
            continue
        results[n] = res

        print(f"\nn={n}: delta={res.delta:.4e}  sigma={res.sigma}  "
              f"achieved_mu={res.achieved_mu:.4f} (target {mu0})")
        print(f"  alphas = {np.round(res.alphas, 4)}  (sum={np.sum(res.alphas):.2e})")
        print(f"  impedances p_0..p_{n + 1} = {np.round(res.impedances, 4)}")
        for N in N_values:
            tn_stop = _extreme_TN(res.a, I0, N, "max")
            tn_pass = _extreme_TN(res.a, I1, N, "min")
            rows.append((n, N, tn_stop, tn_pass))
            print(f"  N={N:3d}:  max T_N over I0 = {tn_stop:.3e}   min T_N over I1(wide) = {tn_pass:.5f}")

    print("\n--- Summary table ---")
    print(f"{'n':>3} {'N':>3} {'max T_N(I0)':>14} {'min T_N(I1 wide)':>17}")
    for n, N, tn_stop, tn_pass in rows:
        print(f"{n:>3} {N:>3} {tn_stop:>14.3e} {tn_pass:>17.5f}")
    print("\nn=9 and n=15 sitting well behind their neighbours here is expected, not a")
    print("bug -- specific degrees are hit-or-miss for a given interval layout (see")
    print("module docstring); n=9 was already a confirmed dead zone for the original")
    print("split I1, and widening I1 simply revealed n=15 as another one.")

    # --- Plot n=13 and n=17: both essentially solve the widened problem for
    # free (n=13 is even slightly better than its split-I1 result), unlike
    # n=9/n=15 (dead zones) or n=7/n=11 (good but not as tight).
    for n in (13, 17):
        if n not in results:
            continue
        res = results[n]
        print(f"\nPlotting n={n} (delta={res.delta:.4e}) ...")
        plot_filter(res.a, N_values, I0=I0, I1=I1, savepath=f"multiband_n{n}_wide_continuation.png")
        print(f"  saved multiband_n{n}_wide_continuation.png")


if __name__ == "__main__":
    main()
