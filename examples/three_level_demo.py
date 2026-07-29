"""Three-level narrative demo for the paper (Section 6.3/6.4): unoptimised
baseline -> SDP-initialised local polish at fixed n -> full degree-scanned
direct optimisation. Isolates *where* the improvement in delta comes from:
redistributing contrasts at fixed complexity (Level 2) vs. adding layers and
searching harder (Level 3).

  Level 1 (unoptimised): n=3, impedances p_1=p_3=2, p_2=sqrt(2) (a symmetric,
  hand-picked structure, not the result of any optimisation). I0/I1 are a
  moderate sub-window of its own natural gap/band (find_bands_gaps), same
  windowing rule as band_informed_demo.py -- a degree-n design has ~n
  natural oscillations across [0,pi], so demanding good behaviour across an
  entire gap/band (rather than a sub-window of it) is an atypical, often
  infeasible ask, more so at small n.

  Level 2 (SDP-initialised, same n=3): design_filter_full solves the lifted
  SDP relaxation and polishes it to a genuine (a, u) pair; that solution is
  reflected onto the realizable branch (ensure_min_phase) and stripped to
  layer parameters (alphas_from_a); those alphas seed a *local* polish over
  alpha_j at n=3 -- design_via_layers with the SDP as the *only* warm start
  (use_continuation_seed=False, small n_restarts), deliberately not using
  the continuation seed or a larger n, to isolate what redistributing
  contrasts at fixed complexity buys on its own.

  Level 3 (direct optimisation, scanned over n): the full algorithm --
  design_via_layers with every warm-start strategy enabled (SDP seed,
  continuation seed, zero-padded smaller-n embeddings), scanned over a
  small range of n and the best delta kept (not the first feasible n --
  local-search quality is not monotonic in n, see multiband_demo.py's n=9
  dead-zone finding). For each n scanned, the SDP relaxation's own raw
  (pre-polish) objective is also recorded as delta_SDP -- a genuine lower
  bound on the true optimum (relaxing constraints can only decrease a
  minimisation's optimal value), reported alongside the achieved delta to
  show relaxation tightness as a function of n, with no theoretical claim
  attached.

Run from the repository root: python3 examples/three_level_demo.py
"""
import numpy as np

from scattering.forward import a_from_alphas, transmission_TN, plot_filter, kappa_B
from scattering.bandgap import find_bands_gaps
from scattering.sdp_design import design_filter_full, design_via_layers, sdp_lower_bound
from scattering.inverse import ensure_min_phase, alphas_from_a


def _extreme_TN(a, intervals, N, mode, n_grid=4000):
    vals = [transmission_TN(a, np.linspace(lo, hi, n_grid), N) for lo, hi in intervals]
    allvals = np.concatenate(vals)
    return float(allvals.max() if mode == "max" else allvals.min())


def main():
    # --- Level 1: unoptimised baseline, n=3, p_1=p_3=2, p_2=sqrt(2) ---
    p1, p2, p3 = 2.0, np.sqrt(2.0), 2.0
    alphas_baseline = np.array([np.log(p1), np.log(p2 / p1), np.log(p3 / p2), np.log(1.0 / p3)])
    print(f"Baseline impedances p_0..p_4 = [1, {p1}, {p2:.4f}, {p3}, 1]")
    print(f"  alphas = {np.round(alphas_baseline, 4)}  (sum={np.sum(alphas_baseline):.2e})")
    a_baseline = a_from_alphas(alphas_baseline)

    bg = find_bands_gaps(a_baseline)
    band_lo, band_hi = bg.bands[0]
    gap_lo, gap_hi, gap_sign = bg.gaps[0]
    print(f"  full gap  = [{gap_lo:.4f}, {gap_hi:.4f}]")
    print(f"  full band = [{band_lo:.4f}, {band_hi:.4f}]")

    I0 = [(gap_lo + 0.70 * (gap_hi - gap_lo), gap_lo + 0.85 * (gap_hi - gap_lo))]
    I1 = [(band_lo + 0.40 * (band_hi - band_lo), band_lo + 0.60 * (band_hi - band_lo))]
    print(f"I0 (stop) = {I0}")
    print(f"I1 (pass) = {I1}")

    mu0 = 0.05  # any mu0>0 works (Prop. 5.4a); N does the work of reaching a given depth
    N_table = (1, 3, 5, 10)
    N_plot = (1, 3, 5)

    # --- Level 2: SDP-initialised local polish, fixed n=3 ---
    print("\n--- Level 2: SDP-initialised local polish (n=3, fixed complexity) ---")
    n_fixed = 3
    delta_sdp_n3 = sdp_lower_bound(n_fixed, I0, I1, mu0)
    print(f"  SDP relaxation lower bound: delta_SDP = {delta_sdp_n3:.4e}")
    full_res_n3 = design_filter_full(n_fixed, I0, I1, mu0)  # polished (a, u), for the warm start below
    a_mp, was_reflected = ensure_min_phase(full_res_n3.a)
    alphas_sdp, info = alphas_from_a(a_mp)
    print(f"  SDP polished+reflected alphas = {np.round(alphas_sdp, 4)} (reflected={was_reflected})")

    res_level2 = design_via_layers(n_fixed, I0, I1, mu0, warm_start_alphas=alphas_sdp,
                                    use_sdp_warm_start=False, use_continuation_seed=False,
                                    n_restarts=5, seed=0)
    print(f"  Level 2 result: status={res_level2.status} delta={res_level2.delta:.4e} "
          f"sigma={res_level2.sigma} achieved_mu={res_level2.achieved_mu:.4f}")

    # --- Level 3: full algorithm, degree-scanned ---
    print("\n--- Level 3: direct optimisation, scanned over n ---")
    n_range = (3, 5, 7, 9)
    level3_results, sdp_bounds, smaller_solutions = {}, {}, {}
    for n in n_range:
        lb = sdp_lower_bound(n, I0, I1, mu0)
        sdp_bounds[n] = lb
        res = design_via_layers(n, I0, I1, mu0, use_sdp_warm_start=True, use_continuation_seed=True,
                                 smaller_solutions=smaller_solutions, n_restarts=15, seed=0)
        if res.status != "optimal":
            print(f"  n={n}: {res.status}  (delta_SDP={lb:.4e})")
            continue
        level3_results[n] = res
        smaller_solutions[n] = res.alphas
        print(f"  n={n}: delta_SDP={lb:.4e}  delta_achieved={res.delta:.4e}  sigma={res.sigma}")

    n_best = min(level3_results, key=lambda n: level3_results[n].delta)
    res_level3 = level3_results[n_best]
    print(f"\n  Best: n={n_best}, delta={res_level3.delta:.4e}")

    # --- Table A: T_N comparison across the three levels ---
    print("\n--- Table A: T_N across levels ---")
    header = (f"{'N':>3} | {'max T_N(I0) L1':>14} {'max T_N(I0) L2':>14} {'max T_N(I0) L3':>14} | "
              f"{'min T_N(I1) L1':>14} {'min T_N(I1) L2':>14} {'min T_N(I1) L3':>14}")
    print(header)
    for N in N_table:
        row = [N]
        for res_a in (a_baseline, res_level2.a, res_level3.a):
            row.append(_extreme_TN(res_a, I0, N, "max"))
        for res_a in (a_baseline, res_level2.a, res_level3.a):
            row.append(_extreme_TN(res_a, I1, N, "min"))
        print(f"{row[0]:>3} | {row[1]:>14.3e} {row[2]:>14.3e} {row[3]:>14.3e} | "
              f"{row[4]:>14.5f} {row[5]:>14.5f} {row[6]:>14.5f}")

    # --- Table B: relaxation quality across the n scanned in Level 3 ---
    # delta_SDP <= delta_achieved must hold mathematically (the achieved
    # design is itself feasible for the relaxed SDP, embedded via A=a a^T,
    # so the relaxed optimum can only be <= it) -- an apparent violation
    # here is not a counterexample, it flags that the SDP solver itself
    # failed to certify a bound anywhere near tight at that n (confirmed:
    # tightening CLARABEL's tolerance to 1e-12 barely moved the reported
    # value and downgraded status to optimal_inaccurate, i.e. the solver is
    # numerically stuck, not merely stopping early). This tends to happen
    # once delta_achieved drops to within the solver's own working
    # precision -- exactly the good-design regime the scan is looking for.
    print("\n--- Table B: SDP relaxation lower bound vs. achieved delta ---")
    print(f"{'n':>3} {'delta_SDP (lower bound)':>24} {'delta_achieved':>16} {'gap':>12}  note")
    gap_n3 = res_level2.delta - delta_sdp_n3
    flag_n3 = "" if gap_n3 >= 0 else "  <- solver precision floor, see caveat below"
    print(f"{n_fixed:>3} {delta_sdp_n3:>24.4e} {res_level2.delta:>16.4e} "
          f"{gap_n3:>12.4e}  (Level 2){flag_n3}")
    any_negative = gap_n3 < 0
    for n in n_range:
        if n not in level3_results:
            continue
        d = level3_results[n].delta
        gap = d - sdp_bounds[n]
        flag = "" if gap >= 0 else "  <- solver precision floor, see caveat below"
        any_negative = any_negative or gap < 0
        print(f"{n:>3} {sdp_bounds[n]:>24.4e} {d:>16.4e} {gap:>12.4e}{flag}")
    if any_negative:
        print("\n  Caveat: delta_SDP <= delta_achieved is a theorem (Sec. 6.3), not an")
        print("  empirical claim -- rows flagged above are the SDP solver failing to")
        print("  certify a bound near delta_achieved's scale, not a violation of it.")

    # --- Plots ---
    print("\nSaving plots ...")
    plot_filter(a_baseline, N_plot, I0=I0, I1=I1, savepath="three_level_1_baseline.png")
    print("  saved three_level_1_baseline.png")
    plot_filter(res_level2.a, N_plot, I0=I0, I1=I1, savepath="three_level_2_sdp_n3.png")
    print("  saved three_level_2_sdp_n3.png")
    plot_filter(res_level3.a, N_plot, I0=I0, I1=I1, savepath=f"three_level_3_best_n{n_best}.png")
    print(f"  saved three_level_3_best_n{n_best}.png")


if __name__ == "__main__":
    main()
