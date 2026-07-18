"""Band-informed design workflow, as suggested by the paper's author:

  1. Start from a simple structure (a single layer of contrast p_1 = 2,
     matching Fig. 2 of the paper) and compute its Floquet discriminant
     kappa_B on the fundamental domain theta = 2*k*h in [0, pi] (kappa_B is
     2*pi-periodic and even, so [0,pi] determines everything).
  2. Read off its natural band/gap structure, and choose I0 (stop) inside
     a gap and I1 (pass) inside a band, each with a guard margin away from
     the band edges (rather than an arbitrary externally-imposed spec).
  3. Solve the SDP design problem for a richer block (n > 1) using those
     bands, and compare the resulting T_N against the original simple
     structure's T_N at the same block count N.

Run from the repository root: python3 examples/band_informed_design.py
"""
import numpy as np

from scattering.layer_stripping import forward_reconstruct, layers_from_a
from scattering.bandgap import find_bands_gaps, shrink_interval
from scattering.design_full import design_filter_full, verify_design
from scattering.polish import polish_design
from scattering.transmission import transmission_TN


def main():
    # Step 1: simple one-layer structure, p_1 = 2 (background p_0 = p_2 = 1).
    alpha0 = np.log(2.0)
    alphas_simple = np.array([alpha0, -alpha0])
    p1_simple, _ = forward_reconstruct(alphas_simple)
    a_simple = p1_simple[::-1]

    bg = find_bands_gaps(a_simple)
    print("Simple 1-layer structure (p_1=2): bands =", bg.bands, " gaps =", bg.gaps)

    I0 = [shrink_interval(bg.gaps[0], margin_frac=0.15)]
    I1 = [shrink_interval(bg.bands[0], margin_frac=0.15)]
    print(f"Chosen I0 (stop) = {I0}, I1 (pass) = {I1}")

    # Step 2/3: design a richer (n=4) block targeting those bands.
    n, mu0 = 4, 0.8
    res = design_filter_full(n, I0, I1, mu0)
    a_pol, u_pol, _ = polish_design(res.a, I0, I1, mu0, res.sigma)
    delta1 = np.sqrt(max(u_pol, 0.0)) - 1.0
    ok = verify_design(a_pol, I0, I1, mu0, res.sigma, u_pol, tol=1e-6)
    print(f"n={n}, mu0={mu0}: delta1={delta1:.4e}, constraints verified={ok} (sigma={res.sigma})")

    stack = layers_from_a(a_pol)
    if stack.was_reflected:
        print("NOTE: optimizer's coefficient vector was not minimum-phase; "
              "reflected to the realizable branch for layer-stripping. "
              "Re-checking constraint (C) on the *realizable* vector:")
        ok_realizable = verify_design(stack.a_used, I0, I1, mu0, res.sigma, u_pol, tol=1e-6)
        print(f"  -> verified on realizable a = {ok_realizable} "
              "(if False: this is the realizability gap described in the write-up)")
    print("alphas:", np.round(stack.alphas, 4), " sum =", round(stack.residual_sum_alpha, 6))
    print("impedances p_0..p_{n+1}:", np.round(stack.impedances, 4))

    # Compare T_N of the simple vs. optimized block, same N.
    N = 8
    th0 = np.linspace(*I0[0], 2000)
    th1 = np.linspace(*I1[0], 2000)
    print(f"\nAt N={N} blocks:")
    print(f"  max T_N on I0 (stop): simple={transmission_TN(a_simple, th0, N).max():.3e}"
          f"   optimized={transmission_TN(a_pol, th0, N).max():.3e}")
    print(f"  min T_N on I1 (pass): simple={transmission_TN(a_simple, th1, N).min():.5f}"
          f"   optimized={transmission_TN(a_pol, th1, N).min():.5f}")


if __name__ == "__main__":
    main()
