"""Demonstration: pick a simple two-layer structure, read I_0/I_1 off its
own kappa_B(theta) by eye (a narrow pass window right next to theta=0,
where kappa_B=1 exactly by the matching constraint and hasn't had room
to swing back toward +-1 yet; a stop window centered on the natural
gap's own minimum), then run the real Stage 2/3 algorithm (not the seed
structure itself) on that geometry.

Seed structure: n=2, rho_1=rho_2=2 (the F2 fixture from Stage 1's own
tests: alpha=(log2,0,-log2)). Its kappa_B has a single symmetric gap
centered at theta=pi, minimum ~-2.10 around theta~1.5-1.65, and is
close to 1 for a while after theta=0.

I_0=[1.4,1.7] (inside the gap, well past kappa_B=-1's crossing at
theta~0.99, so a decent mu_0 has real margin), I_1=[0.05,0.25] (hugging
theta=0, per the mechanism found while investigating the earlier
Instance 1 rerun: a pass band placed close to theta=0 leaves little room
for kappa_B to wander back toward +-1 within it, which is what keeps
s_0>0 -- exactly the effect this demo is meant to show).

Run from the repository root: python3 examples/manual_seed_demo.py
"""
import json

import numpy as np

from scattering.forward import a_from_alphas, kappa_B, transmission_TN
from scattering.driver import degree_scan_stage3, retained_degree_record, save_records_csv, mu0_from_N_max
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

SEED_ALPHA = np.array([np.log(2.0), 0.0, -np.log(2.0)])  # F2 fixture
I0 = [(1.4, 1.7)]
I1 = [(0.05, 0.25)]
EPS0 = EPS1 = 1e-2
N_MAX = 6  # manuscript revision: top-level input in place of mu_0; derived mu0=log(4/eps0)/(2*N_max)~=0.5
N_RANGE = range(2, 9)


def main():
    mu0 = mu0_from_N_max(EPS0, N_MAX)
    a_seed = a_from_alphas(SEED_ALPHA)
    theta_probe = np.linspace(0, np.pi, 5)
    print("Seed structure kappa_B at a few points:", kappa_B(a_seed, theta_probe))
    print(f"I0={I0}  I1={I1}  N_max={N_MAX}  mu0={mu0:.4f}  (cosh(mu0)={np.cosh(mu0):.4f})")
    print()

    records = degree_scan_stage3(N_RANGE, I0, I1, N_MAX, EPS0, EPS1, N_values=(5, 10, 20))
    for r in records:
        print(f"n={r.n} status={r.status:18s} delta_achieved={r.delta_achieved} "
              f"kappa_min={r.kappa_min} s_0={r.s_0} admissible={r.admissible}")

    retained = retained_degree_record(records)
    print()
    if retained is None:
        print("No admissible degree found in range.")
        return

    print(f"Retained: n={retained.n}, delta_achieved={retained.delta_achieved}, "
          f"sigma_star={retained.sigma_star}")

    a = a_from_alphas(np.asarray(retained.alpha))
    theta_I1 = np.linspace(I1[0][0], I1[0][1], 500_000)
    theta_I0 = np.linspace(I0[0][0], I0[0][1], 500_000)
    for N in (1, 3, 5, 10):
        TN_pass_min = transmission_TN(a, theta_I1, N).min()
        TN_stop_max = transmission_TN(a, theta_I0, N).max()
        print(f"N={N:2d}: min T_N over I1 = {TN_pass_min:.8f}   max T_N over I0 = {TN_stop_max:.2e}")

    save_records_csv(records, "examples/manual_seed_demo_scan.csv")
    with open("examples/manual_seed_demo_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)
    plot_design_from_record(retained, I0, I1, N_values=(1, 3, 5, 10), mu0=mu0,
                             savepath="examples/manual_seed_demo_design.png")
    plot_bound_vs_achieved(records, retained=retained, savepath="examples/manual_seed_demo_bound.png")
    print("\nSaved: examples/manual_seed_demo_scan.{csv,json}, "
          "manual_seed_demo_design.png, manual_seed_demo_bound.png")


if __name__ == "__main__":
    main()
