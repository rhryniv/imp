"""Instance 2 (spec Sec. 7, m_0 = 2, "a two-component stop region"): the
Stage 3 degree scan, n = 2..16, I_0 = [1.0,1.4] u [2.6,3.0], I_1 =
[0.2,0.6] u [1.8,2.2] (RH's own choice), mu_0 = 0.05, eps_0 = eps_1 =
1e-2. m_1 = 2 here too (I_1 also has two components), so the Sec. 5.4
gamma/n_min_green closed form does not apply (it requires a single
I_1 component) -- driver.gamma_geometric returns None for this
instance, by design, not a bug.

Exercises the sign-pattern enumeration (2^m_0 = 4 patterns tried per
degree in Phase 1/2 -- see direct.py's own module docstring for why
every feasible pattern is tried, not just the largest-margin one) and
the multi-band case generally. Produces the same Sec. 6 table
(CSV + LaTeX) and JSON dump as instance1_scan.py; the deliverable-3
bound-vs-achieved plot is produced (it does not depend on gamma), the
kappa_B/T_N design plot is produced without a marked mu_0 threshold on
the (now two-band) stop region shading serving the same purpose.

Run from the repository root: python3 examples/instance2_scan.py
"""
import json
import time

import numpy as np

from scattering.driver import degree_scan_stage3, retained_degree_record, save_records_csv, save_records_latex
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(1.0, 1.4), (2.6, 3.0)]
I1 = [(0.2, 0.6), (1.8, 2.2)]
MU0 = 0.05
EPS0 = EPS1 = 1e-2
N_RANGE = range(2, 17)
N_VALUES = (20, 40, 60, 80, 100)


def main():
    t0 = time.time()
    records = degree_scan_stage3(N_RANGE, I0, I1, MU0, EPS0, EPS1, N_values=N_VALUES)
    print(f"scan finished in {time.time() - t0:.1f}s")

    for r in records:
        print(f"n={r.n:2d} status={r.status:20s} underline_delta={r.underline_delta!s:>14.14} "
              f"delta_achieved={r.delta_achieved!s:>14.14} admissible={r.admissible} "
              f"sigma_star={r.sigma_star} dual={r.dual_status}")

    retained = retained_degree_record(records)
    print()
    if retained is None:
        print("No degree in the scanned range is admissible (rule 4).")
    else:
        print(f"Retained (least admissible n): n={retained.n}, delta_achieved={retained.delta_achieved}, "
              f"sigma_star={retained.sigma_star}")

    save_records_csv(records, "examples/instance2_scan.csv")
    save_records_latex(records, "examples/instance2_scan_table.tex")
    with open("examples/instance2_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained, savepath="examples/instance2_bound_vs_achieved.png")

    plot_source = retained if retained is not None else next(
        (r for r in reversed(records) if r.status == "optimal" and r.alpha is not None), None)
    if plot_source is not None:
        plot_design_from_record(plot_source, I0, I1, N_values=(1, 3, 5), mu0=MU0,
                                 savepath="examples/instance2_design.png")
        print(f"design plot uses n={plot_source.n} "
              f"({'retained' if retained is not None else 'best available, none admissible'})")

    print("\nSaved: examples/instance2_scan.{csv,json}, instance2_scan_table.tex, "
          "instance2_bound_vs_achieved.png, instance2_design.png")


if __name__ == "__main__":
    main()
