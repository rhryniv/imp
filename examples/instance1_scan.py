"""Instance 1 (spec Sec. 7, m_0 = m_1 = 1): the Stage 3 degree scan,
n = 2..16, I_0 = [pi/6, pi/4], I_1 = [pi/2, b] with b = 3*pi/4 (< pi, per
the spec's own note that b=pi now fails rule 7 since L=2n is fixed;
b=3*pi/4 chosen by RH), mu_0 = 0.05, eps_0 = eps_1 = 1e-2.

Produces the Sec. 6 emissions table (CSV + LaTeX table body, deliverable
2), the two deliverable-3 plots (kappa_B/T_N for the retained design;
underline_delta(n) vs delta_achieved(n) with n_min_green marked), and a
JSON dump of every DegreeRecord in the scan.

Run from the repository root: python3 examples/instance1_scan.py
"""
import json
import time

import numpy as np

from scattering.driver import degree_scan_stage3, retained_degree_record, save_records_csv, save_records_latex
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(np.pi / 6, np.pi / 4)]
I1 = [(np.pi / 2, 3 * np.pi / 4)]
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
              f"dual={r.dual_status}")

    retained = retained_degree_record(records)
    print()
    if retained is None:
        print("No degree in the scanned range is admissible (rule 4).")
    else:
        print(f"Retained (least admissible n): n={retained.n}, delta_achieved={retained.delta_achieved}, "
              f"sigma_star={retained.sigma_star}")

    save_records_csv(records, "examples/instance1_scan.csv")
    save_records_latex(records, "examples/instance1_scan_table.tex")
    with open("examples/instance1_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained, savepath="examples/instance1_bound_vs_achieved.png")

    plot_source = retained if retained is not None else next(
        (r for r in reversed(records) if r.status == "optimal" and r.alpha is not None), None)
    if plot_source is not None:
        plot_design_from_record(plot_source, I0, I1, N_values=(1, 3, 5), mu0=MU0,
                                 savepath="examples/instance1_design.png")
        print(f"design plot uses n={plot_source.n} "
              f"({'retained' if retained is not None else 'best available, none admissible'})")

    print("\nSaved: examples/instance1_scan.{csv,json}, instance1_scan_table.tex, "
          "instance1_bound_vs_achieved.png, instance1_design.png")


if __name__ == "__main__":
    main()
