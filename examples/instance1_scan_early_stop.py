"""Instance 1 (spec Sec. 7, m_0 = m_1 = 1), rerun with the
stop_at_first_admissible early exit (driver.find_retained_design_stage3):
same parameters as instance1_scan.py -- I_0=[pi/6,pi/4], I_1=[pi/2,b]
with b=3*pi/4, eps_0=eps_1=1e-2, N_max=60 -- but the scan halts the
moment it finds an admissible degree, per rule 4+5 (see
driver.degree_scan_stage3's own docstring for why this is lossless, not
an approximation).

Produces the same per-record fields as instance1_scan.py for whatever
(possibly short) prefix of n=2..16 was actually scanned, under an
"_early_stop" suffix so neither run's output overwrites the other's.

Run from the repository root: python3 examples/instance1_scan_early_stop.py
"""
import json
import time

import numpy as np

from scattering.driver import find_retained_design_stage3, save_records_csv, save_records_latex, mu0_from_N_max
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(np.pi / 6, np.pi / 4)]
I1 = [(np.pi / 2, 3 * np.pi / 4)]
EPS0 = EPS1 = 1e-2
N_MAX = 60
N_RANGE = range(2, 17)
N_VALUES = (20, 40, 60, 80, 100)


def _fmt(x):
    return "None" if x is None else f"{x:.6e}"


def main():
    t0 = time.time()
    retained, records = find_retained_design_stage3(N_RANGE, I0, I1, N_MAX, EPS0, EPS1, N_values=N_VALUES)
    elapsed = time.time() - t0
    print(f"scan finished in {elapsed:.1f}s, {len(records)} degree(s) computed "
          f"(of {len(list(N_RANGE))} in the full range)")

    for r in records:
        print(f"n={r.n:2d} status={r.status:20s} underline_delta={_fmt(r.underline_delta):>14} "
              f"delta_achieved={_fmt(r.delta_achieved):>14} kappa_min={_fmt(r.kappa_min):>14} "
              f"max_kappa_B={_fmt(r.max_kappa_B):>14} admissible={r.admissible} "
              f"time_direct_s={r.time_direct_s:.1f}")

    print()
    if retained is None:
        print("No degree in the scanned range is admissible (rule 4) -- full range was scanned.")
    else:
        print(f"Retained (least admissible n): n={retained.n}, delta_achieved={retained.delta_achieved}, "
              f"sigma_star={retained.sigma_star}, max_kappa_B={retained.max_kappa_B}")

    save_records_csv(records, "examples/instance1_early_stop_scan.csv")
    save_records_latex(records, "examples/instance1_early_stop_scan_table.tex")
    with open("examples/instance1_early_stop_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained, savepath="examples/instance1_early_stop_bound_vs_achieved.png")

    plot_source = retained if retained is not None else next(
        (r for r in reversed(records) if r.status == "optimal" and r.alpha is not None), None)
    if plot_source is not None:
        plot_design_from_record(plot_source, I0, I1, N_values=(1, 3, 5), mu0=mu0_from_N_max(EPS0, N_MAX),
                                 savepath="examples/instance1_early_stop_design.png")
        print(f"design plot uses n={plot_source.n} "
              f"({'retained' if retained is not None else 'best available, none admissible'})")

    print("\nSaved: examples/instance1_early_stop_scan.{csv,json}, instance1_early_stop_scan_table.tex, "
          "instance1_early_stop_bound_vs_achieved.png, instance1_early_stop_design.png")


if __name__ == "__main__":
    main()
