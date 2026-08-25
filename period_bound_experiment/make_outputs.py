"""Build results.csv from results_checkpoint.json (spec Sec 5a)."""
from __future__ import annotations

import csv
import json

with open("results_checkpoint.json") as f:
    R = json.load(f)

COLS = ["n", "d", "L", "sigma", "feasible", "delta", "alpha", "rho",
        "kappa_min_I0", "kappa_max_I1", "Q_max_I0", "beta_J", "d_cap_aposteriori",
        "min_Q", "n_starts_converged", "n_starts_total", "best_start_type",
        "solver_iters", "wall_time_s"]

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for n in [1, 3, 5]:
        for d in range(1, 12):
            for sigma in [1, -1]:
                k = f"n{n}_d{d}_s{sigma}"
                v = R[k]
                row = []
                for col in COLS:
                    val = v.get(col)
                    if col in ("alpha", "rho") and val is not None:
                        val = ";".join(f"{x:.10g}" for x in val)
                    row.append(val)
                w.writerow(row)

print("wrote results.csv")
