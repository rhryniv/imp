"""results_multi.csv: one row per (geometry,n,d,sigma), merging the sweep
checkpoint with reanalyze.py's clustered active-point recount (the raw
per-grid-point counts from the sweep overcount near-flat equioscillation
peaks -- see reanalyze.py's docstring)."""
from __future__ import annotations

import csv
import json

with open("results_multi_checkpoint.json") as f:
    R = json.load(f)
with open("reanalysis.json") as f:
    RE = json.load(f)

COLS = ["geometry", "n", "d", "L", "sigma", "feasible", "delta", "alpha", "rho",
        "A_minus1", "antisym_residual", "sym_residual",
        "n_active_B_per_component", "n_active_C_per_component",
        "kappa_min_per_C_component", "max_abs_kappa_I1", "beta_per_active_C_point",
        "best_start_type", "n_starts_converged", "n_starts_total", "wall_time_s"]

with open("results_multi.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for k, v in R.items():
        row = []
        re_diag = RE.get(k, {}).get("diag") if v["feasible"] else None
        for col in COLS:
            if col == "n_active_B_per_component" and re_diag:
                val = re_diag["n_active_B"]
            elif col == "n_active_C_per_component" and re_diag:
                val = re_diag["n_active_C"]
            elif col == "beta_per_active_C_point" and re_diag:
                val = re_diag["beta_per_active_C"]
            else:
                val = v.get(col)
            if col in ("alpha", "rho", "sigma") and val is not None:
                val = ";".join(f"{x:.10g}" for x in val)
            elif isinstance(val, list):
                val = ";".join(str(x) for x in val)
            row.append(val)
        w.writerow(row)

print("wrote results_multi.csv")
