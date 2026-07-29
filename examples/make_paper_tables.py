"""Generate the LaTeX tables for Section 6.5 (paper) from the two
experiments' saved JSON records (experiment1_level3_scan.json,
experiment2_scan.json), matching the manuscript's existing tab:degree-scan
/ tab:TN column style. Does NOT rerun any SDP/design_direct -- run
experiment1_baseline_scan.py / experiment2_multiband.py first (or after
changing I0/I1/mu0) to regenerate those JSON files.

Run from the repository root: python3 examples/make_paper_tables.py
"""
import json

import numpy as np

from scattering.forward import a_from_alphas, transmission_TN
from scattering.certify import certify

N_TN = (1, 5, 10, 20)


def _fmt(x, sig=2):
    if x is None:
        return "--"
    return f"{x:.{sig}e}".replace("e-0", "e-").replace("e+0", "e+")


def degree_scan_table(records, mu0, caption, label):
    lines = [
        r"\begin{table}[ht]",
        r"\t\centering",
        r"\t\begin{tabular}{r|ccccc}",
        r"\t\t$n$ & status & $\delta_{\mathrm{mag}}$ & $\delta_{\mathrm{achieved}}$ & "
        r"$\delta_{\mathrm{certified}}$ & $\mu_{\min}^{\mathrm{cert}}$ \\",
        r"\t\t\hline",
    ]
    for r in records:
        status = "opt" if r["status"] == "optimal" else r["status"].replace("_", r"\_")
        mu_min_cert = r["mu_min_certified"]
        mu_min_str = "" if mu_min_cert is None else f"{mu_min_cert:.4f}"
        lines.append(
            f"\t\t{r['n']} & {status} & {_fmt(r['delta_mag'])} & {_fmt(r['delta_achieved'])} & "
            f"{_fmt(r['delta_certified'])} & {mu_min_str} \\\\"
        )
    lines += [
        r"\t\end{tabular}",
        f"\t\\caption{{{caption}}}\\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tn_performance_table(a, I0, I1, n_design, delta, caption, label):
    lines = [
        r"\begin{table}[ht]",
        r"\t\centering",
        r"\t\begin{tabular}{r|cc}",
        r"\t\t$N$ & $\max_{I_0}T_N$ & $\min_{I_1}T_N$ \\",
        r"\t\t\hline",
    ]
    for N in N_TN:
        maxT0 = max(float(np.max(transmission_TN(a, np.linspace(lo, hi, 4000), N))) for lo, hi in I0) if I0 else float("nan")
        minT1 = min(float(np.min(transmission_TN(a, np.linspace(lo, hi, 4000), N))) for lo, hi in I1) if I1 else float("nan")
        lines.append(f"\t\t{N} & {_fmt(maxT0, 2)} & {minT1:.4f}" + r" \\")
    lines += [
        r"\t\end{tabular}",
        f"\t\\caption{{{caption.format(n=n_design, delta=f'{delta:.2e}')}}}\\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def best_record(records):
    feasible = [r for r in records if r["status"] == "optimal" and r["delta_achieved"] is not None]
    return min(feasible, key=lambda r: r["delta_achieved"]) if feasible else None


def main():
    out = []

    # --- Experiment 1 ---
    with open("experiment1_level3_scan.json") as f:
        recs1 = json.load(f)
    I0_e1, I1_e1, mu0_e1 = [(0.8, 1.2)], [(0.0, 0.45)], 0.05
    out.append("% ===== Experiment 1: degree scan =====")
    out.append(degree_scan_table(
        recs1, mu0_e1,
        caption=(r"Experiment 1: achieved pass-band ripple vs.\ block complexity~$n$ for the "
                 r"baseline-derived instance $I_0=(0.8,1.2)$, $I_1=(0,0.45)$, $\mu_0=0.05$."),
        label="tab:exp1-degree-scan",
    ))
    best1 = best_record(recs1)
    a1 = a_from_alphas(np.array(best1["alpha"]))
    out.append("\n% ===== Experiment 1: T_N performance at best n =====")
    out.append(tn_performance_table(
        a1, I0_e1, I1_e1, best1["n"], best1["delta_achieved"],
        caption=r"Experiment 1: transmission probability for the best design ($n={n}$, $\delta={delta}$).",
        label="tab:exp1-TN",
    ))

    # --- Experiment 2 ---
    with open("experiment2_scan.json") as f:
        recs2 = json.load(f)
    I0_e2 = [(0.50, 0.65), (2.50, 2.65)]
    I1_e2 = [(1.20, 1.40), (1.80, 2.00)]
    out.append("\n% ===== Experiment 2: degree scan =====")
    out.append(degree_scan_table(
        recs2, 0.05,
        caption=(r"Experiment 2: achieved pass-band ripple vs.\ block complexity~$n$ for the "
                 r"multi-band instance $I_0=(0.50,0.65)\cup(2.50,2.65)$, "
                 r"$I_1=(1.20,1.40)\cup(1.80,2.00)$, $\mu_0=0.05$."),
        label="tab:exp2-degree-scan",
    ))
    best2 = best_record(recs2)
    a2 = a_from_alphas(np.array(best2["alpha"]))
    out.append("\n% ===== Experiment 2: T_N performance at best n =====")
    out.append(tn_performance_table(
        a2, I0_e2, I1_e2, best2["n"], best2["delta_achieved"],
        caption=r"Experiment 2: transmission probability for the best design ($n={n}$, $\delta={delta}$).",
        label="tab:exp2-TN",
    ))

    text = "\n".join(out)
    with open("paper_tables.tex", "w") as f:
        f.write(text)
    print(text)
    print("\nsaved paper_tables.tex")


if __name__ == "__main__":
    main()
