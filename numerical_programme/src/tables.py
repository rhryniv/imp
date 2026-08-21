"""T1 (tab-spec.tex), T2 (tab-results.tex, sandwich table both specs), T3
(tab-constraints.tex, skeleton -- the brief states the user will supply
the constraint-column text, so only headers/structure are emitted).
LaTeX booktabs + siunitx S columns.
"""
from __future__ import annotations

import json

import mpmath as mp

from geometry import setup, delta_n_closed

OUT = "../tables"


def frac_str(lo_hi):
    # renders a (num,den) fraction-of-pi tuple as e.g. "5\pi/6"
    (n1, d1), (n2, d2) = lo_hi
    def one(n, d):
        if d == 1 and n == 1:
            return r"\pi"
        if d == 1:
            return rf"{n}\pi"
        return rf"{n}\pi/{d}"
    return one(n1, d1), one(n2, d2)


def make_t1():
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Specification parameters for Specs A, B, C.}")
    lines.append(r"\label{tab-spec}")
    lines.append(r"\begin{tabular}{l l l S S S S}")
    lines.append(r"\toprule")
    lines.append(r"Spec & $I_1$ & $I_0$ & {$\mu_0$} & {$N_{\max}$} & {$\epsilon_0$} & {$\epsilon_1$} \\")
    lines.append(r"\midrule")
    from geometry import SPECS
    for name in ["A", "B", "C"]:
        geo = setup(name, 30)
        spec = SPECS[name]
        t_str = frac_str(((spec["t_frac"][0], spec["t_frac"][1]), (0, 1)))[0]
        i0_strs = []
        for lo, hi in spec["I0_fracs"]:
            lo_s, hi_s = frac_str((lo, hi))
            i0_strs.append(f"[{lo_s},{hi_s}]")
        i0_str = r" $\cup$ ".join(i0_strs)
        i1_str = f"[0,{t_str}]"
        eps0 = float(geo["eps_0"])
        eps1 = float(geo["eps_1"])
        lines.append(f"{name} & ${i1_str}$ & ${i0_str}$ & {float(geo['mu0']):.0f} & "
                      f"{geo['N_max']} & {eps0:.6e} & {eps1:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(f"{OUT}/tab-spec.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab-spec.tex")


def make_t2():
    with open("../data/e1_results.json") as f:
        e1 = json.load(f)
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{$\delta_{\rm mag}(n)$ sandwich results, Specs A and B, "
                 r"$n=1,\dots,8$. Odd $n$: exact closed form "
                 r"$S/\cosh^2(n\gamma/2)$, verified via the $\hat Q_n$ construction. "
                 r"Even $n$: numeric SDP value only (no closed form), sandwiched between "
                 r"$S/\cosh^2(n\gamma/2)$ and $S/\cosh^2((n-1)\gamma/2)$.}")
    lines.append(r"\label{tab-results}")
    lines.append(r"\begin{tabular}{c S S S S}")
    lines.append(r"\toprule")
    lines.append(r"{$n$} & {$\delta_{\rm mag}(n)$, Spec A} & {closed/sandwich, A} & "
                 r"{$\delta_{\rm mag}(n)$, Spec B} & {closed/sandwich, B} \\")
    lines.append(r"\midrule")
    for name in ["A", "B"]:
        pass
    geoA, geoB = setup("A", 30), setup("B", 30)
    for n in range(1, 9):
        rowA = e1["A"][str(n)]
        rowB = e1["B"][str(n)]
        if n % 2 == 1:
            cA = f"{float(delta_n_closed(n, geoA)):.6e} (exact)"
            cB = f"{float(delta_n_closed(n, geoB)):.6e} (exact)"
        else:
            loA = float(delta_n_closed(n, geoA)); hiA = float(delta_n_closed(n - 1, geoA))
            loB = float(delta_n_closed(n, geoB)); hiB = float(delta_n_closed(n - 1, geoB))
            cA = f"[{loA:.4e}, {hiA:.4e}]"
            cB = f"[{loB:.4e}, {hiB:.4e}]"
        lines.append(f"{n} & {rowA['delta_sdp']:.6e} & \\text{{{cA}}} & "
                      f"{rowB['delta_sdp']:.6e} & \\text{{{cB}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(f"{OUT}/tab-results.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab-results.tex")


def make_t3():
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Constraint summary (skeleton -- text to be supplied).}")
    lines.append(r"\label{tab-constraints}")
    lines.append(r"\begin{tabular}{l l l}")
    lines.append(r"\toprule")
    lines.append(r"Label & Constraint & Notes \\")
    lines.append(r"\midrule")
    for lab in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
        lines.append(f"{lab} & TODO & TODO \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(f"{OUT}/tab-constraints.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab-constraints.tex (skeleton, per brief instruction)")


if __name__ == "__main__":
    make_t1()
    make_t2()
    make_t3()
