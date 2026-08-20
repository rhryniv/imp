"""Tables 3 (tab_designs) and 4 (tab_sweep)."""
from __future__ import annotations

import json

from geometry import setup, delta_n
from task_b import main as task_b_main

DELTA_TARGET = 1e-2


def tab_designs():
    with open("task_c_results.json") as f:
        results = json.load(f)

    lines = [r"\begin{tabular}{cc|ccc|ccc}",
             r"\hline",
             r"geom & $n$ & $\delta_{mag}(n)$ & $\delta^*_n$ & ratio & $L_{best}$ & sign & feasible \\",
             r"\hline"]
    for gname in ["G1", "G2"]:
        geo = setup(gname, 30)
        for n in range(1, 7):
            dmag = float(delta_n(n, geo))
            e = results.get(gname, {}).get(str(n), {})
            dstar = e.get("delta_star")
            if dstar is None:
                lines.append(rf"{gname} & {n} & {dmag:.3e} & --- & --- & --- & --- & no \\")
                continue
            ratio = dstar / dmag
            feas = "yes" if dstar <= DELTA_TARGET else "no"
            lines.append(rf"{gname} & {n} & {dmag:.3e} & {dstar:.3e} & {ratio:.3f} & "
                         rf"{e['L']} & {'+' if e['sign']==1 else '-'} & {feas} \\")
    lines += [r"\hline", r"\end{tabular}"]
    tex = "\n".join(lines)
    with open("tab_designs.tex", "w") as f:
        f.write(tex)
    print(tex)


def tab_sweep():
    tallies = task_b_main()
    lines = [r"\begin{tabular}{c|cc}",
             r"\hline",
             r" & actually constant & not constant \\",
             r"\hline"]
    for gname, t in tallies.items():
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{gname}}}}} \\")
        lines.append(rf"$L \le L_{{cap}}$ (predicted allowed) & {t['predicted_allowed_and_const']} & "
                     rf"{t['predicted_allowed_not_const']} \\")
        lines.append(rf"$L > L_{{cap}}$ (predicted disallowed) & {t['predicted_disallowed_and_const']} & "
                     rf"{t['predicted_disallowed_not_const']} \\")
    lines += [r"\hline", r"\end{tabular}"]
    tex = "\n".join(lines)
    with open("tab_sweep.tex", "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    print("=== tab_designs ===")
    tab_designs()
    print("\n=== tab_sweep ===")
    tab_sweep()
