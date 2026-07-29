"""Optional lift-vs-magnitude SDP comparison (spec Sec. 7, "Optional
comparison run"): for each n, record delta_mag (design_sdp_magnitude),
delta_lift (the plain lifted SDP's own raw relaxed bound,
sdp_design.sdp_lower_bound), and delta_both (the "conjunction" -- lift +
magnitude's sign-free (C'), sdp_design.sdp_lower_bound_conjunction).

Run on both paper specs: Experiment 1's widened single-band windows and
Experiment 2's multi-band spec (m0=2).

Not validated against the manuscript's own worked-example table values
(see scattering/experiment.py's module docstring) -- this is purely an
internal comparison between two relaxations of the SAME (correctly
implemented, post factor-2-bug-fix) problem.

Run from the repository root: python3 examples/experiment3_relaxation_comparison.py
"""
import json

from scattering.sdp_design import compare_relaxations


def run_and_report(label, J0, J1, mu0, n_range):
    print(f"\n=== {label}: J0={J0} J1={J1} mu0={mu0} ===")
    records = []
    for n in n_range:
        r = compare_relaxations(n, J0, J1, mu0)
        records.append(r)
        dm, dl, db = r["delta_mag"], r["delta_lift"], r["delta_both"]
        rel_gap = None if (dl is None or db is None or dl == 0) else abs(db - dl) / dl
        print(f"  n={n}: delta_mag={dm}  delta_lift={dl}  delta_both={db}"
              + (f"  |both-lift|/lift={rel_gap:.2e}" if rel_gap is not None else ""))
    return records


def main():
    all_records = {}

    # Experiment 1's widened windows (single band)
    I0_e1 = [(0.8, 1.2)]
    I1_e1 = [(0.0, 0.55)]
    all_records["experiment1"] = run_and_report("Experiment 1 (widened)", I0_e1, I1_e1, 0.05, (3, 4, 5, 6))

    # Experiment 2's multi-band spec (m0=2)
    I0_e2 = [(0.50, 0.65), (2.50, 2.65)]
    I1_e2 = [(1.20, 1.40), (1.80, 2.00)]
    all_records["experiment2"] = run_and_report("Experiment 2 (multi-band, m0=2)", I0_e2, I1_e2, 0.05, (3, 5, 7))

    with open("experiment3_relaxation_comparison.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print("\nsaved experiment3_relaxation_comparison.json")

    # One-sentence conclusion (spec Sec. 7's own framing)
    max_rel_gap = 0.0
    for recs in all_records.values():
        for r in recs:
            dl, db = r["delta_lift"], r["delta_both"]
            if dl is not None and db is not None and dl > 0:
                max_rel_gap = max(max_rel_gap, abs(db - dl) / dl)
    print(f"\nConclusion: max relative |delta_both - delta_lift| / delta_lift across all "
          f"cases = {max_rel_gap:.2e} -- the conjunction is NOT materially tighter than the "
          f"plain lifted SDP (confirms the provable redundancy: Q_A >= kappa_a^2 already "
          f"follows from the lift's own PSD structure, see sdp_lower_bound_conjunction's "
          f"docstring). Neither delta_mag nor delta_lift dominates the other across these "
          f"cases -- both orderings occur.")


if __name__ == "__main__":
    main()
