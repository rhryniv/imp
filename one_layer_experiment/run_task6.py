"""Task 6 -- the n=1 benchmark against the magnitude bound. High
precision (mpmath, 30 digits): closed forms only (delta_mag(1) is itself
a closed form from the earlier SDP-certificate-test work, not derived
from the transfer matrices here -- nothing to cross-check against
Section 1 in this task, just the two closed forms and their ratio)."""
from __future__ import annotations

import mpmath as mp

from closed_forms import s_minus_mp, delta_min_mp, delta_mag1_mp

mp.mp.dps = 30

MU0 = mp.mpf(1)
U = 3 * mp.pi / 4


def report_for_t(t, label):
    sm = s_minus_mp(MU0, U)
    dstar = delta_min_mp(MU0, t, U)
    dmag = delta_mag1_mp(MU0, t, U)
    ratio = dstar / dmag
    identity_rhs = 2 * sm * (1 + sm) * (1 - mp.cos(U)) / mp.sinh(MU0) ** 2
    identity_err = abs(ratio - identity_rhs)

    print(f"--- {label} (t={mp.nstr(t,6)}) ---")
    print(f"  s_-            = {mp.nstr(sm, 10)}")
    print(f"  delta*_1       = {mp.nstr(dstar, 10)}")
    print(f"  delta_mag(1)   = {mp.nstr(dmag, 10)}")
    print(f"  ratio          = {mp.nstr(ratio, 10)}")
    print(f"  identity RHS   = {mp.nstr(identity_rhs, 10)}")
    print(f"  |ratio - RHS|  = {mp.nstr(identity_err, 6)}")
    return ratio


def run():
    r_baseline = report_for_t(mp.pi / 4, "baseline t=pi/4")
    r_t6 = report_for_t(mp.pi / 6, "t=pi/6")
    r_t3 = report_for_t(mp.pi / 3, "t=pi/3")

    print("\n=== t-independence check ===")
    print(f"  ratio(t=pi/4) = {mp.nstr(r_baseline, 10)}")
    print(f"  ratio(t=pi/6) = {mp.nstr(r_t6, 10)}")
    print(f"  ratio(t=pi/3) = {mp.nstr(r_t3, 10)}")
    print(f"  max pairwise diff = {mp.nstr(max(abs(r_baseline-r_t6), abs(r_baseline-r_t3), abs(r_t6-r_t3)), 6)}")


if __name__ == "__main__":
    run()
