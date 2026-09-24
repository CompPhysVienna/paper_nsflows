"""
Fig. S6: how many dilution steps are needed, and what they cost.

Every time the pool is exhausted the algorithm runs M standard NS iterations
before retraining. Each of them refreshes n_propagate of the K live points: the
highest-energy walker, which is always replaced, plus n_propagate - 1 others
drawn uniformly without replacement. Across iterations the draws are
independent, so a given walker is missed with probability (1 - p)^M with
p = (n_propagate - 1) / (K - 1), and the number of iterations needed to touch
every walker is the coupon-collector problem.

Panel a: probability that every walker has been picked at least n times.
Panel b: the resulting reduction in energy evaluations for the run of Fig. 5f,
whose dilution term is the only one that scales with M. The two marked points
are the M used in this work and the M that makes full replacement near certain.

The analytic curve of panel a is checked against a direct simulation of the
selection; --simulate runs that check and prints the comparison.
"""

import argparse

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

import common as cm
from common import ps

ps.set_style()

K = 10000            # live points
N_PROP = 1000        # walkers refreshed per standard-NS iteration
M_RUN = 100          # the value used for every run in the paper
P_FULL = 0.99        # what we call "full replacement of the live set"
STD_NS_EVALS = 5.05e10

p = (N_PROP - 1) / (K - 1)   # a given walker is among those drawn, per iteration


def p_all_touched(M, n=1):
    """Probability that all K walkers have been picked at least n times."""
    M = np.atleast_1d(M).astype(float)
    hit = binom.sf(n - 1, M, p)                       # P(a given walker picked >= n times)
    with np.errstate(divide="ignore"):
        return np.exp(K * np.log(np.clip(hit, 1e-300, 1.0)))


def main(simulate=False):
    # ---- cost model: only the dilution term scales with M ------------------
    b = cm.energy_budget(cm.run_dir("f"))
    per_iteration = b["mcmc"] / b["n_std_iterations"]
    n_pools = b["n_pools"]
    fixed = b["total"] - b["mcmc"]                    # generation, training, pool draws
    reduction = lambda M: STD_NS_EVALS / (fixed + n_pools * M * per_iteration)

    ceiling = STD_NS_EVALS / fixed
    M_grid = np.arange(1, 251)
    # smallest M for which every walker is refreshed with probability >= P_FULL
    M_full = int(M_grid[p_all_touched(M_grid) >= P_FULL][0])

    fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=0.42)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    # ---- a) coverage -------------------------------------------------------
    for n, c, ls in ((1, "C0", "-"), (2, "C1", "--"), (3, "C2", ":")):
        axa.plot(M_grid, p_all_touched(M_grid, n), color=c, ls=ls, lw=ps.LW,
                 label=rf"$n \geq {n}$")
    axa.axhline(0.99, color="k", ls=":", lw=0.8)
    axa.axvline(M_RUN, color="k", lw=0.8, alpha=0.6)
    axa.annotate(rf"$\mathscr{{M}}={M_RUN}$", xy=(M_RUN, 0.06), xytext=(M_RUN + 8, 0.06),
                 fontsize=ps.ANNOTATION_FONTSIZE)
    axa.set_xlabel(r"Dilution Steps $\mathscr{M}$")
    axa.set_ylabel("Probability All Walkers Refreshed")
    axa.set_xlim(0, 250)
    axa.set_ylim(-0.02, 1.05)
    axa.legend(frameon=False, loc="lower right", fontsize=ps.LEGEND_FONTSIZE_SMALL)

    # ---- b) cost -----------------------------------------------------------
    axb.plot(M_grid, reduction(M_grid), color="C3", lw=ps.LW)
    axb.axhline(ceiling, color="k", ls="--", lw=0.8, alpha=0.6)
    axb.annotate(rf"$\mathscr{{M}}\to 0$: {ceiling:.0f}$\times$",
                 xy=(250, ceiling), xytext=(120, ceiling + 4),
                 fontsize=ps.ANNOTATION_FONTSIZE)
    marks = ((M_RUN, rf"$\mathscr{{M}}={M_RUN}$ (this work)", 8),
             (M_full, rf"$\mathscr{{M}}={M_full}$ ($P_1={P_FULL}$)", -28))
    for M, txt, dy in marks:
        axb.plot([M], [reduction(M)], "o", color="k", ms=ps.MS)
        axb.annotate(txt, xy=(M, reduction(M)), xytext=(M + 8, reduction(M) + dy),
                     fontsize=ps.ANNOTATION_FONTSIZE)
    axb.set_xlabel(r"Dilution Steps $\mathscr{M}$")
    axb.set_ylabel("Reduction in Energy Evaluations")
    axb.set_xlim(0, 250)
    axb.set_ylim(0, ceiling * 1.15)

    ps.savefig_all(fig, cm.FIG_DIR / "fig_dilution", print_scale=ps.PRINT_SCALE)

    # ---- numbers quoted in the text ---------------------------------------
    print(f"p = {p:.6f}   coupon-collector scale ln(K)/-ln(1-p) = {np.log(K)/-np.log(1-p):.1f}")
    print(f"fixed (non-dilution) cost {fixed/1e8:.2f}e8, {n_pools} pools, "
          f"{per_iteration:.0f} evaluations per iteration, ceiling {ceiling:.0f}x")
    print(f"M for P(all refreshed) >= {P_FULL}: {M_full}  -> reduction {reduction(M_full):.0f}x")
    for M in (87, M_RUN, M_full, 159):
        print(f"  M={M:>4}  reduction {reduction(M):>5.0f}x   P(n>=1) {p_all_touched(M)[0]:.4f}   "
              f"P(n>=2) {p_all_touched(M, 2)[0]:.4f}   E[untouched] {K*(1-p)**M:.2f}")

    if simulate:
        rng = np.random.default_rng(0)
        print("\n  simulation check of the independence approximation")
        for M in (87, 100, 132):
            hits = 0
            for _ in range(2000):
                touched = np.zeros(K, dtype=bool)
                for _ in range(M):
                    touched[rng.choice(K, N_PROP, replace=False)] = True
                hits += bool(touched.all())
            print(f"    M={M:>3}: simulated {hits/2000:.4f}   analytic {p_all_touched(M)[0]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true",
                    help="check the analytic coverage against a direct simulation")
    main(**vars(ap.parse_args()))
