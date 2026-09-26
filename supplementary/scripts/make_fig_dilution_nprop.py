"""
Fig. S6: how many walkers to move per dilution iteration.

A dilution stage of M iterations, each moving n_prop of the K walkers through
n_c trial displacements, costs

    M * n_prop * (n_c + 1) = nbar * K * (n_c + 1),   nbar = M * n_prop / K,

so the energy budget is fixed by nbar alone, the mean number of times each walker is
refreshed (the SM calls it n-bar; R there is the reduction in energy evaluations).
What n_prop controls, at fixed nbar, is not the cost but the spread: the
refreshes per walker are Binomial(M, p) with p = (n_prop - 1) / (K - 1), and only
at n_prop = K does every walker get exactly M of them, with no tail at zero.

Panel a: probability that every walker is refreshed, at fixed cost, against n_prop.
Panel b: what guaranteed refreshment costs, as the reduction in energy evaluations
of the run of Fig. 4f when M is set to the smallest value reaching P_1 >= 0.99.

Since P(Bin(M,p) >= 1) = 1 - (1-p)^M is smooth in M, panel a uses that closed form
and does not need M to be an integer. Panel b does, and takes the ceiling.
"""

import numpy as np
import matplotlib.pyplot as plt

import common as cm
from common import ps

ps.set_style()

K = 10000            # live points
N_C = 100            # trial displacements per walker per iteration
N_PROP_RUN = 1000    # walkers moved per iteration in every run of this work
M_RUN = 100          # dilution iterations per stage in every run of this work
P_FULL = 0.99        # what we call "full replacement of the live set"
STD_NS_EVALS = 5.05e10


def coverage(n_prop, M):
    """P(every walker refreshed at least once) after M iterations moving n_prop walkers."""
    n_prop, M = np.atleast_1d(n_prop).astype(float), np.atleast_1d(M).astype(float)
    p = (n_prop - 1) / (K - 1)
    hit = 1 - (1 - p) ** M                       # smooth in M; exactly 1 at n_prop = K
    with np.errstate(divide="ignore"):
        return np.exp(K * np.log(np.clip(hit, 1e-300, 1.0)))


def m_for_full(n_prop):
    """Smallest integer M reaching P_1 >= P_FULL, moving n_prop walkers per iteration."""
    p = np.atleast_1d((np.asarray(n_prop, float) - 1) / (K - 1))
    m = np.ones_like(p)                          # n_prop = K: one sweep touches everything
    partial = p < 1.0
    with np.errstate(divide="ignore"):
        m[partial] = np.log(1 - P_FULL ** (1 / K)) / np.log(1 - p[partial])
    return np.maximum(1, np.ceil(m))


def main():
    b = cm.energy_budget(cm.run_dir("f"))
    n_stages = b["n_pools"] - 1                  # the last pool is never exhausted
    per_walker = N_C + 1                         # evaluations per walker per iteration
    fixed = b["total"] - n_stages * M_RUN * N_PROP_RUN * per_walker

    def reduction(n_prop, M):
        cost = n_stages * M * n_prop * per_walker
        return STD_NS_EVALS / (fixed + cost)

    ceiling = STD_NS_EVALS / fixed
    grid = np.arange(200, K + 1, 10.0)

    fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=0.42)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    # ---- a) coverage at fixed cost ----------------------------------------
    for nbar, c, ls in ((1, "C0", "-"), (3, "C1", "--"), (5, "C4", "-."), (10, "C2", ":")):
        axa.plot(grid / K, coverage(grid, nbar * K / grid), color=c, ls=ls, lw=ps.LW,
                 label=rf"$\bar{{n}} = {nbar}$")
    axa.axhline(P_FULL, color="k", ls=":", lw=0.8)
    axa.plot([N_PROP_RUN / K], coverage(N_PROP_RUN, M_RUN), "o", color="k", ms=ps.MS, zorder=5)
    axa.annotate("this work", xy=(N_PROP_RUN / K, coverage(N_PROP_RUN, M_RUN)[0]),
                 xytext=(0.13, 0.70), fontsize=ps.ANNOTATION_FONTSIZE)
    axa.set_xlabel(r"Walkers Moved per Iteration $n_{\mathrm{prop}}/K$")
    axa.set_ylabel("Probability All Walkers Refreshed")
    axa.set_xlim(0, 1.02)
    axa.set_ylim(-0.02, 1.05)
    axa.legend(frameon=False, loc="lower left")

    # ---- b) what a guarantee costs ----------------------------------------
    M_full = m_for_full(grid)
    axb.plot(grid / K, reduction(grid, M_full), color="C3", lw=ps.LW)
    axb.axhline(ceiling, color="k", ls="--", lw=0.8, alpha=0.6)
    axb.annotate(rf"$\mathscr{{M}}\to 0$: {ceiling:.0f}$\times$", xy=(0.05, ceiling),
                 xytext=(0.05, ceiling + 5), fontsize=ps.ANNOTATION_FONTSIZE)
    r_run = float(reduction(N_PROP_RUN, M_RUN))
    axb.plot([N_PROP_RUN / K], [r_run], "o", color="k", ms=ps.MS, zorder=5)
    axb.annotate(f"this work: {r_run:.0f}" + r"$\times$", xy=(N_PROP_RUN / K, r_run),
                 xytext=(0.09, r_run + 8), fontsize=ps.ANNOTATION_FONTSIZE)
    r_sweep = float(reduction(K, 1))
    axb.plot([1.0], [r_sweep], "o", color="k", ms=ps.MS, zorder=5)
    axb.annotate(f"{r_sweep:.0f}" + r"$\times$", xy=(1.0, r_sweep),
                 xytext=(0.875, r_sweep - 1), fontsize=ps.ANNOTATION_FONTSIZE)
    axb.set_xlabel(r"Walkers Moved per Iteration $n_{\mathrm{prop}}/K$")
    axb.set_ylabel("Reduction in Energy Evaluations")
    axb.set_xlim(0, 1.02)
    axb.set_ylim(0, ceiling * 1.12)

    for ax, tag in ((axa, "a"), (axb, "b")):
        ax.set_title(f"{tag})", loc="left", fontweight="bold")

    ps.savefig_all(fig, cm.FIG_DIR / "fig_dilution_nprop", print_scale=ps.PRINT_SCALE)

    # ---- numbers quoted in the text ---------------------------------------
    print(f"run f: {n_stages} dilution stages, fixed {fixed/1e8:.3f}e8, ceiling {ceiling:.0f}x")
    print(f"this work: n_prop={N_PROP_RUN}, M={M_RUN}, R={M_RUN*N_PROP_RUN/K:.0f}, "
          f"P_1={coverage(N_PROP_RUN, M_RUN)[0]:.2f}, {r_run:.0f}x")
    print(f"{'n_prop':>7} {'M(P>=0.99)':>11} {'R':>6} {'reduction':>10}")
    for n in (500, 1000, 2000, 5000, 7500, K):
        M = m_for_full(np.array([float(n)]))[0]
        print(f"{n:>7} {M:>11.0f} {M*n/K:>6.1f} {float(reduction(n, M)):>9.0f}x")


if __name__ == "__main__":
    main()
