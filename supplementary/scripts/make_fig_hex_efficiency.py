"""
Fig. S4: generation efficiency and cost of the 2:sqrt(3) run.

a) generation efficiency (inverse number of generation attempts per pool) along the NS
   trajectory, for the rectangular cell (N = 9) and for the square-box run of Fig. 5f
   (N = 8), which uses the same NS and training hyperparameters;
b) time spent training the flow and generating each pool in the rectangular cell.
"""

import numpy as np
import matplotlib.pyplot as plt

import common as cm
from common import ps

ps.set_style()

# Printed at \linewidth under the paper's shared PRINT_SCALE.
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=0.39)
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

ax = axes[0]
for key, label, marker, color in [("hex", r"$2:\sqrt{3}$, $N=9$", "^", "C0"),
                                  ("f", r"$1:1$, $N=8$", "o", "C1")]:
    attempts, _ = cm.generation_attempts(cm.run_dir(key))
    progress = 100 * np.arange(len(attempts)) / (len(attempts) - 1)
    ax.plot(progress, 1 / attempts, marker, mfc="none", color=color, label=label)
    print(f"{key}: efficiency min {1 / attempts.max():.2e} at {progress[attempts.argmax()]:.0f}%, "
          f"last {1 / attempts[-1]:.2e}, first {1 / attempts[0]:.2e}")
# Low-efficiency regime of the rectangular cell (cf. the grey bands of Fig. 6).
ax.axvspan(25, 77, color="0.85", alpha=0.5, zorder=0, lw=0)
ax.axhline(1e-2, color="k", ls="--", lw=1.0, alpha=0.5)
ax.set_yscale("log")
ax.set_xlabel("Nested-sampling progress [%]")
ax.set_ylabel(r"Generation efficiency $\eta$")
ax.legend(loc="upper right", handletextpad=0.2, fontsize=ps.ANNOTATION_FONTSIZE)

# b) Same conventions as the timing mosaic of Fig. 5 (new_plotter.ipynb).
ax = axes[1]
run = cm.run_dir("hex")
t = cm.read_timings(run) / 3600
training, generation = t[:, 2], t[:, 3]
pools = np.arange(1, len(t) + 1)
ax.bar(pools, training, width=0.75, color="C0", edgecolor="black", linewidth=0.25,
       label="Network training")
ax.bar(pools, generation, bottom=training, width=0.75, color="C1", alpha=0.85,
       edgecolor="black", linewidth=0.25, label="Pool generation")
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.set_xlabel("Pool generated")
ax.set_ylabel("Time (hours)")
ax.set_ylim(0, 1.45 * (training + generation).max())
ax.margins(x=0.02)
ax.legend(loc="upper right", fontsize=ps.ANNOTATION_FONTSIZE)

# Energy threshold of each training stage on the upper axis: five ticks evenly spaced
# across the panel, labelled with energies spaced geometrically in U - U_min + 1.
conds = np.array([cm.load(run / f"conds_{i:04d}.pt").item() for i in range(len(t))])
shift = 1.0 - conds[-1]
ticks_u = np.geomspace(conds[0] + shift, conds[-1] + shift, 5) - shift
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(np.linspace(pools[0], pools[-1], 5), [f"{u:.1f}" for u in ticks_u],
                  fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
ax_top.set_xlabel(r"$U_\mathrm{max}$", fontsize=ps.ANNOTATION_FONTSIZE, labelpad=4)
ax_top.tick_params(direction="in", length=4, pad=2)
ax_top.spines["right"].set_visible(False)
ax_top.spines["left"].set_visible(False)

# Wall time and energy evaluations: every evaluation of the potential in the run, i.e.
# pool generation, the warm-up and dilution MC stages, training and pool draws.
n_evals = cm.energy_budget(run)["total"]
wall = cm.elapsed_hours(run)
ax.text(0.03, 0.97, f"Wall Time ≈ {wall:.1f} h\nEnergy Eval ≈ {n_evals:.2e}",
        transform=ax.transAxes, ha="left", va="top", fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"))
print(f"hex: training {training.sum():.2f} h, generation {generation.sum():.2f} h, "
      f"std NS {t[:, 0].sum():.2f} h, pool NS {t[:, 1].sum():.2f} h, wall {wall:.2f} h, "
      f"energy evaluations {n_evals:.3e}")

# Panel labels at one common height, above the upper U_max axis of b), each aligned with
# the left edge of its panel. The layout is frozen first so the positions hold on save.
fig.canvas.draw()
fig.set_layout_engine("none")
to_fig = fig.transFigure.inverted()
renderer = fig.canvas.get_renderer()
y_top = max(to_fig.transform((0, a.get_tightbbox(renderer).y1))[1] for a in [*axes, ax_top])
for a, tag in zip(axes, "ab"):
    fig.text(a.get_position().x0, y_top + 0.01, f"{tag})", ha="left", va="bottom",
             fontsize=ps.AXES_TITLESIZE, fontweight="bold")

ps.savefig_all(fig, cm.FIG_DIR / "fig_hex_efficiency", print_scale=ps.PRINT_SCALE)
