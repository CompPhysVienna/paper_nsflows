"""
Fig. S4: generation efficiency and cost of the 2:sqrt(3) run.

a) generation efficiency (inverse number of generation attempts per pool) along the NS
   trajectory, for the rectangular cell (N = 9) and for the square-box run of Fig. 4f
   (N = 8), which uses the same NS and training hyperparameters;
b) time spent training the flow and generating each pool in the rectangular cell.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import common as cm
from common import ps
from nsflows.systems.lennard_jones import lennard_jones

ps.set_style()

# Panel a) uses the axes of Fig. 5: reduced energy U - U_0 on a reversed log abscissa
# (high energy left, basin right) with nested-sampling progress on top, so the two
# figures can be read against each other. PAD puts 0% and 100% exactly at the edges.
PAD = 1.15

# Same criterion as Fig. 5: the low-efficiency regime is the contiguous window around the
# efficiency minimum in which eta stays within SHADE_FACTOR of that minimum.
SHADE_FACTOR = 5.0

# Cells of the two runs, for the U_0 of each. U_0 is the inherent-structure proxy used in
# Fig. 5: the lowest live-point energy of the final live set (see internal_complexity.py).
CELLS = {"hex": (9, 2.864, 3.307), "f": (8, 2.9, 2.9)}


def u_zero(key):
    n, lx, ly = CELLS[key]
    last = cm.load(cm.run_dir(key) / "samples_000000500000.pt")
    last = torch.as_tensor(last).float().reshape(-1, n, 2)
    system = lennard_jones(n_particles=n, dimensions=2, rho=n / (lx * ly),
                           device=torch.device("cpu"), cutin=0.8, lrc=True,
                           aspect_ratio=None if lx == ly else ly / lx)
    with torch.no_grad():
        return float(system.energy(last).numpy().min())


def thresholds(key, n_pools):
    """Training threshold U_max of each pool, in pool order."""
    return np.array([cm.load(cm.run_dir(key) / f"conds_{i:04d}.pt").item()
                     for i in range(n_pools)])


# Printed at \linewidth under the paper's shared PRINT_SCALE.
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=0.39)
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

ax = axes[0]
series = []
for key, label, marker, color in [("hex", r"$2:\sqrt{3}$, $N=9$", "^", "C0"),
                                  ("f", r"$1:1$, $N=8$", "o", "C1")]:
    attempts, _ = cm.generation_attempts(cm.run_dir(key))
    x = thresholds(key, len(attempts)) - u_zero(key)
    series.append((key, x, 1 / attempts))
    ax.plot(x, 1 / attempts, marker, mfc="none", color=color, label=label, zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")

# Anchored on the square-box run, exactly as Fig. 5a is, so the progress axis of the two
# figures is one and the same scale. The rectangular cell fits inside it on both sides
# (its 2861 < 2890 and its 0.476 > 0.122), so nothing is clipped.
anchor = dict((k, x) for k, x, _ in series)["f"]
left, right = anchor.max() * PAD, anchor.min() / PAD
ax.set_xlim(left, right)
span = np.log10(left) - np.log10(right)


def to_pct(xv):
    xv = np.clip(np.asarray(xv, float), 1e-300, None)
    return (np.log10(left) - np.log10(xv)) / span * 100.0


def from_pct(p):
    return 10.0 ** (np.log10(left) - np.asarray(p, float) / 100.0 * span)


for key, x, eff in series:
    pct = to_pct(x)
    order = np.argsort(pct)
    p, e = pct[order], eff[order]
    inside = e < SHADE_FACTOR * e.min()
    lo = hi = int(e.argmin())
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(e) - 1 and inside[hi + 1]:
        hi += 1
    print(f"{key}: efficiency min {eff.min():.2e} at {pct[eff.argmin()]:.0f}%, "
          f"last {eff[-1]:.2e}, first {eff[0]:.2e}, U-U_0 from {x.max():.1f} to "
          f"{x.min():.3f}, curve spans {p.min():.0f}%-{p.max():.0f}%, "
          f"low-efficiency band {p[lo]:.0f}%-{p[hi]:.0f}% "
          f"(eta within {SHADE_FACTOR:g}x of min)")
    # Only the rectangular cell is shaded: it is the subject of this section, and the
    # two bands overlap almost entirely, so drawing both would be unreadable.
    if key == "hex":
        ax.axvspan(float(from_pct(p[lo])), float(from_pct(p[hi])),
                   color="0.5", alpha=0.22, lw=0, zorder=0)

# Dotted grey, as in Fig. 5: a fixed reference level the two final plateaus are compared
# against, not the band criterion.
ax.axhline(1e-2, color="0.5", ls=":", lw=1.0, zorder=1)
ax.set_xlabel(r"$U_{\max} - U_0$")
ax.set_ylabel(r"Generative Efficiency $\eta$")
secax = ax.secondary_xaxis("top", functions=(to_pct, from_pct))
secax.set_xlabel("Nested-Sampling Progress [%]", fontsize=ps.ANNOTATION_FONTSIZE,
                 labelpad=4)
# The secondary axis inherits the parent's log locator, which lands dozens of decade
# ticks on a 0-100 scale; fix them at the same decades Fig. 5 uses.
secax.minorticks_off()
secax.xaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
secax.tick_params(labelsize=ps.ANNOTATION_FONTSIZE_SMALL, direction="in", length=4, pad=2)
# Upper right: the curves occupy the lower left (low energy, recovered efficiency) and
# the upper left (high energy, high efficiency), leaving this corner free.
ax.legend(loc="upper right", handletextpad=0.2, fontsize=ps.ANNOTATION_FONTSIZE)

# b) Same conventions as the timing mosaic of Fig. 4 (new_plotter.ipynb).
ax = axes[1]
run = cm.run_dir("hex")
t = cm.read_timings(run) / 3600
training, generation = t[:, 2], t[:, 3]
pools = np.arange(1, len(t) + 1)
ax.bar(pools, training, width=0.75, color="C0", edgecolor="black", linewidth=0.25,
       label="Network Training")
ax.bar(pools, generation, bottom=training, width=0.75, color="C1", alpha=0.85,
       edgecolor="black", linewidth=0.25, label="Pool Generation")
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.set_xlabel("Pool Generated")
ax.set_ylabel("Time (Hours)")
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
