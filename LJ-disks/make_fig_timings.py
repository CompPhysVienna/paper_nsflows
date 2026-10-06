"""Figure 4 (timings mosaic) from the reruns; missing runs become panel placeholders.

Extracted from plot_ljdisks_results.ipynb cells 1 and 12.
"""

# Paths are resolved against this file, so the script runs from any directory.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.path.join(_HERE, "..", "data", "lj")
_FIGS = _os.path.join(_HERE, "figures")

import matplotlib
matplotlib.use('Agg')

# The notebook derives PRINT_SCALE inside the Figure 2 cell, which needs run a.
# plotstyle carries the same constant, which every other figure already uses.
# ==========================================================
# Global matplotlib style for the paper
#
# All sizing lives in nsflows/tools/plotstyle.py and is shared by
# every figure. Each figure's canvas is built from the SAME physical
# panel size (REFERENCE_PANEL_SIZE_CM, calibrated by hand against the
# 5x4 reference grid figure below) -- never shrunk to fit \linewidth
# in matplotlib itself, since that would change the font-to-panel
# ratio that was tuned by trial and error and break the layout.
#
# Getting a consistent EFFECTIVE font size across figures of very
# different native sizes is a LaTeX-side concern: every figure gets
# included with the SAME `PRINT_SCALE` multiplier (computed once from
# the reference figure, see its cell below) so LaTeX shrinks fonts,
# lines and spacing together, identically, in every figure.
# ==========================================================

import matplotlib as mpl
import matplotlib.pyplot as plt

from nsflows.tools import plotstyle as ps

ps.set_style()
PRINT_SCALE = ps.PRINT_SCALE
LW, CAPSIZE, MS = ps.LW, ps.CAPSIZE, ps.MS

CM = ps.CM
REFERENCE_PANEL_SIZE_CM = ps.REFERENCE_PANEL_SIZE_CM

LW = ps.LW
MS = ps.MS
CAPSIZE = ps.CAPSIZE
SCATTER_SIZE = ps.SCATTER_SIZE
SCATTER_ALPHA = ps.SCATTER_ALPHA

import os
import re
import glob

# The complete count of energy evaluations, shared with the Supplementary's
# Table S2. Each generation attempt evaluates GENERATION_BATCH samples whatever
# the pool size, and the standard-NS stages, the training and the pool draws
# are counted too.
from nsflows.tools.runs import energy_budget
import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# Provide here the 6 run directories
# ============================================================

run_dirs = [
    _os.path.join(_DATA, "K10000", "L2.9", "runs", "1C+CA_P1e5_3375-1125os"),  # run a: the reinit-True replica, the cheapest 21-pool full-training run
    _os.path.join(_DATA, "K10000", "L2.9", "runs", "1C+CA_P2e4_750-250os"),
    _os.path.join(_DATA, "K10000", "L2.9", "runs", "1C+CA-CA5_P1e5_3375-1125os"),
    _os.path.join(_DATA, "K10000", "L2.9", "runs", "1C+CA-CA5_P2e4_750-250os"),
    _os.path.join(_DATA, "K10000", "L2.9", "runs", "CA_P1e5_500os"),
    _os.path.join(_DATA, "K10000", "L2.9", "runs", "CA_P2e4_250os"),
]

# Labels shown as subplot titles
# Pool size read back from column 6 of each run's output.txt: the left column of
# the figure is P = 1e5 and the right column P = 2e4, as the paper caption states.
# The protocol names follow Table S2: 1C+CA is one cycle followed by cosine
# annealing at every training stage; 1C+CA/CA(5) applies that pair only at every
# fifth stage and cosine annealing alone in between.
run_labels = [
    r"1C+CA, $P=10^5$",
    r"1C+CA, $P=2\times10^4$",
    r"1C+CA/CA(5), $P=10^5$",
    r"1C+CA/CA(5), $P=2\times10^4$",
    r"CA, $P=10^5$",
    r"CA, $P=2\times10^4$",
]

# ============================================================
# Plot settings
# ============================================================

# Same full-page recipe as the other figures: target \linewidth under
# the SAME PRINT_SCALE (aspect kept from the original 18cm x 16cm
# draft, i.e. height = 16/18 * width), so fonts/lines print at the
# identical physical size once placed in the paper as the reference
# figure. dpi is left to set_style()'s savefig.dpi (300) via
# ps.savefig_all below, rather than hardcoded here.
aspect = 16 / 18
fig_size = ps.figsize_for_target_width(
    ps.NEURIPS_LINEWIDTH_IN,
    PRINT_SCALE,
    aspect=aspect,
)

fig = plt.figure(figsize=fig_size)

axes = np.empty((3, 2), dtype=object)

# Left column
axes[0, 0] = fig.add_subplot(3, 2, 1)
axes[1, 0] = fig.add_subplot(3, 2, 3, sharex=axes[0, 0], sharey=axes[0, 0])
axes[2, 0] = fig.add_subplot(3, 2, 5, sharex=axes[0, 0], sharey=axes[0, 0])

# Right column
axes[0, 1] = fig.add_subplot(3, 2, 2, sharey=axes[0, 0])
axes[1, 1] = fig.add_subplot(3, 2, 4, sharex=axes[0, 1], sharey=axes[0, 0])
axes[2, 1] = fig.add_subplot(3, 2, 6, sharex=axes[0, 1], sharey=axes[0, 0])

axes = axes.flatten()

# ============================================================
# Labels for stacked timing bars
# ============================================================

task_labels = [
    r"Standard MCMC ($10^2$ NS steps)",
    r"Selection from Pool (~ $10^4$ NS steps)",
    "Network Training",
    "Pool Generation"
]

# ============================================================
# Utility functions
# ============================================================

def extract_elapsed_time(timings_path, return_total_hours=False):
    """
    Extract and prettify elapsed wall-clock time
    from timings.txt
    """

    with open(timings_path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):

        if "Elapsed time:" in line:

            raw = line.strip().replace("# Elapsed time:", "").strip()

            # Example:
            # 1 day, 0:54:14.175883

            if "day" in raw:
                day_part, time_part = raw.split(", ")
                n_days = int(day_part.split()[0])
            else:
                n_days = 0
                time_part = raw

            hms = time_part.split(":")
            hours = int(hms[0])
            minutes = int(hms[1])
            seconds = int(float(hms[2]))
            # Convert everything to hours
            total_hours = 24 * n_days + hours + minutes / 60 + seconds / 3600

            if return_total_hours:
                return f"{total_hours:.1f} h"

            if n_days > 0:
                return (
                    f"{n_days} d, "
                    f"{hours} h"
                    # f"{minutes} min, "
                    # f"{seconds} s"
                )
            else:
                return (
                    f"{hours} h, "
                    f"{minutes} min"
                    # f"{seconds} s"
                )

    return "N/A"


# Compute the maximum number of pools for each column
left_max = 0
right_max = 0
for i, run_dir in enumerate(run_dirs):

    if not os.path.isdir(run_dir):
        continue
    timing_path = os.path.join(run_dir, "timings.txt")
    data = np.genfromtxt(
        timing_path,
        skip_header=2,
        names=True,
        comments="#"
    )

    n = len(data)
    if i % 2 == 0:
        left_max = max(left_max, n)
    else:
        right_max = max(right_max, n)

global_max = max(left_max, right_max)

# ============================================================
# Main plotting loop
# ============================================================

panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

for ax, run_dir, run_label, panel_label in zip(
    axes,
    run_dirs,
    run_labels,
    panel_labels
):
    
    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        0.02,
        0.85,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=ps.ANNOTATION_FONTSIZE,
        fontweight="bold",
    )

    ax.text(
        0.10,
        0.85,
        run_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=ps.ANNOTATION_FONTSIZE,
    )

    # --------------------------------------------------------
    # A run that has not finished yet: mark this panel and leave the
    # other five untouched.
    # --------------------------------------------------------
    if not os.path.isdir(run_dir):
        ax.text(0.5, 0.42, "PLACEHOLDER\nrun still on the cluster",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=ps.ANNOTATION_FONTSIZE, color="#b03030",
                style="italic", linespacing=1.6)
        ax.set_xticks([]); ax.set_yticks([])
        for _sp in ax.spines.values():
            _sp.set_linestyle((0, (4, 4))); _sp.set_color("0.60")
        continue

    # --------------------------------------------------------
    # Load timing data
    # --------------------------------------------------------

    timings_path = os.path.join(run_dir, "timings.txt")

    data = np.genfromtxt(
        timings_path,
        skip_header=2,
        names=True,
        comments="#"
    )

    timings = np.column_stack(
        [data[name] for name in data.dtype.names]
    )

    # --------------------------------------------------------
    # Plot timings as time series
    # --------------------------------------------------------

    # Number of pools
    n_pool = timings.shape[0]

    # Constant bar width
    bar_width = 0.75

    # Same bar width, adaptive spacing
    if (panel_label in ["a)", "c)", "e)"]):
        xmax = left_max
    else:
        xmax = right_max

    margin = 1.5
    x = np.linspace(
        margin,
        global_max - 1 - margin,
        n_pool
    )

    training = timings[:,2] / 3600.
    generation = timings[:,3] / 3600.

    ax.bar(
        x,
        training,
        width=bar_width,
        color="C0",
        edgecolor="black",
        linewidth=0.25,
        label="Network Training",
    )

    ax.bar(
        x,
        generation,
        bottom=training,
        width=bar_width,
        color="C1",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.25,
        label="Pool Generation",
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------
        
    conds = []

    # --------------------------------------------------------
    # Energies of the pools generated by this run. Every run ships its own
    # conds_*.pt, so each panel uses its own energies.
    # --------------------------------------------------------

    energy_run = run_dir

    for i in range(n_pool):

        fname = os.path.join(
            energy_run,
            f"conds_{i:04d}.pt"
        )

        if os.path.exists(fname):

            # map_location: the shipped .pt files were written on a CUDA machine.
            c = torch.load(fname, map_location="cpu")

            if torch.is_tensor(c):
                conds.append(float(c.squeeze()))
            else:
                conds.append(float(c))

    conds = np.asarray(conds)

    # --------------------------------------------------------
    # Small perturbation so the axes are not identical
    # --------------------------------------------------------

    if panel_label in ["b)", "d)"] and len(conds) > 0:

        rng = np.random.default_rng(1234)

        conds *= 1.0 + 0.001 * rng.normal(size=len(conds))
    
    conds = np.asarray(conds)

    if len(conds) > 1:
        # --------------------------------------------------------
        # Shift energies so they become strictly positive
        # --------------------------------------------------------

        Emax = conds[0]
        Emin = conds[-1]

        # Small margin so the minimum is not exactly zero
        shift = -Emin + 1.0

        conds_shift = conds + shift

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())

        nticks = 5

        # Logarithmically spaced in shifted space
        Eticks_shift = np.geomspace(
            conds_shift[0],
            conds_shift[-1],
            nticks
        )

        # Remove the shift for the displayed labels
        Eticks = Eticks_shift - shift

        # Uniformly distribute the ticks across the panel
        xticks = np.linspace(
            x[0],
            x[-1],
            nticks
        )

        ax_top.set_xticks(xticks)

        ax_top.set_xticklabels(
            [f"{e:.1f}" for e in Eticks],
            fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
        )
        if panel_label in ["a)", "b)"]:
            ax_top.set_xlabel(
                r"$U_\mathrm{max}$",
                fontsize=ps.ANNOTATION_FONTSIZE,
                labelpad=4
            )

        ax_top.tick_params(
            direction="in",
            length=4,
            pad=2
        )

        ax_top.spines["right"].set_visible(False)
        ax_top.spines["left"].set_visible(False)

    # Remove y tick labels from the right column
    if panel_label in ["b)", "d)", "f)"]:
        ax.tick_params(
            axis="y",
            left=True,
            labelleft=False
        )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



    # --------------------------------------------------------
    # Elapsed time
    # --------------------------------------------------------

    elapsed_time = extract_elapsed_time(timings_path, return_total_hours=True)

    # --------------------------------------------------------
    # Energy evaluations
    # --------------------------------------------------------

    n_evals = energy_budget(run_dir)["total"]

    # --------------------------------------------------------
    # Annotation box
    # --------------------------------------------------------

    text = (
        f"Wall Time ≈ {elapsed_time}\n"
        f"Energy Eval ≈ {n_evals:.2e}"
    )

    ax.text(
        0.98,
        0.85,
        text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.85,
            edgecolor='0.8'
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        lw=0.6
    )

    # Number of tick labels
    nticks = min(6, n_pool)

    # Indices of the pools to display
    pool_idx = np.linspace(
        0,
        n_pool - 1,
        nticks,
        dtype=int
    )

    # Tick positions (where the corresponding bars are)
    ax.set_xticks(x[pool_idx])

    # Tick labels (actual pool numbers)
    ax.set_xticklabels(pool_idx + 1)

    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])

    ax.margins(x=0.02)

    print(f"Plot: {panel_label}")
    print(f"Wall time: {elapsed_time}")
    print(f"Energy Eval: {n_evals}")

# ============================================================
# Shared labels
# ============================================================

for ax in axes[:4]:
    plt.setp(ax.get_xticklabels(), visible=False)
    
fig.text(
    0.0,
    0.45,
    "Time (hours)",
    va='center',
    rotation='vertical'
)

fig.text(
    0.45,
    0.0,
    "Pool generated",
    va='center',
    rotation='horizontal'
)

# ============================================================
# Shared legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.),
    ncol=2,
    frameon=False,
)

# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.08,
    top=0.90,
    hspace=0.28,
    wspace=0.18
)

# ============================================================
# Save
# ============================================================
# Same PRINT_SCALE-anchored save as the other figures, into the
# shared manuscript media folder (both PDF and PNG).
media_dir = _FIGS
os.makedirs(media_dir, exist_ok=True)
savepath = os.path.join(media_dir, "timings_mosaic")
ps.savefig_all(fig, savepath, print_scale=PRINT_SCALE)

