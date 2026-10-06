"""Figure 2: live sets, generated pools and resampled pools along the NS trajectory.

Built from run a (1C+CA, P=1e5), the run of Fig. 4a and of the first row of
Table S3. The four rows are its pools 2, 7, 12 and 20. The alignment reference
is the final live set of the standard-NS run.

Extracted from plot_ljdisks_results.ipynb cells 0, 1, 3, 5 and 7.
"""



# Paths are resolved against this file, so the script runs from any directory.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.path.join(_HERE, "..", "data", "lj")
_FIGS = _os.path.join(_HERE, "figures")

# The panels superimpose a subsample of each pool, and the training-set augmentation
# that draws it is random. Seed it, so the figure is reproducible rather than merely
# statistically the same from one run to the next.
import numpy as _np
import torch as _torch
_np.random.seed(0)
_torch.manual_seed(0)

import matplotlib
matplotlib.use('Agg')


import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from nsflows.systems.lennard_jones import lennard_jones
from nsflows.systems.uniforms import box_uniform


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

CM = ps.CM
REFERENCE_PANEL_SIZE_CM = ps.REFERENCE_PANEL_SIZE_CM

LW = ps.LW
MS = ps.MS
CAPSIZE = ps.CAPSIZE
SCATTER_SIZE = ps.SCATTER_SIZE
SCATTER_ALPHA = ps.SCATTER_ALPHA

n_particles = 8
dimensions = 2
box_length = 2.9
cutin = 0.8
rho = n_particles/(box_length)**(dimensions)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
box_uniform_2D = box_uniform(n_particles=n_particles, dimensions=dimensions, device=device, box_length=box_length)
LJ_disks = lennard_jones(n_particles=n_particles, dimensions=dimensions, rho=rho, device=device, cutin=cutin, lrc=True)

# Helpers for comparing configurations up to the symmetries of the system: a pi/2
# rotation of the box and a relabelling of the particles. Shared with the other
# notebooks through nsflows.tools.util.
from nsflows.tools.util import (
    remove_outermost_particle,
    dist_matrix,
    hungarian_algorithm,
    align_config,
)


from nsflows.tools.observables import rdf
# counts = list of energies / configurations you want to display as rows
# example:
output_dir = _os.path.join(_DATA, "K10000", "L2.9", "runs", "1C+CA_P1e5_3375-1125os")   # run a
counts = [2, 7, 12, 20]   # the published rows of run a

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

# ------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------
ncols = 5
nrows = len(counts)

# Reference figure: native size built purely from the calibrated
# panel size (square panels) -- this is the layout that was hand-tuned
# to look right, so its font-to-panel ratio must stay untouched here.
# It's meant to span the full \linewidth once placed in the paper.
# PRINT_SCALE (computed after saving, from the ACTUAL post-crop width)
# is what every other figure in the notebook should reuse in its own
# ps.savefig(..., print_scale=PRINT_SCALE) call, to get the same
# effective font size on the page.
panel_size_cm = REFERENCE_PANEL_SIZE_CM
fig_w, fig_h = ps.panel_figsize(ncols=ncols, nrows=nrows, panel_size_cm=panel_size_cm)

fig = plt.figure(
    figsize=(fig_w, fig_h),
    constrained_layout=True
)

gs = GridSpec(
    nrows,
    ncols,
    figure=fig,
    width_ratios=[1, 1, 1, 1, 1],
    height_ratios=[1, 1, 1, 1],
)

axes = np.empty((nrows, ncols), dtype=object)

for row in range(nrows):

    if row == 0:

        # Configuration-space plots
        axes[row, 0] = fig.add_subplot(gs[row, 0])
        axes[row, 1] = fig.add_subplot(
            gs[row, 1],
            sharex=axes[0, 0],
            sharey=axes[0, 0]
        )
        axes[row, 2] = fig.add_subplot(
            gs[row, 2],
            sharex=axes[0, 0],
            sharey=axes[0, 0]
        )

        # RDF plot
        axes[row, 3] = fig.add_subplot(gs[row, 3])
        
        # Energy plot
        axes[row, 4] = fig.add_subplot(gs[row, 4])
    else:

        axes[row, 0] = fig.add_subplot(
            gs[row, 0],
            sharex=axes[0, 0],
            sharey=axes[0, 0]
        )

        axes[row, 1] = fig.add_subplot(
            gs[row, 1],
            sharex=axes[0, 0],
            sharey=axes[0, 0]
        )

        axes[row, 2] = fig.add_subplot(
            gs[row, 2],
            sharex=axes[0, 0],
            sharey=axes[0, 0]
        )

        # RDF shares x only
        axes[row, 3] = fig.add_subplot(
            gs[row, 3],
            sharex=axes[0, 3]
        )

        axes[row, 4] = fig.add_subplot(
            gs[row, 4]
        )

# Handle single-row case
if nrows == 1:
    axes = axes[np.newaxis, :]

# ------------------------------------------------------------
# Helper plotting function
# ------------------------------------------------------------

def plot_config(
    ax,
    data,
    color,
    align_to_reference=None,
    stride=1,
):
    """
    Plot particle configurations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    data : torch.Tensor
        Shape:
            (nsamples, nparticles, ndim)

    color : str
        Matplotlib color.

    align_to_reference : torch.Tensor or None
        Shape:
            (1, nparticles, ndim)

        If provided, configurations are aligned via:
            align_config(data, align_to_reference)

    stride : int
        Subsample configurations before plotting.
    """

    # --------------------------------------------------------
    # Optional subsampling
    # --------------------------------------------------------
    data = data[::stride]

    # --------------------------------------------------------
    # Optional alignment
    # --------------------------------------------------------
    if align_to_reference is not None:

        data = align_config(
            data,
            align_to_reference,
            n_particles,
            dimensions,
            box_length
        )

    # --------------------------------------------------------
    # Move to CPU only once
    # --------------------------------------------------------
    data = data.detach().cpu().numpy()

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.set_aspect("equal")

    ax.scatter(
        data[:, :, 0],
        data[:, :, 1],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        zorder=10,
        color=color,
    )

    ax.set_xlim(-box_length / 2, box_length / 2)
    ax.set_ylim(-box_length / 2, box_length / 2)

    ax.set_xticks([-box_length / 2, 0, box_length / 2])
    ax.set_yticks([-box_length / 2, 0, box_length / 2])

    ax.set_xticklabels([r"$-L/2$", r"$0$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$0$", r"$L/2$"])

# ------------------------------------------------------------
# Column titles
# ------------------------------------------------------------

col_titles = [
    "Live Set",
    "Generated",
    "Resampled",
    r"$g(r)$",
    r"$P(U)$",
]

for j in range(ncols):
    axes[0, j].set_title(col_titles[j])

# ------------------------------------------------------------
# Main loop over rows
# ------------------------------------------------------------

prior_samples = box_uniform_2D.sample(100000)

for row, count in enumerate(counts):

    dataset_filepath = os.path.join(
        output_dir,
        f"dataset_{count:04d}.pt"
    )

    pool_biased_filepath = os.path.join(
        output_dir,
        f"pool_biased_{count:04d}.pt"
    )

    pool_filepath = os.path.join(
        output_dir,
        f"pool_{count:04d}.pt"
    )

    conds_filepath = os.path.join(
        output_dir,
        f"conds_{count:04d}.pt"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if (
        not os.path.exists(dataset_filepath)
        or not os.path.exists(pool_biased_filepath)
        or not os.path.exists(pool_filepath)
        or not os.path.exists(conds_filepath)
    ):
        print(f"Skipping count={count}: missing file.")
        continue

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    # map_location: the shipped .pt files were written on a CUDA machine, so loading
    # them on a CPU- or MPS-only machine needs an explicit target. Mapping to the
    # device selected above also keeps them on the same device as the system objects.
    dataset = torch.load(dataset_filepath, map_location=device)
    pool_biased = torch.load(pool_biased_filepath, map_location=device)
    pool = torch.load(pool_filepath, map_location=device)
    cond = torch.load(conds_filepath, map_location=device)

    r_dataset, g_dataset = rdf(
        dataset,
        n_particles=LJ_disks.n_particles,
        dimensions=LJ_disks.dimensions,
        box_length=LJ_disks.box_length
    )

    r_pool_biased, g_pool_biased = rdf(
        pool_biased,
        n_particles=LJ_disks.n_particles,
        dimensions=LJ_disks.dimensions,
        box_length=LJ_disks.box_length
    )

    r_pool, g_pool = rdf(
        pool,
        n_particles=LJ_disks.n_particles,
        dimensions=LJ_disks.dimensions,
        box_length=LJ_disks.box_length
    )

    dataset_energy = (
        LJ_disks.energy(dataset)
        .detach()
        .cpu()
        .numpy()
    )

    prior_energy = (
        LJ_disks.energy(prior_samples)
        .detach()
        .cpu()
        .numpy()
    )

    pool_biased_energy = (
        LJ_disks.energy(pool_biased)
        .detach()
        .cpu()
        .numpy()
    )

    pool_energy = (
        LJ_disks.energy(pool)
        .detach()
        .cpu()
        .numpy()
    )

    dataset = dataset.view(
        -1,
        LJ_disks.n_particles,
        LJ_disks.dimensions
    )

    pool_biased = pool_biased.view(
        -1,
        LJ_disks.n_particles,
        LJ_disks.dimensions
    )

    pool = pool.view(
        -1,
        LJ_disks.n_particles,
        LJ_disks.dimensions
    )

    # ------------------------------------------------------------
    # Load reference configuration for alignment
    # ------------------------------------------------------------
    reference_path = _os.path.join(_DATA, "K10000", "L2.9", "samples_ref.pt")

    reference_config = torch.load(reference_path, map_location=device)

    reference_config = reference_config.view(
        -1,
        LJ_disks.n_particles,
        LJ_disks.dimensions
    )

    reference_config = remove_outermost_particle(reference_config)

    # Use first configuration as alignment reference
    ref_config = reference_config[[0]].clone()

    # --------------------------------------------------------
    # Plot row
    # --------------------------------------------------------
    plot_config(
        axes[row, 0], 
        dataset, 
        "C0", 
        align_to_reference=ref_config,
        )
    # The pool is larger than the live set, so it is thinned to the same number
    # of configurations before plotting: at equal marker size and alpha, that is
    # what makes the three panels of a row comparable by eye. The published
    # figure used a fixed stride of 10, which is this ratio for its pool of 1e5
    # against a live set of 1e4; hard-coding it makes every other pool size come
    # out too faint.
    pool_stride = max(1, int(round(len(pool) / len(dataset))))
    biased_stride = max(1, int(round(len(pool_biased) / len(dataset))))

    plot_config(
        axes[row, 1], 
        pool_biased, 
        "C2", 
        align_to_reference=ref_config,
        stride=biased_stride,
        )
    plot_config(
        axes[row, 2], 
        pool, 
        "C3", 
        align_to_reference=ref_config,
        stride=pool_stride
        )
    rdf_ax = axes[row, 3]
    rdf_ax.set_aspect("auto")
    rdf_ax.set_box_aspect(1)
    
    rdf_ax.plot(
        r_dataset,
        g_dataset,
        color="C0",
        lw=LW,
        label="Live Set"
    )

    rdf_ax.plot(
        r_pool_biased,
        g_pool_biased,
        color="C2",
        lw=LW,
        ls="--",
        label="Generated Pool"
    )

    rdf_ax.plot(
        r_pool,
        g_pool,
        color="C3",
        lw=LW,
        ls=":",
        label="Resampled Pool"
    )

    rdf_ax.axhline(
        y=1.0,
        color="k",
        linestyle="-.",
        linewidth=LW,
        alpha=0.6
    )

    rdf_ax.spines["right"].set_visible(False)
    rdf_ax.spines["top"].set_visible(False)

    rdf_ax.set_xlim(0, torch.min(LJ_disks.box_length / 2).item())

    # The generated pool before resampling piles up at r -> 0: configurations
    # with overlapping disks, which the rejection resampling then removes. In
    # the lowest row that spike is four times the height of the crystalline
    # peaks and flattens them against the axis. The vertical range is therefore
    # set from the structure outside the core, and the spike runs off the top.
    # In the upper rows there is nothing at small r, so this changes nothing.
    R_CORE = 0.5
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    peak = max(
        float(_np(g)[_np(r) >= R_CORE].max())
        for r, g in ((r_dataset, g_dataset),
                     (r_pool_biased, g_pool_biased),
                     (r_pool, g_pool))
    )
    rdf_ax.set_ylim(0, 1.15 * peak)

    if row == nrows - 1:
        rdf_ax.set_xlabel(r"$r$")

    energy_ax = axes[row, 4]
    energy_ax.set_aspect("auto")
    energy_ax.set_box_aspect(1)

    # common bins

    energy_value_max = (
        cond.item()
        if torch.numel(cond) == 1
        else cond
    )

    energy_value_min = min(
        dataset_energy.min(),
        pool_biased_energy.min(),
        pool_energy.min()
    )

    dataset_std = np.std(dataset_energy)
    # The generated pool is the only distribution here not bounded by U_max, and
    # its upper tail runs orders of magnitude past the live set: in the lowest
    # row the live set has sigma = 0.02 while 5% of the generated pool sits above
    # +484. No linear axis shows both, so this reaches a few times further up
    # than the published 3 sigma to expose more of that tail, and no further --
    # chasing the whole of it flattens the live set into a spike.
    nsigma_up = 10
    nsigma_dw = 1

    emin = energy_value_min - nsigma_dw * dataset_std
    emax = energy_value_max + nsigma_up * dataset_std

    bins = np.linspace(
        emin,
        emax,
        50
    )

    energy_ax.set_xlim(
        emin,
        emax
    )

    hist, edges = np.histogram(
        dataset_energy,
        bins=bins,
        density=True
    )

    energy_ax.stairs(
        hist,
        edges,
        color="C0",
        lw=LW,
        label="Live Set"
    )

    if row==0:
     
        hist, edges = np.histogram(
            prior_energy,
            bins=bins,
            density=True
        )

        centers = 0.5 * (edges[:-1] + edges[1:])


        energy_ax.plot(
            centers,
            hist,
            drawstyle="steps-mid",
            lw=LW,
            linestyle="-.",
            color="C1",
            label="Prior"
        )

    hist, edges = np.histogram(
        pool_biased_energy,
        bins=bins,
        density=True
    )

    centers = 0.5 * (edges[:-1] + edges[1:])


    energy_ax.plot(
        centers,
        hist,
        drawstyle="steps-mid",
        lw=LW,
        linestyle="--",
        color="C2",
        label="Gen. Pool"
    )


    hist, edges = np.histogram(
        pool_energy,
        bins=bins,
        density=True
    )

    energy_ax.stairs(
        hist,
        edges,
        lw=LW,
        linestyle=":",
        color="C3",
        label="Res. Pool"
    )

    if row == nrows - 1:
        energy_ax.set_xlabel(r"$U$")
        
    # Optional row label = energy
    energy_value = cond.item() if torch.numel(cond) == 1 else cond

    axes[row, 0].set_ylabel(
        rf"$U_{{\mathrm{{max}}}} = {energy_value:.2f}$",
    )

# Combined legend for the g(r) and P(U) panels (columns 4 and 5),
# printed above the whole figure instead of inside either panel --
# same pattern as the fig.legend() in the energy-trace figure. Pull
# handles from both axes since between them they cover all 4 series
# (the g(r) panel has the fuller "Generated/Resampled Pool" labels;
# only the P(U) panel plots "Prior").
handles_rdf, labels_rdf = axes[0, 3].get_legend_handles_labels()
handles_pu, labels_pu = axes[0, 4].get_legend_handles_labels()

rdf_by_label = dict(zip(labels_rdf, handles_rdf))
pu_by_label = dict(zip(labels_pu, handles_pu))

legend_entries = [
    ("Live Set", rdf_by_label["Live Set"]),
    ("Generated Pool", rdf_by_label["Generated Pool"]),
    ("Resampled Pool", rdf_by_label["Resampled Pool"]),
    ("Prior", pu_by_label["Prior"]),
]

fig.legend(
    [handle for _, handle in legend_entries],
    [name for name, _ in legend_entries],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(legend_entries),
    frameon=False,
)

axes[0, 3].set_ylim(None, 2.5)
axes[0, 4].set_ylim(None, 0.01)


for row in range(nrows - 1):

    for col in range(ncols):
        if col != 4:
            plt.setp(
                axes[row, col].get_xticklabels(),
                visible=False
            )

for row in range(nrows):

    axes[row, 3].set_xticks(
        [0, torch.min(LJ_disks.box_length / 2).item()]
    )

    # Hide y tick labels on columns 2 and 3
    plt.setp(
        axes[row, 1].get_yticklabels(),
        visible=False
    )

    plt.setp(
        axes[row, 2].get_yticklabels(),
        visible=False
    )

axes[nrows - 1, 3].set_xticklabels(
    [r"$0$", r"$L/2$"]
)
axes[0, 3].set_yticks([0, 1, 2])
axes[0, 3].yaxis.set_major_formatter(
    FormatStrFormatter('%d')
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
# Save both PDF (vector, exact size -- used to anchor PRINT_SCALE)
# and PNG (fast to open, small file -- this figure's scatter clouds
# are heavy as vector points) from the same canvas, into a dedicated
# media folder for the manuscript.
media_dir = _FIGS
os.makedirs(media_dir, exist_ok=True)
savepath = os.path.join(media_dir, "all_energies_grid_rdf_U")

# Save first, then derive PRINT_SCALE from the ACTUAL post-crop width
# (bbox_inches="tight" crops a different amount per figure, so the
# nominal fig_w above is only an estimate).
w_in, h_in = ps.savefig_all(fig, savepath)
PRINT_SCALE = ps.NEURIPS_LINEWIDTH_IN / w_in
print(
    f"PRINT_SCALE = {PRINT_SCALE:.4f} "
    f"-- reuse this exact value in every other figure's "
    f"ps.savefig_all(..., print_scale=PRINT_SCALE) call"
)

plt.close()