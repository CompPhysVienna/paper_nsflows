"""
Fig. S3: live sets, resampled pools, radial distribution functions and energy
distributions along the NS trajectory for N = 9 disks in a 2:sqrt(3) cell (rho = 0.95).
Counterpart of Fig. 2 of the main text; the flow-generated pool before resampling was
not stored for this run, so that column is absent.

Needs the rectangular-cell version of nsflows (branch ``anisotropic_cell``), e.g.
    git archive origin/anisotropic_cell nsflows | tar -x -C /some/dir
    PYTHONPATH=/some/dir python make_fig_hex_configs.py
"""

import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from nsflows.tools.observables import rdf
from nsflows.systems.lennard_jones import lennard_jones

import common as cm
from common import ps

if "aspect_ratio" not in inspect.signature(lennard_jones.__init__).parameters:
    raise RuntimeError("This script needs nsflows from the anisotropic_cell branch (see docstring).")

N, RHO, ASPECT = 9, 0.95, 2 / np.sqrt(3)
COUNTS = [5, 17, 29, 44]

ps.set_style()

system = lennard_jones(n_particles=N, dimensions=2, rho=RHO, device=torch.device("cpu"),
                       cutin=0.8, lrc=True, aspect_ratio=ASPECT)
box = system.box_length
Lx, Ly = box.tolist()
run = cm.run_dir("hex")

# Symmetry check: the rectangle's point group D2 = {E, C2, sigma_x, sigma_y} should leave
# the low-energy crystal invariant, so no alignment is needed in this cell.
final = cm.internal_coordinates(cm.load(run / "samples_000000500000.pt"), N, box)
ref = final[:1]
d2 = {k: cm.D4[k] for k in [r"$E$", r"$C_2$", r"$\sigma_x$", r"$\sigma_y$"]}
print("Matching cost of the final reference configuration under D2:",
      {k: round(float(cm.matching_cost(op(ref), ref, box)[0]), 4) for k, op in d2.items()})
costs = np.stack([cm.matching_cost(op(final[:2000]), ref, box) for op in d2.values()])
print("Median matching cost of the final live set to the reference:", np.median(costs.min(0)))

# Printed at \linewidth under the paper's shared PRINT_SCALE.
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=1.1)
fig, axes = plt.subplots(len(COUNTS), 4, figsize=(fig_w, fig_h), constrained_layout=True)

for row, count in enumerate(COUNTS):
    live = cm.internal_coordinates(cm.load(run / f"dataset_{count:04d}.pt"), N, box)
    pool = cm.internal_coordinates(cm.load(run / f"pool_{count:04d}.pt"), N, box)
    u_bound = cm.load(run / f"conds_{count:04d}.pt").item()

    for ax, x, color in [(axes[row, 0], live, "C0"), (axes[row, 1], pool, "C3")]:
        x = x.numpy()
        ax.scatter(x[:, :, 0], x[:, :, 1], s=ps.SCATTER_SIZE, alpha=ps.SCATTER_ALPHA, color=color,
                   rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-Lx / 2, Lx / 2)
        ax.set_ylim(-Ly / 2, Ly / 2)
        ax.set_xticks([-Lx / 2, 0, Lx / 2], [r"$-L_x/2$", r"$0$", r"$L_x/2$"] if row == len(COUNTS) - 1 else [])
        ax.set_yticks([-Ly / 2, 0, Ly / 2], [r"$-L_y/2$", r"$0$", r"$L_y/2$"] if ax is axes[row, 0] else [])
    axes[row, 0].set_ylabel(rf"$U_{{\max}} = {u_bound:.2f}$")

    ax = axes[row, 2]
    for x, color, ls, label in [(live, "C0", "-", "Live set"), (pool, "C3", ":", "Resampled pool")]:
        r, g = rdf(x.reshape(-1, 2 * N), n_particles=N, dimensions=2, box_length=box)
        ax.plot(np.asarray(r), np.asarray(g).squeeze(), color=color, ls=ls, label=label)
    ax.axhline(1, color="k", ls="-.", lw=0.8)
    ax.set_xlim(0, min(Lx, Ly) / 2)
    if row == len(COUNTS) - 1:
        ax.set_xlabel(r"$r$")

    ax = axes[row, 3]
    u_live = system.energy(live.reshape(-1, 2 * N)).squeeze().numpy()
    u_pool = system.energy(pool.reshape(-1, 2 * N)).squeeze().numpy()
    bins = np.linspace(min(u_live.min(), u_pool.min()), u_bound, 40)
    ax.hist(u_live, bins=bins, density=True, histtype="step", color="C0")
    ax.hist(u_pool, bins=bins, density=True, histtype="step", color="C3", ls=":")
    ax.axvline(u_bound, color="k", lw=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    if row == len(COUNTS) - 1:
        ax.set_xlabel(r"$U$")

for ax, title in zip(axes[0], ["Live Set", "Resampled", r"$g(r)$", r"$P(U)$"]):
    ax.set_title(title)
handles, labels = axes[0, 2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

ps.savefig(fig, cm.FIG_DIR / "fig_hex_configs.png", print_scale=ps.PRINT_SCALE)
