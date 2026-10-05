"""Figure 6 (phase diagram + rho=0.73 snapshot) from the rerun L3.3 run.

Extracted from plot_ljdisks_results.ipynb cells 0, 1, 14, 16, 17 and 19. The
reference live set is the final live set of the new run g, not the unarchived
configuration the repository shipped as samples_ref.pt.
"""



# Paths are resolved against this file, so the script runs from any directory.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DATA = _os.path.join(_HERE, "..", "data", "lj")
_FIGS = _os.path.join(_HERE, "figures")

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

PRINT_SCALE = ps.PRINT_SCALE
LW, CAPSIZE, MS = ps.LW, ps.CAPSIZE, ps.MS


import numpy as np
from scipy.interpolate import make_interp_spline

# ============================================================
# Approximate phase-boundary reconstruction
# ============================================================

# ------------------------------------------------------------
# Fluid-Hexatic coexistence boundary
# ------------------------------------------------------------

rho_fh = np.array([
    0.05,
    0.10,
    0.18,
    0.28,
    0.35,
    0.50,
    0.65,
    0.77
])

T_fh = np.array([
    0.42,
    0.445,
    0.475,
    0.497,
    0.500,
    0.480,
    0.445,
    0.418
])

rho_fh_smooth = np.linspace(rho_fh.min(), rho_fh.max(), 300)

spline_fh = make_interp_spline(rho_fh, T_fh, k=3)
T_fh_smooth = spline_fh(rho_fh_smooth)

# ------------------------------------------------------------
# Fluid-Solid coexistence boundary
# ------------------------------------------------------------

rho_fs = np.array([
    0.77,
    0.78,
    0.79,
    0.80,
    0.81,
    0.82,
    0.83,
    0.835,
    0.84
])

T_fs = np.array([
    0.42,
    0.44,
    0.46,
    0.48,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80
])

# ------------------------------------------------------------
# Hexatic-Solid coexistence boundary
# ------------------------------------------------------------

rho_hs = np.array([
    0.83,
    0.83,
    0.835,
    0.84,
    0.845,
    0.85,
    0.855,
    0.86,
    0.865,
    0.87
])

T_hs = np.array([
    0.38,
    0.42,
    0.45,
    0.48,
    0.50,
    0.60,
    0.70,
    0.75,
    0.78,
    0.80
])

input_dir = _os.path.join(_DATA, "K10000", "L3.3", "runs", "CA_P2e4_250os")


n_particles = 8
dimensions = 2
box_length = 3.3
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

def load_config(count, as_numpy=True):

    # load configurations
    config = torch.load(
        os.path.join(input_dir, "samples_ref.pt")
    ).view(-1, n_particles, dimensions)

    if as_numpy:
        config = config.cpu().numpy()

    umax = torch.load(os.path.join(input_dir, "U_max_ref.pt"))

    return config, umax.item()

count = 500000
config, umax = load_config(count, as_numpy=False)  # shape (B, N, 2)

import os
import matplotlib.pyplot as plt

# ============================================================
# Figure: phase diagram (left) + sampled configuration snapshot
# at L=3.3 (right), as one 1x2 figure -- same pattern as the
# energy-trace figure's ax[0]/ax[1] layout.
#
# Same full-page recipe as the other figures: canvas targets
# \linewidth under the SAME PRINT_SCALE. Both panels are square
# (aspect=1), matching the "square panels" convention used
# throughout the rest of the paper's multi-panel figures -- so the
# whole canvas gets aspect=0.5 (2 panels side by side, each half
# the total width, height = width/2 = one panel's width).
#
# Also dropped, in favor of the shared ps.set_style() rcParams
# (to match every other figure in the notebook): the explicit
# fontsize=16/22 text/label overrides, the thicker lw=1.8-2.2 line
# widths (-> LW), the manual spine linewidth=1.2 (-> AXES_LINEWIDTH
# via rcParams), and the inward/length-6 tick_params + minorticks_on
# (-> the shared outward/length-3 ticks with no minor ticks).
# ============================================================

aspect = 0.5
fig_w, fig_h = ps.figsize_for_target_width(
    ps.NEURIPS_LINEWIDTH_IN,
    PRINT_SCALE,
    aspect=aspect,
)

fig, ax = plt.subplots(
    1, 2,
    figsize=(fig_w, fig_h),
    constrained_layout=True,
    gridspec_kw={"wspace": 0.05},
)

# ------------------------------------------------------------
# Left panel: phase diagram
# ------------------------------------------------------------

# Fluid-Hexatic line
ax[0].plot(
    rho_fh_smooth,
    T_fh_smooth,
    color="black",
    lw=LW,
    zorder=3
)

# Fluid-Solid line
ax[0].plot(
    rho_fs,
    T_fs,
    color="black",
    lw=LW,
    zorder=5
)

# Hexatic-Solid line
ax[0].plot(
    rho_hs,
    T_hs,
    color="black",
    lw=LW,
    zorder=5
)

# ------------------------------------------------------------
# Vertical arrows crossing the phase diagram
# ------------------------------------------------------------

arrow_style = dict(
    arrowstyle="-|>",
    color="black",
    lw=LW,
    mutation_scale=20   # increase arrow head size
)

# rho = 0.73
ax[0].annotate(
    "",
    xy=(0.73, 0.405),   # arrow tip (bottom)
    xytext=(0.73, 0.79),  # start point (top)
    arrowprops=arrow_style,
    zorder=10
)

# rho = 0.95
ax[0].annotate(
    "",
    xy=(0.95, 0.405),
    xytext=(0.95, 0.79),
    arrowprops=arrow_style,
    zorder=10
)

# ------------------------------------------------------------
# Phase labels
# ------------------------------------------------------------

ax[0].text(
    0.33,
    0.61,
    "Fluid",
)

ax[0].text(
    0.90,
    0.61,
    "Solid",
    rotation=90,
    ha="center"
)

# Region below coexistence dome
ax[0].text(
    0.38,
    0.435,
    "Coexistence",
    ha="center"
)

# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax[0].set_xlim(0.05, 1.02)
ax[0].set_ylim(0.40, 0.80)

ax[0].set_xlabel(r"$\rho$")
ax[0].set_ylabel(r"$T$")

ax[0].set_box_aspect(1)

# ------------------------------------------------------------
# Right panel: sampled configuration snapshot
# ------------------------------------------------------------

i = 10
conf_idx = 1000 + i * 500
conf_idx = count
conf_raw = config.cpu().numpy().reshape(-1, n_particles, dimensions)
conf, umax = load_config(conf_idx, as_numpy=True)  # shape (B, N, 2)

ax[1].scatter(
    conf[:, :, 0],
    conf[:, :, 1],
    s=SCATTER_SIZE*2,
    alpha=SCATTER_ALPHA,
    zorder=10,
)

ax[1].set_xticks([-box_length / 2, 0, box_length / 2])
ax[1].set_xticklabels([r"$-L/2$", r"$0$", r"$L/2$"])
ax[1].set_yticks([-box_length / 2, 0, box_length / 2])
ax[1].set_yticklabels([r"$-L/2$", r"$0$", r"$L/2$"])

ax[1].set_xlim(-box_length / 2, box_length / 2)
ax[1].set_ylim(-box_length / 2, box_length / 2)

ax[1].set_aspect('equal', adjustable='box')

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
# Same PRINT_SCALE-anchored save as the other figures, into the
# shared manuscript media folder (both PDF and PNG).
media_dir = _FIGS
os.makedirs(media_dir, exist_ok=True)
savepath = os.path.join(media_dir, "phase_diagram_snapshot")
ps.savefig_all(fig, savepath, print_scale=PRINT_SCALE)
