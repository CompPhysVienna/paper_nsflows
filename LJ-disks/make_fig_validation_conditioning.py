"""Validation of flow-based NS, and the effect of conditioning, as one figure.

a) NS energy trace, standard vs flow-based, with the log-scale inset
b) distribution of the sampled energy bounds
c) fraction of generated samples below the training bound, per architecture
d) relative effective sample size of the same three architectures

Panels a and b share the energy axis. Panels c and d split what used to be one
twin-axis panel: two measures on different scales get one axis each, both anchored
at zero so that bar length means what it looks like. They share the x axis.

The three architectures are told apart by colour (C2/C4/C5). That trio was checked
for colour-vision deficiency rather than chosen by eye: the obvious C2/C3/C4 fails,
green against red separating by only dE 7 in OKLab under deuteranopia. C0 and C1 are
left alone so that they keep meaning "standard" and "flow-based" across the figure.

Run from anywhere:
    <repo>/conda_envs/paper_nsflows/bin/python LJ-disks/make_fig_validation_conditioning.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from nsflows.tools import plotstyle as ps

ps.set_style()
PRINT_SCALE = ps.PRINT_SCALE

# Paths are resolved against this file, so the script runs from any directory.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "lj", "K10000", "L2.9")
FLOW_RUN = os.path.join(DATA, "runs", "CA_P2e4_250os")      # the run of Fig. 5f
COND = os.path.join(DATA, "conditioning_efficiency")
OUT = os.path.join(HERE, "figures", "fig_validation_conditioning")

# Energy of the relaxed minimum-energy structure, from the LBFGS refinement in the
# "Reference minimum energy" section of plot_ljdisks_results.ipynb. Recomputing it
# here would mean repeating the whole alignment chain for one number.
U0 = -14.944960594177246

# Colours: C0/C1 for the two samplers in a and b, C2/C4/C5 for the three
# architectures in c and d (see the module docstring on the CVD check).
C_STD, C_FLOW = "C0", "C1"
C_ARCH = ["C2", "C4", "C5"]

# ---------------------------------------------------------------- panels a, b
it_nf, e_nf = np.loadtxt(os.path.join(FLOW_RUN, "output.txt.gz"),
                         usecols=(0, 3), unpack=True)
it_std, e_std = np.loadtxt(os.path.join(DATA, "reference_from_std_ns.txt.gz"),
                           usecols=(0, 3), unpack=True)
e_nf, e_std = e_nf - U0, e_std - U0

vals_nf, bins_nf = np.histogram(e_nf, bins=200)
vals_std, bins_std = np.histogram(e_std, bins=200)
c_nf = 0.5 * (bins_nf[1:] + bins_nf[:-1])
c_std = 0.5 * (bins_std[1:] + bins_std[:-1])

# ---------------------------------------------------------------- panel c
N_REP, N_LS = 10, 5
SERIES = [                       # subdirectory, legend label
    ("training_window_10K", "Unconditioned"),
    ("training_window_10K_collated_dataset", "Unconditioned 3 Live Sets"),
    ("conditioning_window_10K", "Conditioned 3 Live Sets"),
]


def load(sub, name):
    return np.loadtxt(os.path.join(COND, sub, f"{name}.txt"), usecols=(2, 3), unpack=True)


# ---------------------------------------------------------------- figure
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, PRINT_SCALE, aspect=0.40)
fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
axa = fig.add_subplot(gs[:, 0])
axb = fig.add_subplot(gs[:, 1], sharey=axa)   # a and b are the same energy axis
axc = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[1, 2], sharex=axc)

# --- a) trace -----------------------------------------------------
for x, y, color, ls, lab in ((it_std / 1e5, e_std, C_STD, "-", "Standard Nested Sampling"),
                             (it_nf / 1e5, e_nf, C_FLOW, ":", "Flow-Based Nested Sampling")):
    axa.plot(x, y, color=color, ls=ls, label=lab)
axa.set_ylabel(r"$U_{\max}-U_0$")
axa.set_xlabel(r"Iteration ($\times10^5$)")
axa.set_xticks([0, 5])

ins = inset_axes(axa, width="50%", height="50%", loc="upper right", borderpad=0.6)
ins.plot(it_std / 1e5, e_std, lw=1.0, color=C_STD)
ins.plot(it_nf / 1e5, e_nf, lw=1.0, color=C_FLOW, ls=":")
ins.set_yscale("log")
ins.set_xticks([0, 5])
ins.set_yticks([1e-1, 1e1, 1e3])
ins.tick_params(labelsize=ps.ANNOTATION_FONTSIZE)

# --- b) distribution ----------------------------------------------
axb.plot(vals_std, c_std, color=C_STD, label="Standard Nested Sampling")
axb.plot(vals_nf, c_nf, color=C_FLOW, ls=":", label="Flow-Based Nested Sampling")
axb.set_xlabel("Samples")
axb.set_xscale("log")
axb.tick_params(labelleft=False)

# --- c, d) conditioning --------------------------------------------
x = np.arange(N_LS)
width = 0.27
for k, (sub, lab) in enumerate(SERIES):
    m_g, s_g = load(sub, "eff_generation")
    m_r, s_r = load(sub, "RESS")
    off = (k - 1) * width
    for ax, m, e in ((axc, m_g, s_g), (axd, m_r, s_r)):
        ax.bar(x + off, m, width * 0.92, yerr=e * np.sqrt(N_REP),
               color=C_ARCH[k],
               error_kw=dict(lw=0.8, capsize=1.5, ecolor="0.25"))

axc.set_ylabel("Generated Samples\n" r"with $U(x)<U_{\max}^{\mathrm{train}}$",
               fontsize=ps.ANNOTATION_FONTSIZE)
axd.set_ylabel("RESS", fontsize=ps.ANNOTATION_FONTSIZE)
axc.tick_params(labelbottom=False)
axd.set_xlabel(r"$U_{\max}$")
axd.set_xticks(x, [r"$513$", r"$32.5$", r"$-7.99$", r"$-13.5$", r"$-14.6$"],
               fontsize=ps.ANNOTATION_FONTSIZE)
axc.set_ylim(0, 0.52)
axd.set_ylim(0, 0.21)
for ax in (axc, axd):
    ax.set_xlim(-0.5, N_LS - 0.5)
    ax.tick_params(axis="y", labelsize=ps.ANNOTATION_FONTSIZE)

# --- one legend for the whole figure -------------------------------
# Two columns: the samplers of a and b on the left, the architectures of c and d on
# the right. matplotlib fills columns top to bottom and splits the handles evenly,
# so an invisible third entry pads the left column to the height of the right one;
# without it the first bar would wrap into the left column.
pad = Line2D([], [], ls="none", label="")
legend_handles = [
    Line2D([], [], color=C_STD, ls="-", lw=ps.LW, label="Standard Nested Sampling"),
    Line2D([], [], color=C_FLOW, ls=":", lw=ps.LW, label="Flow-Based Nested Sampling"),
    pad,
    *[Patch(facecolor=C_ARCH[k], label=lab) for k, (_, lab) in enumerate(SERIES)],
]
# Right-aligned rather than centred, so that the bar column sits over panels c and d;
# the column gap is then tuned so that the line column sits over a and b. Measured
# centres: bars 0.852 against a c/d centre of 0.857, lines 0.359 against 0.359.
fig.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 1.00),
           ncol=2, frameon=False, columnspacing=13.0)

for ax, tag in zip((axa, axb, axc, axd), "abcd"):
    ax.set_title(f"{tag})", loc="left", fontweight="bold")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
ps.savefig_all(fig, OUT, print_scale=PRINT_SCALE)
