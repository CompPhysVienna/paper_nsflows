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
FLOW_RUN = os.path.join(DATA, "runs", "CA_P2e4_250os")      # the run of Fig. 4f
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
N_REP, N_GEN, N_LS = 10, 10, 5
TARGETS = [50000, 150000, 250000, 350000, 450000]

# Colour is the training arm, hatching the separation between the live sets of the
# window. The single-live-set arm has no window, so it appears once.
SERIES = [                       # subdirectory, colour index, hatch
    ("training_window_10K",               0, ""),
    ("training_window_10K_step11000",     1, ""),
    ("conditioning_window_10K_step11000", 2, ""),
    ("training_window_10K_step24000",     1, "////"),
    ("conditioning_window_10K_step24000", 2, "////"),
]
ARM_LABELS = ["Unconditioned, 1 Live Set", "Unconditioned, 3 Live Sets",
              "Conditioned, 3 Live Sets"]
SPACING_LABELS = [r"$\Delta = 1.1\times10^4$ Iterations", r"$\Delta = 2.4\times10^4$"]


def load(sub, name):
    """Mean over the networks, and the standard deviation of the training alone.

    Each of the N_REP networks was generated from N_GEN times, so the spread of the
    network means still carries a share of the generation noise. The balanced one-way
    decomposition removes it: MS_within estimates the generation variance and
    (MS_between - MS_within) / N_GEN the training variance.
    """
    col = {"eff_generation": "eff", "RESS": "ress"}[name]
    r = np.genfromtxt(os.path.join(COND, sub, "per_generation.txt"),
                      names=["target", "rep", "gen", "seed", "eff", "ress", "eff_id"],
                      dtype=None, encoding=None)
    m, s = [], []
    for t in TARGETS:
        g = r[r["target"] == t]
        x = np.array([g[g["rep"] == i][col][np.argsort(g[g["rep"] == i]["gen"])]
                      for i in range(N_REP)])
        per_network = x.mean(1)
        ms_within = ((x - per_network[:, None]) ** 2).sum() / (N_REP * (N_GEN - 1))
        ms_between = N_GEN * ((per_network - per_network.mean()) ** 2).sum() / (N_REP - 1)
        m.append(per_network.mean())
        s.append(np.sqrt(max((ms_between - ms_within) / N_GEN, 0.0)))
    return np.array(m), np.array(s)


# ---------------------------------------------------------------- figure
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, PRINT_SCALE, aspect=0.40)
fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
# a and b get equal cells and a third of the width between them; c and d take the
# rest, which they need now that they carry twenty-five bars.
gs = fig.add_gridspec(2, 6, width_ratios=[1.367, 1.367, 1, 1, 1, 1])
axa = fig.add_subplot(gs[:, 0])
axb = fig.add_subplot(gs[:, 1], sharey=axa)   # a and b are the same energy axis
axc = fig.add_subplot(gs[0, 2:])
axd = fig.add_subplot(gs[1, 2:], sharex=axc)

# --- a) trace -----------------------------------------------------
for x, y, color, ls, lab in ((it_std / 1e5, e_std, C_STD, "-", "Standard Nested Sampling"),
                             (it_nf / 1e5, e_nf, C_FLOW, ":", "Flow-Based Nested Sampling")):
    axa.plot(x, y, color=color, ls=ls, label=lab)
axa.set_ylabel(r"$U_{\max}-U_0$")
axa.set_xlabel(r"Iteration ($\times10^5$)")
axa.set_xticks([0, 5])

# Axes.inset_axes, not the axes_grid1 helper: the latter sizes itself against the
# pre-layout bbox and spills into b at this width.
ins = axa.inset_axes([0.40, 0.54, 0.58, 0.42])
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
width = 0.165
for k, (sub, ci, hatch) in enumerate(SERIES):
    m_g, s_g = load(sub, "eff_generation")
    m_r, s_r = load(sub, "RESS")
    off = (k - (len(SERIES) - 1) / 2) * width
    for ax, m, e in ((axc, m_g, s_g), (axd, m_r, s_r)):
        # The stored error is already the training standard deviation, so unlike the
        # earlier single-network version it is not rescaled here.
        ax.bar(x + off, m, width * 0.9, yerr=e, color=C_ARCH[ci], hatch=hatch,
               edgecolor="white" if hatch else "none", linewidth=0.0,
               error_kw=dict(lw=0.7, capsize=1.2, ecolor="0.25"))

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

# --- two legends, each over the panels it describes ----------------
# One legend cannot do this: matplotlib's columnspacing is uniform, so there is no
# way to put a wide gap between the samplers and the bars and a normal gap between
# the three arms and the two spacings. Two figure legends can, each centred on its
# own panels. An invisible third entry pads the sampler column to three rows so
# that, bottom-anchored at a common baseline, its two lines align with the top row
# of the bar block. The panel labels are set first, because the legends sit above
# the tallest panel decoration and the titles are part of it.
for ax, tag in zip((axa, axb, axc, axd), "abcd"):
    ax.set_title(f"{tag})", loc="left", fontweight="bold")

pad = Line2D([], [], ls="none", label="")
line_handles = [
    Line2D([], [], color=C_STD, ls="-", lw=ps.LW, label="Standard Nested Sampling"),
    Line2D([], [], color=C_FLOW, ls=":", lw=ps.LW, label="Flow-Based Nested Sampling"),
    pad,
]
bar_handles = [
    *[Patch(facecolor=C_ARCH[k], label=ARM_LABELS[k]) for k in range(3)],
    Patch(facecolor="0.72", label=SPACING_LABELS[0]),
    Patch(facecolor="0.72", hatch="////", edgecolor="white", linewidth=0.0,
          label=SPACING_LABELS[1]),
]

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
to_fig = fig.transFigure.inverted()
pa, pb, pc = (ax.get_position() for ax in (axa, axb, axc))
y_top = max(to_fig.transform((0, ax.get_tightbbox(renderer).y1))[1]
            for ax in (axa, axb, axc))
fig.set_layout_engine("none")

fig.legend(handles=line_handles, loc="lower center",
           bbox_to_anchor=(0.5 * (pa.x0 + pb.x1), y_top + 0.01), ncol=1, frameon=False,
           fontsize=ps.ANNOTATION_FONTSIZE, handlelength=1.5, labelspacing=0.35)
fig.legend(handles=bar_handles, loc="lower center",
           bbox_to_anchor=(0.5 * (pc.x0 + pc.x1), y_top + 0.01), ncol=2, frameon=False,
           fontsize=ps.ANNOTATION_FONTSIZE, handlelength=1.5, handleheight=1.0,
           columnspacing=2.0, labelspacing=0.35)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
ps.savefig_all(fig, OUT, print_scale=PRINT_SCALE)
