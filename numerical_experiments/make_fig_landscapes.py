"""Publication figure: GW BBH vs atomistic (LJ-8) posterior landscapes.

A 2-row x 4-column panel contrasting the two systems on equal footing.

  Columns (left to right):
    1. one-coordinate cut through the landscape
       (GW: log-likelihood vs mass ratio q at fixed chirp mass;
        LJ: energy along a single particle-swap interpolation)
    2. its 2-coordinate extension
       (GW: the (Mc, q) chirp-mass "banana";
        LJ: the two-swap grid with four permutation-equivalent minima)
    3. local coupling structure from the Hessian / Fisher matrix at the
       main mode (normalised coupling magnitude |H_ij|/sqrt(|H_ii H_jj|))
    4. pairwise normalised mutual information (NMI) between coordinates
       from Laplace samples at the main mode

  Rows: top = GW BBH (15-D), bottom = LJ-8 (16-D).

The story: GW degeneracies are smooth continuous ridges with coupling
*concentrated* in a few named parameter blocks; LJ multimodality is
discrete + combinatorial (hard collision walls between permutation
copies) with coupling *diffuse* across all coordinates.

Data sources (read-only; see --data-dir, default ../data/numerical_experiments/):
  gw_degeneracies.npz   (probe_gw.py)
  lj_symmetries.npz     (probe_lj.py)
  hessian_spectra.npz   (hessian_spectrum.py)
  coupling_mi.npz       (coupling_mi.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from nsflows.tools import plotstyle as ps

HERE = Path(__file__).parent
DEFAULT_DATA_DIR = HERE.parent / "data" / "numerical_experiments"
MEDIA_DIR = HERE / "figures"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

ps.set_style()

GW_C = "#1f6fb4"      # GW accent colour
LJ_C = "#138a36"      # LJ accent colour
TRUTH_C = "#00d0ff"   # truth marker
COST_CMAP = "magma"
COUP_CMAP = "cividis"     # Hessian / Fisher coupling magnitude
NMI_CMAP = "viridis"      # pairwise mutual information
HESS_DECADES = 3          # log color range for normalised |H_ij|


# Short labels for GW parameters (matching param order in the npz).
GW_LABELS = {
    "chirp_mass": r"$\mathcal{M}$", "mass_ratio": r"$q$",
    "a_1": r"$a_1$", "a_2": r"$a_2$", "tilt_1": r"$\theta_1$",
    "tilt_2": r"$\theta_2$", "phi_12": r"$\phi_{12}$", "phi_jl": r"$\phi_{JL}$",
    "luminosity_distance": r"$d_L$", "dec": r"$\delta$", "ra": r"$\alpha$",
    "theta_jn": r"$\theta_{JN}$", "psi": r"$\psi$", "phase": r"$\varphi$",
    "geocent_time": r"$t_c$",
}


def panel_label(ax, text):
    # y=1.02 (same row as the title) worked at the old 9pt title, but at
    # the bigger ANNOTATION_FONTSIZE title the title's own centered span
    # now reaches past x=-0.15 regardless of title length (measured:
    # title x0 ends up left of the label's x0), so put the label on its
    # own row clearly above the title instead of chasing horizontal
    # clearance per-panel.
    ax.text(-0.15, 1.20, text, transform=ax.transAxes,
            fontsize=ps.ANNOTATION_FONTSIZE, fontweight="bold",
            va="bottom", ha="right")


def coupling_magnitude(H, block=None):
    """Off-diagonal |H_ij| normalised to its own maximum (diagonal
    masked). A scale-free *sparsity pattern* of the coupling, robust to
    the soft / near-singular diagonal modes that make a correlation-style
    normalisation explode for the GW Fisher matrix.

    If ``block`` is given, also mask the on-diagonal ``block``x``block``
    sub-blocks (the same-particle coordinate pairs for LJ, ordered
    [x0,y0,x1,y1,...]); the trivial intra-particle x-y coupling is then
    excluded both from the display and from the max used to normalise, so
    the colour scale reflects genuine inter-particle coupling."""
    A = np.abs(H).astype(float)
    np.fill_diagonal(A, np.nan)
    if block:
        for s in range(0, A.shape[0], block):
            A[s:s + block, s:s + block] = np.nan
    A /= np.nanmax(A)
    return A


def mask_diagonal_blocks(M, block):
    """Return a copy of ``M`` with the on-diagonal ``block``x``block``
    sub-blocks set to NaN (same-particle coordinate pairs)."""
    M = M.copy()
    for s in range(0, M.shape[0], block):
        M[s:s + block, s:s + block] = np.nan
    return M


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="directory containing gw_degeneracies.npz, lj_symmetries.npz, "
             "hessian_spectra.npz and coupling_mi.npz (default: the shipped "
             f"{DEFAULT_DATA_DIR}; pass generate_data/output/ to plot from "
             "freshly regenerated data)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    gw = dict(np.load(data_dir / "gw_degeneracies.npz", allow_pickle=True))
    lj = dict(np.load(data_dir / "lj_symmetries.npz", allow_pickle=True))
    hs = dict(np.load(data_dir / "hessian_spectra.npz", allow_pickle=True))
    mi = dict(np.load(data_dir / "coupling_mi.npz", allow_pickle=True))

    # Same full-page recipe as the notebook figures: canvas targets
    # \linewidth under the shared PRINT_SCALE (aspect kept from the
    # original 15.0x7.2in draft), so fonts/lines print at the same
    # physical size as every other figure in the paper.
    fig_w, fig_h = ps.figsize_for_target_width(
        ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect= 10. / 15.0,
    )
    fig = plt.figure(figsize=(fig_w, fig_h))
    # A uniform wspace (plt.subplots' default) has to satisfy the widest
    # column-to-column requirement everywhere, which made every gap as
    # wide as the c->d / g->h ones (colorbar tick labels + the next
    # panel's y-tick labels) even where nothing needs it (a->b, e->f
    # have no colorbar). Explicit per-gap-column widths -- tuned against
    # measured label bboxes -- give each gap only the room it needs.
    gap_ratios = [0.60, 0.46, 0.52]   # a-b/e-f, b-c/f-g, c-d/g-h (panel-width units)
    width_ratios = [1, gap_ratios[0], 1, gap_ratios[1], 1, gap_ratios[2], 1]
    gs = GridSpec(2, 7, figure=fig, width_ratios=width_ratios, wspace=.05, hspace=0.65)
    col_idx = [0, 2, 4, 6]
    axes = np.empty((2, 4), dtype=object)
    for r in range(2):
        for j, c in enumerate(col_idx):
            axes[r, j] = fig.add_subplot(gs[r, c])

    # =====================================================================
    # Row 0 : GW BBH
    # =====================================================================
    logl_truth = float(gw["e0"])
    q_grid = gw["q_grid"]
    mc_grid = gw["mc_grid"]
    truth_q = float(gw["truth_mass_ratio"])
    truth_mc = float(gw["truth_chirp_mass"])
    mc_idx = int(np.argmin(np.abs(mc_grid - truth_mc)))

    # --- (a) GW 1-D: -Delta logL vs q at the truth chirp mass ---
    ax = axes[0, 0]
    cost_q = logl_truth - gw["logl_mq"][mc_idx, :]
    ax.plot(q_grid, cost_q, color=GW_C, lw=1.8)
    ax.axvline(truth_q, color=TRUTH_C, lw=1.2, ls="--", label="injection")
    # "injection" label, rotated 90 deg CCW; both x and y are plain
    # axes-fraction coordinates (0-1) -- tweak these two numbers to
    # reposition it. x=0.78 is roughly where the vline sits; nudge right
    # from there.
    ax.text(0.625, 0.50, "injection", transform=ax.transAxes,
            rotation=90, rotation_mode="anchor",
            color="black", fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
            ha="right", va="top")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"mass ratio $q$")
    ax.set_ylabel(r"$-\Delta\log\mathcal{L}$", labelpad=-5)
    ax.set_title("one coordinate:\n" + r"$q$ at $\mathcal{M}_{\rm inj}$")#, fontsize=ps.TITLE_FONTSIZE)
    # ax.legend(loc="upper center")
    # ax.grid(True, alpha=0.25, which="both")
    panel_label(ax, "a)")

    # --- (b) GW 2-D: (Mc, q) banana ---
    ax = axes[0, 1]
    cost_mq = logl_truth - gw["logl_mq"]
    vmax = np.nanpercentile(cost_mq, 99)
    norm = SymLogNorm(linthresh=1.0, vmin=0, vmax=vmax, base=10)
    im = ax.imshow(cost_mq.T, origin="lower", aspect="auto", cmap=COST_CMAP,
                   norm=norm,
                   extent=(mc_grid.min(), mc_grid.max(),
                           q_grid.min(), q_grid.max()))
    ax.scatter([truth_mc], [truth_q], marker="x", c=TRUTH_C, s=55, lw=2,
               zorder=5)
    ax.set_xlabel(r"chirp mass $\mathcal{M}\;[M_\odot]$")
    ax.set_ylabel(r"mass ratio $q$")
    ax.set_title("two coordinates:\n" + r"$(\mathcal{M}, q)$")#, fontsize=ps.ANNOTATION_FONTSIZE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$-\Delta\log\mathcal{L}$", fontsize=ps.TICK_LABELSIZE, rotation=90, labelpad=-12)
    cb.set_ticks([0, 1e1, 1e2, 1e3])
    cb.set_ticklabels(["0", "", "", r"$10^{3}$"])
    panel_label(ax, "b)")

    # --- (c) GW Fisher coupling magnitude ---
    ax = axes[0, 2]
    names = [GW_LABELS[n] for n in hs["param_names"]]
    Cgw = coupling_magnitude(hs["H_gw"])
    hess_norm = LogNorm(vmin=10.0 ** (-HESS_DECADES), vmax=1.0)
    im = ax.imshow(Cgw, cmap=COUP_CMAP, norm=hess_norm, origin="upper",
                   aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_yticklabels(names, fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_title("Fisher coupling\nat mode")#, fontsize=ps.ANNOTATION_FONTSIZE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|H_{ij}|/\max|H|$", fontsize=ps.TICK_LABELSIZE, labelpad=-15)
    # keep a tick at every decade, but only label the ends (10^0, 10^-3)
    cb.set_ticks([1e0, 1e-1, 1e-2, 1e-3])
    cb.set_ticklabels([r"$10^{0}$", "", "", r"$10^{-3}$"])
    panel_label(ax, "c)")

    # --- (d) GW NMI (per-panel scale; structure not magnitude is shared) ---
    ax = axes[0, 3]
    NMIgw = mi["NMI_gw"].copy()
    np.fill_diagonal(NMIgw, np.nan)
    names_gw_mi = [GW_LABELS.get(n, n) for n in mi["param_names_gw"]]
    im_nmi = ax.imshow(NMIgw, cmap=NMI_CMAP, vmin=0,
                       vmax=float(np.nanmax(NMIgw)), origin="upper",
                       aspect="auto")
    ax.set_xticks(range(len(names_gw_mi)))
    ax.set_yticks(range(len(names_gw_mi)))
    ax.set_xticklabels(names_gw_mi, rotation=90, fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_yticklabels(names_gw_mi, fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_title("coordinate NMI\n")#, fontsize=ps.ANNOTATION_FONTSIZE)
    ax.text(0.65, 0.04,
            f"conc. {float(mi['conc_gw']):.2f}\n"
            r"$\langle\mathrm{NMI}\rangle$="
            f"{float(mi['mean_nmi_gw']):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45,
                      ec="none"))
    cb = fig.colorbar(im_nmi, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("NMI", fontsize=ps.TICK_LABELSIZE, labelpad=-15)
    cb.set_ticks([0, 0.1, 0.2, 0.3])
    cb.set_ticklabels(["0", "", "", "0.3"])
    panel_label(ax, "d)")

    # =====================================================================
    # Row 1 : LJ-8
    # =====================================================================
    e0 = float(lj["e0"])
    ts_swap = lj["ts_swap"]
    e_swap = lj["e_swap"] - e0
    ij = lj["swap_ij"]
    s2 = lj["s_2d"]
    swap2 = lj["swap_2d"]

    # --- (e) LJ 1-D swap ---
    ax = axes[1, 0]
    ax.plot(ts_swap, e_swap, color=LJ_C, lw=1.8)
    ax.axvline(0.0, color=TRUTH_C, lw=1.2, ls="--")
    ax.axvline(1.0, color=TRUTH_C, lw=1.2, ls=":")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(rf"swap $({ij[0]},{ij[1]})$ coordinate $t$")
    ax.set_ylabel(r"$U - U_0$", labelpad=-5)
    ax.set_title("one coordinate\n(single swap)")#, fontsize=ps.ANNOTATION_FONTSIZE)
    # ax.grid(True, alpha=0.25, which="both")
    panel_label(ax, "e)")

    # --- inset: LJ-8 reference configuration and the swap coordinate ---
    box_lj = float(lj["box_length"])
    pos = lj["x0"] % box_lj                      # wrapped positions
    i0, j0 = int(ij[0]), int(ij[1])
    others = [m for m in range(len(pos)) if m not in (i0, j0)]
    axin = ax.inset_axes([0.2, -.025, 0.6, 0.6])
    axin.plot([0, box_lj, box_lj, 0, 0], [0, 0, box_lj, box_lj, 0],
              color="0.5", lw=0.8, zorder=1)
    axin.scatter(pos[others, 0], pos[others, 1], s=20, c="0.65",
                 edgecolors="k", linewidths=0.3, zorder=3)
    axin.scatter(*pos[i0], s=46, c=LJ_C, edgecolors="k", linewidths=0.5,
                 zorder=4)
    axin.scatter(*pos[j0], s=46, c="#d98a13", edgecolors="k",
                 linewidths=0.5, zorder=4)
    d01 = pos[j0] - pos[i0]
    d01 -= box_lj * np.round(d01 / box_lj)       # min-image swap vector
    axin.annotate("", xy=pos[i0] + d01, xytext=pos[i0],
                  arrowprops=dict(arrowstyle="<->", color="k", lw=1.3),
                  zorder=5)
    mid = pos[i0] + 0.5 * d01
    axin.scatter(*mid, marker="x", c="crimson", s=34, lw=1.6, zorder=6)
    axin.text(*(pos[i0] + np.array([-0.12, -0.12])), str(i0), color=LJ_C,
              fontsize=ps.ANNOTATION_FONTSIZE_SMALL, ha="right", va="top", fontweight="bold")
    axin.text(*(pos[j0] + np.array([0.12, 0.12])), str(j0), color="#d98a13",
              fontsize=ps.ANNOTATION_FONTSIZE_SMALL, ha="left", va="bottom", fontweight="bold")
    axin.text(mid[0] + 0.28, mid[1] - 0.22, r"$t$", fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
              ha="center", va="center")
    axin.set_xlim(-0.35, box_lj + 0.35)
    axin.set_ylim(-0.35, box_lj + 0.35)
    axin.set_aspect("equal")
    axin.set_xticks([])
    axin.set_yticks([])
    for s in axin.spines.values():
        s.set_visible(False)

    # --- (f) LJ 2-D two-swap grid ---
    ax = axes[1, 1]
    e2 = lj["e_2d"] - e0
    vmax = np.nanpercentile(e2, 99)
    norm = SymLogNorm(linthresh=1.0, vmin=0, vmax=vmax, base=10)
    im = ax.imshow(e2.T, origin="lower", aspect="auto", cmap=COST_CMAP,
                   norm=norm,
                   extent=(s2.min(), s2.max(), s2.min(), s2.max()))
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        ax.scatter([x], [y], marker="x", c=TRUTH_C, s=45, lw=1.8, zorder=5)
    ax.set_xlabel(rf"swap $({swap2[0]},{swap2[1]})$ coordinate $t_1$")
    ax.set_ylabel(rf"swap $({swap2[2]},{swap2[3]})$ coordinate $t_2$")
    ax.set_title("two coordinates\n(two swaps)")#, fontsize=ps.ANNOTATION_FONTSIZE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$U-U_0$", fontsize=ps.TICK_LABELSIZE, rotation=90, labelpad=-12)
    cb.set_ticks([0, 1e1, 1e2, 1e3])
    cb.set_ticklabels(["0", "", "", r"$10^{3}$"])
    panel_label(ax, "f)")

    # --- (g) LJ Hessian coupling magnitude ---
    ax = axes[1, 2]
    Clj = coupling_magnitude(hs["H_lj"], block=2)
    im = ax.imshow(Clj, cmap=COUP_CMAP, norm=hess_norm, origin="upper",
                   aspect="auto")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xlabel("coordinate index", fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_ylabel("coordinate index", fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_title("Hessian coupling\nat mode")# , fontsize=ps.ANNOTATION_FONTSIZE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|H_{ij}|/\max|H|$", fontsize=ps.TICK_LABELSIZE, labelpad=-15)
    cb.set_ticks([1e0, 1e-1, 1e-2, 1e-3])
    cb.set_ticklabels([r"$10^{0}$", "", "", r"$10^{-3}$"])
    panel_label(ax, "g)")

    # --- (h) LJ NMI (per-panel scale) ---
    ax = axes[1, 3]
    NMIlj = mi["NMI_lj"].copy()
    np.fill_diagonal(NMIlj, np.nan)
    NMIlj = mask_diagonal_blocks(NMIlj, block=2)
    im = ax.imshow(NMIlj, cmap=NMI_CMAP, vmin=0,
                   vmax=float(np.nanmax(NMIlj)), origin="upper",
                   aspect="auto")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xlabel("coordinate index", fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_ylabel("coordinate index", fontsize=ps.ANNOTATION_FONTSIZE_SMALL)
    ax.set_title("coordinate NMI\n")#, fontsize=ps.ANNOTATION_FONTSIZE)
    ax.text(0.65, 0.04,
            f"conc. {float(mi['conc_lj']):.2f}\n"
            r"$\langle\mathrm{NMI}\rangle$="
            f"{float(mi['mean_nmi_lj']):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=ps.ANNOTATION_FONTSIZE_SMALL,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45,
                      ec="none"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("NMI", fontsize=ps.TICK_LABELSIZE, labelpad=-20)
    cb.set_ticks([0, 0.01, 0.02, 0.03])
    cb.set_ticklabels(["0", "", "", "0.03"])
    panel_label(ax, "h)")

    # ---- row labels on the far left ----
    fig.text(0.005, 0.727, "GW BBH (15-D)", rotation=90, va="center",
             ha="left", fontsize=ps.ANNOTATION_FONTSIZE*1.25, fontweight="bold", color=GW_C)
    fig.text(0.005, 0.223, "LJ-8 (16-D)", rotation=90, va="center",
             ha="left", fontsize=ps.ANNOTATION_FONTSIZE*1.25, fontweight="bold", color=LJ_C)

    # Margins tuned by measurement (row-label/left clearance, panel-label
    # + 2-line-title headroom above row 2, tick-label footroom below row
    # 1); wspace/hspace come from the GridSpec above, not from here.
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.07)
    ps.savefig_all(fig, MEDIA_DIR / "fig_landscapes", print_scale=ps.PRINT_SCALE)


if __name__ == "__main__":
    main()
