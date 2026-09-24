"""Publication figure (efficiency variant): internal mode complexity M_k (and
the M_k + drift correction) as predictors of the flow generation *efficiency*
eta = 1/attempts, for two densities.

This is the inverse-y counterpart of fig_internal_complexity_two_density.py:
instead of generation attempts 1/eta on the ordinate we plot the efficiency
eta directly.  The log-space R^2 of each fit is unchanged by the inversion.

  (a) L2.9  -- box 2.9, rho ~ 0.95 (denser): multimodality-limited.
  (b) L3.3  -- box 3.3, rho ~ 0.74 (looser): compression-limited.

Extras vs. the attempts figure:
  * a top axis showing NS progress in %, increasing left -> right (0 % at the
    high-energy start, 100 % near the inherent-structure basin), so the run
    direction is immediately legible;
  * the "(a) L2.9 ..." panel labels moved inside the top-left corner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from nsflows.tools import plotstyle as ps

HERE = Path(__file__).parent
DEFAULT_DATA_DIR = HERE.parent / "data" / "numerical_experiments"
MEDIA_DIR = HERE / "figures"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

ps.set_style()

ATT_C = "#1f6fb4"     # measured efficiency
M_C = "#c0392b"       # M_k-only model
MD_C = "#6c3483"      # M_k + drift model
MCD_C = "#1a7a3a"     # M_k^cum + drift model (best)

PAD = 1.15            # log-axis padding factor (0 %/100 % sit at the edges)


def fit(cols, y):
    X = np.column_stack([np.ones(len(y))] + [np.log10(c) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return 10 ** pred, r2


def panel(ax, U, M, Mcum, drift, att, e_is, label, valid, shade_pct):
    # restrict to the full cumulation window (drop partial-window boundary
    # events, where M_k^cum is un-pooled and drift is undefined).
    U, M, Mcum, drift, att = (a[valid] for a in (U, M, Mcum, drift, att))
    x = U - e_is
    y = np.log10(att)                       # fit still done in attempts space
    M = np.maximum(M, 1e-6)
    Mcum = np.maximum(Mcum, 1e-6)
    drift = np.maximum(drift, 1e-6)
    pMc, r2Mc = fit([Mcum], y)
    pMcd, r2Mcd = fit([Mcum, drift], y)

    # efficiency = 1 / attempts (invert measured points and both models)
    eff = 1.0 / att
    eMc, eMcd = 1.0 / pMc, 1.0 / pMcd

    # sort by energy for clean lines
    o = np.argsort(-x)
    ax.plot(x[o], eMc[o], "--", color=M_C, lw=1.4, alpha=0.85, zorder=2,
            label=r"$M_k^{\mathrm{cum}}$")
    ax.plot(x[o], eMcd[o], "--", color=MCD_C, lw=1.7, alpha=0.9, zorder=2,
            label=r"$M_k^{\mathrm{cum}}+$Drift")
    ax.scatter(x, eff, marker="^", s=42, facecolors="none",
               edgecolors=ATT_C, linewidths=1.3, zorder=4,
               label=r"Measured Efficiency $\eta$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # reference line at eta = 1e-2
    ax.axhline(1e-2, color="0.5", ls="--", lw=0.8, zorder=1)

    # explicit, padded limits so the NS-progress mapping is exact at the edges.
    lo, hi = float(x.min()), float(x.max())
    left, right = hi * PAD, lo / PAD         # high energy left, basin right
    ax.set_xlim(left, right)

    # ax.grid(True, alpha=0.25, which="both")

    # panel label + R^2 of both fits, inside the top-left corner (was
    # the axes title). These differ per panel, so -- unlike the 3
    # series' own labels, now a single shared fig.legend() -- they
    # live here rather than in the legend. Single line first; if it
    # overflows the panel width, switch to "\n".join(...) instead.
    info = (
        rf"{label}    "
        rf"$R^2_{{\rm cum}}={r2Mc:.2f}$, $R^2_{{\rm cum+drift}}={r2Mcd:.2f}$"
    )
    ax.text(0.07, 0.955, info, transform=ax.transAxes,
            fontsize=ps.ANNOTATION_FONTSIZE,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                      alpha=0.9, lw=0.6)
            )
    # top axis: NS progress in %, 0 % (start) at left -> 100 % (basin) at right.
    span = np.log10(left) - np.log10(right)

    def to_pct(xv):
        xv = np.clip(np.asarray(xv, float), 1e-300, None)
        return (np.log10(left) - np.log10(xv)) / span * 100.0

    def from_pct(p):
        return 10.0 ** (np.log10(left) - np.asarray(p, float) / 100.0 * span)

    # shade a NS-progress window (mapped back to the energy axis).
    p0, p1 = shade_pct
    ax.axvspan(float(from_pct(p0)), float(from_pct(p1)),
               color="0.5", alpha=0.22, lw=0, zorder=0)

    secax = ax.secondary_xaxis("top", functions=(to_pct, from_pct))
    # No secax.set_xlabel() here: both panels' bottom/top x-axis labels
    # are identical text over different numeric ranges (can't sharex),
    # so main() puts ONE shared label of each above/below both panels
    # instead of repeating it per panel -- secax is returned so main()
    # can measure its actual tick-label extent to place that shared
    # top label precisely.
    secax.minorticks_off()
    secax.xaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    return r2Mc, r2Mcd, secax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="directory containing internal_complexity_L2.9.npz and "
             "internal_complexity_L3.3.npz (default: the shipped "
             f"{DEFAULT_DATA_DIR}; pass generate_data/output/ to plot from "
             "freshly regenerated data)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    l29 = dict(np.load(data_dir / "internal_complexity_L2.9.npz"))
    l33 = dict(np.load(data_dir / "internal_complexity_L3.3.npz"))

    # Same full-page recipe as the notebook figures: canvas targets
    # \linewidth under the shared PRINT_SCALE (aspect kept from the
    # original 9.6x4.3in draft), so fonts/lines print at the same
    # physical size as every other figure in the paper.
    fig_w, fig_h = ps.figsize_for_target_width(
        ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=4.3 / 9.6,
    )
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)

    *ra, seca = panel(
        axa, l29["Uev"], l29["M"], l29["Mcum"], l29["drift"], l29["attempts"],
        float(l29["e_is"]),
        r"(a) $L = 2.9$  ($\rho\approx0.95$)",
        l29["valid"], (15, 55))
    *rb, secb = panel(
        axb, l33["Uev"], l33["M"], l33["Mcum"], l33["drift"], l33["attempts"],
        float(l33["e_is"]),
        r"(b) $L = 3.3$  ($\rho\approx0.74$)",
        l33["valid"], (25, 80))

    # only the shared (leftmost) y-axis needs its own label now
    axa.set_ylabel(r"Generative Efficiency $\eta$")

    # share a common y-range across both panels
    ylo = min(axa.get_ylim()[0], axb.get_ylim()[0]) #- 0.05 * (axa.get_ylim()[1] - axa.get_ylim()[0])
    yhi = max(axa.get_ylim()[1], axb.get_ylim()[1]) + 0.05 * (axa.get_ylim()[1] - axa.get_ylim()[0])
    axa.set_ylim(ylo, yhi)
    axb.set_ylim(ylo, yhi)

    # Combined legend above both panels, centered -- same pattern as
    # the reference figure's fig.legend() for the g(r)/P(U) panels.
    # Both panels share the same 3 series/colors, so one legend built
    # from either panel's handles covers both; the R^2 values (which
    # DO differ per panel) live in each panel's own label box instead.
    handles, labels = axa.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.15),
        ncol=len(handles), frameon=False,
    )

    fig.tight_layout()

    # Shared x-axis labels: the two panels can't sharex (different
    # energy ranges), but the label TEXT is identical in both, so put
    # one copy of each centered across both panels instead of
    # repeating it per panel. Positioned from the actual rendered
    # tick-label boxes (figure fraction), not a guessed offset.
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()

    def bbox_fig(artist):
        return artist.get_window_extent().transformed(to_fig)

    x_center = (bbox_fig(axa).x0 + bbox_fig(axb).x1) / 2

    bottom_y = min(
        bbox_fig(t).y0
        for ax in (axa, axb) for t in ax.get_xticklabels() if t.get_text()
    )
    fig.text(x_center, bottom_y - 0.02, r"reduced energy $U-U_0$",
              ha="center", va="top", fontsize=mpl.rcParams["axes.labelsize"])

    # Centered in the gap between the secax tick labels and the
    # legend's own bottom edge (rather than padded off just one of
    # them), since that gap is narrow and the legend was measured to
    # actually sit closer than bbox_to_anchor's nominal value suggests.
    secax_top_y = max(
        bbox_fig(t).y1
        for sec in (seca, secb) for t in sec.get_xticklabels() if t.get_text()
    )
    legend_bottom_y = bbox_fig(leg).y0
    ns_label_y = (secax_top_y + legend_bottom_y) / 2
    fig.text(x_center, ns_label_y, "Nested Sampling Progress [%]",
              ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])

    ps.savefig_all(
        fig,
        MEDIA_DIR / "fig_internal_complexity_two_density_efficiency",
        print_scale=ps.PRINT_SCALE,
    )
    print(f"L2.9:  M_cum {ra[0]:.2f}   M_cum+drift {ra[1]:.2f}")
    print(f"L3.3:  M_cum {rb[0]:.2f}   M_cum+drift {rb[1]:.2f}")


if __name__ == "__main__":
    main()
