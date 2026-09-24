"""Concept figure: why D_eff (covariance) misses multimodal structure and
how a higher-moment / negentropy measure recovers it.

This is a *reconstruction* of experiments/overleaf/concepts_internal_complexity.png,
whose original generator was lost. It is a standalone, purely synthetic
illustration (no LJ / NS data). The two quantities it contrasts mirror
``internal_complexity.py:internal_stats``:

  * D_eff : participation ratio of the covariance eigenvalues,
            (sum lambda)^2 / sum lambda^2      (internal_complexity.py:74)
  * M     : sum over eigen-directions of max(0, -kappa_k), i.e. it keeps
            directions with *negative* excess kurtosis (multimodal /
            platykurtic) and discards kappa > 0   (internal_complexity.py:75-79)

Panels (single row):
  (a) Both point clouds overlaid -- identical covariance (one shared
      1-sigma ellipse, hence identical D_eff), yet one is a Gaussian
      (easy) and one is bimodal (hard); D_eff cannot tell them apart.
  (b) A set of standardized 1-D densities p(z); each labelled with its
      negentropy J = int p log(p/phi) and excess kurtosis kappa.
  (c) The same densities in the (kappa, J) moment plane; M keeps kappa<0.

Run:  python make_fig_concepts_internal_complexity.py
Deps: numpy + matplotlib only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

from nsflows.tools import plotstyle as ps

HERE = Path(__file__).parent
MEDIA_DIR = HERE / "figures"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

ps.set_style()

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# D_eff and the moment gate -- mirrors internal_complexity.py:internal_stats
# --------------------------------------------------------------------------
def d_eff(cov: np.ndarray) -> float:
    """Participation ratio of covariance eigenvalues (internal_complexity.py:74)."""
    e = np.clip(np.linalg.eigvalsh(np.asarray(cov)), 0.0, None)
    s = e.sum()
    return float(s * s / max((e ** 2).sum(), 1e-300)) if s > 0 else 1.0


def kept_by_M(kappa: float) -> float:
    """Contribution a direction of excess kurtosis ``kappa`` makes to M:
    max(0, -kappa) -- kept only if multimodal/platykurtic (internal_complexity.py:79)."""
    return max(0.0, -float(kappa))


def panel_label(ax, text):
    ax.text(-0.13, 1.03, text, transform=ax.transAxes,
            fontsize=ps.ANNOTATION_FONTSIZE,
            fontweight="bold", va="bottom", ha="left")


# --------------------------------------------------------------------------
# 1-D densities on a standardized grid (mean 0, unit variance by construction)
# --------------------------------------------------------------------------
# Wide grid so the heavy tails of the Laplace and the sharp/quantized
# scale-mixture are captured when integrating the 2nd/4th moments.
Z = np.linspace(-30.0, 30.0, 24001)
PHI = np.exp(-Z ** 2 / 2) / np.sqrt(2 * np.pi)
# np.trapezoid is NumPy >= 2.0; np.trapz is its name on the 1.x that
# pyproject allows and that the paper environment pins.
_trap = getattr(np, "trapezoid", np.trapz)

def _gauss(z, mu, var):
    return np.exp(-(z - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def density_gaussian(z):
    return _gauss(z, 0.0, 1.0)


def density_bimodal(z, a=0.912):
    var = 1.0 - a ** 2                       # -> unit total variance
    return 0.5 * (_gauss(z, -a, var) + _gauss(z, a, var))


def density_sharp_bimodal(z, a=0.988):
    var = 1.0 - a ** 2
    return 0.5 * (_gauss(z, -a, var) + _gauss(z, a, var))


def density_uniform(z):
    h = np.sqrt(3.0)                          # unit variance -> half-width sqrt(3)
    return np.where(np.abs(z) <= h, 1.0 / (2 * h), 0.0)


def density_laplace(z):
    b = 1.0 / np.sqrt(2.0)                    # unit variance
    return np.exp(-np.abs(z) / b) / (2 * b)


def density_sharp_quantized(z, w=0.85, v1=0.035, v2=6.468):
    """Leptokurtic scale mixture: a sharp narrow spike + a heavy tail.
    With (w, v1, v2) = (0.85, 0.035, 6.468) the total variance is 1, the
    excess kurtosis is +15.8 and the central peak reaches ~1.8 -- the same
    corner of moment space (high kappa, large J) as the original
    'sharp/quantized' curve."""
    return w * _gauss(z, 0.0, v1) + (1 - w) * _gauss(z, 0.0, v2)


def moments(p):
    """Return (mean, var, excess_kurtosis, negentropy J) of pdf ``p`` on Z."""
    p = p / _trap(p, Z)
    m = _trap(Z * p, Z)
    var = _trap((Z - m) ** 2 * p, Z)
    k4 = _trap((Z - m) ** 4 * p, Z)
    kappa = k4 / var ** 2 - 3.0
    nz = p > 1e-12
    J = float(_trap(p[nz] * np.log(p[nz] / PHI[nz]), Z[nz]))
    return float(m), float(var), float(kappa), J


DENSITIES = [
    ("Gaussian",          density_gaussian,       "#1f77b4"),
    ("bimodal",           density_bimodal,        "#d62728"),
    ("sharp bimodal",     density_sharp_bimodal,  "#17becf"),
    ("uniform",           density_uniform,        "#9467bd"),
    ("Laplace (peaked)",  density_laplace,        "#ff7f0e"),
    ("sharp/quantized",   density_sharp_quantized, "#8c564b"),
]


# --------------------------------------------------------------------------
def main() -> None:
    # ---- top-row point clouds -------------------------------------------
    n = 5000
    # (a) anisotropic Gaussian; variance ratio chosen so D_eff ~ 1.48
    var_x, var_y = 1.0, 0.256
    xa = RNG.normal(0, np.sqrt(var_x), n)
    ya = RNG.normal(0, np.sqrt(var_y), n)
    cloud_a = np.column_stack([xa, ya])

    # (b) two modes along x with the SAME total covariance:
    #     within-mode var_x' = var_x - mu^2  ->  total var_x back to var_x
    mu = 0.92                                # larger split -> visibly two modes;
    #                                          total covariance is unchanged
    xb = RNG.normal(0, np.sqrt(var_x - mu ** 2), n) + RNG.choice([-mu, mu], n)
    yb = RNG.normal(0, np.sqrt(var_y), n)
    cloud_b = np.column_stack([xb, yb])

    cov_a = np.cov(cloud_a.T)
    cov_b = np.cov(cloud_b.T)
    deff_a, deff_b = d_eff(cov_a), d_eff(cov_b)

    # ---- density moments -------------------------------------------------
    stats = {}
    for name, fn, _c in DENSITIES:
        _m, _v, kappa, J = moments(fn(Z))
        stats[name] = (kappa, J)
        print(f"  {name:16s}  var={_v:.3f}  kappa={kappa:+.2f}  J={J:.3f}")

    # ---- figure : single row, three columns ------------------------------
    # Same full-page recipe as the notebook figures: canvas targets
    # \linewidth under the shared PRINT_SCALE (aspect kept from the
    # original 11.4x3.5in draft), so fonts/lines print at the same
    # physical size as every other figure in the paper.
    fig_w, fig_h = ps.figsize_for_target_width(
        ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=4.6 / 11.4,
    )
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.35, 1.0], wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])   # both clouds + shared covariance
    ax_b = fig.add_subplot(gs[0, 1])   # densities
    ax_c = fig.add_subplot(gs[0, 2])   # moment space

    lim = (-3.4, 3.4)
    ylim = (-2.2, 2.2)

    def ellipse(cov, color, label, lw=2.2):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * np.sqrt(vals)             # 1-sigma
        return Ellipse((0, 0), w, h, angle=ang, fill=False, lw=lw,
                       edgecolor=color, label=label, zorder=5)

    # (a) both clouds overlaid: identical covariance (one shared ellipse),
    #     yet one is Gaussian (easy) and one is bimodal (hard). D_eff, being
    #     a covariance statistic, sees only the ellipse and cannot tell them
    #     apart.
    nplot = 500                              # sparse subsample for legibility
    ia = RNG.choice(len(cloud_a), nplot, replace=False)
    ib = RNG.choice(len(cloud_b), nplot, replace=False)
    ax_a.scatter(cloud_a[ia, 0], cloud_a[ia, 1], s=34, facecolors="none",
                 edgecolors="#2ca02c", linewidths=1.1, marker="o", alpha=0.9,
                 label="Gaussian (Easy)", zorder=2)
    ax_a.scatter(cloud_b[ib, 0], cloud_b[ib, 1], s=34, c="#d62728",
                 linewidths=1.1, marker="x", alpha=0.9,
                 label="Bimodal (Hard)", zorder=3)
    ax_a.add_patch(ellipse(cov_a, "k",
                           rf"shared cov. ($D_{{\mathrm{{eff}}}}={deff_a:.2f}$)"))
    ax_a.set_xlim(lim); ax_a.set_ylim(ylim)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_xlabel(r"$x_1$"); ax_a.set_ylabel(r"$x_2$")
    # legend moved above the panel and frameless, per the shared
    # "legend outside the axes" convention used elsewhere in the paper.
    leg = ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05),
                      frameon=False, fontsize=ps.TICK_LABELSIZE,
                      handletextpad=0.5, borderpad=0.4)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(1.0)
        except Exception:
            pass
    panel_label(ax_a, "a)")

    # (b) densities  (J and kappa are read off panel c, so keep the legend clean)
    for name, fn, c in DENSITIES:
        ax_b.plot(Z, fn(Z), color=c, lw=1.8, label=name)
    ax_b.set_xlim(-4, 4)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel(r"Standardized Coordinate $z$")
    ax_b.set_ylabel(r"Density $p(z)$")
    # legend moved above the panel, forced to 2 columns (6 entries -> 3
    # rows) so it doesn't run wider than the panel itself.
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2,
                framealpha=1, edgecolor="gray")
    ax_b.grid(True, alpha=0.25)
    panel_label(ax_b, "b)")

    # (c) moment space
    ax_c.axvspan(-100, 0, color="#f7d5d5", alpha=0.7, zorder=0)
    ax_c.axvspan(0, 100, color="#ececec", alpha=0.8, zorder=0)
    ax_c.axvline(0, color="0.3", lw=1.0, zorder=1)
    # per-point label offsets (in points) chosen to avoid overlaps / clipping
    # at the original 7.5pt font. These 6 points sit in two tight clusters
    # in (kappa, J) space (Gaussian/Laplace near the origin; bimodal/
    # sharp-bimodal/uniform all just past the symlog threshold), so bigger
    # text collides fast -- use ANNOTATION_FONTSIZE_SMALL (smaller than
    # the main tick/axis labels, per the same "dense in-panel text" tier
    # used for the reference figure's legends and the timing mosaic) and
    # scale the offsets up by only that (smaller) growth factor.
    #
    # Directions also had to change, not just scale: on this symlog axis
    # "sharp bimodal" (kappa=-1.9) and "sharp/quantized" (kappa=+15.8)
    # used to extend their labels TOWARD each other (right / left resp.)
    # -- fine at the old tiny font, but at the bigger font they now meet
    # in the visually-compressed middle. Likewise "Gaussian" (kappa=0)
    # extended right toward "Laplace" (kappa=3), and "Laplace" extended
    # down into the x-tick labels. Flipped all four to point apart.
    label_fontsize = ps.ANNOTATION_FONTSIZE_SMALL
    _off_scale = label_fontsize / 7.5
    label_off = {
        "Gaussian":         (-5, 0, "right", "center"),
        "bimodal":          (-30, 10, "left", "top"),
        "uniform":          (-5, -3, "left", "top"),
        "sharp bimodal":    (30, 10, "right", "center"),
        "Laplace (peaked)": (0, 5, "center", "bottom"),
        "sharp/quantized":  (-50, 5, "left", "bottom"),
    }
    label_off = {
        name: (dx * _off_scale, dy * _off_scale, ha, va)
        for name, (dx, dy, ha, va) in label_off.items()
    }
    for name, _fn, c in DENSITIES:
        kappa, J = stats[name]
        ax_c.scatter([kappa], [J], s=70, c=c, edgecolors="k", linewidths=0.6,
                     zorder=4)
        dx, dy, ha, va = label_off[name]
        ax_c.annotate(name, (kappa, J), textcoords="offset points",
                      xytext=(dx, dy), ha=ha, va=va, fontsize=label_fontsize, zorder=5)
    ax_c.set_xscale("symlog", linthresh=1.0)
    ax_c.set_xlim(-30, 60)
    ax_c.set_ylim(-0.08, 1.45)
    ax_c.set_xlabel(r"Excess Kurtosis $\kappa$")
    ax_c.set_ylabel(r"Negentropy $J$")
    ax_c.grid(True, alpha=0.4, zorder=1.2, linewidth=0.6)
    ax_c.text(0.03, 0.60, "kept by $M$\n(multimodal,\nHARD)", color="#b22222",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=label_fontsize)
    ax_c.text(0.52, 0.42, "discarded\n(quantized,\nEASY)", color="0.4",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=label_fontsize)
    panel_label(ax_c, "c)")

    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.12)
    ps.savefig_all(
        fig, MEDIA_DIR / "concepts_internal_complexity", print_scale=ps.PRINT_SCALE,
    )


if __name__ == "__main__":
    main()
