"""
Shared matplotlib style for figures going into the NeurIPS 2026 paper.

The calibration model
----------------------
The look of a figure (crowding of tick labels, legend fit, spacing
between panels, ...) is set by the RATIO of font size to panel size,
not by either one alone. So the one thing that must never change
between figures is that ratio: every figure uses the same rcParams
(``set_style()``) AND the same physical panel size in cm
(``REFERENCE_PANEL_SIZE_CM``), calibrated once by hand against a
reference figure (the dense 5x4 grid).

Concretely: build every figure's canvas with `panel_figsize()` /
`figsize_cm()`, which size it in physical cm using that same
constant -- never by fitting it to `\\linewidth`. A big multi-column
figure will naturally come out much wider (in inches) than a small
one; that's fine and expected.

Getting them all onto the page at a *consistent visible font size*
is then a LaTeX-side concern, not a matplotlib one: shrinking a
finished figure by `\\includegraphics[width=...]` scales fonts, line
widths and spacing together, so it preserves the calibrated look --
unlike changing the matplotlib canvas size, which does not rescale
fonts along with it (rcParams sizes are absolute points).

So: pick ONE global scale factor, ``PRINT_SCALE`` (see below), from
how the reference figure needs to be placed, and apply that SAME
factor to every figure's `\\includegraphics` width. `savefig()` will
print exactly what to write in the .tex, given that scale.

Usage
-----
    from nsflows.tools import plotstyle as ps

    ps.set_style()                                        # once
    fig_w, fig_h = ps.panel_figsize(ncols=5, nrows=4)      # reference figure
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    ...
    # This figure is meant to span the full \\linewidth once placed.
    # Compute PRINT_SCALE from the ACTUAL saved width (post-crop),
    # returned by savefig:
    w_in, h_in = ps.savefig(fig, "figures/fig1.pdf")
    PRINT_SCALE = ps.NEURIPS_LINEWIDTH_IN / w_in

    # Every other figure reuses the SAME PRINT_SCALE, whatever its
    # own native size turns out to be, so fonts match fig1's:
    fig_w2, fig_h2 = ps.figsize_cm(2 * ps.REFERENCE_PANEL_SIZE_CM,
                                    1.08 * ps.REFERENCE_PANEL_SIZE_CM)
    ...
    ps.savefig(fig2, "figures/fig2.pdf", print_scale=PRINT_SCALE)
"""

import re
import struct
from pathlib import Path

import matplotlib as mpl

CM = 1 / 2.54
NEURIPS_LINEWIDTH_IN = 5.5  # \linewidth in neurips_2026.sty (single column)

# ---- typography / line constants shared by every figure in the paper ----
FONT_SIZE = 16
AXES_TITLESIZE = 17
AXES_LABELSIZE = 15
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14

# Smaller legend size (the pre-bump default) for legends packed into
# tight spaces -- e.g. the two in-panel legends of the dense 5x4
# reference grid figure, where LEGEND_FONTSIZE crowds the panel. Pass
# explicitly as fontsize=ps.LEGEND_FONTSIZE_SMALL to those `.legend()`
# calls; every other legend keeps using the global LEGEND_FONTSIZE.
LEGEND_FONTSIZE_SMALL = 9

# Small annotation tiers for dense in-panel text that must read
# smaller than the main tick/axis labels but still be legible -- e.g.
# the timing mosaic's panel tags ("a)"), secondary-axis label, and
# the wall-time/energy-eval text box. Defined RELATIVE to
# TICK_LABELSIZE (not as bare literals) so they track it if it's
# tuned again later, instead of drifting out of proportion the way
# the mosaic's old hardcoded 10/9/7pt did across repeated bumps.
ANNOTATION_FONTSIZE = TICK_LABELSIZE - 2
ANNOTATION_FONTSIZE_SMALL = TICK_LABELSIZE - 3

LW = 1.5
MS = 5
CAPSIZE = 2.5
SCATTER_SIZE = 0.25
SCATTER_ALPHA = 0.04
AXES_LINEWIDTH = 0.8
TICK_WIDTH = 0.8
TICK_SIZE = 3

# The physical panel size (in cm) the reference figure was hand-tuned
# at. Every figure in the paper builds its canvas from this SAME
# constant -- it is the calibration anchor, not the page width.
REFERENCE_PANEL_SIZE_CM = 6.0

# Number of columns in the reference 5x4 grid figure (new_plotter.ipynb).
REFERENCE_GRID_NCOLS = 5

# The shared print scale every figure in the paper is included at:
# printed_width = native_width * PRINT_SCALE. Inside the notebook this
# is instead derived live from the reference figure's ACTUAL saved
# width (ps.savefig_all's return value), which is marginally more
# precise since bbox_inches="tight" crops a little; this nominal,
# formula-derived version differs from that by well under 1% and is
# what standalone scripts (which never run the reference figure, so
# have no live value to read back) should use instead, via
# ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, ...).
# Recomputes itself automatically if REFERENCE_PANEL_SIZE_CM or
# REFERENCE_GRID_NCOLS ever change.
PRINT_SCALE = NEURIPS_LINEWIDTH_IN / (REFERENCE_GRID_NCOLS * REFERENCE_PANEL_SIZE_CM * CM)


def set_style():
    """Apply the shared rcParams. Call once, before creating any figure."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",

        "font.size": FONT_SIZE,
        "axes.titlesize": AXES_TITLESIZE,
        "axes.labelsize": AXES_LABELSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,

        "lines.linewidth": LW,
        "lines.markersize": MS,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": AXES_LINEWIDTH,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": TICK_SIZE,
        "ytick.major.size": TICK_SIZE,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,

        "legend.frameon": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def figsize_cm(width_cm, height_cm):
    """Canvas size, in inches, from an explicit physical size in cm."""
    return width_cm * CM, height_cm * CM


def panel_figsize(ncols, nrows, panel_size_cm=REFERENCE_PANEL_SIZE_CM,
                   panel_aspect=1.0):
    """
    Canvas size, in inches, for a grid of ``nrows`` x ``ncols`` panels,
    each ``panel_size_cm`` wide (the SAME value for every figure in the
    paper -- do not shrink this to fit ``\\linewidth``; see module
    docstring). ``panel_aspect`` = panel height / panel width.
    """
    panel_w_in = panel_size_cm * CM
    fig_w = ncols * panel_w_in
    fig_h = nrows * panel_w_in * panel_aspect
    return fig_w, fig_h


def figsize_for_target_width(target_width_in, print_scale, aspect=1.0):
    """
    Canvas size, in inches, for a figure whose FINAL PRINTED width --
    after ``ps.savefig(fig, path, print_scale=print_scale)`` reports
    the ``\\includegraphics`` width to use -- comes out to exactly
    ``target_width_in``.

    This is the inverse of the usual calibration direction (native
    panel size -> whatever it happens to print at): here you fix the
    PRINTED width you want (e.g. ``NEURIPS_LINEWIDTH_IN``, to span the
    same space as the reference figure) and solve for the native
    canvas. Because ``print_scale`` is the SAME number used for every
    figure in the paper, and rcParams (font/line sizes) never change,
    a figure built this way reproduces fonts/line widths at the exact
    same *physical size on the page* as the reference figure -- it
    will just have more (or less) breathing room per panel/label if
    it packs a different number of panels into that same width, which
    is expected and fine.

    aspect : canvas height / canvas width.
    """
    native_w = target_width_in / print_scale
    native_h = native_w * aspect
    return native_w, native_h


def savefig(fig, path, print_scale=None, **kwargs):
    """
    Save ``fig`` and report its native size, plus (if ``print_scale``
    is given) the exact `\\includegraphics` width to use so this figure
    is scaled by the SAME factor as every other figure sharing that
    ``print_scale`` -- which is what keeps fonts/lines visually
    identical across figures of different native sizes.

    ``print_scale`` is a plain multiplier (not a `\\linewidth`
    fraction): compute it once from the reference figure's ACTUAL
    saved width (this function's return value), e.g.::

        w_in, h_in = ps.savefig(fig1, "fig1.pdf")
        PRINT_SCALE = ps.NEURIPS_LINEWIDTH_IN / w_in

    and reuse that same number for every other figure's ``savefig``
    call. Using the post-crop width (rather than the nominal canvas
    size) matters because ``bbox_inches="tight"`` crops each figure by
    a different amount.

    Note ``bbox_inches="tight"`` (set globally in ``set_style()``)
    crops whitespace after the fact, so the saved file can end up a
    touch smaller than the nominal canvas size passed to
    ``panel_figsize``/``figsize_cm`` -- and by a different amount for
    every figure, since it depends on each figure's own whitespace.
    So this reads the size back from the saved file itself (a PDF's
    own /MediaBox, or a PNG's pixel dimensions divided by its DPI --
    the true, post-crop size in both cases) rather than trusting the
    nominal canvas, and reports/uses THAT for ``print_scale``.

    Use a ``.png`` path for figures with lots of drawn elements (e.g.
    dense scatter clouds) -- as vector PDF, every point is a separate
    drawing instruction, which is slow to render/load and bloats file
    size; rasterizing at ``savefig.dpi`` (300, set in ``set_style()``)
    avoids that while still being print-quality at the sizes these
    figures end up placed at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", mpl.rcParams["savefig.bbox"])
    kwargs.setdefault("pad_inches", mpl.rcParams["savefig.pad_inches"])
    dpi = kwargs.get("dpi", mpl.rcParams["savefig.dpi"])
    if dpi == "figure":
        dpi = fig.dpi
    kwargs.setdefault("dpi", dpi)
    fig.savefig(path, **kwargs)

    w_in, h_in = fig.get_size_inches()
    actual_w_in, actual_h_in = _saved_size_in(path, dpi)
    if actual_w_in is not None:
        w_in, h_in = actual_w_in, actual_h_in
        size_note = "actual saved size"
    else:
        size_note = "nominal canvas size, not read back from file"

    msg = f"Saved {path} ({size_note}: {w_in:.3f} x {h_in:.3f} in)"
    if print_scale is not None:
        msg += (
            f" -- include with: "
            f"\\includegraphics[width={w_in * print_scale:.3f}in]{{{path.name}}}"
        )
    print(msg)

    return w_in, h_in


def savefig_all(fig, path, formats=(".pdf", ".png"), print_scale=None, **kwargs):
    """
    Save ``fig`` under the same stem in several formats at once (by
    default PDF + PNG) -- e.g. ``path="figures/fig1"`` (or
    ``"figures/fig1.pdf"``, extension ignored) writes
    ``figures/fig1.pdf`` and ``figures/fig1.png``.

    Every format is written from the SAME in-memory figure, so they
    only differ in file format, not in what's drawn.

    Returns the (width, height) in inches read back from the FIRST
    format's saved file -- use that (not any particular format's own
    number) as the basis for ``PRINT_SCALE``, so every other figure's
    scale is anchored to one consistent number regardless of which
    formats get compared. PDF is a good default first format for this
    since its size comes from an exact vector /MediaBox, rather than
    PNG's pixel-grid rounding.
    """
    stem = Path(path).with_suffix("")
    sizes = []
    for ext in formats:
        w_in, h_in = savefig(
            fig, stem.with_suffix(ext), print_scale=print_scale, **kwargs
        )
        sizes.append((w_in, h_in))
    return sizes[0]


def _saved_size_in(path, dpi):
    """(width, height) in inches of the file actually saved at ``path``
    -- a PDF's own /MediaBox, or a PNG's pixel size / ``dpi``. (None,
    None) if the format isn't recognized or the size can't be read."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_size_in(path)
    if suffix == ".png":
        return _png_size_in(path, dpi)
    return None, None


def _pdf_size_in(path):
    """(width, height) in inches read from a PDF's own /MediaBox, or
    (None, None) if the box can't be found."""
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    match = re.search(
        rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*\]",
        data,
    )
    if not match:
        return None, None
    x0, y0, x1, y1 = (float(v) for v in match.groups())
    points_per_inch = 72.0
    return (x1 - x0) / points_per_inch, (y1 - y0) / points_per_inch


def _png_size_in(path, dpi):
    """(width, height) in inches from a PNG's IHDR pixel dimensions
    divided by ``dpi``, or (None, None) if the file isn't a valid PNG."""
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    width_px, height_px = struct.unpack(">II", data[16:24])
    return width_px / dpi, height_px / dpi
