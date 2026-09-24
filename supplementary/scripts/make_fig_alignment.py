"""
Fig. S1: alignment of low-energy configurations in the square (1:1) box.

a) the alignment reference, with the outermost particle that is excluded from the matching;
b) matching cost of the reference against its own images under the eight operations of D4;
c) final live set of a flow-based run (Fig. 5f) in internal coordinates, unaligned, with
   the lattices of the two crystal orientations overlaid;
d) the same live set after alignment.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


import common as cm
from common import ps

L, N = 2.9, 8
N_SHOW = 10000

# Lattices overlaid on panel c). The low-energy crystal is the triangular lattice of 8 sites
# strained to be commensurate with the box: the box vectors are the lattice vectors
# (3, -2) and (1, 2), whose determinant is 8. Its primitive vectors are therefore fixed,
# b1 = L (1, 1) / 4 along the diagonal and b2 = L (-1, 3) / 8. The second orientation is
# its image under C4.
SUPERCELL = np.array([[3, 1], [-2, 2]])  # columns: box vectors in lattice coordinates
PRIMITIVE = L * np.linalg.inv(SUPERCELL)  # columns: b1, b2
C4 = np.array([[0, -1], [1, 0]])
LATTICE_ORIENTATIONS = [(PRIMITIVE, ":"), (C4 @ PRIMITIVE, "-")]

ps.set_style()

ref_full, ref_reduced = cm.alignment_reference(L, N)

# Stabiliser of the reference: operations mapping it (almost) onto itself.
op_costs = {
    name: cm.matching_cost(cm.remove_outermost_particle(op(ref_full)), ref_reduced, L)[0]
    for name, op in cm.D4.items()
}
print("Matching cost of the reference under D4:")
for name, c in op_costs.items():
    print(f"  {name:>16s}: {c:.4f}")

flow_final = cm.internal_coordinates(
    cm.load(cm.run_dir("f") / "samples_000000500000.pt"), N, L
)[:N_SHOW]
flow_aligned, flow_rotated = cm.align_rot90(flow_final, ref_reduced, L)

std_final = cm.internal_coordinates(cm.load(cm.STD_NS_FINAL), N, L)[:N_SHOW]
_, std_rotated = cm.align_rot90(std_final, ref_reduced, L)

print(f"Rotated by pi/2: flow run {flow_rotated.mean():.3f}, standard NS {std_rotated.mean():.3f}")

# Printed at 0.8 \linewidth under the paper's shared PRINT_SCALE, so fonts match every
# other figure once included at the width reported by savefig.
fig_w, fig_h = ps.figsize_for_target_width(0.8 * ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=1.0)
fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
axes = axes.flatten()


def draw_lattice(ax, primitive, ls, color="k", lw=0.6, alpha=0.5):
    """Bonds along b1, b2 and b2 - b1 of the lattice spanned by the columns of ``primitive``.

    Generalises draw_hexagonal_overlay_diagonal in plot.ipynb to a strained lattice.
    """
    b1, b2 = primitive.T
    n = int(np.ceil(L / min(np.linalg.norm(b1), np.linalg.norm(b2)))) + 3
    points = np.array([i * b1 + j * b2 for i in range(-n, n + 1) for j in range(-n, n + 1)])
    points = points[np.all(np.abs(points) <= L, axis=1)]
    for p in points:
        for bond in (b1, b2, b2 - b1):
            ax.plot(*np.stack([p, p + bond]).T, color=color, lw=lw, ls=ls, alpha=alpha, zorder=1)


def box_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-L / 2, L / 2)
    ax.set_xticks([-L / 2, 0, L / 2], [r"$-L/2$", r"$0$", r"$L/2$"])
    ax.set_yticks([-L / 2, 0, L / 2], [r"$-L/2$", r"$0$", r"$L/2$"])


# a) reference configuration
ax = axes[0]
r = ref_full[0].numpy()
far = np.argmax((r ** 2).sum(-1))
keep = np.arange(N) != far
ax.scatter(r[keep, 0], r[keep, 1], s=60, color="C0", zorder=3)
# The outermost particle sits at the box corner, where its minimum image is ambiguous:
# draw all four periodic images.
corners = np.sign(r[far]) * L / 2
for i, (sx, sy) in enumerate([(1, 1), (-1, 1), (1, -1), (-1, -1)]):
    ax.scatter(r[far, 0] - (1 - sx) * corners[0], r[far, 1] - (1 - sy) * corners[1], s=60,
               facecolor="none", edgecolor="C3", lw=1.5, zorder=3, clip_on=False)
ax.scatter([0], [0], s=60, color="k", marker="+", zorder=4)
box_axes(ax)

# b) matching cost under D4
ax = axes[1]
names = list(op_costs)
costs = np.array([op_costs[n] for n in names])
stabiliser = costs < 0.05
ax.barh(np.arange(8), costs, color=np.where(stabiliser, "C0", "C1"))
ax.set_yticks(np.arange(8), names)
ax.invert_yaxis()
ax.set_xlabel(r"$\langle |\Delta r|^2 \rangle$")

# c) unaligned and d) aligned live set
for ax, x in [(axes[2], flow_final), (axes[3], flow_aligned)]:
    x = x.numpy()
    ax.scatter(x[:, :, 0], x[:, :, 1], s=ps.SCATTER_SIZE, alpha=ps.SCATTER_ALPHA, color="C0",
               rasterized=True)
    box_axes(ax)

for primitive, ls in LATTICE_ORIENTATIONS:
    draw_lattice(axes[2], primitive, ls)

for ax, tag in zip(axes, "abcd"):
    ax.set_title(f"{tag})", loc="left", fontweight="bold")

ps.savefig_all(fig, cm.FIG_DIR / "fig_alignment", print_scale=ps.PRINT_SCALE)
