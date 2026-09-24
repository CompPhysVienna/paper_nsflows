"""
Fig. S2: number of pool draws needed per NS iteration, for the two CA runs of Fig. 5
(e: pool of 1e5, f: pool of 2e4). Ported from the last plotting cell of
LJ-disks/nsflows_vconst.ipynb.

During flow-driven iterations the third column of output.txt is the number of
configurations drawn from the pool before one lies below the current bound (>= 1);
during standard-NS iterations it is the MC acceptance (< 1), masked here. A dotted
line marks every return to standard NS, i.e. every time the pool runs out.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import common as cm
from common import ps

ps.set_style()

runs = [("f", r"$2\times 10^4$", "C0"), ("e", r"$10^5$", "C1")]

# Printed at \linewidth under the paper's shared PRINT_SCALE.
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=0.39)
fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, constrained_layout=True)

for ax, (key, label, color) in zip(axes, runs):
    it, acc, _, is_flow = cm.read_output(cm.run_dir(key))
    picks = np.where(is_flow, acc, np.nan)

    # Pool exhausted: the last flow iteration before a standard-NS stretch.
    empties = it[1:][is_flow[:-1] & ~is_flow[1:]]

    ax.plot(it / 1e5, picks, color=color, lw=0.8, rasterized=True)
    for x in empties:
        ax.axvline(x / 1e5, color="k", ls=":", lw=0.8, alpha=0.7)

    ax.set_yticks([1, np.nanmax(picks)])
    ax.set_ylim(0, None)

    flow_picks = acc[is_flow]
    per_pool = np.split(flow_picks, np.where(np.diff(np.flatnonzero(is_flow)) > 1)[0] + 1)
    print(f"Run {key} (pool {label}): max picks {flow_picks.max():.0f}, "
          f"mean picks {flow_picks.mean():.3f}, fraction of iterations with >1 pick "
          f"{(flow_picks > 1).mean():.3f}, NS iterations per pool {np.mean([len(p) for p in per_pool]):.0f}, "
          f"pools {len(per_pool)}")

axes[-1].set_xlabel(r"NS iteration ($\times 10^5$)")
fig.supylabel("Pool draws per iteration", fontsize=ps.AXES_LABELSIZE)

handles = [mlines.Line2D([], [], color=c, label=f"Pool size {l}") for _, l, c in runs]
handles.append(mlines.Line2D([], [], color="k", ls=":", label="Pool exhausted"))
axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)

ps.savefig_all(fig, cm.FIG_DIR / "fig_pool_picks", print_scale=ps.PRINT_SCALE)
