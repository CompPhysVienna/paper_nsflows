"""
Fig. S6: the two learning-rate schedules of Fig. 5, as used in the production runs.

a) learning rate against optimization step within one training stage, for the long and
   short versions of 1C+CA (runs 5a, 5b) and CA (runs 5e, 5f);
b) validation RESS after every epoch of the same training stages; the parameters with the
   highest value are kept.

For each run the training stage is the one whose energy threshold is closest to U_TARGET,
so that all four curves refer to a comparable constrained ensemble. Adapted from the
learning-rate cells at the end of plot.ipynb.
"""

import re
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import common as cm
from common import ps

U_TARGET = 30.0

RUNS = [
    ("a", "1C+CA (3375+1125)", "C0", "-"),
    ("b", "1C+CA (750+250)", "C0", "--"),
    ("e", "CA (500)", "C1", "-"),
    ("f", "CA (250)", "C1", "--"),
]


def train_logs(run, count):
    """Logs of the protocol stages of one training stage, in order, as (steps, lr, ress).

    Handles both the zero-padded (train_log_0005_0001.txt) and the older unpadded
    (train_log_5_1.txt) file names; steps are cumulative over the protocol stages.
    """
    logs = {}
    for path in glob.glob(str(Path(run) / "train_log_*.txt")):
        m = re.search(r"train_log_(\d+)_(\d+)\.txt$", path)
        if m and int(m.group(1)) == count:
            logs[int(m.group(2))] = np.loadtxt(path, ndmin=2)
    steps, lr, ress, offset = [], [], [], 0
    for stage in sorted(logs):
        log = logs[stage]
        steps.append(offset + log[:, 1])
        lr.append(log[:, -1])
        ress.append(log[:, 9])
        offset += log[-1, 1]
    return np.concatenate(steps), np.concatenate(lr), np.concatenate(ress)


def closest_stage(run, target):
    conds = {}
    for path in glob.glob(str(Path(run) / "conds_*.pt")):
        conds[int(re.search(r"conds_(\d+)\.pt$", path).group(1))] = cm.load(path).item()
    count = min(conds, key=lambda k: abs(conds[k] - target))
    return count, conds[count]


ps.set_style()

# Printed at \linewidth under the paper's shared PRINT_SCALE.
fig_w, fig_h = ps.figsize_for_target_width(ps.NEURIPS_LINEWIDTH_IN, ps.PRINT_SCALE, aspect=0.39)
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

for key, label, color, ls in RUNS:
    run = cm.run_dir(key)
    count, u = closest_stage(run, U_TARGET)
    steps, lr, ress = train_logs(run, count)
    axes[0].plot(steps, lr, color=color, ls=ls, label=label)
    axes[1].plot(steps, ress, color=color, ls=ls, marker="o", ms=3, mfc="none", label=label)
    best = np.argmax(ress)
    axes[1].plot(steps[best], ress[best], "*", color=color, ms=10, zorder=5)
    print(f"{key}: training stage {count}, U_train = {u:.2f}, {int(steps[-1])} steps, "
          f"best validation RESS {ress[best]:.3f} at step {int(steps[best])}")

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Optimization Step")
axes[0].set_ylabel("Learning Rate")

axes[1].set_xscale("log")
axes[1].set_xlabel("Optimization Step")
axes[1].set_ylabel("Validation RESS")
axes[1].legend(loc="upper left", fontsize=ps.ANNOTATION_FONTSIZE)

for ax, tag in zip(axes, "ab"):
    ax.set_title(f"{tag})", loc="left", fontweight="bold")

ps.savefig_all(fig, cm.FIG_DIR / "fig_lr_schedules", print_scale=ps.PRINT_SCALE)
