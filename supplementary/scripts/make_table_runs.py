"""
Tables S2 and S3: computational budget of every flow-based run, and the network size.

Energy evaluations are the complete count of common.energy_budget, split by origin. Note
that the per-attempt batch is GENERATION_BATCH = 2e4 flow samples for both pool sizes.
"""

import torch

from nsflows.network.coupling_blocks import EquivariantRQS
from nsflows.network.circular_shift import circular_shift

import common as cm

rows = [
    # key, cell, pool size, schedule
    ("a", r"1:1", r"$10^5$", r"1C+CA (3375+1125)"),
    ("b", r"1:1", r"$2\times10^4$", r"1C+CA (750+250)"),
    ("c", r"1:1", r"$10^5$", r"1C+CA/CA(5) (3375+1125)"),
    ("d", r"1:1", r"$2\times10^4$", r"1C+CA/CA(5) (750+250)"),
    ("e", r"1:1", r"$10^5$", r"CA (500)"),
    ("f", r"1:1", r"$2\times10^4$", r"CA (250)"),
    ("L3.3", r"1:1, $\rho\approx0.73$", r"$2\times10^4$", r"CA (250)"),
    ("hex", r"$2:\sqrt{3}$", r"$2\times10^4$", r"CA (250)"),
]
labels = {"a": "2, 5a", "b": "5b", "c": "5c", "d": "5d", "e": "5e",
          "f": "3, 5f, 6 (left)", "L3.3": "6 (right)", "hex": "S3, S4"}


def sci(x):
    """Energy evaluations in units of 1e8 (the unit is given in the table header)."""
    return f"{x / 1e8:#.3g}".rstrip(".")


lines = []
for key, cell, pool, schedule in rows:
    run = cm.run_dir(key)
    b = cm.energy_budget(run)
    assert not b["missing_train_logs"], f"{key}: no training logs for {b['missing_train_logs']}"
    total = b["total"]
    lines.append(" & ".join([labels[key], cell, pool, schedule, str(b["n_pools"]),
                             f"{cm.elapsed_hours(run):.1f}", sci(b["generation"]), sci(b["mcmc"]), sci(b["training"]),
                             sci(total)]) + r" \\")
    print(key, b, f"total {total:.3e}")

std = cm.energy_budget(cm.run_dir("std"))
print("standard NS", std, f"{cm.elapsed_hours(cm.run_dir('std')):.1f} h")
lines.append(r"\midrule")
lines.append(" & ".join(["3", "1:1", "--", "standard NS", "--", f"{cm.elapsed_hours(cm.run_dir('std')):.1f}",
                         "--", sci(std["mcmc"]), "--", sci(std["total"])]) + r" \\")

cm.TAB_DIR.mkdir(parents=True, exist_ok=True)
(cm.TAB_DIR / "run_budget.tex").write_text("\n".join(lines) + "\n")

# Network size: the same for every run, since the coupling networks act per particle.
dev = torch.device("cpu")
rqs = sum(p.numel() for p in EquivariantRQS((0,), 7, 2, dev, n_bins=12, conditioned=True).parameters())
shift = sum(p.numel() for p in circular_shift(7, 2, dev).parameters())
print(f"parameters per RQS coupling layer: {rqs}; per circular shift: {shift}")
print(f"square box (28 x [shift, RQS, RQS, shift, RQS, RQS]): {28 * (4 * rqs + 2 * shift)}")
print(f"2:sqrt(3) cell (28 x [RQS, RQS, RQS, RQS]): {28 * 4 * rqs}")
