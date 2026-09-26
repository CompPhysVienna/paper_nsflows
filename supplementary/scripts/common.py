"""
Shared paths, loaders and alignment helpers for the Supplementary Material figures.

Run directories are resolved against ``NSFLOWS_RUN_ROOT``, which defaults to the
``data/lj`` directory of this repository. Only the files these figures read are
shipped there: the compressed ``output.txt``, the generation and training logs,
the per-pool ``conds_*.pt`` and, for the rectangular cell, a few configuration
snapshots. Point the variable at a full run archive to use complete runs.
"""

import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]

from nsflows.tools import plotstyle as ps
SUPP = Path(__file__).resolve().parents[1]
FIG_DIR = SUPP / "figures"
TAB_DIR = SUPP / "tables"

RUN_ROOT = Path(os.environ.get("NSFLOWS_RUN_ROOT", REPO / "data" / "lj"))

# Runs behind the figures of the main text (panel letters of Fig. 4) and of this SM.
RUNS = {
    "a": "K10000/L2.9/runs/1C+CA_P1e5_3375-1125os",
    "b": "K10000/L2.9/runs/1C+CA_P2e4_750-250os",
    "c": "K10000/L2.9/runs/1C+CA-CA5_P1e5_3375-1125os",
    "d": "K10000/L2.9/runs/1C+CA-CA5_P2e4_750-250os",
    "e": "K10000/L2.9/runs/CA_P1e5_500os",
    "f": "K10000/L2.9/runs/CA_P2e4_250os",
    "L3.3": "K10000/L3.3/runs/CA_P2e4_250os",
    "hex": "rect_N9_rho0.95/runs/CA_P2e4_250os",
    "std": "K10000/L2.9/runs/STD_NS",
}

# Final live set of the standard-NS reference run at L = 2.9; its first
# configuration is the alignment reference, as for Fig. 2 of the main text.
STD_NS_FINAL = REPO / "data/lj/K10000/L2.9/samples_ref.pt"



def run_dir(key):
    return RUN_ROOT / RUNS[key]


def load(path):
    return torch.load(path, map_location="cpu")


# ------------------------------------------------------------------
# Run bookkeeping
# ------------------------------------------------------------------

# The run readers and the energy budget live in the library, so the notebook
# that draws the timing figure counts evaluations the same way as Table S2.
from nsflows.tools.runs import (  # noqa: E402
    GENERATION_BATCH,
    read_output,
    generation_attempts,
    read_timings,
    elapsed_hours,
    epochs_per_training,
    energy_budget,
)


def internal_coordinates(x, n_particles, box_length):
    """Shift particle 0 to the origin and wrap with the minimum-image convention."""
    x = x.reshape(-1, n_particles, 2).clone()
    x -= x[:, :1]
    x -= box_length * torch.round(x / box_length)
    return x


def remove_outermost_particle(x):
    """Drop, per configuration, the particle farthest from the reference particle."""
    far = torch.argmax((x ** 2).sum(-1), dim=1)
    keep = torch.ones(x.shape[:2], dtype=torch.bool)
    keep[torch.arange(x.shape[0]), far] = False
    return x[keep].view(x.shape[0], x.shape[1] - 1, 2)


def matching_cost(x, x_ref, box_length):
    """Mean squared minimum-image displacement after optimal (Hungarian) relabelling."""
    d = x[:, :, None, :] - x_ref[:, None, :, :]
    d -= box_length * torch.round(d / box_length)
    c = (d ** 2).sum(-1).numpy()
    return np.array([ci[linear_sum_assignment(ci)].mean() for ci in c])


# The eight operations of the square's point group D4 acting on (x, y).
D4 = {
    r"$E$": lambda y: y,
    r"$C_4$": lambda y: torch.stack([-y[..., 1], y[..., 0]], -1),
    r"$C_2$": lambda y: -y,
    r"$C_4^3$": lambda y: torch.stack([y[..., 1], -y[..., 0]], -1),
    r"$\sigma_x$": lambda y: y * torch.tensor([-1.0, 1.0]),
    r"$\sigma_y$": lambda y: y * torch.tensor([1.0, -1.0]),
    r"$\sigma_d$": lambda y: y[..., [1, 0]],
    r"$\sigma_{d'}$": lambda y: -y[..., [1, 0]],
}


def align_rot90(x, x_ref_reduced, box_length):
    """Rotate by pi/2 every configuration that matches the reference better that way.

    ``x`` holds full configurations in internal coordinates; the matching is done on
    the configuration with its outermost particle removed, and the chosen operation
    is applied to the full configuration. Returns the aligned configurations and a
    boolean mask of those that were rotated.
    """
    reduced = remove_outermost_particle(x)
    cost_e = matching_cost(reduced, x_ref_reduced, box_length)
    cost_r = matching_cost(D4[r"$C_4$"](reduced), x_ref_reduced, box_length)
    rotate = torch.from_numpy(cost_r < cost_e)
    aligned = x.clone()
    aligned[rotate] = D4[r"$C_4$"](x[rotate])
    return aligned, rotate.numpy()


def alignment_reference(box_length=2.9, n_particles=8):
    full = internal_coordinates(load(STD_NS_FINAL), n_particles, box_length)
    return full[:1], remove_outermost_particle(full[:1])
