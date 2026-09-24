"""
Shared paths, loaders and alignment helpers for the Supplementary Material figures.

Run directories are resolved against ``NSFLOWS_RUN_ROOT``, which defaults to the
``data/lj`` directory of this repository. Only the files these figures read are
shipped there: the compressed ``output.txt``, the generation and training logs,
the per-pool ``conds_*.pt`` and, for the rectangular cell, a few configuration
snapshots. Point the variable at a full run archive to use complete runs.
"""

import os
import re
import glob
import gzip
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

# Runs behind the figures of the main text (panel letters of Fig. 5) and of this SM.
RUNS = {
    "a": "K10000/L2.9/runs/1C_P1e5_4500os",
    "b": "K10000/L2.9/runs/1C_P2e4_1000os",
    "c": "K10000/L2.9/runs/1C-CA_P1e5_3375-1125os",
    "d": "K10000/L2.9/runs/1C-CA_P2e4_750-250os",
    "e": "K10000/L2.9/runs/CA_P1e5_500os",
    "f": "K10000/L2.9/runs/CA_P2e4_250os",
    "fig3": "K10000/L2.9/runs/1C-CA_P2e4_750-250os_fig3",
    "L3.3": "K10000/L3.3/runs/CA_P2e4_250os",
    "hex": "rect_N9_rho0.95/runs/CA_P2e4_250os",
    "std": "K10000/L2.9/runs/STD_NS",
}

# Final live set of the standard-NS reference run at L = 2.9; its first
# configuration is the alignment reference, as for Fig. 2 of the main text.
STD_NS_FINAL = REPO / "data/lj/K10000/L2.9/samples_ref.pt"

# Each generation attempt pushes this many base samples through the flow
# (nflows_propagator.max_sample_size), whatever the target pool size.
GENERATION_BATCH = 20000


def run_dir(key):
    return RUN_ROOT / RUNS[key]


def load(path):
    return torch.load(path, map_location="cpu")


# ------------------------------------------------------------------
# Run bookkeeping
# ------------------------------------------------------------------

def _open_text(path):
    """Open a run file, transparently using the gzipped copy when that is what ships."""
    path = Path(path)
    if path.exists():
        return open(path)
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt")
    raise FileNotFoundError(f"neither {path} nor {gz} exists")


def read_output(run):
    """Columns of output.txt: iteration, K, acc, U_max, argmax[, pool size].

    ``acc`` is the MC acceptance during standard-NS iterations, and the number of
    pool draws needed to find a configuration below the bound during flow iterations
    (the latter are the rows with a sixth column).
    """
    it, acc, umax, is_flow = [], [], [], []
    with _open_text(Path(run) / "output.txt") as f:
        for line in f:
            s = line.split()
            it.append(int(s[0]))
            acc.append(float(s[2]))
            umax.append(float(s[3]))
            is_flow.append(len(s) == 6)
    return np.array(it), np.array(acc), np.array(umax), np.array(is_flow)


def generation_attempts(run):
    """Generation attempts and pool size for every pool, in pool order."""
    logs = sorted(glob.glob(str(Path(run) / "generation_log_*.txt")),
                  key=lambda p: int(re.search(r"generation_log_(\d+)\.txt", p).group(1)))
    attempts, pools = [], []
    for log in logs:
        text = open(log).read()
        attempts.append(sum(1 for l in text.splitlines() if l.strip() and not l.startswith("#")))
        pools.append(int(re.search(r"Generated a pool of (\d+)", text).group(1)))
    return np.array(attempts), np.array(pools)


def read_timings(run):
    """Per-pool timings in seconds: STD_NS, NF_NS, TRAIN, GENER."""
    data = np.genfromtxt(Path(run) / "timings.txt", skip_header=2, names=True, comments="#")
    return np.column_stack([data[n] for n in data.dtype.names])


def elapsed_hours(run):
    for line in reversed(open(Path(run) / "timings.txt").readlines()):
        if "Elapsed time:" in line:
            raw = line.split("Elapsed time:")[1].strip()
            days = 0
            if "day" in raw:
                d, raw = raw.split(", ")
                days = int(d.split()[0])
            h, m, s = raw.split(":")
            return 24 * days + int(h) + int(m) / 60 + float(s) / 3600
    return np.nan


def epochs_per_training(run):
    """Number of training epochs of every training stage, summed over its protocol stages."""
    epochs = {}
    for log in glob.glob(str(Path(run) / "train_log_*.txt")):
        count = int(re.search(r"train_log_(\d+)(?:_\d+)?\.txt", log).group(1))
        n = sum(1 for l in open(log) if l.strip() and not l.startswith("#"))
        epochs[count] = epochs.get(count, 0) + n
    return epochs


def energy_budget(run, n_live=10000, n_walkers_moved=1000, n_cycles=100, window=3,
                  test_fraction=0.1):
    """Every evaluation of the potential in a run, split by origin, as the code performs them.

    - mcmc: standard-NS iterations, i.e. the warm-up and the dilution stages. Each evaluates
      the n_walkers_moved cloned walkers once, then once per trial move for n_cycles moves;
    - generation: one evaluation per flow sample, GENERATION_BATCH per attempt;
    - training: energy labels of the whole training window when the dataset is built, and
      the validation loss on the held-out base samples (test_fraction of the window) after
      every epoch, which is also what selects the saved parameters;
    - picks: one evaluation per configuration drawn from the pool, plus one each time the
      pool runs out;
    - init: the initial live set.
    """
    attempts, _ = generation_attempts(run)
    _, acc, _, is_flow = read_output(run)
    n_std = (~is_flow).sum() - 1  # the first row is the initial live set
    n_pools = len(attempts)
    epochs = epochs_per_training(run)
    window_sizes = [n_live * min(k + 1, window) for k in range(n_pools)]
    labels = sum(window_sizes)
    validation = sum(int(test_fraction * w) * epochs.get(k, 0) for k, w in enumerate(window_sizes))
    budget = {
        "mcmc": n_std * n_walkers_moved * (n_cycles + 1),
        "generation": int(attempts.sum()) * GENERATION_BATCH,
        "training": labels + validation,
        "picks": acc[is_flow].sum() + n_pools,
        "init": n_live,
    }
    budget["total"] = sum(budget.values())
    budget["n_std_iterations"] = n_std
    budget["n_pools"] = n_pools
    budget["missing_train_logs"] = [k for k in range(n_pools) if k not in epochs]
    return budget


# ------------------------------------------------------------------
# Internal coordinates and alignment (square box)
# ------------------------------------------------------------------

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
