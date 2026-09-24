"""
Reading the output of a nested sampling run and accounting for its cost.

These helpers are shared by the Supplementary scripts and by the notebook that
draws the timing figure, so that both count energy evaluations the same way.
"""

import re
import glob
import gzip
from pathlib import Path

import numpy as np


# Each generation attempt pushes this many base samples through the flow
# (nflows_propagator.max_sample_size), whatever the target pool size.
GENERATION_BATCH = 20000


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
