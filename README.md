# paper_nsflows

Companion repository for the paper *"Generative Nested Sampling of Atomistic
Thermodynamic Landscapes"* (NSFlows),
[arXiv:2609.03193](https://arxiv.org/abs/2609.03193).

This repository contains `nsflows`, the Python package used to produce all
results reported in the paper: a nested sampling implementation in which the
constrained-prior sampling step is performed by a normalizing flow.

## Which code makes which figure

| Figure | Produced by |
|---|---|
| 1 | [`numerical_experiments/make_fig_landscapes.py`](numerical_experiments/make_fig_landscapes.py) |
| 2, 3, 4, 5, 7 | [`LJ-disks/plot_ljdisks_results.ipynb`](LJ-disks/plot_ljdisks_results.ipynb) |
| 6 | [`numerical_experiments/make_fig_internal_complexity_two_density_efficiency.py`](numerical_experiments/make_fig_internal_complexity_two_density_efficiency.py) |
| 8 | [`numerical_experiments/make_fig_concepts_internal_complexity.py`](numerical_experiments/make_fig_concepts_internal_complexity.py) |

All of them read data included here, so every figure rebuilds from a fresh clone
without re-running any simulation. The runs behind Figures 4 and 5 can also be
reproduced, see [Examples](#examples) below.

## Requirements

- Python >= 3.10
- PyTorch >= 2.2
- NumPy, SciPy, Matplotlib, einops, tqdm

The results in the paper were produced on linux-64 with Python 3.12.2 and
PyTorch 2.2.1 (CUDA 11.8 / cuDNN 8.7). Network training was run on GPU; the
code also runs on CPU, more slowly.

## Installation

Three routes are provided, from the most faithful to the paper's environment to
the most portable. In all cases the final step installs `nsflows` itself in
editable mode.

### 1. Exact environment used in the paper (linux-64)

`conda_envs/spec-file_nsflows.txt` is an explicit conda spec pinning every
package to the exact build used for the published results.

```bash
conda create --name nsflows --file conda_envs/spec-file_nsflows.txt
conda activate nsflows
pip install -e .
```

This is platform-specific: it hardcodes linux-64 package URLs and will not
solve on macOS, Apple Silicon or other platforms. It may also fail if a pinned
build has since been removed from the channels — use route 2 in that case.

### 2. Portable conda environment

`conda_envs/environment.yml` states the same dependencies as version ranges and
lets conda solve them for the current platform.

```bash
conda env create -f conda_envs/environment.yml
conda activate nsflows
pip install -e .
```

This installs the CPU build of PyTorch by default. To get the GPU build used in
the paper, uncomment the `pytorch-cuda=11.8` line in the file before creating
the environment.

### 3. pip / virtualenv

`requirements.txt` pins the exact versions of the direct dependencies, without
the conda-level packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For a GPU build of PyTorch, install it from the appropriate PyTorch index
before running the command above; see https://pytorch.org/get-started/locally/.

### Checking the installation

`nsflows/__init__.py` is empty, so a bare `import nsflows` succeeds even if the
dependencies are missing. Import a module that actually pulls in PyTorch:

```bash
python -c "import nsflows.nested_sampling, nsflows.network.trainer; print('ok')"
```

## Examples

### Two-dimensional double well

The `2D-testsystems/` directory contains two notebooks that run nested sampling on
the same system: a single particle in a two-dimensional double well, defined in
`nsflows.systems.testsystems_2D`. They are the smallest complete demonstration of
the method and run on a laptop CPU, the flow-based one considerably faster on a GPU.

| Notebook | What it does |
|---|---|
| [`2D-testsystems/std_ns.ipynb`](2D-testsystems/std_ns.ipynb) | Standard nested sampling, with a rejection Monte Carlo propagator only. Useful as the baseline the flow-based results are compared against. |
| [`2D-testsystems/nsflows.ipynb`](2D-testsystems/nsflows.ipynb) | Nested sampling with a normalizing flow, which is trained on the live set and then used to generate new configurations below the energy bound. |

Run the cells in order. Both notebooks define their parameters in one cell near the
top, so that is the only place you need to edit to change a run. The flow is
controlled entirely by `turn_on_nf`: it is the iteration at which the flow takes
over, and a negative value disables it, which is what `std_ns.ipynb` uses.

### Lennard-Jones disks

`LJ-disks/` runs the same method on eight Lennard-Jones disks in two dimensions,
the system the paper's results are for. The two state points are selected by the
`density` parameter, 0.95 and 0.73, which fixes the box length and the input data.

| Notebook | What it does |
|---|---|
| [`LJ-disks/std_ns.ipynb`](LJ-disks/std_ns.ipynb) | Standard nested sampling, the baseline |
| [`LJ-disks/nsflows.ipynb`](LJ-disks/nsflows.ipynb) | Nested sampling with the flow |
| [`LJ-disks/multiple_live_sets_conditioning.ipynb`](LJ-disks/multiple_live_sets_conditioning.ipynb) | Flow efficiency against the energy bound, conditioned on it |
| [`LJ-disks/multiple_live_sets_training.ipynb`](LJ-disks/multiple_live_sets_training.ipynb) | The same, retraining at each live set instead |
| [`LJ-disks/plot_ljdisks_results.ipynb`](LJ-disks/plot_ljdisks_results.ipynb) | Figures 2, 3, 4, 5 and 7, from the data included here |

Only the last one is quick. The other four are full runs: the flow-based nested
sampling took between 8 and 32 hours per run on a GPU, and the two efficiency
scans take hours per live set. Reduce `max_ns_iterations`, or `live_sets` and
`total_steps`, to see them work before committing to a full run.

### Output of a run

Each run creates its own directory under `<system>/output/<run-id>/` and
writes there as it goes:

- `output.txt`, one row per iteration: iteration, live points, acceptance, energy bound, index of the replaced walker
- `samples_<iter>.pt` and `U_max_<iter>.pt`, snapshots of the live set and the energy bound every `isavesamp` iterations
- `timings.txt`, wall-clock time split across sampling, flow sampling, training and pool generation

Each run also writes `simulation_summary.txt` and `.json` recording the parameters
it was started with. These directories are not tracked by git. A run interrupted
with `Ctrl-C` still returns its history and writes `timings.txt`, so a partial run
is usable: the plotting cells adapt to however many iterations completed.

### Provided data

`data/` holds the inputs the notebooks need. `data/dw/` covers the double well and
`data/lj/` the Lennard-Jones disks. Both are grouped by the number of live points
`K`, zero-padded to five digits, and `data/lj/` is grouped further by box length.
The notebooks build these paths from `live_samples` and, for the Lennard-Jones
case, from the selected density, so no path needs editing.

| File | Purpose |
|---|---|
| `data/dw/K<K>/samples_init.pt` | Initial live set of `K` points, used to start a run so that every run begins from the same configuration |
| `data/dw/K<K>/reference_from_std_ns.txt` | `output.txt` of a completed standard nested sampling run, used as the reference curve in the energy and density-of-states plots |
| `data/lj/K<K>/L<L>/samples_init.pt` | As above, for the Lennard-Jones disks at box length `L` |
| `data/lj/K<K>/L<L>/samples_ref.pt` | Deep live set of a completed run, used as the reference that configurations are aligned to before plotting |
| `data/lj/K<K>/L<L>/reference_from_std_ns.txt.gz` | As the double-well reference above, compressed (see below). Provided for `L2.9` only |
| `data/lj/K10000/L2.9/live_sets/` | Live sets taken along a nested sampling run, the input to the conditioning and training scans |

For the double well, sets are provided for `K = 1024` (`K01024`) and `K = 10000`
(`K10000`). For the Lennard-Jones disks, `K = 10000` at box lengths `2.9` and `3.3`,
which are the densities 0.95 and 0.73 selected in the notebook. Asking for a value
that is not provided raises a `FileNotFoundError` listing what is available, rather
than silently starting from a mismatched configuration.

The standard nested sampling reference is the one exception: it is provided for
`L2.9` only. At `L3.3` the energy and density-of-states plots simply show the run
itself, and the notebook says so rather than failing.

#### Runs behind the paper figures

`data/lj/K10000/L2.9/runs/` holds the runs that
[`LJ-disks/plot_ljdisks_results.ipynb`](LJ-disks/plot_ljdisks_results.ipynb)
reads. Only the files the figures need are included, not the full runs.

Six of them are the nested sampling runs compared in Figure 5 of the paper, the
time decomposition of successive pool generations, split into network training
and pool generation. They all start from the same initial live
set and run for the same number of nested sampling iterations, and differ only in
the learning-rate schedule and the pool size. The folder name states all three:

    <scheduler>_P<pool size>_<optimisation steps>os

where the scheduler is One Cycle (`1C`), Cosine Annealing (`CA`), or the two
combined (`1C-CA`, 1C applied every fifth epoch and CA for the rest, whose folder
name gives the two step counts separately).

| Folder | Scheduler | Optimisation steps | Pool size |
|---|---|---|---|
| `1C_P1e5_4500os` | 1C | 4500 | 10^5 |
| `1C_P2e4_1000os` | 1C | 1000 | 2x10^4 |
| `1C-CA_P1e5_3375-1125os` | 1C + CA | 3375 (1C) + 1125 (CA) | 10^5 |
| `1C-CA_P2e4_750-250os` | 1C + CA | 750 (1C) + 250 (CA) | 2x10^4 |
| `CA_P1e5_500os` | CA | 500 | 10^5 |
| `CA_P2e4_250os` | CA | 250 | 2x10^4 |

In Figure 5 the 10^5 runs are the left column and the 2x10^4 runs the right,
one row per scheduler. The pool size is not only documented here: it can be read
back from column 6 of each run's `output.txt`, which records the number of
configurations left in the pool.

`data/lj/K10000/L2.9/conditioning_efficiency/` holds the scans behind Figure 4:
how well the flow performs as a function of the energy bound, as the generation
efficiency, the identity efficiency and the RESS. The three folders correspond to
the three curves, and each is produced by one of the notebooks below.

| Folder | Produced by |
|---|---|
| `conditioning_window_10K` | `multiple_live_sets_conditioning.ipynb` |
| `training_window_10K` | `multiple_live_sets_training.ipynb` with `collate_dataset = False` |
| `training_window_10K_collated_dataset` | `multiple_live_sets_training.ipynb` with `collate_dataset = True` |

[`LJ-disks/multiple_live_sets_conditioning.ipynb`](LJ-disks/multiple_live_sets_conditioning.ipynb) and
[`LJ-disks/multiple_live_sets_training.ipynb`](LJ-disks/multiple_live_sets_training.ipynb) train a flow
on live sets taken at several points along a nested sampling run and measure how
well it generates below the corresponding energy bound. They read the live sets
from `data/lj/K10000/L2.9/live_sets/`, which holds the `samples_<iter>.pt` and
`U_max_<iter>.pt` the two notebooks select.

Running them with the parameters as shipped takes hours per live set. The trained
networks are about 88 MB each and are not included, so the `train = False` branch,
which re-evaluates a finished run instead of training, needs a run you produced
yourself; only the resulting efficiency curves are provided here.

The phase diagram in the left panel of Figure 7 is not produced from any of this
data. It is adapted from Y.-W. Li, *Phase behavior of Lennard-Jones particles in
two dimensions*, Physical Review E (2020); only the configuration in the right
panel comes from this work.

#### The compressed reference

The Lennard-Jones reference is one row per nested sampling iteration over half a
million iterations, so it is shipped gzipped: 9.2 MB instead of 26 MB. The notebook
reads it as it is; nothing below is needed to run the examples.

NumPy reads it without unpacking anything:

```python
umax = np.loadtxt("data/lj/K10000/L2.9/reference_from_std_ns.txt.gz", usecols=3, unpack=True)
```

To unpack it anyway, for inspection or for tools that cannot read gzip:

```bash
# keep the archive
gunzip -k data/lj/K10000/L2.9/reference_from_std_ns.txt.gz

# or replace it with the plain file
gunzip data/lj/K10000/L2.9/reference_from_std_ns.txt.gz
```

Both produce `reference_from_std_ns.txt`; the first keeps the archive next to it,
the second removes it. `.gitignore` does not exclude the unpacked file, so delete it
again before committing if you unpack in place.

### Numerical experiments

`numerical_experiments/` holds the scripts for the three figures that are not
produced by the notebooks. Run them directly; each writes a PDF and a PNG into
`numerical_experiments/figures/`.

| Script | Figure | Reads |
|---|---|---|
| `make_fig_landscapes.py` | 1 | `coupling_mi.npz`, `gw_degeneracies.npz`, `hessian_spectra.npz`, `lj_symmetries.npz` |
| `make_fig_internal_complexity_two_density_efficiency.py` | 6 | `internal_complexity_L2.9.npz`, `internal_complexity_L3.3.npz` |
| `make_fig_concepts_internal_complexity.py` | 8 | nothing, it is self-contained |

```bash
cd numerical_experiments
python make_fig_landscapes.py
```

Their inputs are in `data/numerical_experiments/`, as the `.npz` files produced
by the analyses behind those figures. The scripts that generate those `.npz` are
not included here yet; the precomputed data lets the three figures be rebuilt
without them.

## Package layout

```
nsflows/
├── nested_sampling.py    # the nested sampling driver
├── network/              # normalizing flow: splines, coupling blocks, trainer
├── samplers/             # Monte Carlo and flow-based samplers
├── systems/              # test systems: Gaussians, uniforms, Einstein crystal,
│                         #   Lennard-Jones, 2D test systems
├── transformations/      # coordinate transformations and normalization
└── tools/                # observables, plotting style, utilities
```

## Citation

If you use this code, please cite:

> A. Coretti, N. Unglert, S. Falkner, G. K. H. Madsen, and C. Dellago,
> *Generative Nested Sampling of Atomistic Thermodynamic Landscapes*,
> arXiv:2609.03193 (2026).

```bibtex
@article{coretti2026nsflows,
  title         = {Generative Nested Sampling of Atomistic Thermodynamic Landscapes},
  author        = {Coretti, Alessandro and Unglert, Nico and Falkner, Sebastian and Madsen, Georg K. H. and Dellago, Christoph},
  journal       = {arXiv preprint arXiv:2609.03193},
  year          = {2026},
  eprint        = {2609.03193},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.03193}
}
```

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

See [LICENSE](LICENSE).
