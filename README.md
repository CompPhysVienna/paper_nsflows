# paper_nsflows

Companion repository for the paper *"Generative Nested Sampling of Atomistic
Thermodynamic Landscapes"* (NSFlows),
[arXiv:2609.03193](https://arxiv.org/abs/2609.03193).

This repository contains `nsflows`, the Python package used to produce all
results reported in the paper: a nested sampling implementation in which the
constrained-prior sampling step is performed by a normalizing flow.

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

Each run creates its own directory under `2D-testsystems/output/<run-id>/` and
writes there as it goes:

- `output.txt`, one row per iteration: iteration, live points, acceptance, energy bound, index of the replaced walker
- `samples_<iter>.pt` and `U_max_<iter>.pt`, snapshots of the live set and the energy bound every `isavesamp` iterations
- `timings.txt`, wall-clock time split across sampling, flow sampling, training and pool generation

These directories are not tracked by git. A run interrupted with `Ctrl-C` still
returns its history and writes `timings.txt`, so a partial run is usable: the
plotting cells adapt to however many iterations completed.

### Provided data

`data/dw/` holds the inputs the notebooks need, grouped by the number of live
points `K`, zero-padded to five digits. Both notebooks pick the right directory
from the `live_samples` parameter, so no path needs editing.

| File | Purpose |
|---|---|
| `data/dw/K<K>/samples_init.pt` | Initial live set of `K` points, used to start a run so that every run begins from the same configuration |
| `data/dw/K<K>/reference_from_std_ns.txt` | `output.txt` of a completed standard nested sampling run, used as the reference curve in the energy and density-of-states plots |

Sets are currently provided for `K = 1024` (`K01024`) and `K = 10000` (`K10000`).
Setting `live_samples` to any other value raises a `FileNotFoundError` listing what
is available, rather than silently starting from a mismatched configuration.

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
