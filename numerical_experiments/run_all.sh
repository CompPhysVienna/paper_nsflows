#!/usr/bin/env bash
# Regenerate every .npz data file and all the figures from scratch, in
# dependency order.
#
# Data-generation scripts live in generate_data/ and write into
# generate_data/output/ (gitignored). The figure scripts stay in this
# directory and write into ./figures/ (gitignored); by default they read
# their inputs from ../data/numerical_experiments/ (the committed copies),
# but each takes a --data-dir override, which we point at
# generate_data/output/ below so the figures below are plotted from the
# data just regenerated above rather than the committed copies.
#
# Requires the dependencies in requirements.txt (the numexp extra: bilby,
# lalsuite) and the local `nsflows` package. We add the repo root to
# PYTHONPATH so `import nsflows` works without a separate `pip install -e .`
# (as long as this folder stays inside the repository).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
cd "$SCRIPT_DIR"

run() { echo; echo "==== $* ===="; python "$@"; }

# --- data generation -------------------------------------------------------
run generate_data/bh_lj.py                 # -> generate_data/output/lj_minima.npz
run generate_data/hessian_spectrum.py      # reads lj_minima.npz       -> generate_data/output/hessian_spectra.npz
run generate_data/coupling_mi.py           # reads hessian_spectra.npz -> generate_data/output/coupling_mi.npz
run generate_data/probe_gw.py              # -> generate_data/output/gw_degeneracies.npz
run generate_data/probe_lj.py              # -> generate_data/output/lj_symmetries.npz

# Reads the trimmed NS runs bundled in ../data/numerical_experiments/runs/
run generate_data/internal_complexity.py   # -> generate_data/output/internal_complexity_{L2.9,L3.3}.npz

# --- figures (--data-dir points at the just-regenerated data, not the
#              committed copies in ../data/numerical_experiments/; write to
#              ./figures/) -------------------------------------------------
run make_fig_landscapes.py --data-dir generate_data/output   # Figure 1
run make_fig_concepts_internal_complexity.py             # Figure 8 (self-contained)

run make_fig_internal_complexity_two_density_efficiency.py --data-dir generate_data/output  # Figure 6


echo; echo "Done. See generate_data/output/ for regenerated .npz data and ./figures/ for the figures."
