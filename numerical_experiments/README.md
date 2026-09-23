# Numerical experiments

Code behind Figures 1, 6 and 8 of the manuscript (the figures not produced
by the `2D-testsystems/` or `LJ-disks/` notebooks), together with the
data-generation scripts that produced the `.npz` files shipped in
`../data/numerical_experiments/`.

Figure scripts (`make_fig_*.py`, here in `numerical_experiments/`) write PDF
+ PNG into `./figures/` (gitignored). By default `make_fig_landscapes.py`
and `make_fig_internal_complexity_two_density_efficiency.py` (the two that
need input data) read their `.npz` from the committed
`../data/numerical_experiments/`; pass `--data-dir` to point them at
freshly regenerated data instead — e.g. `--data-dir generate_data/output`,
what `run_all.sh` does after regenerating everything. Data-generation
scripts (in `generate_data/`) read and write `.npz`/diagnostic PNGs under
`generate_data/output/` (gitignored).

## Layout

```
numerical_experiments/
├── README.md
├── requirements.txt            # pinned extra deps (bilby, lalsuite) for generate_data/
├── run_all.sh                  # regenerate everything in dependency order
├── figures/                    # fig_*.{pdf,png} land here (gitignored)
│
│   # --- figures ---
├── make_fig_landscapes.py                             # Figure 1
├── make_fig_internal_complexity_two_density_efficiency.py  # Figure 6
├── make_fig_concepts_internal_complexity.py            # Figure 8, self-contained
│
└── generate_data/               # all data-generation scripts; needs the `numexp` extra
    ├── output/                 # regenerated .npz + diagnostic PNGs land here (gitignored)
    ├── _lj_compat.py           # patches lennard_jones.energy's self-mask; imported by the LJ scripts below
    ├── bh_lj.py                # LJ-8 basin hopping         -> output/lj_minima.npz
    ├── hessian_spectrum.py     # Hessian/Fisher at mode     -> output/hessian_spectra.npz   [GW]
    ├── coupling_mi.py          # Laplace-sample NMI         -> output/coupling_mi.npz        [GW]
    ├── probe_gw.py             # GW degeneracy probes       -> output/gw_degeneracies.npz    [GW]
    ├── probe_lj.py             # LJ symmetry probes         -> output/lj_symmetries.npz
    └── internal_complexity.py  # M_k/M_k^cum/drift vs 1/eta -> output/internal_complexity_{L2.9,L3.3}.npz   [RAW]
```

Every script in `generate_data/` needs the `numexp` extra installed — either
`pip install -e .[numexp]` from the repository root, or
`pip install -r requirements.txt` here for the pinned, tested versions —
even though only the three `[GW]`-marked scripts (`hessian_spectrum.py`,
`coupling_mi.py`, `probe_gw.py`) actually import `bilby`; the rest need
only numpy/scipy/matplotlib/torch plus the local `nsflows` package, kept
together with the GW ones for a single install step and a single output
directory. They resolve `_lj_compat` and `output/` relative to their own
file location, so they run correctly either as `python generate_data/bh_lj.py`
from `numerical_experiments/` (as `run_all.sh` does) or as
`python bh_lj.py` from inside `generate_data/`.


`[RAW]` marks `internal_complexity.py`, which reads NS run directories
(`generation_log.zip`, `U_max.zip`, `samples.zip`). The two runs behind Figure 6
are bundled under `../data/numerical_experiments/runs/<label>/`, trimmed to the
snapshots the script actually touches: the 45 generation events plus the final
live set used for the E_IS proxy, 46 of 999, so 23 MB per run rather than
500 MB. They reproduce `internal_complexity_*.npz` exactly. Pass run specs on
the command line, `LABEL:RUNDIR:BOX`, to point it at full runs of your own.

The bundled runs are the ones used for the figure: `L2.9` is the flow run with a
pool of 2x10^4 and cosine annealing over 250 optimisation steps, shipped in full
as `../data/lj/K10000/L2.9/runs/CA_P2e4_250os/`, and `L3.3` is the corresponding
run at the lower density.

## Reproducing

Rebuild just the figures from the committed data in
`../data/numerical_experiments/`:

```bash
cd numerical_experiments
python make_fig_landscapes.py
python make_fig_internal_complexity_two_density_efficiency.py
python make_fig_concepts_internal_complexity.py
```

Regenerate the data and rebuild the figures from it in one go (writes data
into `generate_data/output/`, then calls the figure scripts with
`--data-dir generate_data/output` so they plot the fresh data rather than
the committed copies):

```bash
./run_all.sh
```

To rebuild a single figure from freshly regenerated data without running
everything:

```bash
python make_fig_landscapes.py --data-dir generate_data/output
```

## Data-flow / run order

```
generate_data/bh_lj.py ───────────────► lj_minima.npz
                                    │
generate_data/hessian_spectrum.py ◄┘ ──► hessian_spectra.npz
                                    └──► generate_data/coupling_mi.py ──► coupling_mi.npz
generate_data/probe_gw.py ────────────────────────────────────────────► gw_degeneracies.npz
generate_data/probe_lj.py ────────────────────────────────────────────► lj_symmetries.npz
generate_data/internal_complexity.py ◄── runs/{L2.9,L3.3} (bundled, trimmed) ──► internal_complexity_{L2.9,L3.3}.npz

make_fig_landscapes.py                                     ◄── gw_degeneracies, lj_symmetries, hessian_spectra, coupling_mi
make_fig_internal_complexity_two_density_efficiency.py     ◄── internal_complexity_{L2.9,L3.3}
```

## Notes

- `make_fig_*.py` save both `.pdf` and `.png` into `./figures/`.
- The data-generation defaults reproduce the manuscript settings; each
  script exposes them via `--help` (grid resolution, temperatures, NS
  iterations, seeds, etc.).
- The GW likelihood build prints a few warnings from bilby/lal on
  startup — these are expected.
