"""Internal mode complexity M_k, cumulated M_k^cum, D_eff, target drift and
the (logged) generation attempts for one or more NS runs, written to
output/internal_complexity_<label>.npz for make_fig_internal_complexity.py.

For each generation event the script reads the logged generation efficiency
(attempts = 1/eff) and the energy threshold Umax, matches Umax to the nearest
live-point snapshot, and from that snapshot's sorted pair-distance fingerprint
computes:
  * M       -- internal mode complexity of the single live set,
  * Mcum    -- M of the pooled last N_CUM=3 live sets (the flow's training
               input),
  * Deff    -- fingerprint-covariance participation ratio (single set),
  * drift   -- displacement of the fingerprint mean across the cumulation
               window.
The first N_CUM-1 events have a partial cumulation window (Mcum un-pooled,
drift undefined) and are flagged via 'valid'.

This is a DATA-GENERATION script: it needs torch + the local nsflows package
AND the raw NS run directories (generation_log.zip, U_max.zip, samples.zip),
which are large and NOT bundled in this folder. By default it reads the runs
under ../experiments/; pass run specs on the command line to override:

    python internal_complexity.py LABEL:RUNDIR:BOX [LABEL:RUNDIR:BOX ...]

The precomputed output/internal_complexity_*.npz ship alongside, so the
figure rebuilds without re-running this script.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from nsflows.systems.lennard_jones import lennard_jones

HERE = Path(__file__).parent
OUT = HERE / "output"
N, CUTIN, TOL, N_CUM = 8, 0.8, 1e-3, 3
ITER = re.compile(r"_(\d+)\.(?:pt|txt)$")

# label : (run directory, box length).  The loose run is the newer L3.3.
DEFAULT_RUNS = {
    "L2.9": (HERE.parent.parent / "experiments" / "L2.9", 2.9),
    "L3.3": (HERE.parent.parent / "experiments" / "L3.3_new", 3.3),
}


def fingerprints(configs, box, n=N):
    K = configs.shape[0]
    pos = configs.reshape(K, n, 2)
    rij = pos[:, :, None, :] - pos[:, None, :, :]
    rij -= box * np.round(rij / box)
    r = np.sqrt((rij ** 2).sum(-1))
    iu = np.triu_indices(n, k=1)
    d = r[:, iu[0], iu[1]]
    d.sort(axis=1)
    return d


def internal_stats(fp):
    K = fp.shape[0]
    mu = fp.mean(0)
    fpc = fp - mu
    C = 0.5 * ((fpc.T @ fpc) / max(K - 1, 1)
               + ((fpc.T @ fpc) / max(K - 1, 1)).T)
    e, v = np.linalg.eigh(C)
    e = np.clip(e, 0.0, None)
    s = e.sum()
    d_eff = float(s * s / max((e ** 2).sum(), 1e-300)) if s > 0 else 1.0
    M = 0.0
    for k in np.where(e > TOL * e.max())[0]:
        z = (fpc @ v[:, k]) / np.sqrt(max(e[k], 1e-30))
        z -= z.mean()
        M += max(0.0, -(float((z ** 4).mean()) - 3.0))
    return M, d_eff, mu


def load_pt(zf, name):
    with zf.open(name) as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu",
                          weights_only=False)


def compute(label: str, rundir: Path, box: float):
    d = Path(rundir)
    # ---- logs: threshold + efficiency ----
    zl = zipfile.ZipFile(d / "generation_log.zip")
    logs = sorted(zl.namelist(),
                  key=lambda s: int(re.search(r"_(\d+)\.txt$", s).group(1)))
    Uev, eff = [], []
    for nm in logs:
        t = zl.read(nm).decode()
        Uev.append(float(re.search(r"Umax = ([-\d.]+)", t).group(1)))
        eff.append(float(re.search(r"Generation efficiency: ([-\d.eE]+)",
                                   t).group(1)))
    Uev = np.array(Uev)
    attempts = 1.0 / np.array(eff)

    # ---- per-snapshot thresholds ----
    zu = zipfile.ZipFile(d / "U_max.zip")
    un = sorted((n for n in zu.namelist() if n.endswith(".pt")),
                key=lambda s: int(ITER.search(s).group(1)))
    umax = np.array([float(load_pt(zu, n)) for n in un])
    j = np.array([int(np.argmin(np.abs(umax - u))) for u in Uev])

    # ---- fingerprints at each unique event snapshot (cached) ----
    zs = zipfile.ZipFile(d / "samples.zip")
    sn = sorted((n for n in zs.namelist() if n.endswith(".pt")),
                key=lambda s: int(ITER.search(s).group(1)))
    fpc = {}
    for s in sorted({int(x) for x in j}):
        cfg = load_pt(zs, sn[s]).float().numpy().astype(np.float64)
        fpc[s] = fingerprints(cfg, box)

    # single-snapshot and cumulated (last N_CUM events, pooled as the flow
    # sees its training set) internal complexity / D_eff
    M = np.empty(len(j)); Deff = np.empty(len(j))
    Mcum = np.empty(len(j)); Deffcum = np.empty(len(j))
    Mu = np.empty((len(j), N * (N - 1) // 2))
    for i in range(len(j)):
        M[i], Deff[i], Mu[i] = internal_stats(fpc[int(j[i])])
        idx = [int(j[k]) for k in range(max(0, i - (N_CUM - 1)), i + 1)]
        pooled = np.concatenate([fpc[s] for s in idx], axis=0)
        Mcum[i], Deffcum[i], _ = internal_stats(pooled)

    # approximate drift (fingerprint-mean displacement across the window).
    # the first N_CUM-1 events have a partial cumulation window and an
    # undefined drift; they are flagged via 'valid' and excluded from the
    # figure/fits to avoid a boundary spike.
    drift = np.array([np.linalg.norm(Mu[i] - Mu[max(0, i - (N_CUM - 1))])
                      for i in range(len(j))])
    drift = np.maximum(drift, 1e-6)
    valid = np.arange(len(j)) >= (N_CUM - 1)

    # ---- E_IS proxy: lowest live-point energy in the last snapshot ----
    system = lennard_jones(n_particles=N, dimensions=2, rho=N / box ** 2,
                           device=torch.device("cpu"), cutin=CUTIN, lrc=True)
    last = load_pt(zs, sn[-1]).float()
    with torch.no_grad():
        e_is = float(system.energy(last).cpu().numpy().reshape(-1).min())

    OUT.mkdir(exist_ok=True)
    out = OUT / f"internal_complexity_{label}.npz"
    np.savez(out, Uev=Uev, attempts=attempts, M=M, Mcum=Mcum,
             Deff=Deff, Deffcum=Deffcum, drift=drift, valid=valid,
             e_is=e_is, box=box)
    print(f"{label}: {len(Uev)} events  attempts[{attempts.min():.0f},"
          f"{attempts.max():.0f}]  E_IS={e_is:.3f}  -> {out.name}")


def main() -> None:
    if len(sys.argv) > 1:
        runs = {}
        for spec in sys.argv[1:]:
            label, rundir, box = spec.split(":")
            runs[label] = (Path(rundir), float(box))
    else:
        runs = DEFAULT_RUNS
    for label, (rundir, box) in runs.items():
        compute(label, rundir, box)


if __name__ == "__main__":
    main()
