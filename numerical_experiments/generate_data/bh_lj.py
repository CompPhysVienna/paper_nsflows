"""Basin-hopping enumeration of LJ-8 (2D, PBC) inherent structures.

Procedure (Wales / Doye style):
1. Run ``scipy.optimize.basinhopping`` from each of several diverse
   starts: random uniform in the box, plus perturbations of the
   reference FCC configuration. Each call performs a Metropolis chain
   over local minima, perturbing the current minimum and re-quenching
   with L-BFGS using the torch-autograd gradient.
2. Pool every local minimum visited across all chains.
3. Deduplicate with a symmetry-invariant fingerprint (the sorted vector
   of PBC pair distances), which is invariant under particle
   permutation and PBC translation.

Sanity-checked finding: the reference FCC (square lattice with basis)
at rho ~ 0.95 in the (2.9, 2.9) box used by the rest of the codebase is
a saddle stationary point, not a local minimum — tiny perturbations
collapse to lower-energy configurations near E ~ -14 to -15.

Output: experiments/basin_hopping/output/lj_minima.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import basinhopping, minimize

import _lj_compat  # noqa: F401  # patches lennard_jones.energy self-mask
from nsflows.systems.lennard_jones import lennard_jones


def make_system(n_particles: int, box_length: float, cutin: float):
    device = torch.device("cpu")
    rho = n_particles / box_length ** 2
    return lennard_jones(
        n_particles=n_particles, dimensions=2, rho=rho,
        device=device, cutin=cutin, lrc=True,
    )


def energy_and_grad_factory(system):
    def fn(x_flat):
        x = torch.tensor(x_flat, dtype=torch.float32, requires_grad=True)
        e = system.energy(x.unsqueeze(0))[0]
        g = torch.autograd.grad(e, x)[0]
        return float(e.item()), g.detach().numpy().astype(np.float64)
    return fn


def sorted_pair_distances(x_flat: np.ndarray, n: int, dim: int,
                          box: float) -> np.ndarray:
    pos = x_flat.reshape(n, dim)
    rij = pos[:, None, :] - pos[None, :, :]
    rij -= box * np.round(rij / box)
    r = np.linalg.norm(rij, axis=-1)
    iu = np.triu_indices(n, k=1)
    return np.sort(r[iu])


def fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


class Catalog:
    """In-memory catalog of distinct minima, keyed by sorted pair
    distance fingerprint."""

    def __init__(self, n: int, dim: int, box: float, tol: float):
        self.n, self.dim, self.box, self.tol = n, dim, box, tol
        self.energies: list[float] = []
        self.fingerprints: list[np.ndarray] = []
        self.configs: list[np.ndarray] = []
        self.counts: list[int] = []
        self.first_seen: list[int] = []

    def add(self, e: float, x: np.ndarray, iter_index: int) -> int:
        fp = sorted_pair_distances(x, self.n, self.dim, self.box)
        for j, ref in enumerate(self.fingerprints):
            if fingerprint_distance(fp, ref) < self.tol:
                self.counts[j] += 1
                if e < self.energies[j]:
                    self.energies[j] = e
                    self.configs[j] = x.copy()
                    self.fingerprints[j] = fp
                return j
        self.energies.append(e)
        self.fingerprints.append(fp)
        self.configs.append(x.copy())
        self.counts.append(1)
        self.first_seen.append(iter_index)
        return len(self.energies) - 1


def run_basinhopping(fn, x0, *, niter: int, stepsize: float, T: float,
                     seed: int, catalog: Catalog, max_energy: float,
                     start_idx: int):
    """Run scipy.optimize.basinhopping from x0; record every accepted
    local minimum into the catalog."""

    rng = np.random.default_rng(seed)

    class _RandomDisplace:
        def __init__(self, step):
            self.step = step

        def __call__(self, x):
            return x + rng.uniform(-self.step, self.step, size=x.shape)

    take_step = _RandomDisplace(stepsize)
    visits = []

    def _callback(x, f, accept):
        if np.isfinite(f) and f < max_energy:
            visits.append((float(f), np.array(x)))
            catalog.add(float(f), np.array(x), start_idx + len(visits))

    minimizer_kwargs = dict(method="L-BFGS-B", jac=True,
                             options=dict(maxiter=500, gtol=1e-7, ftol=1e-12))
    basinhopping(
        fn, x0, niter=niter, T=T, take_step=take_step,
        minimizer_kwargs=minimizer_kwargs, callback=_callback,
        seed=seed,
    )
    return visits


def quench(fn, x0):
    res = minimize(fn, x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500, gtol=1e-7, ftol=1e-12))
    return float(res.fun), res.x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--box-length", type=float, default=2.9)
    parser.add_argument("--cutin", type=float, default=0.8)
    parser.add_argument("--n-chains", type=int, default=20,
                        help="number of independent basin-hopping chains")
    parser.add_argument("--niter-per-chain", type=int, default=100,
                        help="basin-hopping steps per chain")
    parser.add_argument("--stepsize", type=float, default=0.4,
                        help="perturbation step (physical units)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="basin-hopping Metropolis temperature")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedup-tol", type=float, default=5e-3)
    parser.add_argument("--max-energy", type=float, default=100.0)
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "output" / "lj_minima.npz"))
    args = parser.parse_args()

    system = make_system(args.n_particles, args.box_length, args.cutin)
    n_dofs = args.n_particles * 2
    box = args.box_length
    fn = energy_and_grad_factory(system)

    rng = np.random.default_rng(args.seed)
    catalog = Catalog(args.n_particles, 2, box, args.dedup_tol)

    # FCC reference (almost certainly a saddle, but useful seed)
    x_fcc = system.init_conf(random=False, asNumpy=True).reshape(-1).astype(np.float64)

    # Seed catalog with a quench of FCC+small noise so the lowest-lying
    # minima of the FCC-connected component appear early.
    for k in range(4):
        x_seed = x_fcc + rng.normal(scale=0.02, size=n_dofs)
        e_seed, x_seed_min = quench(fn, x_seed)
        if e_seed < args.max_energy:
            catalog.add(e_seed, x_seed_min, -1)

    visits_count = 0
    t0 = time.time()
    for c in range(args.n_chains):
        # Half the chains start from random uniform, half from a
        # perturbed permutation of FCC. The latter biases the search
        # toward FCC-connected structures; the former toward random
        # packings.
        if c < args.n_chains // 2:
            x0 = rng.uniform(0.0, box, size=n_dofs)
        else:
            x = x_fcc.reshape(args.n_particles, 2).copy()
            perm = rng.permutation(args.n_particles)
            x = x[perm]
            x = x + rng.normal(scale=0.1, size=x.shape)
            x0 = x.reshape(-1)

        visits = run_basinhopping(
            fn, x0,
            niter=args.niter_per_chain, stepsize=args.stepsize,
            T=args.temperature, seed=args.seed + c,
            catalog=catalog, max_energy=args.max_energy,
            start_idx=visits_count,
        )
        visits_count += len(visits)
        elapsed = time.time() - t0
        print(f"  chain {c+1}/{args.n_chains}  visits={len(visits)}  "
              f"unique={len(catalog.energies)}  elapsed={elapsed:.1f}s",
              flush=True)

    # Sort catalog by energy
    order = np.argsort(catalog.energies)
    energies = np.array([catalog.energies[k] for k in order])
    fingerprints = np.stack([catalog.fingerprints[k] for k in order]) \
        if catalog.fingerprints else np.zeros((0, 0))
    configs = np.stack([catalog.configs[k] for k in order]) \
        if catalog.configs else np.zeros((0, n_dofs))
    counts = np.array([catalog.counts[k] for k in order], dtype=int)

    e_fcc, _ = quench(fn, x_fcc + 1e-8 * rng.standard_normal(n_dofs))
    print(f"\nFCC reference quench: E = {e_fcc:.4f}  "
          f"(FCC itself: {fn(x_fcc)[0]:.4f}; gradient norm "
          f"{np.linalg.norm(fn(x_fcc)[1]):.2e})")
    print(f"Found {len(energies)} unique minima.")
    if len(energies) > 0:
        print("Top of catalog (lowest 15):")
        for k, (e, n) in enumerate(zip(energies[:15], counts[:15])):
            print(f"  #{k:2d}: E = {e:+9.4f}   visits {n}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        catalog_energies=energies,
        catalog_fingerprints=fingerprints,
        catalog_configs=configs,
        catalog_counts=counts,
        e_fcc=float(fn(x_fcc)[0]),
        e_fcc_quench=e_fcc,
        x_fcc=x_fcc,
        box_length=box, n_particles=args.n_particles,
        dedup_tol=args.dedup_tol, max_energy=args.max_energy,
        n_chains=args.n_chains,
        niter_per_chain=args.niter_per_chain,
        stepsize=args.stepsize,
        temperature=args.temperature,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
