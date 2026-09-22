"""Probe the 8-particle 2D Lennard-Jones energy along KNOWN symmetry
directions in configuration space.

A random direction through 16-D LJ space (probe in experiments/posterior)
showed only one basin because the model's hidden multimodality comes from
discrete symmetries — particle relabelings and PBC translations — that
are almost surely orthogonal to a random vector. Here we walk along
those symmetry directions explicitly.

Probes:
* **PBC translation** — shift every particle by ``t * delta`` with delta
  a fixed 2D direction, wrap into the box. Energy must be exactly flat;
  this is a sanity check.
* **1D swap interpolation** — straight line ``x(t) = (1-t) x0 + t P x0``
  where ``P`` swaps particles ``i`` and ``j``. Endpoints are the FCC
  ground state; the midpoint puts both particles on top of each other
  and produces a huge collision barrier.
* **2D two-swap grid** — independent linear interpolations along two
  disjoint swaps as the two axes. Expected: four equivalent FCC basins
  at the corners of the unit square, with high-energy barriers between.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _lj_compat  # noqa: F401  # patches lennard_jones.energy self-mask
from nsflows.systems.lennard_jones import lennard_jones


def build_lj(n_particles: int = 8, box_length: float = 2.9, cutin: float = 0.8):
    device = torch.device("cpu")
    rho = n_particles / box_length ** 2
    return lennard_jones(
        n_particles=n_particles, dimensions=2, rho=rho,
        device=device, cutin=cutin, lrc=True,
    ), device


def energy_batch(system, configs: np.ndarray) -> np.ndarray:
    """configs: shape (B, N, 2) in physical units. Returns (B,)."""
    x = torch.from_numpy(
        configs.reshape(configs.shape[0], -1).astype(np.float32)
    ).to(system.device)
    with torch.no_grad():
        e = system.energy(x).cpu().numpy().reshape(-1)
    return e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--box-length", type=float, default=2.9)
    parser.add_argument("--cutin", type=float, default=0.8)
    parser.add_argument("--n-points-1d", type=int, default=401)
    parser.add_argument("--n-points-2d", type=int, default=121)
    parser.add_argument("--swap-ij", type=int, nargs=2, default=(0, 1),
                        help="indices of the two particles to swap (1D probe)")
    parser.add_argument("--swap-2d", type=int, nargs=4,
                        default=(0, 1, 2, 3),
                        help="two disjoint pairs (i1 j1 i2 j2) for the 2D probe")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "output" / "lj_symmetries.npz"))
    args = parser.parse_args()

    system, device = build_lj(args.n_particles, args.box_length, args.cutin)
    box = args.box_length

    x0_flat = system.init_conf(random=False, asNumpy=True).reshape(-1)
    N = args.n_particles
    x0 = x0_flat.reshape(N, 2)
    e0 = float(energy_batch(system, x0[None])[0])

    # ------------------------------------------------------------------
    # 1) PBC translation in a fixed direction
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    direction_2d = rng.standard_normal(2)
    direction_2d /= np.linalg.norm(direction_2d)
    # Sweep t over one full period in the box (|t| up to box_length).
    ts_trans = np.linspace(-box, box, args.n_points_1d)
    configs_trans = np.empty((args.n_points_1d, N, 2))
    for k, t in enumerate(ts_trans):
        shifted = x0 + t * direction_2d
        shifted = shifted - box * np.floor(shifted / box)  # PBC wrap
        configs_trans[k] = shifted
    e_trans = energy_batch(system, configs_trans)

    # ------------------------------------------------------------------
    # 2) 1D swap interpolation
    # ------------------------------------------------------------------
    i, j = args.swap_ij
    x_swap = x0.copy()
    x_swap[[i, j]] = x_swap[[j, i]]

    # Apply the minimum-image convention to the displacement vector
    # (particles travel along the shortest periodic path).
    def min_image_diff(a, b, box_len):
        d = b - a
        d -= box_len * np.round(d / box_len)
        return d

    delta = min_image_diff(x0, x_swap, box)
    ts_swap = np.linspace(-0.2, 1.2, args.n_points_1d)  # show overshoot
    configs_swap = np.empty((args.n_points_1d, N, 2))
    for k, t in enumerate(ts_swap):
        cfg = x0 + t * delta
        cfg = cfg - box * np.floor(cfg / box)
        configs_swap[k] = cfg
    e_swap = energy_batch(system, configs_swap)

    # ------------------------------------------------------------------
    # 3) 2D two-swap grid: independent swap of (i1,j1) on axis-1 and
    #    (i2,j2) on axis-2.
    # ------------------------------------------------------------------
    i1, j1, i2, j2 = args.swap_2d
    x_swap1 = x0.copy(); x_swap1[[i1, j1]] = x_swap1[[j1, i1]]
    x_swap2 = x0.copy(); x_swap2[[i2, j2]] = x_swap2[[j2, i2]]
    delta1 = min_image_diff(x0, x_swap1, box)
    delta2 = min_image_diff(x0, x_swap2, box)

    g = np.linspace(-0.2, 1.2, args.n_points_2d)
    T1, T2 = np.meshgrid(g, g, indexing="ij")
    M = args.n_points_2d
    configs_2d = np.empty((M * M, N, 2))
    for ii in range(M):
        for jj in range(M):
            cfg = x0 + T1[ii, jj] * delta1 + T2[ii, jj] * delta2
            cfg = cfg - box * np.floor(cfg / box)
            configs_2d[ii * M + jj] = cfg
    e_2d = energy_batch(system, configs_2d).reshape(M, M)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ts_trans=ts_trans, e_trans=e_trans, direction_2d=direction_2d,
        ts_swap=ts_swap, e_swap=e_swap, swap_ij=np.array([i, j]),
        s_2d=g, e_2d=e_2d, swap_2d=np.array([i1, j1, i2, j2]),
        e0=e0, x0=x0, box_length=box,
        n_particles=N,
    )
    print(f"Wrote {args.output}")
    print(f"  FCC energy: {e0:.4f}")
    print(f"  Translation: range=[{e_trans.min():.3g}, {e_trans.max():.3g}] "
          f"(flat if exactly = {e0:.4f}; max deviation "
          f"{np.abs(e_trans - e0).max():.2e})")
    print(f"  Swap 1D: range=[{e_swap.min():.3g}, {e_swap.max():.3g}]")
    print(f"  Two-swap 2D: range=[{e_2d.min():.3g}, {e_2d.max():.3g}]")


if __name__ == "__main__":
    main()
