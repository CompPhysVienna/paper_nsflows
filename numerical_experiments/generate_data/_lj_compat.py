"""Local patch for nsflows.systems.lennard_jones.lennard_jones.energy.

Importing this module replaces ``lennard_jones.energy`` with a version that
masks self-interactions by particle index instead of by ``r < tol``. The
original logic also zeroed out *genuine* near-overlaps between distinct
particles, producing an artificial dip in the energy whenever a probe
walked through a coincident configuration.

Opt in by adding ``import _lj_compat  # noqa: F401`` at the top of any
experiment script before instantiating ``lennard_jones``. Idempotent.
"""
import torch

from nsflows.systems.lennard_jones import lennard_jones


def _energy_fixed(self, x):
    pos = x.view((-1, self.n_particles, self.dimensions))

    rij = pos[:, :, None, :] - pos[:, None, :, :]
    rij -= self.box_length * torch.round(rij / self.box_length)
    r2 = torch.clamp(torch.sum(rij ** 2, dim=-1).unsqueeze(-1), min=self.tol)
    r6 = torch.clamp(r2 ** 3, min=self.tol)
    r12 = torch.clamp(r6 ** 2, min=self.tol)

    lj_energy = 4 * self.epsilon * (
        self.sigma ** 12 / r12 - self.sigma ** 6 / r6
    ) - self.ecutoff

    mask = getattr(self, "_self_mask", None)
    if mask is None or mask.shape[1] != self.n_particles or mask.device != self.device:
        mask = torch.eye(
            self.n_particles, dtype=torch.bool, device=self.device,
        )[None, :, :, None]
        self._self_mask = mask

    cond_cutoff = (r2 < self.cutoff_sq) & ~mask
    e_part = torch.where(cond_cutoff, lj_energy, torch.zeros_like(lj_energy))

    if self.cutin is not None:
        r = torch.sqrt(r2)
        lin_energy = self.slope * (r - self.cutin) + self.ecutin
        e_part = torch.where((r < self.cutin) & ~mask, lin_energy, e_part)

    return 0.5 * torch.sum(e_part, dim=(1, 2)) + self.n_particles * self.etail


if not getattr(lennard_jones.energy, "_self_mask_patched", False):
    _energy_fixed._self_mask_patched = True
    lennard_jones.energy = _energy_fixed
