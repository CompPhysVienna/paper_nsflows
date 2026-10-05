"""The repulsive core must stay repulsive all the way to zero separation."""

import torch

from nsflows.systems.lennard_jones import lennard_jones


def _system(n_particles=8, rho=0.95):
    return lennard_jones(n_particles=n_particles, dimensions=2, rho=rho,
                         device=torch.device("cpu"), cutin=0.8, lrc=True)


def test_coincident_particles_are_not_free():
    """Self-pairs were excluded by distance rather than by index, so two *distinct*
    particles closer than sqrt(tol) contributed nothing. Coincident particles then
    came out below the well depth: the strongest repulsion in the system read as
    the most attractive configuration a sampler could find."""
    s = _system()
    base = s.init_conf(random=False).view(8, 2).clone()

    def energy_at(separation):
        x = base.clone()
        x[1] = x[0] + torch.tensor([separation, 0.0])
        return float(s.energy(x.reshape(1, -1)))

    reference = energy_at(1e-5)          # just outside the old exclusion radius
    for separation in (1e-6, 1e-7, 1e-9, 0.0):
        e = energy_at(separation)
        assert e > 0.0, f"separation {separation:g} gives energy {e:.3f}"
        assert abs(e - reference) < 1.0, (
            f"energy jumps from {reference:.3f} at 1e-5 to {e:.3f} at {separation:g}")


def test_core_energy_is_monotonic_approaching_contact():
    s = _system()
    base = s.init_conf(random=False).view(8, 2).clone()
    seps = [0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0]
    energies = []
    for d in seps:
        x = base.clone()
        x[1] = x[0] + torch.tensor([d, 0.0])
        energies.append(float(s.energy(x.reshape(1, -1))))
    for a, b, da, db in zip(energies, energies[1:], seps, seps[1:]):
        assert b >= a - 1e-3, (
            f"energy falls from {a:.3f} at separation {da:g} to {b:.3f} at {db:g}")


def test_self_pairs_contribute_nothing():
    """A single particle has no pairs at all, so only the tail correction is left.

    Low density, so that the one-particle box is still wide enough for the cutoff.
    """
    s = _system(n_particles=1, rho=0.05)
    x = torch.zeros(1, 2)
    assert abs(float(s.energy(x)) - s.etail) < 1e-6
