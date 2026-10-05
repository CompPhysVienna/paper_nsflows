"""The orthorhombic cell used for the triangular-lattice runs.

A cell whose sides differ changes three things: the box is no longer set by the
lattice spacing, the point group loses its axis permutations, and a displacement
that is a sensible fraction of one side is not a sensible fraction of the other.
"""

import numpy as np
import pytest
import torch

from nsflows.samplers.monte_carlo import (
                                          rejection_monte_carlo,
                                          resolve_step_size)
from nsflows.systems.lennard_jones import lennard_jones
from nsflows.systems.uniforms import box_uniform
from nsflows.tools.util import octahedral_transformation

DEVICE = torch.device("cpu")
ASPECT = 2 / np.sqrt(3)          # triangular lattice in a rectangular cell


def _system(n_particles=9, rho=0.95, aspect_ratio=ASPECT):
    return lennard_jones(n_particles=n_particles, dimensions=2, rho=rho,
                         device=DEVICE, cutin=0.8, lrc=True,
                         aspect_ratio=aspect_ratio)


# --------------------------------------------------------------------------
# the cell
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_particles,rho", [(9, 0.95), (8, 0.95), (16, 0.7), (5, 0.4)])
def test_density_is_exact_in_both_cells(n_particles, rho):
    """The square path rescales the lattice box to hit rho; the orthorhombic path
    computes the box from rho directly. Both must land on it exactly.

    The square path is the one that regressed when the branch was merged: without
    the rescaling every cell that is not an exact lattice fill comes out at the
    wrong density.
    """
    for aspect_ratio in (None, ASPECT):
        s = _system(n_particles, rho, aspect_ratio)
        assert abs(n_particles / float(s.volume) - rho) < 1e-6, (
            f"N={n_particles} rho={rho} aspect_ratio={aspect_ratio}: "
            f"N/V = {n_particles / float(s.volume)}")


def test_cell_has_the_requested_aspect_ratio():
    s = _system()
    lx, ly = s.box_length.tolist()
    assert abs(ly / lx - ASPECT) < 1e-5, (lx, ly)
    assert s.orthorhombic_cell is True


def test_square_cell_is_not_flagged_orthorhombic():
    assert _system(aspect_ratio=None).orthorhombic_cell is False


def test_cutoff_respects_the_shorter_side():
    """The minimum-image convention is set by the smallest side, not the largest."""
    s = _system()
    assert s.cutoff < 0.5 * min(s.box_length.tolist())


# --------------------------------------------------------------------------
# the point group
# --------------------------------------------------------------------------

def _group(orthorhombic, n=400):
    torch.manual_seed(0)
    ops = torch.stack([octahedral_transformation(2, DEVICE, orthorhombic)[0]
                       for _ in range(n)])
    return torch.unique(ops, dim=0)


def test_orthorhombic_cell_drops_the_axis_permutations():
    """Swapping axes of different length does not map the cell onto itself, so the
    group is D2 rather than D4."""
    ortho, square = _group(True), _group(False)
    assert ortho.shape[0] == 4, ortho.shape[0]
    assert square.shape[0] == 8, square.shape[0]
    # every orthorhombic operation is diagonal, i.e. reflections only
    off_diagonal = ortho - ortho * torch.eye(2)
    assert float(off_diagonal.abs().max()) == 0.0
    # the square group is not
    assert float((square - square * torch.eye(2)).abs().max()) > 0.0


# --------------------------------------------------------------------------
# the step size
# --------------------------------------------------------------------------

def test_scalar_step_becomes_the_same_fraction_of_each_side():
    s = _system()
    step = resolve_step_size(s, 1.2, 2, DEVICE)
    box = s.box_length
    assert step.shape == (2,)
    # equal fraction of each side, normalised on the longest
    fractions = (step / box).tolist()
    assert abs(fractions[0] - fractions[1]) < 1e-6, fractions
    assert abs(float(step.max()) - 1.2) < 1e-6


def test_scalar_step_is_unchanged_in_a_square_cell():
    s = _system(n_particles=8, aspect_ratio=None)
    step = resolve_step_size(s, 1.2, 2, DEVICE)
    assert torch.allclose(step, torch.full((2,), 1.2), atol=1e-6), step



def test_explicit_vector_step_is_taken_verbatim():
    s = _system()
    step = resolve_step_size(s, [0.3, 0.7], 2, DEVICE)
    assert torch.allclose(step, torch.tensor([0.3, 0.7]))
    with pytest.raises(ValueError):
        resolve_step_size(s, [0.3, 0.7, 0.1], 2, DEVICE)



def test_walkers_stay_inside_the_orthorhombic_cell():
    s = _system()
    torch.manual_seed(0)
    x = torch.stack([s.init_conf(random=True) for _ in range(32)])
    prop = rejection_monte_carlo(system=s, n_cycles=20, step_size=0.5, transform=False)
    prop.x0 = x
    out, _, _ = prop.sample_space(N=32, energy_bound=1e6)

    half = s.box_length / 2
    pos = out.view(-1, s.n_particles, 2)
    assert bool((pos.abs() <= half + 1e-4).all()), (
        f"a walker left the cell: max |x| = {pos.abs().amax(dim=(0, 1)).tolist()}, "
        f"half box = {half.tolist()}")


def test_base_distribution_fills_the_orthorhombic_cell():
    s = _system()
    prior = box_uniform(n_particles=9, dimensions=2, device=DEVICE,
                        box_length=float(s.box_length[0]), aspect_ratio=ASPECT)
    torch.manual_seed(0)
    z = prior.sample(4000, transform=False).view(-1, 9, 2)
    lo, hi = z.amin(dim=(0, 1)), z.amax(dim=(0, 1))
    half = s.box_length / 2
    assert bool((lo >= -half - 1e-3).all()) and bool((hi <= half + 1e-3).all())
    # and it really is wider in y than in x
    spread = (hi - lo)
    assert spread[1] > spread[0], spread.tolist()
    assert abs(float(spread[1] / spread[0]) - ASPECT) < 0.05, float(spread[1] / spread[0])
