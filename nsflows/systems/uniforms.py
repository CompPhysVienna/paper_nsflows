import numpy as np
import torch

from nsflows.systems.base import base_system
from nsflows.tools.util import sample_spherical


class box_uniform(base_system):

    def __init__(self, n_particles, dimensions, device, box_length=1., aspect_ratio=None):
        super().__init__(n_particles, dimensions, device)
        """
        - box_length (float/sequence): edge length of the cell. A scalar gives a cube,
            a sequence of length `dimensions` gives one length per axis.
        - aspect_ratio (float/None): with a scalar box_length, L_y / L_x of an
            orthorhombic 2D cell, matching lennard_jones(aspect_ratio=...).
        """

        self.name = "box-uniform"

        # box_length is kept per axis, so the prior matches an orthorhombic cell.
        # A scalar stays equivalent to the isotropic case it replaces.
        if np.isscalar(box_length):
            if aspect_ratio is None:
                lengths = [float(box_length)] * dimensions
            else:
                assert dimensions == 2, "aspect_ratio is only supported in two dimensions"
                lengths = [float(box_length), float(box_length) * float(aspect_ratio)]
        else:
            lengths = [float(l) for l in box_length]
            assert len(lengths) == dimensions, "box_length must have one entry per dimension"

        self.box_length = torch.tensor(lengths, device=device, dtype=torch.float32)
        self.h_box_length = self.box_length / 2


    # The energy is the one of a standard uniform distribution
    def energy(self, x):
        
        return torch.zeros(x.shape[0], device=self.device).unsqueeze(-1)
        

    def init_conf(self):

        u = np.random.uniform(size=(self.n_particles, self.dimensions))

        return u * self.box_length.cpu().numpy() - self.h_box_length.cpu().numpy()


    def sample(self, n_samples, beta=1, transform=True):

        x = (torch.rand((n_samples, self.n_particles, self.dimensions),
                        device=self.device, dtype=torch.float32) - .5) * self.box_length
        x = x.reshape(n_samples, self.dofs)

        if transform:
            # transform in internal coordinates (center particle 0)
            x[:,0:self.dimensions] = torch.zeros(self.dimensions, device=self.device, dtype=torch.float32)

        return x