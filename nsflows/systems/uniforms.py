import numpy as np
import torch

from nsflows.systems.base import base_system
from nsflows.tools.util import sample_spherical


class box_uniform(base_system):

    def __init__(self, n_particles, dimensions, device, box_length=1.):
        super().__init__(n_particles, dimensions, device)

        self.name = "box-uniform"

        self.box_length = box_length
        self.h_box_length = box_length/2


    # The energy is the one of a standard uniform distribution
    def energy(self, x):
        
        return torch.zeros(x.shape[0], device=self.device).unsqueeze(-1)
        

    def init_conf(self):

        return np.random.uniform(low=-self.h_box_length, high=self.h_box_length, size=(self.n_particles, self.dimensions))


    def sample(self, n_samples, beta=1, transform=True):

        x = (torch.rand((n_samples, self.dofs), device=self.device, dtype=torch.float32) - .5) * self.box_length

        if transform:
            # transform in internal coordinates (center particle 0)
            x[:,0:self.dimensions] = torch.zeros(self.dimensions, device=self.device, dtype=torch.float32)

        return x