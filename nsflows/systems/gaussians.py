import numpy as np
import torch

from nsflows.systems.base import base_system
from nsflows.tools.util import sample_spherical, truncated_chi

class normal(base_system):

    def __init__(self, n_particles, dimensions, device):
        super().__init__(n_particles, dimensions, device)

        self.name = "normal"
                

    # The energy is the one of a standard normal distribution
    def energy(self, x):
        
        return 1/2*(x[:,0]**2 + x[:,1]**2).unsqueeze(-1)


    def init_conf(self, lower_bound, upper_bound):
        raise NotImplementedError


    def sample(self, n_samples, transform=True):

        x = torch.randn(n_samples, self.dofs, device=self.device)

        if transform:
            # transform in internal coordinates (center particle 0)
            x[:,0:self.dimensions] = torch.zeros(self.dimensions, device=self.device, dtype=torch.float32)

        return x