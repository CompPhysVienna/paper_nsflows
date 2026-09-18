import numpy as np
import torch

from nsflows.systems.base import base_system

class double_well(base_system):
    
    def __init__(self, n_particles, dimensions, device, eps=1., c=1., d=1.):
        super().__init__(n_particles, dimensions, device)

        self.eps = eps
        self.c = c
        self.d = d

        self.PBC = False


    def energy(self, x):

        return self.eps*(self.c*(x[:,0]**2 - 1)**2 + (x[:,0]-x[:,1])**2 + self.d*(x[:,0]+x[:,1])).unsqueeze(-1)


    def init_conf(self, lower_bounds=[-2.5,-5], upper_bounds=[2.5, 5], random=True, asNumpy=False):
        
        assert random, "Initial configuration for test systems can only be random"

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        if random:
            rndm = np.random.rand(self.n_particles, self.dimensions)
            conf = lower_bounds + (upper_bounds - lower_bounds)*rndm

        if asNumpy:
            return conf
        else:
            return torch.from_numpy(((conf).ravel()).astype(np.float32)).to(self.device)