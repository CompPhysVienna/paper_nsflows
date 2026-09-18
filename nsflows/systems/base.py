import torch.nn as nn

class base_system(nn.Module):

    def __init__(self, n_particles, dimensions, device):
        super(base_system, self).__init__() 
        
        self.device = device

        self.n_particles = n_particles
        self.dimensions = dimensions
        self.dofs = self.n_particles*self.dimensions

        self.PBC = False
        
    def energy(self, x):
        raise NotImplementedError

    def init_conf(self):
        raise NotImplementedError
    
    def sample(self, N, beta=1):
        raise NotImplementedError