import torch
from torch import nn

from nsflows.transformations.base import base_transform

class NormalizeBox(base_transform):
    
    def __init__(self, n_particles, dimensions, box_length, device):
        super(NormalizeBox, self).__init__(device)
        
        self._name = "normalize_box"

        self.n_particles = n_particles
        self.dimensions = dimensions
        
        self.box_length = box_length


    def F_data2network(self, data, return_Jacobian=True):

        # print(f"Applying {self._name} data 2 network")

        # Normalization: data is in [-L/2,L/2], network input is in [-1,1]
        datap = data.view(-1, self.n_particles, self.dimensions)
        data_transformed = (datap * 2 / self.box_length).view(-1, self.n_particles*self.dimensions)

        if return_Jacobian:
            log_det_data2network = (torch.log(2 / self.box_length)).repeat(data.shape[0], 1).sum(dim=-1, keepdims=True)
        else:
            log_det_data2network = None

        return data_transformed, log_det_data2network


    def F_network2data(self, sample, return_Jacobian=True):

        # print(f"Applying {self._name} network 2 data")

        # Inverse normalization: network output is in [-1,1], data is in [-L/2, L/2]
        samplep = sample.view(-1, self.n_particles, self.dimensions)
        sample_transformed = (samplep * self.box_length / 2).view(-1, self.n_particles*self.dimensions)

        if return_Jacobian:
            log_det_network2data = (torch.log(self.box_length / 2)).repeat(sample.shape[0], 1).sum(dim=-1, keepdims=True)
        else:
            log_det_network2data = None

        return sample_transformed, log_det_network2data