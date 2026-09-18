import torch

from nsflows.transformations.base import base_transform

class remove_origin(base_transform):
    
    def __init__(self, n_particles, dimensions, device):
        super(remove_origin, self).__init__(device)
        
        self._name = "remove_origin"

        self.n_particles = n_particles
        self.dimensions = dimensions
                
            
    def F_data2network(self, data, return_Jacobian=True):
        
        # print(f"Applying {self._name} data 2 network")

        datap = data.view(-1, self.n_particles, self.dimensions)[:,1:,:]
        data_transformed = datap.view(-1, (self.n_particles-1)*self.dimensions)
        
        if return_Jacobian:
            log_det_data2network = data_transformed.new_zeros(data.shape[0], 1)
        else:
            log_det_data2network = None
            
        return data_transformed, log_det_data2network
    
    
    def F_network2data(self, sample, return_Jacobian=True):
        
        # print(f"Applying {self._name} network 2 data")

        zeros = torch.zeros((sample.shape[0], self.dimensions), device=self.device)
        sample_transformed = torch.cat((zeros, sample.clone()), dim = -1)
        
        if return_Jacobian:
            log_det_network2data = sample_transformed.new_zeros(sample.shape[0], 1)
        else:
            log_det_network2data = None
            
        return sample_transformed, log_det_network2data