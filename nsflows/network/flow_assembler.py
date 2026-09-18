import numpy as np
import torch
from torch import nn

from nsflows.network.base import base_generator
from nsflows.tools.util import mean_per_condition

class flow_assembler(base_generator):

    def __init__(self, prior, posterior, blocks, prior_sided_transformation_layers=[], post_sided_transformation_layers=[], device=None, init_zeros=False, k = 1., j = 1.):
        super(flow_assembler, self).__init__(prior, posterior, device)

        if device is None:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.prior_sided_transformation_layers = nn.ModuleList(prior_sided_transformation_layers)
        self.posterior_sided_transformation_layers = nn.ModuleList(post_sided_transformation_layers)

        self.n_prior_sided_tl = len(prior_sided_transformation_layers)
        self.n_post_sided_tl = len(post_sided_transformation_layers)

        self.blocks = nn.ModuleList(blocks)
        
        self.init_zeros = init_zeros
        self.apply(self.init_weights)

        self.k = k
        self.j = j

    def init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            if self.init_zeros:
                nn.init.constant_(module.weight, 0.0)
            else:
                module.weight.data.normal_(mean=0.0, std=0.01)

            if module.bias is not None:
                if self.init_zeros:
                    nn.init.constant_(module.bias, 0.0)
                else:
                    module.bias.data.normal_(mean=0.0, std=0.01)


    def F_xz(self, x, condition=None):

        z, logdetJ_xz = x.clone(), x.new_zeros(x.shape[0], 1)

        for t in range(self.n_post_sided_tl):

            z, part_logdetJ_data2network = self.posterior_sided_transformation_layers[t].F_data2network(z)
            logdetJ_xz += part_logdetJ_data2network

        for block in reversed(self.blocks):

            z, part_logdetJ_xz = block(z, inverse=True, condition=condition)
            logdetJ_xz += part_logdetJ_xz

        for t in reversed(range(self.n_prior_sided_tl)):

            z, part_logdetJ_network2data = self.prior_sided_transformation_layers[t].F_network2data(z)
            logdetJ_xz += part_logdetJ_network2data
        
        return z, logdetJ_xz
    
    def F_zx(self, z, condition=None):

        x, logdetJ_zx = z.clone(), z.new_zeros(z.shape[0], 1)

        for t in range(self.n_prior_sided_tl):

            x, part_logdetJ_data2network = self.prior_sided_transformation_layers[t].F_data2network(x)
            logdetJ_zx += part_logdetJ_data2network

        for block in self.blocks:

            x, part_logdetJ_zx = block(x, inverse=False, condition=condition)
            logdetJ_zx += part_logdetJ_zx

        for t in reversed(range(self.n_post_sided_tl)):

            x, part_logdetJ_network2data = self.posterior_sided_transformation_layers[t].F_network2data(x)
            logdetJ_zx += part_logdetJ_network2data

        return x, logdetJ_zx
    

    def loss_xz(self, x, condition_x=None):
        
        z, log_detJ_xz = self.F_xz(x, condition_x)

        logp_xz = -self.prior.energy(z)
        logw_xz = None
        
        if condition_x is not None:
            loss = mean_per_condition(-(logp_xz + log_detJ_xz), condition_x).mean()
        else:
            loss = -(logp_xz + log_detJ_xz).mean()
        
        return loss, logw_xz

    def loss_zx(self, z, U_max, condition_z=None, energy_z=None):

        x, log_detJ_zx = self.F_zx(z, condition_z)
        
        energy_x = self.posterior.energy(x)
        dE = energy_x - U_max

        # pen_zx = -torch.where(dE > 0, 1/2*torch.square(dE), torch.zeros(1, dtype=torch.float32, device=self.device))
        pen_zx = -torch.where(dE > 0, dE, torch.zeros(1, dtype=torch.float32, device=self.device))
        # pen_zx = .1 * torch.log(torch.sigmoid(-dE / .1) + 1e-12)

        if energy_z is not None:
            logp_zx = -torch.where(dE < 0, torch.zeros(1, device=self.device), torch.inf*torch.ones(1, device=self.device))
            logp_z  = -energy_z
            logw_zx = (logp_zx - logp_z + log_detJ_zx).squeeze(-1)
        else:
            logw_zx = None

        if condition_z is not None:
            loss = mean_per_condition(-(self.k*pen_zx + self.j*log_detJ_zx), condition_z).mean()
            avg_logp_zx = mean_per_condition((-self.k*pen_zx), condition_z).mean()
        else:
            loss = -(self.k*pen_zx + self.j*log_detJ_zx).mean()
            avg_logp_zx = (-self.k*pen_zx).mean()

        return loss, logw_zx, avg_logp_zx