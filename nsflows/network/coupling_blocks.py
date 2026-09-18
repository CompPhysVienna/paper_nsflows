import torch
import warnings
from torch import nn
from einops import rearrange

from nsflows.network.splines import rational_quadratic_spline
from nsflows.tools.util import get_target_indices
from nsflows.network.coupling_networks import EquivariantTransformer
from nsflows.network.base import InputOutsideDomain

class EquivariantRQS(nn.Module):

    def __init__(self, target_coordinates, n_particles, dimensions, device, n_bins=8, left=-1, right=1, bottom=-1, top=1, conditioned=False):
        super(EquivariantRQS, self).__init__()

        self.n_particles = n_particles
        self.dimensions = dimensions

        self.n_bins = n_bins
        self.left   = left
        self.right  = right
        self.bottom = bottom
        self.top    = top

        in_dim, out_dim = dimensions-len(target_coordinates), len(target_coordinates)
        identity_indices, transformed_indices = get_target_indices(target_coordinates, n_particles, dimensions)

        self.identity_indices = identity_indices
        self.transformed_indices = transformed_indices

        # Choosing the network to get rational quadratic spline parameters
        self.network = EquivariantTransformer(in_dim, out_dim * (3 * self.n_bins), device=device, conditioned=conditioned)

    def forward(self, x, inverse=False, condition=None):

        # Split input
        x_identity = x[:, self.identity_indices]
        x_transformed = x[:, self.transformed_indices]

        # Rational Quadratic splines parameter via equivariant transformer
        parameters = self.network(x_identity, condition) # Parameters of the transformation are function of the untransformed input
        parameters = rearrange(parameters, "b (d p) -> b d p", d = len(self.transformed_indices)) # p = (widths, heights, slopes) * n_bins, d = n_particles

        widths = parameters[:, :, :self.n_bins]
        heights = parameters[:, :, self.n_bins:2*self.n_bins]
        slopes = parameters[:, :, 2*self.n_bins:]
        # Make spline periodic
        slopes = torch.cat([slopes, slopes[..., [0]]], dim=-1)

        # Part of input transformed through a function of the untransformed input xt = (f_xi)(xt)
        try:
            x_transformed, part_log_det = rational_quadratic_spline(
                                            x_transformed, 
                                            widths, heights, slopes, 
                                            inverse=inverse, 
                                            left=self.left, right=self.right,
                                            bottom=self.bottom, top=self.top,
                                            enable_identity_init=True)
        except InputOutsideDomain:
            exceeded_left = (x_transformed - self.left).min()
            exceeded_right = (x_transformed - self.right).max()
            warnings.warn(
                f"InputOutsideDomain: min {self.left} - {exceeded_left.item()}; "
                f"max {self.right} + {exceeded_right.item()}",
                UserWarning
            )
            x_transformed, part_log_det = rational_quadratic_spline(
                                            x_transformed.clamp(self.left, self.right), 
                                            widths, heights, slopes, 
                                            inverse=inverse, 
                                            left=self.left, right=self.right,
                                            bottom=self.bottom, top=self.top,
                                            enable_identity_init=True)

        x[:, self.transformed_indices] = x_transformed

        return x, part_log_det.sum(1, keepdim=True)