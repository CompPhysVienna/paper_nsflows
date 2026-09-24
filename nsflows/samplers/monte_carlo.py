import numpy as np
import torch

from nsflows.samplers.base import base_sampler

class rejection_monte_carlo(base_sampler):
    
    """
    Implements the Rejection Monte Carlo sampling algorithm.

    Parameters:
    - system: Object representing the physical system, providing energy computation and configuration details.
    - step_size: Maximum step size for particle movement.
    - n_cycles: Number of Metropolis sampling cycles to perform.
    - transform: Whether to transform sampled coordinates into internal coordinates (center particle 0).
    """

    def __init__(self, system, n_cycles, step_size, transform=True):
        super(rejection_monte_carlo, self).__init__(system, n_cycles, step_size)

        self.system = system
        self.dimensions = system.dimensions  # Number of spatial dimensions.
        self.n_particles = system.n_particles  # Number of particles in the system.
        self.dofs = system.dofs  # Degrees of freedom (n_particles * dimensions).
        self.device = system.device  # Torch device (CPU/GPU).

        # Maximum displacement per step, kept per axis. A scalar is scaled by
        # L_alpha / max(L), so in an orthorhombic cell each axis is explored in
        # proportion to its length; in a square cell every factor is 1 and the
        # behaviour is unchanged.
        if np.isscalar(step_size):
            self.step_size = (float(step_size) * self.system.box_length
                              / torch.max(self.system.box_length))
        else:
            self.step_size = torch.as_tensor(step_size, dtype=torch.float32, device=self.device)
            assert self.step_size.shape == (self.dimensions,), \
                f"step_size must be a scalar or have shape ({self.dimensions},)"
        self.n_cycles = n_cycles  # Number of cycles for sampling.
        
        self.x0 = None  # Stores the initial configuration.

        self.transform = transform  # Enable/disable transformation into internal coordinates.


    def rejection_cycle(self, x, u_x, energy_bound, dx):
        """
        Perform one cycle of the Rejection algorithm.

        Parameters:
        - x: Current configurations of the system, shape (n_samples, dofs).
        - u_x: Current energies of the configurations, shape (n_samples,).
        - dx: Maximum step size for displacements.
        - energy_bound: Energy above which to reject the move.

        Returns:
        - x: Updated configurations.
        - u_x: Updated energies.
        - acc: Fraction of accepted moves.
        """

        n_samples = x.shape[0]

        # ---- Propose random displacements ----
        shift = torch.zeros((n_samples, self.dofs), device=self.device)

        # pick one random particle per sample
        selected_particles = torch.randint(self.n_particles, size=(n_samples,), device=self.device)

        rows = torch.arange(n_samples, device=self.device).repeat_interleave(self.dimensions)
        cols = (
            selected_particles.unsqueeze(1) * self.dimensions
            + torch.arange(self.dimensions, device=self.device)
        ).reshape(-1)

        rand_disp = (torch.rand((n_samples, self.dimensions), device=self.device) * 2 - 1) * dx
        shift[rows, cols] = rand_disp.reshape(-1)

        xp = x + shift  # proposed configs, still (n_samples, dofs)

        # ---- Compute proposed energies ----
        u_xp = self.system.energy(xp).squeeze(axis=1)

        # ---- Accept/reject ----
        mask = (u_xp < energy_bound)

        if mask.any():
            xp_mask = xp[mask].reshape(-1, self.n_particles, self.dimensions)

            # Apply periodic boundary conditions
            if self.system.PBC:
                xp_mask -= self.system.box_length * torch.round(
                    xp_mask / self.system.box_length
                )

            # Flatten back to dofs and update accepted configs
            x[mask] = xp_mask.reshape(-1, self.dofs)
            u_x[mask] = u_xp[mask]

        # ---- Acceptance ratio ----
        acc = mask.float().mean()

        return x, u_x, acc
    

    def sample_space(self, N, energy_bound):
        """
        Generate samples using the Metropolis Monte Carlo algorithm.

        Parameters:
        - N: Number of configurations to sample.
        - energy_bound: Energy above which to reject the move.

        Returns:
        - x: Final configurations after sampling.
        - u_x: Energies of the sampled configurations.
        - tot_acc: Fraction of accepted moves during sampling.
        """
        # Initialize configurations (either random or from the last sampled state).
        if self.x0 is None:
            x = torch.stack([self.system.init_conf() for i in range(N)])
        else:
            indx = np.random.choice(np.arange(0, self.x0.shape[0]), replace=(N > self.x0.shape[0]), size=N)
            x = self.x0[indx]

        # Compute initial energies.
        u_x = self.system.energy(x).squeeze(axis=1)

        # Sampling phase.
        avg_acc = 0
        for cycle in range(self.n_cycles):
            x, u_x, acc = self.rejection_cycle(x, u_x, energy_bound, self.step_size)
            avg_acc += acc
        
        avg_acc /= self.n_cycles

        # Optionally transform configurations into internal coordinates.
        if self.transform:
            xp = x.view(-1, self.n_particles, self.dimensions)
            c = xp[:, 0].clone()  # Center coordinates of particle 0.
            xp -= c.unsqueeze(1)
            if self.system.PBC:
                xp -= self.system.box_length * torch.round(xp / self.system.box_length)
            x = xp.view(-1, self.dofs)

        # Store the last sampled state for future use.
        self.x0 = x.clone()

        return x, u_x, avg_acc
