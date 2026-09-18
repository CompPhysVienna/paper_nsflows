import numpy as np
import torch

from nsflows.systems.base import base_system

class lennard_jones(base_system):

    def __init__(self, n_particles, dimensions, rho, device, lattice_type="FCC", epsilon = 1, sigma = 1, cutoff = None, cutin = None, lrc = True, tol = 1.e-12):
        super(lennard_jones, self).__init__(n_particles, dimensions, device)
        """
        Initializes the model function with parameters.

        Parameters:
        - n_particles (int): Number of particles.
        - dimensions (int): Dimensionality of the space.
        - rho (float): Density of the system in reduced units.
        - device: PyTorch device on which the computation will be performed.
        - lattice_type (str): Type of the lattice for initial conditions. 
            Only FCC structure (default) is allowed. If None, particles are initialized randomly.
        - epsilon (float): Lennard-Jones potential epsilon parameter.
        - sigma (float): Lennard-Jones potential sigma parameter.
        - cutoff (float/str/None): Cutoff distance for Lennard-Jones potential. 
            If "wca" then the class generates a WCA potential.
            If None the default value of 2.5 is used in reduced units. 
        - cutin (float): Distance used for linearization at short distance in reduced units.
            Zero raises an exception, use None for standard non-linearized LJ potential.
        - lrc (bool): Computes long-range corrections for energy and pressure.
        - tol (float): Tolerance used to clamp small values of distance.
        """

        assert self.dimensions == 2, "Only two dimensions are supported"

        # Name of the model
        if cutoff == "wca":
            self.name = "wca"
        else:
            self.name = "lennard-jones"

        # Lennard-Jones paramters
        self.epsilon = epsilon
        self.sigma = sigma

        # Periodic boundary conditions, length of the box and density of the system
        self.PBC = True
        self.orthorhombic_cell = False

        n_elem = {"FCC" : 2}
        self.lattice_type = lattice_type
        
        # Computing length of the box from density
        self.rho = rho # reduced units

        self.a = (n_elem[self.lattice_type]/self.rho)**(1/self.dimensions) # reduced units
        n = int(np.ceil((n_particles/n_elem[self.lattice_type])**(1/self.dimensions)))
        L = n*self.a*np.ones(self.dimensions) # reduced units
        V = L.prod()
        s = ((self.n_particles/V)/(n_elem[self.lattice_type]*(n**self.dimensions)/V))**(1/self.dimensions)
        self.box_length = torch.from_numpy((s*L).astype(np.float32)).to(self.device) # reduced units
        self.volume = self.box_length.prod().cpu().numpy()
        
        assert np.abs(self.rho - self.n_particles/self.volume) < np.sqrt(tol), f"Error in computing box length from density: rho = {self.rho}, N/V = {self.n_particles/self.volume}"
        min_box_dimension = min(self.box_length.cpu().numpy())

        # Setting cutoff for interactions and correcting for it when not wca
        self.lr_corrections = lrc
        if cutoff is None:
            self.cutoff = np.min((2.5*self.sigma, 0.49*min_box_dimension)) # reduced units
        elif cutoff == "wca":
            self.lr_corrections = False
            self.cutoff = (2**(1./6.))*self.sigma # reduced units
        else:
            self.cutoff = cutoff # reduced units
        assert (self.cutoff < 0.5*min_box_dimension), "Short range cutoff cannot be more than half of the box length"
        self.cutoff_sq = self.cutoff**2
        
        # Constant value used to shift the energy
        self.ecutoff = 4 * self.epsilon * ( (self.sigma/self.cutoff)**(12) - (self.sigma/self.cutoff)**(6) )

        # Long-range corrections for energy and pressure
        # To be multiplied by the number of particles in energy caluclation (already divided by two in the expressions below)
        if self.lr_corrections:
            self.etail = 1/5*np.pi*self.rho*(self.epsilon*self.sigma**2)*(2*(self.sigma/self.cutoff)**(10) - 5*(self.sigma/self.cutoff)**(4))
            self.ptail = 3/5*np.pi*self.rho*(self.epsilon*self.sigma**2)*(4*(self.sigma/self.cutoff)**(10) - 5*(self.sigma/self.cutoff)**(4))
        else:
            self.etail = 0
            self.ptail = 0

        # Linearization of interaction at short distances
        # This avoids explosions if particles are too close
        self.cutin = cutin
        if self.cutin is not None:
            assert self.cutin < self.cutoff, f"cutin = {self.cutin} must be smaller than cutoff = {self.cutoff}"
            
            # Calculations of constant linearization parameters
            self.slope = -24 * self.epsilon / self.cutin * ( 2 * (self.sigma/self.cutin)**(12) - (self.sigma/self.cutin)**(6) )
            self.ecutin = 4 * self.epsilon * ( (self.sigma/self.cutin)**(12) - (self.sigma/self.cutin)**(6) ) - self.ecutoff

        # Tolerance to avoid numerical instabilities when backpropagating through the model
        self.tol = tol
        self.sqrt_tol = np.sqrt(tol)

        # Set initial condition to None
        self.x0 = None


    def energy(self, x):
        """
        Computes the potential energy of the system.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (B, N*D), where B is the batch size,
                            N is the number of particles, and D is the dimensionality of each particle.

        Returns:
        - e_tot (torch.Tensor): Potential energy of the system, with shape (B, 1).
        """

        # Reshape input tensor to represent particle positions
        pos = x.view((-1, self.n_particles, self.dimensions))

        # Pairwise differences between particles
        rij = pos[:, :, None, :] - pos[:, None, :, :]             
        # Apply periodic boundary conditions
        rij -= self.box_length*torch.round(rij/self.box_length)    
        # Compute square distances between particles
        r2 = torch.clamp(torch.sum(rij**2, dim=-1).unsqueeze(-1), min=self.tol) 
        r6 = torch.clamp(r2**3, min=self.tol)
        r12 = torch.clamp(r6**2, min=self.tol)

        # Compute Lennard-Jones potential energy
        lj_energy = 4 * self.epsilon * ( (self.sigma)**(12)/r12 - (self.sigma)**(6)/r6 ) - self.ecutoff

        # Evaluate condition for applying cutoff and avoiding self interactions
        cond_cutoff = (r2 < self.cutoff_sq) & (r2 > self.tol)
        # Compute partial energies and zero when outside cutoff and when same particle 
        e_part = torch.where(cond_cutoff, lj_energy, torch.zeros(lj_energy.shape, device=self.device)) 

        # Add linear term if required
        if self.cutin is not None:
            # Compute linear term
            r = torch.sqrt(r2)
            lin_energy = self.slope * (r - self.cutin) + self.ecutin
            # Evaluate condition for applying linear term and avoiding self interactions
            cond_cutin = (r < self.cutin) & (r > self.sqrt_tol)
            # Evaluate partial energy
            e_part = torch.where(cond_cutin, lin_energy, e_part) 

        # Sum up potential energy contributions and remove overcounting
        e_tot = .5*torch.sum(e_part, dim=(1,2)) + self.n_particles*self.etail

        return e_tot 


    def init_conf(self, random=False, asNumpy=False):
        
        if random:
            conf = np.random.uniform(size=[self.n_particles, self.dimensions])*self.box_length.cpu().numpy()
        else:
            conf = self.fcc_2d()

        if asNumpy:
            return conf
        else:
            return torch.from_numpy(((conf).ravel()).astype(np.float32)).to(self.device)


    def fcc_2d(self):

        n_cells = int(np.round(np.power(self.n_particles/2, 1/2)))

        if self.n_particles != 2*n_cells**2:
            raise Exception("Wrong number of particles, expected {}!".format(2*n_cells**2))

        a = self.box_length.cpu().numpy()/n_cells
        fcc_displ = a/2

        n_placed = 0
        conf = np.zeros([self.n_particles, 2])
        
        for i in range(n_cells):
            for j in range(n_cells):

                conf[n_placed, 0] = a[0] * i
                conf[n_placed, 1] = a[1] * j
  
                conf[n_placed + 1, 0] = a[0] * i + fcc_displ[0]
                conf[n_placed + 1, 1] = a[1] * j + fcc_displ[1]

                n_placed += 2

        return conf