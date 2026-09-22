import uuid
import os
import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from itertools import combinations

def generate_log_spaced_numbers(n: int, k: float = 3, lower_bound: float = -1, upper_bound: float = 1) -> np.ndarray:
    """
    Generate n numbers equally spaced in log space between lower_bound and upper_bound,
    with controllable point density near the lower bound.
    
    Parameters:
    n (int): Number of points to generate
    k (float, optional): Density control parameter. 
        - K > 1: More points concentrated near lower bound
        - K < 1: More points concentrated near upper bound
        - K = 1: Uniform log spacing
        Defaults to 3.0.
    lower_bound (float, optional): Lower bound of the range. Defaults to -1.0.
    upper_bound (float, optional): Upper bound of the range. Defaults to 1.0.
    
    Returns:
    numpy.ndarray: Array of n equally spaced points in log space
    """

    # Adjust bounds to start from 0
    adjusted_upper = upper_bound - lower_bound
    
    # Generate base points
    base_points = np.linspace(0, 1, n) ** (k)
    
    # Scale and translate points
    log_points = base_points * adjusted_upper
    
    # Translate points back to original lower bound
    return log_points + lower_bound


def biased_exponential(shape: tuple, k: float = 3, lower: float = -1, upper: float = 1, device: str = 'cpu') -> torch.Tensor:
    """
    Generates random numbers of a given shape between lower and upper bounds,
    biased towards the lower bound using an inverted exponential transformation.
    
    Parameters:
    shape (tuple): Shape of the output tensor
    k (float): Controls the bias strength (higher k -> stronger bias)
    lower (float): Lower bound of the support (default: -1)
    upper (float): Upper bound of the support (default: 1)
    device (str): Device to perform computations ('cpu' or 'cuda')
    
    Returns:
    torch.Tensor: Tensor of biased random numbers
    """

    u = torch.rand(shape, device=device)
    k_tensor = torch.tensor(k, dtype=torch.float32, device=device)  # Ensure k is a tensor
    
    return lower + (torch.exp(-k_tensor * u) - torch.exp(-k_tensor)) / (1 - torch.exp(-k_tensor)) * (upper - lower)

def reset_optimizer_and_scheduler(optimizer, scheduler, initial_lr):
    
    # Reset optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

    # Reset scheduler
    scheduler.last_epoch = -1  # Reset state for StepLR and similar
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.best = float('inf')  # Reset best metric
        scheduler.num_bad_epochs = 0  # Reset counter
        scheduler.cooldown_counter = 0 # Reset cooldown counter

    return
    

def transform_dataset(x, n_particles, dimensions, box_length, PBC=True):

    dofs = n_particles*dimensions
    
    # transform in internal coordinates (center particle 0)
    xp = x.view(-1, n_particles, dimensions)
    c = xp[:,0].clone()
    xp -= c.unsqueeze(1)
    if PBC:
        xp -= box_length*torch.round(xp/box_length)
    x = xp.view(-1, dofs)

    return x


def octahedral_transformation(dimensions, device):
        
        identity = torch.eye(dimensions, dtype=torch.float32, device=device)
        permuted_axes = identity[:, torch.randperm(dimensions)]            
        reflect = torch.randint(0, 2, size=(dimensions,), device=device) * 2 - 1
        
        return (reflect*permuted_axes).unsqueeze(0)


def dist_matrix(target, reference, n_particles, dimensions, box_length = None):

    reference = reference.reshape((reference.shape[0], n_particles, dimensions))
    target = target.reshape((target.shape[0], n_particles, dimensions))

    rep_reference = reference.repeat(1,1,n_particles).reshape(reference.shape[0], n_particles**2, dimensions)
    rep_target = target.repeat(1, n_particles,1)
    
    v_i0 = rep_reference - rep_target
    if box_length is not None:
        v_i0 -= box_length*torch.round(v_i0/box_length)

    sq_dist_array = torch.square(torch.norm(v_i0, dim=-1))

    cost_matrix = sq_dist_array.reshape(target.shape[0], n_particles, n_particles)

    return cost_matrix

    
def hungarian_algorithm(x, cost_matrix, n_particles, dimensions):
    
    x = x.reshape((x.shape[0], n_particles, dimensions))
    mapped_x = x.new_zeros(x.shape)
    
    for i in range(cost_matrix.shape[0]):
        
        indices = linear_sum_assignment(cost_matrix[i].cpu().numpy())
        mapped_x[i, indices[0],:] = x[i, indices[1],:]
    
    return mapped_x.reshape(x.shape[0], n_particles * dimensions)


def sample_spherical(n_samples, dimensions=2, device=None):
    
    x = torch.randn((dimensions, n_samples), device=device)
    norm = torch.norm(x, dim=0)
    y = (x/norm).T
    
    return y


def truncated_chi(rcut, n_samples, df=2, device=None):
    
    r = torch.tensor([], device=device)
    while r.shape[0] < n_samples:
    
        # Generate samples from standard normal distribution
        normal_samples = torch.randn((n_samples, df), device=device)    
        # Square the samples
        squared_samples = normal_samples.pow(2)
        # Sum up the squared samples
        chi_square_samples = squared_samples.sum(dim=-1)
        # Degrees of freedom adjustment
        chi_square_samples *= df

        rho = torch.sqrt(chi_square_samples)
        rho = rho[rho < rcut]
        r = torch.cat((r, rho), dim=0)

    return r[:n_samples]


# This sampling method is taken from [Williams et al]
# [Williams et al] Michael J. Williams et al., Phys. Rev. D 103, 103006 (2021). DOI: 10.1103/PhysRevD.103.103006
def truncated_normal(n_samples, n_particles, dimensions, rcut, device):

    y = sample_spherical(n_samples*n_particles, dimensions=dimensions, device=device)
    rho = truncated_chi(rcut, n_samples*n_particles, df=dimensions, device=device)
    
    return ((y.T*rho).T).reshape(n_samples, n_particles*dimensions)


def truncated_gaussian(n_samples, n_particles, dimensions, rcut, device, loc=0, scale=1):

    samples = torch.tensor([], device=device)
    while samples.shape[0] < n_samples*n_particles:
        # Step 1: Generate sample from Gaussian distribution
        x = torch.randn(size=(n_samples*n_particles, dimensions), device=device)*scale
        
        # Step 2: Check if sample is within truncation range
        x = x[torch.norm(x, dim=-1) < rcut]
        samples = torch.cat((samples, x + loc))
    
    return samples[:n_samples*n_particles].reshape(n_samples, n_particles*dimensions)


def remove_from_tensor(tensor, indices):

    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    
    return tensor[mask]


def delete_tensor_where(tensor, mask):

    return tensor[mask]


def get_targets(dimensions, n_blocks):
    targets = []

    A = range(dimensions)
    for r in range(1, len(A)):
        for i in combinations(A, r):
            targets.append(i)

    return targets*n_blocks


def get_target_indices(target, n_particles, dimensions):

    coordinate_indices =  np.arange(n_particles*dimensions)
    mask = np.ones(n_particles*dimensions, dtype=bool)
    for indx in target:
        mask[indx::dimensions] = 0

    return coordinate_indices[mask], coordinate_indices[~mask]


def ress(log_w):

    with torch.no_grad():

        if torch.all(log_w == -torch.inf):
            return torch.tensor(0.0, device=log_w.device)
        
        sig = torch.nn.Softmax(dim=0)
        ress = 1/torch.sum(sig(log_w)**2)/log_w.shape[0]

    return ress


def mean_per_condition(values, conditions):
    
    unique_conditions, unique_counts = conditions.unique(return_counts=True)
    
    if len(unique_conditions) == 1:
        return values
    elif len(unique_conditions) == values.shape[0]:
        return values
    
    unique_conditions = unique_conditions.reshape(-1, 1)
    unique_counts = unique_counts.reshape(-1, 1)

    unique_conditions_list = unique_conditions.flatten().tolist()
    conditions_list = conditions.flatten().tolist()

    condition_map = {k:v for k,v in zip(unique_conditions_list, range(len(unique_conditions_list)))}

    indices_np = np.fromiter(map(condition_map.get, conditions_list), dtype=int)
    indices = torch.from_numpy(indices_np).to(values.device).reshape(-1, 1).expand(-1, values.shape[1])

    # Computing mean over values for each condition
    grouped_values = torch.zeros(unique_conditions.shape[0], values.shape[1], dtype=torch.float, device=values.device).scatter_add_(0, indices, values) / unique_counts

    return grouped_values


def searchsorted(bin_locations, inputs, eps=1e-6):

    bin_locations[..., -1] += eps

    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def torch_ns_cbrt(x):
    
    ans = torch.sign(x)*torch.exp(torch.log(torch.abs(x))/3.0)
    
    return ans


def torch_ns_sqrt(x):
    
    ans = torch.exp((torch.log(torch.abs(x))) / 2.0)
    
    return ans


def generate_unique_identifier():
    
    return str(uuid.uuid4())


def remove_empty_directories(root_folder):

    for root, dirs, files in os.walk(root_folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if directory is empty
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")


def generate_output_directory(run_id, root_folder="./output"):

    output_dir = os.path.join(root_folder, run_id)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Generated output directory: {output_dir}")
    
    return output_dir


def box_length_from_density(rho, n_particles, dimensions):
    """
    Compute the side length of a square/cubic box from density.

    Parameters
    ----------
    rho : float
        Number density (N/V).
    n_particles : int
        Number of particles.
    dimensions : int
        Number of spatial dimensions.

    Returns
    -------
    L : float
        Box side length.
    """

    volume = n_particles / rho
    L = volume ** (1.0 / dimensions)

    return L


def density_from_box_length(L, n_particles, dimensions):
    """
    Compute number density from box side length.

    Parameters
    ----------
    L : float
        Box side length.
    n_particles : int
        Number of particles.
    dimensions : int
        Number of spatial dimensions.

    Returns
    -------
    rho : float
        Number density (N/V).
    """

    volume = L ** dimensions
    rho = n_particles / volume

    return rho


def remove_outermost_particle(xp):
    """
    Drop the particle farthest from the origin in each configuration.

    Parameters
    ----------
    xp : torch.Tensor
        (B, N, D) configurations in internal coordinates.

    Returns
    -------
    xp_trimmed : torch.Tensor
        (B, N-1, D) configurations with the outermost particle removed.
    """

    # distance from center
    r2 = torch.sum(xp**2, dim=-1)

    # find farthest particle per batch
    far_idx = torch.argmax(r2, dim=1)

    B, N, D = xp.shape
    mask = torch.ones((B, N), dtype=torch.bool, device=xp.device)
    mask[torch.arange(B), far_idx] = False

    xp_trimmed = xp[mask].view(B, N-1, D)

    return xp_trimmed


def identity_transform(x):

    return x


def rotate_pi2_transform(x):

    y = x.clone()
    x0 = y[..., 0].clone()
    y[..., 0] = -y[..., 1]
    y[..., 1] = x0

    return y


def align_rot90_hungarian(x, x_ref, n_particles, dimensions, box_length=None):
    """
    For each configuration, decide whether the identity or a pi/2 rotation brings
    it closest to the reference, after optimally permuting particle labels.

    Parameters
    ----------
    x : torch.Tensor
        (B, N, 2) configurations to align.
    x_ref : torch.Tensor
        (B, N, 2) or (1, N, 2) reference configuration.
    n_particles : int
        Number of particles in x, i.e. N.
    dimensions : int
        Number of spatial dimensions.
    box_length : float or None
        If given, distances use the minimum image convention with this box length.

    Returns
    -------
    best_sym : torch.Tensor
        (B,) index of the best transform, 0 for identity and 1 for the rotation.
    """

    B = x.shape[0]

    transforms = [identity_transform, rotate_pi2_transform]

    best_cost = torch.full((B,), float("inf"), device=x.device)
    best_sym = torch.zeros(B, dtype=torch.long, device=x.device)

    for s in range(2):

        x_s = transforms[s](x)

        cost_matrix = dist_matrix(
            x_s,
            x_ref,
            n_particles,
            dimensions,
            box_length
        )

        # hungarian_algorithm returns a flattened (B, N*D) tensor
        x_perm = hungarian_algorithm(
            x_s,
            cost_matrix,
            n_particles,
            dimensions
        ).reshape(B, n_particles, dimensions)

        cost = torch.mean((x_perm - x_ref) ** 2, dim=(1, 2))

        mask = cost < best_cost

        best_cost[mask] = cost[mask]
        best_sym[mask] = s

    return best_sym


def apply_rot90_symmetry_batched(x, sym):
    """
    Apply the pi/2 rotation to the configurations flagged by `sym`.

    Parameters
    ----------
    x : torch.Tensor
        (B, N, 2) configurations.
    sym : torch.Tensor
        (B,) as returned by align_rot90_hungarian.

    Returns
    -------
    x_out : torch.Tensor
        (B, N, 2) configurations with the rotation applied where sym == 1.
    """

    x_out = x.clone()

    mask = sym == 1

    if mask.any():
        x0 = x_out[mask, :, 0].clone()
        x_out[mask, :, 0] = -x_out[mask, :, 1]
        x_out[mask, :, 1] = x0

    return x_out


def align_config(config, ref_config, n_particles, dimensions, box_length=None):
    """
    Align configurations to a reference up to a pi/2 rotation and a relabelling
    of the particles. The outermost particle is dropped before the comparison,
    so that the reference only has to fix the inner ones, but the rotation is
    then applied to the full configuration.

    Parameters
    ----------
    config : torch.Tensor
        (B, N, 2) configurations to align.
    ref_config : torch.Tensor
        (B, N-1, 2) or (1, N-1, 2) reference, with its outermost particle
        already removed.
    n_particles : int
        Number of particles in config, i.e. N.
    dimensions : int
        Number of spatial dimensions.
    box_length : float or None
        If given, distances use the minimum image convention with this box length.

    Returns
    -------
    aligned_config : torch.Tensor
        (B, N, 2) aligned configurations.
    """

    # remove outermost particle
    red_config = remove_outermost_particle(config)

    # compute symmetry (identity or pi/2)
    symm = align_rot90_hungarian(
        red_config,
        ref_config,
        n_particles-1,
        dimensions,
        box_length=box_length
    )

    # apply symmetry to the FULL configuration
    aligned_config = apply_rot90_symmetry_batched(config, symm)

    return aligned_config