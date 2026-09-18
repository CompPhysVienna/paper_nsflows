import numpy as np
import torch

def rdf(x, n_particles, dimensions, box_length, cutoff=None, n_bins=100, batch_size=None, log_weights=None):

    if isinstance(box_length, torch.Tensor):
        Lmin = torch.min(box_length).item()/2
        volume = box_length.prod().item()
    elif isinstance(box_length, np.ndarray):
        Lmin = np.min(box_length)/2
        volume = box_length.prod()
    elif isinstance(box_length, (float, int)):
        Lmin = box_length/2
        volume = box_length**dimensions
    else:
        raise TypeError(f"Unsupported type {type(box_length)}")

    if cutoff is None:
        cutoff = Lmin

    dr = cutoff/n_bins
    r = torch.linspace(0., cutoff, n_bins)

    N = x.shape[0]
    if batch_size is None:
        batch_size = N
    n_batches = N//batch_size
    gofr_batch = np.zeros((n_batches, n_bins))
    
    for batch in range(n_batches):

        x_batch = x[batch*batch_size:(batch+1)*batch_size]

        # Reshape input tensor to represent particle positions
        pos = x_batch.view((-1, n_particles, dimensions))

        # Pairwise differences between particles
        rij = pos[:, :, None, :] - pos[:, None, :, :]                 
        # Apply periodic boundary conditions
        rij -= box_length*torch.round(rij/box_length)  
        # Compute square distances between particles  
        r = torch.sqrt(torch.sum(rij**2, dim=-1))                  
        # Evaluate condition for applying cutoff and avoiding self interactions
        cond_cutoff = (r < cutoff) & (r > 0)
        r = r[cond_cutoff]

        if log_weights is not None:
            lw = log_weights[batch*batch_size:(batch+1)*batch_size]
            w = torch.exp(lw - lw.max())
            weights_all = w.view(batch_size,1,1).repeat(1, n_particles, n_particles)/w.mean()
            weights = weights_all[cond_cutoff].cpu().numpy()
        else:
            weights = None

        gofr_batch[batch], r = np.histogram(r.cpu().numpy(), dr*np.arange(n_bins+1), weights=weights)

    gofr = gofr_batch.sum(axis=0)

    V_bins = np.array([np.pi*dr**2*(2*b + 1) for b in range(n_bins)])
    rho = n_particles/volume

    return r[:-1]+0.5*dr, gofr/N/(V_bins*(n_particles-1)*rho)