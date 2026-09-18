import datetime
import os
import torch
import numpy as np
from collections import deque

from nsflows.network.dataset import PBCDataset, ConditionedDataset

def nested_sampling(K : int, 
                    system, 
                    std_propagator, 
                    nf_propagator = None, 
                    init_samples_filepath : str | None = None, 
                    max_iters : int = 10000, 
                    n_propagate : int = 1, 
                    update_step : bool = False,
                    turn_on_nf : int = 2500, 
                    alternate_std_ns_iters : int = 0, 
                    n_pool : int = 10000, 
                    load_nf_parameters : bool = False, 
                    reinitialize_nf_parameters : bool = True,
                    itrain : int = 0,
                    cumulate_n_dataset : int = 1,
                    training_protocol : list = [{"w_xz" : 1, 
                                                 "w_zx" : 0, 
                                                 "batch_size" : 128, 
                                                 "conds_per_batch" : -1,
                                                 "total_steps" : 1000,
                                                 "lr" : 1e-3, 
                                                 "save_best" : False, 
                                                 "optimizer" : None, 
                                                 "scheduler" : None,}],
                    iprint : int = 0, 
                    isavesamp : int = 100,
                    save_biased_pool : bool = False, 
                    outputdir : str = "./",
                    disable_pbar : bool = False):

    if turn_on_nf > 0:
        assert turn_on_nf <= max_iters+1, "turn_on_nf must be less than maximum number of iterations"

    # The flow is only ever used from iteration turn_on_nf onwards. Set turn_on_nf to a
    # negative value and leave nf_propagator as None to run standard nested sampling.
    if turn_on_nf >= 0 and nf_propagator is None:
        raise ValueError(
            f"nf_propagator is None but the flow would be switched on at iteration {turn_on_nf}. "
            "Pass a propagator, or set turn_on_nf to a negative value to run standard nested sampling."
        )
    
    st_ns_timings = []
    nf_ns_timings = []
    train_timings = []
    gener_timings = []

    # Initialization of variables
    move_indices = np.empty(n_propagate, dtype=int)
    sample_indices = np.empty(n_propagate, dtype=int)
    acc = torch.zeros(max_iters)
    umax = torch.zeros(max_iters+1)
    uargmax = torch.zeros(max_iters+1)
    data_samples = deque(maxlen=cumulate_n_dataset)
    data_conditions = deque(maxlen=cumulate_n_dataset)

    std_ns_iter = 0
    
    if init_samples_filepath is None:
        samples = torch.stack([system.init_conf(random=True) for i in range(K)])
    elif isinstance(init_samples_filepath, str):
        map_location = system.device if nf_propagator is None else nf_propagator.device
        samples = torch.load(init_samples_filepath, map_location=map_location)
        assert len(samples) == K, f"Initial configuration read from {init_samples_filepath} incompatible with given K (K={K}, len(samples) = {len(samples)})"
    else:
        raise Exception(f"Variable 'init_samples_filepath' must be a string. Provided: {type(init_samples_filepath)}")

    U_samples = system.energy(samples).squeeze()
    U_max, U_argmax = torch.max(U_samples), torch.argmax(U_samples)

    # Create a list of particles from which we can choose
    # This is basically needed to avoid choosing the particle with the highest energy, which must be eliminated
    particles_to_choose_from = np.arange(K)
    particles_to_choose_from = particles_to_choose_from[particles_to_choose_from != U_argmax.item()]

    umax[0] = U_max.item()
    uargmax[0] = U_argmax.item()

    normalizing_flows = False
    fine_tune = False

    # Number of iterations actually completed, so an interrupted run can return
    # its history without the unfilled (zero) tail of the preallocated arrays.
    completed_iters = 0

    if iprint > 0:
        print(0, samples.shape[0], 1., U_max.item(), U_argmax.item())
        with open(os.path.join(outputdir, "output.txt"), "w+") as f:
            f.write(f"{0} {samples.shape[0]} {1.} {U_max.item()} {U_argmax.item()}\n")

    # Start timers
    run_start_time = st_ns_timing_s = datetime.datetime.now()

    def write_timings(run_end_time, interrupted=False):
        # A run that never switches the flow on appends nothing to any of the timing
        # lists, so close the standard-NS stretch that is still open here. The network
        # columns (NF_NS, TRAIN, GENER) stay at zero for such a run.
        if not (st_ns_timings or nf_ns_timings or train_timings or gener_timings):
            st_ns_timings.append((run_end_time - st_ns_timing_s).total_seconds())

        timings = [st_ns_timings, nf_ns_timings, train_timings, gener_timings]
        max_len = max(len(timing) for timing in timings)
        timings = np.array([timing + [0.] * (max_len - len(timing)) for timing in timings]).T
        header = f"Run started: {run_start_time}\n\nSTD_NS\tNF_NS\tTRAIN\tGENER"
        footer = f"Run ended: {run_end_time}\nElapsed time: {run_end_time - run_start_time}"
        if interrupted:
            footer += "\nRun interrupted before reaching max_iters"
        np.savetxt(os.path.join(outputdir, "timings.txt"), timings, header=header, footer=footer)

    try:

        # This is the iterative loop for the nested sampling
        for iter in range(max_iters):
            
            if iter == turn_on_nf:
                # Check that this is done once!
                normalizing_flows = True
                st_ns_timing_e = datetime.datetime.now()
                st_ns_timings.append((st_ns_timing_e - st_ns_timing_s).total_seconds())

                if iter > 0:
                    nf_ns_timings.append(0.)

                # if scheduler is not None:
                #     for param_group in optimizer.param_groups:
                #         initial_lr = param_group['lr']

            if normalizing_flows:
                # If the pool is empty there are things to do:
                # - Train the network on the current set of live points
                # - Generate a new pool of configuration from which to get samples
                if nf_propagator.empty_pool:
                    # (Re)Initialize weights and biases
                    if reinitialize_nf_parameters:
                        nf_propagator.initialize_weights()
                    elif itrain > 0:
                        fine_tune = True
                        if (nf_propagator.training_counter % itrain) == 0:
                            nf_propagator.initialize_weights()
                            fine_tune = False
                            
                    # Training is performed using the set of live points
                    # TODO: Check this one, seems unnecessary
                    # This continues the training when max_generation_attempts in nf_propagator is set < inf
                    while nf_propagator.empty_pool:
        
                        if isavesamp > 0:
                            torch.save(samples.squeeze(), os.path.join(outputdir, f"dataset_{nf_propagator.training_counter:04d}.pt"))
                            torch.save(U_max, os.path.join(outputdir, f"conds_{nf_propagator.training_counter:04d}.pt"))

                        data_samples.append(samples)
                        conditions = torch.ones((samples.shape[0], 1), device=nf_propagator.device)*U_max
                        data_conditions.append(conditions)

                        data_tensor = torch.cat(list(data_samples), dim=0)
                        conditions_tensor = torch.cat(list(data_conditions), dim=0)
                        
                        if nf_propagator.flow.posterior.PBC:
                            samples_dataset = PBCDataset(flow=nf_propagator.flow, 
                                            data_tensor=data_tensor, 
                                            test_fraction=0.1, 
                                            shuffle_data=True, 
                                            conditions_tensor=conditions_tensor,
                                            transform=True,
                                            augment=True)
                        else:    
                            samples_dataset = ConditionedDataset(flow=nf_propagator.flow, 
                                            data_tensor=data_tensor, 
                                            test_fraction=0.1, 
                                            shuffle_data=True, 
                                            conditions_tensor=conditions_tensor,
                                            transform=False,
                                            augment=False)
                        
                        assert nf_propagator.transform == samples_dataset.transform, "Check compatibility between transform in dataset and nf_propagator"
                                               
                        if load_nf_parameters:
                            nf_propagator.load_parameters()
                            train_timings.append(0.)
                        else:
                            train_timing_s = datetime.datetime.now()
                            nf_propagator.train(train_dataset = samples_dataset, 
                                                training_protocol = training_protocol,
                                                fine_tune = fine_tune, 
                                                outputdir = outputdir)
                            train_timing_e = datetime.datetime.now()
                            train_timings.append((train_timing_e - train_timing_s).total_seconds())

                        # Once training is done, pool can be generated (when generating no gradient is needed)
                        with torch.no_grad():

                            # We generate pool of samples using normalizing flows
                            gener_timing_s = datetime.datetime.now()
                            nf_propagator.generate(n_pool, energy_bound = U_max, outputdir=outputdir, disable_pbar=disable_pbar, save_biased_pool=save_biased_pool)
                            gener_timing_e = datetime.datetime.now()
                            gener_timings.append((gener_timing_e - gener_timing_s).total_seconds())

                            if isavesamp > 0:
                                if save_biased_pool:
                                    torch.save(nf_propagator.pool_biased.squeeze(), os.path.join(outputdir, f"pool_biased_{(nf_propagator.generating_counter-1):04d}.pt"))
                                torch.save(nf_propagator.pool.squeeze(), os.path.join(outputdir, f"pool_{(nf_propagator.generating_counter-1):04d}.pt"))
                        
                        # Log timings
                        timings = [st_ns_timings, nf_ns_timings, train_timings, gener_timings]
                        max_len = max(len(timing) for timing in timings)
                        timings = np.array([timing + [0.] * (max_len - len(timing)) for timing in timings]).T
                        header = f"Run started: {run_start_time}\n\nSTD_NS\tNF_NS\tTRAIN\tGENER"
                        np.savetxt(os.path.join(outputdir, "timings.txt"), timings, header=header)

                        nf_ns_timing_s = datetime.datetime.now()

                clone_indx = np.random.choice(particles_to_choose_from)

                # When the pool is not empty then sample one configuration at a time via Normalizing flow 
                nf_propagator.x0 = samples[clone_indx].unsqueeze(0)
                samples[U_argmax], U_samples[U_argmax], acc[iter] = nf_propagator.sample_space(N = 1, energy_bound = U_max)
                
                if nf_propagator.empty_pool:
                    if alternate_std_ns_iters > 0:
                        normalizing_flows = False
                        std_ns_iter = 0
                        st_ns_timing_s = datetime.datetime.now()
                    else:
                        st_ns_timings.append(0.)
                    nf_ns_timing_e = datetime.datetime.now()
                    nf_ns_timings.append((nf_ns_timing_e - nf_ns_timing_s).total_seconds())
                    
            # Standard Nested Sampling algorithm       
            else:
                std_ns_iter += 1
                # Here we select the walkers to keep and the ones to remove.
                # By allowing U_argmax to be here we are effectively cloning a particle
                # as the position of U_argmax will be replaced by an evolved clone
                sample_indices[0] = U_argmax
                if n_propagate > 1:
                    sample_indices[1:] = np.random.choice(particles_to_choose_from, n_propagate-1, replace=False)
                # Here we chose the indices to move
                # As U_argmax must be eliminated it must not appear here.
                # In principle, here we could choose with replacement (see line below)
                # move_indices = np.random.choice(particles_to_choose_from, n_propagate, replace=True)
                move_indices[0] = np.random.choice(particles_to_choose_from)
                move_indices[1:] = sample_indices[1:]

                # These are the particles that I move
                std_propagator.x0 = samples[move_indices]
                # Here I move them
                samples[sample_indices], U_samples[sample_indices], acc[iter] = std_propagator.sample_space(N = n_propagate, energy_bound = U_max)

                if update_step:
                    if acc[iter] < 0.5:
                        std_propagator.step_size *= 0.5
                
                if iter > turn_on_nf and std_ns_iter == alternate_std_ns_iters:
                    
                    normalizing_flows = True
                    st_ns_timing_e = datetime.datetime.now()
                    st_ns_timings.append((st_ns_timing_e - st_ns_timing_s).total_seconds())

            U_max, U_argmax = torch.max(U_samples), torch.argmax(U_samples)
            umax[iter+1] = U_max.item()
            uargmax[iter+1] = U_argmax.item()
            completed_iters = iter + 1
                
            # Create a list of particles from which we can choose
            # This is basically needed to avoid choosing the particle with the highest energy, which must be eliminated
            particles_to_choose_from = np.arange(K)
            particles_to_choose_from = particles_to_choose_from[particles_to_choose_from != U_argmax.item()]

            if iprint > 0:
                if (iter+1) % iprint == 0:

                    if normalizing_flows:
            
                        print(iter+1, samples.shape[0], acc[iter].item(), U_max.item(), U_argmax.item(), nf_propagator.pool_size)
                        with open(os.path.join(outputdir, "output.txt"), "a") as f:
                            f.write(f"{iter+1} {samples.shape[0]} {acc[iter].item()} {U_max.item()} {U_argmax.item()} {nf_propagator.pool_size}\n")

                        if nf_propagator.empty_pool:
                            print(f"\nPool is now empty: average attempts = {nf_propagator.average_attempts}\n")

                    else:
                        print(iter+1, samples.shape[0], acc[iter].item(), U_max.item(), U_argmax.item())
                        with open(os.path.join(outputdir, "output.txt"), "a") as f:
                            f.write(f"{iter+1} {samples.shape[0]} {acc[iter].item()} {U_max.item()} {U_argmax.item()}\n")
                
            if isavesamp > 0:
                if (iter+1) % isavesamp == 0:
                    torch.save(samples.squeeze(), os.path.join(outputdir, "samples.pt"))
                if (iter+1) > isavesamp and (iter + 1) % isavesamp == 0:
                    torch.save(samples, os.path.join(outputdir, f"samples_{(iter+1):012d}.pt"))
                    torch.save(U_max, os.path.join(outputdir, f"U_max_{(iter+1):012d}.pt"))
        
        # Stop timer
        run_end_time = datetime.datetime.now()

        # Log timings
        write_timings(run_end_time)

        return samples.squeeze(), U_samples.squeeze(), acc, umax
    
    except KeyboardInterrupt:

        # Log timings for the part of the run that did happen.
        write_timings(datetime.datetime.now(), interrupted=True)

        # acc and umax were preallocated to max_iters; drop the tail that was never
        # filled, so the caller gets the history of the iterations that actually ran.
        return samples.squeeze(), U_samples.squeeze(), acc[:completed_iters], umax[:completed_iters+1]