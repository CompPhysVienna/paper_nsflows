import numpy as np
import torch
import os
import math
from tqdm import tqdm

from nsflows.samplers.base import base_sampler
from nsflows.network.trainer import Trainer

from nsflows.tools.util import remove_from_tensor, ress

class nflows_propagator(base_sampler):
    
    def __init__(self, flow, conditioned : bool, max_generation_attempts : None | float = None, transform : bool = False, max_sample_size = 20000):
        super(nflows_propagator, self).__init__(system=flow.posterior, n_cycles=0, step_size=0)

        # Device to perform computations
        self.device = flow.device

        # Specifics of target system
        self.n_particles = flow.posterior.n_particles
        self.dimensions = flow.posterior.dimensions
        self.dofs = flow.posterior.dofs
        if flow.posterior.PBC:
            self.box_length = flow.posterior.box_length

        self.max_generation_attempts = max_generation_attempts
        self.empty_pool = True
        self.pool_size = None
        self.max_sample_size = max_sample_size

        self.full_training_counter = 0
        self.training_counter = 0
        self.generating_counter = 0

        self.conditioned = conditioned
        self.transform = transform

        # Initialization of the flow  
        self.flow = flow

        # Trainer for the network
        self.flow_trainer = Trainer(self.flow)

        return

    def initialize_weights(self):

        print("Initializing Network weights")

        self.flow.apply(self.flow.init_weights)

        return


    def train(self, train_dataset, training_protocol, fine_tune, outputdir="./"):
        
        if fine_tune:
            print("\nFine-tune training: executing last training stage only.")

            # "{save_best}_flow_parameters_{self.full_training_counter:04d}_{(len(training_protocol) - 2):04d}.pt"
            # is needed to load parameters of the last full training up to the last training step of the protocol
            # so if one has 3 protocols, fine tune performs only number 3 and loads parameters after last full training until step 2
            if training_protocol[-2]["save_best"]:
                self.load_parameters(params_path = os.path.join(outputdir, f"best_flow_parameters_{self.full_training_counter:04d}_{(len(training_protocol) - 2):04d}.pt"))
            else:
                self.load_parameters(params_path = os.path.join(outputdir, f"flow_parameters_{self.full_training_counter:04d}_{(len(training_protocol) - 2):04d}.pt"))
        else:
            print(f"\nFull training: training in {len(training_protocol)} stages")
        metrics = []
        self.average_attempts = 0
        self.n_sample_space = 0

        for stage, train_param in enumerate(training_protocol):
            # Train the flow here
            if fine_tune:
                # When fine-tuning, only the last stage of the protocol is executed, the others are skipped
                if stage < len(training_protocol) - 1:
                    continue

            total_steps = train_param["total_steps"]
            batch_size = train_param["batch_size"]
            steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
            n_epochs = math.ceil(total_steps / steps_per_epoch)
            exact_epochs = total_steps / steps_per_epoch

            if train_param["optimizer"] is not None:
                optimizer = getattr(torch.optim, train_param["optimizer"])([p for p in self.flow.parameters() if p.requires_grad], lr=train_param["end_lr"])
            else:
                optimizer = None
            
            if train_param["scheduler"] == "LambdaLR":
                def lr_lambda(step):
                    return  train_param["start_lr"] / train_param["end_lr"] + step * (1 - train_param["start_lr"] / train_param["end_lr"]) / n_epochs

                scheduler = getattr(torch.optim.lr_scheduler, train_param["scheduler"])(
                    optimizer,
                    lr_lambda=lr_lambda
                )
            elif train_param["scheduler"] == "OneCycleLR":

                div_factor = train_param["max_lr"]/train_param["start_lr"]
                final_div_factor = train_param["start_lr"]/train_param["end_lr"]

                scheduler = getattr(torch.optim.lr_scheduler, train_param["scheduler"])(
                    optimizer,
                    max_lr=train_param["max_lr"],                      # peak LR
                    pct_start=0.4,
                    # anneal_strategy="linear",
                    div_factor=div_factor,
                    final_div_factor=final_div_factor,
                    epochs=n_epochs,
                    steps_per_epoch=steps_per_epoch,
                )
            elif train_param["scheduler"] == "CosineAnnealingLR":

                for pg in optimizer.param_groups:
                    pg["lr"] = train_param["start_lr"]

                scheduler = getattr(torch.optim.lr_scheduler, train_param["scheduler"])(
                    optimizer,
                    T_max=n_epochs,
                    eta_min=train_param["end_lr"], 
                )

            elif train_param["scheduler"] is None:
                scheduler = None
            else:
                raise NotImplementedError
            
            print(f"Stage {stage+1} of {len(training_protocol)}")
            metrics.append(self.flow_trainer.training_routine(train_dataset, 
                                                     w_zx=train_param["w_zx"], 
                                                     w_xz=train_param["w_xz"], 
                                                     n_epochs=n_epochs, 
                                                     batch_size=batch_size, 
                                                     conds_per_batch=train_param["conds_per_batch"],
                                                     n_dump=1, 
                                                     n_save=0,
                                                     save_best=train_param["save_best"], 
                                                     save_dir=outputdir,
                                                     stage = stage, 
                                                     counter=self.training_counter, 
                                                     optimizer=optimizer, 
                                                     scheduler=scheduler,
                                                     clip_grad_norm=100,
                                                     )
            )

            if train_param["save_best"]:
                self.load_parameters(params_path = os.path.join(outputdir, f"best_flow_parameters_{self.training_counter:04d}_{stage:04d}.pt"))
                
            print()

        if not fine_tune:
            self.full_training_counter = self.training_counter
        self.training_counter += 1
        
        return metrics


    def load_parameters(self, params_path = "./flow_parameters.pt"):
        
        print(f"\nLoading Parameters from file {params_path}")

        # Load parameters from previous training
        self.flow.load_state_dict(torch.load(params_path))
        
        return


    def generate(self, n_pool, energy_bound, 
                 outputdir="./", 
                 disable_pbar=False, 
                 save_biased_pool=False):
        
        self.pool = []
        self.pool_biased = []

        with open(os.path.join(outputdir, f"generation_log_{self.generating_counter:04d}.txt"), "w+") as f:
            f.write(f"# Generation {self.generating_counter}\n\n")
        n_generated = 0
        attempted_generation = 0
        acc = 0
        acc_biased = 0
        avg_ress = 0

        if n_pool > self.max_sample_size:
            n_samples = self.max_sample_size
        else:
            n_samples = n_pool

        with torch.no_grad():
            with tqdm(total=n_pool, desc="Generating samples", unit="samples", disable=disable_pbar) as pbar:
                while n_generated < n_pool:

                    # We sample from the truncated Gaussian
                    z = self.flow.prior.sample(n_samples, transform=self.transform)
                    # This must be sampled from the dataset otherwise we have a problem for the first set of live points
                    condition_z = torch.zeros((z.shape[0], 1), device=self.device)
                    
                    # Transforming through normalizing flow
                    target_x, logJ_zx = self.flow.F_zx(z, condition=condition_z)
                    energy_x_target = self.flow.posterior.energy(target_x)

                    # Computing weights for resampling:
                    # since probability in target space has to be uniform, to each configuration a probability of 1 is assigned. 
                    # Note that, as in the standard case, partition function is not important
                    # log_prob_zx = -torch.where(energy_x_target < energy_bound, torch.zeros(1, device=self.device), torch.inf*torch.ones(1, device=self.device))
                    log_prob_zx = torch.where(
                        energy_x_target < energy_bound,
                        torch.zeros_like(energy_x_target),
                        torch.full_like(energy_x_target, -torch.inf)
                    )
                    log_prob_z = -self.flow.prior.energy(z)        
                    log_w = (log_prob_zx - log_prob_z + logJ_zx).squeeze(-1)
                    avg_ress += ress(log_w)

                    # Reweighting from Williams et al
                    log_u = torch.log(torch.rand(log_w.shape, device=self.device))
                    log_w = torch.nan_to_num(log_w, neginf=-1e30)
                    indx_will = torch.where((log_w - torch.max(log_w)) >= log_u)[0]
                    target_x_resampled = target_x[indx_will]
                    valid_samples = target_x_resampled.shape[0]
                    n_generated += valid_samples
                    acc += (valid_samples/n_samples)

                    # Even if generation is conditioned, some samples will be out of the boundaries. 
                    # Here we take care of those, removing them from the pool.
                    U_pool_biased = energy_x_target.squeeze()
                    cond_biased = U_pool_biased < energy_bound
                    acc_biased += (cond_biased.sum()/n_samples)

                    pbar.update(min(valid_samples, n_pool - (n_generated - valid_samples)))
                    with open(os.path.join(outputdir, f"generation_log_{self.generating_counter:04d}.txt"), "a") as f:
                        f.write(f"{attempted_generation} {n_generated/n_pool}\n")

                    if valid_samples > 0:                
                        if save_biased_pool:
                            red_indx = torch.randperm(target_x.shape[0], device=self.device)[:valid_samples]
                            self.pool_biased.append(target_x[red_indx])
                        self.pool.append(target_x_resampled)
                    attempted_generation += 1

                    if self.max_generation_attempts is not None and attempted_generation > self.max_generation_attempts:
                        print(f"Trying generating for more than {self.max_generation_attempts} times.")
                        print(f"Acceptances: acc_b = {acc_biased/attempted_generation}, acc = {acc/attempted_generation}")
                        print(f"Average relative effective sample size: {avg_ress/attempted_generation}")
                        print(f"Retraining the model.")
                        return
        
        if save_biased_pool:
            self.pool_biased = torch.cat(self.pool_biased, dim=0)[:n_pool]
        self.pool = torch.cat(self.pool, dim=0)[:n_pool]

        self.pool_size = self.pool.shape[0]
        self.empty_pool = False

        print(f"Generated pool n. {self.generating_counter} for Umax = {energy_bound}")
        if save_biased_pool:
            print(f"Generated a biased pool of {self.pool_biased.shape[0]} with an acceptance of {acc_biased/attempted_generation}")
        print(f"Generated a pool of {self.pool.shape[0]} with an acceptance of {acc/attempted_generation}")
        print(f"Generation repeated {attempted_generation} times")
        print(f"Average relative effective sample size: {avg_ress/attempted_generation}")
        print(f"Generation efficiency: {1/attempted_generation}\n")
        with open(os.path.join(outputdir, f"generation_log_{self.generating_counter:04d}.txt"), "a") as f:
            f.write(f"# Generated pool n. {self.generating_counter} for Umax = {energy_bound}\n")
            if save_biased_pool:
                f.write(f"# Generated a biased pool of {self.pool_biased.shape[0]} with an acceptance of {acc_biased/attempted_generation}\n")
            f.write(f"# Generated a pool of {self.pool.shape[0]} with an acceptance of {acc/attempted_generation}\n")
            f.write(f"# Generation repeated {attempted_generation} times\n")
            f.write(f"# Average relative effective sample size: {avg_ress/attempted_generation}\n")
            f.write(f"# Generation efficiency: {1/attempted_generation}\n")

        self.generating_counter +=1

        return 


    # This method chooses points from the sampling pool generated by the flow and uses them to substitute highest energy point
    def sample_space(self, N, energy_bound):

        x = []
        u_x = []
        n_confs = 0
        attempt = 0             
        while self.pool.shape[0] >= N:
            
            # randomly chooses a sample from the pool
            indices = np.random.choice(self.pool.shape[0], N, replace = False)
            xp = self.pool[indices].clone()
            u_xp = self.flow.posterior.energy(xp).squeeze()
                
            # Checks if the energy of the new point is lower than the boundary. 
            # If it is accepts else rejects.
            mask = u_xp < energy_bound
            n_confs += mask.sum()
            x.append(xp[mask])
            u_x.append(u_xp[mask])

            # delete the point from the pool either ways
            self.pool = remove_from_tensor(self.pool, indices)
            
            attempt += 1
            if n_confs >= N:
                break

        if self.pool.shape[0] < N and n_confs < N:
            
            x = self.x0
            u_x = self.flow.posterior.energy(x)
            
            self.average_attempts /= self.n_sample_space
            
            self.empty_pool = True
            self.pool_size = None
        
        else:
            x = torch.cat(x, dim=0)
            u_x = torch.cat(u_x, dim=0)

            self.average_attempts += attempt
            self.n_sample_space +=1

            self.pool_size = self.pool.shape[0]

        return x[:N], u_x[:N], attempt