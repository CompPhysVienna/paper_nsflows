import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from nsflows.tools.util import ress

class Trainer(object):

    def __init__(self, network):
        """
        Trainer wrapper for a conditional normalizing flow network.
        
        Args:
            network: the normalizing flow model to be trained.
        """
        self.network = network
        self.device = network.device

    def training_routine(self, 
                         train_dataset : torch.utils.data.Dataset, 
                         w_xz : float = 1, 
                         w_zx : float = 1, 
                         n_epochs : int = 500, 
                         batch_size : int = 10000, 
                         conds_per_batch : int = -1, 
                         n_dump : int = 1, 
                         n_save : int = 1,
                         save_best : bool = False, 
                         save_dir : str = "./", 
                         stage : int | None = None,
                         counter : int | None = None, 
                         optimizer = None, 
                         scheduler = None, 
                         early_stopping_lr_threshold : float = 0,
                         disable_screen_log : bool = False,
                         clip_grad_norm : float = 0) -> np.ndarray:
        """
        Main training loop for the conditional normalizing flow.

        Args:
            train_dataset (torch.utils.data.Dataset): dataset providing training and test data. 
                Must implement __getitem__ and __len__. 
                In this implementation it is also expected to provide helper methods 
                `_sample_conditions`, `normalize_conditions`, and `get_test_data`.            
            w_xz (float): weight for the x→z negative log-likelihood loss.
            w_zx (float): weight for the z→x reconstruction/constraint loss.
            n_epochs (int): number of training epochs.
            batch_size (int): mini-batch size.
            conds_per_batch (int): number of sampled conditions per batch (if <0, set to batch_size).
            n_dump (int): frequency (in epochs) for evaluation/logging.
            n_save (int): frequency (in epochs) for saving intermediate parameters.
            save_best (bool): whether to save model parameters corresponding to the best validation score.
            save_dir (str): directory for saving logs and checkpoints.
            counter (int | None): optional counter to distinguish multiple training runs.
            optimizer: PyTorch optimizer (default: Adam with lr=1e-4).
            scheduler: PyTorch learning rate scheduler (ReduceLROnPlateau or OneCycleLR supported).
            early_stopping_lr_threshold (float): stop training if LR drops below this value (only with ReduceLROnPlateau).
            disable_screen_log (bool): disable printing logs to stdout.

        Returns:
            np.ndarray: array of training and validation metrics collected across epochs.
        """

        # Filenames for logs and saved parameter checkpoints
        if counter is None and stage is None:
            log_filename = f"train_log.txt"
            parameters_filename = f"flow_parameters.pt"
            best_parameters_filename = f"best_flow_parameters.pt"            
            int_parameters_filename = "flow_parameters_{}.pt"
        else:
            if counter is None:
                log_filename = f"train_log_{stage:04d}.txt"
                parameters_filename = f"flow_parameters_{stage:04d}.pt"
                best_parameters_filename = f"best_flow_parameters_{stage:04d}.pt"
                int_parameters_filename = "flow_parameters_" + f"{stage:04d}" + "_{}.pt"
            elif stage is None:
                log_filename = f"train_log_{counter:04d}.txt"
                parameters_filename = f"flow_parameters_{counter:04d}.pt"
                best_parameters_filename = f"best_flow_parameters_{counter:04d}.pt"
                int_parameters_filename = "flow_parameters_" + f"{counter:04d}" + "_{}.pt"
            else:
                log_filename = f"train_log_{counter:04d}_{stage:04d}.txt"
                parameters_filename = f"flow_parameters_{counter:04d}_{stage:04d}.pt"
                best_parameters_filename = f"best_flow_parameters_{counter:04d}_{stage:04d}.pt"
                int_parameters_filename = "flow_parameters_" + f"{counter:04d}" + "_" + f"{stage:04d}" + "_{}.pt"
    
            
        # Default conditions per batch
        if conds_per_batch < 0:
            conds_per_batch = batch_size

        # Best model saving requires single dump frequency
        if save_best:
            best_perf = np.inf
            assert n_dump == 1, "save_best requires n_dump = 1"

        # Logging weights (if zero still logged in validations)
        w_xz_log = w_xz if w_xz > 0 else 1
        w_zx_log = w_zx if w_zx > 0 else 1

        # Training dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        n_steps = n_epochs * len(train_dataloader)

        # Default optimizer (Adam)
        if optimizer is None:
            optimizer = torch.optim.Adam([p for p in self.network.parameters() if p.requires_grad], lr=1e-4)        

        # Scheduler sanity checks
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) \
            or isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) \
            or isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                assert n_dump == 1, "Chosen scheduler requires n_dump = 1"
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                assert scheduler.total_steps == len(train_dataloader) * n_epochs, \
                       "Problem with total_steps (epochs*steps_per_epochs) in One Cycle LR"

        # Initialize losses
        nll = loss_ecut = torch.tensor(0., requires_grad=True).to(self.device)
        ecut_violation = torch.tensor(0.).to(self.device)
        total_norm = torch.tensor(0.).to(self.device)

        # Container for metrics (per epoch)
        epoch_metrics = []
        step = 0
        best_epoch = 0

        try:
            for epoch in range(1, n_epochs + 1):    
                
                # Switch network to training mode
                self.network.train()
                for i, sample in enumerate(train_dataloader):

                    # Data and conditions from dataset
                    x = sample[0]
                    _batch_size = x.shape[0]
                    cond_x = sample[2]
                    
                    # Compute x2z loss if enabled
                    if w_xz > 0:
                        nll, _ = self.network.loss_xz(x, condition_x=cond_x)

                    # Compute z2x loss if enabled
                    if w_zx > 0:
                        # Sample latent codes from prior
                        z = self.network.prior.sample(_batch_size, transform=train_dataset.transform)
                        # Sample normalized conditions
                        cond_z = train_dataset._sample_conditions(conds_per_batch)
                        cond_z = (cond_z.repeat((_batch_size + conds_per_batch - 1)//conds_per_batch, 1))[:_batch_size]
                        # Unnormalize conditions for evaluation
                        U_max = train_dataset.normalize_conditions(cond_z, inverse=True) 
                        # Evaluate reconstruction/constraint loss
                        loss_ecut, _, ecut_violation = self.network.loss_zx(z, U_max=U_max, condition_z=cond_z)

                    # Weighted total loss
                    loss_total = w_xz*nll + w_zx*loss_ecut

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss_total.backward()
                    if clip_grad_norm > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=clip_grad_norm)  # huge value, effectively no clipping
                    optimizer.step()

                    # Step LR scheduler if OneCycleLR
                    if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()
                    step += 1

                # ===== Validation & Logging =====
                if n_dump > 0:
                    if epoch == 1 or epoch % n_dump == 0:
                    
                        self.network.eval()

                        val_nll, val_loss_ecut, val_ress_zx, val_ecut_violation = [], [], [], []
                        with torch.no_grad():
                            # Retrieve test data from dataset
                            test_data_x, _, test_cond_x, test_data_z, test_energy_z, test_cond_z = train_dataset.get_test_data()
                            n_batches_test_data_x = max(test_data_x.shape[0] // batch_size, 1)
                            n_batches_test_data_z = max(test_data_z.shape[0] // batch_size, 1)
                            
                            # Evaluate NLL (x2z)
                            for b in range(n_batches_test_data_x):
                                condition_x = test_cond_x[b*batch_size:(b+1)*batch_size]
                                val_nll_part, _ = self.network.loss_xz(
                                    test_data_x[b*batch_size:(b+1)*batch_size],
                                    condition_x=condition_x
                                )
                                val_nll.append(val_nll_part)
                            val_nll = torch.vstack(val_nll).mean()

                            # Evaluate reconstruction/constraint loss (z2x)
                            for b in range(n_batches_test_data_z):
                                condition_z = test_cond_z[b*batch_size:(b+1)*batch_size]
                                U_max = train_dataset.normalize_conditions(condition_z, inverse=True)
                                val_loss_ecut_part, val_logw_zx_part, val_ecut_violation_part = self.network.loss_zx(
                                    test_data_z[b*batch_size:(b+1)*batch_size],
                                    U_max=U_max, 
                                    condition_z=condition_z,
                                    energy_z=test_energy_z[b*batch_size:(b+1)*batch_size],
                                )
                                val_loss_ecut.append(val_loss_ecut_part)
                                val_ress_zx.append(ress(val_logw_zx_part))
                                val_ecut_violation.append(val_ecut_violation_part)
                            val_loss_ecut = torch.vstack(val_loss_ecut).mean()
                            val_ress_zx = torch.vstack(val_ress_zx).mean()
                            val_ecut_violation = torch.vstack(val_ecut_violation).mean()
                            
                            # Total validation loss
                            val_loss_total = w_xz*val_nll + w_zx*val_loss_ecut

                            # Step ReduceLROnPlateau scheduler and early stopping
                            if scheduler is not None:
                                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    scheduler.step(val_loss_total)  
                                    if scheduler.get_last_lr()[0] < early_stopping_lr_threshold:
                                        break
                                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                                    scheduler.step()
                                elif isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                                    scheduler.step()

                            # Build training progress message
                            train_msg = f"{epoch/n_epochs*100:6.2f} % | epoch {epoch}/{n_epochs} step {step}/{n_steps} | train: xz (NLL) = {w_xz_log*nll:.3f}"
                            train_msg += f" zx (ECUT) = {w_zx_log*loss_ecut:.5g}"
                            train_msg += f" loss = {loss_total:.5g} ecut_v = {w_zx_log*ecut_violation:.5g} grad_norm = {total_norm:.5g} | eval: xz (NLL) = {w_xz_log*val_nll:.3f} zx (ECUT) = {w_zx_log*val_loss_ecut:.5g} loss = {val_loss_total:.5g} ecut_v = {w_zx_log*val_ecut_violation:.5g} ress_zx = {val_ress_zx:.5g}"
                            if scheduler is not None:
                                train_msg += f" | lr = {scheduler.get_last_lr()[0]:.3e}"
                            train_msg += f" | {epoch/n_epochs*100:6.2f} %"
                            if not disable_screen_log:
                                print(train_msg)

                            # Store metrics for logging
                            losses = [epoch, step, 
                                      w_xz_log*nll.item(), 
                                      w_zx_log*loss_ecut.item(), 
                                      w_zx_log*ecut_violation.item(),
                                      total_norm.item(), 
                                      w_xz_log*val_nll.item(), 
                                      w_zx_log*val_loss_ecut.item(), 
                                      w_zx_log*val_ecut_violation.item(),
                                      val_ress_zx.item()]
                            if scheduler is not None:
                                losses.append(scheduler.get_last_lr()[0])
                            epoch_metrics.append(losses)

                # ===== Model Saving =====
                if save_best:
                    # perf = val_nll.item() + val_ecut_violation.item()
                    perf = -val_ress_zx.item()
                    if perf < best_perf:
                        best_perf = perf
                        best_epoch = epoch
                        torch.save(self.network.state_dict(), os.path.join(save_dir, best_parameters_filename))

                if n_save > 0:
                    if epoch % n_save == 0 and epoch != n_epochs:
                        torch.save(self.network.state_dict(), os.path.join(save_dir, int_parameters_filename.format(epoch)))

                        # Save intermediate logs
                        with open(os.path.join(save_dir, log_filename), "w+") as log_file:
                            log_file.write("# epoch\tstep\tnll\tecut\tecut_viol\tgrad_norm\tval_nll\tval_ecut\tval_ecut_viol\tval_ress_zx")
                            if scheduler is not None:
                                log_file.write("\tLR")
                            log_file.write("\n")
                            for row in epoch_metrics:
                                log_file.write("\t".join(map(str, row)) + "\n")
            
            # Save final model parameters
            torch.save(self.network.state_dict(), os.path.join(save_dir, parameters_filename))
            
            # Save final log file
            with open(os.path.join(save_dir, log_filename), "w+") as log_file:
                log_file.write("# epoch\tstep\tnll\tecut\tecut_viol\tgrad_norm\tval_nll\tval_ecut\tval_ecut_viol\tval_ress_zx")
                if scheduler is not None:
                    log_file.write("\tLR")
                log_file.write("\n")
                for row in epoch_metrics:
                    log_file.write("\t".join(map(str, row)) + "\n")
                if save_best:
                    log_file.write(f"# Best epoch: {best_epoch}\n")
    
        except KeyboardInterrupt:
            
            # Save parameters if training is interrupted
            torch.save(self.network.state_dict(), os.path.join(save_dir, parameters_filename))
            return np.array(epoch_metrics)
        
        return np.array(epoch_metrics)
