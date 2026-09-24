import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial

from nsflows.tools.util import octahedral_transformation

class ConditionedDataset(Dataset):
    
    def __init__(self, 
                 flow, 
                 data_tensor, 
                 test_fraction, 
                 conditions_tensor, 
                 shuffle_data=False, 
                 transform=True, 
                 augment=True, 
                 energy_labels=None,
                 conditions_distribution="single",
                 sample_conditions_from_dataset=False):
        
        super(ConditionedDataset, self).__init__()        
        
        self.device = flow.device
        self.conditioned = True

        flow.eval()

        n_test = int(test_fraction * data_tensor.shape[0])
        random_generator = torch.Generator(device=flow.device)

        if shuffle_data:
            shuffle_indices = torch.randperm(data_tensor.shape[0], generator=random_generator, device=flow.device)
            data_tensor = data_tensor[shuffle_indices]
            conditions_tensor = conditions_tensor[shuffle_indices]
            if energy_labels is not None:
                energy_labels = energy_labels[shuffle_indices]
        
        self.condition_values = torch.unique(conditions_tensor)
        self.num_condition_values = self.condition_values.numel()
        self.max_condition, self.min_condition = torch.max(self.condition_values), torch.min(self.condition_values)
        # set conditions between 1 and 0 if multiple values, else set it to 1
        self.conditions_tensor_norm = self.normalize_conditions(conditions_tensor)

        self.data_dimensions  = flow.posterior.dimensions
        self.data_n_particles = flow.posterior.n_particles

        self.train_data_x = data_tensor[:-n_test]
        self.test_data_x  = data_tensor[-n_test:]
        if energy_labels is None:
            self.energy_train_x = flow.posterior.energy(self.train_data_x)
            self.energy_test_x  = flow.posterior.energy(self.test_data_x)
        else:
            self.energy_train_x = energy_labels[:-n_test]
            self.energy_test_x  = energy_labels[-n_test:]

        if augment:
            assert transform, "Cannot augment without transforming"

        self.augment       = augment
        self.transform     = transform

        self.test_data_z   = flow.prior.sample(n_test, transform=self.transform)
        self.energy_test_z = flow.prior.energy(self.test_data_z)
        
        if self.num_condition_values == 1 and sample_conditions_from_dataset == False:
            sample_conditions_from_dataset = True
            print(f"Warning: with single condition values the condition sampler is automatically set to sample_conditions_from_dataset = {sample_conditions_from_dataset}")
            print(f"To silence this warning set sample_conditions_from_dataset to True in class initialization")
        if conditions_distribution == "single" and sample_conditions_from_dataset == False:
            sample_conditions_from_dataset = True
            print(f"Warning: with conditions_distribution = single the condition sampler is automatically set to sample_conditions_from_dataset = {sample_conditions_from_dataset}")
            print(f"To silence this warning set sample_conditions_from_dataset to True in class initialization")

        self._sample_conditions = partial(self.sample_conditions, method=conditions_distribution, from_dataset=sample_conditions_from_dataset)
        self.condition_test_z = self._sample_conditions(n_test)
        self.condition_train_x = self.conditions_tensor_norm[:-n_test]
        self.condition_test_x = self.conditions_tensor_norm[-n_test:]

    def __len__(self):

        return self.train_data_x.shape[0]

    def __getitem__(self, idx):

        train_item = self.train_data_x[idx].clone()
        train_label = self.energy_train_x[idx]
        train_condition = self.condition_train_x[idx] if self.conditioned else None

        return train_item, train_label, train_condition
    
    
    def get_test_data(self):
        
        return self.test_data_x, self.energy_test_x, self.condition_test_x , self.test_data_z, self.energy_test_z, self.condition_test_z
    

    def normalize_conditions(self, conditions, inverse=False):

        if self.num_condition_values > 1:
            if inverse:
                return (self.max_condition - self.min_condition)*conditions + self.min_condition
            else:
                return (conditions - self.min_condition)/(self.max_condition - self.min_condition)
        else:
            if inverse:
                return 0*conditions + self.min_condition
            else:
                return 0*conditions
            

    def sample_conditions(self, n_samples, method="single", from_dataset=False):
        
        if method == "single":
            samples = torch.zeros((n_samples, 1), device=self.device)

        elif method == "uniform":
            if from_dataset:
                weights = torch.ones(self.num_condition_values, device=self.device)
                indices = torch.multinomial(weights, n_samples, replacement=True)
                samples = self.conditions_tensor_norm[indices].view(n_samples, 1)
            else:
                samples = torch.rand(n_samples, 1, device=self.device)  # Uniform between 0 and 1

        elif method == "skewed_low":
            if from_dataset:
                weights = torch.linspace(1.0, 0.01, steps=self.num_condition_values, device=self.device)
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, n_samples, replacement=True)
                samples = self.conditions_tensor_norm[indices].view(n_samples, 1)
            else:
                # Use Beta distribution with alpha < 1 and beta = 1
                # This skews the distribution toward 0
                alpha = 0.5  # controls the skewness
                beta = 1.0
                samples = torch.distributions.Beta(alpha, beta).sample((n_samples, 1)).to(device=self.device)

        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return samples


class PBCDataset(Dataset):
    
    def __init__(self, 
                 flow, 
                 data_tensor, 
                 test_fraction, 
                 conditions_tensor, 
                 shuffle_data=False, 
                 transform=True, 
                 augment=True, 
                 energy_labels=None,
                 conditions_distribution="single",
                 sample_conditions_from_dataset=False):
        super(PBCDataset, self).__init__()        
        
        self.device = flow.device
        self.conditioned = True

        flow.eval()

        n_test = int(test_fraction * data_tensor.shape[0])
        random_generator = torch.Generator(device=flow.device)

        if shuffle_data:
            shuffle_indices = torch.randperm(data_tensor.shape[0], generator=random_generator, device=flow.device)
            data_tensor = data_tensor[shuffle_indices]
            conditions_tensor = conditions_tensor[shuffle_indices]
            if energy_labels is not None:
                energy_labels = energy_labels[shuffle_indices]
        
        self.condition_values = torch.unique(conditions_tensor)
        self.num_condition_values = self.condition_values.numel()
        self.max_condition, self.min_condition = torch.max(self.condition_values), torch.min(self.condition_values)
        # set conditions between 1 and 0 if multiple values, else set it to 1
        self.conditions_tensor_norm = self.normalize_conditions(conditions_tensor)

        self.data_dimensions  = flow.posterior.dimensions
        self.data_n_particles = flow.posterior.n_particles
        self.data_box_length  = flow.posterior.box_length
        # Axis permutations are not a symmetry of an orthorhombic cell.
        self.data_orthorhombic_cell = getattr(flow.posterior, "orthorhombic_cell", False)

        assert flow.posterior.PBC, "PBCDataset can only be used for dataset with PBC"

        self.train_data_x = data_tensor[:-n_test]
        self.test_data_x  = data_tensor[-n_test:]
        if energy_labels is None:
            self.energy_train_x = flow.posterior.energy(self.train_data_x)
            self.energy_test_x  = flow.posterior.energy(self.test_data_x)
        else:
            self.energy_train_x = energy_labels[:-n_test]
            self.energy_test_x  = energy_labels[-n_test:]

        if augment:
            assert transform, "Cannot augment without transforming"

        self.augment       = augment
        self.transform     = transform

        self.test_data_z   = flow.prior.sample(n_test, transform=self.transform)
        self.energy_test_z = flow.prior.energy(self.test_data_z)
        
        if self.num_condition_values == 1 and sample_conditions_from_dataset == False:
            sample_conditions_from_dataset = True
            print(f"Warning: with single condition values the condition sampler is automatically set to sample_conditions_from_dataset = {sample_conditions_from_dataset}")
            print(f"To silence this warning set sample_conditions_from_dataset to True in class initialization")
        if conditions_distribution == "single" and sample_conditions_from_dataset == False:
            sample_conditions_from_dataset = True
            print(f"Warning: with conditions_distribution = single the condition sampler is automatically set to sample_conditions_from_dataset = {sample_conditions_from_dataset}")
            print(f"To silence this warning set sample_conditions_from_dataset to True in class initialization")

        self._sample_conditions = partial(self.sample_conditions, method=conditions_distribution, from_dataset=sample_conditions_from_dataset)
        self.condition_test_z = self._sample_conditions(n_test)
        self.condition_train_x = self.conditions_tensor_norm[:-n_test]
        self.condition_test_x = self.conditions_tensor_norm[-n_test:]


    def __len__(self):

        return self.train_data_x.shape[0]


    def __getitem__(self, idx):

        if self.augment:
            # center random particle
            train_item_p = (self.train_data_x[idx].clone()).view(self.data_n_particles, self.data_dimensions)
            indx = np.random.randint(self.data_n_particles)
            c = train_item_p[indx].clone()
            train_item_p -= c
            train_item_p -= self.data_box_length*torch.round(train_item_p/self.data_box_length)
            train_item_p[[indx, 0]] = train_item_p[[0, indx]]

            # octahedral transformations
            base_oct = octahedral_transformation(self.data_dimensions, self.device,
                                                 orthorhombic_cell=self.data_orthorhombic_cell)
            full_oct = base_oct.repeat(self.data_n_particles, 1, 1)
            train_item = torch.bmm(train_item_p.unsqueeze(1), full_oct).reshape(self.data_n_particles*self.data_dimensions)
        elif self.transform:
            # center first particle
            train_item_p = (self.train_data_x[idx].clone()).view(self.data_n_particles, self.data_dimensions)
            c = train_item_p[0].clone()
            train_item_p -= c
            train_item_p -= self.data_box_length*torch.round(train_item_p/self.data_box_length)
            train_item = train_item_p.reshape(self.data_n_particles*self.data_dimensions)
        else:
            train_item = self.train_data_x[idx].clone()
        train_label = self.energy_train_x[idx]
        train_condition = self.condition_train_x[idx] if self.conditioned else None

        return train_item, train_label, train_condition
    
    
    def get_test_data(self):
        
        return self.test_data_x, self.energy_test_x, self.condition_test_x , self.test_data_z, self.energy_test_z, self.condition_test_z
    
    def normalize_conditions(self, conditions, inverse=False):

        if self.num_condition_values > 1:
            if inverse:
                return (self.max_condition - self.min_condition)*conditions + self.min_condition
            else:
                return (conditions - self.min_condition)/(self.max_condition - self.min_condition)
        else:
            if inverse:
                return 0*conditions + self.min_condition
            else:
                return 0*conditions
            

    def sample_conditions(self, n_samples, method="single", from_dataset=False):

        if method == "single":
            samples = torch.zeros((n_samples, 1), device=self.device)

        elif method == "uniform":
            if from_dataset:
                weights = torch.ones(self.num_condition_values, device=self.device)
                indices = torch.multinomial(weights, n_samples, replacement=True)
                samples = self.conditions_tensor_norm[indices].view(n_samples, 1)
            else:
                samples = torch.rand(n_samples, 1, device=self.device)  # Uniform between 0 and 1

        elif method == "skewed_low":
            if from_dataset:
                weights = torch.linspace(1.0, 0.01, steps=self.num_condition_values, device=self.device)
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, n_samples, replacement=True)
                samples = self.conditions_tensor_norm[indices].view(n_samples, 1)
            else:
                # Use Beta distribution with alpha < 1 and beta = 1
                # This skews the distribution toward 0
                alpha = 0.5  # controls the skewness
                beta = 1.0
                samples = torch.distributions.Beta(alpha, beta).sample((n_samples, 1)).to(device=self.device)

        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return samples