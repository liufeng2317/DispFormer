import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor
from .data_augmentation import *
from typing import List


def train_collate_fn(batch):
    data, data_mask, labels,labels_used_layer = zip(*batch)
    # Find the maximum length in the batch
    max_length = max(d.size(1) for d in data)
    
    # Pad the data to the maximum length
    padded_data = [torch.cat([d, torch.ones(d.size(0), max_length - d.size(1))*(-1)], dim=1) if d.size(1) < max_length else d for d in data]
    padded_data = torch.stack(padded_data, dim=0)
    
    # change the zero input to -1
    mask = padded_data <= 0
    padded_data[mask] = -1
    
    # Labels are already tensors, no need to convert
    labels = torch.stack(labels)  # Stack labels into a single tensor
    
    # Labels do not need padding
    padded_data_mask = (padded_data[:,1, :] == -1) & (padded_data[:,2, :] == -1)
    
    # labels used 
    labels_uselayer = torch.stack(labels_used_layer)
    return padded_data,padded_data_mask,labels,labels_uselayer

def test_collate_fn(batch):
    data, data_mask = zip(*batch)
    # Find the maximum length in the batch
    max_length = max(d.size(1) for d in data)
    
    # Pad the data to the maximum length
    padded_data = [torch.cat([d, torch.ones(d.size(0), max_length - d.size(1))*(-1)], dim=1) if d.size(1) < max_length else d for d in data]
    padded_data = torch.stack(padded_data, dim=0)
    
    # change the zero input to -1
    mask = padded_data <= 0
    padded_data[mask] = -1
    
    # Labels do not need padding
    padded_data_mask = (padded_data[:,1, :] == -1) & (padded_data[:,2, :] == -1)
    
    return padded_data,padded_data_mask

class DispersionDatasets(Dataset):
    def __init__(self, 
                 input_data_path: str = "", 
                 input_label_path: str = "", 
                 train: bool = True,
                 interp_layer: bool = False,
                 layer_thickness: float = 0.5,
                 layer_number: int = 100,
                 layer_used_range: List[float] = [0, 100],
                 layer_interp_kind: str = "nearest",
                 num_workers: int = 0,
                 augmentation_train_data: bool = True,
                 noise_level: float = 0.02,
                 mask_ratio: float = 0.1,
                 remove_phase_ratio: float = 0.1,
                 remove_group_ratio: float = 0.1,
                 max_masking_length: int = 30):
        
        self.input_data_path = input_data_path
        self.input_label_path = input_label_path
        self.layer_thickness = layer_thickness
        self.layer_number = layer_number
        self.layer_interp_kind = layer_interp_kind
        self.layer_used_start = layer_used_range[0]
        self.layer_used_end = layer_used_range[1]
        self.augmentation_train_data = augmentation_train_data
        
        # Augmentation parameters
        self.noise_level = noise_level
        self.mask_ratio = mask_ratio
        self.remove_phase_ratio = remove_phase_ratio
        self.remove_group_ratio = remove_group_ratio
        self.max_masking_length = max_masking_length
        
        # Load and process input dataset
        self.input_dataset = self._load_input_data(input_data_path)
        self.input_masks = (self.input_dataset[:, 1, :] == -1) & (self.input_dataset[:, 2, :] == -1)

        # Load and process output dataset
        self.train = train
        if train:
            self.output_dataset = self._load_output_data(input_label_path, interp_layer, num_workers)
    
    def _load_input_data(self, path: str) -> torch.Tensor:
        """Load input dataset from the specified path."""
        try:
            input_dataset = np.load(path)["data"].transpose(0, 2, 1)
            return torch.tensor(input_dataset, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error loading input data from {path}: {e}")

    def _load_output_data(self, path: str, interp_layer: bool, num_workers: int) -> torch.Tensor:
        """Load output dataset from the specified path and optionally interpolate."""
        try:
            output_dataset = np.load(path)["data"].transpose(0, 2, 1)
            if interp_layer:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    output_dataset = list(executor.map(self._interp_vs, output_dataset))
                output_dataset = np.array(output_dataset)
            return torch.tensor(output_dataset, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Error loading output data from {path}: {e}")

    def augmentation(self, input_data):
        """Apply data augmentation by adding noise, masking, or removing phase/group velocities."""
        
        # Early exit if no augmentation is needed
        if self.noise_level <= 0 and self.mask_ratio <= 0 and self.remove_group_ratio <= 0 and self.remove_phase_ratio <= 0:
            return input_data

        # Add Gaussian noise if noise_level > 0
        if self.noise_level > 0:
            input_data = add_gaussian_noise(input_data, noise_level=self.noise_level)
        
        # Apply random masking if mask_ratio > 0
        if self.mask_ratio > 0:
            input_data = random_masking(input_data, mask_ratio=self.mask_ratio)
        
        # Randomly remove phase or group velocity if either ratio > 0
        if self.remove_group_ratio > 0 or self.remove_phase_ratio > 0:
            input_data = random_remove_phase_or_group(
                input_data, 
                remove_phase_ratio=self.remove_phase_ratio, 
                remove_group_ratio=self.remove_group_ratio, 
                masking_value=-1
            )
        
        return input_data


    def _interp_vs(self, output_data):
        """Interpolate 1D velocity model."""
        depth, vs = output_data[0, :], output_data[1, :]
    
        # Create an interpolation function
        F = interpolate.interp1d(depth, vs, kind=self.layer_interp_kind, fill_value="extrapolate")
        
        # Generate interpolated depth points
        interp_depth = np.arange(depth.min(), depth.max(), self.layer_thickness)
        interp_vs = F(interp_depth)
        
        # Check the length of the interpolated velocity data
        num_interp_points = len(interp_vs)

        if num_interp_points < self.layer_number:
            # Calculate the number of elements to add
            num_to_add = self.layer_number - num_interp_points
            
            # Calculate the additional depths and vs values
            additional_depths = interp_depth[-1] + self.layer_thickness * np.arange(1, num_to_add + 1)
            additional_vs = np.full(num_to_add, vs[-1])  # Use the last vs value for padding
            
            # Concatenate the additional values
            interp_depth = np.concatenate((interp_depth, additional_depths))
            interp_vs = np.concatenate((interp_vs, additional_vs))
        else:
            # Trim the interpolated arrays to the desired layer number
            interp_depth = interp_depth[:self.layer_number]
            interp_vs = interp_vs[:self.layer_number]

        return np.vstack((interp_depth, interp_vs))

    
    def vary_length(self, input_data, masking_value=-1, min_data_length=30, min_end_idx=105):
        """
        Randomly vary the length of valid data by masking parts of the input data.

        Parameters:
        -----------
        input_data : torch.Tensor or np.ndarray
            Input data where:
            - Row 0: Periods
            - Row 1: Phase velocities
            - Row 2: Group velocities
        
        masking_value : int, optional
            The value to use for masking (default is -1).

        min_data_length : int, optional
            Minimum length of the valid data region (default is 30).

        min_end_idx : int, optional
            Minimum end index for valid data (default is 105).

        Returns:
        --------
        torch.Tensor or np.ndarray
            The input data with regions masked according to the specified rules.
        """
        num_points = input_data.shape[1]
        
        # Randomly choose the starting index and length for the valid data
        mask_begin_idx = np.random.randint(0, 50)
        mask_length = np.random.randint(min_data_length, max(min_data_length + 1, num_points - min_data_length))
        mask_end_idx = mask_begin_idx + mask_length
        
        # Ensure the mask end index is at least `min_end_idx`
        mask_end_idx = max(min_end_idx, mask_end_idx)
        
        # Ensure mask_end_idx does not exceed the length of input_data
        mask_end_idx = min(mask_end_idx, num_points)

        # Apply masking to regions outside the valid range
        input_data[1:, :mask_begin_idx] = masking_value  # Mask the beginning
        input_data[1:, mask_end_idx:] = masking_value    # Mask the end
        
        return input_data


    def __getitem__(self, index):
        """
            input_data:[3,n]:period phase_vel group_vel
            output_data: [2,n]: thickness,vs
            input_mask: ignore the input or not
            self.layer_usage: usage layer
        """
        input_data = self.input_dataset[index].clone()
        if self.augmentation_train_data and self.train:
            input_data = self.augmentation(input_data)

        # Vary input length
        input_data = self.vary_length(input_data)
        
        # Masks
        input_mask = (input_data[1, :] <= 0) & (input_data[2, :] <= 0)
        phase_mask = input_data[1, :] > 0
        group_mask = input_data[2, :] > 0

        # Caching phase and group data
        phase_data = input_data[:, phase_mask]
        group_data = input_data[:, group_mask]
        
        # Phase depth calculation
        if phase_mask.any():
            phase_min_period = phase_data[0, :].min()
            phase_max_period = phase_data[0, :].max()
            phase_velocity = phase_data[1, :]
            max_depth_phase = 1.1 * (phase_max_period * phase_velocity[phase_data[0, :].argmax()])
            min_depth_phase = (1/3) * (phase_min_period * phase_velocity[phase_data[0, :].argmin()])
        else:
            min_depth_phase, max_depth_phase = None, None

        # Group depth calculation
        if group_mask.any():
            group_min_period = group_data[0, :].min()
            group_max_period = group_data[0, :].max()
            group_velocity = group_data[2, :]
            max_depth_group = 1.1 * (group_max_period * group_velocity[group_data[0, :].argmax()])
            min_depth_group = (1/2) * (group_min_period * group_velocity[group_data[0, :].argmin()])
        else:
            min_depth_group, max_depth_group = None, None

        # Calculate min and max depth indices
        min_depth = min(filter(None, [min_depth_phase, min_depth_group]), default=None)
        max_depth = max(filter(None, [max_depth_phase, max_depth_group]), default=None)

        min_depth_idx = max(0, int(min_depth // 0.5)) if min_depth is not None else 0
        max_depth_idx = min(400, int(max_depth // 0.5)) if max_depth is not None else 400

        if self.train:
            output_data = self.output_dataset[index]
            used_layer = torch.tensor([min_depth_idx, max_depth_idx], dtype=torch.int)
            return input_data, input_mask, output_data, used_layer
        else:
            return input_data, input_mask

    def __len__(self):
        return len(self.input_dataset)
