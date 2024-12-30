import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor
from .data_augmentation import *
from typing import List

def train_collate_fn(batch):
    """
    Collate function for batching data during training. This function takes a list of samples (tuples of data, masks, 
    labels, and labels_used_layer), processes them to ensure uniformity in sequence length, applies padding, 
    and prepares the data for feeding into a neural network model.

    Parameters:
    - batch (list of tuples)                : A list where each element is a tuple containing:
        - data (torch.Tensor)               : The input data for the model, typically of shape (batch_size, seq_length).
        - data_mask (torch.Tensor)          : A mask for the input data, indicating valid (1) and padded (0) values.
        - labels (torch.Tensor)             : The target labels corresponding to the input data.
        - labels_used_layer (torch.Tensor)  : A tensor indicating which layers were used for the labels.

    Returns:
    - padded_data (torch.Tensor): The input data padded to the maximum sequence length within the batch, 
      with padding values set to -1 for consistency.
    - padded_data_mask (torch.Tensor): A mask for the padded input data, marking positions that are padded.
    - labels (torch.Tensor): A stacked tensor of labels for the entire batch.
    - labels_uselayer (torch.Tensor): A stacked tensor of labels_used_layer for the entire batch.

    This function performs the following:
        1. Pads the input data sequences to the maximum length within the batch.
        2. Adjusts any input values that are zero to -1 (for consistency in handling padding).
        3. Creates a mask to indicate padded positions in the input data.
        4. Stacks the labels and labels_used_layer tensors into single tensors for easy batch processing.
    """
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
    """
    Collate function for batching data during testing. This function takes a list of samples (tuples of data and 
    masks), processes them to ensure uniform sequence length, applies padding, and prepares the data for testing 
    a model.

    Parameters:
    - batch (list of tuples): A list where each element is a tuple containing:
        - data (torch.Tensor): The input data for the model, typically of shape (batch_size, seq_length).
        - data_mask (torch.Tensor): A mask for the input data, indicating valid (1) and padded (0) values.

    Returns:
    - padded_data (torch.Tensor): The input data padded to the maximum sequence length within the batch, 
      with padding values set to -1 for consistency.
    - padded_data_mask (torch.Tensor): A mask for the padded input data, marking positions that are padded.

    This function performs the following:
        1. Pads the input data sequences to the maximum length within the batch.
        2. Adjusts any input values that are zero to -1 (for consistency in handling padding).
        3. Creates a mask to indicate padded positions in the input data.
    """
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
    """ used in zero-shot & few-shot learning
    This class represents a dataset for DispFormer, suitable for training or evaluating a model
    that predicts subsurface velocity models.The dataset supports both training and evaluation modes, 
    with options to augment the data, add noise, and mask parts of the input sequences to improve model robustness and generalization.

    Attributes:
    - input_data_path (str): Path to the input dispersion data file, which contains three columns per sample: 
      [period, phase velocity, group velocity].
    - input_label_path (str): Path to the velocity model file, which contains four columns per sample: 
      [depth, P-wave velocity (vp), S-wave velocity (vs), density (rho)].
    - train (bool): A flag indicating whether the dataset is used for training or testing.
    - interp_layer (bool): Whether to automatically interpolate the layers to have equal thickness.
        - layer_thickness (float): The thickness of each interpolated layer (used when `interp_layer=True`).
        - layer_number (int): The number of layers in the velocity model, used for interpolation and layer extraction.
        - layer_interp_kind (str): The interpolation method for adjusting the layer thickness, e.g., 'nearest', 'linear'.
    - layer_used_range (List[float]): The range of layers (depth range) to use from the velocity model. (not used in pre-training)
    - num_workers (int): The number of workers for parallel data loading (used when loading large datasets).
    - augmentation_train_data (bool): Whether to apply data augmentation during training (such as noise, masking, etc.).
        - noise_level (float): The standard deviation of the noise added to the dispersion data during training (for augmentation).
        - mask_ratio (float): The fraction of the input data to randomly mask during training as part of the data augmentation.
        - remove_phase_ratio (float): The fraction of phase velocity data to randomly remove for augmentation during training.
        - remove_group_ratio (float): The fraction of group velocity data to randomly remove for augmentation during training.
        - max_masking_length (int): The maximum length of sequences to mask in the dispersion data during training.

    Methods:
    - __len__: Returns the total number of samples in the dataset.
    - __getitem__: Loads and returns a sample from the dataset, including both input data and corresponding labels.
    - _load_input_data: Loads the dispersion data from the input file (not implemented here, can be customized).
    - _load_output_data: Loads the velocity model data from the label file (not implemented here, can be customized).
    - _interp_vs: Interpolates the layers to a uniform thickness based on the specified settings.
    - augmentation: Optionally applies augmentation techniques (e.g., noise, masking) to the training data.

    Usage:
    The `DispersionDatasets` class is designed to be used with PyTorch's `DataLoader` for efficient data loading 
    and batching during training and evaluation. It can handle both the dispersion data (input) and the velocity 
    model labels (output), along with optional data augmentation and masking for training.

    Example:
        ```python
        dataset = DispersionDatasets(input_data_path="path_to_dispersion_data.csv", 
                                    input_label_path="path_to_velocity_model.csv", 
                                    train=True, 
                                    augmentation_train_data=True)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        ```
    """
    def __init__(self, 
                 input_data_path: str = "", 
                 input_label_path: str = "", 
                 train: bool = True,
                 interp_layer: bool = False,
                 layer_thickness: float = 0.5,
                 layer_number: int = 100,
                 layer_used_range: List[float] = [0, 100],
                 layer_interp_kind: str = "nearest",
                 num_workers: int = 10,
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
        
        # Load input dataset
        self.input_dataset = self._load_input_data(input_data_path)
        self.input_masks = (self.input_dataset[:, 1, :] == -1) & (self.input_dataset[:, 2, :] == -1)
        
        self.train = train
        if train:
            # Load and process output dataset
            self.output_dataset = self._load_output_data(input_label_path, interp_layer, num_workers)
            self.used_layers = self._compute_used_layers()

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

    def _compute_used_layers(self):
        """Computes the used layers for each sample in the output dataset."""
        used_layers = torch.zeros((self.output_dataset.shape[0], 2), dtype=torch.int)
        for i in range(self.output_dataset.shape[0]):
            air_layer_mask = self.output_dataset[i, 1, :] < 0
            water_layer_mask = self.output_dataset[i, 1, :] == 0
            air_layers_num = int(torch.sum(air_layer_mask))
            water_layers_num = int(torch.sum(water_layer_mask))
            used_layer = torch.tensor([max(air_layers_num + water_layers_num, self.layer_used_start), self.layer_used_end], dtype=torch.int)
            used_layers[i] = used_layer
        return used_layers

    def augmentation(self, input_data):
        """Applies data augmentation methods to the input data."""
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

    def __getitem__(self, index):
        """Returns input and output data with masks."""
        input_data = self.input_dataset[index].clone()
        if self.augmentation_train_data and self.train:
            input_data = self.augmentation(input_data)
            input_mask = (input_data[1, :] <= 0) & (input_data[2, :] <= 0)
        else:
            input_mask = self.input_masks[index]
        
        if self.train:
            output_data = self.output_dataset[index]
            used_layer = self.used_layers[index]
            return input_data, input_mask, output_data, used_layer
        else:
            return input_data, input_mask

    def __len__(self):
        return len(self.input_dataset)