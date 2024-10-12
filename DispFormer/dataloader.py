import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor
from .data_augmentation import *


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
    def __init__(self, input_data_path="", 
                 input_label_path="", 
                 train            = True,
                 interp_layer     = False,
                 layer_thickness  = 0.5,
                 layer_number     = 100,
                 layer_used_range = [0,100],
                 layer_interp_kind="nearest",
                 num_workers=4,
                 augmentation_train_data = True,
                 noise_level=0.02,
                 mask_ratio=0.1,
                 remove_phase_ratio=0.1,
                 remove_group_ratio=0.1,
                 max_masking_length = 30
                 ):  # Add num_workers for parallel processing
        self.input_data_path         = input_data_path
        self.input_label_path        = input_label_path
        self.layer_thickness         = layer_thickness
        self.layer_number            = layer_number
        self.layer_interp_kind       = layer_interp_kind
        self.layer_used_start        = layer_used_range[0]
        self.layer_used_end          = layer_used_range[1]
        self.augmentation_train_data = augmentation_train_data
        
        # Augmentation parameters
        self.noise_level = noise_level
        self.mask_ratio  = mask_ratio
        self.remove_phase_ratio = remove_phase_ratio
        self.remove_group_ratio = remove_group_ratio
        self.max_masking_length = max_masking_length
        
        # Load and process input dataset [period, phase velocity, group velocity]
        input_dataset       = np.load(input_data_path)["data"].transpose(0, 2, 1)
        self.input_dataset  = torch.tensor(input_dataset, dtype=torch.float32)
        self.input_masks    = (self.input_dataset[:, 1, :] == -1) & (self.input_dataset[:, 2, :] == -1)

        # Load and process output dataset
        self.train = train
        if train:
            output_dataset = np.load(input_label_path)["data"].transpose(0, 2, 1)

            if interp_layer:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    output_dataset = list(executor.map(self.interp_vs, output_dataset))
                output_dataset = np.array(output_dataset)
            self.output_dataset = torch.tensor(output_dataset, dtype=torch.float32)

            self.used_layers = torch.zeros((self.output_dataset.shape[0], 2), dtype=torch.int)
            for i in range(self.output_dataset.shape[0]):
                air_layer_mask = self.output_dataset[i, 1, :] < 0
                water_layer_mask = self.output_dataset[i, 1, :] == 0
                air_layers_num = int(torch.sum(air_layer_mask))
                water_layers_num = int(torch.sum(water_layer_mask))
                used_layer = torch.tensor([max(air_layers_num + water_layers_num, self.layer_used_start), self.layer_used_end], dtype=torch.int)
                self.used_layers[i] = used_layer

    def augmentation(self, input_data):
        """Augmentation"""
        input_data = add_gaussian_noise(input_data,noise_level=self.noise_level)
        input_data = random_masking(input_data,mask_ratio=self.mask_ratio)
        input_data = begin_end_masking(input_data,masking_value=-1,max_masking_length=np.random.randint(self.max_masking_length))
        input_data = random_remove_phase_or_group(input_data,remove_phase_ratio=self.remove_phase_ratio,remove_group_ratio=self.remove_group_ratio,masking_value=-1)
        # input_data = add_random_padding(input_data,padding_value=-1,max_padding_length=np.random.randint(20))
        return input_data

    def interp_vs(self, output_data):
        """Interpolate 1D velocity model"""
        depth, vs = output_data[0, :], output_data[1, :]
        F = interpolate.interp1d(depth, vs, kind=self.layer_interp_kind, fill_value="extrapolate")
        interp_depth = np.arange(depth.min(), depth.max(), self.layer_thickness)
        interp_vs = F(interp_depth)
        
        # padding or clip the datasets
        if len(interp_vs) < self.layer_number:
            # Calculate the number of elements to add
            num_to_add = self.layer_number - len(interp_vs)
            # Calculate the additional depths and vs values
            additional_depths = interp_depth[-1] + self.layer_thickness * np.arange(1, num_to_add + 1)
            additional_vs = np.full(num_to_add, vs[-1])  # Use the last vs value for padding
            # Append the additional values to interp_depth and interp_vs
            interp_depth = np.concatenate([interp_depth, additional_depths])
            interp_vs = np.concatenate([interp_vs, additional_vs])
        else:
            interp_depth = interp_depth[:self.layer_number]
            interp_vs = interp_vs[:self.layer_number]
        return np.vstack((interp_depth, interp_vs))

    def __getitem__(self, index):
        """
            input_data:[3,n]:period phase_vel group_vel
            output_data: [2,n]: thickness,vs
            input_mask: ignore the input or not
            self.layer_usage: usage layer
        """
        input_data = self.input_dataset[index]
        if self.augmentation_train_data and self.train:
            input_data = self.augmentation(input_data)
            input_mask = (input_data[1, :] <= 0) & (input_data[2, :] <= 0)
        else:
            input_mask = self.input_masks[index]
        
        if self.train:
            output_data      = self.output_dataset[index]
            used_layer       = self.used_layers[index]
            return input_data, input_mask,output_data, used_layer
        else:
            return input_data, input_mask

    def __len__(self):
        return len(self.input_dataset)


# ############# normalize version ##################
# class DispersionDatasets(Dataset):
#     def __init__(self, input_data_path="", 
#                  input_label_path="", 
#                  train=True,
#                  interp_layer=False,
#                  layer_thickness=0.5,
#                  layer_number=100,
#                  layer_used_range=[0,100],
#                  layer_interp_kind="nearest",
#                  augmentation_train_data=True,
#                  num_workers=4):  # Add num_workers for parallel processing
#         self.input_data_path = input_data_path
#         self.input_label_path = input_label_path
#         self.layer_thickness = layer_thickness
#         self.layer_number = layer_number
#         self.layer_interp_kind = layer_interp_kind
#         self.layer_used_start = layer_used_range[0]
#         self.layer_used_end = layer_used_range[1]
#         self.augmentation_train_data = augmentation_train_data
        
#         # Load and process input dataset [period, phase velocity, group velocity]
#         input_dataset = np.load(input_data_path)["data"].transpose(0, 2, 1)
#         self.input_dataset = torch.tensor(input_dataset, dtype=torch.float32)

#         self.input_masks = (self.input_dataset[:, 1, :] == -1) & (self.input_dataset[:, 2, :] == -1)

#         # Load and process output dataset
#         self.train = train
#         if train:
#             output_dataset = np.load(input_label_path)["data"].transpose(0, 2, 1)

#             if interp_layer:
#                 with ThreadPoolExecutor(max_workers=num_workers) as executor:
#                     output_dataset = list(executor.map(self.interp_vs, output_dataset))
#                 output_dataset = np.array(output_dataset)
#             self.output_dataset = torch.tensor(output_dataset, dtype=torch.float32)

#             self.used_layers = torch.zeros((self.output_dataset.shape[0], 2), dtype=torch.int)
#             for i in range(self.output_dataset.shape[0]):
#                 air_layer_mask = self.output_dataset[i, 1, :] < 0
#                 water_layer_mask = self.output_dataset[i, 1, :] == 0
#                 air_layers_num = int(torch.sum(air_layer_mask))
#                 water_layers_num = int(torch.sum(water_layer_mask))
#                 used_layer = torch.tensor([max(air_layers_num + water_layers_num, self.layer_used_start), self.layer_used_end], dtype=torch.int)
#                 self.used_layers[i] = used_layer

#     def augmentation(self, input_data, noise_level=0.02, mask_ratio=0.1):
#         """Augmentation"""
#         input_data = add_gaussian_noise(input_data, noise_level=noise_level)
#         input_data = random_masking(input_data, mask_ratio=mask_ratio)
#         input_data = add_random_padding(input_data, padding_value=-1, max_padding_length=np.random.randint(20))
#         return input_data

#     def interp_vs(self, output_data):
#         """Interpolate 1D velocity model"""
#         depth, vs = output_data[0, :], output_data[1, :]
#         F = interpolate.interp1d(depth, vs, kind=self.layer_interp_kind, fill_value="extrapolate")
#         interp_depth = np.arange(depth.min(), depth.max(), self.layer_thickness)
#         interp_vs = F(interp_depth)
        
#         # padding or clip the datasets
#         if len(interp_vs) < self.layer_number:
#             # Calculate the number of elements to add
#             num_to_add = self.layer_number - len(interp_vs)
#             # Calculate the additional depths and vs values
#             additional_depths = interp_depth[-1] + self.layer_thickness * np.arange(1, num_to_add + 1)
#             additional_vs = np.full(num_to_add, vs[-1])  # Use the last vs value for padding
#             # Append the additional values to interp_depth and interp_vs
#             interp_depth = np.concatenate([interp_depth, additional_depths])
#             interp_vs = np.concatenate([interp_vs, additional_vs])
#         else:
#             interp_depth = interp_depth[:self.layer_number]
#             interp_vs = interp_vs[:self.layer_number]
#         return np.vstack((interp_depth, interp_vs))

#     def __getitem__(self, index):
#         """
#             input_data:[3,n]:period phase_vel group_vel
#             output_data: [2,n]: thickness,vs
#             input_mask: ignore the input or not
#             self.layer_usage: usage layer
#         """
#         input_data = self.input_dataset[index]
        
#         if self.augmentation_train_data and self.train:
#             input_data = self.augmentation(input_data, noise_level=0.02, mask_ratio=0.1)
#             input_mask = (input_data[1, :] < 0) & (input_data[2, :] <0)
#         else:
#             input_mask = self.input_masks[index]
        
#         # Normalize phase and group velocities to 0-1
#         phase_velocity = input_data[1, :]
#         phase_velocity_mask = phase_velocity>0
#         group_velocity = input_data[2, :]
#         group_velocity_mask = group_velocity>0
#         phase_min,phase_max = phase_velocity[phase_velocity_mask].min(),phase_velocity[phase_velocity_mask].max()
#         group_min,group_max = group_velocity[group_velocity_mask].min(),group_velocity[group_velocity_mask].max()
#         input_data[1, phase_velocity_mask] = (phase_velocity[phase_velocity_mask] - phase_min) / (phase_max - phase_min)
#         input_data[2, group_velocity_mask] = (group_velocity[group_velocity_mask] - group_min) / (group_max - group_min)
        
#         if self.train:
#             output_data = self.output_dataset[index]
#             used_layer  = self.used_layers[index]
#             return input_data, input_mask, output_data, used_layer
#         else:
#             return input_data, input_mask
    
#     def __len__(self):
#         return len(self.input_dataset)