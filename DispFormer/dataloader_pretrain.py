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
                 num_workers      =4,
                 augmentation_train_data = True,
                 noise_level      = 0.02,
                 mask_ratio       = 0.1,
                 remove_phase_ratio=0.1,
                 remove_group_ratio=0.1,
                 max_masking_length= 30
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

    def augmentation(self, input_data):
        """Augmentation"""
        if self.noise_level>0:
            input_data = add_gaussian_noise(input_data,noise_level=self.noise_level)
            
        if self.mask_ratio>0:
            input_data = random_masking(input_data,mask_ratio=self.mask_ratio)
            
        if self.remove_group_ratio>0 or self.remove_phase_ratio >0:
            input_data = random_remove_phase_or_group(input_data,remove_phase_ratio=self.remove_phase_ratio,remove_group_ratio=self.remove_group_ratio,masking_value=-1)
        
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
    
    def vary_length(self, input_data,masking_value=-1,min_data_length=30,min_end_idx = 105):
        # Determine the number of mask values to add
        mask_begin_idx = np.random.randint(0, 50)
        mask_length    = np.random.randint(min_data_length, np.max([min_data_length+1,input_data.shape[1]-min_data_length]))
        mask_end_idx   = mask_begin_idx + mask_length
        mask_end_idx   = np.max([min_end_idx,mask_end_idx])
        input_data[1:,:mask_begin_idx]    = masking_value
        input_data[1:,mask_end_idx:]      = masking_value
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
        input_data = self.vary_length(input_data)
        
        input_mask = (input_data[1, :] <= 0) & (input_data[2, :] <= 0)
        phase_mask = input_data[1, :] > 0
        group_mask = input_data[2, :] > 0
        if sum(phase_mask)>0:
            phase_min_period,phase_max_period = input_data[0,:][phase_mask].min(),input_data[0,:][phase_mask].max()
            # the maximum depth
            max_depth_phase = 1.1 * (phase_max_period*input_data[1,np.argwhere(input_data[0,:]==phase_max_period)[0]])[0]
            # the minimum depth 
            min_depth_phase = 1/3 * (phase_min_period*input_data[1,np.argwhere(input_data[0,:] == phase_min_period)[0]])[0]
        else:
            min_depth_phase = None
            max_depth_phase = None
            
        if sum(group_mask)>0:
            group_min_period,group_max_period = input_data[0,:][group_mask].min(),input_data[0,:][group_mask].max()
            # the maximum depth
            max_depth_group = 1.1 * (group_max_period*input_data[2,np.argwhere(input_data[0,:]==group_max_period)[0]])[0]
            # the minimum depth 
            min_depth_group = 1/2 * (group_min_period*input_data[2,np.argwhere(input_data[0,:]==group_min_period)[0]])[0]
        else:
            min_depth_group = None
            max_depth_group = None
        
        # the maximum depth
        if min_depth_group is None:
            min_depth = min_depth_phase
        elif min_depth_phase is None:
            min_depth = min_depth_group
        else:
            min_depth       = np.min([min_depth_phase,min_depth_group])
        min_depth_idx   = int(min_depth//0.5)
        
        # the maximum depth
        if max_depth_group is None:
            max_depth = max_depth_phase
        elif max_depth_phase is None:
            max_depth = max_depth_group
        else:
            max_depth       = np.max([max_depth_phase,max_depth_group])
        max_depth_idx   = np.min([400,int(max_depth//0.5)])
        
        if self.train:
            output_data    = self.output_dataset[index]
            used_layer     = torch.tensor([min_depth_idx,max_depth_idx], dtype=torch.int)
            return input_data, input_mask,output_data, used_layer
        else:
            return input_data, input_mask

    def __len__(self):
        return len(self.input_dataset)
