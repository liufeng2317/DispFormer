import numpy as np 
import torch


def add_gaussian_noise(input_data,noise_level=0.05):
    """
    Apply Gaussian noise to phase_vel and group_vel channels.
    """
    phase_vel = input_data[1, :]  # phase_vel channel
    group_vel = input_data[2, :]  # group_vel channel
    
    # Compute standard deviation of the channel data
    phase_vel_std = phase_vel[phase_vel != -1].std()
    group_vel_std = group_vel[group_vel != -1].std()
    
    # Define noise standard deviations (1% of the standard deviation)
    phase_vel_noise_std = noise_level * phase_vel_std
    group_vel_noise_std = noise_level * group_vel_std
    
    # Generate Gaussian noise
    phase_vel_noise = torch.normal(mean=0.0, std=phase_vel_noise_std, size=phase_vel.size())
    group_vel_noise = torch.normal(mean=0.0, std=group_vel_noise_std, size=group_vel.size())
    
    # Add noise to the channels
    noisy_phase_vel = phase_vel + phase_vel_noise
    noisy_group_vel = group_vel + group_vel_noise
    
    # Update input_data with noisy values
    mask = (input_data[1, :] == -1) & (input_data[2, :] == -1)
    input_data[1, :] = noisy_phase_vel
    input_data[2, :] = noisy_group_vel
    input_data[1,mask] = -1
    input_data[2,mask] = -1
    return input_data

def random_masking(input_data, mask_ratio=0.1):
    """
    Apply random masking to simulate missing data.
    Ensure that the first and last two points are not masked.
    """
    # Get the number of points in the data
    num_points = input_data.shape[1]
    
    # Calculate the number of points that can be masked
    num_maskable_points = num_points - 4  # Reserve first and last two points
    
    if num_maskable_points <= 0:
        # Not enough points to mask
        return input_data
    
    # Generate mask for the maskable range
    maskable_data = input_data  # Exclude first and last two points
    mask1 = torch.rand(maskable_data.shape[1]) < mask_ratio
    mask2 = torch.rand(maskable_data.shape[1]) < mask_ratio
    masked_data = maskable_data.clone()
    masked_data[1,mask1] = -1  # Assuming -1 is used to indicate masked/invalid data
    masked_data[2,mask2] = -1  # Assuming -1 is used to indicate masked/invalid data
    
    # Reconstruct data with unmasked edges
    data_with_mask = masked_data
    
    return data_with_mask

def begin_end_masking(input_data,masking_value=-1,max_masking_length=10):
    # Determine the number of mask values to add
    mask_length = np.random.randint(0, max_masking_length + 1)
    
    # determin mask to being or end
    mask_begin = np.random.randint(-5, 5) >0
    mask_phase = np.random.randint(-5, 5) >0
    mask_group = np.random.randint(-5, 5) >0    
    if mask_begin:
        if mask_phase:
            input_data[1,:np.min([mask_length,input_data.shape[1]])] = masking_value
        if mask_group:
            input_data[2,:np.min([mask_length,input_data.shape[1]])] = masking_value
    else:
        if mask_phase:
            input_data[1,np.min([input_data.shape[1],input_data.shape[1]-mask_length]):] = masking_value
        if mask_group:
            input_data[2,np.min([input_data.shape[1],input_data.shape[1]-mask_length]):] = masking_value
    return input_data

def random_remove_phase_or_group(input_data,remove_phase_ratio = 0.1,remove_group_ratio = 0.1,masking_value=-1):
    p_remove_phase = np.random.random()
    p_remove_group = np.random.random()
    if p_remove_phase < remove_phase_ratio and p_remove_group > remove_phase_ratio:
        input_data[1,:] =  masking_value
    if p_remove_phase > remove_phase_ratio and p_remove_group < remove_group_ratio:
        input_data[2,:] = masking_value
    return input_data

def add_random_padding(input_data, padding_value=-1, max_padding_length=10):
    """
    Apply random padding of padding_value to the end of the input_data.

    Parameters:
    input_data (ndarray): Input data of shape (3, seq_length).
    padding_value (float): The value used for padding.
    max_padding_length (int): The maximum number of padding values to add.

    Returns:
    ndarray: The input_data with random padding applied.
    """
    seq_length = input_data.shape[1]
    
    # Determine the number of padding values to add
    padding_length = np.random.randint(0, max_padding_length + 1)
    
    # Create padding array
    padding = torch.tensor(np.full((3, padding_length), padding_value))
    
    left_or_right = np.random.randint(-5,5)
    if left_or_right>0:
        # Append padding to the data
        padded_data = torch.concatenate((input_data, padding), axis=1)
    else:
        padded_data = torch.concatenate((padding,input_data), axis=1)
    return padded_data