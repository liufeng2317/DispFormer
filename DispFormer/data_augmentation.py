import numpy as np 
import torch


def add_gaussian_noise(input_data, noise_level=0.05):
    """
    Apply Gaussian noise to phase velocity and group velocity channels.

    Parameters:
    -----------
    input_data : torch.Tensor
        A 2D tensor where:
        - Row 0: Periods
        - Row 1: Phase velocities
        - Row 2: Group velocities

    noise_level : float, optional
        The noise level as a fraction of the standard deviation of the channel data (default is 0.05).

    Returns:
    --------
    torch.Tensor
        The input data with added Gaussian noise to phase and group velocity channels.
    """
    # Mask invalid data points (-1) in phase and group velocities
    valid_phase_mask = input_data[1, :] != -1
    valid_group_mask = input_data[2, :] != -1

    # Compute standard deviation only for valid phase and group velocity values
    phase_vel_std = input_data[1, valid_phase_mask].std() if valid_phase_mask.any() else 0
    group_vel_std = input_data[2, valid_group_mask].std() if valid_group_mask.any() else 0

    # Define noise standard deviations
    phase_vel_noise_std = noise_level * phase_vel_std
    group_vel_noise_std = noise_level * group_vel_std

    # Generate Gaussian noise only for valid data points
    phase_vel_noise = torch.normal(mean=0.0, std=phase_vel_noise_std, size=input_data[1, valid_phase_mask].shape)
    group_vel_noise = torch.normal(mean=0.0, std=group_vel_noise_std, size=input_data[2, valid_group_mask].shape)

    # Add noise to the valid phase and group velocities
    input_data[1, valid_phase_mask] += phase_vel_noise
    input_data[2, valid_group_mask] += group_vel_noise

    return input_data

def random_masking(input_data, mask_ratio=0.1):
    """
    Apply random masking to simulate missing data.

    Parameters:
    -----------
    input_data : torch.Tensor
        A 2D tensor where:
        - Row 0: Periods
        - Row 1: Phase velocities
        - Row 2: Group velocities

    mask_ratio : float, optional
        The ratio of points to be randomly masked (default is 0.1).

    Returns:
    --------
    torch.Tensor
        The input data with randomly masked phase and group velocity values.
    """
    # Get the number of points in the data
    num_points = input_data.shape[1]

    # Check if there are enough points to apply masking
    if num_points <= 4:
        return input_data  # Not enough points to apply masking

    # Clone the input data to avoid modifying the original
    masked_data = input_data.clone()

    # Generate random mask based on the specified ratio
    mask = torch.rand(num_points) < mask_ratio

    # Apply the mask to both phase and group velocities
    masked_data[1, mask] = -1  # Mask phase velocity
    masked_data[2, mask] = -1  # Mask group velocity

    return masked_data

def begin_end_masking(input_data,masking_value=-1,max_masking_length=10):
    """
        Randomly masks values at the beginning or end of the input data to simulate missing data.

        Parameters:
        -----------
        input_data : np.ndarray
            A 2D array where:
            - Row 0: Periods
            - Row 1: Phase velocities
            - Row 2: Group velocities

        masking_value : int, optional
            Value used for masking (default is -1).

        max_masking_length : int, optional
            Maximum number of values to mask (default is 10).

        Returns:
        --------
        np.ndarray
            The input data with masked values.
    """
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

def random_remove_phase_or_group(input_data, remove_phase_ratio=0.1, remove_group_ratio=0.1, masking_value=-1):
    """
    Randomly removes phase or group velocity data based on specified removal ratios,
    ensuring that both phase and group velocities are not removed simultaneously.

    Parameters:
    -----------
    input_data : np.ndarray
        A 2D array where:
        - Row 0: Periods
        - Row 1: Phase velocities
        - Row 2: Group velocities

    remove_phase_ratio : float, optional
        Probability threshold for removing phase velocity data (default is 0.1).

    remove_group_ratio : float, optional
        Probability threshold for removing group velocity data (default is 0.1).

    masking_value : int, optional
        Value used for masking (default is -1).

    Returns:
    --------
    np.ndarray
        The input data with some phase or group velocity values removed.
    """
    random_vals = np.random.random(2)
    
    # Check if both should be masked, if so, only mask one randomly
    if random_vals[0] < remove_phase_ratio and random_vals[1] < remove_group_ratio:
        if np.random.random() < 0.5:
            input_data[1, :] = masking_value  # Mask phase velocity
        else:
            input_data[2, :] = masking_value  # Mask group velocity
    else:
        if random_vals[0] < remove_phase_ratio:
            input_data[1, :] = masking_value  # Mask phase velocity
        if random_vals[1] < remove_group_ratio:
            input_data[2, :] = masking_value  # Mask group velocity

    return input_data


def add_random_padding(input_data, padding_value=-1, max_padding_length=10):
    """
    Add random padding to the input data either at the beginning or the end.

    Parameters:
    -----------
    input_data : torch.Tensor
        A 2D tensor where the first dimension represents different channels 
        (e.g., periods, phase velocities, group velocities) and the second 
        dimension represents the sequence length.

    padding_value : int, optional
        The value to use for padding (default is -1).

    max_padding_length : int, optional
        The maximum length of padding to add (default is 10).

    Returns:
    --------
    torch.Tensor
        The input data with random padding added at either the beginning or the end.
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