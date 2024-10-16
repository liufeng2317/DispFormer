import numpy as np
from disba import PhaseDispersion,GroupDispersion

def gen_init_model(t,cg_obs,thick,area=False):
    """
    generate the initial model based on empirical formula 
    developed by Thomas M.Brocher (2005).
    ---------------------
    Input Parameters:
        t : 1D numpy array 
            => period of observaton dispersion points
        cg_obs: 1D numpy array 
            => phase velocity of observation dispersion points
        thick : 1D numpy array 
            => thickness of each layer
    Output: the initialize model
        thick : 1D numpy array 
            => thickness
        vs : 1D numpy array 
            => the shear wave velocity
        vp : 1D numpy array 
            => the compress wave velocity
        rho: 1D numpy array 
            => the density
    --------------------
    Output parameters:
        model:Dict 
            => the generated model
    """
    wavelength  = t*cg_obs
    nlayer      = len(thick)
    lambda2L    = 0.65      # the depth faction 0.63L
    beta        = 0.92      # the poisson's ratio
    eqv_lambda = lambda2L*wavelength
    lay_model = np.zeros((nlayer,2))
    lay_model[:,0] = thick
    for i in range(nlayer-1):
        if i == 0:
            up_bound = 0
        else:
            up_bound = up_bound + lay_model[i-1,0] # the top-layer's depth
        low_bound = up_bound + lay_model[i,0] # the botton-layer's depth
        # vs for every layer
        lambda_idx = np.argwhere((eqv_lambda>up_bound) & (eqv_lambda<low_bound))
        if len(lambda_idx)>0:
            lay_model[i,1] = np.max(cg_obs[lambda_idx])/beta # phase velocity -> vs
        else:
            lambda_idx = np.argmin(np.abs(eqv_lambda - low_bound))
            lay_model[i,1] = cg_obs[lambda_idx]/beta
    # set the last layer
    lay_model[nlayer-1,0] = 0
    lay_model[nlayer-1,1] = np.max(cg_obs)*1.1
    thick = lay_model[:,0]
    vs = lay_model[:,1]
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2+ 0.2683*vs**3 - 0.0251*vs**4
    depth = np.cumsum(thick)
    
    mask = depth>120
    vp[mask] = vs[mask]*1.79
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return thick,vp,vs,rho 
    else:
        return model

def gen_model(depth,vs,area=False,Brocher=True):
    """
    generate the initial model based on empirical formula 
    developed by Thomas M.Brocher (2005).
    ---------------------
    Input Parameters:
        thick : Array(1D) 
            => the thickness of layer 
        vs    : Array(1D)
            => the shear wave velocity
        area  : boolen 
            => the output format
    --------------------
    Output parameters:
        model:Dict 
            the generated model
    """
    depth       = np.array(depth)
    thickness   = np.diff(depth)
    thickness   = np.insert(thickness,-1,thickness[-1]) 
    vs = np.array(vs)
    if Brocher:
        vp = 0.9409 + 2.0947*vs - 0.8206*vs**2+ 0.2683*vs**3 - 0.0251*vs**4
    else:
        vp = 1.79*vs
    mask = depth>120
    vp[mask] = vs[mask]*1.79
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    model = {
        "thick":thickness,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return thickness,vp,vs,rho
    else:
        return model
    
def smooth_data(y, window_size=7):
    """
    Smooth the input data using a moving average filter, with padding to reduce edge effects.
    
    Parameters:
    y (array-like): The input data to be smoothed.
    window_size (int): Size of the moving average window (default is 7).
    
    Returns:
    y_smooth (array-like): Smoothed data.
    """

    # Save the first data point to preserve it after smoothing
    first_point = y[0]

    # Create a moving average window (a window of ones normalized by the window size)
    window = np.ones(window_size) / window_size

    # Pad the input data using 'reflect' mode to minimize edge effects during convolution
    y_padded = np.pad(y, (window_size // 2, window_size // 2), mode='reflect')

    # Apply the convolution between the padded data and the smoothing window
    y_smooth = np.convolve(y_padded, window, mode='valid')

    # Replace the first point of the smoothed data with the original first point
    y_smooth[0] = first_point
    
    return y_smooth


#################################################################
#################################################################

def NMSE(output, target):
    """
    Computes the Normalized Mean Squared Error (NMSE) between the output and target tensors.
    
    Parameters:
    output (torch.Tensor): Predicted output.
    target (torch.Tensor): Ground truth target.
    
    Returns:
    torch.Tensor: The NMSE value.
    """
    return torch.sum(((output - target) / target) ** 2)

def NMSE_np(output, target):
    """
    Computes the Normalized Mean Squared Error (NMSE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The NMSE value.
    """
    return np.sum(((output - target) / target) ** 2)


def MAPE_np(output, target):
    """
    Computes the Mean Absolute Percentage Error (MAPE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The MAPE value, expressed as a percentage.
    """
    return np.mean(np.abs((output - target) / target)) * 100


def MSE_np(output, target):
    """
    Computes the Mean Squared Error (MSE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The MSE value, scaled by 1000.
    """
    return np.mean((output - target) ** 2) * 1000


def MAE_np(output, target):
    """
    Computes the Mean Absolute Error (MAE) using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output.
    target (np.array): Ground truth target.
    
    Returns:
    float: The MAE value, scaled by 1000.
    """
    return np.mean(np.abs(output - target)) * 1000


def MAE_layers_np(output, target):
    """
    Computes the Mean Absolute Error (MAE) for each layer (dimension) of the output using NumPy arrays.
    
    Parameters:
    output (np.array): Predicted output with multiple layers.
    target (np.array): Ground truth target with multiple layers.
    
    Returns:
    np.array: The MAE value for each layer, scaled by 100.
    """
    results = np.mean(np.abs(output - target), axis=0) * 100
    return results


#################################################################
#################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os
def plot_matrix(matrix, learning_rates, sparse_nums, plot_base_path, 
                metric_name="Metric", save_name="",
                show=True):
    """
    Plot a 2D heatmap of the matrix with annotations and highlight the minimum value in each row.

    Args:
    - matrix (np.array): The matrix to be plotted.
    - learning_rates (list): List of learning rates (column labels).
    - sparse_nums (list): List of sparse numbers (row labels).
    - plot_base_path (str): Base path for saving the plot.
    - metric_name (str): Name of the metric the matrix represents (e.g., 'NMSE', 'MSE', 'MAE', 'MAPE').
    - save_name (str): Name of the file to save the plot.
    - show (bool): If True, display the plot. If False, close the plot after saving.

    Returns:
    None
    """
    
    # Convert the input matrix to a numpy array for easier plotting
    matrix_np = np.array(matrix)

    # Create a figure for the heatmap
    plt.figure(figsize=(8, 6))
    
    # Plot the matrix as a heatmap using a 'cool' colormap
    plt.imshow(matrix_np, cmap='cool', aspect='auto', interpolation='none')

    # Annotate each cell of the matrix with the corresponding value
    for i in range(len(sparse_nums)):
        for j in range(len(learning_rates)):
            plt.text(j, i, f"{matrix_np[i, j]:.2f}", ha='center', va='center', color='k')

    # Find and highlight the minimum value in each row
    ax = plt.gca()  # Get the current axes
    for i in range(len(sparse_nums)):
        min_index = np.argmin(matrix_np[i, :])  # Find the index of the minimum value in the row
        # Draw a red rectangle around the cell with the minimum value
        rect = patches.Rectangle((min_index - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Add a colorbar to the plot
    plt.colorbar(label=metric_name)

    # Set the x-axis and y-axis labels
    plt.xticks(np.arange(len(learning_rates)), learning_rates)
    plt.yticks(np.arange(len(sparse_nums)), sparse_nums)
    plt.xlabel('Learning Rate')
    plt.ylabel('Sparse Num')

    # Add a title to the plot
    plt.title(f'{metric_name} vs Learning Rate and Sparse Num')

    # Save the plot to a file if a save name is provided
    if not save_name == "":
        plt.savefig(os.path.join(plot_base_path, save_name), bbox_inches='tight', dpi=300)

    # Show the plot or close it based on the 'show' flag
    if show:
        plt.show()
    else:
        plt.close()

#################################################################
#################################################################
import torch
def cal_misfits_transformer(model,criterion,test_loader,device='cpu'):
    model.eval()  # Set the model to evaluation mode
    misfits = 0
    target_vs,inv_vs,target_used_layer = [],[],[]
    with torch.no_grad():
        loss_batch = 0
        for batch_inputs,batch_data_mask,batch_targets,batch_layer_usage in test_loader:
            batch_inputs, batch_targets,batch_data_mask = batch_inputs.to(device), batch_targets.to(device), batch_data_mask.to(device)
            input_data = batch_inputs[:,:,:].clone()
            input_mask = batch_data_mask[:,:].clone()

            outputs = model(input_data,input_mask)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i,0]:batch_layer_usage[i,1]], batch_targets[i,1,batch_layer_usage[i,0]:batch_layer_usage[i,1]])
            loss_batch += loss.item()
            inv_vs.extend(outputs.cpu().detach().numpy())
            target_vs.extend(batch_targets[:,:,:].cpu().detach().numpy())
            target_used_layer.extend(batch_layer_usage.cpu().detach().numpy())
            # break
        misfits += loss_batch
    return misfits

def predict_res_transformer(model,criterion,test_loader,device='cpu'):
    model.eval()  # Set the model to evaluation mode
    misfits = 0
    target_vs,inv_vs,target_used_layer,all_inputs = [],[],[],[]
    with torch.no_grad():
        loss_batch = 0
        for batch_inputs,batch_data_mask,batch_targets,batch_layer_usage in test_loader:
            batch_inputs, batch_targets,batch_data_mask = batch_inputs.to(device), batch_targets.to(device), batch_data_mask.to(device)
            input_data = batch_inputs[:,:,:].clone()
            input_mask = batch_data_mask[:,:].clone()

            outputs = model(input_data,input_mask)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i,0]:batch_layer_usage[i,1]], batch_targets[i,1,batch_layer_usage[i,0]:batch_layer_usage[i,1]])
            loss_batch += loss.item()
            inv_vs.extend(outputs.cpu().detach().numpy())
            target_vs.extend(batch_targets[:,:,:].cpu().detach().numpy())
            target_used_layer.extend(batch_layer_usage.cpu().detach().numpy())
            all_inputs.extend(input_data.cpu().detach().numpy())
            # break
        misfits += loss_batch
    inv_vs = np.array(inv_vs)
    target_vs = np.array(target_vs)
    all_inputs = np.array(all_inputs)
    return target_vs,inv_vs,all_inputs

def cal_misfits_cnn(model,criterion,test_loader,device='cpu'):
    model.eval()  # Set the model to evaluation mode
    misfits = 0
    target_vs,inv_vs,target_used_layer = [],[],[]
    with torch.no_grad():
        loss_batch = 0
        for batch_inputs,batch_data_mask,batch_targets,batch_layer_usage in test_loader:
            batch_inputs, batch_targets,batch_data_mask = batch_inputs.to(device), batch_targets.to(device), batch_data_mask.to(device)
            input_data = batch_inputs[:,:,:].clone()
            input_mask = batch_data_mask[:,:].clone()

            outputs = model(input_data)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i,0]:batch_layer_usage[i,1]], batch_targets[i,1,batch_layer_usage[i,0]:batch_layer_usage[i,1]])
            loss_batch += loss.item()
            inv_vs.extend(outputs.cpu().detach().numpy())
            target_vs.extend(batch_targets[:,:,:].cpu().detach().numpy())
            target_used_layer.extend(batch_layer_usage.cpu().detach().numpy())
            # break
        misfits += loss_batch
    return misfits

def predict_res_cnn(model,criterion,test_loader,device='cpu'):
    model.eval()  # Set the model to evaluation mode
    misfits = 0
    target_vs,inv_vs,target_used_layer,all_inputs = [],[],[],[]
    with torch.no_grad():
        loss_batch = 0
        for batch_inputs,batch_data_mask,batch_targets,batch_layer_usage in test_loader:
            batch_inputs, batch_targets,batch_data_mask = batch_inputs.to(device), batch_targets.to(device), batch_data_mask.to(device)
            input_data = batch_inputs[:,:,:].clone()
            input_mask = batch_data_mask[:,:].clone()

            outputs = model(input_data)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i,0]:batch_layer_usage[i,1]], batch_targets[i,1,batch_layer_usage[i,0]:batch_layer_usage[i,1]])
            loss_batch += loss.item()
            inv_vs.extend(outputs.cpu().detach().numpy())
            target_vs.extend(batch_targets[:,:,:].cpu().detach().numpy())
            target_used_layer.extend(batch_layer_usage.cpu().detach().numpy())
            all_inputs.extend(input_data.cpu().detach().numpy())
            # break
        misfits += loss_batch
    inv_vs = np.array(inv_vs)
    target_vs = np.array(target_vs)
    all_inputs = np.array(all_inputs)
    return target_vs,inv_vs,all_inputs

def predict_res_sfnet(model,criterion,test_loader,device='cpu'):
    model.eval()  # Set the model to evaluation mode
    misfits = 0
    target_vs,inv_vs,target_used_layer,all_inputs = [],[],[],[]
    with torch.no_grad():
        loss_batch = 0
        for batch_inputs,batch_data_mask,batch_targets,batch_layer_usage in test_loader:
            batch_inputs, batch_targets,batch_data_mask = batch_inputs[:,1:,:].to(device), batch_targets.to(device), batch_data_mask.to(device)
            input_data = batch_inputs[:,:,:].clone()
            input_mask = batch_data_mask[:,:].clone()

            outputs = model(input_data)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i,0]:batch_layer_usage[i,1]], batch_targets[i,1,batch_layer_usage[i,0]:batch_layer_usage[i,1]])
            loss_batch += loss.item()
            inv_vs.extend(outputs.cpu().detach().numpy())
            target_vs.extend(batch_targets[:,:,:].cpu().detach().numpy())
            target_used_layer.extend(batch_layer_usage.cpu().detach().numpy())
            all_inputs.extend(input_data.cpu().detach().numpy())
            # break
        misfits += loss_batch
    inv_vs = np.array(inv_vs)
    target_vs = np.array(target_vs)
    all_inputs = np.array(all_inputs)
    return target_vs,inv_vs,all_inputs

#################################################################
#################################################################
def plot_single_station_cmp_res(disp_loc,inputs_disp,target_vs,Transformer_inv_vs,
                                sta_idx,depth_idx,
                                save_path="",show=True):
    ############################################################
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plt.scatter(disp_loc[:,0],disp_loc[:,1],c = target_vs[:,1,depth_idx],s=5,cmap='jet_r')
    plt.scatter(disp_loc[sta_idx,0],disp_loc[sta_idx,1],s=60,facecolor=None,edgecolor='k',marker='v',label='select station')
    # plt.scatter(train_disp_loc[::sparse_num,0],train_disp_loc[::sparse_num,1],s=2,c='k',marker='.',label='training sets')
    plt.legend(fontsize =11,loc='upper left')
    # plt.title(f"depth:{depth_idx*0.5} km")
    plt.xlabel("Longitude (°)", fontsize=12)
    plt.ylabel("Latitude (°)", fontsize=12)
    plt.tick_params(labelsize=12)

    ############################################################
    plt.subplot(222)
    plt.step(target_vs[sta_idx,1,:],target_vs[sta_idx,0,:]       ,where='post',c='k',linestyle='--',label="True")
    plt.step(Transformer_inv_vs[sta_idx,:],target_vs[sta_idx,0,:]        ,where='post',c='g',linestyle='-' ,label="Transformer")
    plt.legend(fontsize =12)
    plt.gca().invert_yaxis()
    plt.xlabel("S-wave velocity (km/s)", fontsize=12)
    plt.ylabel("Depth (km)", fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid()

    ###########################################################
    mask = (inputs_disp[sta_idx,1,:]>0) + (inputs_disp[sta_idx,2,:]>0)
    t = inputs_disp[sta_idx,0,mask]

    Transformer_depth,Transformer_vs = np.arange(Transformer_inv_vs.shape[1])*0.5,Transformer_inv_vs[sta_idx]
    Transformer_thickness,Transformer_vp,Transformer_vs,Transformer_rho= gen_model(depth=Transformer_depth,vs=Transformer_vs,area=True)
    Transformer_vel_model = np.hstack((Transformer_thickness.reshape(-1,1),Transformer_vp.reshape(-1,1),Transformer_vs.reshape(-1,1),Transformer_rho.reshape(-1,1)))
    Transformer_pd = PhaseDispersion(*Transformer_vel_model.T)
    Transformer_gd = GroupDispersion(*Transformer_vel_model.T)
    Transformer_phase_disp = [Transformer_pd(t, mode=i, wave="rayleigh") for i in range(1)]
    Transformer_group_disp = [Transformer_gd(t, mode=i, wave='rayleigh') for i in range(1)]

    plt.subplot(223)
    mask = inputs_disp[sta_idx,1,:]>0
    plt.scatter(inputs_disp[sta_idx,0,mask],inputs_disp[sta_idx,1,mask]           ,c='k',s=30,label='target ')
    plt.scatter(Transformer_phase_disp[0].period,Transformer_phase_disp[0].velocity   ,c='g',s=10,label='Transformer')
    # plt.legend()
    plt.xlabel("Period (s)", fontsize=12)
    plt.ylabel("Phase velocity (km/s)", fontsize=12)
    plt.tick_params(labelsize=12)

    plt.subplot(224)
    mask = inputs_disp[sta_idx,2,:]>0
    plt.scatter(inputs_disp[sta_idx,0,mask],inputs_disp[sta_idx,2,mask]           ,c='k',s=30,label='target ')
    plt.scatter(Transformer_group_disp[0].period,Transformer_group_disp[0].velocity                 ,c='g',s=10,label="Transformer")
    # plt.legend()
    plt.xlabel("Period (s)", fontsize=12)
    plt.ylabel("Group velocity (km/s)", fontsize=12)
    plt.tick_params(labelsize=12)
    
    if not save_path == "":
        plt.savefig(save_path,bbox_inches='tight',dpi=300)
    
    if show == True:
        plt.show()
    else:
        plt.close()