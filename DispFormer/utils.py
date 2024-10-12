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
    # 保留第一个点
    first_point = y[0]

    # 创建平滑窗口
    window = np.ones(window_size) / window_size

    # 使用reflect填充以减轻边缘效应
    y_padded = np.pad(y, (window_size // 2, window_size // 2), mode='reflect')
    y_smooth = np.convolve(y_padded, window, mode='valid')

    # 将第一个点替换为原始数据中的第一个点
    y_smooth[0] = first_point
    
    return y_smooth

#################################################################
#################################################################

def NMSE(output, target):
    return torch.sum(((output - target) / target) ** 2)

# NMSE: defined by self
def NMSE_np(output, target):
    return np.sum(((output - target) / target) ** 2)

# MAPE: Mean Absolute Percentage Error
def MAPE_np(output, target):
    return np.mean(np.abs((output - target) / target)) * 100

# MSE: Mean Squared Error
def MSE_np(output, target):
    return np.mean((output - target) ** 2)*1000

# MAE: Mean Absolute Error
def MAE_np(output, target):
    return np.mean(np.abs(output - target))*1000

# MAE for each layers
def MAE_layers_np(output,target):
    results =  np.mean(np.abs(output - target),axis=0)*100
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
    绘制2D热力图并标记每行的最小值
    
    Args:
    - matrix (np.array): 待绘制的矩阵
    - learning_rates (list): 学习率列表（列标签）
    - sparse_nums (list): 稀疏数列表（行标签）
    - plot_base_path (str): 保存路径的基础路径
    - metric_name (str): 矩阵对应的指标名称 (如 'NMSE', 'MSE', 'MAE', 'MAPE')
    - save_name (str): 保存文件的名称
    """
    # 转换为numpy数组，便于绘图
    matrix_np = np.array(matrix)

    # 绘制2D热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix_np, cmap='cool', aspect='auto', interpolation='none')

    # 在格点上显示数值
    for i in range(len(sparse_nums)):
        for j in range(len(learning_rates)):
            plt.text(j, i, f"{matrix_np[i, j]:.2f}", ha='center', va='center', color='k')

    # 查找每行的最小值并用红色方框标记
    ax = plt.gca()  # 获取当前坐标轴
    for i in range(len(sparse_nums)):
        min_index = np.argmin(matrix_np[i, :])  # 找到每行的最小值索引
        # 添加红色方框
        rect = patches.Rectangle((min_index - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # 颜色条、标签和标题
    plt.colorbar(label=metric_name)
    plt.xticks(np.arange(len(learning_rates)), learning_rates)
    plt.yticks(np.arange(len(sparse_nums)), sparse_nums)
    plt.xlabel('Learning Rate')
    plt.ylabel('Sparse Num')
    plt.title(f'{metric_name} vs Learning Rate and Sparse Num')

    # 保存图片
    if not save_name == "":
        plt.savefig(os.path.join(plot_base_path, save_name), bbox_inches='tight', dpi=300)

    # 显示图像
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