import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append("../../../")
from DispFormer.dataloader_pretrain import *
from DispFormer.plots import *
from DispFormer.model.dispformer import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import json

# load training parameters
with open('settings.json', 'r') as f:
    settings = json.load(f)

# Paths for input data and saving the model
# Paths for input data and saving the model
input_data_path = "PATH-TO-DISPERSION-DATASETS"
input_label_path= "PATH-TO-VELOCITY-MODEL"
save_path       = "PATH-TO-SAVE-Model"

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
if __name__ == "__main__":
    ######################################################
    #                       Datasets
    ######################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = DispersionDatasets(
        input_data_path=input_data_path,
        input_label_path=input_label_path,
        train=True,
        interp_layer=True,
        layer_thickness = 0.5,
        layer_number    = 400,
        layer_used_range= [0, 400],
        num_workers     = settings['training']['num_workers'],
        augmentation_train_data=True,
        noise_level = 0.02,
        mask_ratio  = 0.1,
        remove_phase_ratio=0,
        remove_group_ratio=0,
        max_masking_length=0
    )

    collect_fn = train_collate_fn
    train_loader = DataLoader(
        dataset,
        batch_size=settings['training']['batch_size'],
        shuffle=True,
        collate_fn=collect_fn,
        num_workers=settings['training']['num_workers']
    )

    ######################################################
    #                       Model
    ######################################################
    model = DispersionTransformer(
        settings['model']['model_dim'],
        settings['model']['num_heads'],
        settings['model']['num_layers'],
        settings['model']['output_dim'],
        device=device
    ).to(device)

    ######################################################
    #                       Training
    ######################################################

    def NMSE(output, target):
        return torch.sum(((output - target) / target) ** 2)

    criterion = NMSE
    optimizer = optim.Adam(model.parameters(), lr=settings['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=500, gamma=0.75, last_epoch=-1)

    # Training Loop
    pbar = tqdm(range(settings['training']['num_epochs']))
    losses = []
    for epoch in pbar:
        loss_batch = 0
        for batch_inputs, batch_data_mask, batch_targets, batch_layer_usage in train_loader:
            batch_inputs, batch_data_mask, batch_targets = batch_inputs.to(device), batch_data_mask.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs, batch_data_mask)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i, 0]:batch_layer_usage[i, 1]], batch_targets[i, 1, batch_layer_usage[i, 0]:batch_layer_usage[i, 1]])
            loss.backward()
            optimizer.step()
            loss_batch += loss.item()
        scheduler.step()
        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
        elif loss_batch < np.array(losses).min():
                torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
        losses.append(loss_batch)
        if (epoch + 1) % settings['training']['save_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"transformer_epoch_{epoch+1}.pt"))
        pbar.set_description(f'Epoch [{epoch+1}/{settings["training"]["num_epochs"]}], Loss: {loss_batch:.4f}')

    losses = np.array(losses)
    torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
    np.savetxt(os.path.join(save_path, "loss.txt"), losses)