import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append("../../../../")
from DispFormer.dataloader import *
from DispFormer.plots import *
from DispFormer.model.dispformer import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import json

# load training parameters
with open('settings.json', 'r') as f:
    settings = json.load(f)

# Paths for input data and saving the model
train_data_path         = "PATH-TO-FINETUING-TRAINING-DATASET"
train_label_path        = "PATH-TO-FINETUING-TRAINING-VELOCITY-MODEL"
valid_data_path         = "PATH-TO-FINETUING-VALIDATION-DATASET"
valid_label_path        = "PATH-TO-FINETUING-VALIDATION-VELOCITY-MODEL"
pretrained_model_path   = f"PATH-TO-FINETUNED-MODEL"

sparse_num = 20
save_path               = "PATH-TO-SAVING-PATH"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == "__main__":
    ######################################################
    #                       Datasets
    ######################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = DispersionDatasets(
        input_data_path=train_data_path,
        input_label_path=train_label_path,
        train=True,
        interp_layer=True,
        layer_thickness=0.5,
        layer_number=400,
        layer_used_range=[0, 400],
        augmentation_train_data=False,
        num_workers=settings['training']['num_workers']
    )

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    train_indices = indices[::sparse_num]
    np.random.shuffle(train_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    collect_fn = train_collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings['training']['batch_size'],
        collate_fn=collect_fn,
        sampler=train_sampler,
        num_workers=settings['training']['num_workers']
    )
    
    valid_dataset = DispersionDatasets(
        input_data_path=valid_data_path,
        input_label_path=valid_label_path,
        train=True,
        interp_layer=True,
        layer_thickness=0.5,
        layer_number=400,
        layer_used_range=[0, 400],
        augmentation_train_data=False,
        num_workers=settings['training']['num_workers']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=settings['training']['batch_size'],
        collate_fn=collect_fn,
        shuffle=False,
        num_workers=settings['training']['num_workers'],
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
    model = load_model(model=model,load_path=pretrained_model_path,device=device)
    ######################################################
    #                       Training
    ######################################################
    def NMSE(output, target):
        return torch.sum(((output - target) / target) ** 2)
    
    criterion = NMSE
    optimizer = optim.Adam(model.parameters(), lr=settings['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1000, gamma=0.75, last_epoch=-1)

    # Training Loop
    pbar = tqdm(range(settings['training']['num_epochs']))
    train_losses = []
    valid_losses = []

    for epoch in pbar:
        # Training phase
        model.train()
        train_loss_batch = 0
        for batch_inputs, batch_data_mask, batch_targets, batch_layer_usage in train_loader:
            batch_inputs, batch_data_mask, batch_targets = batch_inputs.to(device), batch_data_mask.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs, batch_data_mask)
            loss = 0
            for i in range(batch_layer_usage.shape[0]):
                loss += criterion(outputs[i][batch_layer_usage[i, 0]:batch_layer_usage[i, 1]], batch_targets[i, 1, batch_layer_usage[i, 0]:batch_layer_usage[i, 1]])
            loss.backward()
            optimizer.step()
            train_loss_batch += loss.item()
        
        train_losses.append(train_loss_batch)
        scheduler.step()
        
        # Validation phase
        model.eval()
        valid_loss_batch = 0
        with torch.no_grad():  # No need to calculate gradients during validation
            for batch_inputs, batch_data_mask, batch_targets, batch_layer_usage in valid_loader:
                batch_inputs, batch_data_mask, batch_targets = batch_inputs.to(device), batch_data_mask.to(device), batch_targets.to(device)
                outputs = model(batch_inputs, batch_data_mask)
                loss = 0
                for i in range(batch_layer_usage.shape[0]):
                    loss += criterion(outputs[i][batch_layer_usage[i, 0]:batch_layer_usage[i, 1]], batch_targets[i, 1, batch_layer_usage[i, 0]:batch_layer_usage[i, 1]])
                valid_loss_batch += loss.item()

        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
        elif valid_loss_batch < np.array(valid_losses).min():
            torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
        
        valid_losses.append(valid_loss_batch)

        # Save model at specified epochs
        if (epoch + 1) % settings['training']['save_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"transformer_epoch_{epoch+1}.pt"))

        pbar.set_description(f'Epoch [{epoch+1}/{settings["training"]["num_epochs"]}], Train Loss: {train_loss_batch:.4f}, Valid Loss: {valid_loss_batch:.4f}')

    # Save final model and losses
    # torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
    np.savetxt(os.path.join(save_path, "train_losses.txt"), np.array(train_losses))
    np.savetxt(os.path.join(save_path, "valid_losses.txt"), np.array(valid_losses))
