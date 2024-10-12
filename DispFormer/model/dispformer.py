import torch
import torch.nn as nn

def load_model(model, load_path, device):
    """
    Load the model from the specified path.

    Parameters:
    - model: The model instance to load the state dictionary into.
    - load_path: The path to the saved model state dictionary.
    - device: The device to load the model onto.

    Returns:
    - model: The model loaded with the state dictionary.
    """
    # Load the state dictionary
    state_dict = torch.load(load_path, map_location=device)
    
    # If the model was saved using DDP, it will have 'module.' prefix
    if 'module.' in list(state_dict.keys())[0]:
        # Remove the 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:]  # Remove 'module.' from the key
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

class PeriodPositionalEncoding(nn.Module):
    def __init__(self, model_dim, min_period=0.005, max_period=200, device=torch.device('cpu')):
        super(PeriodPositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.min_period = min_period
        self.max_period = max_period
        self.device = device

    def forward(self, periods):
        # periods shape: (batch_size, seq_length)
        periods = periods.float().to(self.device)

        # Handle invalid periods (e.g., period=0)
        invalid_mask = (periods <= 0) | (periods != periods)  # Check for zero or NaN
        valid_periods = periods.clone()
        valid_periods[invalid_mask] = self.min_period  # Replace invalid periods with min_period for calculation

        # Convert periods to logarithmic scale and normalize
        min_period_tensor = torch.tensor(self.min_period, device=self.device, dtype=torch.float)
        max_period_tensor = torch.tensor(self.max_period, device=self.device, dtype=torch.float)
        periods_normalized = (torch.log(valid_periods) - torch.log(min_period_tensor)) / \
                             (torch.log(max_period_tensor) - torch.log(min_period_tensor))

        # Reshape for broadcasting
        position = periods_normalized.unsqueeze(-1)  # shape: (batch_size, seq_length, 1)

        # Create div_term for positional encoding
        div_term = torch.exp(torch.arange(0, self.model_dim, 2, dtype=torch.float, device=self.device) *
                             -(torch.log(torch.tensor(10000.0, device=self.device)) / self.model_dim))

        # Compute the positional encodings
        pos_enc = torch.zeros((periods.shape[0], periods.shape[1], self.model_dim), device=self.device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)

        # Mask out the positional encodings for invalid periods
        pos_enc[invalid_mask] = 0  # Set to all zeros for invalid data

        return pos_enc

class DispersionTransformer(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, output_dim, device=torch.device('cpu')):
        super(DispersionTransformer, self).__init__()
        self.device = device
        self.phase_encoding = nn.Linear(1, model_dim).to(device)
        self.group_encoding = nn.Linear(1, model_dim).to(device)
        self.period_position_encoding = PeriodPositionalEncoding(model_dim=model_dim, device=device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads).to(device),
            num_layers=num_layers
        ).to(device)
        self.conv_fusion = nn.Conv1d(model_dim * 3, model_dim, kernel_size=3, padding=1).to(device)
        self.fc_fuse = nn.Linear(model_dim, model_dim).to(device)  # Linear layer to fuse model_dim features
        self.fc_out  = nn.Linear(model_dim, output_dim).to(device)

    def forward(self, input_data, mask=None):
        """
            mask for the padding data [period,phase velocity, group velocity] <=0
        """
        # input_data shape: (batch_size, 3, seq_length)
        period_data, phase_velocity, group_velocity = input_data[:, 0, :], input_data[:, 1, :], input_data[:, 2, :]

        # Prepare velocity for embedding: (batch_size, seq_length, model_dim)
        phase_embedding             = self.phase_encoding(phase_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        group_embedding             = self.group_encoding(group_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_position_embedding   = self.period_position_encoding(period_data)         # (batch_size, seq_length, model_dim)

        # Apply mask to embedding (if mask is provided)
        if mask is not None:
            phase_velocity_mask       = phase_velocity<=0
            group_velocity_mask       = group_velocity<=0
            phase_embedding           = phase_embedding.masked_fill(phase_velocity_mask.unsqueeze(-1), 0)
            group_embedding           = group_embedding.masked_fill(group_velocity_mask.unsqueeze(-1), 0)
            period_position_embedding = period_position_embedding.masked_fill(mask.unsqueeze(-1), 0)
        
        # Combine embeddings by stacking them along a new dimension (dim=1)
        combined_embedding = torch.cat([period_position_embedding, phase_embedding, group_embedding], dim=2)  # (batch_size, seq_length, model_dim * 3)
        
        combined_embedding = combined_embedding.permute(0, 2, 1)  # (batch_size, model_dim * 3, seq_length)
        
        # Apply convolutional layer for feature fusion
        fused_features = self.conv_fusion(combined_embedding)  # (batch_size, model_dim, seq_length)
        
        # Apply transformer encoder with mask to ignore padded positions
        transformer_output = self.transformer_encoder(fused_features.permute(2, 0, 1), src_key_padding_mask=mask)  # (seq_length, batch_size, model_dim)
        transformer_output = transformer_output.permute(1, 2, 0)  # (batch_size, model_dim, seq_length)

        # Apply linear layer to fuse model_dim features at each time step
        time_step_fused = self.fc_fuse(transformer_output.permute(0, 2, 1))  # (batch_size, seq_length, model_dim)
        
        # Aggregate the fused features over the sequence length (if needed)
        pooled_output = torch.mean(time_step_fused, dim=1)  # (batch_size, model_dim)
        
        # Apply fully connected layer for final output
        output = self.fc_out(pooled_output)  # (batch_size, output_dim)
        output *= 6.5  # Scale the output
        
        return output
    
    
class DispersionTransformer_linearEmbedding(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, output_dim, device=torch.device('cpu')):
        super(DispersionTransformer_linearEmbedding, self).__init__()
        self.device = device
        self.phase_encoding = nn.Linear(1, model_dim).to(device)
        self.group_encoding = nn.Linear(1, model_dim).to(device)
        self.period_encoding = nn.Linear(1,model_dim).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads).to(device),
            num_layers=num_layers
        ).to(device)
        self.conv_fusion = nn.Conv1d(model_dim * 3, model_dim, kernel_size=3, padding=1).to(device)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, output_dim).to(device)

    def forward(self, input_data, mask=None):
        # input_data shape: (batch_size, 3, seq_length)
        period_data, phase_velocity, group_velocity = input_data[:, 0, :], input_data[:, 1, :], input_data[:, 2, :]
        
        # Prepare velocity for embedding: (batch_size, seq_length, model_dim)
        phase_embedding  = self.phase_encoding(phase_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        group_embedding  = self.group_encoding(group_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_embedding = self.period_encoding(period_data.unsqueeze(-1))  # (batch_size, seq_length, model_dim)

        # Apply mask to embedding (if mask is provided)
        if mask is not None:
            phase_embedding  = phase_embedding.masked_fill(mask.unsqueeze(-1), 0)
            group_embedding  = group_embedding.masked_fill(mask.unsqueeze(-1), 0)
            period_embedding = period_embedding.masked_fill(mask.unsqueeze(-1), 0)
        
        # Combine embeddings by stacking them along a new dimension (dim=1)
        combined_embedding = torch.cat([period_embedding, phase_embedding, group_embedding], dim=2)  # (batch_size, seq_length, model_dim * 3)
        combined_embedding = combined_embedding.permute(0, 2, 1)  # (batch_size, model_dim * 3, seq_length)
        
        # Apply convolutional layer for feature fusion
        fused_features = self.conv_fusion(combined_embedding)  # (batch_size, model_dim, seq_length)
        
        # Apply transformer encoder with mask to ignore padded positions
        transformer_output = self.transformer_encoder(fused_features.permute(2, 0, 1), src_key_padding_mask=mask)  # (seq_length, batch_size, model_dim)
        transformer_output = transformer_output.permute(1, 2, 0)  # (batch_size, model_dim, seq_length)

        # Apply pooling layer
        pooled_output = self.pool(transformer_output).squeeze(-1)  # (batch_size, model_dim)

        # Fully connected layer for final output
        output = self.fc(pooled_output)  # (batch_size, output_dim)
        output *= 6.5  # Scale the output
        
        return output
    
class DispersionTransformer_linear_posEmbedding(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, output_dim, device=torch.device('cpu')):
        super(DispersionTransformer_linear_posEmbedding, self).__init__()
        self.device = device
        self.phase_encoding = nn.Linear(1, model_dim).to(device)
        self.group_encoding = nn.Linear(1, model_dim).to(device)
        self.period_encoding = nn.Linear(1,model_dim).to(device)
        self.period_position_encoding = PeriodPositionalEncoding(model_dim=model_dim, device=device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads).to(device),
            num_layers=num_layers
        ).to(device)
        self.conv_fusion = nn.Conv1d(model_dim * 3, model_dim, kernel_size=3, padding=1).to(device)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, output_dim).to(device)

    def forward(self, input_data, mask=None):
        # input_data shape: (batch_size, 3, seq_length)
        period_data, phase_velocity, group_velocity = input_data[:, 0, :], input_data[:, 1, :], input_data[:, 2, :]
        
        # Prepare velocity for embedding: (batch_size, seq_length, model_dim)
        phase_embedding = self.phase_encoding(phase_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        group_embedding = self.group_encoding(group_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_embedding = self.period_encoding(period_data.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_position_embedding = self.period_position_encoding(period_data)  # (batch_size, seq_length, model_dim)
        # Apply mask to embedding (if mask is provided)
        if mask is not None:
            phase_embedding  = phase_embedding.masked_fill(mask.unsqueeze(-1), 0)
            group_embedding  = group_embedding.masked_fill(mask.unsqueeze(-1), 0)
            period_embedding = period_embedding.masked_fill(mask.unsqueeze(-1), 0)
            period_position_embedding = period_position_embedding.masked_fill(mask.unsqueeze(-1), 0)
        
        # Combine embeddings by stacking them along a new dimension (dim=1)
        combined_embedding = torch.cat([period_embedding+period_position_embedding, phase_embedding, group_embedding], dim=2)  # (batch_size, seq_length, model_dim * 3)
        combined_embedding = combined_embedding.permute(0, 2, 1)  # (batch_size, model_dim * 3, seq_length)
        
        # Apply convolutional layer for feature fusion
        fused_features = self.conv_fusion(combined_embedding)  # (batch_size, model_dim, seq_length)
        
        # Apply transformer encoder with mask to ignore padded positions
        transformer_output = self.transformer_encoder(fused_features.permute(2, 0, 1), src_key_padding_mask=mask)  # (seq_length, batch_size, model_dim)
        transformer_output = transformer_output.permute(1, 2, 0)  # (batch_size, model_dim, seq_length)

        # Apply pooling layer
        pooled_output = self.pool(transformer_output).squeeze(-1)  # (batch_size, model_dim)

        # Fully connected layer for final output
        output = self.fc(pooled_output)  # (batch_size, output_dim)
        output *= 6.5  # Scale the output

        return output


class DispersionTransformer_nofusion(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, output_dim, device=torch.device('cpu')):
        super(DispersionTransformer_nofusion, self).__init__()
        self.device = device
        self.phase_encoding = nn.Linear(1, model_dim).to(device)
        self.group_encoding = nn.Linear(1, model_dim).to(device)
        self.period_position_encoding = PeriodPositionalEncoding(model_dim=model_dim, device=device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads).to(device),
            num_layers=num_layers
        ).to(device)
        # Commented out Conv1d layer used in previous version
        # self.conv_fusion = nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1).to(device)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, output_dim).to(device)

    def forward(self, input_data, mask=None):
        """
            mask for the padding data [period,phase velocity, group velocity] <=0
        """
        # input_data shape: (batch_size, 3, seq_length)
        period_data, phase_velocity, group_velocity = input_data[:, 0, :], input_data[:, 1, :], input_data[:, 2, :]
        phase_velocity_mask = phase_velocity<=0
        group_velocity_mask = group_velocity<=0

        # Prepare velocity for embedding: (batch_size, seq_length, model_dim)
        phase_embedding             = self.phase_encoding(phase_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        group_embedding             = self.group_encoding(group_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_position_embedding   = self.period_position_encoding(period_data)  # (batch_size, seq_length, model_dim)

        # Apply mask to embedding (if mask is provided)
        if mask is not None:
            phase_embedding           = phase_embedding.masked_fill(phase_velocity_mask.unsqueeze(-1), 0)
            group_embedding           = group_embedding.masked_fill(group_velocity_mask.unsqueeze(-1), 0)
            period_position_embedding = period_position_embedding.masked_fill(mask.unsqueeze(-1), 0)
        
        # Combine embeddings by stacking them along a new dimension (dim=1)
        # combined_embedding = torch.cat([period_position_embedding, phase_embedding, group_embedding], dim=2)  # (batch_size, seq_length, model_dim * 3)
        combined_embedding = period_position_embedding + phase_embedding + group_embedding
        combined_embedding = combined_embedding.permute(0, 2, 1)  # (batch_size, model_dim, seq_length)
        
        # Commented out convolutional feature fusion used in previous version
        # fused_features = self.conv_fusion(combined_embedding)  # (batch_size, model_dim, seq_length)

        # Apply transformer encoder with mask to ignore padded positions
        transformer_output = self.transformer_encoder(combined_embedding.permute(2, 0, 1), src_key_padding_mask=mask)  # (seq_length, batch_size, model_dim)
        transformer_output = transformer_output.permute(1, 2, 0)  # (batch_size, model_dim, seq_length)

        # Apply pooling layer
        pooled_output = self.pool(transformer_output).squeeze(-1)  # (batch_size, model_dim)
        
        # Fully connected layer for final output
        output = self.fc(pooled_output)  # (batch_size, output_dim)
        output *= 6.5                    # Scale the output
        return output
