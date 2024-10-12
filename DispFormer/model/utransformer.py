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
        periods = periods.float().to(self.device)
        invalid_mask = (periods <= 0) | (periods != periods)
        valid_periods = periods.clone()
        valid_periods[invalid_mask] = self.min_period
        
        min_period_tensor = torch.tensor(self.min_period, device=self.device, dtype=torch.float)
        max_period_tensor = torch.tensor(self.max_period, device=self.device, dtype=torch.float)
        periods_normalized = (torch.log(valid_periods) - torch.log(min_period_tensor)) / \
                             (torch.log(max_period_tensor) - torch.log(min_period_tensor))
                             
        position = periods_normalized.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2, dtype=torch.float, device=self.device) *
                             -(torch.log(torch.tensor(10000.0, device=self.device)) / self.model_dim))
        
        pos_enc = torch.zeros((periods.shape[0], periods.shape[1], self.model_dim), device=self.device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        pos_enc[invalid_mask] = 0
        
        return pos_enc

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, device=torch.device('cpu')):
        super(TransformerBlock, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads).to(device),
            num_layers=num_layers
        ).to(device)
    
    def forward(self, x, mask=None):
        return self.transformer_encoder(x.permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2)
    
class UShapeTransformer(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, output_dim, device=torch.device('cpu')):
        super(UShapeTransformer, self).__init__()
        self.device = device
        
        # Encoder Embeddings
        self.phase_encoding  = nn.Linear(1, model_dim).to(device)
        self.group_encoding  = nn.Linear(1, model_dim).to(device)
        self.period_encoding = nn.Linear(1, model_dim).to(device)
        self.period_position_encoding = PeriodPositionalEncoding(model_dim=model_dim, device=device)

        # Transformer Encoder and Decoder Blocks
        self.encoder1   = TransformerBlock(model_dim, num_heads, num_layers).to(device)
        self.encoder2   = TransformerBlock(model_dim, num_heads, num_layers).to(device)
        
        self.bottleneck = TransformerBlock(model_dim, num_heads, num_layers).to(device)

        # Additional Transformer layers in the decoding path
        self.decoder2 = TransformerBlock(model_dim, num_heads, num_layers).to(device)
        self.decoder1 = TransformerBlock(model_dim, num_heads, num_layers).to(device)
        
        # Fusion and Output layers
        self.conv_fusion = nn.Conv1d(model_dim * 3, model_dim, kernel_size=3, padding=1).to(device)
        self.fc_fuse = nn.Linear(model_dim, model_dim).to(device)  # Linear layer to fuse model_dim features
        self.fc_out = nn.Linear(model_dim, output_dim).to(device)

    def forward(self, input_data, mask=None):
        period_data, phase_velocity, group_velocity = input_data[:, 0, :], input_data[:, 1, :], input_data[:, 2, :]

        # Embeddings
        phase_embedding  = self.phase_encoding(phase_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        group_embedding  = self.group_encoding(group_velocity.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_embedding = self.period_encoding(period_data.unsqueeze(-1))  # (batch_size, seq_length, model_dim)
        period_position_embedding = self.period_position_encoding(period_data)  # (batch_size, seq_length, model_dim)

        # Combine embeddings by stacking them along a new dimension (dim=1)
        combined_embedding = torch.cat([period_embedding+period_position_embedding, phase_embedding, group_embedding], dim=2)  # (batch_size, seq_length, model_dim * 3)
        combined_embedding = combined_embedding.permute(0, 2, 1)  # (batch_size, model_dim * 3, seq_length)
        
        combined_embedding = self.conv_fusion(combined_embedding)  # (batch_size, model_dim, seq_length)

        # U-Shape Transformer Encoding-Decoding with skip connections
        enc1 = self.encoder1(combined_embedding.permute(2, 0, 1))  # Encoder 1
        enc2 = self.encoder2(enc1)  # Encoder 2
        bottleneck = self.bottleneck(enc2)  # Bottleneck

        # Decoder without upsampling
        dec2 = self.decoder2(bottleneck+enc2)   # Decoder 2 directly
        dec1 = self.decoder1(dec2+enc1)         # Decoder 1 directly

        # Apply linear layer to fuse model_dim features at each time step
        time_step_fused = self.fc_fuse(dec1.permute(1, 0, 2))  # (batch_size, seq_length, model_dim)
        
        # Aggregate the fused features over the sequence length (if needed)
        pooled_output = torch.mean(time_step_fused, dim=1)  # (batch_size, model_dim)
        
        # Apply fully connected layer for final output
        output = self.fc_out(pooled_output)  # (batch_size, output_dim)
        output = output * 6.5  # Scale the output
        
        return output
