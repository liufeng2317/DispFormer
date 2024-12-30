import torch
import torch.nn as nn
import torch.nn.functional as F

class S2vpBlock1(nn.Module):
    def __init__(self, in_filter_num, out_filter_num, strides=1, device='cpu'):
        super(S2vpBlock1, self).__init__()
        self.device = device  # Store device

        # 1D Convolution: kernel_size=7, padding=3 keeps the same length if stride=1
        self.conv1 = nn.Conv1d(in_channels=in_filter_num, out_channels=out_filter_num, kernel_size=7, stride=strides, padding=3).to(device)
        self.bn1 = nn.BatchNorm1d(out_filter_num).to(device)
        
        # 1D Convolution: kernel_size=5, padding=2 also keeps length
        self.conv2 = nn.Conv1d(in_channels=out_filter_num, out_channels=out_filter_num, kernel_size=5, stride=1, padding=2).to(device)
        self.bn2 = nn.BatchNorm1d(out_filter_num).to(device)
        
        # Downsampling layer: applied if strides != 1 or input and output channels are different
        self.downsample = nn.Conv1d(in_channels=in_filter_num, out_channels=out_filter_num, kernel_size=1, stride=strides).to(device) if strides != 1 or in_filter_num != out_filter_num else nn.Identity()

        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        identity = self.downsample(x)  # Match dimensions if needed

        out = self.conv1(x)            # First convolution
        out = self.bn1(out)            # Batch normalization
        out = self.relu(out)              # ReLU activation (in-place free)

        out = self.conv2(out)          # Second convolution
        out = self.bn2(out)            # Batch normalization

        out = out+ identity                # Residual connection
        out = self.relu(out)              # Final ReLU activation (in-place free)
        return out


class S2vpNet(nn.Module):
    def __init__(self, seq_length=80, init_features=20, output_dim=400, device='cpu'):
        super(S2vpNet, self).__init__()
        self.device = device

        # Stem: Conv1D transforms input from 2 to init_features
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=init_features, kernel_size=1, stride=1).to(device),
            nn.BatchNorm1d(init_features).to(device),
            nn.ReLU()
        )

        # Stacked layers of S2vpBlock1 with increasing filter sizes and downsampling
        self.layer1 = self._make_layer(init_features,     init_features     , blocks=1).to(device)
        self.layer2 = self._make_layer(init_features,     init_features * 2 , blocks=1, strides=2).to(device)  # Downsampling reduces length by 2
        self.layer3 = self._make_layer(init_features * 2, init_features * 4 , blocks=1, strides=2).to(device)  # Further downsampling
        self.layer4 = self._make_layer(init_features * 4, init_features * 8 , blocks=1, strides=2).to(device)
        self.layer5 = self._make_layer(init_features * 8, init_features * 16, blocks=1, strides=2).to(device)

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            self._make_upsample_block(init_features * 16, init_features * 8).to(device),  # Upsamples and reduces filter count
            self._make_upsample_block(init_features * 16, init_features * 4).to(device),
            self._make_upsample_block(init_features * 8, init_features * 2).to(device),
            self._make_upsample_block(init_features * 4, init_features).to(device)
        ])

        # Final convolutional output layer
        self.conv_out = nn.Conv1d(in_channels=2*init_features, out_channels=1, kernel_size=3, padding=1).to(device)
        self.bn_out = nn.BatchNorm1d(1).to(device)
        
        # Fully connected layers for final output
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                     # [batch_size, flattened_size]
            nn.Linear(seq_length, 1024).to(device),
            nn.ReLU(),
            # nn.Dropout(0.0),
            nn.Linear(1024, 1024).to(device),
            nn.ReLU(),
            # nn.Dropout(0.0),
            nn.Linear(1024, output_dim).to(device),
            nn.Sigmoid()
        )

    def _make_layer(self, in_filter_num, out_filter_num, blocks, strides=1):
        layers = [S2vpBlock1(in_filter_num, out_filter_num, strides, device=self.device)]
        for _ in range(1, blocks):
            layers.append(S2vpBlock1(out_filter_num, out_filter_num, device=self.device))
        return nn.Sequential(*layers)

    def _make_upsample_block(self, in_filter_num, out_filter_num):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),   # Upsamples length by 2
            nn.Conv1d(in_channels=in_filter_num, out_channels=out_filter_num, kernel_size=3, padding=1).to(self.device),
            nn.BatchNorm1d(out_filter_num).to(self.device),
            nn.ReLU()
        )

    def crop_and_concat(self, upsampled, bypass):
        # Get the seq_length of both tensors
        upsampled_len = upsampled.size(2)
        bypass_len = bypass.size(2)
        
        # Calculate cropping if necessary
        if upsampled_len > bypass_len:
            # Crop the upsampled tensor to match bypass seq_length
            crop_start = (upsampled_len - bypass_len) // 2
            crop_end = crop_start + bypass_len
            upsampled = upsampled[:, :, crop_start:crop_end]
        
        # Concatenate along the channel dimension (dim=1)
        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        x1 = self.stem(x)                 # [batch_size, init_features, seq_length]
        x1 = self.layer1(x1)              # [batch_size, init_features, seq_length]
        x2 = self.layer2(x1)              # [batch_size, init_features * 2, seq_length // 2]
        x3 = self.layer3(x2)              # [batch_size, init_features * 4, seq_length // 4]
        x4 = self.layer4(x3)              # [batch_size, init_features * 8, seq_length // 8]
        x5 = self.layer5(x4)              # [batch_size, init_features * 16, seq_length // 16]
        
        x6 = self.upsample_blocks[0](x5)  # [batch_size, init_features * 8, seq_length // 8]
        x6 = self.crop_and_concat(x6, x4)   # Crop and concatenate with x4
        
        x7 = self.upsample_blocks[1](x6)  # [batch_size, init_features * 4, seq_length // 4]
        x7 = self.crop_and_concat(x7, x3)   # Crop and concatenate with x3
        
        x8 = self.upsample_blocks[2](x7)  # [batch_size, init_features * 2, seq_length // 2]
        x8 = self.crop_and_concat(x8, x2)   # Crop and concatenate with x2

        x9 = self.upsample_blocks[3](x8)  # [batch_size, init_features, seq_length]
        x9 = self.crop_and_concat(x9, x1)   # Crop and concatenate with x1

        x10 = self.conv_out(x9)          # [batch_size, 1, seq_length]
        x10 = self.bn_out(x10)
        
        out = self.fc_layers(x10)         # Flattened and passed through FC layers
        out = out* 6.5  # Scale the output
        return out

# # Example usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = S2vpNet(device=device).to(device)  # Ensure the model is on the correct device
# inputs = torch.randn(8, 2, 512).to(device)  # batch size of 8, 2 channels, sequence length 512
# outputs = model(inputs)
# print(outputs.shape)
