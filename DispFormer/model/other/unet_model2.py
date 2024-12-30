import torch
import torch.nn as nn
from collections import OrderedDict

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, init_features=32, output_dim=10, device=torch.device('cpu')):
        super(UNet1D, self).__init__()
        self.device = device
        self.num_layers = num_layers
        features = init_features

        # Encoding layers
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.encoders.append(UNet1D._block(in_channels, features, name=f"enc{i+1}").to(device))
            else:
                self.encoders.append(UNet1D._block(features, features * 2, name=f"enc{i+1}").to(device))
                features *= 2
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2).to(device))

        # Bottleneck layer
        self.bottleneck = UNet1D._block(features, features * 2, name="bottleneck").to(device)
        features *= 2  # Update features for bottleneck

        # Decoding layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.upconvs.append(nn.ConvTranspose1d(features, features // 2, kernel_size=2, stride=2).to(device))
            self.decoders.append(UNet1D._block(features, features // 2, name=f"dec{num_layers-i}").to(device))
            features //= 2

        self.conv = nn.Conv1d(features, out_channels, kernel_size=1).to(device)
        self.fc = nn.Linear(out_channels * init_features, output_dim).to(device)

    def forward(self, x):
        x = x.to(self.device)
        enc_features = []

        # Encoding path
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding path
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            x = self._crop_and_concat(enc_features[self.num_layers - 1 - i], x)
            x = self.decoders[i](x)

        conv_output = self.conv(x)  # (batch_size, out_channels, seq_len)
        flattened_output = conv_output.view(conv_output.size(0), -1)  # Flatten the output
        output = self.fc(flattened_output)  # Apply linear layer to map to output_dim
        output *= 6.5  # Scale the output
        return output

    def _crop_and_concat(self, enc_feature, dec_feature):
        if enc_feature.size(2) > dec_feature.size(2):
            enc_feature = enc_feature[:, :, :dec_feature.size(2)]
        return torch.cat((enc_feature, dec_feature), dim=1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm1d(features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm1d(features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
