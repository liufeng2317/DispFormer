import torch
import torch.nn as nn
from collections import OrderedDict

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32, output_dim=10, device=torch.device('cpu')):
        super(UNet1D, self).__init__()
        self.device = device
        features = init_features

        # Encoding layers
        self.encoder1 = UNet1D._block(in_channels, features, name="enc1").to(device)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2).to(device)
        self.encoder2 = UNet1D._block(features, features * 2, name="enc2").to(device)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2).to(device)
        self.encoder3 = UNet1D._block(features * 2, features * 4, name="enc3").to(device)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2).to(device)
        self.encoder4 = UNet1D._block(features * 4, features * 8, name="enc4").to(device)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2).to(device)

        # Bottleneck layer
        self.bottleneck = UNet1D._block(features * 8, features * 16, name="bottleneck").to(device)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2).to(device)
        self.decoder4 = UNet1D._block(features * 8 * 2, features * 8, name="dec4").to(device)
        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2).to(device)
        self.decoder3 = UNet1D._block(features * 4 * 2, features * 4, name="dec3").to(device)
        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2).to(device)
        self.decoder2 = UNet1D._block(features * 2 * 2, features * 2, name="dec2").to(device)
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2).to(device)
        self.decoder1 = UNet1D._block(features * 2, features, name="dec1").to(device)

        self.conv = nn.Conv1d(features, out_channels, kernel_size=1).to(device)
        self.output_dim = output_dim
        self.fc = nn.Linear(out_channels * init_features, output_dim).to(device)

    def forward(self, x):
        x = x.to(self.device)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self._crop_and_concat(enc4, dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = self._crop_and_concat(enc3, dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self._crop_and_concat(enc2, dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self._crop_and_concat(enc1, dec1)
        dec1 = self.decoder1(dec1)

        conv_output = self.conv(dec1)  # (batch_size, out_channels, seq_len)
        flattened_output = conv_output.view(conv_output.size(0), -1)  # Flatten the output
        output = self.fc(flattened_output)  # Apply linear layer to map to output_dim
        output *= 6.5  # Scale the output
        return output

    def _crop_and_concat(self, enc_feature, dec_feature):
        """
        Center crop the encoder feature map to match the decoder feature map size and concatenate them.
        """
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
