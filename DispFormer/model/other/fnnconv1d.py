import torch
import torch.nn as nn
from collections import OrderedDict

class FNN1D(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 seq_length=80, 
                 init_features=256, 
                 hidden_dims=[256, 512, 256, 128, 64, 32, 16], 
                 output_dim=400, device=torch.device('cpu')):
        super(FNN1D, self).__init__()
        self.device = device
        features = init_features

        # 1D卷积层
        self.conv1 = nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False).to(device)
        self.relu1 = nn.ReLU(inplace=True).to(device)

        # 计算展平后的输入维度
        self.fc_input_dim = features * seq_length

        # 全连接层
        self.fc_layers = nn.ModuleList()
        prev_dim = self.fc_input_dim
        for dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_dim, dim).to(device))
            self.fc_layers.append(nn.ReLU(inplace=True).to(device))
            prev_dim = dim

        # 输出层
        self.fc_output = nn.Linear(prev_dim, output_dim).to(device)

    def forward(self, x):
        x = x.to(self.device)

        # 通过1D卷积层
        x = self.conv1(x)
        x = self.relu1(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 通过全连接层
        for layer in self.fc_layers:
            x = layer(x)

        # 输出层
        output = self.fc_output(x)
        output *= 6.5  # Scale the output
        return output
