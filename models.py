import torch.nn as nn
import os
from efficient_kan import KAN

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ConvBNRelu3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0, stride=1):
        super(ConvBNRelu3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNRelu2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1, padding=0):
        super(ConvBNRelu2D, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Enc_AE(nn.Module):
    def __init__(self, channel, output_dim, windowSize):
        super(Enc_AE, self).__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.windowSize = windowSize
        self.conv1 = ConvBNRelu3D(
            in_channels=1, out_channels=8, kernel_size=(7, 3, 3), stride=1, padding=0)
        self.conv2 = ConvBNRelu3D(
            in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.conv3 = ConvBNRelu3D(
            in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32 * (self.channel - 12),
                                  out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.sem_conv = nn.Conv2d(64, 16, kernel_size=1)
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),      
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          
            nn.Linear(512, 2 * output_dim)
        )
        self.kan_layer = KAN([2 * output_dim, output_dim])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0], -1, x.shape[3], x.shape[4]])
        x = self.conv4(x)
        map = self.pool(x)
        h = map.reshape([map.shape[0], -1])
        z = self.projector(h)
        H = self.kan_layer(z)
        return H