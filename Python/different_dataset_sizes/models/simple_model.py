# coding:utf-8
"""
  Purpose:  Simple CNN and Scatt+CNN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio import Scattering2D


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conva = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.convb = nn.Conv2d(in_channels=64, out_channels=81, kernel_size=3, padding=1, stride=2)

        self.conv1 = nn.Conv2d(in_channels=81, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conva(x)
        x = F.relu(x)
        x = self.convb(x)
        x = F.relu(x)

        # Common base
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 64)
        x = self.fc1(x)
        return x


class ScattCNN(nn.Module):
    def __init__(self, input_shape=(28, 28), J=2, L=8):
        super(ScattCNN, self).__init__()

        self.scattering = Scattering2D(J=J, shape=input_shape)
        if torch.cuda.is_available():
            print("Move scattering to GPU")
            self.scattering = self.scattering.cuda()
        self.K = 1 + J * L + L ** 2 * (J - 1) * J // 2
        self.scatt_output_shape = tuple([x // 2 ** J for x in input_shape])
        self.bn = nn.BatchNorm2d(self.K)
        self.conv = nn.Conv2d(in_channels=self.K, out_channels=self.K, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_channels=self.K, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.scattering(x)
        x = x.view(-1, self.K, *self.scatt_output_shape)
        x = self.bn(x)
        x = self.conv(x)
        x = F.relu(x)

        # Common base
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 64)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    print("CNN only model:")
    model = CNN().to("cuda")
    summary(model, input_size=(1, 28, 28))
    print("")
    print("Scatt + CNN model:")
    model = ScattCNN().to("cuda")
    summary(model, input_size=(1, 28, 28))
