# coding:utf-8
"""
  Purpose:  Base model, CNN only, has to replace scattering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Base(nn.Module):

    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 75, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(75, 75, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        return x


# class Base(nn.Module):
#
#     def __init__(self):
#         super(Base, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv7 = nn.Conv2d(512, 651, kernel_size=3, stride=1, padding=1)
#         self.conv8 = nn.Conv2d(651, 651, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.conv4(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.conv5(x)
#         x = F.relu(x)
#         x = self.conv6(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#         x = self.conv7(x)
#         x = F.relu(x)
#         x = self.conv8(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#         return x


if __name__ == '__main__':
    from torchsummary import summary

    print("CNN:")
    model = Base().to("cuda")
    summary(model, input_size=(3, 256, 256))
    print("")
