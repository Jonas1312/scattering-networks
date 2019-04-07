# coding:utf-8
"""
  Purpose:  Vanilla ResNet18 and Scatt + ResNet18 models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio import Scattering2D


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ScattResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10, input_shape=(32, 32), J=2, L=8):
        super(ScattResNet18, self).__init__()

        self.scattering = Scattering2D(J=J, shape=input_shape)
        if torch.cuda.is_available():
            print("Move scattering to GPU")
            self.scattering = self.scattering.cuda()
        self.K = 1 + J * L + L ** 2 * (J - 1) * J // 2
        self.scatt_output_shape = tuple([x // 2 ** J for x in input_shape])
        self.bn = nn.BatchNorm2d(self.K * 3)

        self.in_planes = self.K * 3
        self.conv1 = nn.Conv2d(self.K * 3, self.K * 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.K * 3)
        self.layer1 = self._make_layer(block, self.K * 3, 3, stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.scattering(x)
        out = self.bn(out.view(-1, self.K * 3, 8, 8))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    print("CNN only model:")
    model = ResNet18().to("cuda")
    summary(model, input_size=(3, 32, 32))
    print("")
    print("ScattCNN model:")
    model = ScattResNet18().to("cuda")
    summary(model, input_size=(3, 32, 32))
