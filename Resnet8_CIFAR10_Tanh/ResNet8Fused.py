import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedBasicBlock(nn.Module):
    """BasicBlock after Conv-BN fusion.

    All BatchNorm2d layers are folded into the preceding Conv2d layers, so the
    inference graph only contains Conv2d, Tanh, residual addition, avgpool, and
    the final Linear layer.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = torch.tanh(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = torch.tanh(out)
        return out


class ResNet8CIFAR10Fused(nn.Module):
    """Fused CIFAR-style ResNet-8 for inference/export."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.block1 = FusedBasicBlock(16, 16, stride=1)
        self.block2 = FusedBasicBlock(16, 32, stride=2)
        self.block3 = FusedBasicBlock(32, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


ResNet8Fused = ResNet8CIFAR10Fused
