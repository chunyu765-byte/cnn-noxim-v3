import torch
import torch.nn as nn


class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Input: 3 x 32 x 32
        # Layer 1 output: 16 x 32 x 32 = 16384
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        # Layer 2 output: 16 x 16 x 16 = 4096
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3 output: 48 x 16 x 16 = 12288
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1)

        # Layer 4 output: 48 x 8 x 8 = 3072
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 5 output: 96 x 8 x 8 = 6144
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)

        # Layer 6 output: 64 x 8 x 8 = 4096
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)

        # Layer 7 output: 64 x 8 x 8 = 4096
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Layer 8 output: 64 x 4 x 4 = 1024
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 9 output: 512
        self.fc1 = nn.Linear(1024, 512)

        # Layer 10 output: 256
        self.fc2 = nn.Linear(512, 256)

        # Layer 11 output: 10 logits, no activation for CrossEntropyLoss
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.max_pooling_1(x)

        x = torch.tanh(self.conv2(x))
        x = self.max_pooling_2(x)

        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        x = self.max_pooling_3(x)

        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


AlexNet = AlexNetCIFAR10


if __name__ == '__main__':
    model = AlexNetCIFAR10(num_classes=10)
    x = torch.randn(3, 3, 32, 32)
    y = model(x)
    print('Output shape:', y.shape)
