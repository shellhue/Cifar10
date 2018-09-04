import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        stride = 1
        if in_channels != out_channels:
            stride = 2
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(16, 16,  num_blocks)
        self.layer2 = self.make_layer(16, 32, num_blocks)
        self.layer3 = self.make_layer(32, 64, num_blocks)
        self.fc = nn.Linear(64, 10)

    def make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels))
            else:
                layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out


def ResNet20():
    return ResNet(num_blocks=3)


def ResNet32():
    return ResNet(num_blocks=5)


def ResNet44():
    return ResNet(num_blocks=7)


def ResNet56():
    return ResNet(num_blocks=9)


def ResNet110():
    return ResNet(num_blocks=18)


def ResNet1202():
    return ResNet(num_blocks=200)

