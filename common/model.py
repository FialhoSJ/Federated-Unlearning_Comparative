from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class SmallCifarCNN(nn.Module):
    """Small deterministic baseline backbone for smoke tests and sanity runs."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.feature_dim = 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x):
        return self.classifier[:3](self.features(x))

    def classify_features(self, features):
        return self.classifier[3](features)

    def forward(self, x):
        return self.classify_features(self.forward_features(x))


class ResNet18GroupNorm(nn.Module):
    """CIFAR-sized ResNet18 without batch-running statistics across clients."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.feature_dim = 512
        self.network = resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(8, channels))
        self.network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.network.maxpool = nn.Identity()
        self.network.fc = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        net = self.network
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        return net.avgpool(x).flatten(1)

    def classify_features(self, features):
        return self.network.fc(features)

    def forward(self, x):
        return self.classify_features(self.forward_features(x))
