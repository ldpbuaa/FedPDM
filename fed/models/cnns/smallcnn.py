import torch
import torch.nn as nn

from ..base import ModelBase


class SmallCNN(ModelBase):
    name = 'small'
    input_size = {'cifar':(3, 32, 32), 'fashionmnist': (1, 28, 28)}
    filters = [128, 256, 512, 512]

    def __init__(self, num_classes=10, input_channel=1, scaling = 1):
        super().__init__(num_classes)
        f1, f2, f3, f4 = self.filters
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, f1, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(f1, f2, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(f2, f3, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(f4, f4*2),
            nn.Linear(f4*2, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
