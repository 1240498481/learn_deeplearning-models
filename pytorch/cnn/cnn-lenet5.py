import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='logs/LeNet5/')


class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            # 3x224x224 -> 18x220x220
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.Tanh(),
            # 18x220x220 -> 18x110x110
            nn.MaxPool2d(kernel_size=2),
            # 18x110x110 -> 48x106x106
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Tanh(),
            # 48x106x106 -> 48x53x53
            nn.MaxPool2d(kernel_size=2)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d((50, 50))

        self.classifier = nn.Sequential(
            # nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Linear(self.AvgPool.output_size[0] * self.AvgPool.output_size[1] * 16 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(10, False)
model.to('cuda:0')

writer.add_graph(model, torch.rand(1, 3, 224, 224).to('cuda:0'))
writer.close()