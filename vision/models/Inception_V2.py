import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def ConvBNReLUFactorization(in_channels, out_channels, kernel_sizes, paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_sizes, stride=1, padding=paddings),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels3reduce, kernel_size=1),
            ConvBNReLU(out_channels3reduce, out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(out_channels3, out_channels3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModuleB(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(out_channels2reduce, out_channels2reduce, kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(out_channels2reduce, out_channels2, kernel_sizes=[3, 1], paddings=[1, 0]),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels3reduce, kernel_size=1),
            ConvBNReLUFactorization(out_channels3reduce, out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(out_channels3reduce, out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]),
            ConvBNReLUFactorization(out_channels3reduce, out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]),
            ConvBNReLUFactorization(out_channels3reduce, out_channels3, kernel_sizes=[3, 1], paddings=[1, 0]),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModuleC(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels1, kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(out_channels2reduce, out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(out_channels2reduce, out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])

        self.branch3_conv1 = ConvBNReLU(in_channels, out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(out_channels3reduce, out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(out_channels3, out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(out_channels3, out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)

        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)], dim=1)

        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):
    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels1reduce, kernel_size=1),
            ConvBNReLU(out_channels1reduce, out_channels1, kernel_size=3, stride=2, padding=1),
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(out_channels2, out_channels2, kernel_size=3, stride=2, padding=1),
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(128, out_channels=768, kernel_size=5, stride=1)
        self.auxiliary_dropout = nn.Dropout(0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_channels=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out


class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV2, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            InceptionV2ModuleA(192, 64, 64, 64, 64, 96, 32),
            InceptionV2ModuleA(256, 64, 64, 96, 64, 96, 64),
            InceptionV3ModuleD(320, 128, 160, 64, 96),
        )

        self.block4 = nn.Sequential(
            InceptionV2ModuleB(576, 224, 64, 96, 96, 128, 128),
            InceptionV2ModuleB(576, 192, 96, 128, 96, 128, 128),
            InceptionV2ModuleB(576, 160, 128, 160, 128, 128, 128),
            InceptionV2ModuleB(576, 96, 128, 192, 160, 160, 128),
            InceptionV3ModuleD(576, 128, 192, 192, 256)
        )

        self.block5 = nn.Sequential(
            InceptionV2ModuleC(1024, 352, 192, 160, 160, 112, 128),
            InceptionV2ModuleC(1024, 352, 192, 160, 192, 112, 128),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(1024, num_classes)

    def forwrd(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


if __name__ == '__main__':
    writers = SummaryWriter(log_dir='../logs/Inception_V2')
    writers.add_graph(InceptionV2(), torch.rand(1, 3, 224, 224))
    writers.close()
