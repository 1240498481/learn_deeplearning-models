import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


"""
参考地址： https://zhuanlan.zhihu.com/p/506610815
"""


writer = SummaryWriter(log_dir='logs/mobilenet_v1')


"""
MobileNetV1
"""
def conv_bn(in_channel, out_channel, stride=1):
    """
    传统卷积块：Conv+BN+Act
    """
    return nn.Sequential(
        # 卷积层，卷积核大小：3
        nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
        # 归一化层
        nn.BatchNorm2d(out_channel),

        # ReLU6激活函数
        nn.ReLU6(inplace=True)
    )


def conv_dsc(in_channel, out_channel, stride=1):
    """
        深度可分离卷积：DW+BN+Act + Conv+BN+Act
        groups：用于控制输入和输出之间的连接方式，如果in_channel被分成groups份，则此卷积层将使用每个组的内部卷积通道，
    而不是将所有通道用于单个卷积，这对于实现轻量级模型的深度可分离卷积层非常有效，因为它将大大降低计算成本，同时保持模型准确性。
        通常，在深度可分离卷积层中，groups的值被设置为in_channel的值，以便将每个输入通道分别与一个内部卷积通道进行关联性。
        如果groups参数的默认值仍然是1，则所有输入通道都将被连接到输出，这就将导致标准卷积操作。
    """
    return nn.Sequential(
        # 输入：in_channels，输出：in_channel，卷积核大小：3
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
        nn.BatchNorm2d(in_channel),
        nn.ReLU6(inplace=True),

        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self, in_dim=3, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes
        self.stage1 = nn.Sequential(
            # [3x224x224] - > [32x112x112]
            conv_bn(in_dim, 32, 2),
            # [32x112x112] -> [32x112x112]
            # [32x112x112] -> [64x112x112]
            conv_dsc(32, 64, 1),

            # [64x112x112] -> [64x56x56]
            # [64x56x56] -> [128x56x56]
            conv_dsc(64, 128, 2),

            # [128x56x56] -> [128x56x56]
            # [128x56x56] -> [128x56x56]
            conv_dsc(128, 128, 1),

            # [128x56x56] -> [128x28x28]
            # [128x28x28] -> [256x28x28]
            conv_dsc(128, 256, 2),

            # [256x28x28] -> [256x28x28]
            # [256x28x28] -> [256x28x28]
            conv_dsc(256, 256, 1),
        )

        self.stage2 = nn.Sequential(
            # [256x28x28] -> [256x14x14]
            # [256x14x14] -> [512x14x14]
            conv_dsc(256, 512, 2),

            # [512x14x14] -> [512x14x14]
            # [512x14x14] -> [512x14x14]
            conv_dsc(512, 512, 1),

            # [512x14x14] -> [512x14x14]
            # [512x14x14] -> [512x14x14]
            conv_dsc(512, 512, 1),

            # [512x14x14] -> [512x14x14]
            # [512x14x14] -> [512x14x14]
            conv_dsc(512, 512, 1),

            # [512x14x14] -> [512x14x14]
            # [512x14x14] -> [512x14x14]
            conv_dsc(512, 512, 1),

            # [512x14x14] -> [512x14x14]
            # [512x14x14] -> [512x14x14]
            conv_dsc(512, 512, 1),
        )

        self.stage3 = nn.Sequential(
            # [512x14x14] -> [512x7x7]
            # [512x7x7] -> [1024x7x7]
            conv_dsc(512, 1024, 2),

            # [1024x7x7] -> [1024x7x7]
            # [1024x7x7] -> [1024x7x7]
            conv_dsc(1024, 1024, 1),
        )

        # [1024x7x7] -> [1024x1x1]
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # [1024] -> [num_classes]
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


mobileNet_V1 = MobileNetV1(num_classes=5)


writer.add_graph(mobileNet_V1, torch.rand(1, 3, 224, 224))
writer.close()













"""
MobileNetV2
"""
writer = SummaryWriter(log_dir='logs/mobilenet_v2')


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor

    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_dim=3, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # 第一个卷积层
        features.append(ConvBNReLU(in_dim, input_channel, stride=2))
        # 创建7个InvertedResidual模块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            # n = 1 + 2 + 3 + 4 + 3 + 3 + 1 = 17， 一共17个InvertedResidual模块
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 最后一个卷积模块
        features.append(ConvBNReLU(input_channel, last_channel, 1))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


mobileNet_V2 = MobileNetV2(num_classes=5)
writer.add_graph(mobileNet_V2, torch.rand(1, 3, 224, 224))
writer.close()




"""
MobileNetV3
"""
from typing import Callable, List, Optional
from functools import partial
from torch import Tensor
import torch.nn.functional as F


writer = SummaryWriter(log_dir='logs/mobilenet_v3')


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    """
    全连接注意力机制模块
    """
    def __init__(self, input_c: int, squeeze_factor: int=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:
    def __init__(self, input_c: int, kernel: int, expanded_c: int, out_c: int, use_se: bool,
                 activation: str, stride: int, width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError('illegal stride value.')

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_c != cnf.input_c:
            layers.append(
                ConvBNActivation(cnf.input_c, cnf.expanded_c, kernel_size=1,
                                 norm_layer=norm_layer, activation_layer=activation_layer)
            )

        layers.append(
            ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel,
                             stride=cnf.stride, groups=cnf.expanded_c, norm_layer=norm_layer,
                             activation_layer=activation_layer)
        )

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        layers.append(
            ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer,
                             activation_layer=nn.Identity)
        )
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int, in_dim=3, num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty.')
        elif not (isinstance(inverted_residual_setting, List) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(
            ConvBNActivation(in_dim, firstconv_output_c, kernel_size=3, stride=2,
                             norm_layer=norm_layer, activation_layer=nn.Hardswish)
        )
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(
            ConvBNActivation(lastconv_input_c, lastconv_output_c, kernel_size=1,
                             norm_layer=norm_layer, activation_layer=nn.Hardswish)
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(in_dim=3, num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, 'RE', 1),
        bneck_conf(16, 3, 64, 24, False, 'RE', 2),
        bneck_conf(24, 3, 72, 24, False, 'RE', 1),

        bneck_conf(24, 5, 72, 40, True, 'RE', 2),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),

        bneck_conf(40, 3, 240, 80, False, 'HS', 2),
        bneck_conf(80, 3, 200, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, True, 'HS', 1),
        bneck_conf(80, 3, 400, 112, True, 'HS', 1),
        bneck_conf(112, 3, 672, 112, True, 'HS', 1),

        bneck_conf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        in_dim=in_dim,
        num_classes=num_classes
    )


def mobilenet_v3_small(in_dim=3, num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1

        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),

        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),

        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]

    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        in_dim=in_dim,
        num_classes=num_classes
    )


mobileNet_V3 = mobilenet_v3_large(num_classes=5)
writer.add_graph(mobileNet_V3, torch.rand(1, 3, 224, 224))
writer.close()
