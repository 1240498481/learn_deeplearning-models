import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from vision.transforms._presets import ImageClassification
from vision.utils import _log_api_usage_once
from vision.models._api import register_model, Weights, WeightsEnum
from vision.models._meta import _IMAGENET_CATEGORIES
from vision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torch.utils.tensorboard import SummaryWriter


__all__ = [
    'DenseNet',
    'DenseNet121_Weights',
    'DenseNet161_Weights',
    'DenseNet169_Weights',
    'DenseNet201_Weights',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
]


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float,
                 memory_efficient: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method
    def forward(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception('Memory Efficient not supported in JIT')

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float, memory_efficient: bool=False) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i +1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    def __init__(self, growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False,) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ('norm0', nn.BatchNorm2d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress, check_hash=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(growth_rate: int,
              block_config: Tuple[int, int, int, int],
              num_init_features: int,
              weights: Optional[WeightsEnum],
              progress: bool,
              **kwargs: Any,
              ) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model


_COMMON_META = {
    'min_size': (29, 29),
    'categories': _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/116",
    "_docs": """These weights are ported from LuaTorch.""",
}


class DenseNet121_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet121-a639ec97.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 7978856,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.434,
                    "acc@5": 91.972,
                }
            },
            "_ops": 2.834,
            "_file_size": 30.845,
        },
    )
    DEFAULT = IMAGENET1K_V1


class DenseNet161_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet161-8d451a50.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 28681000,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.138,
                    "acc@5": 93.560,
                }
            },
            "_ops": 7.728,
            "_file_size": 110.369,
        },
    )
    DEFAULT = IMAGENET1K_V1


class DenseNet169_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 14149480,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.600,
                    "acc@5": 92.806,
                }
            },
            "_ops": 3.36,
            "_file_size": 54.708,
        },
    )
    DEFAULT = IMAGENET1K_V1


class DenseNet201_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet201-c1103571.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 20013928,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.896,
                    "acc@5": 93.370,
                }
            },
            "_ops": 4.291,
            "_file_size": 77.373,
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet121_Weights.IMAGENET1K_V1))
def densenet121(*, weights: Optional[DenseNet121_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    weights = DenseNet121_Weights.verify(weights)

    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet161_Weights.IMAGENET1K_V1))
def densenet161(*, weights: Optional[DenseNet161_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    weights = DenseNet161_Weights.verify(weights)

    return _densenet(48, (6, 12, 36, 24), 96, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet169_Weights.IMAGENET1K_V1))
def densenet169(*, weights: Optional[DenseNet169_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    weights = DenseNet169_Weights.verify(weights)

    return _densenet(32, (6, 12, 32, 32), 64, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=('pretrained', DenseNet201_Weights.IMAGENET1K_V1))
def densenet201(*, weights: Optional[DenseNet201_Weights]=None, progress: bool=True, **kwargs: Any) -> DenseNet:
    weights = DenseNet201_Weights.verify(weights)

    return _densenet(32, (6, 12, 48, 32), 64, weights, progress, **kwargs)


writer = SummaryWriter(log_dir='../logs/DenseNet/densenet121')
writer.add_graph(densenet121(), torch.rand(1, 3, 224, 224))


writer = SummaryWriter(log_dir='../logs/DenseNet/densenet161')
writer.add_graph(densenet161(), torch.rand(1, 3, 224, 224))


writer = SummaryWriter(log_dir='../logs/DenseNet/densenet169')
writer.add_graph(densenet169(), torch.rand(1, 3, 224, 224))


writer = SummaryWriter(log_dir='../logs/DenseNet/densenet201')
writer.add_graph(densenet201(), torch.rand(1, 3, 224, 224))
writer.close()