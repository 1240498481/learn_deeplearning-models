import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
NUM_EPOCHS = 20
NUM_CLASSES = 10

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 判断输入通道是1 or 3
GRAYSCALE = False

# 数据切分
train_indices = torch.arange(0, 48000)
valid_indices = torch.arange(48000, 50000)

# 加载数据
train_and_valid = datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)

train_dataset = Subset(train_and_valid, train_indices)
valid_dataset = Subset(train_and_valid, valid_indices)

test_dataset = datasets.CIFAR10(root='data', train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

torch.manual_seed(0)


def _bn_function_factory(norm, relu, conv):
    """
        主要作用是为denseNet模型中的稠密连接块（Dense Block）构建一个内部的BN层，从而实现对输入特征向量的标准化。
        参数norm，relu，conv分别代表着BN层，ReLU激活函数和卷积层。
        在函数内部，将BN层和ReLU激活函数应用于输入特征向量上，并将它们的结果拼接在一起，得到一个concated_features特征向量，
    然后将这个特征向量传递给卷积层进行处理，得到新的bottlenecl_output特征向量。最终，bn_function函数返回的就是这个处理后的
    特征向量，作为该稠密连接块的输出。
        在denseNet中，稠密连接块采用了上述_bn_function_factory函数构建的内部BN层，并将其应用于稠密单元内的所有输入，
    以实现对输入特征向量的标准化，这样有助于加快训练速度并提高模型性能。
    """
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        """
            作用是定义密集块(Dense Block)的基本组成单元-密集层(Dense Layer)
            在denseNet的实现中，密基层是由若干个卷积层构成的，其中每个卷积层的输入都是由前面所有层的输出所组成（及稠密连接），
        输出的通道数也是预先定义好的，经过若干个这样的密基层之后，输入的特征图就被累积、压缩和混合，产生了复杂而有信息量的特征表示。
            总的来说，此类的作用是为denseNet实现中的密集块提供了一个高效和易于理解的方式来定义密集层，并且保证反向传播提示时有效和正确。

        :param num_input_features: 是一个整数值，代表输入特征图的通道数。
            在denseNet中，当我们连接多个密集块时，需要确保每个密集块的输入特征图的通道数与前一个密集块的输出通道数相等。
        :param growth_rate:
        :param bn_size:
        :param drop_rate:
        :param memory_efficient: 控制卷积层的计算方式。
            如果被设置为True，则在密基层中的卷积计算是内存和计算量更小的版本，但这可能导致模型精度稍微降低，
            如果被设置为False，会使用更加准确的、精度更高的内部计算方式，但这可能导致在使用较大数据集时内存和计算速度较慢。
            这个标志只能改变denseNet中密基层内部的卷积层的计算方式，对其他卷积层和网络的其他部分并没有作用。
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        # 生成新的特征，如果drop_rate大于0，则进行随机失活，并传递到下一层
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        """
            由多个密集块组成，每个密集块又是由多个卷积层组成。
        :param num_layers:
        :param num_input_features:
        :param bn_size:
        :param growth_rate:
        :param drop_rate:
        :param memory_efficient:
        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module(f'denselayer{i+1}', layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        """
            主要用来连接相邻的两个_DenseBlock，并在它们之间执行下采样操作以降低特征图大小。
            其中，归一化层，激活函数层和卷积层的作用是将输入特征图的通道数从num_input_features -> num_output_features，以
        减少网络中的参数数量和计算复杂度。
            而池化层用于下采样，将特征图的大小减半。这样，输出的特征图就可以作为下一个_DenseBlock的输入。
        :param num_input_features:
        :param num_output_features:
        """
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_featuremaps=64,
                 bn_size=4, drop_rate=0,
                 num_classes=1000,
                 memory_efficient=False,
                 grayscale=False):
        """

        :param growth_rate: 每个_DenseBlock中卷积层输出的通道数，也就是每个密集块中增加的特征图通道数
        :param block_config: 网络每个DenseBlock块的层数
        :param num_init_featuremaps: 每个卷积层的输入特征图的通道数
        :param bn_size: 指定了在_DenseBlock中的卷积层之后，BN层应该使用的特征图通道数
        :param drop_rate: 是否进行随机失活
        :param num_classes: 输出类别
        :param memory_efficient: 是否使用内存和计算量更小的版本
        :param grayscale: 表示输入的图像的通道
        """
        super(DenseNet121, self).__init__()

        if grayscale:
            in_channels = 1
        else:
            in_channels = 3

        # 获取第一层的输出
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv2d(in_channels, num_init_featuremaps, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_features=num_init_featuremaps)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])
        )

        # 每个卷积层的输入特征图的通道数
        num_features = num_init_featuremaps
        # 循环创建_DenseBlock块
        for i, num_layers in enumerate(block_config):
            # 创建块
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            # 将当前块的输出特征也保存下来，方便传给下一层
            self.features.add_module(f'denseblock{i+1}', block)
            # 在一次连续操作之后输出的特征图的通道数，即增加后的通道数。原来的通道数 + 传过来的通道数
            num_features = num_features + num_layers * growth_rate
            # 假如没有结束，则将通道数变为原来的1/2
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # 添加最后一层的归一化层
        self.features.add_module(f'norm5', nn.BatchNorm2d(num_features))
        # 获取线性层，输出为类目数
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用kaiming初始化方法对神经网络的权重进行初始化，该方法旨在解决梯度消失和梯度爆炸问题，特别是在较深的神经网络中。
                # 它基于每层权重矩阵的输入与输出通道数来计算正态分布的标准差，从而改进了传统的均匀分布初始化方法。
                # 这有助于保证前向传播时信号在激活函数的非线性范围内传递，从而有效避免梯度消失问题和加速训练收敛。
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                # 这两行代码使用常数初始化方法初始化神经网络层的权重和偏置。
                # 在这里，权重被初始化为1，偏置被初始化为0
                # 常数初始化方法是一种简单但有效的初始化方法，它可以用于任何类型的网络层。
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 偏置被初始化为0
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


torch.manual_seed(RANDOM_SEED)

# 初始化模型
model = DenseNet121(num_classes=NUM_CLASSES, grayscale=GRAYSCALE)
model.to(DEVICE)


writer = SummaryWriter('logs')
writer.add_graph(model, torch.rand(1, 3, 224, 224).to(DEVICE))

writer.close()


# 优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 计算准确率
def compute_acc(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    model.eval()

    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        assert predicted_labels.size() == targets.size()
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


start_time = time.time()

cost_list = []
train_acc_list, valid_acc_list = [], []

# 开始训练
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        optimizer.step()

        cost_list.append(cost.item())
        if not batch_idx % 150:
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | Batch {batch_idx:03d}/{len(train_loader):03d} | Cost: {cost:.4f}')

    model.eval()
    with torch.set_grad_enabled(False):
        train_acc = compute_acc(model, train_loader, device=DEVICE)
        valid_acc = compute_acc(model, valid_loader, device=DEVICE)

        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d}\n'
              f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f}')
        # 此处添加准确率的时候，一定要先移到CPU中，再转为numpy，否则会报错！
        train_acc_list.append(train_acc.cpu().numpy())
        valid_acc_list.append(valid_acc.cpu().numpy())

    elapsed = (time.time() - start_time) / 60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')


plt.plot(cost_list, label='Minibatch cost')
plt.plot(np.convolve(cost_list, np.ones(200,)/200, mode='valid'), label='Running average')
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.legend()
plt.show()

plt.plot(np.arange(1, NUM_EPOCHS+1), train_acc_list, label='Training')
plt.plot(np.arange(1, NUM_EPOCHS+1), valid_acc_list, label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

with torch.set_grad_enabled(False):
    test_acc = compute_acc(model=model, data_loader=test_loader, device=DEVICE)
    valid_acc = compute_acc(model=model, data_loader=valid_loader, device=DEVICE)
    train_acc = compute_acc(model=model, data_loader=train_loader, device=DEVICE)

print(f'Train Acc: {train_acc:.2f}%')
print(f'Validtion Acc: {valid_acc:.2f}%')
print(f'Test Acc: {test_acc:.2f}%')




