import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from collections import Counter
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import sys
from torchmetrics import ConfusionMatrix
import matplotlib
from mlxtend.plotting import plot_confusion_matrix

sys.path.append('../../pytorch_ipynb')


# 批次大小
BATCH_SIZE = 256
# 训练轮数
NUM_EPOCHS = 40
# 学习率
LEARNING_RATE = 0.0001
# CPU核数
NUM_WORKERS = 4


# 创建nn模型
class PyTorchModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            # 输入通道：3，输出通道：64，卷积核大小：11，步长：4，填充2
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 最大池化，池化核大小：3，步长：2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输入通道：64，输出通道192，卷积核大小：5，填充：2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 最大池化
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输入通道：192，输出通道：384，卷积核：3，填充：1
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 输入通道：384，输出通道：256，卷积核大小：3，填充：1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 输入通道：256，输出通道：256，卷积核大小：3，填充：1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 最大池化
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应平均池化层，用来将输入的图像或特征映射调整为指定的输出大小（此处为6，6）
        # 该层常用于处理不同大小的输入数据，并能使得在全连接层之前需要展开的向量大小固定。
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 分类层
        self.classifier = nn.Sequential(
            # 随机失活一部分神经元，有助于减轻模型过拟合的现象。
            nn.Dropout(0.5),
            # 全连接层，输入256*6*6，输出4096
            nn.Linear(256 * 6 * 6, 4096),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 随机失活
            nn.Dropout(0.5),
            # 全连接层，输入4096，输出4096
            nn.Linear(4096, 4096),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 全连接层，输入4096，输出类目数
            nn.Linear(4096, num_classes)
        )

    # 定义该层或模型实例如何进行前向传播计算，并处理输入数据以生成输出结果。
    def forward(self, x):
        # 通过全卷积的特征层，提取特征
        x = self.features(x)
        # 使用自适应平均池化，处理输出大小
        x = self.avgpool(x)
        # 用于将张量扁平化，主要作用是将输入张量按照其维度展开成1维，和全连接层对应
        x = torch.flatten(x, start_dim=1)
        # 使用全连接层输出类目
        logits = self.classifier(x)
        return logits


# 将pytorch模型转为pytorch_lightning模型
class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置模型
        self.model = model
        # 假如模型包含dropout，则设置dropout
        if hasattr(model, 'dropout_proba'):
            self.dropout_proba = model.dropout_proba
        # 保存模型的超参数，会创建一个hparams.yaml文件，其中包含超参数
        self.save_hyperparameters(ignore=['model'])
        # 计算训练集上的准确率
        self.train_acc = torchmetrics.Accuracy()
        # 计算验证集上的准确率
        self.valid_acc = torchmetrics.Accuracy()
        # 计算测试集上的准确率
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # 返回模型
        return self.model(x)

    def _shared_step(self, batch):
        # 获取提取的体征和真实标签
        features, true_labels = batch
        # 预测
        logits = self(features)
        # 采用交叉熵函数，计算预测值和真实值的损失，该函数组合了softmax函数和负对数似然损失
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        # 预测标签
        predicted_labels = torch.argmax(logits, dim=1)
        # 返回损失，真实标签，预测标签
        return loss, true_labels, predicted_labels

    # 训练
    def training_step(self, batch, batch_idx):
        # 获取损失，真实标签，预测标签
        loss, true_labels, predicted_labels = self._shared_step(batch)
        # 输出loss
        self.log('train_loss', loss)
        # 模型验证
        self.model.eval()
        # 反向传播，计算梯度
        with torch.no_grad():
            # 损失，真实标签，预测标签
            _, true_labels, predicted_labels = self._shared_step(batch)
        # 通过预测标签和真实标签，计算准确率
        self.train_acc(predicted_labels, true_labels)
        # 输出准确率
        self.log('train_acc', self.train_acc, on_epoch=True, on_step=False)
        # 再次训练
        self.model.train()
        # 返回loss
        return loss

    # 验证
    def validation_step(self, batch, batch_idx):
        # 损失，真实标签，预测标签
        loss, true_labels, predicted_labels = self._shared_step(batch)
        # 验证损失
        self.log('valid_loss', loss)
        # 验证准确率
        self.valid_acc(predicted_labels, true_labels)
        # 输出
        self.log('valid_acc', self.valid_acc, on_epoch=True, on_step=False, prog_bar=True)

    # 测试
    def test_step(self, batch, batch_idx):
        # 损失，真实标签，预测标签
        loss, true_labels, predicted_labels = self._shared_step(batch)
        # 测试准确率
        self.test_acc(predicted_labels, true_labels)
        # 输出
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)

    # 优化函数
    def configure_optimizers(self):
        # 采用Adam优化函数，学习率为0.0001
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

"""
注释掉的这段，主要是用来做数据展示的，对实际的训练，并没有任何帮助

# 下载训练数据
train_dataset = datasets.CIFAR10(
    root='datasets/data', train=True, transform=transforms.ToTensor(), download=True
)
# 加载数据，一个batch256，采用4个CPU核数加载，
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    # 当数据集无法被batch_size整除时，将舍去最后一个不足部分的批次
    drop_last=True,
    # 随机打乱顺序
    shuffle=True,
)

# 下载测试集
test_dataset = datasets.CIFAR10(
    root='datasets/data', train=False, transform=transforms.ToTensor()
)
# 加载测试集
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    # 不能被batch_size整除时，不舍弃最后一轮
    drop_last=False,
    # 不打乱数据
    shuffle=False,
)

# 跟踪可迭代对象中每个元素出现的次数
train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())


# Counter()函数的most_common(n)方法可以获取计数器字典中出现次数最多的前n个元素，并按照他们的频率进行排序
# 参数n是一个可选参数，如果未指定，则返回所有元素及其计数
majority_class = test_counter.most_common(1)[0]
# 打印出现次数最多的元素第一个下标
print(f'Majority class: {majority_class[0]}')

# 计算所占整体数据的比重  出现次数最多的元素次数/所有出现次数的和
baseline_acc = majority_class[1] / sum(test_counter.values())
print(f'Accuracy when always predicting the majority class: {baseline_acc:.2f} ({baseline_acc*100:.2f})')


# 创建一个8x8英寸的画布
plt.figure(figsize=(8, 8))
# 关闭坐标轴的显示，隐藏x轴，y轴，标签以及边界框线
plt.axis('off')
# 设置标题
plt.title('Training images')
# 显示图片，并取出前64个进行显示
plt.imshow(np.transpose(torchvision.utils.make_grid(
    # 显示前64个图像
    images[:64],
    # 设定直接相邻两张图片的间隔为2像素
    padding=2,
    # 转置操作，因为pytorch处理图像时采用的CxHxW，而plt.imshow()默认采用RGB模型，所以需要进行维度顺序的转换
    normalize=True
), (1, 2, 0)))
plt.show()

"""


# 数据模型类
class DataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        # 数据的路径
        self.data_path = data_path

    # 完成数据处理相关操作
    def prepare_data(self) -> None:
        # 下载数据
        datasets.CIFAR10(root=self.data_path, download=True)
        # 对训练数据进行修改
        self.train_transform = transforms.Compose([
            # 重置为70x70
            transforms.Resize((70, 70)),
            # 随机截取64x64
            transforms.RandomCrop((64, 64)),
            # 转为Tensor
            transforms.ToTensor(),
        ])
        # 对测试数据进行修改
        self.test_transform = transforms.Compose([
            # 重置大小为70x70
            transforms.Resize((70, 70)),
            # 随机截取64x64
            transforms.CenterCrop((64, 64)),
            # 转为Tensor
            transforms.ToTensor(),
        ])
        return

    # 执行
    def setup(self, stage=None):
        # 训练数据
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )
        # 测试数据
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )
        # 数据划分
        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    # 训练数据加载
    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        return train_loader

    # 验证数据加载
    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        return valid_loader

    # 测试数据加载
    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        return test_loader


# 设置随机种子
torch.manual_seed(1)
# 传入数据路径
data_module = DataModule(data_path='./datasets/data')

# 获取模型
pytorch_model = PyTorchModel(num_classes=10)

# 转为LightningModel模型
lightning_model = LightningModel(pytorch_model, learning_rate=LEARNING_RATE)

# 回调函数，在训练期间根据特定事物（如每个step完成，每个epoch终止等）自动被触发并执行某些操作的函数
callbacks = [
    # save_top_k：保存比前1个结果更好的结果
    # mode：max为越大越好
    # monitor：监听的变量为valid_acc
    ModelCheckpoint(
        save_top_k=1, mode='max', monitor='valid_acc'
    )
]
# CSV日志
logger = CSVLogger(save_dir='logs/', name='my-model')

# 设置训练超参数
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=callbacks,
    progress_bar_refresh_rate=50,
    accelerator='auto',
    devices='0',
    logger=logger,
    deterministic=False,
    log_every_n_steps=10,
)

start_time = time.time()
# 开始运行
trainer.fit(model=lightning_model, datamodule=data_module)
# 结束时间
runtime = (time.time() - start_time) / 60
# 打印耗时
print(f'Training took {runtime:.2f} min in total.')

# 读取csv logger日志
metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')

# 写入日志
aggreg_metrics = []
agg_col = 'epoch'
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

# 显示训练曲线
df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[['train_loss', 'valid_loss']].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Loss')
df_metrics[['train_acc', 'valid_acc']].plot(grid=True, legend=True, xlabel='Epoch', ylabel='ACC')
plt.show()

# 调用最好的模型进行测试
trainer.test(model=lightning_model, data_module=data_module, ckpt_path='best')
# 最好的模型的路径
path = trainer.checkpoint_callback.best_model_path
print(path)
# 加载LightningModel模型
lightning_model = LightningModel.load_from_checkpoint(path, model=pytorch_model)
# 模型验证
lightning_model.eval()

# 加载测试数据
test_dataloader = data_module.test_dataloader()
# 获取计算准确率的函数
acc = torchmetrics.Accuracy()

# 计算测试集上的准确率
for batch in test_dataloader:
    features, true_labels = batch

    with torch.no_grad():
        logits = lightning_model(features)

    predicted_labels = torch.argmax(logits, dim=1)
    acc(predicted_labels, true_labels)

# predicted_labels[:5]

# 计算准确率
test_acc = acc.compute()
print(f'Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
# 获取标签类别
class_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# 创建混淆矩阵图，通过将预测类别和真实类别进行比较，混淆矩阵可以帮助我们了解分类器的整体性能以及它所犯的错误类型和程度
cmat = ConfusionMatrix(num_classes=len(class_dict))
# 循环加载数据
for x, y in test_dataloader:
    with torch.no_grad():
        pred = lightning_model(x)
    cmat(pred, y)
# 转为numpy类型
cmat_tensor = cmat.compute()
cmat = cmat_tensor.numpy()
# 画图显示
fig, ax = plot_confusion_matrix(
    conf_mat=cmat,
    class_names=class_dict.values(),
    norm_colormap=matplotlib.colors.LogNorm()
)
plt.show()


