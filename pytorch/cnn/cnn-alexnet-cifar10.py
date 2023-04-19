import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from pytorch.helper_evaluate import compute_accuracy
from pytorch.helper_data import get_dataloaders_cifar10
from pytorch.helper_train import train_classifier_simple_v1


# 假如CUDA可用，则将Torch后端设置为cuDNN确定性模型
if torch.cuda.is_available():
    '''
        在cuDNN确定性模式下，卷积和其他操作使用的算法是确定性的，这意味着使用相同的输入和参数多次运行
    相同的操作将产生相同的输出。这对在GPU上训练神经网络时实现可重复性可能有所帮助。
        但是，这种模式可能会导致性能降低。
    '''
    torch.backends.cudnn.deterministic = True


# 设置随机种子
def set_all_seeds(seed):
    # 设置随机种子，这样在完全相同的条件下返回的结果通常是相同的
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 设置cuDNN确定性模型
def set_deterministic():
    # 假如CUDA可用
    if torch.cuda.is_available():
        # cuDNN禁止基于运行时的自动调整算法的性能测试，并使用默认的算法来运行卷积操作
        torch.backends.cudnn.benchmark = False
        # 开启cuDNN确定性模式
        torch.backends.cudnn.deterministic = True
    # 设置随机种子，并且用于确保运行模型时产生的随机数是确定性的，deterministic(确定性)
    torch.set_deterministic(True)


# 设置超参数
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
NUM_EPOCHS = 40
NUM_CLASSES = 10
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 设置随机种子
set_all_seeds(RANDOM_SEED)

# 训练数据进行预处理
train_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                      transforms.RandomCrop((64, 64)),
                                      transforms.ToTensor()])
# 测试数据进行预处理
test_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                      transforms.CenterCrop((64, 64)),
                                      transforms.ToTensor()])
# 加载数据
train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
    batch_size=BATCH_SIZE,
    num_workers=2,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    validation_fraction=0.1
)

# 打印训练数据
print('Training Set:\n')
for images, labels in train_loader:
    print(f'Image batch dimensions: {images.size()}')
    print(f'Image label dimensions: {labels.size()}')
    print(labels[:10])
    break

# 打印验证数据
print('\nValidation Set:')
for images, labels in valid_loader:
    print(f'Image batch dimensions: {images.size()}')
    print(f'Image label dimensions: {labels.size()}')
    print(labels[:10])
    break

# 打印测试数据
print('\nTesting Set:')
for images, labels in train_loader:
    print(f'Image batch dimensions: {images.size()}')
    print(f'Image label dimensions: {labels.size()}')
    print(labels[:10])
    break


# 搭建AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 输入通道3，输出64，卷积核11，步长4，填充2
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # inplace = True，则会直接在原地修改输入张量，而不是创建一个新的输出张量，这可以减少内存占用和计算开销，
            # 但同时会使导数的计算变的比较复杂，因为这样会修改输入张量的值，因此梯度的传播需要反向传播算法进行调整。
            nn.ReLU(inplace=True),
            # 最大池化层，核大小3，步长2
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应池化层，将输出结果改成6x6大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return probas       # ???? 应该是 return probas吧。


# 设置随机数生成器的种子，以确保随机数生成器按照相同的顺序生成随机数序列，作用主要是提高模型的可重复性和可验证性
torch.manual_seed(RANDOM_SEED)
# 获取模型
model = AlexNet(NUM_CLASSES)
# 加载到GPU
model.to(DEVICE)
# 优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 调用训练函数
log_dict = train_classifier_simple_v1(num_epochs=NUM_EPOCHS,
                                      model=model,
                                      optimizer=optimizer,
                                      device=DEVICE,
                                      train_loader=train_loader,
                                      valid_loader=valid_loader,
                                      logging_interval=50)
# 获取损失函数
loss_list = log_dict['train_loss_per_batch']
# 画图
plt.plot(loss_list, label='Minibatch loss')
plt.plot(np.convolve(loss_list, np.ones(200,)/200, mode='valid'), label='Running average')
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.legend()
plt.show()

# 画图
plt.plot(np.arange(1, NUM_EPOCHS+1), log_dict['train_acc_per_epoch'], label='Training')
plt.plot(np.arange(1, NUM_EPOCHS+1), log_dict['valid_acc_per_epoch'], label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# 验证在训练集，验证集，测试集上的准确率
with torch.set_grad_enabled(False):
    train_acc = compute_accuracy(model=model, data_loader=train_loader, device=DEVICE)
    test_acc = compute_accuracy(model=model, data_loader=test_loader, device=DEVICE)
    valid_acc = compute_accuracy(model=model, data_loader=valid_loader, device=DEVICE)

print(f'Train Acc: {valid_acc:.2f}%')
print(f'Validation Acc: {valid_acc:.2f}%')
print(f'Test Acc: {test_acc:.2f}%')
