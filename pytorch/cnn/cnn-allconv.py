import time
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random_seed = 1
learning_rate = 0.001
num_epochs = 15
batch_size = 256
num_classes = 10

train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for images, labels in train_loader:
    print(f'Image batch dimensions: {images.shape}')
    print(f'Image label dimensions: {labels.shape}')
    break


class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        # 28x28x1 -> 28x28x4
        self.conv_1 = torch.nn.Conv2d(1, 4, kernel_size=(3, 3), stride=1, padding=1)
        # 28x28x4 -> 14x14x4
        self.conv_2 = torch.nn.Conv2d(4, 4, kernel_size=(3, 3), stride=2, padding=1)
        # 14x14x4 -> 14x14x8
        self.conv_3 = torch.nn.Conv2d(4, 8, kernel_size=(3, 3), stride=1, padding=1)
        # 14x14x8 -> 7x7x8
        self.conv_4 = torch.nn.Conv2d(8, 8, kernel_size=(3, 3), stride=2, padding=1)
        # 7x7x8   -> 7x7x16
        self.conv_5 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=2, padding=1)
        # 7x7x16  -> 4x4x16
        self.conv_6 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=2, padding=1)
        # 4x4x16 -> 4x4xself.num_classes
        self.conv_7 = torch.nn.Conv2d(16, self.num_classes, kernel_size=(3, 3), stride=1)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)

        out = self.conv_2(out)
        out = F.relu(out)

        out = self.conv_3(out)
        out = F.relu(out)

        out = self.conv_4(out)
        out = F.relu(out)

        out = self.conv_5(out)
        out = F.relu(out)

        out = self.conv_6(out)
        out = F.relu(out)

        out = self.conv_7(out)
        out = F.relu(out)

        logits = F.adaptive_avg_pool2d(out, 1)
        # drop width
        # 对张量 out 进行自适应平均池化，并将最后一列挤压（squeeze）掉，从而得到池化后的结果。
        logits.squeeze_(-1)
        # drop height
        logits.squeeze_(-1)
        probas = torch.softmax(logits, dim=1)
        return logits, probas


torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)

        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


start_time = time.time()
for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)
        # 训练
        logits, probas = model(features)
        # 计算损失
        cost = F.cross_entropy(logits, targets)
        # 优化器清零
        optimizer.zero_grad()
        # 反向传播
        cost.backward()
        # 优化器更新值
        optimizer.step()
        if not batch_idx % 50:
            print(f'Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Cost: {cost}')
    # 模型验证
    model = model.eval()
    print(f'Epoch: {epoch+1}/{num_epochs} training accuracy: {compute_accuracy(model, train_loader)}')

    print(f'Time elapsed: {(time.time() - start_time)/60} min')

print(f'Total Training Time: {(time.time() - start_time)/60} min')

print(f'Test accuracy: {compute_accuracy(model, test_loader)}')