import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 从 lab1 的实验数据中不难看出，之前实现的MLP在测试集上的最高准确率为98.14%，此时超参数配置为：
# config = {
#     'batch_size': 64,
#     'epochs': 20,
#     'lr': 0.001,
#     'hidden_layers': [128, 128],   # 每层神经元个数，[] 表示无隐藏层，这里的列表表示两个隐藏层，每个隐藏层有 128 神经元
#     'activation': 'relu',           # 'relu', 'tanh', 'sigmoid'
#     'dropout': 0.2,                 # dropout 概率，0 表示不使用
#     'use_batch_norm': False,
# }
# 因此在此我们先复现这个结果，作为后续CNN实验的基准

# ---------------------------
# 1. 超参数（基准配置）
# ---------------------------
config = {
    'batch_size': 64,
    'epochs': 20,
    'lr': 0.001,
    'hidden_layers': [128, 128],   # 每层神经元个数，[] 表示无隐藏层，这里的列表表示两个隐藏层，每个隐藏层有 128 神经元
    'activation': 'relu',           # 'relu', 'tanh', 'sigmoid'
    'dropout': 0.2,                 # dropout 概率，0 表示不使用
    'use_batch_norm': False,
}

# ---------------------------
# 2. 数据加载与预处理
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),          # 将 PIL 图像转为 Tensor，并除以 255 归一化到 [0,1]
    transforms.Lambda(lambda x: x.view(-1))  # 展平为 784 维向量
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 从训练集中划分 10% 作为验证集
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# ---------------------------
# 3. 可配置的 MLP 模型
# ---------------------------
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation: {name}')

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_layers=[128,128],
                 activation='relu', dropout=0.0, use_batch_norm=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# 4. 训练函数
# ---------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    return train_losses, val_losses, train_accs, val_accs

# # ---------------------------
# # 5. 绘制损失与准确率曲线
# # ---------------------------
# def plot_curves(train_losses, val_losses, train_accs, val_accs, title_suffix=''):
#     epochs = range(1, len(train_losses)+1)
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,2,1)
#     plt.plot(epochs, train_losses, 'b-', label='Train Loss')
#     plt.plot(epochs, val_losses, 'r-', label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(f'Loss Curves {title_suffix}')

#     plt.subplot(1,2,2)
#     plt.plot(epochs, train_accs, 'b-', label='Train Acc')
#     plt.plot(epochs, val_accs, 'r-', label='Val Acc')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.title(f'Accuracy Curves {title_suffix}')
#     plt.show()

# ---------------------------
# 6. 运行基准模型
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = FlexibleMLP(input_dim=784, output_dim=10,
                    hidden_layers=config['hidden_layers'],
                    activation=config['activation'],
                    dropout=config['dropout'],
                    use_batch_norm=config['use_batch_norm'])
print(model)

train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, epochs=config['epochs'],
    lr=config['lr'], device=device
)
# plot_curves(train_losses, val_losses, train_accs, val_accs, title_suffix='(Baseline)')

# 最终测试集评估
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        test_total += target.size(0)
        test_correct += (predicted == target).sum().item()
print(f'Test accuracy: {test_correct / test_total:.4f}')

# 复现完成

# 搭建一个最小化CNN,确认机制正确

# ---------------------------
# 7. CNN 数据加载：保留图像空间结构
# ---------------------------
# CNN 的输入需要保持为 (batch_size, channels, height, width)，
# 因此这里不能使用前面 MLP 的 flatten transform。
cnn_config = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
}

cnn_transform = transforms.Compose([
    transforms.ToTensor(),          # 输出形状为 (1, 28, 28)，保留通道、高、宽
])

cnn_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=cnn_transform)
cnn_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=cnn_transform)

cnn_train_size = int(0.9 * len(cnn_train_dataset))
cnn_val_size = len(cnn_train_dataset) - cnn_train_size
cnn_train_dataset, cnn_val_dataset = torch.utils.data.random_split(
    cnn_train_dataset,
    [cnn_train_size, cnn_val_size]
)

cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=cnn_config['batch_size'], shuffle=True)
cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=cnn_config['batch_size'], shuffle=False)
cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=cnn_config['batch_size'], shuffle=False)


# ---------------------------
# 8. 最小化 CNN 模型
# ---------------------------
class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            # 输入:  (batch, 1, 28, 28)
            # 输出:  (batch, 16, 28, 28)
            # padding=1 使 3x3 卷积后图像高宽保持不变。
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 输出: (batch, 16, 14, 14)
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

    def print_shape_flow(self, batch_size=4):
        x = torch.randn(batch_size, 1, 28, 28)
        print('\nMinimalCNN tensor shape flow:')
        print(f'Input:          {tuple(x.shape)}')
        x = self.conv_block[0](x)
        print(f'After Conv2d:   {tuple(x.shape)}')
        x = self.conv_block[1](x)
        print(f'After ReLU:     {tuple(x.shape)}')
        x = self.conv_block[2](x)
        print(f'After MaxPool:  {tuple(x.shape)}')
        x = self.classifier[0](x)
        print(f'After Flatten:  {tuple(x.shape)}')
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        print(f'Output logits:  {tuple(x.shape)}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


minimal_cnn = MinimalCNN()
minimal_cnn.print_shape_flow()
print(minimal_cnn)
print(f'MinimalCNN trainable parameters: {count_parameters(minimal_cnn)}')

cnn_train_losses, cnn_val_losses, cnn_train_accs, cnn_val_accs = train_model(
    minimal_cnn,
    cnn_train_loader,
    cnn_val_loader,
    epochs=cnn_config['epochs'],
    lr=cnn_config['lr'],
    device=device
)

cnn_test_acc = evaluate_accuracy(minimal_cnn, cnn_test_loader, device=device)
print(f'MinimalCNN test accuracy: {cnn_test_acc:.4f}')

#

# ---------------------------
# 9. LeNet-style CNN：Conv -> Pool -> Conv -> Pool -> FC -> FC
# ---------------------------
class LeNetStyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: (batch, 1, 28, 28)
            # 输出: (batch, 6, 24, 24)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            # 输出: (batch, 6, 12, 12)
            nn.MaxPool2d(kernel_size=2),

            # 输出: (batch, 16, 8, 8)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            # 输出: (batch, 16, 4, 4)
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def print_shape_flow(self, batch_size=4):
        x = torch.randn(batch_size, 1, 28, 28)
        print('\nLeNetStyleCNN tensor shape flow:')
        print(f'Input:          {tuple(x.shape)}')
        x = self.features[0](x)
        print(f'After Conv1:    {tuple(x.shape)}')
        x = self.features[1](x)
        x = self.features[2](x)
        print(f'After Pool1:    {tuple(x.shape)}')
        x = self.features[3](x)
        print(f'After Conv2:    {tuple(x.shape)}')
        x = self.features[4](x)
        x = self.features[5](x)
        print(f'After Pool2:    {tuple(x.shape)}')
        x = self.classifier[0](x)
        print(f'After Flatten:  {tuple(x.shape)}')
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        print(f'After FC1:      {tuple(x.shape)}')
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        print(f'After FC2:      {tuple(x.shape)}')
        x = self.classifier[5](x)
        print(f'Output logits:  {tuple(x.shape)}')


lenet_cnn = LeNetStyleCNN()
lenet_cnn.print_shape_flow()
print(lenet_cnn)
print(f'LeNetStyleCNN trainable parameters: {count_parameters(lenet_cnn)}')

lenet_train_losses, lenet_val_losses, lenet_train_accs, lenet_val_accs = train_model(
    lenet_cnn,
    cnn_train_loader,
    cnn_val_loader,
    epochs=cnn_config['epochs'],
    lr=cnn_config['lr'],
    device=device
)

lenet_test_acc = evaluate_accuracy(lenet_cnn, cnn_test_loader, device=device)
print(f'LeNetStyleCNN test accuracy: {lenet_test_acc:.4f}')
