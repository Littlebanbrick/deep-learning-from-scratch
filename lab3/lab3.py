"""
Lab 3: CNN for MNIST Handwritten Digit Recognition
===================================================

从 Task 1 (MLP 基准) 一路做到 Task 5 (错分分析) 的完整实验脚本。
所有终端输出可重定向到 results/result.txt，所有图表保存到 results/。

运行方式（在 lab3/ 目录下）：
    conda activate dlfs
    python -u lab3.py >> results/result.txt
"""

import os
import time
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无显示环境也能保存图片
import matplotlib.pyplot as plt

# =====================================================================
# 0. 全局设置：随机种子、设备、输出目录
# =====================================================================
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LAB_DIR, 'data')
RESULTS_DIR = os.path.join(LAB_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed=42):
    """每个实验前调用：保证模型初始化与训练过程可复现，
    且各 Task 4 变体从相同的初始权重出发，做到“只改一个变量”的受控对比。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation: {name}')


# =====================================================================
# 1. 训练 / 评估 / 绘图 工具函数
# =====================================================================
def train_model(model, train_loader, val_loader, epochs, lr,
                device=DEVICE, weight_decay=0.0):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    epoch_times = []

    for epoch in range(epochs):
        t0 = time.time()
        # --- 训练阶段 ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
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

        # --- 验证阶段 ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        val_loss /= total
        val_acc = correct / total

        dt = time.time() - t0
        epoch_times.append(dt)
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f'Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f} | {dt:.1f}s')

    return {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
    }


def evaluate_accuracy(model, data_loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


def plot_curves(hist, save_path, title):
    """单个实验的 loss / accuracy 曲线，保存为图片。"""
    epochs = range(1, len(hist['train_losses']) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['train_losses'], 'b-', label='Train Loss')
    plt.plot(epochs, hist['val_losses'], 'r-', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title(f'Loss — {title}')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['train_accs'], 'b-', label='Train Acc')
    plt.plot(epochs, hist['val_accs'], 'r-', label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title(f'Accuracy — {title}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_overlay(hists, labels, save_path, title):
    """把多个实验的验证曲线叠加在一张图上，便于横向对比。"""
    epochs = range(1, len(hists[0]['train_losses']) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for h, l in zip(hists, labels):
        plt.plot(epochs, h['val_losses'], label=l)
    plt.xlabel('Epoch'); plt.ylabel('Val Loss'); plt.legend()
    plt.title(f'Val Loss — {title}')
    plt.subplot(1, 2, 2)
    for h, l in zip(hists, labels):
        plt.plot(epochs, h['val_accs'], label=l)
    plt.xlabel('Epoch'); plt.ylabel('Val Acc'); plt.legend()
    plt.title(f'Val Acc — {title}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# =====================================================================
# 2. 数据加载
# =====================================================================
# 统一使用 transforms.ToTensor()（归一化到 [0,1]），与 lab1 保持一致，
# 使 CNN 与 MLP 的对比只受模型结构影响，不受预处理差异干扰。
BATCH_SIZE = 64
N_TRAIN, N_VAL = 54000, 6000

mlp_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),     # 展平为 784 维向量
])
cnn_transform = transforms.Compose([transforms.ToTensor()])             # 保留 (1,28,28)
aug_transform = transforms.Compose([                                     # Task 4D 数据增强，仅训练集
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])


def make_split(dataset_full, seed=42):
    """用固定种子的 generator 划分 90/10 训练/验证集，
    保证不同 transform 的数据集使用完全相同的索引划分。"""
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset_full, [N_TRAIN, N_VAL], generator=g)


mlp_train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=mlp_transform)
mlp_test = datasets.MNIST(DATA_DIR, train=False, transform=mlp_transform)
mlp_train, mlp_val = make_split(mlp_train_full)

cnn_train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=cnn_transform)
cnn_test = datasets.MNIST(DATA_DIR, train=False, transform=cnn_transform)
cnn_train, cnn_val = make_split(cnn_train_full)

cnn_train_full_aug = datasets.MNIST(DATA_DIR, train=True, download=True, transform=aug_transform)
cnn_train_aug, _ = make_split(cnn_train_full_aug)        # 验证集仍用非增强的 cnn_val

mlp_train_loader = DataLoader(mlp_train, batch_size=BATCH_SIZE, shuffle=True)
mlp_val_loader = DataLoader(mlp_val, batch_size=BATCH_SIZE, shuffle=False)
mlp_test_loader = DataLoader(mlp_test, batch_size=BATCH_SIZE, shuffle=False)

cnn_train_loader = DataLoader(cnn_train, batch_size=BATCH_SIZE, shuffle=True)
cnn_val_loader = DataLoader(cnn_val, batch_size=BATCH_SIZE, shuffle=False)
cnn_test_loader = DataLoader(cnn_test, batch_size=BATCH_SIZE, shuffle=False)

cnn_train_aug_loader = DataLoader(cnn_train_aug, batch_size=BATCH_SIZE, shuffle=True)


# =====================================================================
# 3. 模型定义
# =====================================================================
class FlexibleMLP(nn.Module):
    """lab1 中的可配置 MLP，复现最佳配置 [128,128]+ReLU+Dropout0.2 作为基准。"""
    def __init__(self, input_dim=784, output_dim=10, hidden_layers=[128, 128],
                 activation='relu', dropout=0.0, use_batch_norm=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MinimalCNN(nn.Module):
    """Task 2: 单卷积块最小 CNN，主要用于验证 tensor shape 流转。"""
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                   # (B,16,14,14)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))

    def print_shape_flow(self, batch_size=4):
        x = torch.randn(batch_size, 1, 28, 28)
        print('MinimalCNN tensor shape flow:')
        print(f'  Input:          {tuple(x.shape)}')
        x = self.conv_block[0](x); print(f'  After Conv2d:   {tuple(x.shape)}')
        x = self.conv_block[1](x); print(f'  After ReLU:     {tuple(x.shape)}')
        x = self.conv_block[2](x); print(f'  After MaxPool:  {tuple(x.shape)}')
        x = self.classifier[0](x); print(f'  After Flatten: {tuple(x.shape)}')
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x); print(f'  Output logits: {tuple(x.shape)}')


class LeNetStyleCNN(nn.Module):
    """Task 3/4: 可配置 LeNet 风格 CNN。
       Conv(5)->Act->Pool -> Conv(5)->Act->Pool -> FC(120)->Act->FC(84)->Act->(Dropout)->FC(10)
       通过 channels / activation / pool / dropout 四个旋钮控制 Task 4 的各对比实验。"""
    def __init__(self, channels=(6, 16), activation='tanh', pool='max', dropout=0.0):
        super().__init__()
        act = lambda: get_activation(activation)
        Pool = nn.MaxPool2d if pool == 'max' else nn.AvgPool2d
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5), act(),   # (B,c1,24,24)
            Pool(kernel_size=2),                      # (B,c1,12,12)
            nn.Conv2d(c1, c2, kernel_size=5), act(),  # (B,c2,8,8)
            Pool(kernel_size=2),                      # (B,c2,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 4 * 4, 120), act(),
            nn.Linear(120, 84), act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def print_shape_flow(self, batch_size=4):
        x = torch.randn(batch_size, 1, 28, 28)
        print('LeNetStyleCNN tensor shape flow:')
        print(f'  Input:          {tuple(x.shape)}')
        x = self.features[0](x); print(f'  After Conv1:    {tuple(x.shape)}')
        x = self.features[1](x); x = self.features[2](x); print(f'  After Pool1:    {tuple(x.shape)}')
        x = self.features[3](x); print(f'  After Conv2:    {tuple(x.shape)}')
        x = self.features[4](x); x = self.features[5](x); print(f'  After Pool2:    {tuple(x.shape)}')
        x = self.classifier[0](x); print(f'  After Flatten: {tuple(x.shape)}')
        x = self.classifier[1](x); print(f'  After FC1:     {tuple(x.shape)}')
        x = self.classifier[3](x); print(f'  After FC2:     {tuple(x.shape)}')
        x = self.classifier[6](x); print(f'  Output logits: {tuple(x.shape)}')


# =====================================================================
# 实验记录容器
# =====================================================================
RESULTS = []

def record(name, model, hist, test_acc):
    RESULTS.append({
        'name': name,
        'params': count_parameters(model),
        'train_acc': hist['train_accs'][-1],
        'val_acc': hist['val_accs'][-1],
        'test_acc': test_acc,
        'avg_epoch_time': hist['avg_epoch_time'],
        'hist': hist,
        'model': model,
    })


# =====================================================================
# Task 1: 复现 MLP 基准 (lab1 最佳配置)
# =====================================================================
print('\n' + '=' * 70)
print('Task 1: MLP Baseline (lab1 best: [128,128] + ReLU + Dropout 0.2)')
print('=' * 70)
set_seed()
mlp = FlexibleMLP(hidden_layers=[128, 128], activation='relu', dropout=0.2)
print(mlp)
print(f'MLP trainable parameters: {count_parameters(mlp)}')
mlp_hist = train_model(mlp, mlp_train_loader, mlp_val_loader, epochs=20, lr=1e-3)
mlp_test = evaluate_accuracy(mlp, mlp_test_loader)
print(f'MLP test accuracy: {mlp_test:.4f}')
plot_curves(mlp_hist, os.path.join(RESULTS_DIR, 'task1_mlp_curves.png'), 'MLP Baseline')
record('MLP [128,128]+ReLU+Dropout0.2', mlp, mlp_hist, mlp_test)


# =====================================================================
# Task 2: Minimal CNN (single conv block)
# =====================================================================
print('\n' + '=' * 70)
print('Task 2: Minimal CNN (single conv block)')
print('=' * 70)
set_seed()
minimal = MinimalCNN()
minimal.print_shape_flow()
print(minimal)
print(f'MinimalCNN trainable parameters: {count_parameters(minimal)}')
min_hist = train_model(minimal, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3)
min_test = evaluate_accuracy(minimal, cnn_test_loader)
print(f'MinimalCNN test accuracy: {min_test:.4f}')
plot_curves(min_hist, os.path.join(RESULTS_DIR, 'task2_minimal_cnn_curves.png'), 'MinimalCNN')
record('MinimalCNN (1 conv block)', minimal, min_hist, min_test)


# =====================================================================
# Task 3: LeNet-style CNN (Tanh + MaxPool, channels (6,16))
#   —— 同时作为 Task 4 各控制变量实验的基准
# =====================================================================
print('\n' + '=' * 70)
print('Task 3: LeNet-style CNN (Tanh + MaxPool, channels (6,16))')
print('=' * 70)
set_seed()
lenet = LeNetStyleCNN(channels=(6, 16), activation='tanh', pool='max')
lenet.print_shape_flow()
print(lenet)
print(f'LeNetStyleCNN trainable parameters: {count_parameters(lenet)}')
lenet_hist = train_model(lenet, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3)
lenet_test = evaluate_accuracy(lenet, cnn_test_loader)
print(f'LeNetStyleCNN test accuracy: {lenet_test:.4f}')
plot_curves(lenet_hist, os.path.join(RESULTS_DIR, 'task3_lenet_curves.png'), 'LeNet (Tanh+MaxPool)')
record('LeNet Tanh+MaxPool (6,16)', lenet, lenet_hist, lenet_test)


# =====================================================================
# Task 4: 控制变量对比（每次只改一个因素，其余与 Task 3 基准相同）
# =====================================================================

# ---- 4A: 激活函数 Tanh vs ReLU（MaxPool, channels (6,16)）----
print('\n' + '=' * 70)
print('Task 4A: Activation — Tanh vs ReLU (MaxPool, channels (6,16))')
print('=' * 70)
print('Tanh 已在 Task 3 训练，这里补训 ReLU 版本。')
set_seed()
lenet_relu = LeNetStyleCNN(channels=(6, 16), activation='relu', pool='max')
print(f'LeNet (ReLU) trainable parameters: {count_parameters(lenet_relu)}')
relu_hist = train_model(lenet_relu, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3)
relu_test = evaluate_accuracy(lenet_relu, cnn_test_loader)
print(f'LeNet (ReLU) test accuracy: {relu_test:.4f}')
record('LeNet ReLU+MaxPool (6,16)', lenet_relu, relu_hist, relu_test)
plot_overlay([lenet_hist, relu_hist], ['Tanh', 'ReLU'],
             os.path.join(RESULTS_DIR, 'task4a_activation_curves.png'),
             'Activation: Tanh vs ReLU')

# ---- 4B: 池化类型 MaxPool vs AvgPool（Tanh, channels (6,16)）----
print('\n' + '=' * 70)
print('Task 4B: Pooling — MaxPool vs AvgPool (Tanh, channels (6,16))')
print('=' * 70)
print('MaxPool 已在 Task 3 训练，这里补训 AvgPool 版本。')
set_seed()
lenet_avg = LeNetStyleCNN(channels=(6, 16), activation='tanh', pool='avg')
print(f'LeNet (AvgPool) trainable parameters: {count_parameters(lenet_avg)}')
avg_hist = train_model(lenet_avg, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3)
avg_test = evaluate_accuracy(lenet_avg, cnn_test_loader)
print(f'LeNet (AvgPool) test accuracy: {avg_test:.4f}')
record('LeNet Tanh+AvgPool (6,16)', lenet_avg, avg_hist, avg_test)
plot_overlay([lenet_hist, avg_hist], ['MaxPool', 'AvgPool'],
             os.path.join(RESULTS_DIR, 'task4b_pooling_curves.png'),
             'Pooling: Max vs Avg')

# ---- 4C: 通道数 (6,16) vs (16,32) vs (32,64)（Tanh+MaxPool）----
print('\n' + '=' * 70)
print('Task 4C: Channel width — (6,16) vs (16,32) vs (32,64) (Tanh+MaxPool)')
print('=' * 70)
chan_hists = [lenet_hist]
chan_labels = ['(6,16)']
for ch in [(16, 32), (32, 64)]:
    set_seed()
    m = LeNetStyleCNN(channels=ch, activation='tanh', pool='max')
    print(f'\n-- channels {ch}, trainable parameters: {count_parameters(m)} --')
    h = train_model(m, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3)
    t = evaluate_accuracy(m, cnn_test_loader)
    print(f'LeNet channels {ch} test accuracy: {t:.4f}')
    record(f'LeNet Tanh+MaxPool {ch}', m, h, t)
    chan_hists.append(h); chan_labels.append(str(ch))
plot_overlay(chan_hists, chan_labels,
             os.path.join(RESULTS_DIR, 'task4c_channels_curves.png'),
             'Channel width comparison')

# ---- 4D: 正则化 Dropout / WeightDecay / Augmentation（基准 Tanh+MaxPool (6,16)）----
print('\n' + '=' * 70)
print('Task 4D: Regularisation — Dropout / WeightDecay / Augmentation')
print('=' * 70)

# (a) Dropout 0.5，加在最终分类头之前
set_seed()
m_drop = LeNetStyleCNN(channels=(6, 16), activation='tanh', pool='max', dropout=0.5)
print(f'LeNet + Dropout0.5 trainable parameters: {count_parameters(m_drop)}')
drop_hist = train_model(m_drop, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3)
drop_test = evaluate_accuracy(m_drop, cnn_test_loader)
print(f'LeNet + Dropout0.5 test accuracy: {drop_test:.4f}')
record('LeNet + Dropout0.5', m_drop, drop_hist, drop_test)

# (b) Weight decay 1e-4（L2 正则，写在优化器里）
set_seed()
m_wd = LeNetStyleCNN(channels=(6, 16), activation='tanh', pool='max')
print(f'LeNet + WeightDecay1e-4 trainable parameters: {count_parameters(m_wd)}')
wd_hist = train_model(m_wd, cnn_train_loader, cnn_val_loader, epochs=15, lr=1e-3, weight_decay=1e-4)
wd_test = evaluate_accuracy(m_wd, cnn_test_loader)
print(f'LeNet + WeightDecay1e-4 test accuracy: {wd_test:.4f}')
record('LeNet + WeightDecay1e-4', m_wd, wd_hist, wd_test)

# (c) 数据增强 RandomAffine（小幅旋转+平移，MNIST 仍可辨认）
set_seed()
m_aug = LeNetStyleCNN(channels=(6, 16), activation='tanh', pool='max')
print(f'LeNet + RandomAffine aug trainable parameters: {count_parameters(m_aug)}')
aug_hist = train_model(m_aug, cnn_train_aug_loader, cnn_val_loader, epochs=15, lr=1e-3)
aug_test = evaluate_accuracy(m_aug, cnn_test_loader)
print(f'LeNet + RandomAffine aug test accuracy: {aug_test:.4f}')
record('LeNet + RandomAffine aug', m_aug, aug_hist, aug_test)

plot_overlay([lenet_hist, drop_hist, wd_hist, aug_hist],
             ['Baseline', '+Dropout0.5', '+WeightDecay1e-4', '+Augmentation'],
             os.path.join(RESULTS_DIR, 'task4d_regularization_curves.png'),
             'Regularisation comparison')


# =====================================================================
# Task 5: Error analysis on the best CNN
# =====================================================================
print('\n' + '=' * 70)
print('Task 5: Error analysis')
print('=' * 70)
cnn_results = [r for r in RESULTS if not r['name'].startswith('MLP')]
best = max(cnn_results, key=lambda r: r['test_acc'])
print(f'Best CNN: {best["name"]}  (test acc = {best["test_acc"]:.4f})')
best_model = best['model']

best_model.eval()
images, trues, preds, confs = [], [], [], []
with torch.no_grad():
    for data, target in cnn_test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        out = best_model(data)
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
        for i in range(data.size(0)):
            if pred[i].item() != target[i].item():
                images.append(data[i].cpu().squeeze().numpy())
                trues.append(target[i].item())
                preds.append(pred[i].item())
                confs.append(conf[i].item())

print(f'Total misclassified: {len(images)} / {len(cnn_test)}')
pairs = Counter((t, p) for t, p in zip(trues, preds))
print('Most confused (true -> pred) pairs:')
for (t, p), c in pairs.most_common(8):
    print(f'  {t} -> {p}: {c}')

# 画前 20 个错分样本
n_show = min(20, len(images))
plt.figure(figsize=(12, 8))
for k in range(n_show):
    plt.subplot(4, 5, k + 1)
    plt.imshow(images[k], cmap='gray')
    plt.title(f'true={trues[k]} pred={preds[k]}\nconf={confs[k]:.2f}', fontsize=8)
    plt.axis('off')
plt.suptitle(f'Misclassified samples — {best["name"]}')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'task5_misclassified.png'), dpi=120)
plt.close()


# =====================================================================
# 汇总对比表
# =====================================================================
print('\n' + '=' * 70)
print('Summary')
print('=' * 70)
header = f'{"Model":<32}{"Params":>10}{"TrainAcc":>10}{"ValAcc":>10}{"TestAcc":>10}{"s/epoch":>10}'
print(header)
print('-' * len(header))
for r in RESULTS:
    print(f'{r["name"]:<32}{r["params"]:>10}{r["train_acc"]:>10.4f}{r["val_acc"]:>10.4f}'
          f'{r["test_acc"]:>10.4f}{r["avg_epoch_time"]:>9.1f}s')

csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'params', 'train_acc', 'val_acc', 'test_acc', 'avg_epoch_time_s'])
    for r in RESULTS:
        w.writerow([r['name'], r['params'],
                    f'{r["train_acc"]:.4f}', f'{r["val_acc"]:.4f}',
                    f'{r["test_acc"]:.4f}', f'{r["avg_epoch_time"]:.1f}'])

print(f'\nSaved figures and summary to: {RESULTS_DIR}')
print('Done.')
