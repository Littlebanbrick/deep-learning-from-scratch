"""
minimal-rnn / sine_rnn.py
=========================
最小 RNN 演示：用 nn.RNN 预测正弦波的下一个点。

任务（many-to-one 回归）
-----------------------
- 输入：正弦波上连续的 T=20 个采样点，形状 (batch, T, 1)
- 输出：第 T+1 个点的值（标量）

为什么选正弦波
--------------
零数据依赖、CPU 秒级训练、可视化直观，能清楚展示 RNN「用历史预测未来」
的记忆作用。这是理解 PyTorch RNN API 的最小载体。

nn.RNN 内部到底在做什么（一句话）
----------------------------------
对每个时间步 t：
    h_t = tanh( W_ih · x_t  +  W_hh · h_{t-1}  +  b )
nn.RNN 就是把这个递推式在整条序列上向量化跑一遍。我们只取最后一个时间步
的 h_T，接一个 Linear 头回归出预测值。

跑法
----
    python sine_rnn.py

产出（保存在 minimal-rnn/results/）
------------------------------------
- loss_curve.png        训练/验证 MSE 曲线
- prediction.png        单步预测：真实 vs 预测
- autoregressive.png    多步自回归预测：给一段种子，RNN 自由滚动预测
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # 无界面环境也能存图

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 0. 配置
# ============================================================
SEED = 42
SEQ_LEN = 20          # 用过去 20 个点预测第 21 个
HIDDEN_SIZE = 32      # RNN 隐藏维度
NUM_LAYERS = 1        # 堆叠几层 RNN（1 层最简单）
BATCH_SIZE = 16
EPOCHS = 150
LR = 1e-3

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. 数据：生成正弦波 + 滑动窗口
# ============================================================
def make_sine_data(n_points: int = 2000) -> np.ndarray:
    """生成带一点点噪声的正弦波，使任务非平凡（否则 RNN 会退化成「复制上一个点」）。"""
    t = np.linspace(0, 80, n_points)
    y = np.sin(t) + 0.05 * np.random.randn(n_points)
    return y.astype(np.float32)


def build_windows(y: np.ndarray, seq_len: int):
    """滑动窗口：X[i] = y[i : i+seq_len]，target[i] = y[i+seq_len]。

    返回 X 形状 (N, seq_len, 1)，target 形状 (N, 1)。
    最后那个特征维 1 是给 nn.RNN 的 input_size。
    """
    n = len(y) - seq_len
    X = np.stack([y[i:i + seq_len] for i in range(n)])        # (N, seq_len)
    target = y[seq_len:]                                       # (N,)
    # 加上特征维 -> (N, seq_len, 1) 与 (N, 1)
    X = X[:, :, None]
    target = target[:, None]
    return X, target


def prepare_data():
    y = make_sine_data()
    X, target = build_windows(y, SEQ_LEN)

    # 前 70% 训练，后 30% 验证；窗口不跨段，避免泄漏。
    split = int(len(X) * 0.7)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = target[:split], target[split:]

    # 用 PyTorch 标准数据管道：TensorDataset + DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, y


# ============================================================
# 2. 模型：nn.RNN + Linear 回归头
# ============================================================
class SineRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        # batch_first=True 表示输入形状是 (batch, seq, feature)，更直观。
        # nonlinearity 默认 'tanh'，对应上面注释里的递推式。
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        # 取最后一个时间步的隐藏状态 h_T，映射成标量预测。
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # h0: (num_layers, batch, hidden_size)，不传时默认全零。
        out, h_n = self.rnn(x)
        # out 形状 (batch, seq_len, hidden_size)：每个时间步的隐藏状态
        # h_n 形状 (num_layers, batch, hidden_size)：最后一个时间步的隐藏状态
        # many-to-one：只要最后一步。
        last = out[:, -1, :]          # (batch, hidden_size)
        pred = self.head(last)        # (batch, 1)
        return pred


# ============================================================
# 3. 训练循环
# ============================================================
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)

        # --- val ---
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                total += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = total / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")
    return history


# ============================================================
# 4. 评估与可视化
# ============================================================
def plot_loss(history, path):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Loss curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_prediction(model, y, path, seq_len=SEQ_LEN):
    """对整条波形的每个窗口做单步预测，画出真实 vs 预测。"""
    model.eval()
    X, target = build_windows(y, seq_len)
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy().ravel()
    true = target.ravel()

    plt.figure(figsize=(11, 4))
    plt.plot(true, label="true", alpha=0.8)
    plt.plot(pred, label="predicted (1-step)", alpha=0.8)
    # 画一条竖线分隔训练/验证段
    split = int(len(X) * 0.7)
    plt.axvline(split, color="gray", linestyle="--", alpha=0.6, label="train | val")
    plt.xlabel("window index")
    plt.ylabel("value")
    plt.title("1-step prediction over the whole wave")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_autoregressive(model, y, path, seed_len=SEQ_LEN, future=200):
    """多步自回归预测：给前 seed_len 个真实点作为种子，
    之后每步用模型自己的预测当输入继续滚，看 RNN 能「记住」多久。

    注意：这里每个预测步只喂「最近 seq_len 个点」做一次 many-to-one 预测，
    模型权重不变；误差完全来自 RNN 对自身预测的反馈累积。
    """
    model.eval()
    # 从波形中后段（验证区）取一段作为种子，确保模型没见过这段的未来。
    start = int(len(y) * 0.75)
    window = list(y[start:start + seed_len])
    truth = y[start + seed_len:start + seed_len + future]

    preds = []
    with torch.no_grad():
        for _ in range(future):
            x = torch.tensor(window[-seed_len:], dtype=torch.float32)[None, :, None]
            p = model(x).item()
            preds.append(p)
            window.append(p)  # 把预测塞回窗口尾部，继续滚

    x_axis = np.arange(future)
    plt.figure(figsize=(11, 4))
    plt.plot(x_axis, truth, label="true future", alpha=0.8)
    plt.plot(x_axis, preds, label="autoregressive prediction", alpha=0.8)
    plt.xlabel("steps ahead")
    plt.ylabel("value")
    plt.title(f"Autoregressive {future}-step prediction from a {seed_len}-point seed")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ============================================================
# 5. main
# ============================================================
def main():
    print("=" * 64)
    print("minimal-rnn: predict the next point of a sine wave with nn.RNN")
    print("=" * 64)

    train_loader, val_loader, y = prepare_data()

    # 打印一次形状，帮助理解数据流
    xb, yb = next(iter(train_loader))
    print(f"\n[shape] batch x: {tuple(xb.shape)}  (batch, seq_len, input_size)")
    print(f"[shape] batch y: {tuple(yb.shape)}  (batch, 1)")

    model = SineRNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:\n{model}")
    print(f"Trainable params: {n_params}")

    # 一次 forward sanity check：确认输出形状正确
    with torch.no_grad():
        out = model(xb)
    print(f"[shape] model output: {tuple(out.shape)}  (should be (batch, 1))")

    print(f"\nTraining {EPOCHS} epochs on CPU ...")
    history = train(model, train_loader, val_loader)

    # 保存模型权重
    torch.save(model.state_dict(), HERE / "sine_rnn.pt")

    plot_loss(history, RESULTS / "loss_curve.png")
    plot_prediction(model, y, RESULTS / "prediction.png")
    plot_autoregressive(model, y, RESULTS / "autoregressive.png")

    print("\nSaved figures:")
    for name in ["loss_curve.png", "prediction.png", "autoregressive.png"]:
        print(f"  {RESULTS / name}")
    print("\nDone. 看 autoregressive.png 观察 RNN 预测如何随步数发散——那是它「记忆」的边界。")


if __name__ == "__main__":
    main()
