// English Report Template — Lab 3

// ===== Code Block Style =====
#show raw.where(block: true): set block(
  fill: luma(250),
  inset: 6pt,
  radius: 3pt,
)
#show raw.where(block: true): set text(
  size: 9pt,
  font: ("Liberation Mono", "Noto Serif CJK SC"),
)
#show raw.where(block: true): set par(leading: 1.15em)
#show raw.where(block: false): set text(
  font: ("Liberation Mono", "Noto Serif CJK SC"),
)

// ===== Page =====
#set page(
  paper: "a4",
  margin: (left: 2.6cm, right: 2.6cm, top: 2.4cm, bottom: 2.8cm),
  numbering: "1",
  number-align: bottom + center,
  header: context [
    #text(size: 12pt, fill: gray.darken(25%))[deep-learning-from-scratch]
    #h(1fr)
    #text(size: 12pt, fill: gray.darken(25%))[#datetime.today().display("[month repr:short] [day], [year]")]
  ],
  footer: context align(center)[
    #text(size: 11pt, fill: gray.darken(50%))[#counter(page).display()]
  ],
)

// ===== Typography =====
#set text(
  font: ("Liberation Serif", "Noto Serif CJK SC"),
  size: 14pt,
  lang: "en",
)
#show figure.caption: set text(
  size: 11pt,
  font: ("Liberation Serif", "Noto Serif CJK SC"),
  style: "italic",
  fill: luma(90),
)
#show figure.caption: set par(leading: 1.0em)

#set par(
  justify: true,
  first-line-indent: 0em,
  leading: 0.75em,
  spacing: 1em,
)

// ===== Heading hierarchy =====
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [
  #v(0.9em)
  #text(size: 20pt, weight: "bold", it.body)
  #v(0.35em)
]
#show heading.where(level: 2): it => [
  #v(0.55em)
  #text(size: 15pt, weight: "semibold", it.body)
  #v(0.25em)
]
#show heading.where(level: 3): it => [
  #v(0.35em)
  #text(size: 13pt, weight: "semibold", it.body)
  #v(0.15em)
]

// ===== Cover =====
#align(center + horizon)[
  #v(0%)
  #text(size: 35pt, weight: "bold")[
    Lab 3
    \
  ]
  #v(0%)
  #text(size: 25pt, weight: "bold")[
    Convolutional Neural Networks
    \
    for MNIST Handwritten Digit Recognition
  ]
  #image("icon_ZJU.png", width: 40%)
  #v(2em)
  #text(size: 20pt)[Author: Chuanyu Wang]
  #v(0.5em)
  #text(size: 20pt)[Time: 2026-7-7]
  #v(0.5em)
]

#pagebreak()

// ===== Abstract =====
#block[
  #text(size: 15pt, weight: "semibold")[Abstract]
  #v(0.4em)
  本实验在 Lab 1 的 MLP 基础上，引入卷积神经网络（CNN）重新处理 MNIST 手写数字识别任务。我们从复现 Lab 1 最佳 MLP 基准出发，依次构建单卷积块的最小 CNN、LeNet-5 风格 CNN，并以 LeNet 为基础在四个维度上展开受控对比实验：激活函数（Tanh vs ReLU）、池化类型（MaxPool vs AvgPool）、通道宽度（$(6,16)$ / $(16,32)$ / $(32,64)$）以及正则化手段（Dropout / Weight Decay / 数据增强）。所有实验在相同的数据划分、归一化、batch size 与优化器下进行，每个实验前重置随机种子以保证各变体从相同初始权重出发。实验结果表明：仅含 44k 参数的 LeNet 即可达到 98.84% 的测试准确率，超越 118k 参数的 MLP 基准（98.06%）；其中通道扩展至 $(16,32)$ 的变体取得最佳表现 98.89%。本报告记录了 tensor shape 流转、训练曲线、参数对比与错分分析，并讨论了过拟合、收益递减与正则化等关键现象。
]

#pagebreak()

// ============================================================
// 正文
// ============================================================

= 一、硬件配置与软件环境

== 硬件配置

- CPU: Intel(R) Core(TM) Ultra 9 285H (16 logical cores)
- 内存: 32 GB
- 无独立 GPU，全部实验在 CPU 上完成

== 软件环境

- 操作系统: Ubuntu 26.04 LTS
- Python 版本: 3.10.20
- 主要依赖库及版本:

```text
PyTorch version: 2.12.1+cu130
TorchVision version: 0.27.1+cu130
NumPy version: 2.2.6
Matplotlib version: 3.10.x
```

== 实验约定

- 随机种子固定为 42（`torch.manual_seed(42)` 与 `np.random.seed(42)`），并在每个实验前重置，保证各 Task 4 变体从相同初始权重出发，实现“只改一个变量”的受控对比。
- 数据归一化统一使用 `transforms.ToTensor()`（缩放到 $[0, 1]$），*不*叠加 MNIST 均值/方差归一化。理由是与 Lab 1 的 MLP 基准保持一致，使 CNN 与 MLP 的对比只受模型结构影响，不受预处理差异干扰（Lab 要求明确写出此选择）。
- 训练集 60 000 张按 90/10 划分为 54 000/6 000（train/val），用固定种子的 `generator` 划分，保证 MLP 与 CNN 使用完全相同的索引。
- 全部 batch_size = 64，优化器为 Adam，`lr = 1e-3`，损失函数为 `CrossEntropyLoss`（输入为原始 logits）。

#pagebreak()

= 二、Task 1: 复现 MLP 基准

直接复用 Lab 1 的最佳配置作为后续 CNN 要超越的基准。模型结构为 `Flatten -> Linear(784, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 10)`，训练 20 个 epoch。

#figure(
  image("results/task1_mlp_curves.png", width: 100%),
  caption: [Task 1 MLP 基准的训练/验证损失与准确率曲线。训练损失一路平滑下降至 0.035，验证损失在第 7 轮后停住并在 0.075–0.092 之间来回震荡。],
)

训练损失从 0.3977 平滑降至 0.0350，验证损失在前 7 轮快速下降后停止改善，进入震荡期。验证准确率在第 14 轮达到峰值 0.9810 后略回落。最终测试准确率为 *0.9806*，参数量 118 282。

这里的“锯齿”在后续所有实验中都会反复出现，值得先解释清楚。验证损失后期在低点附近震荡、而训练损失继续平滑下降，本质是三类因素的叠加：

+ *mini-batch 梯度噪声*：每个 epoch 的权重更新来自约 844 个 batch 的随机梯度，方向本身带噪声；Adam 全程使用固定 $lr = 10^(-3)$ 不衰减，后期权重已接近最优解，固定步长反复“跨过”最优点，loss 因此在最低点附近来回震荡。
+ *验证集小 + loss 是连续值*：val 集仅 6 000 张，单张高置信度的错分会贡献一个极大的 $-log(epsilon)$，使 Val Loss 对个别样本特别敏感；而 Val Acc 是 0/1 计数，相对稳定——这就是为什么每张图右侧 Acc 曲线总比左侧 Loss 曲线平滑。
+ *过拟合信号*：最根本地，蓝线（train）继续降、红线（val）停住甚至上跳，正是 train/val gap 拉开的过程。震荡不是 bug，而是过拟合的可视化。

#pagebreak()

= 三、Task 2: 最小 CNN 与 Tensor Shape 流转

Task 2 的核心目标不是准确率，而是验证对卷积、池化、展平如何改变 tensor shape 的理解。模型为单卷积块：

```python
Conv2d(1, 16, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
    -> Flatten -> Linear(3136, 128) -> ReLU -> Linear(128, 10)
```

`padding=1` 使 $3 times 3$ 卷积后高宽不变（仍 $28 times 28$），`MaxPool2d(2)` 后变 $14 times 14$，展平维度为 $16 times 14 times 14 = 3136$。shape 流转打印如下：

```text
Input:          (4, 1, 28, 28)
After Conv2d:   (4, 16, 28, 28)
After ReLU:     (4, 16, 28, 28)
After MaxPool:  (4, 16, 14, 14)
After Flatten:  (4, 3136)
Output logits:  (4, 10)
```

#figure(
  image("results/task2_minimal_cnn_curves.png", width: 100%),
  caption: [Task 2 MinimalCNN 的训练/验证损失与准确率曲线。],
)

测试准确率 0.9858，已超过 MLP 基准。值得注意的是该模型参数量高达 *402 986*——比 LeNet 还多——原因是 16 通道在 $14 times 14$ 上展平维度（3136）很大，导致第一个 `Linear` 层参数 $3136 times 128 approx 400k$。这本身就是一个讨论点：卷积层参数虽少，但若不在卷积阶段充分降采样就直接展平，全连接层会变得极其臃肿。Task 3 的 LeNet 通过两次池化把空间尺寸压到 $4 times 4$，避免了这一问题。

#pagebreak()

= 四、Task 3: LeNet-5 风格 CNN

实现一个 LeNet-5 风格的两层卷积网络作为 Task 4 各对比实验的基准：

```python
Conv2d(1, 6, 5) -> Tanh -> MaxPool(2)
    -> Conv2d(6, 16, 5) -> Tanh -> MaxPool(2)
    -> Flatten -> Linear(256, 120) -> Tanh
    -> Linear(120, 84) -> Tanh -> Linear(84, 10)
```

MNIST 输入是 $28 times 28$（非 LeNet 原文的 $32 times 32$），$5 times 5$ 卷积无 padding 时形状流为 $28 -> 24 -> 12 -> 8 -> 4$，因此第一层 FC 输入为 $16 times 4 times 4 = 256$。完整 shape 流转：

```text
Input:         (4, 1, 28, 28)
After Conv1:   (4, 6, 24, 24)
After Pool1:   (4, 6, 12, 12)
After Conv2:   (4, 16, 8, 8)
After Pool2:   (4, 16, 4, 4)
After Flatten: (4, 256)
After FC1:     (4, 120)
After FC2:     (4, 84)
Output logits: (4, 10)
```

#figure(
  image("results/task3_lenet_curves.png", width: 100%),
  caption: [Task 3 LeNet (Tanh + MaxPool, channels (6,16)) 的训练/验证曲线。训练损失降至 0.0062，验证损失在 0.05 附近震荡，gap 明显小于 MinimalCNN。],
)

测试准确率 *0.9884*，参数量仅 *44 426*——约为 MLP 的 1/2.7、MinimalCNN 的 1/9，却取得更高准确率。这直接体现了卷积的两个核心归纳偏置：*局部感受野*与*权值共享*。MLP 把 $28 times 28$ 展平为 784 维向量，破坏了像素间的二维邻接关系，每个隐藏神经元需要独立学习一个覆盖全图的模式；而卷积核在整张图上共享同一组权重，每个神经元只看 $5 times 5$ 的局部窗口，参数量从 $O(text("width") times text("height"))$ 降到 $O(text("kernel")^2)$。

#pagebreak()

= 五、Task 4: CNN 设计选择对比

以 Task 3 的 LeNet (Tanh + MaxPool, channels $(6,16)$) 为基准，每次只改动一个因素，其余完全相同。所有变体在训练前都重置 `set_seed(42)`，保证从同一组初始权重出发。对比只画验证曲线（`plot_overlay`），因为横向对比关注的是泛化表现。

== Task 4A: 激活函数 — Tanh vs ReLU

```text
Tanh  test acc = 0.9884   (末5轮 val acc: 0.9860 0.9860 0.9867 0.9853 0.9877)
ReLU  test acc = 0.9888   (末5轮 val acc: 0.9875 0.9903 0.9883 0.9902 0.9892)
```

#figure(
  image("results/task4a_activation_curves.png", width: 100%),
  caption: [Task 4A 激活函数对比：Tanh vs ReLU 的验证损失与验证准确率。ReLU 后期 val acc 更高更稳。],
)

两者准确率接近，ReLU 略胜且后期 val acc 更稳定。这与现代 CNN 偏好 ReLU 的原因一致：ReLU 在正区间导数恒为 1，*非饱和*，缓解了深层网络的梯度消失；同时 $max(0, z)$ 产生稀疏激活（部分神经元输出恰为 0），降低神经元间的共适应；此外计算只需一次比较，远快于 Tanh 的指数运算。经典 LeNet 使用 Tanh 是 1998 年的时代选择（ReLU 当时尚未普及），而现代网络普遍采用 ReLU 及其变体（Leaky ReLU、ELU 等）。

== Task 4B: 池化类型 — MaxPool vs AvgPool

```text
MaxPool  test acc = 0.9884
AvgPool  test acc = 0.9861
```

#figure(
  image("results/task4b_pooling_curves.png", width: 100%),
  caption: [Task 4B 池化对比：MaxPool vs AvgPool 的验证曲线。],
)

MaxPool 略好。直觉上，MaxPool 保留局部窗口内*最强响应*，对笔画这种“稀疏尖锐”的特征尤为契合——一条笔画经过的像素激活值高，MaxPool 能稳定地把它保留下来，对小的平移/形变也鲁棒（只要最强响应仍在窗口内，输出就不变）。AvgPool 做平均平滑，会把尖锐的笔画响应与背景的低响应平均掉，相当于在特征图上做了低通滤波，可能模糊掉区分性强的边缘信息。

*关于池化丢弃了什么*：池化层丢弃了*位置信息*——它只保留窗口内的最大值或均值，不再知道这个最大值出现在窗口的哪个位置。这是用空间精度换取平移不变性的有意取舍，也是池化作为“局部平移鲁棒性”机制的代价。

#pagebreak()

== Task 4C: 通道宽度 — $(6,16)$ / $(16,32)$ / $(32,64)$

#figure(
  image("results/task4c_channels_curves.png", width: 100%),
  caption: [Task 4C 通道宽度对比：三组配置的验证曲线几乎重叠。],
)

三组配置的准确率与参数量如下表。准确率几乎不随通道数变化，$(32,64)$ 甚至略低于 $(16,32)$，而每 epoch 耗时几乎线性增长。

#figure(
  table(
    columns: 4,
    stroke: 0.3pt,
    align: (center, center, center, center),
    [通道配置], [参数量], [测试准确率], [s/epoch],
    [$(6,16)$],  [44 426],  [0.9884], [9.2],
    [$(16,32)$], [85 822],  [#text(fill: red)[0.9889]], [13.5],
    [$(32,64)$], [186 110], [0.9885], [25.5],
  ),
  caption: [Task 4C 通道宽度对比。$(16,32)$ 取得最佳准确率，$(32,64)$ 参数翻倍反而略降，且每 epoch 慢近 3 倍。],
)

这是*收益递减*（diminishing returns）的典型例子。MNIST 太简单，baseline 已逼近 $approx 99%$ 的天花板，进一步加宽通道只换来更多的过拟合风险与计算量，没有泛化收益。$(6,16)$ $->$ $(16,32)$ 参数翻倍准确率仅 $+0.0005$；$(16,32)$ $->$ $(32,64)$ 再翻倍准确率反而 $-0.0004$。在更复杂的任务（CIFAR、ImageNet）上加宽的收益更明显，但本实验说明：*加宽通道并非总是值得*，需结合任务难度与过拟合信号权衡。

== Task 4D: 正则化 — Dropout / Weight Decay / 数据增强

以基准 LeNet (Tanh + MaxPool, $(6,16)$) 为对照，分别添加三种正则化手段：

#figure(
  image("results/task4d_regularization_curves.png", width: 100%),
  caption: [Task 4D 正则化对比：Baseline / +Dropout0.5 / +WeightDecay1e-4 / +Augmentation 的验证曲线。],
)

#figure(
  table(
    columns: 4,
    stroke: 0.3pt,
    align: (left, center, center, center),
    [配置], [Train Acc], [Val Acc], [Test Acc],
    [Baseline],         [0.9980], [0.9877], [0.9884],
    [+Dropout0.5],      [0.9954], [0.9855], [0.9878],
    [+WeightDecay1e-4], [0.9966], [0.9847], [0.9853],
    [+Augmentation],    [0.9823], [#text(fill: red)[0.9898]], [0.9876],
  ),
  caption: [Task 4D 正则化对比。数据增强显著压低了 train acc（0.9823）却取得最高 val acc（0.9898）；三种手段对 test acc 几乎无提升。],
)

三点观察：

+ *正则化主要压低 train acc，缩小 train/val gap*，而非提升 test acc。Dropout 把 train acc 从 0.9980 压到 0.9954，数据增强压到 0.9823，但 test acc 都在 0.987–0.988 之间，与 baseline 几乎无差。这是因为 baseline 已逼近 MNIST 的 $approx 99%$ 天花板，过拟合尚未严重到损害 test 性能。
+ *数据增强的 val loss 低于 train loss，是正常现象而非错误*。注意增强实验中 train loss 是在「旋转/平移过的、更难的」增强数据上计算的，而 val loss 是在「原始干净的」数据上计算的，两者不可直接比大小。读图时若误以为“val 比 train 好”是模型异常，就会得出错误结论。
+ *数据增强有效抑制了过拟合*：它的 val acc 后期最高（0.9898），且 train/val gap 从 baseline 的 $approx 0.010$ 缩到 $approx 0.008$。只是 15 epoch 内模型尚未完全从更难的增强数据中“学完”，test acc 还没追上 baseline。若延长训练，增强版本大概率会后来居上——这回答了 Lab 的问题：“正则化提升准确率，还是只是放缓训练？”——在本实验中主要是*放缓过拟合、缩小 gap*，对 test acc 的提升有限。

#pagebreak()

= 六、Task 5: 错分分析

在最佳 CNN（LeNet Tanh+MaxPool, $(16,32)$, test acc 0.9889）上分析误分类样本。共 111/10 000 张测试图被错分。

#figure(
  image("results/task5_misclassified.png", width: 100%),
  caption: [Task 5 最佳 CNN 的前 20 个错分样本。每张图标注真值、预测值及模型对预测类的置信度。],
)

最常混淆的 (true -> pred) 对如下：

```text
9 -> 4: 10
5 -> 3: 9
2 -> 7: 6
6 -> 0: 6
8 -> 7: 5
2 -> 1: 4
8 -> 9: 4
8 -> 3: 4
```

观察：

- *9↔4 与 5→3 是高频混淆对*。9 与 4 在手写体中常共享顶部闭合环结构，5 与 3 的轮廓极为相似——这些是*视觉上确实模糊*的对，并非模型无理取闹。
- *8 容易被误判为 7/9/3*：8 由两个环组成，若书写时某个环不闭合，拓扑结构就接近 9 或 3。
- 置信度普遍不高（多在 0.5–0.8），说明模型对这些样本本就不确定——这与样本本身的视觉模糊性一致。少数高置信度错分（如置信度 >0.9 的）通常是书写极端潦草的样本。
- 相比 MLP 基准，CNN 的错误更多集中在“形状相近的数字对”上，而非全图结构误判，说明 CNN 学到了更贴近笔画拓扑的特征。

#pagebreak()

= 七、汇总对比

#figure(
  table(
    columns: 6,
    stroke: 0.3pt,
    align: (left, center, center, center, center, center),
    [模型], [参数量], [Train Acc], [Val Acc], [Test Acc], [s/epoch],
    [MLP $[128,128]$+ReLU+Dropout0.2], [118 282], [0.9885], [0.9783], [0.9806], [8.0],
    [MinimalCNN (1 conv block)],        [402 986], [0.9980], [0.9848], [0.9858], [13.0],
    [LeNet Tanh+MaxPool $(6,16)$],     [44 426],  [0.9980], [0.9877], [0.9884], [9.2],
    [LeNet ReLU+MaxPool $(6,16)$],     [44 426],  [0.9961], [0.9892], [0.9888], [8.4],
    [LeNet Tanh+AvgPool $(6,16)$],    [44 426],  [0.9971], [0.9843], [0.9861], [9.0],
    [LeNet Tanh+MaxPool $(16,32)$],    [85 822],  [0.9976], [0.9878], [#text(fill: red)[0.9889]], [13.5],
    [LeNet Tanh+MaxPool $(32,64)$],    [186 110], [0.9978], [0.9868], [0.9885], [25.5],
    [LeNet + Dropout0.5],               [44 426],  [0.9954], [0.9855], [0.9878], [9.4],
    [LeNet + WeightDecay1e-4],         [44 426],  [0.9966], [0.9847], [0.9853], [10.6],
    [LeNet + RandomAffine aug],        [44 426],  [0.9823], [0.9898], [0.9876], [21.8],
  ),
  caption: [Lab 3 全部 10 组实验汇总。最佳 CNN 为 LeNet Tanh+MaxPool $(16,32)$，test acc 0.9889。],
)

#pagebreak()

= 八、报告思考题

== Q1: 为什么在第一层之前展平图像会丢失空间结构？

MNIST 图像本是一个 $28 times 28$ 的二维像素阵列，相邻像素共同构成笔画的局部几何（边缘、端点、转角）。`Flatten` 将其拉成 784 维的一维向量后，二维邻接关系被彻底破坏：像素 $(i, j)$ 与 $(i, j+1)$ 在原图中相邻，但在展平后的向量中相隔 1 个位置，而 $(i, j)$ 与 $(i+1, j)$ 却相隔 28 个位置。MLP 的第一个全连接层中，每个隐藏神经元对所有 784 个像素各学一个独立权重，必须从零开始重新发现“哪些像素在空间上相邻、共同构成笔画”这一先验。而卷积层天然以二维块为单位读取输入，邻接关系被结构性地保留在核的滑动窗口里。

== Q2: 局部感受野与权值共享如何减少参数量？

*局部感受野*指每个卷积核只看 $k times k$ 的窗口而非全图，单层参数从 $text("width") times text("height") times c_"out"$ 降到 $k^2 times c_"in" times c_"out"$。以 Task 3 为例，第一个 `Conv2d(1, 6, 5)` 仅 $1 times 6 times 5 times 5 + 6 = 156$ 个参数，而等价的 MLP 隐藏层若看全图需要 $784 times 6 + 6 approx 4710$ 个。

*权值共享*指同一个卷积核在整张图的所有空间位置上复用同一组权重——同一个 $5 times 5$ 模板从左上滑到右下，只学一份参数。这使得参数量与输入空间尺寸*无关*，只与核大小和通道数有关。LeNet 总共 44k 参数就达到了 MLP 118k 都达不到的准确率，正是这两个归纳偏置的功劳。

== Q3: 池化的作用是什么？它可能丢弃了哪些信息？

池化在空间维度上下采样，作用有三：(1) *降低计算量*与后续层参数量（空间尺寸减半，展平维度变 $1/4$）；(2) 提供*局部平移鲁棒性*——只要最强响应仍在窗口内，输出不变，对小位移不敏感；(3) 扩大后续层的*有效感受野*。

它丢弃的主要是*精确位置信息*：MaxPool 只记窗口内最大值，不知道它出现在哪个位置；AvgPool 把所有位置平均掉，更彻底地丢失分布形状。此外池化不可逆，下采样后的低分辨率特征无法恢复原始细节。这也是近年来一些架构（如 ResNet 后段、全卷积网络）用 `stride>1` 卷积替代池化的原因之一。

== Q4: CNN 是否超过了 Lab 1 的 MLP 基准？哪个设计选择贡献最大？

是。MLP 基准 test acc 0.9806，所有 CNN 变体（除 MinimalCNN 外的 LeNet 系列均 $>= 0.9861$）都超过了它，最佳 LeNet $(16,32)$ 达 0.9889，提升 $+0.0083$。

贡献最大的设计选择是*卷积本身*——从 MLP 换到最朴素的 LeNet $(6,16)$ 就已从 0.9806 跳到 0.9884（$+0.0078$），几乎吃掉了全部提升。Task 4 的各变体（激活、池化、通道、正则化）之间准确率差异都在 $0.003$ 以内，属于微调。这说明对 MNIST 这种简单任务，“用不用卷积”是量级差异，而“卷积内部怎么配”是细枝末节。

== Q5: 比较 MLP 与 CNN 的参数量。更准确的模型一定参数更多吗？

不一定，本实验是反例。MLP 118k 参数 test acc 0.9806，LeNet 44k 参数 test acc 0.9884——CNN 参数少了 2.7 倍，准确率反而更高。MinimalCNN 有 403k 参数却只有 0.9858，反而不如 44k 的 LeNet。可见参数量与准确率并非单调相关：模型的*归纳偏置*是否匹配数据结构更重要。CNN 的卷积+池化把先验知识（局部性、平移不变性）写进了架构里，用更少参数表达了更适合图像的模式；而 MinimalCNN 虽然也用了卷积，但没在卷积阶段充分降采样就把 $14 times 14 times 16 = 3136$ 维直接喂给全连接层，导致参数堆在 FC 层，效率低下。

== Q6: 哪些误分类数字对模型最难？这些错误合理吗？

最难的是 *9↔4*（10 次）和 *5→3*（9 次），其次是 *2→7、6→0、8→7/9/3*。这些错误*合理*：9 与 4 在手写体中常共享顶部闭合环；5 与 3 轮廓几乎一致，仅一个笔画的开合不同；8 由两个环组成，某个环不闭合时拓扑上就退化成 9 或 3。从错分样本图看，置信度普遍不高（0.5–0.8），说明模型对这些样本本就犹豫，与人类的视觉模糊判断一致。少量高置信度错分对应书写极端潦草、连人都难辨的样本。

== Q7: 本实验如何与 LeNet-5 论文相关联？

LeNet-5（LeCun et al., 1998, *Gradient-Based Learning Applied to Document Recognition*）是首个成功应用于手写数字识别的卷积神经网络，奠定了现代 CNN 的基本范式：交替的卷积层与池化层提取层次化特征，最后接全连接层分类。本实验的 Task 3 直接复现了它的核心结构——`Conv-Pool-Conv-Pool-FC-FC-FC`、Tanh 激活、$(6,16)$ 通道配置——只是把输入从原论文的 $32 times 32$ 改成了 MNIST 的 $28 times 28$，并相应调整了 FC 输入维度。

Task 4 的对比则重现了 LeNet 之后社区对这一架构的逐项反思：用 ReLU 替代 Tanh（4A，现代 CNN 的标准做法）、用 MaxPool 替代原论文的 AvgPool（4B）、加宽通道（4C，对应 VGG/ResNet 的容量探索）、加入现代正则化（4D）。从这个角度看，本实验是一次“压缩版的 CNN 十年演进史”——从 1998 年的 LeNet 出发，逐步替换为现代组件，观察每一步的边际收益。

#pagebreak()

= 九、总结

本实验从复现 Lab 1 的 MLP 基准出发，逐步构建最小 CNN、LeNet-5 风格 CNN，并在四个维度上展开受控对比，最终取得 98.89% 的测试准确率。贯穿实验的核心结论：

- *卷积优于全连接*：仅 44k 参数的 LeNet 超越 118k 参数的 MLP，归功于局部感受野与权值共享这两个归纳偏置。
- *架构选择 > 细节调参*：从 MLP 到 LeNet 是量级跃升（$+0.008$），而激活、池化、通道、正则化之间的差异都在 $0.003$ 以内。
- *收益递减*：MNIST 上加宽通道到 $(32,64)$ 参数翻 4 倍、慢 3 倍却无收益，提醒实践中需结合任务难度权衡容量。
- *正则化主要抑制过拟合而非提升准确率*：在已逼近天花板的简单任务上，Dropout / Weight Decay / 数据增强缩小 train/val gap，但对 test acc 提升有限。
- *错分合理*：模型的高频混淆对（9↔4、5↔3）正是视觉上确实模糊的数字对，错误与人类直觉一致。

从 Lab 1 到 Lab 3，从 784 维向量的盲目连接到结构化的卷积核滑动，我们看到了“先验知识如何写入架构”带来的质变——这正是从 MLP 走向现代深度学习的关键一步。
