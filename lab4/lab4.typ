// English Report Template — Lab 4

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
    Lab 4
    \
  ]
  #v(0%)
  #text(size: 25pt, weight: "bold")[
    Transfer Learning with 
    \
    Pretrained ResNet18
    on Oxford-IIIT Pet
  ]
  #image("icon_ZJU.png", width: 40%)
  #v(2em)
  #text(size: 20pt)[Author: Chuanyu Wang]
  #v(0.5em)
  #text(size: 20pt)[Time: 2026-7-8]
  #v(0.5em)
]

#pagebreak()

// ===== Abstract =====
#block[
  #text(size: 15pt, weight: "semibold")[Abstract]
  #v(0.4em)
  本实验使用在 ImageNet（1.28M 图像、1000 类）上预训练的 ResNet18，通过迁移学习适配到 Oxford-IIIT Pet 数据集（37 类猫狗品种，约 3 300 张训练图）。我们实现并对比了三种迁移策略：从零训练（Task 2）、特征提取（Task 3，冻结 backbone）、微调（Task 4，差异化学习率解冻整个网络）。最终在 Task 5 通过数据规模消融实验定量验证了"预训练的收益随数据减少而增大"，在 Task 6 用 t-SNE 可视化解释了预训练 backbone 为何有效。实验结果：从零训练仅取得 0.361 测试准确率（11.2M 参数在小数据上严重过拟合）；冻结 backbone 仅训练 18 981 个头部参数即达 0.885；微调节点进一步将整个网络解冻以差异化学习率训练 10 个 epoch，取得最佳 0.897。数据规模消融显示 20% 数据下预训练带来 +0.72 的提升，100% 数据下缩至 +0.52——这一趋势是迁移学习作为标准实践的核心论据。
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
scikit-learn version: 1.7.x
```

== 实验约定

- 随机种子固定为 42 ，并在每个 Task 前重置，保证三种策略从相同数据划分与相同头部初始化出发。
- 数据集为 Oxford-IIIT Pet，37 类，共约 7 400 张图。`trainval` 划分按 90/10 分为 train / val（固定 `generator` 划分索引，保证三种策略一致），标准 `test` 划分留作最终评测。
- 输入统一 Resize 到 $224 times 224$ 以匹配 ResNet 输入。*归一化使用 ImageNet 均值/方差*（`mean = [0.485, 0.456, 0.406]`，`std = [0.229, 0.224, 0.225]`）。
- 训练时 batch_size = 64，优化器 Adam，损失 `CrossEntropyLoss`（输入为 raw logits）。
- 学习率：从零训练与冻结头部均 `lr = 1e-3`；微调 backbone 用 `1e-4`、head 用 `1e-3`（差异化学习率）。
- 全部训练在 CPU 上完成。三个主要训练 run（Task 2 scratch、Task 3 frozen、Task 4 fine-tune）合计约 60 分钟，其中 Task 2 最慢（每 epoch 约 122 秒）。

#pagebreak()

= 二、Task 1: 数据准备与 Sanity Check

通过 `torchvision.datasets.OxfordIIITPet(download=True)` 拉取数据集（约 800 MB）。构建 train / val / test 三个 DataLoader，并对训练集施加水平翻转增强。

#figure(
  image("results/task1_sample_grid.png", width: 110%),
  caption: [Task 1 训练集样本网格。每张图标注品种名，并经过 ImageNet 反归一化（unnormalize）以还原自然色彩用于人眼检查。],
)

== ImageNet 归一化的强制性

预训练 ResNet18 的第一层卷积与所有 BatchNorm 层的 running statistics 都是在 ImageNet 分布（`mean = [0.485, 0.456, 0.406]`）下学得的。若改用 `[0.5, 0.5, 0.5]` 或不归一化，第一层卷积看到的输入分布与预训练时截然不同，相当于把一个被精心调教过的特征提取器丢到了一个它从未见过的数据分布上——其输出立即失真，后续 BN 层的 running stats 也全部失配。这是迁移学习中最容易被忽略但代价最高的细节。本实验全程使用 ImageNet 统计量归一化，并在可视化时通过反归一化还原像素便于人眼检查（如上图）。

#figure(
  image("results/task1_class_distribution.png", width: 85%),
  caption: [37 类品种的样本量分布。各类约 200 张图，分布基本均衡，无需重采样。],
)

#pagebreak()

= 三、Task 2: 从零训练 ResNet18（Baseline）

实例化一个*无*预训练权重的 ResNet18，仅替换最后的 `fc` 层为 `Linear(512, 37)`。全部 11.2M 参数随机初始化，从零开始训练 20 个 epoch，`lr = 1e-3`。

```python
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 37)
```

#figure(
  image("results/task2_scratch_curves.png", width: 100%),
  caption: [Task 2 从零训练 ResNet18 的训练/验证曲线。训练损失一路降至 0.36，但验证损失在 2.4 附近震荡不降，train/val gap 巨大。],
)

== 灾难性的过拟合

训练准确率从 7% 平滑攀升至 90.9%，而验证准确率在前几个 epoch 仅爬到约 30% 就基本停滞，最终 test acc 仅 *0.361*。训练损失 0.36 与验证损失 2.40 之间出现了近一个数量级的 gap——这是典型的*灾难性过拟合*：模型在 3 300 张训练图上记住了样本，却学不到能泛化到测试集的模式。

根本原因在数据量与模型容量的严重失配。ResNet18 有 11.2M 参数，而训练集仅 3 300 张图，参数比图多出三个数量级。如此大的容量配上如此少的数据，模型有充足的能力去"死记"训练集而非学习可泛化的视觉特征——这正是 Task 3 用预训练 + 冻结所要解决的问题。

== 每 epoch 时间

从零训练每 epoch 约 122 秒，是整个 lab 最慢的 run。原因在于 backbone 每一层都要计算梯度并更新权重，计算量最大。

#pagebreak()

= 四、Task 3: 特征提取（冻结 Backbone）

加载 ImageNet 预训练的 ResNet18，替换 `fc` 为新的 `Linear(512, 37)`，并*冻结整个 backbone*——只训练头部 18 981 个参数。

```python
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 37)   # 新头部默认 requires_grad=True
```

== BatchNorm 的 eval 模式处理

冻结 backbone 时必须同时把 BN 层切到 eval 模式，否则 BN 的 running stats 会被 3 300 张宠物图的小 batch 统计污染。具体做法是先 `model.eval()`（全网络进入 eval），再 `model.fc.train()`（只让头部回到 train）：

```python
model.eval()        # 全网络 eval：BN 用 ImageNet running stats，不更新
model.fc.train()    # 仅头部 train：因为 fc 无 BN，等价于正常训练
```

这是 Task 3 的关键细节。若漏掉这一步，BN 的 running mean / var 会被宠物数据的小 batch 覆盖，预训练时学到的 ImageNet 统计量被破坏，准确率会明显下降。

#figure(
  image("results/task3_frozen_curves.png", width: 100%),
  caption: [Task 3 冻结 backbone 训练头部的曲线。验证准确率第 1 epoch 即达 0.80，最终稳定在 0.91，test acc 0.885。],
)

== 冻结的威力

验证准确率在第 1 个 epoch 就达到 0.80（对比 Task 2 同期仅 0.07），到第 3 个 epoch 已 0.88，最终 test acc *0.885*。这个数字背后有两个事实：

+ *预训练 backbone 已经提供了高质量特征*。ImageNet 1.28M 图像学到的视觉特征（边缘、纹理、物体部件）对宠物分类同样有效，头部只需要学一个从 512 维特征到 37 类的线性映射即可——这是一个近似线性的简单问题，所以 20 epoch 内就收敛。
+ *只训练 18 981 个参数*。参数量比 Task 2 少了 580 倍，却没有过拟合——因为 backbone 是固定的特征提取器，参数不会随训练样本数增加而过拟合训练集。

每 epoch 仅 43 秒（Task 2 的 1/3），因为前向只需要算 backbone 一次，反向只更新头部。

#pagebreak()

= 五、Task 4: 微调（解冻整个网络）

从 Task 3 的 checkpoint 出发，解冻整个网络，用*差异化学习率*继续训练 10 个 epoch：backbone `1e-4`，head `1e-3`。

```python
model = build_resnet(pretrained=True, freeze=False)  # 解冻
state = torch.load("checkpoints/task3_frozen.pt")
model.load_state_dict(state)                          # 续训

optimizer = optim.Adam([
    {"params": backbone_params, "lr": 1e-4},   # 预训练：小步
    {"params": model.fc.parameters(), "lr": 1e-3},  # 新头：大步
])
```

== 两个关键设计选择

+ *BN 模式切回 train*。Task 3 把 BN 保持 eval（统计量冻结在 ImageNet 值），Task 4 反过来——整个网络进入 `train()` 模式，BN 的 running stats 跟随宠物数据的特征分布更新。因为现在 backbone 权重在动，特征分布也在变，BN 必须跟上这个新分布。两件事（冻结 vs 解冻）的 BN 处理是*刻意相反*的，对应两种不同的语义：冻结时特征不变，统计量也不应变；解冻时特征在变，统计量必须跟着变。
+ *差异化学习率*。backbone 已经编码了 ImageNet 学到的有用特征，过大学习率会 *catastrophic forgetting*（灾难性遗忘）——几步就把精心学到的权重冲掉。所以 backbone 用小步 `1e-4`：足够往宠物方向特化，又不足以遗忘 ImageNet。head 是新初始化的、且 sits on a moving backbone（它下面的特征每步都在变），需要更大的 `1e-3` 来追踪这个移动目标。

#figure(
  image("results/task4_finetune_curves.png", width: 100%),
  caption: [Task 4 微调曲线。第 1 epoch 出现明显下凹（BN 切换瞬态），随后恢复并超过 Task 3 水平。train acc 在 ep8 后达 1.0000，但 val loss 仍在下降。],
)

== BN-mode-switch 瞬态

Task 3 终点 val acc 0.9130，Task 4 第 1 epoch 跌到 0.8859，第 2 epoch 回升到 0.8995，到第 7 epoch 达到峰值 0.9212。这个第 1 epoch 的下凹不是 bug：解冻的瞬间 BN 从 eval（用冻结的 ImageNet running stats）切到 train（用当前 batch 的统计量更新 running stats），running mean / var 从 ImageNet 值向宠物特征分布过渡，第一波前向传播的统计量处于过渡态，输出自然短暂失真。一旦 BN 的 running stats 重新稳定（1–2 个 epoch），准确率即恢复并超过冻结水平。

== 温和过拟合

train acc 在第 8 epoch 达到 1.0000（完全记住训练集），但 val loss 仍在缓慢下降（0.344 → 0.280）。这与 Task 2 的"灾难性过拟合"不同：这里 train/val gap 虽然存在，但 val 还在改善，属于*温和过拟合*。最终 test acc *0.8972*，比 Task 3 提升 +1.22 pp。

== 为何收益只有 +1.22 pp

微调相对冻结的提升不大，原因在于 ImageNet 预训练特征已经与宠物任务高度匹配——ImageNet 1000 个类别里有约 150 个就是猫狗品种，预训练 backbone 本就在"猫狗视觉"上受过良好训练。冻结阶段已经榨取了大部分价值，留给微调的边际空间有限。微调的价值更多体现在边缘品种的细微区分（如不同猫品种的毛色纹理），提升有限但真实。

#pagebreak()

= 六、Task 5: 数据规模消融（Headline 实验）

这是整个 lab 的核心论证。在两种数据规模（20% ≈ 660 张、100%）下分别重复 Task 2（scratch）与 Task 3（frozen），共四个数据点，绘制单张对比图。

为保证公平，20% 子采样使用固定 `seed=42` 的 `generator`，保证 scratch 与 frozen 在 20% 下用*完全相同*的 660 张图——这样两个策略之间唯一的变量就是"是否预训练"。

#figure(
  image("results/task5_data_size_ablation.png", width: 85%),
  caption: [Task 5 数据规模消融。横轴为训练数据比例，纵轴为 test acc。两条线的差距在 20% 处最大（0.72），在 100% 处缩小（0.52）。],
)

#figure(
  table(
    columns: 5,
    stroke: 0.3pt,
    align: (left, center, center, center, center),
    [数据量], [scratch test acc], [frozen test acc], [gap], [scratch train acc],
    [20%（≈660 张）], [0.128], [#text(fill: red)[0.850]], [0.722], [0.841],
    [100%（3 300 张）], [0.361], [0.885], [0.524], [0.909],
  ),
  caption: [Task 5 消融数据。20% 数据下 scratch 几乎完全失败（train 84% / test 13%，灾难性过拟合）；frozen 仍达 0.85。],
)

== 两个关键观察

+ *数据越少，预训练帮助越大*。20% 数据下 gap = 0.722，100% 数据下 gap = 0.524。这是迁移学习的核心论据：当数据稀缺时，与其让 11.2M 参数从零在 660 张图上瞎学，不如直接复用 ImageNet 学到的通用视觉特征。
+ *scratch 的 train acc 在 20% 时高达 0.841，但 test 仅 0.128*。这是过拟合的极端形态——模型把 660 张图背下来了，却对测试集一无所知。frozen 在同样的 660 张图上 train acc 0.994 / test acc 0.850，gap 小得多，因为只有 18 981 个参数可训练，无法死记，只能学到真正泛化的特征。

== 为何这是迁移学习的中心论证

在真实场景中，标注数据是稀缺资源——医学影像、工业缺陷、遥感地物，每个样本都需要领域专家标注。Task 5 量化地告诉我们：在 660 张图的小数据场景，从零训练几乎必然失败，而预训练 + 冻结可以拿到 0.85 的可用准确率。这就是为什么迁移学习是深度学习实践中的默认起点而非可选项。

#pagebreak()

= 七、Task 6: 特征可视化（t-SNE）

用冻结的预训练 backbone（`fc` 替换为 `Identity`）提取测试集的 512 维倒数第二层特征，用 `sklearn.manifold.TSNE`（`init="pca"`，`perplexity=30`）降到 2D，按品种着色。对照基线是对原始像素展平后直接做 t-SNE。

#figure(
  image("results/task6_tsne_side_by_side.png", width: 80%),
  caption: [Task 6 t-SNE 对比。左：原始像素空间（150 528 维展平）几乎无结构；右：预训练 backbone 的 512 维特征已按品种聚类。],
)

== 像素空间（左）

原始像素展平为 $224 times 224 times 3 = 150#h(2pt)528$ 维向量后做 t-SNE，散点几乎是一个无结构的色斑，不同品种的点完全混在一起。这说明像素空间里，一只柯基和一只波斯猫的"距离"与两只柯基之间的距离没有本质差异——像素级的相似度不携带语义。

== 预训练特征空间（右）

预训练 backbone 的 512 维特征做 t-SNE 后，同品种的点聚成清晰的簇，猫（圆形标记）与狗（三角形标记）在宏观上也分成了两大团。这说明 backbone 已经学到了与"品种"语义高度对齐的视觉特征——这些特征是在 ImageNet 1.28M 图像上训练得到的副产品，却天然适用于宠物分类。

== 这解释了 Task 3 为何有效

Task 3 冻结 backbone 后只训练一个线性头部就达到 0.885 准确率，根本原因就在这张图：预训练特征空间里，不同品种*线性可分*——一个 `Linear(512, 37)` 就足以完成分类。换句话说，冻结阶段的任务被简化成了一个"似凸的线性分类问题"，所以才收敛得快、过拟合轻。

#pagebreak()

= 八、汇总对比

#figure(
  table(
    columns: 6,
    stroke: 0.3pt,
    align: (left, center, center, center, center, center),
    [策略], [Pretrained?], [可训练参数], [Final Train Acc], [Final Test Acc], [s/epoch],
    [Task 2 scratch],  [否], [11 195 493], [0.909], [0.361], [121.7],
    [Task 3 frozen],  [是], [18 981],      [0.988], [0.885], [42.8],
    [Task 4 fine-tune],[是], [11 195 493], [1.000], [#text(fill: red)[0.897]], [114.7],
  ),
  caption: [Lab 4 三种策略汇总。微调取得最佳 test acc 0.897，但冻结以 1/580 的参数量达到 0.885，性价比最高。],
)

三点总结：

- *预训练 vs 从零*：Task 2 → Task 3，从 0.361 跃升到 0.885（+0.524），是整个 lab 最大的单步提升。预训练把 backbone 从"随机初始化的累赘"变成"高质量特征提取器"。
- *冻结 vs 微调*：Task 3 → Task 4，从 0.885 提升到 0.897（+0.012），收益有限。因为 ImageNet 1000 类里本就包含大量猫狗品种，预训练特征已高度适配宠物任务，冻结已提取了大部分价值。
- *参数效率*：Task 3 仅用 18 981 个参数就达到 0.885，而 Task 2 用 11.2M 参数只得到 0.361——参数多寡不决定泛化，*参数是否承载了正确的先验*才是关键。

#pagebreak()

= 九、报告思考题

== Q1: 什么是残差连接？它解决了什么问题？与梯度消失的关系？

残差连接（skip connection / shortcut connection）是 ResNet 的核心结构：在每个 BasicBlock 中，把块的输入 $x$ 直接加到块的输出上，使块学习的不再是完整映射 $H(x)$，而是残差 $F(x) = H(x) - x$。

```
        ┌─────────────────────────────┐
   x ───┤ 3×3 conv → BN → ReLU        │
   │    │ 3×3 conv → BN               ├──┐
   │    └─────────────────────────────┘  │ F(x)
   └────────────── + ────────────────────┘──→ ReLU → out
```

它解决的是*深层网络的退化问题*。朴素地堆叠卷积层，理论上更深的网络至少不会比浅网络差（把多出的层学成恒等映射即可），但实践中 SGD 很难在深层 plain 网络上找到这样的恒等映射——训练误差反而随深度增加而上升，这是 He et al. (2016) 观察到的"退化"现象，与过拟合不同（过拟合是 train 低 val 高，退化是 train 也高）。

与*梯度消失*的关系：朴素深层网络的梯度在反向传播时需要经过一连串乘法（每层一个 Jacobian），深层链式相乘容易让梯度指数级缩小，浅层参数几乎收不到信号。残差连接提供了一条梯度回传的"高速公路"——梯度可以绕过 block 的权重层直接回传到浅层：

$ (partial L)/(partial x) = (partial L)/(partial "out") dot (1 + (partial F)/(partial x)) $

其中 $1$ 这一项就是 skip connection 的贡献，它保证即使 $(partial F) / (partial x)$ 很小，梯度仍能通过 $+1$ 这条路径无损回传。这就是为什么 ResNet 能训到 152 层而梯度不消失——skip connection 是梯度的"旁路通道"。

== Q2: 区分 backbone 与 classification head。为什么替换 ImageNet 预训练模型的头部是必须的？

*Backbone*（骨干网络）指网络的特征提取部分——在 ResNet18 中是从 `conv1` 到 `avgpool` 之前的所有层，输出是 512 维特征向量。它的职责是把一张 $224 times 224 times 3$ 的图像压缩成一个紧凑的语义表示。*Classification head*（分类头）指最后的 `Linear(512, 1000)`，它把 512 维特征映射到 ImageNet 的 1000 个类别 logit。

替换头部是必须的，原因有二：

+ *类别数不匹配*。ImageNet 头部输出 1000 维（对应 1000 个 ImageNet 类别），而宠物任务只有 37 类。维度对不上，根本无法直接用。
+ *语义不匹配*。即使硬凑维度，ImageNet 头部学到的是"这张图属于 1000 个 ImageNet 概念中哪一个"的映射，而我们要的是"属于 37 个品种中哪一个"。两者是不同的分类问题，旧头部的权重对宠物任务毫无意义——它会把一张柯基的 512 维特征映射到 ImageNet 的某个类别 logit 上（可能是"狗"这个大类，但绝不是"柯基"）。

所以替换头部 = 保留 backbone 学到的通用视觉特征，丢掉头部学到的 ImageNet 特定的类别映射，重新学一个 512→37 的映射。这就是迁移学习"特征提取器通用、分类器专用"思想的具体体现。

== Q3: 为什么冻结 backbone 在小数据上有帮助？若在 660 张图上从零初始化头部微调 11.2M 参数会发生什么？

冻结 backbone 的本质是*把可训练参数从 11.2M 降到 18 981*。在 660 张图（20% 数据）这种小样本场景下：

- 11.2M 参数 vs 660 张图：参数比样本多 4 个数量级，模型有充足容量"死记"每张训练图的特征（包括噪声、光照、背景），而非学习可泛化的视觉模式。这正是 Task 5 中 scratch 20% 的灾难：train acc 0.84、test acc 0.13——背下了 660 张图却对测试集一无所知。
- 18 981 参数 vs 660 张图：参数量与样本量在同一量级，模型没有余力去记忆，只能学到真正能泛化的线性映射（从预训练 backbone 提取的高质量特征到 37 个类别）。这就是 frozen 20% 能达 0.85 的原因。

若在 660 张图上从零初始化头部、解冻整个 11.2M 参数做"scratch-style 微调"（即 Task 2 的做法），会发生什么？——就是我们 Task 5 的 scratch 20% 结果：test acc 0.13。问题不在头部初始化方式，而在"11.2M 参数 + 660 张图"这个组合本身就让过拟合不可避免。冻结 backbone 把可训练参数压到与数据量匹配的量级，才是治本之道。

== Q4: 为什么微调对 backbone 用比 head 更小的学习率？若对两者都用 1e-3 会发生什么？

backbone 用小学习率（`1e-4`）、head 用大学习率（`1e-3`）的差异化设计，源于两类参数的初始状态截然不同：

- *backbone 是预训练好的*，权重已经编码了 ImageNet 学到的高质量视觉特征。我们只想"温和地推动"它从通用视觉特征向宠物特化方向偏移，而不是推翻重来。大学习率会在几步之内把精心学到的权重冲掉，造成 *catastrophic forgetting*（灾难性遗忘）——backbone 退化回随机初始化水平，等于把预训练价值扔了。
- *head 是新初始化的*（Task 3 训练好的，但相对于一个"正在移动的 backbone"，它仍然需要快速追踪变化）。backbone 每步微调都改变其下的特征分布，head 必须跟上这个移动目标，所以需要较大学习率保持灵活性。

若对两者都用 `1e-3`：backbone 在前几个 step 就会被大幅扰动，ImageNet 学到的通用特征（边缘、纹理、物体部件）被冲散，backbone 退化成接近随机初始化的状态。此时 Task 4 退化为"在 3 300 张图上从零训练整个 ResNet18"——也就是 Task 2 的灾难性过拟合场景。预期结果是 test acc 大幅下降，可能跌回 0.4–0.5 区间，BN 的 running stats 也会在剧烈扰动下失稳。这正是为什么"差异化学习率"是微调的标准做法而非可选优化。

== Q5: 从 Task 5 消融描述 scratch 与 pretrained 的 gap 如何随数据规模变化？为什么这是迁移学习的中心论据？

从 Task 5 数据：

- 20% 数据（660 张）：scratch 0.128 vs frozen 0.850，gap = *0.722*
- 100% 数据（3 300 张）：scratch 0.361 vs frozen 0.885，gap = *0.524*

gap 随数据量增加而*缩小*（0.72 → 0.52）。趋势解读：

- 数据极少时，scratch 几乎必然失败（11.2M 参数无法在 660 张图上泛化），而 frozen 几乎不受影响（只训练 18 981 个头部参数，且 backbone 是免费的通用特征提取器）。此时预训练是"救命"级别的差距。
- 数据增多时，scratch 开始能学到一些东西（0.13 → 0.36），但仍然远不及 frozen。即便到 3 300 张图，scratch 也只到 0.36——因为 11.2M 参数对 3 300 张图仍嫌过多，过拟合依旧严重。

*为什么这是中心论据*：在真实场景中，标注数据始终是稀缺资源——医学影像需要放射科医生标注，工业缺陷需要产线工程师标注，遥感地物需要地学专家标注。Task 5 量化地告诉我们：在这些"数据天然稀缺"的场景里，从零训练几乎注定失败，而迁移学习能提供一个可用的 baseline。这不是"锦上添花"的提升，而是"从不可用到可用"的质变。这就是为什么迁移学习是深度学习实践中的*默认起点*而非可选项——它把深度学习的适用门槛从"需要百万级标注"降到了"只需要几千标注"。

== Q6: 比较 t-SNE 图。聚类的有无说明了 backbone 学到了什么？

像素空间 t-SNE（左图）：散点几乎是无结构的色斑，不同品种的点完全混在一起，猫和狗也没有宏观分离。

预训练特征空间 t-SNE（右图）：同品种的点聚成清晰的簇，猫（圆点）与狗（三角）在宏观上分成两大团，部分视觉相近的品种（如不同短毛猫）的簇也彼此靠近。

聚类说明 backbone 学到的不是"像素级的相似度"，而是*语义级的相似度*：

- 同品种的图像虽在像素空间差异巨大（不同光照、姿态、背景），但在 backbone 的 512 维特征空间里距离很近——backbone 已学会忽略这些表面变化，提取出"品种身份"这个本质特征。
- 这种语义特征是在 ImageNet 1.28M 图像上训练得到的副产品。ImageNet 的 1000 个类别里有约 150 个是猫狗品种，所以 backbone 本就在"猫狗视觉"上受过良好训练——这正是它在宠物任务上几乎"开箱即用"的原因。
- 像素空间没有聚类，说明原始像素不携带品种语义——一只柯基和一只波斯猫在像素层面的差异，与两只柯基在不同光照下的差异，对像素度量来说没有本质区别。深度学习的价值正在于：通过学习，把"像素空间里不可分的"问题，变换到"特征空间里线性可分"的问题。Task 3 冻结阶段只训一个 `Linear(512, 37)` 就达 0.885，正是因为特征空间已经线性可分。

== Q7: 连接到 Lab 3。Lab 3 的归纳偏置*写进架构*，Lab 4 的归纳偏置从何而来？两者是互补还是竞争？

*Lab 3 的归纳偏置写在架构里*。卷积核的局部感受野（$k times k$ 窗口）与权值共享（同一核在整张图滑动）是 CNN 的两个核心归纳偏置，它们被硬编码在 `Conv2d` 的实现里——无论你怎么训练，这两个偏置都在起作用。这些偏置表达的是"图像中相邻像素共同构成局部模式，且同一模式可在图像不同位置复用"的先验。LeNet 44k 参数超越 MLP 118k 参数，正是这两个架构级归纳偏置的功劳。

*Lab 4 的归纳偏置来自预训练权重*。ResNet18 的架构本身（卷积 + 残差连接）确实也包含 Lab 3 那种架构级归纳偏置，但 Lab 4 真正的核心是"ImageNet 预训练权重"——这 11.2M 个参数的*取值*编码了另一个层次的归纳偏置：从 1.28M 张自然图像中学到的"什么是边缘、什么是纹理、什么是物体部件、什么是猫狗轮廓"。这是一种*数据驱动的归纳偏置*，不是写在架构里的，而是从数据里学出来的。

*两者是互补的，而非竞争*：

- 架构级归纳偏置（Lab 3）让网络"会看局部、会共享权重"，这是它能有效利用图像结构的*能力前提*。没有卷积，就算给 1.28M 张图，一个 MLP 也学不到等质量的视觉特征。
- 数据级归纳偏置（Lab 4）让网络"已经见过很多自然图像"，这是它在新任务上不需要从零学起的*知识前提*。没有预训练，再好的 ResNet 架构在 660 张图上也只能过拟合。

两者叠加才构成完整的迁移学习：架构提供"能学好视觉特征"的能力，预训练提供"已经学好了"的状态。Task 2 从零训练 ResNet18 失败，正是因为它有架构级归纳偏置却缺数据级归纳偏置——能力有了，但知识没有，11.2M 参数在 3 300 张图上只能学到过拟合的噪声。从这个角度看，Lab 4 是 Lab 3 的自然延伸：Lab 3 证明了"架构里的先验重要"，Lab 4 证明了"权重里的先验同样重要，且在小数据下更关键"。

#pagebreak()

= 十、总结

本实验在 Oxford-IIIT Pet 数据集上完整实践了迁移学习的三种策略，并用消融与可视化论证了其有效性。核心结论：

- *预训练价值远超架构调参*。Task 2 → Task 3，从 0.361 跃升到 0.885（+0.524），是整个 lab 最大的单步提升。这一步没有改架构、没有调超参，只是把"随机初始化"换成"ImageNet 预训练"。
- *数据越少，预训练越关键*。Task 5 量化显示 20% 数据下 gap 0.72、100% 数据下 gap 0.52。这是迁移学习作为小数据场景默认起点的核心论据。
- *冻结即可，微调收益有限*。Task 3 → Task 4 仅 +1.22 pp，因为 ImageNet 预训练特征已与宠物任务高度匹配。微调的价值在边缘品种的细微区分，提升有限但真实。
- *差异化学习率是微调的标准做法*。backbone 用小步（`1e-4`）避免灾难性遗忘，head 用大步（`1e-3`）追踪移动的 backbone。两者用同一学习率会让 backbone 退化为接近随机初始化。
- *预训练特征空间线性可分*。Task 6 t-SNE 显示 backbone 的 512 维特征已按品种聚类，这就是 Task 3 冻结阶段只训一个线性头就能达到 0.885 的根本原因。

从 Lab 3 到 Lab 4，我们看到了归纳偏置的两次注入：Lab 3 把"局部性"和"权值共享"写进架构，Lab 4 把"自然图像的通用视觉结构"写进权重。前者提供能力，后者提供知识。两者叠加，才让深度学习在数据稀缺的真实场景中真正可用。
