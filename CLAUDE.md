# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

This is an educational repository for learning deep learning fundamentals from scratch. It belongs to a student at Zhejiang University and contains three sub-projects that build on each other progressively.

## User Profile and Collaboration Preferences

The user is a Computer Science and Technology student at Zhejiang University, entering sophomore year. They are early in their deep learning studies. They have solid Python fundamentals and are learning NumPy/PyTorch/deep learning frameworks through this project. They have studied enough linear algebra and calculus to read formulas; detailed derivations are useful when explicitly requested, but are not always needed.

### Communication

- Communicate with the user in Chinese by default. English technical terms such as `backpropagation`, `autograd`, `SGD`, `BatchNorm`, `Dropout`, `loss`, and `gradient` are fine and often preferred.
- Keep explanations teaching-oriented. This is a learning repository, not a production system.
- When explaining math, state the key formula and intuition. Do not expand every calculus derivation unless the user asks for it or the derivation is central to the current learning goal.

### Implementation Style

- The user usually expects the agent to implement code on their behalf, but only after they have clearly indicated that they understand the relevant concept and know what the agent is doing.
- Preserve the educational nature of the code. Prefer clear, readable implementations over production abstractions or excessive optimization.
- Avoid introducing libraries, frameworks, or file structure changes beyond the lab's intended scope unless the user explicitly asks.
- The user should be able to explain the code afterward. Do not hide important logic behind unnecessary helper layers.

### Lab and Report Style

- For experiment design, follow the style of `lab1/lab_requirement.md` and `lab2/lab_requirement.md`: incremental tasks, fixed seeds, visible metrics, plots, and short analysis questions.
- For report writing, follow the existing `.typ` files: Typst reports with academic formatting, English report titles/section framing where appropriate, and Chinese technical discussion where natural.
- Keep the tone student-like and defensible. Avoid over-polished, generic, or obviously AI-written prose.
- Generated plots or PDFs that are useful for reports should be saved in visible project locations near the relevant lab. Temporary inspection artifacts may go under a project-local `tmp/` directory, which should be ignored by git if created.

### Execution and Dependencies

- Do not install dependencies automatically. The user must manually download or install all dependencies.
- Do not start training runs automatically unless the user explicitly asks. The user wants to start all training themselves.
- If a proposed run may be time-consuming on CPU, especially anything plausibly approaching or exceeding one hour, warn the user first that it may take some time. The user does not have a GPU.
- It is acceptable to inspect code, run lightweight non-training checks, and compile or inspect report files when that does not require new dependencies or long computation.

### Git Workflow

- Before starting a new substantive step, check the git status.
- The user's preferred rhythm is: before doing a new step, commit the previous completed change; do not commit the current in-progress change. The current change should become the "previous change" for the next step.
- When Codex materially authors or assists with a committed change, include a commit trailer such as `Co-Authored-By: Codex <noreply@openai.com>` unless the user specifies a different Codex identity.
- Never automatically commit unrelated user changes. If unrelated dirty files exist, leave them alone unless the user explicitly asks.
- Do not delete generated figures, reports, datasets, or user-visible outputs unless the user explicitly asks or they are clearly temporary artifacts created during the current task.

## Project Structure

```
deep-learning-from-scratch/
├── car_or_truck/           # First neural network: binary car/truck classifier
├── lab1/                   # MNIST MLP with hyperparameter exploration (PyTorch)
├── lab2/                   # NumPy-only manual NN implementation (linear/logistic regression + 1-hidden-layer NN)
├── lab3/                   # CNN for MNIST (tensor-shape tracking, controlled design comparisons)
├── lab4/                   # Transfer learning with pretrained ResNet18 on Oxford-IIIT Pet
├── minimal-rnn/            # Minimal RNN stub: predict next sine point with nn.RNN
├── lab5/                   # Sequence modeling: RNN → LSTM/GRU (BPTT, vanishing gradients, hand-written LSTM)
├── lab6/                   # Attention and Transformer from scratch (the keystone of the advanced track)
├── lab7/                   # Vision Transformer: CV + Transformer confluence, controlled ViT-vs-CNN comparison
├── lab-gen/                # Autoencoders and VAE (generative modeling, reparameterisation, β-VAE)
├── lab-opt/                # Optimizers from scratch (SGD/momentum/RMSProp/Adam, verified vs torch.optim)
├── papers/                 # Classic CV papers (LeNet→AlexNet→VGG→…→YOLO)
├── requirements.txt        # Python dependencies
└── CLAUDE.md               # This file
```

The labs form a progression. The core chain is hand-written MLP/backprop (lab2) → CNN (lab3) → transfer learning (lab4), with a parallel sequence track that grows the `minimal-rnn` stub into RNN/LSTM (lab5) → Transformer (lab6) → Vision Transformer (lab7, where the CV and sequence tracks merge). `lab-gen` (autoencoders/VAE) and `lab-opt` (optimizers) are optional branches reachable after lab6 / lab2 respectively.

### Advanced labs (lab5–lab7, lab-gen, lab-opt) — status

These five labs currently contain only a `lab_requirement.md` each. They are **planned but not yet started** — no implementation code, no report. They were authored as a curriculum so the user can work through them incrementally. When the user begins one, treat its `lab_requirement.md` as the spec (same authority as lab1/lab2's). The requirements follow the established style: incremental tasks, fixed seeds, visible metrics/plots, controlled one-axis comparisons, hand-implementation verified against library/autograd, CPU-only with time warnings, and Typst reports.

### car_or_truck/ — Binary Classification Introduction

Simple 2-feature binary classifier (car vs. truck) built with PyTorch. Introduces the full ML pipeline: synthetic data generation, feature normalization, model definition via `nn.Sequential`, training loop, and evaluation. Two variants exist:
- `car_or_truck.py` — Linearly separable data, basic training
- `car_or_truck_overlap.py` — Overlapping data, adds early stopping, validation set, and 5-fold cross-validation

Detailed learning notes are in `note_2026_6_12.md` and `note_2026_6_13.md`.

### lab1/ — MNIST Hyperparameter Exploration

Configurable MLP (`FlexibleMLP` class in `lab1.py`) for MNIST digit classification. Uses a config dict pattern to systematically vary:
- **Hidden layer count** (0, 1, 2, 3 layers) → Experiment A
- **Neuron count** (16, 64, 128, 512) → Experiment B
- **Activation functions** (ReLU, Tanh, Sigmoid) → Experiment C
- **Dropout** (0.0, 0.2, 0.5) → Experiment D
- **Batch Normalization** → Experiment E

Each experiment variant is a separate script (`lab1_*.py`). The lab report is in `lab1.typ` (Typst format).

**Key architectural pattern**: The `FlexibleMLP` class builds a `nn.Sequential` network dynamically from a config dict, inserting optional BatchNorm and Dropout layers between Linear + activation as specified.

### lab2/ — Manual NumPy Neural Networks

Implements learning algorithms from scratch with NumPy only (no autograd). Four incremental tasks:
- **Task 1** (`task1_numpy_basics.py`): NumPy fundamentals — array ops, standardization, sigmoid (numerically stable), binary cross-entropy loss
- **Task 2** (`task2_linear_regression.py`): Linear regression via mini-batch SGD — manual gradient derivation for MSE loss
- **Task 3** (`task3_logistic_regression.py`): Logistic regression on moons dataset — manual gradients for binary cross-entropy + decision boundary visualization
- **Task 4** (`task4.py`): Single-hidden-layer NN (ReLU → Sigmoid) with full manual backpropagation + PyTorch gradient verification

**Key architectural pattern**: Each task implements forward pass, backward pass (manual chain rule), and mini-batch gradient descent in a loop, with full-batch evaluation at each epoch end for monitoring. Task 4 validates correctness by comparing against PyTorch's `autograd`.

## Dependencies

- PyTorch 2.x + torchvision (for MNIST data loading)
- NumPy
- scikit-learn (data splitting, dataset generation)
- matplotlib (visualization)

Install: `pip install -r requirements.txt`

## Common Commands

### Running scripts

All scripts are standalone Python files. Run any directly:

```bash
python car_or_truck/car_or_truck.py
python lab1/lab1.py
python lab2/task2_linear_regression.py
```

### Generating lab reports

Reports are written in Typst (`.typ` files). Compile with:

```bash
typst compile lab1/lab1.typ
```

## Key Patterns to Know

1. **Config-driven experiments**: lab1 uses a Python dict for all hyperparameters, making it easy to run variants by changing one key.
2. **Manual gradient descent**: lab2 tasks derive and code gradients by hand — no autograd until Task 4's verification step.
3. **Mini-batch SGD**: All training uses mini-batch gradient descent with per-epoch full-dataset evaluation for monitoring.
4. **Numerically stable sigmoid**: Uses a branching implementation (`z >= 0` vs `z < 0`) to avoid overflow in `exp(-z)`.

## Random Seeds

All scripts fix `torch.manual_seed(42)` or `np.random.seed(42)` for reproducibility.
