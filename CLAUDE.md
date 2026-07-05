# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

This is an educational repository for learning deep learning fundamentals from scratch. It belongs to a student at Zhejiang University and contains three sub-projects that build on each other progressively.

## Project Structure

```
deep-learning-from-scratch/
├── car_or_truck/           # First neural network: binary car/truck classifier
├── lab1/                   # MNIST MLP with hyperparameter exploration (PyTorch)
├── lab2/                   # NumPy-only manual NN implementation (linear/logistic regression + 1-hidden-layer NN)
├── requirements.txt        # Python dependencies
└── CLAUDE.md               # This file
```

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
