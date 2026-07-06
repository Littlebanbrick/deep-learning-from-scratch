# Deep Learning from Scratch

This repository is an educational deep learning workspace focused on building intuition from first principles. It progresses from simple synthetic classification tasks to MNIST image classification with multilayer perceptrons, manual NumPy implementations of gradient-based learning, and convolutional neural networks.

The project is maintained as a learning record and lab environment. The code favors clarity, reproducibility, and direct connection to core deep learning concepts over production-level abstraction.

## Project Overview

The repository currently contains four main learning components:

```text
deep-learning-from-scratch/
├── car_or_truck/   # Introductory binary classification with PyTorch
├── lab1/           # MNIST classification with MLPs and hyperparameter studies
├── lab2/           # NumPy implementations of regression, classification, and backpropagation
├── lab3/           # CNN-based MNIST experiments
├── papers/         # Reading list for document recognition and CNN development
└── requirements.txt
```

## Learning Path

### 1. `car_or_truck/`

This section introduces the basic machine learning workflow through a small binary classification problem. It covers:

- Synthetic data generation
- Feature normalization
- Simple PyTorch models
- Training and validation loops
- Early stopping and cross-validation in the overlapping-data variant

It serves as the first step before moving to larger image classification tasks.

### 2. `lab1/`: MNIST with Multilayer Perceptrons

Lab 1 uses PyTorch to train fully connected neural networks on MNIST. The main purpose is to study how model design choices affect accuracy and generalization.

Experiments include:

- Number of hidden layers
- Number of neurons per layer
- Activation functions: ReLU, Tanh, Sigmoid
- Dropout
- Batch normalization

The central limitation explored in this lab is that MLPs flatten each image into a vector, which discards the original spatial structure of the 28x28 digit image.

### 3. `lab2/`: Manual NumPy Neural Networks

Lab 2 removes high-level autograd support and implements core learning algorithms manually using NumPy.

Tasks include:

- NumPy fundamentals and vectorization
- Linear regression with gradient descent
- Logistic regression with binary cross-entropy
- A single-hidden-layer neural network with manual backpropagation
- Gradient comparison with PyTorch autograd

This lab is intended to build a concrete understanding of forward propagation, loss functions, gradients, and the chain rule.

### 4. `lab3/`: CNNs for MNIST

Lab 3 revisits MNIST after studying convolutional neural networks. It compares the Lab 1 MLP baseline with CNN models that preserve spatial structure.

Current models include:

- A reproduced MLP baseline
- A minimal CNN
- A LeNet-style CNN following the classic pattern:

```text
Conv -> Pool -> Conv -> Pool -> FC -> FC
```

The goal is to verify why convolution, local receptive fields, weight sharing, and pooling are effective for image recognition.

## Papers

The `papers/` directory contains selected classic and modern papers related to document recognition and convolutional networks, including:

- High-performance reading machines
- Gradient-based learning applied to document recognition
- AlexNet
- VGG
- GoogLeNet
- MobileNetV2
- EfficientNet
- YOLO

These papers provide historical and technical context for the progression from traditional document recognition systems to modern CNN-based visual recognition.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib

All experiments are designed to run on CPU. Fixed random seeds are used where appropriate to improve reproducibility.

## Running Experiments

Most scripts are standalone and can be run directly.

Examples:

```bash
python car_or_truck/car_or_truck.py
python lab1/lab1.py
python lab2/task2_linear_regression.py
python lab3/lab3.py
```

MNIST data may be downloaded automatically by `torchvision` when running Lab 1 or Lab 3. Dataset folders are not intended to be committed.

## Reports

Some labs include Typst reports:

```bash
typst compile lab1/lab1.typ
typst compile lab2/lab2.typ
```

Generated plots and reports are kept near the relevant lab when they are useful for analysis.

## Repository Principles

- Prefer readable educational code over unnecessary abstraction.
- Keep experiments reproducible through fixed seeds and documented hyperparameters.
- Change one experimental variable at a time when comparing models.
- Record both quantitative metrics and qualitative observations.
- Use manual implementations when the goal is conceptual understanding.
- Use PyTorch when the goal is model-building practice and experimentation.

## Status

This repository is actively evolving as the learning sequence progresses. Lab 3 is currently focused on CNN experiments for MNIST and will likely expand with controlled comparisons of activation functions, pooling methods, channel counts, regularization, and error analysis.
