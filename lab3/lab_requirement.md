# Lab 3: CNN for MNIST Handwritten Digit Recognition

## 1. Objective

This lab aims to verify your understanding of convolutional neural networks by revisiting the MNIST handwritten digit classification task from Lab 1. In Lab 1, the MLP flattened each 28x28 image into a 784-dimensional vector, which destroyed the spatial structure of the image. Even with careful tuning, this approach has a natural performance ceiling because the model must learn spatial relationships from scratch.

In this lab, you will build CNN-based classifiers for MNIST and compare them with the MLP baseline. By completing this lab, you should be able to:

- Explain why flattening an image loses useful spatial information.
- Implement a basic CNN using PyTorch.
- Understand how convolution, padding, stride, pooling, and fully connected layers affect tensor shapes.
- Compare MLP and CNN performance under similar training settings.
- Analyse how CNN design choices affect accuracy, training speed, parameter count, and overfitting.

The expected result is that a reasonably designed CNN should exceed the Lab 1 MLP baseline and reach above 98% test accuracy on MNIST.

## 2. Prerequisites

You are expected to have completed:

- Lab 1: MNIST classification with MLP.
- Lab 2: manual forward/backward implementation and gradient-based learning.
- Andrew Ng's CNN-related deep learning lectures, including convolution, padding, stride, pooling, and basic CNN architectures.

You should be comfortable with Python, PyTorch, tensor shapes, and the training loop pattern used in Lab 1.

## 3. Software Environment

- Python 3.8 or above
- PyTorch
- torchvision
- NumPy
- matplotlib

All experiments should run on CPU. Use a fixed random seed (`torch.manual_seed(42)`) for reproducibility.

Do not introduce additional deep learning frameworks. The purpose of this lab is to understand CNNs, not to use a high-level training wrapper.

## 4. Dataset

Use the same MNIST dataset as Lab 1:

- 60,000 training images
- 10,000 test images
- 28x28 grayscale input
- 10 classes, corresponding to digits 0 through 9

Unlike Lab 1, do not flatten the image before the convolutional layers. The input tensor should have shape:

```python
(batch_size, 1, 28, 28)
```

Pixel values should be converted to floating point and normalised. You may use either:

- `transforms.ToTensor()` only, which scales pixels to `[0, 1]`; or
- `transforms.Normalize((0.1307,), (0.3081,))`, the common MNIST normalisation.

State clearly which choice you use in the report.

## 5. Tasks

Complete the tasks in order. Each task should build on the previous one.

### Task 1: Reproduce the MLP Baseline

Start from your best or representative Lab 1 MLP model. Train it under a fixed setting and record:

- Training loss and accuracy
- Validation or test loss and accuracy
- Final test accuracy
- Number of trainable parameters

This model will serve as the baseline for CNN comparison.

Suggested baseline:

```python
Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 10)
```

Use `CrossEntropyLoss`, so the final layer should output raw logits rather than probabilities.

### Task 2: Build a Minimal CNN

Implement a simple CNN with one convolutional block:

```python
Conv2d(1, 16, kernel_size=3, padding=1)
ReLU
MaxPool2d(kernel_size=2)
Flatten
Linear(...)
ReLU
Linear(..., 10)
```

You must explicitly compute and print the tensor shape after each major stage for one dummy batch:

- Input
- After convolution
- After pooling
- After flattening
- Output logits

This task is not only about accuracy. It is mainly to verify that you understand how CNN tensor shapes flow through the model.

### Task 3: Build a LeNet-Style CNN

Implement a stronger CNN inspired by LeNet-5:

```python
Conv2d(1, 6, kernel_size=5)
Tanh or ReLU
AvgPool2d or MaxPool2d
Conv2d(6, 16, kernel_size=5)
Tanh or ReLU
AvgPool2d or MaxPool2d
Flatten
Linear(...)
Activation
Linear(..., 10)
```

You do not need to reproduce LeNet-5 exactly. MNIST inputs are 28x28 rather than the 32x32 input used in the original LeNet-5 paper, so adjust shapes carefully.

Record:

- Final test accuracy
- Training curves
- Parameter count
- Whether this model exceeds the Lab 1 MLP baseline

### Task 4: Compare Key CNN Design Choices

Using your LeNet-style model as the base, run controlled experiments by changing one factor at a time.

#### Experiment A: Activation Function

Compare:

- `Tanh`
- `ReLU`

Observe convergence speed and final accuracy. Explain why modern CNNs usually prefer ReLU, while classic LeNet used tanh-like nonlinearities.

#### Experiment B: Pooling Type

Compare:

- `AvgPool2d`
- `MaxPool2d`

Observe whether max pooling improves performance or convergence. Explain the intuition behind pooling as local translation robustness.

#### Experiment C: Number of Channels

Compare at least three channel configurations, for example:

- `(6, 16)` classic LeNet-style
- `(16, 32)` moderate CNN
- `(32, 64)` wider CNN

Record parameter count and test accuracy. Discuss whether increasing channels is always worth the extra computation.

#### Experiment D: Regularisation

Add one or more of the following:

- Dropout before the final classifier
- Weight decay in the optimizer
- Data augmentation with small random affine transforms

Do not use heavy augmentation. MNIST should remain recognisable. Suggested transform:

```python
transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
```

Observe whether regularisation improves test accuracy or mainly slows down training.

### Task 5: Error Analysis

After training your best CNN, inspect at least 20 misclassified test images. Save a figure showing several examples with:

- The true label
- The predicted label
- The model confidence for the predicted class

Answer:

- Which digits are most often confused?
- Are the mistakes visually ambiguous to humans?
- Does the CNN make fewer obvious mistakes than the MLP baseline?

## 6. Suggested Training Settings

Use settings similar to Lab 1 unless you have a clear reason to change them:

- Batch size: 64 or 128
- Epochs: 10 to 20
- Optimizer: Adam or SGD with momentum
- Loss: `torch.nn.CrossEntropyLoss`
- Learning rate:
  - Adam: start with `1e-3`
  - SGD with momentum: start with `1e-2`

For each experiment, keep the training setting fixed unless the experiment explicitly studies the optimizer or learning rate.

If CPU training is slow, reduce epochs first. Do not change multiple experimental factors at once.

## 7. Required Outputs

Your final lab folder should contain:

- `lab3.py` or multiple scripts such as `task1_mlp_baseline.py`, `task2_minimal_cnn.py`, and `task3_lenet_cnn.py`
- Loss and accuracy plots for the MLP baseline and at least two CNN variants
- A comparison table containing:
  - Model name
  - Number of trainable parameters
  - Final train accuracy
  - Final test accuracy
  - Training time per epoch, if measured
- A misclassification analysis figure for the best CNN
- A report file, preferably `lab3.typ` or `report.md`

## 8. Report Questions

Answer the following questions in your report:

1. Why does flattening an image before the first layer lose spatial structure?
2. How do local receptive fields and shared weights reduce the number of parameters compared with an MLP?
3. What is the role of pooling? What information might pooling discard?
4. Did the CNN exceed the Lab 1 MLP baseline? If yes, which design choice contributed most? If no, what might have gone wrong?
5. Compare the parameter count of the MLP and CNN. Does the more accurate model necessarily have more parameters?
6. Which misclassified digits are hardest for the model? Are these errors reasonable?
7. How does this experiment connect to the LeNet-5 paper?

## 9. Evaluation Criteria

- Correct PyTorch implementation of MLP and CNN models.
- Correct handling and explanation of tensor shapes.
- Fair comparison between MLP and CNN under controlled settings.
- Clear experimental tables and plots.
- Thoughtful analysis of spatial structure, weight sharing, pooling, and generalisation.
- Meaningful misclassification analysis rather than only reporting accuracy.
- Reproducibility through fixed random seeds and documented hyperparameters.

## 10. Hints

- Use `model.train()` during training and `model.eval()` during evaluation.
- Wrap evaluation code with `torch.no_grad()`.
- Remember that `CrossEntropyLoss` expects raw logits, not softmax probabilities.
- Use a helper function to count parameters:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- To debug shape errors, pass a dummy tensor through the model:

```python
x = torch.randn(4, 1, 28, 28)
logits = model(x)
print(logits.shape)
```

- If you use `nn.Sequential`, `nn.Flatten()` is useful before the first linear layer.
- For MNIST, a small CNN should already perform very well. If accuracy is poor, first check normalisation, tensor shape, loss function, and whether the model outputs logits.
