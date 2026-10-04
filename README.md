# my_micrograd

A tiny scalar-valued automatic differentiation engine and neural network library built from scratch for learning purposes.

This project is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## What I implemented

### Automatic differentiation

`engine.py` implements a `Value` class that supports:

- addition
- multiplication
- subtraction
- division
- exponentiation
- ReLU
- automatic computation graph construction
- reverse-mode automatic differentiation
- gradient accumulation
- topological ordering for backpropagation

Example:

```python
from engine import Value

a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a
e = d * 2

e.backward()

print(e.data)  # 16.0
print(a.grad)  # 8.0
print(b.grad)  # 4.0
```

## Neural networks

`nn.py` implements:

- `Module`
- `Neuron`
- `Layer`
- `MLP`

Example:

```python
from nn import MLP

model = MLP(2, [4, 1])

x = [1.0, 2.0]

score = model(x)

score.backward()
```

The hidden layers use ReLU activations, while the final layer is linear.

## Training

The repository contains two simple training examples.

`train.py` trains a small model to learn:

```text
y = x1 + x2
```

`train_classifier.py` trains a small MLP for binary classification using:

- max-margin loss
- L2 regularization
- gradient descent

The training process follows:

```text
forward
→ calculate loss
→ zero gradients
→ backward
→ update parameters
```

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover:

- gradient accumulation
- shared computation graph nodes
- chain rule
- ReLU
- powers
- division
- branched computation graphs

Some gradients were also compared against PyTorch as a reference implementation.

## Why I built this

The goal of this project is to understand what frameworks such as PyTorch do underneath APIs like:

```python
loss.backward()
optimizer.step()
```

Rather than treating automatic differentiation and neural-network training as black boxes, this project implements the core ideas from scratch.