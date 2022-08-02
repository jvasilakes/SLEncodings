# Subjective Logic Encodings

`subjective_encodings` is a small PyTorch extension for encoding and predicting Subjective Logic distributions in place of
hard labels.

# Installation

The only requirement is PyTorch. This release was tested with `torch=1.12.0+cu102`.

```
python setup.py develop
```

By using `develop`, the package will be automatically updated when changes are pulled from github.


# Uninstallation

```
python setup.py develop --uninstall
```


# Usage

`subjective_encodings` provides two prediction layers that can plug in directly to any classification model, in place of the usual
output layer. These are `subjective_encodings.BetaLayer` for binary prediction tasks and `subjective_encodings.DirichletLayer` for 
multi-label prediction tasks. 

```python
import torch
import torch.distributions as D

import subjective_encodings as sle

N = 5
input_dim = 10
hidden_dim = 5
label_dim = 2

# Binary classification
x = torch.randn(N, input_dim)
y = torch.randint(label_dim, size=(N,1))
y_enc = sle.encode_labels(y, label_dim, collate=True)

model = torch.nn.Sequential(
	torch.nn.Linear(input_dim, hidden_dim),
	torch.nn.Tanh(),
	sle.BetaLayer(hidden_dim))
output = model(x)
loss = D.kl_divergence(outputs, enc)


# Multi-label classification (4 labels)
label_dim = 4
y = torch.randint(num_labels, size=(N,1))
y_sl = sle.encode_labels(y, label_dim, collate=True)

model = torch.nn.Sequential(
	torch.nn.Linear(input_dim, hidden_dim),
	torch.nn.Tanh(),
	sle.DirichletLayer(hidden_dim, num_labels))
output = model(x)
loss = D.kl_divergence(outputs, enc)
```
