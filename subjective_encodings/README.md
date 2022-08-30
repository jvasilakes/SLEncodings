# Subjective Logic Encodings

Subjecti Logic Encodings (`sle`) is a small PyTorch extension for encoding and predicting Subjective Logic distributions in place of hard labels.

## Installation

The only requirement is PyTorch. This release was tested with `torch=1.12.0+cu102`.

```
python setup.py develop
```

By using `develop`, the package will be automatically updated when changes are pulled from github.


## Uninstallation

```
python setup.py develop --uninstall
```


## Usage

`sle` provides a ready-to-use prediction layer at `sle.layers.SLELayer` that can plug in directly to any classification model, in place of the usual output layer.

```python
import torch
import torch.distributions as D

import sle

N = 5
x_dim = 10
hidden_dim = 5
y_dim = 2

# Binary classification
x = torch.randn(N, x_dim)
y = torch.randint(y_dim, size=(N,1))
y_enc = sle.encode_labels(y, y_dim)
collated = sle.collate.sle_default_collate(y_enc)

model = torch.nn.Sequential(
	torch.nn.Linear(x_dim, hidden_dim),
	torch.nn.Tanh(),
	sle.layers.SLELayer(hidden_dim, y_dim))
output = model(x)
# output will be a SLBeta instance
loss = D.kl_divergence(collated, output)


# Multi-label classification (4 labels)
label_dim = 4
y_idxs = torch.randint(num_labels, size=(N,))
y = torch.zeros((N, y_dim))
y[torch.arange(N), y_idxs] = 1  # one-hot encoding
y_enc = sle.encode_labels(y)
collated = sle.collate.sle_default_collate(y_enc)
# collated will be a SLDirichlet instance

model = torch.nn.Sequential(
	torch.nn.Linear(x_dim, hidden_dim),
	torch.nn.Tanh(),
	sle.layers.SLELayer(hidden_dim, y_dim))
output = model(x)
# output will be a SLDirichlet instance
loss = D.kl_divergence(collated, output)
```
