import torch
import torch.distributions as D

from .distributions import SLBeta, SLDirichlet
from .layers import BetaLayer, DirichletLayer  # noqa


EPS = 1e-18


def encode_labels(labels, num_labels, collate=False):
    labels = torch.as_tensor(labels)
    if labels.dim() == 0:
        raise ValueError(f"Input to encode_labels must be a 1 or 2 dimensional vector.")  # noqa
    elif labels.dim() == 1:
        labels = labels.unsqueeze(1)
    elif labels.dim() > 2:
        raise ValueError(f"Input to encode_labels must be a 1 or 2 dimensional vector.")  # noqa

    if labels.size(1) > 1:
        raise ValueError("Multi-label tasks not yet supported.")

    encoded = [encode_one(y, num_labels) for y in labels]
    if collate is True:
        encoded = collate_sle_labels(encoded)
    return encoded


def encode_one(label, num_labels):
    # Binary
    if num_labels in [1, 2]:
        return label2beta(label)
    # Multi-class
    elif num_labels > 2:
        return label2dirichlet(label, num_labels)
    else:
        raise ValueError(f"Unsupported number of unique labels {num_labels}")


def collate_sle_labels(encoded):
    collated_params = {}
    param_names = encoded[0].parameters().keys()
    for param_name in param_names:
        param_vals = [sl_lab.parameters()[param_name]
                      for sl_lab in encoded]
        collated = torch.stack(param_vals, dim=0)
        collated_params[param_name] = collated
    dist_cls = encoded[0].__class__
    encoded = dist_cls(**collated_params)
    return encoded


def label2beta(label):
    if label not in [0, 1]:
        raise ValueError("Only binary {0,1} values are supported")

    u = torch.tensor(0. + EPS)
    if label == 1:
        b = torch.tensor(1. - EPS)
        d = torch.tensor(0.)
    else:
        b = torch.tensor(0.)
        d = torch.tensor(1. - EPS)
    return SLBeta(b, d, u)


def label2dirichlet(label, num_labels):
    if label not in range(num_labels):
        raise ValueError(f"label {label} not in {num_labels}")

    beliefs = torch.zeros(num_labels)
    beliefs[label] = torch.tensor(1. - EPS)
    unc = torch.tensor([0. + EPS])
    return SLDirichlet(beliefs, unc)


def cross_entropy(dist1: D.Distribution, dist2: D.Distribution):
    return D.kl_divergence(dist1, dist2) + dist1.entropy()
