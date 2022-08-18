import torch
import torch.distributions as D

from .layers import BetaLayer, DirichletLayer  # noqa F401
from .distributions import SLBeta, SLDirichlet


# Smoothing one-hot labels so KL-divergence is non-zero.
EPS = 1e-6


def encode_labels(labels, num_labels, uncertainties=None, priors=None):
    labels = torch.as_tensor(labels)
    if labels.dim() == 0:
        raise ValueError(f"Input to encode_labels must be a 1 or 2 dimensional vector.")  # noqa
    elif labels.dim() == 1:
        labels = labels.unsqueeze(1)
    elif labels.dim() > 2:
        raise ValueError(f"Input to encode_labels must be a 1 or 2 dimensional vector.")  # noqa

    if labels.size(1) > 1:
        raise ValueError("Multi-label tasks not yet supported.")

    if uncertainties is None:
        uncertainties = [None] * len(labels)
    if priors is None:
        priors = [None] * len(labels)
    encoded = [encode_one(y, num_labels, u, a)
               for (y, u, a) in zip(labels, uncertainties, priors)]
    return encoded


def encode_one(label, num_labels, u=0, a=None):
    # Binary
    if num_labels in [1, 2]:
        return label2beta(label, u=u, a=a)
    # Multi-class
    elif num_labels > 2:
        return label2dirichlet(label, num_labels, u=u, a=a)
    else:
        raise ValueError(f"Unsupported number of unique labels {num_labels}")


def label2beta(label, u=0, a=None):
    if label not in [0, 1]:
        raise ValueError("Only binary {0,1} values are supported")

    u = torch.as_tensor(u, dtype=torch.float32)
    if u == 0:
        u += EPS
    if label == 1:
        b = 1. - u
        d = torch.tensor(0.)
    else:
        b = torch.tensor(0.)
        d = 1. - u
    return SLBeta(b, d, u, a)


def label2dirichlet(label, num_labels, u=0, a=None):
    if label not in range(num_labels):
        raise ValueError(f"label {label} not in {num_labels}")

    u = torch.as_tensor([u], dtype=torch.float32)
    if u == 0:
        u += EPS
    beliefs = torch.zeros(num_labels)
    beliefs[label] = 1. - u
    return SLDirichlet(beliefs, u, a)


def cross_entropy(dist1: D.Distribution, dist2: D.Distribution):
    return D.kl_divergence(dist1, dist2) + dist1.entropy()
