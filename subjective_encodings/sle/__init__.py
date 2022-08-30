import torch
import torch.distributions as D

from .distributions import SLBeta, SLDirichlet


# Smoothing one-hot labels so KL-divergence is okay for optimization.
EPS = 1e-6


def encode_labels(ys, uncertainties=None, priors=None):
    """
    ys: a one-hot or probabilistic encoding of labels.
    """
    if len(ys.shape) != 2:
        raise ValueError("Input to encode_labels must be 2 dimensional.")
    ys = torch.as_tensor(ys)
    if (ys.sum(axis=1) == 0).all() is False:
        raise ValueError("ys must sum to 1 per annotation.")

    if uncertainties is None:
        # uncertainties = torch.zeros_like(ys)
        uncertainties = torch.zeros((ys.shape[0],))
    if priors is None:
        priors = [None] * ys.shape[0]
    encoded = [encode_one(y, u, a) for (y, u, a)
               in zip(ys, uncertainties, priors)]
    return encoded


def old_encode_labels(labels, num_labels, uncertainties=None, priors=None):
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
        uncertainties = [0.] * len(labels)
    if priors is None:
        priors = [None] * len(labels)
    encoded = [encode_one(y, num_labels, u, a)
               for (y, u, a) in zip(labels, uncertainties, priors)]
    return encoded


def encode_one(y, u=0, a=None):
    """
    y: Tensor() for a single label of shape (K,), where K is the label dim.
    """
    label_dim = y.shape[0]
    if label_dim in [1, 2]:
        return label2beta(y, u=u, a=a)
    elif label_dim > 2:
        return label2dirichlet(y, u=u, a=a)
    else:
        raise ValueError(f"Unsupported number of unique labels {label_dim}")


def old_encode_one(label, num_labels, u=0, a=None):
    # Binary
    if num_labels in [1, 2]:
        return label2beta(label, u=u, a=a)
    # Multi-class
    elif num_labels > 2:
        return label2dirichlet(label, num_labels, u=u, a=a)
    else:
        raise ValueError(f"Unsupported number of unique labels {num_labels}")


def label2beta(label, u=0, a=None):
    u = torch.as_tensor(u, dtype=torch.float32)

    # If a scalar, label is taken to be
    # the probability of belief.
    if label.dim() == 0:
        b = label
        d = 1. - label
    # vector of [belief, disbelief]
    elif label.dim() == 1:
        assert len(label) == 2
        b, d = label
    else:
        raise ValueError(f"Unsupported label shape {label.shape}")

    if u == 0:
        u = EPS
    b = b - (EPS / 2)
    d = d - (EPS / 2)
    return SLBeta(b, d, u, a)


def label2dirichlet(label, u=0, a=None):
    """
    label assumed to be one hot or vector of probabilities.
    """
    if u == 0:
        u = EPS
    u = torch.as_tensor([u], dtype=torch.float32)
    beliefs = label - (u * label)
    return SLDirichlet(beliefs, u, a)


def fuse(sldists, max_uncertainty=False):
    if len(sldists) == 1:
        return sldists[0]
    fused = sldists[0]
    for d in sldists[1:]:
        fused = fused.cumulative_fusion(d)
    if max_uncertainty is True:
        fused = fused.max_uncertainty()
    return fused


def cross_entropy(dist1: D.Distribution, dist2: D.Distribution):
    assert type(dist1) == type(dist2)
    return D.kl_divergence(dist1, dist2) + dist1.entropy()
