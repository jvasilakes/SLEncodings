import torch
import torch.distributions as D

from .distributions import SLBeta, SLDirichlet
from .layers import SLELayer  # noqa, make available at package level


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
    encoded = batch_encode(ys, uncertainties, priors)
    # encoded = [encode_one(y, u, a) for (y, u, a)
    #            in zip(ys, uncertainties, priors)]
    return encoded


def batch_encode(ys, uncertainties=None, priors=None):
    label_dim = ys.shape[1]
    if label_dim in [1, 2]:
        pass
    elif label_dim > 2:
        if uncertainties is None:
            uncs = torch.zeros((ys.size(0), 1), dtype=torch.float32).fill_(EPS)
        else:
            uncs = torch.as_tensor(uncertainties, dtype=torch.float32)
            if uncs.dim() == 1:
                uncs = uncs.unsqueeze(1)
        uncs[uncs == 0.] = EPS  # Smoothing dogmatic opinions
        beliefs = ys - (uncs * ys)
        if priors is None:
            priors = torch.ones(ys.size(), dtype=torch.float32)
            priors /= torch.ones(ys.size(0), 1).fill_(ys.size(1))
        else:
            priors = torch.as_tensor(priors, dtype=torch.float32)
            if priors.dim() == 1:
                priors = priors.unsqueeze(1)
        return [SLDirichlet(b, u, a) for (b, u, a) in
                zip(beliefs, uncs, priors)]
    else:
        raise ValueError(f"Unsupported number of labels {label_dim}")


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


def label2beta(label, u=0, a=None):
    u = torch.as_tensor(u, dtype=torch.float32)

    # If a scalar, label is taken to be
    # the probability of belief.
    if label.dim() == 0:
        b = label
        d = 1. - label
    elif label.dim() == 1:
        # vector of [belief]
        if label.shape[0] == 1:
            b = label[0]
            d = 1. - b
        # vector of [belief, disbelief]
        elif label.shape[0] == 2:
            b, d = label
        else:
            raise ValueError(f"Unsupported label shape {label.shape}")
    else:
        raise ValueError(f"Unsupported label shape {label.shape}")

    if u == 0:
        u = EPS
    if b > d:
        b = b - (EPS / 2)
    else:
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
