import torch
from .distributions import SLBeta, SLDirichlet
from .layers import SLELayer  # noqa, make available at package level
from .collate import sle_default_collate


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

    label_dim = ys.shape[1]
    if label_dim in [1, 2]:
        dists = encode_betas(ys, uncertainties, priors)
    elif label_dim > 2:
        dists = encode_dirichlets(ys, uncertainties, priors)
    else:
        raise ValueError(f"Unsupported number of labels {label_dim}")
    return dists


def encode_betas(ys, uncertainties=None, priors=None):
    if uncertainties is None:
        uncs = torch.zeros(ys.size(0), dtype=torch.float32).fill_(EPS)
    else:
        uncs = torch.as_tensor(uncertainties, dtype=torch.float32)
        uncs = uncs.squeeze()
        uncs[torch.isclose(uncs, torch.tensor(0.))] = EPS  # Smoothing dogmatic opinions  # noqa
    beliefs = ys - (uncs * ys)
    disbeliefs = 1. - (beliefs + uncs)
    if priors is None:
        priors = torch.zeros(ys.size(0), dtype=torch.float32).fill_(0.5)
    else:
        priors = torch.as_tensor(priors, dtype=torch.float32)
        priors = priors.squeeze()
    return [SLBeta(b, d, u, a) for (b, d, u, a) in
            zip(beliefs, disbeliefs, uncs, priors)]


def encode_dirichlets(ys, uncertainties=None, priors=None):
    if uncertainties is None:
        uncs = torch.zeros((ys.size(0), 1), dtype=torch.float32).fill_(EPS)
    else:
        uncs = torch.as_tensor(uncertainties, dtype=torch.float32)
        if uncs.dim() == 1:
            uncs = uncs.unsqueeze(1)
        uncs[torch.isclose(uncs, torch.tensor(0.))] = EPS  # Smoothing dogmatic opinions  # noqa
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


def fuse(dists):

    def compute_denominator(u1, u2):
        denom = (u1 + u2) - (u1 * u2)
        return denom

    def compute_b(b1, b2, u1, u2, denom):
        b_num = (b1 * u2) + (b2 * u1)
        b = b_num / denom
        return b

    def compute_u(u1, u2, denom):
        u_num = u1 * u2
        u = u_num / denom
        return u

    def compute_a(a1, a2, u1, u2):
        a_num1 = (a1 * u2) + (a2 * u1)
        a_num2 = (a1 + a2) * (u1 * u2)
        a_denom = (u1 + u2) - (2 * u1 * u2)
        a = (a_num1 - a_num2) / a_denom
        return a

    dist_cls = dists[0].__class__
    assert dist_cls in [SLBeta, SLDirichlet], "Can only fuse SLBeta or SLDirichlet"  # noqa
    assert all([isinstance(d, dist_cls) for d in dists]), "Found both SLBeta and SLDirichlet!"  # noqa

    if len(dists) == 1:
        return dists[0]

    b, u, a = dists[0].b, dists[0].u, dists[0].a
    for dist in dists[1:]:
        denom = compute_denominator(u, dist.u)
        b = compute_b(b, dist.b, u, dist.u, denom)
        u = compute_u(u, dist.u, denom)
        a = compute_a(a, dist.a, u, dist.u)

    args = [b, u, a]
    if dist_cls == SLBeta:
        d = 1. - (b + u)
        args = [b, d, u, a]
    return dist_cls(*args)


def fast_fuse(dists):
    raise NotImplementedError()
    dist_cls = dists[0].__class__
    assert dist_cls in [SLBeta, SLDirichlet], "Can only fuse SLBeta or SLDirichlet"  # noqa
    assert all([isinstance(d, dist_cls) for d in dists]), "Found both SLBeta and SLDirichlet!"  # noqa

    if len(dists) == 1:
        return dists[0]

    def helper(dists):
        if len(dists) == 2:
            a, b = dists
            T = (a.b * b.u) + (b.b * a.u)
            V = a.u + b.u
            U = a.u * b.u
            return T, V, U

        T, V, U = helper(dists[:-1])
        c = dists[-1]
        new_T = (c.u * T) + (c.b * U)
        new_V = U + (c.u * V)
        new_U = c.u * U
        return new_T, new_V, new_U

    T, V, U = helper(dists)
    K = len(dists) - 1
    denom = (V - (K * U))
    b = T / denom
    u = U / denom
    # a = Not Implemented
    args = [b, u]
    if dist_cls == SLBeta:
        d = 1 - (b + u)
        args = [b, d, u]
    return dist_cls(*args)
