import torch
import numpy as np
import matplotlib.pyplot as plt
import ternary
from ternary.helpers import simplex_iterator


def clear_plots():
    f = plt.figure()
    f.clear()
    plt.close(f)


def plot_beta(dist, title=None):
    clear_plots()
    x = torch.as_tensor(np.linspace(0, 1, 100), dtype=torch.float32)
    y = torch.exp(dist.log_prob(x))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    return fig


def plot_dirichlet(dist, cmap="Blues", title=None):
    if dist.event_shape[0] != 3:
        raise ValueError("Can't plot Dirichlets with >3 dimensions.")
    clear_plots()
    data = {}
    for (i, j, k) in simplex_iterator(scale=100, boundary=True):
        datum = torch.exp(dist.log_prob(torch.as_tensor([i, j, k]) / 100))
        if torch.isinf(datum):
            datum = 1.
        elif torch.isnan(datum):
            datum = 0.
        else:
            datum = datum.item()
        data[(i, j)] = datum

    _, ax = plt.subplots(figsize=(5, 5))
    fig, tax = ternary.figure(scale=100, ax=ax)
    tax.heatmap(data, cmap=cmap, colorbar=False)
    tax.boundary(linewidth=1)
    plt.axis("off")
    if title is not None:
        tax.set_title(title)
    return tax


if __name__ == "__main__":
    from distributions import SLBeta, SLDirichlet
    b = SLBeta(0.2, 0.7, 0.1)
    b.plot()
    d = SLDirichlet([0.6, 0.1, 0.1], [0.2])
    d.plot()
