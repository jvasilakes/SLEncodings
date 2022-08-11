import torch
import matplotlib.pyplot as plt
import ternary
from ternary.helpers import simplex_iterator


def plot_dirichlet(dist, cmap="Blues", title=None):
    data = {}
    for (i, j, k) in simplex_iterator(scale=100, boundary=True):
        datum = dist.log_prob(torch.as_tensor([i, j, k]) / 100)
        if torch.isinf(datum):
            datum = 1.
        elif torch.isnan(datum):
            datum = 0.
        else:
            datum = datum.item()
        data = [(i, j)] = datum

    fig, tax = ternary.figure(scale=100)
    tax.heatmap(data, cmap=cmap, colorbar=False)
    tax.boundary(linewidth=1)
    plt.axis("off")
    if title is not None:
        tax.set_title(title)
    return tax
