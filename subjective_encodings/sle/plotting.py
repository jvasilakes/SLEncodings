import torch
import matplotlib.pyplot as plt
import ternary
from ternary.helpers import simplex_iterator


def plot_dirichlet(dist, cmap="Blues", title=None):
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
    from distributions import SLDirichlet
    d = SLDirichlet([0.6, 0.1, 0.1], [0.2])
    p = plot_dirichlet(d)
    p.show()
