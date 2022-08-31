import argparse

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

import src.dataloaders as dataloaders
import src.aggregators as aggregators


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("datadirs", nargs='+', type=str)
    parser.add_argument("-u", type=float, default=0.0,
                        help="Total uncertainty")
    parser.add_argument("--annotator_id", type=int, default=None)
    return parser.parse_args()


def main(args):
    dl = dataloaders.load(args.dataset_name, *args.datadirs)
    data = aggregators.MultiAnnotatorDataset(**dl.train)
    X = data.X
    if X.shape[1] > 2:
        X = TSNE(n_components=2).fit_transform(X)

    gold_y = data.gold_y.argmax(axis=-1).numpy()
    if args.annotator_id is None:
        y = gold_y
        print(f"total_uncertainty: {args.u}")
    else:
        y = np.array([yi.argmax() for yi in data.Y[:, args.annotator_id, :]])
        md = data.metadata[0][args.annotator_id]
        md.pop("example_id")
        md["total_uncertainty"] = 1. - (0.5 * (md["annotator_reliability"] + md["annotator_certainty"]))  # noqa
        acc = (y == gold_y).sum() / len(y)
        md["accuracy"] = acc
        print(md)

    z = get_logisitic_regression_entropies(X, y, args.u)

    fig, ax = plt.subplots(figsize=(8, 4))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    ax.contourf(xx, yy, z, cmap="Blues")
    ax.scatter(X0, X1, c=y, cmap="spring", s=20, alpha=0.8, edgecolor='k')
    ax.set_xticks([])
    ax.set_yticks([])

    sm = ScalarMappable(norm=plt.Normalize(z.min(), z.max()), cmap="Blues")
    sm.set_array([])
    cbar = fig.colorbar(sm, ticks=[0, 1])
    cbar.ax.set_ylim([0, 1])
    cbar.ax.set_yticklabels([0, 1])
    cbar.ax.set_ylabel("Entropy", rotation=270)
    fig.tight_layout()
    plt.show()


def get_logisitic_regression_entropies(X, y, u):
    lr = LogisticRegression().fit(X, y)

    # x and y coordinates, not inputs and outputs
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    probs = lr.predict_proba(Xmesh)
    probs = recalibrate_probs(probs, 1. - args.u)
    n_classes = probs.shape[-1]
    ent = stats.entropy(probs, axis=1)
    ent_max = np.log(n_classes)
    z = (ent / ent_max).reshape(xx.shape)
    return z


def recalibrate_probs(probs, gamma):
    exponent = gamma * np.log(probs)
    num = np.exp(exponent)
    denom = np.exp(exponent).sum(axis=1)[:, np.newaxis]
    return num / denom


def make_meshgrid(x, y, h=0.02):
    x_min = x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


if __name__ == "__main__":
    args = parse_args()
    main(args)
