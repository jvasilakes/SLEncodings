import argparse

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.linear_model import LogisticRegression

import src.datasets as datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    parser.add_argument("-u", type=float, default=1.0,
                        help="Total uncertainty")
    return parser.parse_args()


def main(args):
    data = datasets.MultiAnnotatorDataset.from_file(args.filepath)
    X = np.array([exs[0]['x'].numpy() for exs in data.examples])
    y = np.array([exs[0]["preferred_y"].item() for exs in data.examples])
    lr = LogisticRegression().fit(X, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    probs = lr.predict_proba(Xmesh)
    probs = recalibrate_probs(probs, args.u)
    n_classes = probs.shape[-1]
    ent = stats.entropy(probs, axis=1)
    ent_max = np.log(n_classes)
    z = (ent / ent_max).reshape(xx.shape)

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
