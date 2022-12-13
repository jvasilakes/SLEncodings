import argparse

import torch
import torch.distributions as D
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

import src.dataloaders as dataloaders
import src.aggregators as aggregators

from sle import SLBeta, SLDirichlet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("datadirs", nargs='+', type=str)
    parser.add_argument("-u", type=float, default=0.0,
                        help="Total uncertainty")
    parser.add_argument("--annotator_id", type=int, default=None)
    parser.add_argument("--model", type=str, default="lr",
                        help="lr or path to model checkpoint.")
    return parser.parse_args()


def main(args):
    dl = dataloaders.load(args.dataset_name, *args.datadirs)
    data = aggregators.MultiAnnotatorDataset(**dl.train)
    # data.X is a list of tensors
    X = data.X
    if X[0].dim() > 1:
        X = [x.flatten() for x in X]
    X = torch.vstack(X).numpy()
    if X.shape[1] > 2:
        X = TSNE(n_components=2).fit_transform(X)

    gold_y = [y.argmax() for y in data.gold_y]
    if args.annotator_id is None:
        y = gold_y
    else:
        y = np.array([yi.argmax() for yi in data.Y[:, args.annotator_id, :]])
        md = data.metadata[0][args.annotator_id]
        md.pop("example_id")
        md["total_uncertainty"] = 1. - (0.5 * (md["annotator_reliability"] + md["annotator_certainty"]))  # noqa
        acc = (y == gold_y).sum() / len(y)
        md["accuracy"] = acc
        print(md)

    if args.model == "lr":
        z = get_logisitic_regression_entropies(X, y, args.u)
        z_points = y
    else:
        z = get_model_entropies(X, y, args.model)
        #z_points = get_model_entropies_points(X, y, args.model)
        z_points = get_model_preds_points(X, y, args.model)
    print(z.mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    ax.contourf(xx, yy, z, cmap="Blues")
    # ax.scatter(X0, X1, c=y, cmap="spring", s=20, alpha=0.8, edgecolor='k')
    ax.scatter(X0, X1, c=z_points, cmap="spring", s=20, alpha=0.8, edgecolor='k')  # noqa
    ax.set_xticks([])
    ax.set_yticks([])

    sm = ScalarMappable(norm=plt.Normalize(z.min(), z.max()), cmap="Blues")
    sm.set_array([])
    cbar = fig.colorbar(sm)
    #cbar.ax.set_ylim([0, 1])
    #cbar.ax.set_yticklabels([0, 1])
    cbar.ax.set_ylabel("Entropy", rotation=270)
    fig.tight_layout()
    plt.show()


def get_logisitic_regression_entropies(X, y, u):
    lr = LogisticRegression().fit(X, y)
    print(f"LR train acc: {lr.score(X, y):.4f}")

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


def get_model_entropies_points(X, y, model_ckpt):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_ckpt)
    model.to(DEVICE)
    model.eval()

    output = model({'X': torch.tensor(X, dtype=torch.float32).to(DEVICE)})
    if torch.is_tensor(output):
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
        entropies = D.Categorical(probs).entropy()
        ent_max = np.log(probs.size(1))
        norm_ents = entropies / ent_max
    elif isinstance(output, (SLBeta, SLDirichlet)):
        with torch.no_grad():
            norm_ents = output.entropy()
            max_ent = output.get_uniform().entropy()[0]
            min_ent = -30.
            norm_ents[torch.where(norm_ents > max_ent)] = min_ent
            norm_ents[torch.where(norm_ents < min_ent)] = min_ent

    return norm_ents.cpu().numpy()


def get_model_preds_points(X, y, model_ckpt):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_ckpt)
    model.to(DEVICE)
    model.eval()

    output = model({'X': torch.tensor(X, dtype=torch.float32).to(DEVICE)})
    if torch.is_tensor(output):
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
    elif isinstance(output, (SLBeta, SLDirichlet)):
        with torch.no_grad():
            probs = output.mean

    hard_preds = probs.argmax(dim=1)
    soft_preds = probs[torch.arange(probs.size(0)), hard_preds] + hard_preds
    #soft_preds /= soft_preds.max()
    return soft_preds.cpu().numpy()


def get_model_entropies(X, y, model_ckpt):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_ckpt)
    model.to(DEVICE)
    model.eval()

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    output = model({'X': torch.tensor(Xmesh, dtype=torch.float32).to(DEVICE)})
    if torch.is_tensor(output):
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
        entropies = D.Categorical(probs).entropy()
        ent_max = np.log(probs.size(1))
        norm_ents = entropies / ent_max
    elif isinstance(output, (SLBeta, SLDirichlet)):
        with torch.no_grad():
            norm_ents = output.entropy()
            max_ent = output.get_uniform().entropy()[0]
            min_ent = -30.
            norm_ents[torch.where(norm_ents > max_ent)] = max_ent
            norm_ents[torch.where(norm_ents < min_ent)] = min_ent

    return norm_ents.cpu().numpy().reshape(xx.shape)


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
