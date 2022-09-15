import json
import argparse

import numpy as np
import torch
import torch.distributions as D
import scipy.stats as stats
import scipy.spatial.distance as distance
from sklearn.metrics import precision_recall_fscore_support

import sle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output", type=str,
                        help="""Path to model outputs json file.""")
    parser.add_argument("label_type", type=str,
                        choices=["discrete", "sle"])
    return parser.parse_args()


def main(args):
    outputs = [json.loads(line) for line in open(args.model_output)]
    hard_metrics = compute_hard_metrics(outputs, args.label_type)
    print(json.dumps(hard_metrics, indent=2, ensure_ascii=False))

    soft_metrics = compute_soft_metrics(outputs, args.label_type)
    print(json.dumps(soft_metrics, indent=2, ensure_ascii=False))


def compute_hard_metrics(outputs, label_type):
    gold_y = []
    ann_y = []
    pred = []
    for op in outputs:
        gold_y.append(predict(op["gold_y"], "discrete"))
        ann_y.append(predict(op['Y'], label_type))
        pred.append(predict(op["model_output"], label_type))

    gp, gr, gf, _ = precision_recall_fscore_support(
        gold_y, pred, average="macro")
    ap, ar, af, _ = precision_recall_fscore_support(
        ann_y, pred, average="macro")

    return {"gold": {"precision↑": gp.item(),
                     "recall↑": gr.item(),
                     "f1↑": gf.item()},
            "ann": {"precision↑": ap.item(),
                    "recall↑": ar.item(),
                    "f1↑": af.item()}
            }


def predict(model_output, label_type):
    """
    label_type = "discrete"
        Predict discrete labels from a categorical distribution
        e.g., output of softmax/sigmoid.
    label_type = "sle"
        Predict discrete labels from a Beta/Dirichlet distribution
    """
    if label_type == "discrete":
        probs = torch.softmax(torch.tensor(model_output), dim=0)
        return np.argmax(probs)
    elif label_type == "sle":
        if 'd' in model_output.keys():
            dist = sle.SLBeta(**model_output)
        else:
            dist = sle.SLDirichlet(**model_output)
        return np.argmax(dist.mean)
    else:
        raise ValueError(f"Unsupported label_type '{label_type}'.")


def compute_soft_metrics(model_output, label_type):
    ann_y = []
    pred = []
    for op in model_output:
        ann_y.append(get_distribution(op['Y'], label_type))
        pred.append(get_distribution(op["model_output"], label_type))

    kl_mean, kl_sd = forward_kl(ann_y, pred)
    jsd_mean, jsd_sd = jensen_shannon_divergence(ann_y, pred)
    sim, corr = normalized_entropy_scores(ann_y, pred)

    return {"kl↓": kl_mean.item(),
            "jsd↓": jsd_mean.item(),
            "nes↑": sim.item(),
            "nec↑": corr.item()}


def get_distribution(model_output, label_type):
    """
    label_type = "discrete"
        Return a categorical distribution
    label_type = "sle"
        Return a SLBeta or SLDirichlet distribution
    """
    if label_type == "discrete":
        # model_output is already a categorical distribution
        probs = torch.softmax(torch.tensor(model_output), dim=0)
        return D.Categorical(probs)
    elif label_type == "sle":
        if 'd' in model_output.keys():
            return sle.SLBeta(**model_output)
        else:
            return sle.SLDirichlet(**model_output)


def forward_kl(gold_dists, pred_dists):
    kls = []
    for (g, p) in zip(gold_dists, pred_dists):
        kl = D.kl_divergence(g, p)
        kls.append(kl)
    return (np.mean(kls), np.std(kls))


def jensen_shannon_divergence(gold_dists, pred_dists):
    jsds = []
    for (g, p) in zip(gold_dists, pred_dists):
        if isinstance(g, D.Categorical):
            jsds.append(distance.jensenshannon(g.probs, p.probs))
        else:
            # SLBeta or SLDirichlet
            # sample 100 bernoulli/categorical distributions
            gold_probs = g.sample((10,))
            pred_probs = p.sample((10,))
            # compute JSD between each sample
            indiv_jsds = distance.jensenshannon(gold_probs, pred_probs, axis=1)
            jsds.append(indiv_jsds.mean())
    # return the mean JSD over distributions and samples
    return (np.mean(jsds), np.std(jsds))


def normalized_entropy_scores(gold_dists, pred_dists):
    distribution_cls = gold_dists[0].__class__
    if distribution_cls == sle.SLBeta:
        uniform_args = [0., 0., 1.]
    elif distribution_cls == sle.SLDirichlet:
        K = len(gold_dists[0].b)
        uniform_args = [np.zeros(K), [1.]]
    elif distribution_cls == D.Categorical:
        K = len(gold_dists[0].probs)
        uniform_args = [torch.ones((K,)) / K]
    max_ent = distribution_cls(*uniform_args).entropy()
    gold_ents = [g.entropy() / max_ent for g in gold_dists]
    pred_ents = [p.entropy() / max_ent for p in pred_dists]
    # Cosine SIMILARITY, not distance
    sim = 1. - distance.cosine(gold_ents, pred_ents)
    # Pearson's rho
    corr = np.abs(stats.pearsonr(gold_ents, pred_ents)[0])
    return (sim, corr)


if __name__ == "__main__":
    args = parse_args()
    main(args)
