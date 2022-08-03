import os
import re
import json
import argparse
from glob import glob
from collections import Counter

import numpy as np
from scipy.stats import entropy
from torch.distributions import kl_divergence
from sklearn.metrics import precision_recall_fscore_support

from metrics import cross_entropy
import sle
import sle.distributions as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str,
                        help="Path to output.json")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold probability for prediction.")
    return parser.parse_args()


def main(args):
    cmdlog = os.path.join(args.output_dir, "cmdlog.json")
    hyperparams = json.load(open(cmdlog))
    model_outputs = load_model_outputs(args.output_dir, args.epoch)

    if hyperparams["label_type"] == "sl":
        evaluate_subjective_logic(model_outputs, args.threshold)
    elif hyperparams["label_type"] == "discrete":
        evaluate_discrete(model_outputs, args.threshold)
    else:
        raise ValueError(f"Unsupported label: {hyperparams['label_type']}")


def expand_dims(labels):
    labels = np.reshape(labels, (-1, 1))
    labels_inv = np.abs(labels - 1.)
    return np.hstack([labels, labels_inv])


def evaluate_discrete(model_outputs, threshold):
    preferred_labels = []
    hard_labels = []
    soft_labels = []
    hard_predictions = []
    soft_predictions = []
    for example in model_outputs.values():
        preferred_labels.append(int(example["preferred_y"]))
        lab = example['y']
        hard_labels.append(int(lab >= 0.5))
        soft_labels.append(lab)
        probs = np.array(example["model_output"]["probs"])
        soft_predictions.append(probs)
        pred = int(probs[0] >= threshold)
        hard_predictions.append(pred)

    p, r, f, _ = precision_recall_fscore_support(
            preferred_labels, hard_predictions, average=None)

    hard_labels_exp = expand_dims(hard_labels)
    soft_labels_exp = expand_dims(soft_labels)
    soft_preds_exp = expand_dims(soft_predictions)

    hard_ent = entropy(hard_labels_exp, axis=1).mean()
    hard_ce = cross_entropy(hard_labels_exp, soft_preds_exp).mean()
    hard_kldiv = entropy(hard_labels_exp, soft_preds_exp, axis=1).mean()

    soft_ent = entropy(soft_labels_exp, axis=1).mean()
    soft_ce = cross_entropy(soft_labels_exp, soft_preds_exp).mean()
    soft_kldiv = entropy(soft_labels_exp, soft_preds_exp, axis=1).mean()

    label_counts = Counter(preferred_labels)
    print("Data Statistics")
    for (lab, count) in label_counts.items():
        print(f"  {lab}: {count}")
    print()

    print("Results (hard_labels)")
    print(f"  Entropy: {hard_ent:.4f}")
    print(f"  Cross-Entropy: {hard_ce:.4f}")
    print(f"  KL Divergence: {hard_kldiv:.4f}")
    print(f"  kl+entropy = {hard_kldiv + hard_ent:.4f}")
    for label in label_counts.keys():
        print(f"  Label: {label}")
        for (metric, results) in [("prec", p), ("rec", r), ("f1", f)]:
            print(f"   {metric}: {results[label]:.4f}")

    print()
    print("Results (soft_labels)")
    print(f"  Entropy: {soft_ent:.4f}")
    print(f"  Cross-Entropy: {soft_ce:.4f}")
    print(f"  KL Divergence: {soft_kldiv:.4f}")
    print(f"  kl+entropy = {soft_kldiv + soft_ent:.4f}")


def evaluate_subjective_logic(model_outputs, threshold):
    preferred_labels = []
    hard_labels = []
    soft_labels = []
    label_dists = []
    hard_predictions = []
    soft_predictions = []
    pred_dists = []
    for example in model_outputs.values():
        preferred_labels.append(int(example["preferred_y"]))
        y_dist = dist.SLBeta(**example['y'])
        label_dists.append(y_dist)
        hard_labels.append(int(y_dist.mean >= 0.5))
        soft_labels.append(y_dist.mean)

        pred_dist = dist.SLBeta(**example["model_output"])
        pred_dists.append(pred_dist)
        soft_predictions.append(pred_dist.mean)
        hard_predictions.append(int(pred_dist.mean >= 0.5))

    p, r, f, _ = precision_recall_fscore_support(
            preferred_labels, hard_predictions, average=None)

    label_ent = np.array([d.entropy() for d in label_dists])
    ce = np.array([sle.cross_entropy(yd, pd) for (yd, pd)
                   in zip(label_dists, pred_dists)])
    kldiv = np.array([kl_divergence(yd, pd) for (yd, pd)
                      in zip(label_dists, pred_dists)])
    zipped = zip(kldiv, label_ent, ce, label_dists, pred_dists)
    for (k, e, c, yd, pd) in zipped:
        ke_str = f"{k+e:.4f}"
        c_str = f"{c:.4f}"
        if ke_str != c_str:
            print(yd, pd)
            print(f"{k:.4f} + {e:.4f} = {k+e:.4f} = {c:.4f}")
            input()

    label_counts = Counter(preferred_labels)
    print("Data Statistics")
    for (lab, count) in label_counts.items():
        print(f"  {lab}: {count}")
    print()

    print("Results")
    print(f"  Entropy: {np.mean(label_ent):.4f}")
    print(f"  Cross-Entropy: {np.mean(ce):.4f}")
    print(f"  KL Divergence: {np.mean(kldiv):.4f}")
    print(f"  kl+entropy = {np.mean(kldiv + label_ent):.4f}")
    for label in label_counts.keys():
        print(f"  Label: {label}")
        for (metric, results) in [("prec", p), ("rec", r), ("f1", f)]:
            print(f"   {metric}: {results[label]:.4f}")


def load_model_outputs(dirpath, epoch=-1):
    globpath = os.path.join(dirpath, "outputs_epoch*.json")
    output_files = glob(globpath)
    epoch_re = re.compile(r'(?<=\=)[0-9]+')

    epochs_and_output_files = []
    for path in output_files:
        search_res = epoch_re.search(path)
        if search_res is not None:
            file_epoch = int(search_res.group(0))
            epochs_and_output_files.append((file_epoch, path))
    sorted_files = sorted(epochs_and_output_files, key=lambda x: x[0])
    if epoch == -1:
        fpath = sorted_files[-1][1]
    else:
        try:
            fpath = dict(sorted_files)[epoch]
        except KeyError:
            raise KeyError(f"Outputs for epoch {epoch} not found.")

    return json.load(open(fpath))


if __name__ == "__main__":
    args = parse_args()
    main(args)
