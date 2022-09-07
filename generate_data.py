import os
import json
import argparse

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import src.aggregators as aggregators


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the generated data.")
    parser.add_argument("--n-examples", type=int, default=100,
                        help="The total number of examples.")
    parser.add_argument("--n-features", type=int, default=5,
                        help="The input dimension.")
    parser.add_argument("--n-annotators", type=int, default=3,
                        help="The number of annotator per example.")
    parser.add_argument("--reliability", type=str, default="high",
                        choices=["perfect", "high", "high-outlier", "medium",
                                 "low", "low-outlier"],
                        help="Distribution of annotator reliability.")
    parser.add_argument("--certainty", type=str, default="high",
                        choices=["perfect", "high", "high-outlier", "medium",
                                 "low", "low-outlier"],
                        help="Distribution of annotator certainty.")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=False)
    np.random.seed(args.random_seed)

    label_set = set(range(3))
    annotators = get_annotators(args.n_annotators, label_set,
                                args.reliability, args.certainty)

    data = generate_data(args.n_examples, args.n_features, annotators,
                         random_seed=args.random_seed)

    train_idxs, other_idxs = train_test_split(
        range(args.n_examples), train_size=0.8,
        random_state=args.random_seed)
    val_idxs, test_idxs = train_test_split(
        range(len(other_idxs)), test_size=0.5,
        random_state=args.random_seed)

    train = [data[i] for i in train_idxs]
    val = [data[i] for i in val_idxs]
    test = [data[i] for i in test_idxs]

    train_outpath = os.path.join(args.outdir, "train.json")
    save_data(train, train_outpath)

    val_outpath = os.path.join(args.outdir, "val.json")
    save_data(val, val_outpath)

    test_outpath = os.path.join(args.outdir, "test.json")
    save_data(test, test_outpath)

    metadata = {"n_examples": args.n_examples,
                "n_features": args.n_features,
                "n_classes": 3,
                "n_annotators": args.n_annotators,
                "reliability": args.reliability,
                "certainty": args.certainty,
                "random_seed": args.random_seed}
    with open(os.path.join(args.outdir, "metadata.json"), 'w') as outF:
        json.dump(metadata, outF)


def get_annotators(n, label_set, reliability, certainty):
    params_map = {"perfect": (10, 1e-18),
                  "high": (10, 1),
                  "high-outlier": (10, 1),
                  "medium": (10, 10),
                  "low": (1, 10),
                  "low-outlier": (1, 10)
                  }

    reliability_params = params_map[reliability]
    reliabilities = np.random.beta(*reliability_params, size=n)
    if reliability == "high-outlier":
        reliabilities[0] = 0.1
    elif reliability == "low-outlier":
        reliabilities[0] = 0.9

    certainty_params = params_map[certainty]
    certainties = np.random.beta(*certainty_params, size=n)
    if certainty == "high-outlier":
        certainties[0] = 0.1
    elif certainty == "low-outlier":
        certainties[0] = 0.9

    return [aggregators.Annotator(label_set, reliabilities[i], certainties[i])
            for i in range(n)]


def generate_data(n_examples, n_features, annotators, random_seed=0):
    n_redundant = min(0, n_features-2)
    X, y = make_classification(n_examples, n_features,
                               n_classes=3, n_clusters_per_class=1,
                               n_redundant=n_redundant, shift=0.5,
                               class_sep=0.75, flip_y=0.0,
                               random_state=random_seed)

    data = []
    for (i, (x_i, pref_y)) in enumerate(zip(X, y)):
        datum = {"example_id": i,
                 'x': list(x_i), "preferred_y": int(pref_y),
                 "annotations": []}
        for (j, annotator) in enumerate(annotators):
            ann_y = annotator.annotate(pref_y)
            datum["annotations"].append(
                    {"annotator_id": annotator.id,
                     "annotator_reliability": annotator.reliability,
                     "annotator_certainty": annotator.certainty,
                     "y": int(ann_y)}
                    )
        data.append(datum)
    return data


def save_data(dataset, outpath):
    with open(outpath, 'w') as outF:
        for datum in dataset:
            json.dump(datum, outF)
            outF.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
