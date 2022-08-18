import os
import argparse

import src.datasets as datasets


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
    parser.add_argument("--confidence", type=str, default="high",
                        choices=["perfect", "high", "high-outlier", "medium",
                                 "low", "low-outlier"],
                        help="Distribution of annotator confidence.")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    os.makedirs(args.outdir, exist_ok=False)
    full_dataset = datasets.MultiAnnotatorDataset(
        args.n_examples, args.n_features, annotators=args.n_annotators,
        reliability=args.reliability, confidence=args.confidence,
        random_seed=args.random_seed)
    train, val, test = datasets.split_dataset(full_dataset)

    train_outpath = os.path.join(args.outdir, "train.json")
    train.save(train_outpath)
    val_outpath = os.path.join(args.outdir, "val.json")
    val.save(val_outpath)
    test_outpath = os.path.join(args.outdir, "test.json")
    test.save(test_outpath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
