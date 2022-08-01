import os
import json
import argparse

import torch
import numpy as np
from tqdm import trange

import data as D
from models import LinearNet, SLNet, AggregatingSLNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save log files.")
    parser.add_argument("--datapath", type=str, default=None,
                        help="Path to data file")
    parser.add_argument("--label-type", type=str, default="discrete",
                        choices=["discrete", "sl"])
    parser.add_argument("--label-aggregation", type=str, default=None,
                        choices=["fuse", "vote", "freq", "sample"])
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--n-annotators", type=int, default=3)
    parser.add_argument('-T', "--annotator-trustworthiness", type=str,
                        choices=["perfect", "high", "high-outlier", "medium",
                                 "low", "low-outlier"], default="high",
                        help="Distribution of annotator trustworthiness.")
    parser.add_argument("--hidden-dim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--generate-data-only", action="store_true",
                        help="""If set, generate and save data, but
                                don't train a model.""")
    return parser.parse_args()


def main(args):
    np.random.seed(0)
    dataclass = get_data_class(args.label_type, args.label_aggregation)
    if args.datapath is not None:
        dataset = dataclass.from_file(args.datapath)
    else:
        dataset = dataclass(args.n_features, args.n_examples,
                            annotators=args.n_annotators,
                            trustworthiness=args.annotator_trustworthiness,
                            random_seed=args.random_seed)
    print(dataset)
    os.makedirs(args.outdir, exist_ok=False)
    data_outpath = os.path.join(args.outdir, "data.json")
    dataset.save(data_outpath)

    if args.generate_data_only is True:
        return

    modelclass = get_model_class(args.label_type, args.label_aggregation)
    model = modelclass(dataset.n_features, args.hidden_dim,
                       dataset.label_dim, lr=args.lr)
    print(model)
    model.train()

    save_cmdline_args(args, args.outdir)

    losses = []
    for epoch in trange(args.epochs):
        epoch_losses = []
        if (epoch+1) % 10 == 0:
            model.eval()
            save_model_outputs(model, dataset, epoch=epoch, outdir=args.outdir)
            model.train()
        for (n, datum) in enumerate(dataset):
            output = model(datum["x"])
            loss = model.compute_loss(output, datum)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            epoch_losses.append(loss.item())
        losses.append(epoch_losses)

    model.eval()
    save_model_outputs(model, dataset, epoch=epoch, outdir=args.outdir)

    losslog = os.path.join(args.outdir, "losses.json")
    with open(losslog, 'w') as outF:
        json.dump(losses, outF)


def save_cmdline_args(args, outdir):
    outpath = os.path.join(outdir, "cmdlog.json")
    with open(outpath, 'w') as outF:
        json.dump(args.__dict__, outF, indent=2)


def get_data_class(label_type, label_aggregation):
    lookup = {
            ("discrete", None): D.MultiAnnotatorDataset,
            ("discrete", "vote"): D.VotingAggregatedDataset,
            ("discrete", "freq"): D.FrequencyAggregatedDataset,
            ("discrete", "sample"): D.CatSampleAggregatedDataset,
            ("sl", None): D.SubjectiveLogicDataset,
            ("sl", "fuse"): D.CumulativeFusionDataset,
            ("sl", "sample"): D.SLSampleAggregatedDataset,
            }
    try:
        return lookup[(label_type, label_aggregation)]
    except KeyError:
        raise KeyError(f"Unsupported (label_type, label_aggregation): ({label_type}, {label_aggregation})")  # noqa


def get_model_class(label_type, label_aggregation):
    lookup = {
            ("discrete", None): LinearNet,
            ("discrete", "vote"): LinearNet,
            ("discrete", "freq"): LinearNet,
            ("discrete", "sample"): LinearNet,
            ("sl", None): AggregatingSLNet,
            ("sl", "fuse"): SLNet,
            ("sl", "sample"): LinearNet,
            }
    try:
        return lookup[(label_type, label_aggregation)]
    except KeyError:
        raise KeyError(f"Unsupported (label_type, label_aggregation): ({label_type}, {label_aggregation})")  # noqa


def save_model_outputs(model, dataset, epoch=None, outdir=None):
    outputs = {}
    model.eval()
    seen = set()
    for datum in dataset:
        datum_cp = dict(datum)
        uuid = datum_cp.pop("uuid")
        if uuid in seen:
            continue
        seen.add(uuid)
        output = model(datum["x"])
        if "distribution" in output:
            output = output["distribution"].parameters()
            datum_cp['y'] = datum_cp['y'].parameters()
        datum_cp["model_output"] = output
        datum_cp = convert_tensors_to_items(datum_cp)
        outputs[uuid] = datum_cp

    if outdir is None:
        outdir = '.'
    outpath = os.path.join(outdir, f"outputs_epoch={epoch}.json")
    with open(outpath, 'w') as outF:
        json.dump(outputs, outF)


def convert_tensors_to_items(collection):
    if torch.is_tensor(collection):
        if collection.dim() > 0:
            return collection.tolist()
        else:
            return collection.item()

    if isinstance(collection, dict):
        cp = dict(collection)
        for key in cp.keys():
            cp[key] = convert_tensors_to_items(cp[key])
    elif isinstance(collection, (list, tuple, set)):
        cp = type(collection)(collection)
        for i in range(len(cp)):
            cp[i] = convert_tensors_to_items(cp[i])
    else:
        return collection

    return cp


if __name__ == "__main__":
    args = parse_args()
    main(args)
